import time
from tqdm import tqdm
import torch
import wandb
import numpy as np 

from src.visualization import print_loss
from src.utility import format_time


def init_wandb(model, opt_name, fusion_name='unknown', num_params=-1, batch_size=64, epochs=15):
  """
  Initialize a Weights & Biases run for a given fusion model.

  Args:
      model (nn.Module): The PyTorch model to track.
      fusion_name (str): Short name of the fusion strategy (e.g. "early_fusion").
      num_params (int): Total number of trainable parameters of the model.
      opt_name (str): Name of the optimizer (e.g. "Adam").
      batch_size (int, optional): Batch size used during training.
      epochs (int, optional): Number of training epochs.

  Returns:
      wandb.sdk.wandb_run.Run: The initialized W&B run object.
  """

  config = {
    # "embedding_size": embedding_size,      ## TODO: ändert die sich? hab ich die bei fusion?
    "optimizer_type": opt_name,
    "fusion_strategy": fusion_name,
    "model_architecture": model.__class__.__name__,
    "batch_size": batch_size,
    "num_epochs": epochs,
    "num_parameters": num_params
  }

  run = wandb.init(
    project="cilp-extended-assessment",
    name=f"{fusion_name}_run",
    config=config,
    reinit='finish_previous',                           # allows starting a new run inside one script
  )

  return run


def train_model(model, optimizer, input_fn, loss_fn, epochs, train_dataloader, val_dataloader, model_save_path, target_idx=-1, log_to_wandb=False, device=None):
    """
    Generic training loop for all fusion models.

    Args:
        model (nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        input_fn (callable): Function that maps a batch to model inputs.
                             Takes a batch tuple and returns a tuple of tensors.
        loss_fn (callable): Loss function (e.g. CrossEntropyLoss).
        epochs (int): Number of training epochs.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader): Dataloader for validation data.
        model_save_path (str or Path): Where to save the best model checkpoint.
        target_idx (int): If using multi-target labels, index of the target
                          to use (-1 for all / default).
        log_to_wandb (bool): If True, log metrics to Weights & Biases.
        model_name (str or None): Optional label for logging / printing.

    Returns:
        dict: Dictionary containing training history:
              {
                "train_losses": [...],
                "valid_losses": [...],
                "epoch_times": [...],
                "best_valid_loss": float,
                "best_model_state_dict": dict,
                "num_params": int,
                "max_gpu_mem_mb": float,
              }
    """
    train_losses = []
    valid_losses = []
    epoch_times = []
    valid_accs = []

    best_val_loss = float('inf')

    # Track peak GPU memory usage (if CUDA is available)
    max_gpu_mem_mb = 0.0
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    for epoch in tqdm(range(epochs)):
        start_time = time.time()                  # to track the train time per model
        print(f"Epoch and start time: {epoch} und {start_time}")

        # ----- Training loop -----
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            rgb, lidar_xyza, position = batch
            rgb = rgb.to(device)
            lidar_xyza = lidar_xyza.to(device)
            position = position.to(device)

            optimizer.zero_grad()
            target = batch[target_idx].to(device)
            outputs = model(*input_fn(batch, device))

            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        train_losses.append(train_loss)
        print_loss(epoch, train_loss, outputs, target, is_train=True)

        # ----- Validation loop -----
        model.eval()
        valid_loss = 0.0
        correct = 0.0
        total = 0.0
        with torch.no_grad():
          for step, batch in enumerate(val_dataloader):
              target = batch[target_idx].to(device)
              outputs = model(*input_fn(batch, device))
              
              # calculate loss
              valid_loss += loss_fn(outputs, target).item()
              
              # calculate accuracy 
              preds = outputs.argmax(dim=1)
              correct += (preds == target).sum().item()
              total += target.size(0)

        valid_loss = valid_loss / (step + 1)
        valid_losses.append(valid_loss)

        valid_acc = correct / total
        valid_accs.append(valid_acc)

        print_loss(epoch, valid_loss, outputs, target, is_train=False)

        # Save best model based on validation loss
        if valid_loss < best_val_loss:
          best_val_loss = valid_loss
          torch.save(model.state_dict(), model_save_path)
          print('Found and saved better weights for the model')
        
        # accuracy of best validation-loss epoch
        best_epoch = int(np.argmin(valid_losses))
        best_valid_acc = valid_accs[best_epoch]

        # calculate epoch times
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        epoch_time_formatted = format_time(epoch_time)

        # GPU memory usage
        if use_cuda:
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            max_gpu_mem_mb = max(max_gpu_mem_mb, gpu_mem_mb)
        else:
            gpu_mem_mb = 0.0

        # wandb logging
        if log_to_wandb:
            wandb.log(
                {
                    "model": model.__class__.__name__,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_time": epoch_time_formatted,
                    "max_gpu_mem_mb_epoch": gpu_mem_mb if use_cuda else 0.0,
                }
            )

    return {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "valid_accs": valid_accs,
        "epoch_times": epoch_times,
        "best_valid_loss": float(best_val_loss),
        "best_valid_acc": float(best_valid_acc),
        "max_gpu_mem_mb": float(max_gpu_mem_mb),
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }




def compute_class_weights_old(dataset, num_classes=2):
    """
    Compute inverse-frequency class weights from a dataset.
    """
    # Grab all labels from the prebuilt samples list (cheap)
    labels = [sample["label"] for sample in dataset.samples]
    labels = torch.tensor(labels, dtype=torch.long)

    # Count occurrences of each class
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # Inverse-frequency style weights (rarer class -> larger weight)
    class_weights = class_counts.sum() / (class_counts + 1e-6)

    # Normalize so mean(weight) = 1 (optional but nice)
    class_weights = class_weights / class_weights.mean()

    return class_weights


def get_inputs(batch, device=None):
    """
    Prepare inputs for intermediate/late fusion models.

    Returns RGB and XYZA tensors separately so that each modality
    can be passed to its own encoder.

    Args:
        batch (tuple): (rgb, xyz, label) from the dataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (inputs_rgb, inputs_xyz)
    """
    inputs_rgb = batch[0].to(device)
    inputs_xyz = batch[1].to(device)
    return (inputs_rgb, inputs_xyz)


def get_early_inputs(batch, device=None):
    """
    Prepare inputs for the early fusion model.

    Concatenates RGB and XYZA along the channel dimension to obtain
    an 8-channel tensor.

    Args:
        batch (tuple): (rgb, xyz, label) from the dataset.

    Returns:
        tuple[torch.Tensor]: Single-element tuple (inputs_mm_early,).
    """
    inputs_rgb = batch[0].to(device)
    inputs_xyz = batch[1].to(device)

    # Concatenate along channel dimension: (B, 4, H, W) + (B, 4, H, W) -> (B, 8, H, W)
    inputs_mm_early = torch.cat((inputs_rgb, inputs_xyz), 1)
    return (inputs_mm_early,)


def get_rgb_inputs(batch, device):
    rgb, lidar_xyza, pos = batch
    return (rgb.to(device),)


def train_with_batch_loss(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    batch_loss_fn,      # (model, batch, device) -> loss
    epochs,
    model_save_path,
    log_to_wandb=False,
    device=None,
    extra_args=None,
):
    """
    Generic loop for models trained from a batch-level loss:
    - CILP contrastive (Stage 1)
    - Projector MSE (Stage 2)

    batch_loss_fn must take (model, batch, device) and return a scalar loss.
    """
    train_losses = []
    valid_losses = []
    epoch_times = []

    best_val_loss = float("inf")
    max_gpu_mem_mb = 0.0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        start_time = time.time()
        print(f"[{'BatchLoss'}] Epoch {epoch+1}/{epochs}")

        # ----- TRAIN -----
        model.train()
        train_loss_epoch = 0.0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, _ = batch_loss_fn(model, batch, device, **(extra_args or {}))
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= (step + 1)
        train_losses.append(train_loss_epoch)
        print(f"  train loss: {train_loss_epoch:.4f}")

        # ----- VALID -----
        model.eval()
        valid_loss_epoch = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                loss, _ = batch_loss_fn(model, batch, device, **(extra_args or {}))
                valid_loss_epoch += loss.item()
        valid_loss_epoch /= (step + 1)
        valid_losses.append(valid_loss_epoch)
        print(f"  valid loss: {valid_loss_epoch:.4f}")

        # Save best model
        if valid_loss_epoch < best_val_loss:
            best_val_loss = valid_loss_epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"  → New best saved (val loss {best_val_loss:.4f})")

        # Epoch timing
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        epoch_time_formatted = format_time(epoch_time)

        # GPU stats
        if use_cuda:
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            max_gpu_mem_mb = max(max_gpu_mem_mb, gpu_mem_mb)
        else:
            gpu_mem_mb = 0.0

        # wandb
        if log_to_wandb:
            wandb.log({
                "model": model.__class__.__name__,
                "epoch": epoch + 1,
                "train_loss": train_loss_epoch,
                "valid_loss": valid_loss_epoch,
                "epoch_time": epoch_time_formatted,
                "max_gpu_mem_mb_epoch": gpu_mem_mb,
                "lr": optimizer.param_groups[0]["lr"],
            })

    return {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "epoch_times": epoch_times,
        "best_valid_loss": float(best_val_loss),
        "max_gpu_mem_mb": float(max_gpu_mem_mb),
    }