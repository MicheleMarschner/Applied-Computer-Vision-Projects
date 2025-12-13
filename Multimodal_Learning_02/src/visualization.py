import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fiftyone as fo
import torch
import torch.nn.functional as Func
import wandb

from src.utility import format_positions


def print_loss(epoch, loss, outputs, target, is_train=True, is_debug=False):
    """
    Print a formatted loss line and optionally one example prediction.

    Args:
        epoch (int): Current epoch index.
        loss (float or Tensor): Loss value for this epoch.
        outputs (torch.Tensor): Model predictions for the current batch.
        target (torch.Tensor): Ground-truth targets for the current batch.
        is_train (bool): If True, label as training loss; else validation.
        is_debug (bool): If True, also print one prediction/target pair.
    """

    loss_type = "train loss:" if is_train else "valid loss:"
    print("epoch", str(epoch), loss_type, str(loss))

    if is_debug:
        print("example pred:", format_positions(outputs[0].tolist()))
        print("example real:", format_positions(target[0].tolist()))


## Final: löschen
def plot_losses(losses, title="Training & Validation Loss Comparison", figsize=(10,6)):
    """
    Legacy plotting helper to show train/valid losses for multiple models.

    Args:
        losses (dict): Mapping model_name -> {"train_losses": [...],
                                              "valid_losses": [...]}.
        title (str): Plot title.
        figsize (tuple): Matplotlib figure size.
    """
    plt.figure(figsize=figsize)

    for model_name, log in losses.items():
        train = log["train_losses"]
        valid = log["valid_losses"]

        # plot train + valid with different line styles
        plt.plot(train, label=f"{model_name} - train", linewidth=2)
        plt.plot(valid, label=f"{model_name} - valid", linestyle="--", linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_losses(loss_dict, title="Validation Loss per Model", ylabel="Loss", xlabel="Epoch"):
    """
    Plot validation loss curves for multiple models.

    Args:
        loss_dict (dict): Mapping "model_name" -> list_of_losses (same length).
        title (str): Plot title.
        ylabel (str): Label for y-axis.
        xlabel (str): Label for x-axis.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Auto-generate x-axis based on first model
    any_key = next(iter(loss_dict))
    epochs = range(len(loss_dict[any_key]))

    for model_name, losses in loss_dict.items():
        ax.plot(epochs, losses, label=model_name)

    ax.xlabel(xlabel)
    ax.ylabel(ylabel)
    ax.title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    
    fig.tight_layout()
    return fig, ax


def build_fusion_comparison_df(metrics, name_map=None):
    rows = []
    for key, m in metrics.items():
        avg_train_loss = float(np.mean(m["train_losses"]))
        avg_valid_loss = float(np.mean(m["valid_losses"]))
        avg_epoch_time = float(np.mean(m["epoch_times"]))
        rows.append({
            "Fusion Strategy": name_map.get(key, key) if name_map else key,
            "Avg Valid Loss": avg_valid_loss,
            "Best Valid Loss": float(m["best_valid_loss"]),
            "Num of params": int(m["num_params"]),
            "Avg time per epoch (min:s)": avg_epoch_time,
            "GPU Memory (MB, max)": float(m["max_gpu_mem_mb"]),
        })
    return pd.DataFrame(rows)


def build_pairwise_downsampling_tables(metrics, name_map):
    """
    Automatically finds all model pairs <base>_maxpool and <base>_strided,
    builds comparison tables, and uses display names from name_map.

    Args:
        metrics (dict): Contains all logged results.
        name_map (dict): Maps base model names to display strings.

    Returns:
        dict: { display_name: DataFrame }
    """
    tables = {}

    # 1. Find all base names
    base_names = set()
    for key in metrics.keys():
        if key.endswith("_pool"):
            base_names.add(key.replace("_pool", ""))
        elif key.endswith("_stride"):
            base_names.add(key.replace("_stride", ""))

    # 2. Build table for each base
    for base in base_names:
        key_pool   = f"{base}_pool"
        key_stride = f"{base}_stride"

        if key_pool not in metrics or key_stride not in metrics:
            print(f"Skipping {base}: missing maxpool/strided pair.")
            continue

        m_pool   = metrics[key_pool]
        m_stride = metrics[key_stride]

        # Collect fields needed for Task 4
        val_loss_pool   = float(m_pool["best_valid_loss"])
        val_loss_stride = float(m_stride["best_valid_loss"])

        params_pool     = int(m_pool["num_params"])
        params_stride   = int(m_stride["num_params"])

        time_pool       = float(np.sum(m_pool["epoch_times"]))
        time_stride     = float(np.sum(m_stride["epoch_times"]))

        acc_pool        = float(m_pool["best_valid_acc"])
        acc_stride      = float(m_stride["best_valid_acc"])

        # Human-readable display name
        display_name = name_map.get(base, base)

        # Build the DataFrame
        df = pd.DataFrame([
            {
                "Metric": "Validation Loss (best)",
                "MaxPool2d": val_loss_pool,
                "Strided Conv": val_loss_stride,
                "Difference (Strided - MaxPool)": val_loss_stride - val_loss_pool,
            },
            {
                "Metric": "Parameters",
                "MaxPool2d": params_pool,
                "Strided Conv": params_stride,
                "Difference (Strided - MaxPool)": params_stride - params_pool,
            },
            {
                "Metric": "Training Time (s)",
                "MaxPool2d": time_pool,
                "Strided Conv": time_stride,
                "Difference (Strided - MaxPool)": time_stride - time_pool,
            },
            {
                "Metric": "Final Accuracy",
                "MaxPool2d": acc_pool,
                "Strided Conv": acc_stride,
                "Difference (Strided - MaxPool)": acc_stride - acc_pool,
            },
        ])

        tables[display_name] = df

    return tables


def build_grouped_dataset(name: str, pairs: dict, persistent: bool = True, overwrite: bool = True):
    """
    Build a grouped FiftyOne dataset with slices: 'rgb' and 'lidar' from matched pairs.

    Args:
        name: FiftyOne dataset name.
        pairs: dict[class_name -> list[{stem, rgb, lidar}]] as returned by match_lidar_rgb.
        persistent: whether to persist the dataset in FiftyOne.
        overwrite: delete existing dataset with same name.

    Returns:
        fiftyone.core.dataset.Dataset
    """

    # Delete existing dataset if it exists
    if overwrite and name in fo.list_datasets():
        print(f"Deleting existing dataset: {name}")
        fo.delete_dataset(name)

    # Create new grouped dataset
    print(f"Creating new dataset: {name}")
    dataset = fo.Dataset(name, persistent=persistent)
    dataset.add_group_field("group", default="rgb")

    print(f"✅ Created grouped dataset: {name}")
    
    samples = []
    for class_name, class_pairs in pairs.items():
        label_str = "cube" if class_name == "cubes" else "sphere"

        for item in class_pairs:
            # Create group
            group = fo.Group()

            # Create RGB sample
            rgb_sample = fo.Sample(
                filepath=str(item["rgb"]),
                group=group.element("rgb"),
                label=fo.Classification(label=label_str),
            )

            # Create PCD sample
            pcd_sample = fo.Sample(
                filepath=str(item["lidar"]),
                group=group.element("lidar"),
                label=fo.Classification(label=label_str),
            )

            samples.extend([rgb_sample, pcd_sample])

    # Add all samples to dataset
    dataset.add_samples(samples)

    return dataset


def plot_class_distributions(
    total_per_class: dict,
    splits: dict,
    split_names=("train", "val"),
    figsize=(8, 6),
    title_full="Class distribution (full dataset)",
    title_splits="Train vs Validation",
):
    """
    Plot class distributions for the full dataset and for two dataset splits.

    Creates a two-panel bar chart showing (1) total samples per class and
    (2) class counts in two splits (e.g., train vs validation).
    """
    # --- Data ---
    class_names = list(total_per_class.keys())
    counts_full = [total_per_class[c] for c in class_names]

    split1_name, split2_name = split_names
    split1_counts = [len(splits[split1_name][c]) for c in class_names]
    split2_counts = [len(splits[split2_name][c]) for c in class_names]

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # left: full dataset distribution
    axes[0].bar(class_names, counts_full, color="steelblue")
    axes[0].set_title(title_full)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    # right: train vs validation split
    x = range(len(class_names))
    width = 0.35

    axes[1].bar([i - width/2 for i in x], split1_counts, width=width, label=split1_name.capitalize())
    axes[1].bar([i + width/2 for i in x], split2_counts, width=width, label=split2_name.capitalize())
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_title(title_splits)
    axes[1].set_xlabel("Class")
    axes[1].legend()

    # show plots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)

    return fig, axes


def plot_similarity_matrix(
        model, 
        dataloader, 
        device, 
        max_b=32, 
        normalize="softmax",
        temperature=1.0,
        title="CILP similarity matrix (RGB → LiDAR)"
):
    """
    Plot a (possibly normalized) RGB→LiDAR similarity matrix.

    Returns:
        fig (matplotlib.figure.Figure)
    """
    model.eval()
    rgb, lidar, _ = next(iter(dataloader))
    rgb, lidar = rgb.to(device), lidar.to(device)

    with torch.no_grad():
        logits_per_img, _ = model(rgb, lidar)  # (B, B)

    B = min(logits_per_img.size(0), max_b)
    sim = logits_per_img[:B, :B].cpu()

    if normalize == "softmax":
        sim = Func.softmax(sim / temperature, dim=1)

    sim = sim.detach().cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(sim)
    ax.set_title(title + (" (softmax)" if normalize == "softmax" else ""))
    ax.set_xlabel("LiDAR index")
    ax.set_ylabel("RGB index")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    return fig


def plot_retrieval_examples(
    model,
    dataloader,
    device,
    k=5,
    mode="mismatches_then_correct",  # "correct" | "mismatches" | "mismatches_then_correct" | "random"
    normalize="softmax",             # None | "softmax"
    temperature=1.0,
    title="RGB → LiDAR retrieval examples",
):
    """
    Creates a 2xk grid: top row = RGB queries, bottom row = retrieved LiDAR.
    Returns:
        fig (matplotlib.figure.Figure)
        table (wandb.Table)  [created lazily; requires wandb at runtime when used]
    """
    model.eval()
    rgb, lidar, *_ = next(iter(dataloader))
    rgb, lidar = rgb.to(device), lidar.to(device)

    with torch.no_grad():
        logits_per_img, _ = model(rgb, lidar)  # (B,B)

    B = logits_per_img.size(0)
    gt = torch.arange(B, device=device)

    # probabilities + preds + confidence
    if normalize == "softmax":
        probs = Func.softmax(logits_per_img / temperature, dim=1)
        preds = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
    else:
        preds = logits_per_img.argmax(dim=1)
        conf = torch.softmax(logits_per_img / temperature, dim=1).max(dim=1).values  # for display only

    correct_mask = preds.eq(gt)

    # ---- choose which indices to show ----
    all_idx = torch.arange(B, device=device)
    if mode == "random":
        idx_show = all_idx[torch.randperm(B, device=device)[:min(k, B)]]
    elif mode == "mismatches":
        wrong = all_idx[~correct_mask]
        idx_show = wrong[:min(k, len(wrong))] if len(wrong) > 0 else all_idx[torch.argsort(conf)[:min(k, B)]]
    elif mode == "correct":
        corr = all_idx[correct_mask]
        if len(corr) == 0:
            idx_show = all_idx[torch.argsort(conf)[:min(k, B)]]
        else:
            corr_sorted = corr[torch.argsort(conf[corr], descending=True)]
            idx_show = corr_sorted[:min(k, len(corr_sorted))]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    k_eff = int(min(k, len(idx_show)))

    # ---- build wandb.Table lazily (so function can be used without wandb init) ----
    table = wandb.Table(columns=[
        "RGB index", "Predicted LiDAR index", "GT LiDAR index", "Correct", "Confidence"
    ])
    for i in idx_show[:k_eff]:
        i = int(i)
        table.add_data(i, int(preds[i]), int(gt[i]), bool(correct_mask[i]), float(conf[i].item()))

    # ---- plot 2 x k grid ----
    fig, axes = plt.subplots(2, k_eff, figsize=(2.8 * k_eff, 5.2))
    if k_eff == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, i in enumerate(idx_show[:k_eff]):
        i = int(i)
        p = int(preds[i])
        is_correct = bool(correct_mask[i])
        c = float(conf[i].item())

        # RGB query
        rgb_img = rgb[i].permute(1, 2, 0).detach().cpu().numpy()
        rgb_img = np.clip(rgb_img, 0, 1)
        axes[0, col].imshow(rgb_img)
        axes[0, col].set_title(f"RGB {i}")
        axes[0, col].axis("off")

        # Retrieved LiDAR (depth map)
        lidar_img = lidar[p].squeeze().detach().cpu().numpy()
        axes[1, col].imshow(lidar_img, cmap="viridis")
        axes[1, col].set_title(f"LiDAR {p} | p={c:.2f} " + ("✅" if is_correct else "❌"))
        axes[1, col].axis("off")

        # colored border for readability
        for sp in axes[1, col].spines.values():
            sp.set_visible(True)
            sp.set_linewidth(3)
            sp.set_edgecolor("green" if is_correct else "red")

    fig.suptitle(f"{title} (mode={mode})", y=1.02)
    fig.tight_layout()

    return fig, table
