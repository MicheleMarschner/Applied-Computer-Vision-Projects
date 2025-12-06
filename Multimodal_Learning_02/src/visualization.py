import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    plt.figure(figsize=(8,5))

    # Auto-generate x-axis based on first model
    any_key = next(iter(loss_dict))
    epochs = range(len(loss_dict[any_key]))

    for model_name, losses in loss_dict.items():
        plt.plot(epochs, losses, label=model_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()


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

        acc_pool        = float(m_pool["final_valid_acc"])
        acc_stride      = float(m_stride["final_valid_acc"])

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