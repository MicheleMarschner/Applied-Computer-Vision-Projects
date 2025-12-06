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