import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from fiftyone import ViewField as F
from collections import Counter
import pandas as pd
from collections import defaultdict
from PIL import Image
import numpy as np
import math
import seaborn as sns
import pandas as pd

import torch


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(image[0].detach().cpu()))


def show_generated_images_grid(images, prompts, w_tests):
    """
    Display all generated images in a grid with guidance weight and prompt labels.

    Args:
        images (torch.Tensor): Shape (N, 3, H, W), values in [-1, 1].
        prompts (list[str]): Text prompts (length P).
        w_tests (list[float]): Guidance weights (length W).
    """

    images = (images + 1) / 2               # Shift from [-1, 1] space to [0, 1] space
    images = images.clamp(0, 1)             # Clip any artifacts that fell outside the valid range as 
                                            # because of Guidance w the model produced values like -1.77 and 2.33
    P = len(prompts)
    W = len(w_tests)
    N = images.shape[0]

    assert N == P * W, "Number of images must equal len(prompts) * len(w_tests)"

    plt.figure(figsize=(2.2 * W, 2.4 * P))

    for i in range(N):
        w_idx = i // P
        p_idx = i % P

        img = images[i].detach().cpu()
        ax = plt.subplot(P, W, p_idx * W + w_idx + 1)
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"w={w_tests[w_idx]:+.1f}", fontsize=8)
        ax.axis("off")

        # Optional: show prompt only on the left
        if w_idx == 0:
            ax.set_ylabel(prompts[p_idx], fontsize=8)

    plt.tight_layout()
    plt.show()


def show_images_from_disk_in_saved_order(
    saved_samples,
    cols=5,
    figsize_per_img=(3, 3),
):
    """
    Display images from disk in the exact order of `saved_samples`.

    Each entry in saved_samples:
        (filepath, prompt, w_val)

    Titles show:
        i = generation index
        w = guidance weight
        prompt = text prompt
    """
    n = len(saved_samples)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_img[0] * cols, figsize_per_img[1] * rows),
    )

    axes = axes.flatten() if n > 1 else [axes]

    for i, (filepath, prompt, w_val) in enumerate(saved_samples):
        img = Image.open(filepath).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(
            f"i={i}  w={w_val:+.1f}\n{prompt}",
            fontsize=9,
        )
        axes[i].axis("off")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    

def compare_generated_vs_real_roses(
    generated_images, prompt_idx, prompts, w_tests, real_rose_path
):
    P = len(prompts)
    n_cols = 1 + len(w_tests)

    plt.figure(figsize=(2.2 * n_cols, 3))

    # --- real image (explicit) ---
    img_real = Image.open(real_rose_path).convert("RGB")
    ax = plt.subplot(1, n_cols, 1)
    ax.imshow(img_real)
    ax.set_title("Real red rose")
    ax.axis("off")

    # --- generated images ---
    for w_idx, w in enumerate(w_tests):
        i = w_idx * P + prompt_idx
        img = (generated_images[i].cpu() + 1) / 2

        ax = plt.subplot(1, n_cols, w_idx + 2)
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f"w={w:+.1f}")
        ax.axis("off")

    plt.suptitle(prompts[prompt_idx], y=1.05)
    plt.tight_layout()
    plt.show()


def plot_samples_from_view(view, n=10, cols=5):
    """Plot samples from a FiftyOne view with conditioning label, prediction, and confidence."""
    samples = view.take(n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 2.4, rows * 2.4))

    for i, sample in enumerate(samples):
        # Load image from the filepath stored in the FiftyOne sample
        img = Image.open(sample.filepath)

        # Read stored fields (adapted to your dataset schema)
        cond = sample["conditioning"].label
        pred = sample["prediction"].label
        conf = sample["prediction"].confidence

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")

        # Show conditioning label (requested digit), predicted label (digit/IDK), and confidence
        ax.set_title(f"Cond: {cond} | Pred: {pred}\nConf: {conf:.2f}", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_confidence_histograms(dataset, bins=20, figsize=(7, 4)):
    """Plot confidence histograms for IDK vs non-IDK predictions in a FiftyOne dataset."""
    idk_conf = dataset.match(F("prediction.label") == "IDK").values("prediction.confidence")
    ok_conf  = dataset.match(F("prediction.label") != "IDK").values("prediction.confidence")

    plt.figure(figsize=figsize)
    plt.hist(ok_conf, bins=bins, alpha=0.7, label="Non-IDK")
    plt.hist(idk_conf, bins=bins, alpha=0.7, label="IDK")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    return plt.figure


def idk_frequency_table(dataset):
  """Return a table of IDK counts and rates per conditioning digit."""
  idk_view = dataset.match(F("prediction.label") == "IDK")
  counts = Counter(s["conditioning"].label for s in idk_view)
  all_counts = Counter(s["conditioning"].label for s in dataset)

  df_idk = pd.DataFrame(sorted(all_counts.items(), key=lambda x: int(x[0])), columns=["digit", "total"])
  df_cnt = pd.DataFrame(sorted(counts.items(), key=lambda x: int(x[0])), columns=["digit", "idk_count"])

  df = df_idk.merge(df_cnt, on="digit", how="left").fillna({"idk_count": 0})
  df["idk_count"] = df["idk_count"].astype(int)

  total_idk = max(len(idk_view), 1)
  df["idk_share"] = (df["idk_count"] / total_idk).round(3)
  df["idk_rate_per_digit"] = (df["idk_count"] / df["total"]).round(3)

  return df


def guidance_stats(dataset, guidance_list):
    """Print coverage and IDK rate for each guidance weight."""
    stats = defaultdict(dict)

    for w in sorted(guidance_list):
        view = dataset.match(F("guidance_w") == w)
        total = len(view)
        idk = len(view.match(F("prediction.label") == "IDK"))

        stats[w]["coverage"] = 1 - idk / total if total > 0 else 0.0
        stats[w]["idk_rate"] = idk / total if total > 0 else 0.0

    for w in sorted(stats):
        print(f"w={w}: coverage={stats[w]['coverage']:.2%}, idk={stats[w]['idk_rate']:.2%}")

    return stats


def report_coverage_accuracy(dataset):
    """Print coverage and accuracy metrics for an IDK classifier on a FiftyOne dataset."""
    total = len(dataset)

    idk_view = dataset.match(F("prediction.label") == "IDK")
    idk = len(idk_view)

    covered_view = dataset.match(F("prediction.label") != "IDK")
    covered = len(covered_view)

    coverage = covered / total if total > 0 else 0.0

    correct_covered_view = covered_view.match(
        F("prediction.label") == F("conditioning.label")
    )
    correct = len(correct_covered_view)

    acc_covered = correct / covered if covered > 0 else 0.0
    acc_standard = correct / total if total > 0 else 0.0

    print(f"Total Test Images:    {total}")
    print(f"IDK Responses:        {idk}")
    print(f"Covered Responses:    {covered}")
    print("-" * 30)
    print(f"COVERAGE:             {coverage:.2%}")
    print(f"ACCURACY (Covered):   {acc_covered:.2%}")
    print(f"ACCURACY (Standard):  {acc_standard:.2%}")
    print("=" * 50)

    return {
        "total": total,
        "idk": idk,
        "covered": covered,
        "coverage": coverage,
        "correct_covered": correct,
        "accuracy_covered": acc_covered,
        "accuracy_standard": acc_standard,
    }


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def confidence_report(
    dataset,
    label_path="prediction.label",
    conf_path="prediction.confidence",
    idk_label="IDK",
    bins=20,
    figsize=(8, 3),
    density=False,
    show=True,
    show_table=True,
):
    """Summarize + plot confidence for digit predictions vs IDK in a FiftyOne Dataset/View."""
    # Pull fields (may contain None)
    confs_raw = np.array(dataset.values(conf_path), dtype=object)
    labels_raw = np.array(dataset.values(label_path), dtype=object)

    # Clean confidence -> float, missing -> NaN
    confs = np.full(confs_raw.shape, np.nan, dtype=float)
    has_conf = np.array([c is not None for c in confs_raw], dtype=bool)
    confs[has_conf] = confs_raw[has_conf].astype(float)

    # Clean labels -> str, missing handled by mask
    has_label = np.array([l is not None for l in labels_raw], dtype=bool)
    labels = labels_raw.astype(str)

    # Valid rows: label+confidence present and finite
    valid = has_conf & has_label & np.isfinite(confs)

    # Groups
    idk_mask = valid & (labels == idk_label)
    digit_mask = valid & (labels != idk_label)

    digits = confs[digit_mask]
    idks = confs[idk_mask]

    def _stats(x):
        return None if x.size == 0 else {
            "N": int(x.size),
            "Mean": float(x.mean()),
            "Median": float(np.median(x)),
        }

    stats_digits = _stats(digits)
    stats_idk = _stats(idks)

    # Make a neat table for reporting
    stats_table = pd.DataFrame({"Digits": stats_digits, "IDK": stats_idk}).T
    stats_table.index.name = "Group"

    if show_table:
        try:
            from IPython.display import display
            display(stats_table.style.format({"Mean": "{:.3f}", "Median": "{:.3f}"}))
        except Exception:
            print(stats_table.to_string())

    # Valid arrays for downstream plotting
    labels_valid = labels[valid]
    confs_valid = confs[valid]

    # Histogram plot
    plt.figure(figsize=figsize)
    plt.hist(digits, bins=bins, alpha=0.7, label="Digit", density=density)
    plt.hist(idks, bins=bins, alpha=0.7, label="IDK", density=density)
    plt.title("Confidence distributions: digits vs IDK")
    plt.xlabel("Confidence")
    plt.ylabel("Density" if density else "Count")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()

    return {
        "labels": labels_valid,
        "confs": confs_valid,
        "digits": digits,
        "idks": idks,
        "stats_digits": stats_digits,
        "stats_idk": stats_idk,
        "stats_table": stats_table,
        "valid_mask": valid,
    }



def boxplot_confidence_by_label(
    report: dict,
    order=None,
    figsize=(10, 3),
    title="Confidence by predicted label",
    xlabel="Predicted label",
    ylabel="Confidence",
    grid=False,
    show=True,
):
    """Boxplot confidence grouped by predicted label using `report['labels']` + `report['confs']`."""
    # Pull arrays produced by confidence_report() (already cleaned to "valid" rows)
    labels = np.asarray(report["labels"], dtype=object)
    confs = np.asarray(report["confs"], dtype=float)

    # Default order for MNIST-style labels: IDK first, then 0..9
    if order is None:
        order = ["IDK"] + [str(i) for i in range(10)]

    # Build a small table for pandas' boxplot, enforce category order on x-axis
    df = pd.DataFrame({"label": labels.astype(str), "confidence": confs})
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)

    # Plot: one box per label category
    plt.figure(figsize=figsize)
    df.boxplot(column="confidence", by="label", grid=grid)
    plt.title(title)
    plt.suptitle("")  # remove pandas' default "Boxplot grouped by ..."
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if show:
        plt.show()

    return df


def make_grid(filepaths, n=25, pad=2, bg=255):
    """Create a square-ish n-image grid (as a single numpy array) from filepaths."""
    paths = list(filepaths)[:n]
    imgs = [Image.open(p).convert("L") for p in paths]  # MNIST -> grayscale
    w, h = imgs[0].size

    # Aim for a roughly square layout (e.g., 25 -> 5x5)
    cols = int(math.sqrt(n))
    rows = int(math.ceil(n / cols))

    # Pre-allocate a white canvas and paste each image in
    grid = np.full(
        (rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad),
        bg,
        dtype=np.uint8,
    )

    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        y0, x0 = r * (h + pad), c * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = np.array(img)

    return grid


def show_confidence_grids(
    dataset,
    label_path="prediction.label",
    conf_path="prediction.confidence",
    idk_label="IDK",
    n=25,
    pad=2,
    figsize=(15, 5),
    sort_idk="low",  # "low" | "high" | None
):
    """Show 3 grids: top-conf digits, low-conf digits, and IDK samples (optionally sorted)."""
    # Build the three views from the dataset
    digits_view = dataset.match(F(label_path) != idk_label)
    top_digits  = digits_view.sort_by(conf_path, reverse=True)
    low_digits  = digits_view.sort_by(conf_path)

    idk_view = dataset.match(F(label_path) == idk_label)
    if sort_idk == "low":
        idk_view = idk_view.sort_by(conf_path)
    elif sort_idk == "high":
        idk_view = idk_view.sort_by(conf_path, reverse=True)

    # Convert each view into a single grid image
    top_grid = make_grid(top_digits.values("filepath"), n=n, pad=pad)
    low_grid = make_grid(low_digits.values("filepath"), n=n, pad=pad)
    idk_grid = make_grid(idk_view.values("filepath"), n=n, pad=pad)

    # Plot as a "subfigure" style row with 3 columns
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    ax[0].imshow(top_grid, cmap="gray")
    ax[0].set_title("Top confident digit predictions")
    ax[0].axis("off")

    ax[1].imshow(low_grid, cmap="gray")
    ax[1].set_title("Lowest-confidence digit predictions")
    ax[1].axis("off")

    ax[2].imshow(idk_grid, cmap="gray")
    ax[2].set_title("IDK predictions")
    ax[2].axis("off")

    fig.tight_layout()
    plt.show()

    # Return views too, so you can reuse them later (e.g., for annotation/export)
    return {"top_digits": top_digits, "low_digits": low_digits, "idk_view": idk_view}


def labeled_eval_table(
    dataset,
    gt_path="gt.label",
    pred_path="prediction.label",
    conf_path="prediction.confidence",
    order=None,
    require_labels=True,
):
    """Summarize accuracy + per-pred-label stats on samples with manual gt labels."""
    assert dataset is not None, "dataset is None"

    # Pull once into a dataframe (raw values can include None)
    df = pd.DataFrame({
        "gt":   dataset.values(gt_path),
        "pred": dataset.values(pred_path),
        "conf": dataset.values(conf_path),
    })

    # Keep only manually labeled samples
    df = df[df["gt"].notna()].copy()
    N = len(df)
    print(f"Labeled samples: {N} / {len(dataset)}")
    if require_labels:
        assert N > 0, f"No manual labels found. Fill {gt_path} in the FiftyOne App first."

    # Clean types
    df["gt"] = df["gt"].astype(str)
    df["pred"] = df["pred"].astype(str)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")  # None -> NaN
    df["correct"] = df["pred"] == df["gt"]

    # --- Prints (same as your version) ---
    n_correct = int(df["correct"].sum())
    n_wrong = int((~df["correct"]).sum())
    print(f"\nCorrect: {n_correct} ({n_correct/N:.1%})")
    print(f"Wrong:   {n_wrong} ({n_wrong/N:.1%})")

    is_idk_pred = df["pred"] == "IDK"
    n_idk = int(is_idk_pred.sum())
    n_digit = int((~is_idk_pred).sum())
    print(f"\nPredicted IDK:   {n_idk} ({n_idk/N:.1%})")
    print(f"Predicted DIGIT: {n_digit} ({n_digit/N:.1%})")

    # --- Table: per predicted label ---
    per_pred = (
        df.groupby("pred")
          .agg(
              count=("pred", "size"),
              labeled_pct=("pred", lambda s: 100 * len(s) / N),
              wrong_pct=("correct", lambda s: 100 * (~s).mean()),
              mean_conf=("conf", "mean"),
              median_conf=("conf", "median"),
          )
          .reset_index()
          .rename(columns={
              "pred": "label",
              "labeled_pct": "labeled [%]",
              "wrong_pct": "wrong [%]",
          })
    )

    # order labels: 0..9 then IDK (others afterwards)
    if order is None:
        order = [str(i) for i in range(10)] + ["IDK"]
    per_pred["__order"] = per_pred["label"].apply(lambda x: order.index(x) if x in order else 999)
    per_pred = per_pred.sort_values(["__order", "label"]).drop(columns="__order")

    # format: 3 decimals
    for c in ["labeled [%]", "wrong [%]", "mean_conf", "median_conf"]:
        per_pred[c] = per_pred[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")

    per_pred = per_pred[["label", "count", "labeled [%]", "wrong [%]", "mean_conf", "median_conf"]]
    per_pred = per_pred.rename(columns={"label": ""})  # nicer display

    print("\nPer predicted label:")
    display(per_pred)

    return df, per_pred



def plot_confusion_matrix_seaborn(df, gt_col="gt", pred_col="pred", order=None, figsize=(7,6)):
    """Plot a GT×Pred confusion matrix using seaborn heatmap."""
    if order is None:
        order = [str(i) for i in range(10)] + ["IDK"]

    cm = pd.crosstab(df[gt_col], df[pred_col], dropna=False).reindex(
        index=order, columns=order, fill_value=0
    )

    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, annot=True, fmt="d", cbar=True, square=True)
    ax.set_title("Confusion matrix (GT rows × Pred cols)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground truth label")
    plt.tight_layout()
    plt.show()
    return cm


def plot_accuracy_coverage(df, figsize=(6,4)):
    """Accuracy-on-covered vs coverage curve."""
    plt.figure(figsize=figsize)
    plt.plot(df["coverage"], df["acc_covered"], marker="o", markersize=3)
    plt.xlabel("Coverage (fraction not IDK)")
    plt.ylabel("Accuracy on covered")
    plt.title("Accuracy–Coverage curve (cascaded IDK policy)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_reject_tradeoff(df, figsize=(6,4)):
    """Good vs wasted rejects vs threshold."""
    plt.figure(figsize=figsize)
    plt.plot(df["threshold"], df["good_reject"], label="Good rejects")
    plt.plot(df["threshold"], df["wasted_reject"], label="Wasted rejects")
    plt.xlabel("Threshold")
    plt.ylabel("Count (on labeled subset)")
    plt.title("Reject trade-off vs threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_matplotlib(cm, labels, figsize=(7,6), title="Confusion matrix (GT × Pred)"):
    """Plot a confusion matrix DataFrame (cm) using matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm.values)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground truth label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.iat[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()



def pick_rows_for_thresholds(df, report_ts, tol=1e-6):
    """
    For each desired threshold in report_ts, pick the closest row in df (robust to linspace grids).
    """
    out_idx = []
    for t in report_ts:
        idx = (df["threshold"] - t).abs().idxmin()
        if abs(df.loc[idx, "threshold"] - t) > tol:
            print(f"Note: requested t={t} not in df; using closest t={df.loc[idx,'threshold']:.6f}")
        out_idx.append(idx)
    return df.loc[out_idx].copy()

def make_threshold_report_table(df):
    """
    Returns a compact table for specific thresholds, with:
    - rate columns rounded to 3 decimals
    - count columns kept as ints
    """
    report_ts = np.array([0.95, 0.90, 0.85, 0.80, 0.78, 0.70, 0.60, 0.50, 0.10], dtype=float)

    cols = [
        "threshold",
        "coverage",
        "acc_covered",
        "idk_rate",
        "good_reject",
        "wasted_reject",
        "hard_reject",
        "soft_reject",
        "n_answered",
        "n_abstained",
    ]

    sub = pick_rows_for_thresholds(df, report_ts)[cols]

    # round rate-like columns; keep counts as ints
    rate_cols = ["threshold", "coverage", "acc_covered", "idk_rate"]
    sub[rate_cols] = sub[rate_cols].round(3)

    count_cols = ["good_reject", "wasted_reject", "hard_reject", "soft_reject", "n_answered", "n_abstained"]
    sub[count_cols] = sub[count_cols].astype(int)

    return sub.sort_values("threshold", ascending=False)