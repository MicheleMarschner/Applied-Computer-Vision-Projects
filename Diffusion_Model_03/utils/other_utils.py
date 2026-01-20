
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datetime import datetime
import os
import random
import numpy as np
import cv2
import albumentations as A
import math
from torchvision.transforms import ToPILImage
import fiftyone as fo

from utils import config


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(image[0].detach().cpu()))


def get_timestamp():
    """Return a human-readable timestamp string (YYYY-MM-DD_HH-MM) for naming files or runs."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def set_seeds(seed=51):
    """
    Set seeds for complete reproducibility across all libraries and operations.

    Args:
        seed (int): Random seed value
    """
    # Set environment variables before other imports
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # OpenCV
    cv2.setRNGSeed(seed)

    # Albumentations (for data augmentation)
    try:
        A.seed_everything(seed)
    except AttributeError:
        # Older versions of albumentations
        pass

    # PyTorch deterministic algorithms (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some operations don't have deterministic implementations
        print("Warning: Some operations may not be deterministic")

    print(f"All random seeds set to {seed} for reproducibility")


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


class MyDataset(Dataset):
    """
    Simple image dataset that loads RGB images from class-based folders.
    """
    def __init__(self, root_dir, rgb_transform=None, classes=config.CLASSES):
        self.root_dir = Path(root_dir)
        self.rgb_transform = rgb_transform
        self.samples = []  # list of (rgb_path, lidar_path, label)

        for label, class_name in enumerate(classes):
            rgb_dir = self.root_dir / class_name

            rgb_files = sorted(
                list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))
            )

            for rgb_path in rgb_files:
                self.samples.append((rgb_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        rgb_path, label = self.samples[idx]

        # RGB: PIL -> tensor (CPU)
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb = self.rgb_transform(rgb_img) if self.rgb_transform else rgb_img

        return rgb, label


class GeneratedListDataset(Dataset):
    """
    Loads images from a list of (filepath, prompt, w) tuples.
    """
    def __init__(self, saved_samples, transform=None):
        self.samples = saved_samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # stored as (filepath, prompt, w_val)
        filepath = self.samples[idx][0]

        img = Image.open(filepath).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
    

def plot_samples_from_view(view, n=10, threshold=0.5, cols=5):
    """Plot samples from a FiftyOne view with GT, prediction, confidence, and threshold."""
    samples = view.take(n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 2.2, rows * 2.2))

    for i, sample in enumerate(samples):
        img = Image.open(sample.filepath)

        gt = sample["ground_truth"].label
        pred = sample["prediction_with_idk"].label
        conf = sample["prediction_with_idk"].confidence

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")

        ax.set_title(
            f"GT: {gt} | Pred: {pred}\n"
            f"Conf: {conf:.2f}  (τ={threshold})",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()


def save_samples_to_disk(generated_images, text_prompts, w, save_dir, repetition_size, n_weights, n_samples):
    """
    Save generated images ([-1, 1]) as PNGs and return per-sample metadata
    (path, prompt, guidance weight w), assuming sampling order is grouped by
    repetitions × w_tests × text_prompts.
    """

    # Save generated images to disk for downstream evaluation
    to_pil = ToPILImage()

    # Track saved image paths together with their prompts and guidance values
    saved_samples = []

    print("Saving images to disk...")
    assert len(generated_images) == n_samples, (
        f"generated_images={len(generated_images)} != {n_samples}"
    )

    for i, img_tensor in enumerate(generated_images):
        # Recover prompt and guidance value from the sampling order
        idx_within_rep = i % (repetition_size * n_weights)
        prompt = text_prompts[idx_within_rep % repetition_size]
        w_val = w[idx_within_rep // repetition_size]

        # Map model output from [-1, 1] to [0, 1] for image saving and clip any artifacts that fell outside the valid range
        img_norm = ((img_tensor + 1) / 2).clamp(0, 1).detach().cpu()
        pil_img = to_pil(img_norm)

        filename = os.path.join(
            save_dir, f"flower_w{w_val:+.1f}_p{idx_within_rep % repetition_size}_{i}.png"
        )
        pil_img.save(filename)

        saved_samples.append((filename, prompt, float(w_val)))

    print("All images saved.")
    return saved_samples

# Mnist Model Architecture
class MNISTClassifier(nn.Module):
    '''Lightweight CNN baseline for MNIST digit classification (used for IDK experiments).'''
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
    
def create_FiftyOne_dataset(samples, extracted_embeddings, clip_scores):
    """
    Create (or replace) a FiftyOne dataset of generated images with metadata
    (prompt, guidance w, CLIP score) and flattened U-Net embeddings for
    FiftyOne Brain analysis (uniqueness/representativeness).
    """

    # Delete existing dataset if it exists
    if config.FIFTYONE_DATASET_EXPERIMENTS_NAME in fo.list_datasets():
        print(f"Deleting existing dataset: {config.FIFTYONE_DATASET_EXPERIMENTS_NAME}")
        fo.delete_dataset(config.FIFTYONE_DATASET_EXPERIMENTS_NAME)

    dataset = fo.Dataset(name=config.FIFTYONE_DATASET_EXPERIMENTS_NAME)

    # Build a FiftyOne dataset where each image is paired with prompt, guidance w,
    # CLIP score, and a flattened U-Net embedding (used for embedding-based analysis)
    fo_samples = []

    print("Building FiftyOne dataset...")

    for i, (filepath, prompt, w_val) in enumerate(samples):
        # FiftyOne Brain expects a 1D embedding vector per sample for distance computations
        raw_embedding = extracted_embeddings[i]                 # e.g., (512, 8, 8)
        flat_embedding = raw_embedding.flatten().cpu().numpy() # (512*8*8,)

        sample = fo.Sample(filepath=filepath)

        # Store fields for filtering and analysis in the FiftyOne App
        sample["ground_truth"] = fo.Classification(label=prompt)
        sample["w"] = float(w_val)
        sample["clip_score"] = float(clip_scores[i])
        sample["unet_embedding"] = flat_embedding

        fo_samples.append(sample)

    # Add all samples in one call for efficiency
    dataset.add_samples(fo_samples)
    print(f"Added {len(fo_samples)} samples to the dataset.")

    return dataset

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confidence_distribution(model, loader, device):
    """
    Computes and visualizes the classifier's confidence distribution, split by
    whether predictions are correct or incorrect.

    Output:
      - A plot with two confidence histograms (correct vs incorrect)
      - Two lists: (confidences_correct, confidences_incorrect)

    Confidence definition:
      - For each image, the classifier produces a probability distribution over classes.
      - Confidence is the maximum class probability (max softmax / max probability).

    Probability computation assumption:
      - This code assumes `model(images)` returns log-probabilities (e.g., output of log_softmax).
      - In that case, `torch.exp(log_probs)` converts them to probabilities.
      - If `model(images)` returns logits instead, replace `torch.exp(output)` with
        `torch.softmax(output, dim=1)`.
    """
    model.eval()

    # Store max-probability confidences for correct vs incorrect predictions
    conf_correct = []
    conf_incorrect = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Model output is treated as log-probabilities; exponentiation yields probabilities
            log_probs = model(images)
            probs = torch.exp(log_probs)

            # Confidence: max class probability; Prediction: argmax class index
            max_confidence, pred_class = torch.max(probs, dim=1)

            # Separate confidences by correctness to analyze calibration/uncertainty behavior
            is_correct = pred_class.eq(labels)
            conf_correct.extend(max_confidence[is_correct].cpu().numpy())
            conf_incorrect.extend(max_confidence[~is_correct].cpu().numpy())

    # --------------------- Plot ---------------------
    # Visualization goal: show whether incorrect predictions tend to have lower confidence.
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 110

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    color_correct = "#8EC5FF"
    color_incorrect = "#FFB3C7"

    # Step histograms make overlap/comparison easy (distinct from filled KDE plot style)
    ax.hist(
        conf_correct,
        bins=30,
        density=True,
        histtype="step",
        linewidth=2.2,
        alpha=0.95,
        label="Correct",
        color=color_correct,
    )
    ax.hist(
        conf_incorrect,
        bins=30,
        density=True,
        histtype="step",
        linewidth=2.2,
        alpha=0.95,
        label="Incorrect",
        color=color_incorrect,
    )

    # Median lines provide a simple summary of where the bulk of each distribution lies
    med_correct = float(np.median(conf_correct)) if len(conf_correct) else np.nan
    med_incorrect = float(np.median(conf_incorrect)) if len(conf_incorrect) else np.nan
    if not np.isnan(med_correct):
        ax.axvline(med_correct, linestyle="--", linewidth=1.8, alpha=0.8, color=color_correct)
    if not np.isnan(med_incorrect):
        ax.axvline(med_incorrect, linestyle="--", linewidth=1.8, alpha=0.8, color=color_incorrect)

    ax.set_title("Confidence distribution (Correct vs Incorrect)", fontsize=13, pad=10)
    ax.set_xlabel("Confidence (max class probability)", fontsize=12, labelpad=8)
    ax.set_ylabel("Density", fontsize=12, labelpad=8)

    ax.set_xlim(0.0, 1.0)
    ax.tick_params(direction="out", length=5, width=1.4, labelsize=11)

    sns.despine(ax=ax)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.show()

    return conf_correct, conf_incorrect


def find_stability_limit(coverages, accuracies):
    """
    Identifies the "elbow" point of an accuracy–coverage curve.

    Intended use:
      - When sweeping an IDK confidence threshold, coverage decreases as the model
        rejects more samples, while accuracy on the remaining (accepted) samples
        typically increases.
      - The elbow is a heuristic point where returns diminish: accuracy gains start
        flattening relative to the loss in coverage.

    Method:
      - Treat the curve as points (coverage_i, accuracy_i).
      - Compute perpendicular distance of each point to the straight line between
        the first and last points.
      - Return the index with the maximum distance (classic elbow heuristic).
    """
    cov = np.asarray(coverages, dtype=float)
    acc = np.asarray(accuracies, dtype=float)

    start = np.array([cov[0], acc[0]])
    end = np.array([cov[-1], acc[-1]])

    numerator = np.abs(
        (end[1] - start[1]) * cov
        - (end[0] - start[0]) * acc
        + end[0] * start[1]
        - end[1] * start[0]
    )
    denominator = np.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2) + 1e-12

    distances = numerator / denominator
    return int(np.argmax(distances))


def plot_acc_coverage_curve(coverages, accuracies):
    """
    Plots the accuracy–coverage tradeoff for an IDK-threshold sweep and highlights
    the elbow point.

    Inputs:
      - coverages: list/array of acceptance rates (fraction of samples not rejected)
      - accuracies: list/array of accuracy on accepted samples for each threshold

    Output:
      - A plot showing the curve (points correspond to thresholds)
      - Returns (elbow_accuracy, elbow_coverage) for reporting or threshold selection

    Visualization choices:
      - Markers show that the curve consists of discrete threshold settings.
      - Vertical/horizontal guide lines make the elbow readable.
      - A small annotation summarizes elbow coordinates.
    """
    elbow_idx = find_stability_limit(coverages, accuracies)
    elbow_acc = float(accuracies[elbow_idx])
    elbow_cov = float(coverages[elbow_idx])

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 110

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(
        coverages,
        accuracies,
        linewidth=2.6,
        marker="o",
        markersize=4.5,
        label="Threshold sweep",
    )

    ax.axvline(elbow_cov, linestyle="--", linewidth=1.6, alpha=0.75)
    ax.axhline(elbow_acc, linestyle="--", linewidth=1.6, alpha=0.75)
    ax.scatter(elbow_cov, elbow_acc, s=70, zorder=5, label="Elbow")

    ax.annotate(
        f"Elbow\ncov={elbow_cov:.3f}\nacc={elbow_acc:.3f}",
        xy=(elbow_cov, elbow_acc),
        xytext=(10, -10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
    )

    ax.set_title("Accuracy–Coverage curve (IDK option)", fontsize=13, pad=10)
    ax.set_xlabel("Coverage (fraction accepted)", fontsize=12, labelpad=8)
    ax.set_ylabel("Accuracy on accepted samples", fontsize=12, labelpad=8)

    sns.despine(ax=ax)
    ax.legend(frameon=True, fontsize=10)
    ax.tick_params(direction="out", length=5, width=1.4, labelsize=11)

    plt.tight_layout()
    plt.show()

    return elbow_acc, elbow_cov



