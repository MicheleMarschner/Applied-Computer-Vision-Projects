
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
import torch.nn.functional as Func
from fiftyone import ViewField as F
from collections import Counter
import pandas as pd
from collections import defaultdict

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

# Mnist Model Architecture for IDK class bonus task  
class ModernLeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.5)

        # Will store the latest fc1 activations (batch x 84)
        self.last_embedding = None

    def forward(self, x):
        x = self.pool(Func.relu(self.conv1(x)))
        x = self.pool(Func.relu(self.conv2(x)))
        x = Func.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        emb = self.fc1(x)          # pre-activation embedding
        self.last_embedding = emb  # save for inspection/extraction

        x = Func.relu(emb)
        x = self.dropout(x)
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


def rescale_to_unit_interval_and_stretch(x_gen):
    """
    x_gen: diffusion output in [-1, 1], shape typically [1, 1, 28, 28]
    returns: [0, 1] tensor, same shape, with contrast stretching
    """
    img_01 = (x_gen.clamp(-1, 1) + 1.0) / 2.0

    mn = img_01.amin(dim=(-2, -1), keepdim=True)
    mx = img_01.amax(dim=(-2, -1), keepdim=True)
    den = (mx - mn).clamp_min(1e-6)  # avoid divide-by-zero
    img_01 = (img_01 - mn) / den

    return img_01.clamp(0, 1)


def normalize_for_lenet(img):
    return (img - 0.1307) / 0.3081       # MNIST Normalize((0.1307,), (0.3081,))


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
