import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datetime import datetime
import os
import random
import numpy as np
import cv2
import albumentations as A
from pathlib import Path

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
        # stored as (filepath, prompt, w_val) -> we only need filepath
        filepath = self.samples[idx][0]

        img = Image.open(filepath).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # We just return the image, FID doesn't need labels
        return img