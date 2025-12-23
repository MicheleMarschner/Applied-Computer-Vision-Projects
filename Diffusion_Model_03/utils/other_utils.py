from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datetime import datetime
import os
import random
import numpy as np
import cv2
import albumentations as A

from utils import config


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(image[0].detach().cpu()))
    

def to_image(tensor, to_pil=True):
    tensor = (tensor + 1) / 2
    ones = torch.ones_like(tensor)
    tensor = torch.min(torch.stack([tensor, ones]), 0)[0]
    zeros = torch.zeros_like(tensor)
    tensor = torch.max(torch.stack([tensor, zeros]), 0)[0]
    if not to_pil:
        return tensor
    return transforms.functional.to_pil_image(tensor)


def plot_generated_images(noise, result):
    """
    Plots the input noise and the generated image side by side.
    """
    plt.figure(figsize=(8,8))
    nrows = 1
    ncols = 2
    samples = {
        "Noise" : noise,
        "Generated Image" : result
    }
    for i, (title, img) in enumerate(samples.items()):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_title(title)
        show_tensor_image(img)
    plt.show()


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