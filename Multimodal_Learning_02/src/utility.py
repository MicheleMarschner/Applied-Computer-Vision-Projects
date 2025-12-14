import os
import random
import torch
from torch.utils.data import Subset
import numpy as np
import cv2
import albumentations as A
from pathlib import Path
import shutil
import subprocess


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


# Additional reproducibility considerations:
def create_deterministic_training_dataloader(dataset, batch_size, shuffle=True, seed=51, **kwargs):
    """
    Create a DataLoader with deterministic shuffling behaviour.

    Args:
        dataset (Dataset): PyTorch Dataset instance.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle data.
        **kwargs: Additional DataLoader keyword arguments.

    Returns:
        DataLoader: Training DataLoader with reproducible sampling.
    """
    # Use a generator with fixed seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        **kwargs
    )


def format_time(seconds):
    """
    Convert a duration in seconds to a human-readable 'MMm SSs' string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted duration, e.g. "02m 15s".
    """
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}m {s:02d}s"


def get_torch_xyza(lidar_depth, azimuth, zenith):
    """
    Convert LiDAR depth + angular coordinates into an XYZA tensor.

    Args:
        lidar_depth (torch.Tensor): Radial distances of shape (H, W).
        azimuth (torch.Tensor): Azimuth angles in radians, shape (H,).
        zenith (torch.Tensor): Zenith angles in radians, shape (W,).

    Returns:
        torch.Tensor: Stacked tensor of shape (4, H, W) containing
            X, Y, Z coordinates and a validity mask A.
    """
    # Broadcast azimuth (per row) and zenith (per column) to full image grid
    # and convert polar coordinates into Cartesian coordinates.
    x = lidar_depth * torch.sin(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    y = lidar_depth * torch.cos(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    z = lidar_depth * torch.sin(-zenith[None, :])

    # A is a binary mask: 1 for valid points, 0 for "no return" / far-away
    a = torch.where(lidar_depth < 50.0,
                    torch.ones_like(lidar_depth),
                    torch.zeros_like(lidar_depth))

    xyza = torch.stack([x, y, z, a], dim=0)
    return xyza


def format_positions(positions):
    """
    Format a sequence of numerical positions as nicely aligned strings.

    Args:
        positions (Iterable[float]): Sequence of scalar values.

    Returns:
        list[str]: List of strings with 4 decimal places.
    """
    return ['{0: .4f}'.format(x) for x in positions]


def create_random_subset(size, dataset):
    """
    Create a random subset of a given dataset.

    Args:
        size (int): Desired number of samples in the subset.
        dataset (torch.utils.data.Dataset): Dataset to sample from.

    Returns:
        torch.utils.data.Subset: Random subset of the given dataset.
    """
    # Sample unique indices uniformly at random from the dataset
    indices = np.random.choice(size, size=size, replace=False)
    return Subset(dataset, indices)



    """
    Check whether cached LiDAR point clouds (PCD) are available.
    """
    pcd_dir = cache_dir / "cubes" / "pcd"
    return pcd_dir.exists() and any(pcd_dir.glob("*.pcd"))


def has_cached_pointclouds(cache_dir: Path, classes=["cubes", "spheres"]) -> bool:
    """
    Check whether cached LiDAR point clouds (PCD) are available.
    """
    # Check cache for all requested classes
    for cls in classes:
        pcd_dir = cache_dir / cls / "pcd"
        if not (pcd_dir.exists() and any(pcd_dir.glob("*.pcd"))):
            return False
    return True


def prepare_lidar_pointclouds(
    raw_dataset_dir: Path,
    local_pointcloud_dir: Path,
    cache_dir: Path,
    converter_script: Path,
    classes=["cubes", "spheres"],
):
    """
    Prepare LiDAR point clouds for local use.

    If cached point clouds exist, they are copied from the cache directory.
    Otherwise, point clouds are generated from raw depth data and can later
    be persisted as cache.

    Returns:
        str: "cache" if loaded from cache, "computed" if newly generated.
    """
    if has_cached_pointclouds(cache_dir, classes):
        print("Loading cached LiDAR point clouds")
        shutil.copytree(cache_dir, local_pointcloud_dir, dirs_exist_ok=True)
        return "cache"

    print("Computing LiDAR point clouds from raw data")
    local_pointcloud_dir.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        input_dir = raw_dataset_dir / cls
        output_dir = local_pointcloud_dir / cls / "pcd"
        output_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "python",
                str(converter_script),
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )

    return "computed"



