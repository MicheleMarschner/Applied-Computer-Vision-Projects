from pathlib import Path

from PIL import Image
from tqdm import tqdm
import random

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np

from src.utility import create_random_subset, create_deterministic_training_dataloader, get_torch_xyza


class AssessmentDataset(Dataset):
    """
    Fully preprocessed dataset:

    - Loads raw RGB + depth
    - Loads azimuth/zenith per class
    - Converts depth → XYZA ONCE
    - Applies transform_rgb ONCE
    - Applies transform_lidar ONCE
    - Caches everything in memory

    Then __getitem__ is extremely fast.
    """
    def __init__(
        self,
        root_dir,
        start_idx=0,
        end_idx=None,
        transform_rgb=None,
        transform_lidar=None,
        shuffle=True,
        seed=51
    ):
        self.root_dir = Path(root_dir)

        # These MUST be deterministic transforms (Resize, ToTensor, Normalize...)
        self.transform_rgb = transform_rgb
        self.transform_lidar = transform_lidar

        self.classes = ["cubes", "spheres"]
        self.label_map = {"cubes": 0, "spheres": 1}

        samples = []
        self.az = {}
        self.ze = {}

        print(f"Scanning RAW dataset in {root_dir}...")

        # -------- 1. Scan the dataset & load azimuth/zenith --------
        for cls in self.classes:
            cls_dir   = self.root_dir / cls
            rgb_dir   = cls_dir / "rgb"
            lidar_dir = cls_dir / "lidar"

            # load az/zen
            az_path = cls_dir / "azimuth.npy"
            ze_path = cls_dir / "zenith.npy"
            if not az_path.exists() or not ze_path.exists():
                raise FileNotFoundError(f"Missing azimuth/zenith in {cls_dir}")

            self.az[cls] = torch.from_numpy(np.load(az_path)).float()
            self.ze[cls] = torch.from_numpy(np.load(ze_path)).float()

            # match stems
            rgb_files   = sorted(rgb_dir.glob("*.png"))
            lidar_files = sorted(lidar_dir.glob("*.npy"))

            rgb_stems   = {f.stem for f in rgb_files}
            lidar_stems = {f.stem for f in lidar_files}
            matching    = sorted(rgb_stems & lidar_stems)

            print(f"{cls}: {len(matching)} paired samples")

            for stem in matching:
                samples.append(
                    {
                        "class": cls,
                        "rgb_path": rgb_dir / f"{stem}.png",
                        "depth_path": lidar_dir / f"{stem}.npy",
                        "label": self.label_map[cls],
                    }
                )

        # -------- Optional shuffle --------
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if end_idx is None:
            end_idx = len(samples)
        self.samples = samples[start_idx:end_idx]

        # Prepare storage
        self.rgb_tensors = []
        self.lidar_tensors = []
        self.labels = []

        # -------- 2. PRECOMPUTE EVERYTHING --------
        print("Precomputing RGB + XYZA tensors into RAM...")

        for item in tqdm(self.samples, desc="Preprocessing"):
            cls = item["class"]
            az  = self.az[cls]
            ze  = self.ze[cls]

            # --- RGB ---
            rgb_img = Image.open(item["rgb_path"])
            if self.transform_rgb:
                rgb_tensor = self.transform_rgb(rgb_img)  # applied once
            else:
                rgb_tensor = transforms.ToTensor()(rgb_img)

            # --- LiDAR XYZA ---
            depth_np = np.load(item["depth_path"])
            depth_t  = torch.from_numpy(depth_np).float()
            xyza = get_torch_xyza(depth_t, az, ze)        # (4,H,W)

            if self.transform_lidar:
                xyza = self.transform_lidar(xyza)        # applied once

            self.rgb_tensors.append(rgb_tensor)
            self.lidar_tensors.append(xyza)
            self.labels.append(item["label"])

        print(f"Dataset ready: {len(self.samples)} samples preprocessed.")

    # -------- Fast loaders --------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            self.rgb_tensors[idx],
            self.lidar_tensors[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


## Final: überdenken woher datenset kommen soll
class AssessmentXYZADataset(Dataset):
    """
    Dataset for the CILP XYZ + RGB assessment data.

    It expects the following folder structure:

        root/
          cubes/
            rgb/*.png
            lidar_xyza/*.npy
          spheres/
            rgb/*.png
            lidar_xyza/*.npy

    Each sample consists of an RGB image, a LiDAR XYZA tensor and a class label.
    """
    def __init__(self, root_dir, start_idx=0, end_idx=None,
                 transform_rgb=None, transform_lidar=None, shuffle=True, seed=51):
        """
        Args:
            root_dir (str or Path): Root directory of the dataset.
            start_idx (int): Start index (inclusive) for slicing the dataset.
            end_idx (int or None): End index (exclusive); if None use all.
            transform_rgb (callable or None): Transform applied to RGB images.
            transform_lidar (callable or None): Transform applied to LiDAR tensors.
            shuffle (bool): If True, shuffle the full list of samples once.
        """
        self.root_dir = Path(root_dir)
        self.transform_rgb = transform_rgb
        self.transform_lidar = transform_lidar

        self.classes = ["cubes", "spheres"]
        self.label_map = {"cubes": 0, "spheres": 1}

        samples = []

        print(f"Scanning dataset in {root_dir}...")
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            rgb_dir = cls_dir / "rgb"
            lidar_dir = cls_dir / "lidar_xyza"

            rgb_files = sorted(rgb_dir.glob("*.png"))
            print(f"{cls}: {len(rgb_files)} RGB files found. Matching XYZA...")

            for rgb_path in tqdm(rgb_files, desc=f"{cls} matching", leave=False):
                stem = rgb_path.stem
                lidar_path = lidar_dir / f"{stem}.npy"
                if lidar_path.exists():
                    samples.append({
                        "rgb": rgb_path,
                        "lidar_xyza": lidar_path,
                        "label": self.label_map[cls],
                    })

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if end_idx is None:
            end_idx = len(samples)
        self.samples = samples[start_idx:end_idx]

        # Preload LiDAR tensors into memory since they are small and fast to cache
        print(f"Preloading LiDAR XYZA tensors into RAM...")
        self.lidar_tensors = []
        for item in tqdm(self.samples, desc="Loading XYZA", leave=False):
            lidar_np = np.load(item["lidar_xyza"])        # (4, H, W)
            lidar_t  = torch.from_numpy(lidar_np).float() # CPU tensor
            self.lidar_tensors.append(lidar_t)

        print(
            f"Dataset ready: {len(self.samples)} samples loaded.\n"
            f"Slice [{start_idx}:{end_idx}]"
        )

    def __len__(self):
        """Return the number of samples in this dataset slice."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a single (rgb, lidar, label) triplet.

        Returns:
            tuple: (rgb_tensor, lidar_tensor, label_tensor)
        """
        item  = self.samples[idx]
        lidar = self.lidar_tensors[idx]

        # RGB image is loaded on the fly
        rgb = Image.open(item["rgb"])
        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)

        if self.transform_lidar:
            lidar = self.transform_lidar(lidar)

        label = torch.tensor(item["label"], dtype=torch.long)
        return rgb, lidar, label


class AssessmentCILPDataset(Dataset):
    def __init__(self, root_dir, transform_rgb=None):
        self.root_dir = Path(root_dir)
        self.rgb = []
        self.lidar = []
        self.class_idx = []

        for class_name in ["cubes", "spheres"]:
            rgb_dir   = self.root_dir / class_name / "rgb"
            lidar_dir = self.root_dir / class_name / "lidar"

            rgb_files = sorted(rgb_dir.glob("*.png"))

            for rgb_path in rgb_files:
                file_number = rgb_path.stem
                lidar_npy = lidar_dir / f"{file_number}.npy"
                if not lidar_npy.exists():
                    continue

                # RGB
                rgb_img = Image.open(rgb_path)
                rgb_tensor = transform_rgb(rgb_img)

                # LiDAR depth (1 channel)
                lidar_arr = np.load(lidar_npy)           # (H, W)
                lidar_tensor = torch.from_numpy(lidar_arr).float().unsqueeze(0)

                self.rgb.append(rgb_tensor)
                self.lidar.append(lidar_tensor)

                if class_name.startswith("cube"):
                    self.class_idx.append(torch.tensor([0.], dtype=torch.float32))
                else:
                    self.class_idx.append(torch.tensor([1.], dtype=torch.float32))

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        return (
            self.rgb[idx],
            self.lidar[idx],
            self.class_idx[idx],
        )
    

def match_lidar_rgb(classes, rgb_root: Path, pcd_root: Path):
    """
    Match RGB (.png) and LiDAR (.pcd) files by filename stem per class.

    Returns:
        dict[class_name, list[dict]] with keys: "stem", "rgb", "lidar".
    """
    pairs = {}

    for class_name in classes:
        rgb_dir = rgb_root / class_name / "rgb"
        pcd_dir = pcd_root / class_name / "pcd"

        # Check if directories exist
        assert rgb_dir.exists(), f"RGB directory not found: {rgb_dir}"
        assert pcd_dir.exists(), f"PCD directory not found: {pcd_dir}"

        # Collect files
        rgb_files = sorted(rgb_dir.glob("*.png"))
        pcd_files = sorted(pcd_dir.glob("*.pcd"))

        print(f"[{class_name}] RGB: {len(rgb_files)} | PCD: {len(pcd_files)}")

        # Match by stem
        rgb_stems = {f.stem for f in rgb_files}
        pcd_stems = {f.stem for f in pcd_files}
        matching = rgb_stems & pcd_stems

        # Store matched pairs
        pairs[class_name] = [
            {
                "stem": stem,
                "rgb": rgb_dir / f"{stem}.png",
                "lidar": pcd_dir / f"{stem}.pcd",
            }
            for stem in sorted(matching)
        ]

        if len(matching) == 0:
            print(f"⚠️  No matching pairs found for class '{class_name}'")
        else:
            print(f"✅ {class_name}: {len(matching)} matched pairs")

    return pairs


def compute_dataset_mean_std(root_dir, img_size=64):
    """
    Estimate the per-channel mean and std for the RGB+LiDAR data.

    Args:
        root_dir (str or Path): Root directory passed to AssessmentXYZADataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            mean and std tensors with shape (C,).
    """

    stats_transforms = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToImage(),
      transforms.ToDtype(torch.float32, scale=True),  # [0,1], 4 channels
    ])

    stats_dataset = AssessmentXYZADataset(
        root_dir=root_dir,
        start_idx=0,
        end_idx=None,          # or e.g. 1000 to subsample
        transform_rgb=stats_transforms,
    )

    subset_size = min(2000, len(stats_dataset)*0.3)
    subset_for_stats = create_random_subset(size=subset_size, dataset=stats_dataset)

    loader = DataLoader(subset_for_stats, batch_size=64, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    total = 0

    for images, _, _ in tqdm(loader, desc="Computing mean/std"):
        images = images.float()       # B, C, H, W
        batch_size = images.size(0)

        # compute mean over batch (channels only!)
        mean += images.mean(dim=[0, 2, 3]) * batch_size

        # compute std over batch
        std += images.std(dim=[0, 2, 3]) * batch_size

        total += batch_size

    mean /= total
    std /= total

    return mean, std


def compute_dataset_mean_std_neu(root_dir, img_size=64, seed=51):
    """
    Estimate the per-channel mean and std for the RGB+LiDAR data.

    Args:
        root_dir (str or Path): Root directory passed to AssessmentXYZADataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            mean and std tensors with shape (C,).
    """
    stats_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),  # [0,1], 4 channels
    ])

    stats_dataset = AssessmentXYZADataset(
        root_dir=root_dir,
        transform_rgb=stats_transforms,
        transform_lidar=None,
        shuffle=False,
        seed=seed
    )

    subset_size = min(2000, len(stats_dataset)*0.3)
    subset_for_stats = create_random_subset(size=subset_size, dataset=stats_dataset)

    loader = DataLoader(subset_for_stats, batch_size=64, shuffle=False)

    # Accumulate running sum and sum of squares to compute mean/std
    channel_sum = torch.zeros(4)
    channel_sq_sum = torch.zeros(4)
    num_pixels = 0

    for rgb, _, _ in tqdm(loader, desc="Computing mean/std"):
        # rgb shape: (B, C, H, W)
        b, c, h, w = rgb.shape
        num_pixels += b * h * w
        channel_sum += rgb.sum(dim=[0, 2, 3])
        channel_sq_sum += (rgb ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean ** 2)
    return mean, std


def get_dataloaders(root_dir, valid_batches, num_workers=2, test_frac=0.15, batch_size=64, img_transforms=None, seed=51):
    """
    Create train / val / test datasets + dataloaders.

    - Validation set has exactly VALID_BATCHES * BATCH_SIZE samples.
    - Validation DataLoader with drop_last=True → exactly VALID_BATCHES batches.
    - Train/test share the remaining samples via a random split.
    """

    # 1) Base dataset: no internal shuffle, full range
    base_dataset = AssessmentXYZADataset(
        root_dir,
        start_idx=0,
        end_idx=None,
        transform_rgb=img_transforms,    # adapt if you also pass lidar transforms
        transform_lidar=None,
        shuffle=False,                   # important: shuffling done via indices
        seed=51
    )

    N = len(base_dataset)
    val_size = valid_batches * batch_size

    if N <= val_size:
        raise ValueError(
            f"Dataset too small: N={N}, but need at least "
            f"{val_size + 1} samples to have train+test as well."
        )

    # remaining samples after reserving validation
    remaining = N - val_size
    test_size = int(remaining * test_frac)
    train_size = remaining - test_size

    print(f"Total samples: {N}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 2) One random permutation of all indices (deterministic)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()

    train_idx = perm[:train_size]
    val_idx   = perm[train_size:train_size + val_size]
    test_idx  = perm[train_size + val_size:]

    # 3) Subsets
    train_ds = Subset(base_dataset, train_idx)
    val_ds   = Subset(base_dataset, val_idx)
    test_ds  = Subset(base_dataset, test_idx)

    # 4) DataLoaders
    train_loader = create_deterministic_training_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,   # for evaluation, we usually don't drop examples
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_ds, train_loader, val_ds, val_loader, test_ds, test_loader


def get_cilp_dataloaders(
    root_dir,
    valid_batches,
    num_workers=2,
    test_frac=0.10,
    batch_size=64,
    img_transforms=None,
    seed=51,
):
    """
    Train/val/test splits + dataloaders for the CILP depth-only dataset.

    Uses AssessmentCILPDataset → lidar/*.npy as a single channel.
    """
    base_dataset = AssessmentCILPDataset(
        root_dir=root_dir,
        transform_rgb=img_transforms,
    )

    N = len(base_dataset)
    val_size = valid_batches * batch_size

    if N <= val_size:
        raise ValueError(f"CILP dataset too small: N={N}, need > {val_size}.")

    remaining = N - val_size
    test_size = int(remaining * test_frac)
    train_size = remaining - test_size

    print(f"[CILP] Total samples: {N}")
    print(f"[CILP] Train: {train_size}, Val: {val_size}, Test: {test_size}")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()

    train_idx = perm[:train_size]
    val_idx   = perm[train_size:train_size + val_size]
    test_idx  = perm[train_size + val_size:]

    train_ds = Subset(base_dataset, train_idx)
    val_ds   = Subset(base_dataset, val_idx)
    test_ds  = Subset(base_dataset, test_idx)

    train_loader = create_deterministic_training_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_ds, train_loader, val_ds, val_loader, test_ds, test_loader
