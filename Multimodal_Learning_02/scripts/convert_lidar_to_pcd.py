#!/usr/bin/env python3
"""
Convert .npy LiDAR depth files to .pcd point cloud format.

This script converts LiDAR depth measurements stored as .npy files into
PCD (Point Cloud Data) format for visualization in FiftyOne and other tools.
It uses the same coordinate transformation logic as utils.py.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_depth_to_xyz(lidar_depth, azimuth, zenith, max_range=50.0):
    """
    Convert LiDAR depth data to XYZ coordinates.

    Uses the same transformation as utils.py:103-109:
    - x = depth * sin(-azimuth) * cos(-zenith)
    - y = depth * cos(-azimuth) * cos(-zenith)
    - z = depth * sin(-zenith)

    Args:
        lidar_depth: 64x64 array of depth measurements
        azimuth: 1D array of azimuth angles (corresponds to rows)
        zenith: 1D array of zenith angles (corresponds to columns)
        max_range: Maximum LiDAR range (points at this distance are filtered)

    Returns:
        xyz: Nx3 numpy array of point cloud coordinates
    """
    # Broadcast to create 2D angle arrays matching depth shape
    azimuth_2d = azimuth[:, np.newaxis]  # Shape: (64, 1)
    zenith_2d = zenith[np.newaxis, :]    # Shape: (1, 64)

    # Calculate XYZ coordinates (same as utils.py)
    x = lidar_depth * np.sin(-azimuth_2d) * np.cos(-zenith_2d)
    y = lidar_depth * np.cos(-azimuth_2d) * np.cos(-zenith_2d)
    z = lidar_depth * np.sin(-zenith_2d)

    # Filter out max-range points (no-return beams)
    valid_mask = lidar_depth < max_range

    # Stack and flatten to Nx3 array
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]

    xyz = np.stack([x_valid, y_valid, z_valid], axis=1)

    return xyz


def convert_npy_to_pcd(npy_path, pcd_path, azimuth, zenith, add_color=True):
    """
    Convert a single .npy file to .pcd format.

    Args:
        npy_path: Path to input .npy file
        pcd_path: Path to output .pcd file
        azimuth: Azimuth angles array
        zenith: Zenith angles array
        add_color: If True, color points by height (z-coordinate)
    """
    # Load depth data
    lidar_depth = np.load(npy_path)

    # Convert to XYZ
    xyz = convert_depth_to_xyz(lidar_depth, azimuth, zenith)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Optional: Add color based on height (z-coordinate) for better visualization
    if add_color and len(xyz) > 0:
        z_values = xyz[:, 2]
        z_min, z_max = z_values.min(), z_values.max()

        # Normalize z to [0, 1]
        if z_max > z_min:
            z_norm = (z_values - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z_values)

        # Create rainbow colormap (similar to visualization in notebooks)
        # Blue (low) -> Green (mid) -> Red (high)
        colors = np.zeros((len(xyz), 3))
        colors[:, 0] = z_norm  # Red channel
        colors[:, 1] = 1 - np.abs(z_norm - 0.5) * 2  # Green channel (peaks at 0.5)
        colors[:, 2] = 1 - z_norm  # Blue channel

        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save as PCD
    o3d.io.write_point_cloud(str(pcd_path), pcd)

    return len(xyz)


def main():
    parser = argparse.ArgumentParser(description='Convert .npy LiDAR files to .pcd format')
    parser.add_argument('dataset_path', type=str,
                        help='Path to dataset directory (e.g., data/replicator_data_cubes/)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable height-based coloring')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for PCD files (default: <dataset_path>/pcd/)')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    # Verify dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return 1

    # Load angle data (same for all files in dataset)
    azimuth_path = dataset_path / 'azimuth.npy'
    zenith_path = dataset_path / 'zenith.npy'
    lidar_dir = dataset_path / 'lidar'

    if not azimuth_path.exists() or not zenith_path.exists():
        print(f"Error: Could not find azimuth.npy and zenith.npy in {dataset_path}")
        return 1

    if not lidar_dir.exists():
        print(f"Error: Could not find lidar/ directory in {dataset_path}")
        return 1

    print(f"Loading angle data from {dataset_path}...")
    azimuth = np.load(azimuth_path)
    zenith = np.load(zenith_path)
    print(f"  Azimuth shape: {azimuth.shape}")
    print(f"  Zenith shape: {zenith.shape}")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dataset_path / 'pcd'

    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get all .npy files in lidar directory
    npy_files = sorted(lidar_dir.glob('*.npy'))
    print(f"Found {len(npy_files)} .npy files to convert")

    if len(npy_files) == 0:
        print(f"Warning: No .npy files found in {lidar_dir}")
        return 0

    # Convert all files with progress bar
    total_points = 0
    add_color = not args.no_color

    for npy_path in tqdm(npy_files, desc="Converting", unit="file"):
        pcd_path = output_dir / f"{npy_path.stem}.pcd"
        num_points = convert_npy_to_pcd(npy_path, pcd_path, azimuth, zenith, add_color)
        total_points += num_points

    print(f"\nConversion complete!")
    print(f"  Converted {len(npy_files)} files")
    print(f"  Total points: {total_points:,}")
    print(f"  Average points per file: {total_points / len(npy_files):.1f}")
    print(f"  Output location: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
