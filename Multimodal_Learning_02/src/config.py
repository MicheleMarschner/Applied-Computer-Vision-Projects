from pathlib import Path
import torch
import os

# --------------------
# Reproducibility
# --------------------
SEED = 51

# --------------------
# General experiment
# --------------------
CLASSES = ["cubes", "spheres"]
NUM_CLASSES = len(CLASSES)
LABEL_MAP = {"cubes": 0, "spheres": 1}

IMG_SIZE = 64
BATCH_SIZE = 32
VALID_BATCHES = 10
N = 12500

# --------------------
# Paths
# --------------------
TMP_ROOT = Path("/content")         
DRIVE_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/Applied Computer Vision/Applied-Computer-Vision-Projects/Multimodal_Learning_02/")

RAW_DATA = TMP_ROOT / "data"                        
TMP_TRANSFORMED_DATA_PATH = TMP_ROOT / "data_transformed"  
DRIVE_TRANSFORMED_DATA_PATH = DRIVE_ROOT / "data_transformed"

CHECKPOINTS = DRIVE_ROOT / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# --------------------
# General
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()  # Number of CPU cores