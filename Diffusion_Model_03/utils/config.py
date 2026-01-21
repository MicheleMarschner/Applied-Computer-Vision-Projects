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

# Constants for Assignment 3 and further experiments
BATCH_SIZE = 32
CLIP_FEATURES = 512
TIMESTEPS = 400       # Number of timesteps
IMG_SIZE = 32
INCEPTION_IMG_SIZE = 299
IMG_CH = 3
INPUT_SIZE = (IMG_CH, IMG_SIZE, IMG_SIZE)
FIFTYONE_DATASET_NAME = "generated_flowers_experiment"
FIFTYONE_DATASET_EXPERIMENTS_NAME = "generated_optimized_flowers_experiment"
W_TESTS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
W_OPT = [1.0, 2.0, 3.0]
CLASSES = ["daisy", "roses", "sunflowers"]
HF_BASE_REPO_ID = "mmarschn/generated_flowers_experiment"
HF_EXPERIMENT_REPO_ID = "mmarschn/generated_optimized_flowers_experiment"


# Constants for Bonus
N_SAMPLES = 200  
BATCH_SIZE_CLASSIFIER = 64
FIFTYONE_BONUS_DATASET_NAME = "mnist_idk_experiment"
CONFIDENCE_THRESHOLD = 0.99
GUIDANCE_LIST = [0.0, 1.0, 2.0]
BATCH_ADD = 500

# --------------------
# Paths
# --------------------
TMP_ROOT = Path("/content")         
DRIVE_ROOT = Path("/content/drive/MyDrive/Applied-Computer-Vision-Projects/Diffusion_Model_03")              # !! Change this path if the project is located elsewhere (repeat in notebooks)

SAVE_DIR = TMP_ROOT / "generated_flowers"
os.makedirs(SAVE_DIR, exist_ok=True) # Ensure folder exists

EXPORT_DIR = TMP_ROOT / "generated_flowers_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

TEMP_IMG_DIR = TMP_ROOT / "mnist_temp"
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

EXPORT_MNIST_DIR = TMP_ROOT / "mnist_idk_export"
os.makedirs(EXPORT_MNIST_DIR, exist_ok=True)

DATA_DIR = TMP_ROOT / "data/cropped_flowers"
os.makedirs(DATA_DIR, exist_ok=True)

CHECKPOINTS_DIR = DRIVE_ROOT / "checkpoints" 
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_MODEL_PATH = CHECKPOINTS_DIR / "best_mnist_classifier.pth"
CLASSIFIER_IDK_MODEL_PATH = CHECKPOINTS_DIR / "idk_classifier_lenet_2.pth"
UNET_MODEL_PATH = CHECKPOINTS_DIR / "uNet.pth"
UNET_MNIST_MODEL_PATH = CHECKPOINTS_DIR / "ddpm_unet_conditioned.pth"
UNET_MNIST_UNCOND_MODEL_PATH = CHECKPOINTS_DIR / "ddpm_unet_unconditioned.pth"

# --------------------
# General
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count()  # Number of CPU cores