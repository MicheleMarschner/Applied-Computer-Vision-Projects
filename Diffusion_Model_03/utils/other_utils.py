import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
import os
import random
import numpy as np
import cv2
import albumentations as A


def save_animation(xs, gif_name, interval=300, repeat_delay=5000):
    fig = plt.figure()
    plt.axis('off')
    imgs = []

    for x_t in xs:
        im = plt.imshow(x_t, animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=repeat_delay)
    animate.save(gif_name)


def load_fashionMNIST(data_transform, train=True):
    return torchvision.datasets.FashionMNIST(
        "./data/",
        download=True,
        train=train,
        transform=data_transform,
    )


def load_transformed_fashionMNIST(img_size, batch_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)
    train_set = load_fashionMNIST(data_transform, train=True)
    test_set = load_fashionMNIST(data_transform, train=False)
    data = torch.utils.data.ConcatDataset([train_set, test_set])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return data, dataloader


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