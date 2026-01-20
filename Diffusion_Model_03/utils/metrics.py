import torch
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import config, other_utils


def calculate_clip_score(clip_preprocess_val, tokenizer, clip_scorer, image_path, text_prompt, device=None):
    """
    Computes a CLIP similarity score between an image and a text prompt.

    The image and text are embedded using a pretrained OpenCLIP model, L2-normalized,
    and compared via cosine similarity (dot product of normalized embeddings).

    Args:
        image_path (str | Path): Path to the image file on disk.
        text_prompt (str): Text prompt to compare against.

    Returns:
        float: Cosine similarity score (higher means stronger semantic alignment).
    """
    # Preprocess and move to the same device as the CLIP model
    image = clip_preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer([text_prompt]).to(device)

    with torch.no_grad():
        image_features = clip_scorer.encode_image(image)
        text_features = clip_scorer.encode_text(text)

        # Normalize to turn dot product into cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (image_features @ text_features.T).item()

    return score


def calculate_fid(real_embeddings, gen_embeddings):
    """
    Compute Fréchet Inception Distance (FID) from two sets of feature vectors by
    comparing their Gaussian means and covariances (lower is better).
    """

    # Calculate mean and covariance
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)

    # Sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2)

    # Product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical error handling
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Final FID calculation
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def init_inception_for_fid(device, img_size=299):
    """
    Initialize pretrained InceptionV3 for FID: return 2048-D features (fc=Identity)
    and the ImageNet-normalized 299x299 preprocessing transform.
    """
    inception = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        transform_input=False,
    ).to(device)

    inception.fc = torch.nn.Identity()
    inception.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    inception_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return inception, inception_transform

inception, inception_transform = init_inception_for_fid(config.DEVICE, config.INCEPTION_IMG_SIZE)


def get_inception_features_from_raw(dataset_path, batch_size, model, device=None, num_workers=0):
    """
    Extracts 2048-dimensional InceptionV3 feature embeddings for a dataset of images.

    Args:
        raw_dataset): dataset of the original images
        model (nn.Module): Pretrained InceptionV3 feature extractor (fc = Identity).

    Returns:
        np.ndarray: Array of shape (N, 2048) containing feature embeddings for all images.
    """
    raw_dataset = other_utils.MyDataset(dataset_path, inception_transform, config.CLASSES)
    raw_dataloader  = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features = []                         # Stores feature batches
    with torch.no_grad():
      for img, _ in tqdm(raw_dataloader):
          img = img.to(device)
          f = model(img)                # Runs Inception forward pass
          features.append(f.cpu().numpy())  # Transform to numpy for later mathematical operations
    return np.concatenate(features, axis=0) # Concatenate batches to one array


def get_inception_features_from_files(saved_samples, batch_size, model, transform, device=None, num_workers=0):
    """
    Loads images from disk and extracts InceptionV3 feature embeddings.

    Args:
        saved_samples: List of tuples containing image filepaths.
        model: Pretrained feature extractor.
        transform: Image preprocessing pipeline (resize/normalize).

    Returns:
        np.ndarray: Feature matrix of shape (N, 2048).
    """
    dataset = other_utils.GeneratedListDataset(saved_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features = []

    with torch.no_grad():
        for img_batch in tqdm(loader, desc="Extracting Generated Features"):
            img_batch = img_batch.to(device)

            # The transform handles Resize, Scale, and Normalize
            f = model(img_batch)
            features.append(f.cpu().numpy())

    return np.concatenate(features, axis=0)



def calculate_fid_score(samples, dataset_path):
    """
    Compute FID between real images in `dataset_path` and generated images in `samples`
    by extracting InceptionV3 features for both sets and comparing their feature
    statistics (lower is better).
    """

    # Extract features from real images on disk
    real_embeddings = get_inception_features_from_raw(dataset_path, config.BATCH_SIZE, inception, device=config.DEVICE, num_workers=config.NUM_WORKERS)

    print("Real Embeddings:", real_embeddings.shape)

    print(f"Total generated: {len(samples)}")

    # Extract features from generated images on disk
    gen_embeddings = get_inception_features_from_files(
        saved_samples=samples,
        batch_size=config.BATCH_SIZE,
        model=inception,
        transform=inception_transform,
        device=config.DEVICE,
        num_workers=config.NUM_WORKERS
    )

    print("Generated Embeddings:", gen_embeddings.shape)

    # Compute FID Score: checks if the images look "real" compared to the original dataset
    fid_score = calculate_fid(real_embeddings, gen_embeddings)
    return fid_score


def compute_idk_tradeoff(conf_correct, conf_incorrect, num_thresholds=100):
    """
    Computes accuracy–coverage tradeoff for an "IDK" reject option using only
    confidence scores (no logits needed).

    Interpretation
    --------------
    For a given threshold t:
      - Accept sample if confidence >= t, otherwise reject as "IDK".
      - Accuracy(covered) = accepted_correct / (accepted_correct + accepted_incorrect)
      - Coverage = (accepted_correct + accepted_incorrect) / total_samples
    """
    conf_correct = np.asarray(conf_correct, dtype=float)
    conf_incorrect = np.asarray(conf_incorrect, dtype=float)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    accuracies_covered = []
    coverages = []

    total_samples = len(conf_correct) + len(conf_incorrect)

    for t in thresholds:
        accepted_correct = int((conf_correct >= t).sum())
        accepted_incorrect = int((conf_incorrect >= t).sum())
        accepted_total = accepted_correct + accepted_incorrect

        if accepted_total > 0:
            acc_cov = accepted_correct / accepted_total
            cov = accepted_total / total_samples
        else:
            # If nothing is accepted, accuracy-on-covered is undefined; use 1.0 by convention
            acc_cov = 1.0
            cov = 0.0

        accuracies_covered.append(acc_cov)
        coverages.append(cov)

    return thresholds, np.array(accuracies_covered), np.array(coverages)


def select_threshold_by_target_accuracy(thresholds, accuracies_covered, target_accuracy=0.9985):
    """
    Selects the smallest threshold that achieves at least `target_accuracy`
    on accepted samples.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    accuracies_covered = np.asarray(accuracies_covered, dtype=float)

    for t, acc in zip(thresholds, accuracies_covered):
        if acc >= target_accuracy:
            return float(t)

    # If no threshold reaches the target, return the strictest one
    return float(thresholds[-1])
