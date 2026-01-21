import torch
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

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
    # CLIP preprocess + batch dim + device move
    image = clip_preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)
    # Tokenize prompt to CLIP text input format
    text = tokenizer([text_prompt]).to(device)

    with torch.no_grad():
        # Encode into shared CLIP embedding space
        image_features = clip_scorer.encode_image(image)
        text_features = clip_scorer.encode_text(text)

        # L2-normalize so dot product = cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (image_features @ text_features.T).item()

    return score


def calculate_fid(real_embeddings, gen_embeddings):
    """
    Compute Fréchet Inception Distance (FID) from two sets of feature vectors by
    comparing their Gaussian means and covariances (lower is better).
    """
    # Gaussian stats in feature space
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)

    # ||mu1 - mu2||^2 term
    ssdiff = np.sum((mu1 - mu2)**2)

    # sqrtm can produce tiny imaginary parts -> drop them
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def init_inception_for_fid(device, img_size=299):
    """
    Initialize pretrained InceptionV3 for FID: return 2048-D features (fc=Identity)
    and the ImageNet-normalized 299x299 preprocessing transform.
    """
    # Pretrained InceptionV3; replace classifier head so forward() returns 2048-d features
    inception = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        transform_input=False,
    ).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    # ImageNet normalization expected by pretrained weights
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    inception_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return inception, inception_transform

# Global Inception feature extractor + transform used for all FID computations
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

    features = []
    with torch.no_grad():
      for img, _ in tqdm(raw_dataloader):
          img = img.to(device)
          f = model(img)                    # 2048-d features
          features.append(f.cpu().numpy())  # numpy for mean/cov later
    return np.concatenate(features, axis=0)


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
            f = model(img_batch)            # 2048-d features
            features.append(f.cpu().numpy())
    return np.concatenate(features, axis=0)



def calculate_fid_score(samples, dataset_path):
    """
    Compute FID between real images in `dataset_path` and generated images in `samples`
    by extracting InceptionV3 features for both sets and comparing their feature
    statistics (lower is better).
    """
    # Real/reference features
    real_embeddings = get_inception_features_from_raw(
        dataset_path, config.BATCH_SIZE, inception, device=config.DEVICE, num_workers=config.NUM_WORKERS
    )
    print("Real Embeddings:", real_embeddings.shape)
    print(f"Total generated: {len(samples)}")

    # Generated features
    gen_embeddings = get_inception_features_from_files(
        saved_samples=samples,
        batch_size=config.BATCH_SIZE,
        model=inception,
        transform=inception_transform,
        device=config.DEVICE,
        num_workers=config.NUM_WORKERS
    )
    print("Generated Embeddings:", gen_embeddings.shape)

    return calculate_fid(real_embeddings, gen_embeddings)


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

    return float(thresholds[-1])


def top_digit_excluding_idk(prob_vec, idk_index=10):
    # "Forced digit" baseline: exclude IDK class, then take argmax among digits
    p = np.array(prob_vec, dtype=float)
    p[idk_index] = -1.0  # exclude IDK
    digit_idx = int(np.argmax(p))
    digit_conf = float(p[digit_idx])
    return str(digit_idx), digit_conf

def cascaded_policy(raw_pred_label, raw_conf, threshold, idk_label="IDK"):
    """Hard IDK stays; else soft IDK if conf<threshold; else keep digit."""
    # Hard IDK: model explicitly abstained
    if raw_pred_label == idk_label:
        return idk_label, "Hard"
    # Soft IDK: abstain if digit confidence is below threshold
    if float(raw_conf) < float(threshold):
        return idk_label, "Soft"
    return str(raw_pred_label), "None"


def evaluate_threshold(gt, pred, conf, probs, threshold, idk_label="IDK", idk_index=10):
    """Coverage/accuracy + good vs wasted rejects for one threshold."""
    gt = np.asarray(gt, dtype=object)
    pred = np.asarray(pred, dtype=object)
    conf = np.asarray(conf, dtype=float)

    final = np.empty(len(pred), dtype=object)
    rej_type = np.empty(len(pred), dtype=object)

    # Apply cascaded abstention policy to every sample
    for i, (rp, rc) in enumerate(zip(pred, conf)):
        fp, rt = cascaded_policy(rp, rc, threshold, idk_label=idk_label)
        final[i] = fp
        rej_type[i] = rt

    hard_reject = int(np.sum(rej_type == "Hard"))
    soft_reject = int(np.sum(rej_type == "Soft"))

    covered = final != idk_label
    coverage = covered.mean() if len(final) else 0.0
    acc_covered = (final[covered] == gt[covered]).mean() if covered.any() else np.nan

    idk_mask = final == idk_label
    
    # "Good" vs "wasted" rejects: compare abstention against forced-digit baseline
    good_reject = wasted_reject = 0
    for i in np.where(idk_mask)[0]:
      digit_pred, _ = top_digit_excluding_idk(probs[i], idk_index=idk_index)
      if digit_pred == gt[i]:
          wasted_reject += 1
      else:
          good_reject += 1

    n_total = int(len(final))
    n_answered = int(np.sum(covered))
    n_abstained = int(np.sum(idk_mask))

    return {
    "threshold": float(threshold),
    "coverage": float(coverage),
    "idk_rate": float(idk_mask.mean() if len(final) else 0.0),
    "acc_covered": float(acc_covered) if np.isfinite(acc_covered) else np.nan,
    "good_reject": int(good_reject),
    "wasted_reject": int(wasted_reject),
    "hard_reject": hard_reject,
    "soft_reject": soft_reject,
    "n_total": n_total,
    "n_answered": n_answered,
    "n_abstained": n_abstained,
    }

def sweep_thresholds(gt, pred, conf, probs, thresholds, idk_label="IDK", idk_index=10):
    """Run evaluate_threshold over many thresholds."""
    # Run the same evaluation across thresholds to build curves/tables
    rows = [
        evaluate_threshold(gt, pred, conf, probs, t, idk_label=idk_label, idk_index=idk_index)
        for t in thresholds
    ]
    
    return pd.DataFrame(rows)


def select_threshold(df, target_acc=0.95):
    """Max coverage among thresholds reaching target_acc; else max accuracy."""
    cand = df[df["acc_covered"] >= target_acc].sort_values("coverage", ascending=False)
    return cand.iloc[0] if len(cand) else df.sort_values("acc_covered", ascending=False).iloc[0]


def get_labeled_predictions(
    dataset,
    gt_path="gt.label",
    pred_path="prediction.label",
    conf_path="prediction.confidence",
    probs_path="probabilities",
    idk_label="IDK",
    require_labels=True,
):
    """Return (df_labeled, gt, pred, conf, probs) for samples with manual GT labels."""
    assert dataset is not None, "dataset is None"

    # Pull fields once from FiftyOne; all lists align to dataset order
    gt_raw    = dataset.values(gt_path)
    pred_raw  = dataset.values(pred_path)
    conf_raw  = dataset.values(conf_path)
    probs_raw = dataset.values(probs_path)

    # Small eval table (used for confusion matrix + filtering to labeled samples)
    df = pd.DataFrame({
        "gt": gt_raw,
        "pred": pred_raw,
        "conf": conf_raw,
    })

    # Only keep samples with manual GT labels
    df = df[df["gt"].notna()].copy()
    N = len(df)
    print(f"Labeled samples: {N} / {len(gt_raw)}")
    if require_labels:
        assert N > 0, f"No manual labels found. Fill {gt_path} in the FiftyOne App first."

    df["gt"] = df["gt"].astype(str)
    df["pred"] = df["pred"].astype(str)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")  # None -> NaN

    df["correct"] = df["pred"] == df["gt"]
    df["is_idk_pred"] = df["pred"] == idk_label

    # Use df.index (original dataset indices) to pull the matching probability vectors
    idx = df.index.to_numpy()
    gt   = df["gt"].to_numpy(dtype=object)
    pred = df["pred"].to_numpy(dtype=object)
    conf = df["conf"].to_numpy(dtype=float)
    probs = [probs_raw[i] for i in idx]

    return df, gt, pred, conf, probs
