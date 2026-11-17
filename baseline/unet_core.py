"""Shared U-Net components for MARIDA v0.4. Use scripts 15, 16, 17 in PyCharm.

MARIDA label mask: 0 = UNLABELED (ignored); 1..15 = classes.
Network labels: 0..14 for 15 classes; 255 = IGNORE.
"""
from pathlib import Path
import random
import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import Dataset
from settings import CLASSES, DATA_FILE

NUM_CLASSES = len(CLASSES)
NUM_BANDS = 11
IGNORE = 255
DATA_DIR = DATA_FILE.parent
PATCHES_DIR = DATA_DIR / "patches"
SPLITS_DIR = DATA_DIR / "splits"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_split_paths(split, lookup=None):
    """Use official split manifest, never random patch splitting."""
    manifest = SPLITS_DIR / (split + "_X.txt")
    if not manifest.is_file():
        raise FileNotFoundError("Missing MARIDA official split: " + str(manifest))
    if lookup is None:
        lookup = {}
        if not PATCHES_DIR.is_dir():
            raise FileNotFoundError("Missing MARIDA patches folder: " + str(PATCHES_DIR))
        for path in PATCHES_DIR.rglob("*"):
            if path.suffix.lower() not in (".tif", ".tiff"):
                continue
            if path.stem.lower().endswith(("_cl", "_conf")):
                continue
            key = path.name.lower()
            if key in lookup:
                raise ValueError("Duplicate GeoTIFF filename: " + key)
            lookup[key] = path
    names = []
    for raw in manifest.read_text(encoding="utf-8-sig").splitlines():
        name = raw.strip().strip('"\'').replace("\\", "/").split("/")[-1]
        if not name or name.startswith("#"):
            continue
        if not name.lower().endswith((".tif", ".tiff")):
            name += ".tif"
        if not name.lower().startswith("s2_"):
            name = "S2_" + name
        names.append(name.lower())
    if len(names) != len(set(names)):
        raise ValueError("Duplicate patch names in " + str(manifest))
    missing = [x for x in names if x not in lookup]
    if missing:
        raise FileNotFoundError(f"{split} split: missing {len(missing)} TIFF images. Example: {missing[0]}")
    if not names:
        raise ValueError("Empty split file: " + str(manifest))
    return [lookup[name] for name in names], set(names), lookup


def get_all_split_paths():
    train_paths, train_names, lookup = get_split_paths("train")
    val_paths, val_names, _ = get_split_paths("val", lookup)
    test_paths, test_names, _ = get_split_paths("test", lookup)
    if train_names & val_names or train_names & test_names or val_names & test_names:
        raise AssertionError("Official split filenames overlap: STOP to avoid data leakage")
    return {"train": train_paths, "val": val_paths, "test": test_paths}


def load_raw_image(path):
    array = np.asarray(tifffile.imread(path), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 11-band GeoTIFF: {path}: {array.shape}")
    if array.shape[0] == NUM_BANDS:
        return np.ascontiguousarray(array)
    if array.shape[-1] == NUM_BANDS:
        return np.ascontiguousarray(np.moveaxis(array, -1, 0))
    raise ValueError(f"Unexpected band layout: {path}: {array.shape}")


def load_mask(path, suffix):
    filename = path.with_name(path.stem + suffix + ".tif")
    if not filename.is_file():
        raise FileNotFoundError("Missing annotation mask: " + str(filename))
    result = np.squeeze(tifffile.imread(filename))
    if result.ndim != 2:
        raise ValueError(f"Expected 2D mask: {filename}: {result.shape}")
    return result


def load_label(path):
    label = np.asarray(load_mask(path, "_cl"), dtype=np.int64)
    if not np.isin(label, np.arange(NUM_CLASSES + 1)).all():
        raise ValueError(f"Mask has unexpected class IDs: {path}")
    return label


def training_statistics(train_paths, stride=8):
    """Learn normalization and class weights from TRAIN only, no val/test information."""
    sums = np.zeros(NUM_BANDS, dtype=np.float64)
    sumsq = np.zeros(NUM_BANDS, dtype=np.float64)
    counts = np.zeros(NUM_BANDS, dtype=np.int64)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    debris_patch_flags = []
    for idx, path in enumerate(train_paths, 1):
        x = load_raw_image(path)[:, ::stride, ::stride].reshape(NUM_BANDS, -1)
        good = np.isfinite(x)
        clean = np.where(good, x, 0.0).astype(np.float64)
        sums += clean.sum(axis=1)
        sumsq += (clean * clean).sum(axis=1)
        counts += good.sum(axis=1)
        labels = load_label(path)
        valid = (labels >= 1) & (labels <= NUM_CLASSES)
        class_counts += np.bincount(labels[valid] - 1, minlength=NUM_CLASSES)
        debris_patch_flags.append(bool(np.any(labels == 1)))
        if idx % 150 == 0:
            print(f"Train-only statistics: {idx}/{len(train_paths)} patches")
    if np.any(counts == 0):
        raise ValueError("A spectral band contains no finite train values.")
    means = sums / counts
    variance = np.maximum(sumsq / counts - means ** 2, 1e-8)
    stds = np.sqrt(variance)
    if class_counts[0] == 0:
        raise ValueError("Training split has no labeled Marine Debris pixels")
    # Cap weighting of rare classes to reduce sensitivity to very few labels.
    weights = np.sqrt((class_counts.max() + 1) / (class_counts + 1))
    weights = np.clip(weights, 1.0, 8.0)
    weights[class_counts == 0] = 1.0
    return means.astype(np.float32), stds.astype(np.float32), class_counts, weights.astype(np.float32), debris_patch_flags


class MaridaSegmentationDataset(Dataset):
    """Reads a whole 11-band image. Only labeled pixels contribute to loss/metrics."""
    def __init__(self, paths, means, stds, augment=False):
        self.paths = list(paths)
        self.means = np.asarray(means, dtype=np.float32).reshape(NUM_BANDS, 1, 1)
        self.stds = np.maximum(np.asarray(stds, dtype=np.float32).reshape(NUM_BANDS, 1, 1), 1e-4)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        x = load_raw_image(path)
        raw_label = load_label(path)
        confidence = np.asarray(load_mask(path, "_conf"), dtype=np.int64)
        if raw_label.shape != x.shape[1:] or confidence.shape != raw_label.shape:
            raise ValueError("Image / label / confidence shape mismatch: " + path.name)
        x = np.where(np.isfinite(x), x, self.means)
        x = np.clip((x - self.means) / self.stds, -8, 8).astype(np.float32)
        y = np.where(raw_label == 0, IGNORE, raw_label - 1).astype(np.int64)
        pixel_weight = np.ones_like(raw_label, dtype=np.float32)
        pixel_weight[confidence == 2] = 0.5
        pixel_weight[confidence == 3] = 1.0 / 3.0
        pixel_weight[raw_label == 0] = 0.0
        if self.augment:
            if np.random.random() < 0.5:
                x, y, pixel_weight = x[:, :, ::-1], y[:, ::-1], pixel_weight[:, ::-1]
            if np.random.random() < 0.5:
                x, y, pixel_weight = x[:, ::-1, :], y[::-1, :], pixel_weight[::-1, :]
            k = np.random.randint(0, 4)
            x, y, pixel_weight = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k), np.rot90(pixel_weight, k)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(pixel_weight)),
                str(path))


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet11(nn.Module):
    """Small four-level U-Net: 11 channels in, 15 semantic classes out."""
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = ConvBlock(NUM_BANDS, base)
        self.enc2 = ConvBlock(base, 2 * base)
        self.enc3 = ConvBlock(2 * base, 4 * base)
        self.enc4 = ConvBlock(4 * base, 8 * base)
        self.pool = nn.MaxPool2d(2)
        self.center = ConvBlock(8 * base, 16 * base)
        self.up4 = nn.ConvTranspose2d(16 * base, 8 * base, 2, stride=2)
        self.dec4 = ConvBlock(16 * base, 8 * base)
        self.up3 = nn.ConvTranspose2d(8 * base, 4 * base, 2, stride=2)
        self.dec3 = ConvBlock(8 * base, 4 * base)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = ConvBlock(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = ConvBlock(2 * base, base)
        self.output = nn.Conv2d(base, NUM_CLASSES, 1)

    def forward(self, x):
        import torch.nn.functional as F
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.enc4(self.pool(c))
        z = self.center(self.pool(d))
        z = self.dec4(torch.cat([self.up4(z), d], dim=1))
        z = self.dec3(torch.cat([self.up3(z), c], dim=1))
        z = self.dec2(torch.cat([self.up2(z), b], dim=1))
        z = self.dec1(torch.cat([self.up1(z), a], dim=1))
        return self.output(z)


def confusion_from_batch(confusion, predicted, truth):
    """Accumulate 15x15 confusion on annotated pixels ONLY."""
    p = predicted.detach().cpu().numpy().reshape(-1)
    t = truth.detach().cpu().numpy().reshape(-1)
    valid = t != IGNORE
    if np.any(valid):
        confusion += np.bincount(NUM_CLASSES * t[valid] + p[valid],
                                 minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return confusion


def metrics_from_cm(cm):
    tp = int(cm[0, 0])
    fp = int(cm[:, 0].sum() - tp)
    fn = int(cm[0, :].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    per_class_f1 = []
    for i in range(NUM_CLASSES):
        den = cm[i, :].sum() + cm[:, i].sum()
        per_class_f1.append(float(2 * cm[i, i] / den) if den else float("nan"))
    return dict(tp=tp, fp=fp, fn=fn, debris_f1=f1, debris_iou=iou,
                debris_precision=precision, debris_recall=recall,
                evaluated_pixels=int(cm.sum()),
                macro_f1_present=float(np.nanmean(per_class_f1)))


def make_rgb(raw_image):
    rgb = raw_image[[3, 2, 1]].transpose(1, 2, 0).copy()
    for k in range(3):
        values = rgb[:, :, k]
        finite = values[np.isfinite(values)]
        if len(finite):
            lo, hi = np.percentile(finite, [2, 98])
            rgb[:, :, k] = np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)
    return np.nan_to_num(rgb, nan=0.0, posinf=1, neginf=0)
