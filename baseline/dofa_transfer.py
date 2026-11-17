"""DOFA ViT-B -> frozen 11-band features -> trainable dense segmentation decoder.

Requires official DOFA source (Torch Hub), original pretrained checkpoint (Hugging Face),
and the existing unet_core.py data loader. NO checkpoint/encoder source is bundled.

This is FROZEN FEATURE TRANSFER, not end-to-end fine-tuning of the foundation model.
0 in MARIDA raw mask = unlabeled; IGNORE=255 for neural target.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from settings import BANDS, CLASSES, PROJECT_DIR
from unet_core import (MaridaSegmentationDataset, IGNORE, NUM_CLASSES)

# MARIDA settings.py order (central wavelengths in MICROMETERS, not nanometers).
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.
EXPECTED_BANDS = ["nm440", "nm490", "nm560", "nm665", "nm705", "nm740",
                  "nm783", "nm842", "nm865", "nm1600", "nm2200"]
WAVELENGTH_UM = [0.440, 0.490, 0.560, 0.665, 0.705, 0.740,
                 0.783, 0.842, 0.865, 1.610, 2.190]
MODEL_REPO = "zhu-xlab/DOFA:master"
WEIGHT_REPO = "earthflow/DOFA"
WEIGHT_FILE = "DOFA_ViT_base_e100.pth"
PATCH_GRID = 16  # 256 / DOFA ViT-B patch size 16
EMBED_DIM = 768
CACHE_ROOT = PROJECT_DIR / "baseline" / "cache" / "dofa_v06"
META_FILE = CACHE_ROOT / "cache_metadata.json"
RESULTS_DIR = PROJECT_DIR / "results" / "dofa_v06"
FIGURES_DIR = PROJECT_DIR / "assets" / "figures" / "dofa_v06"
MODELS_DIR = PROJECT_DIR / "models" / "dofa_v06"


def check_bands():
    if list(BANDS) != EXPECTED_BANDS or len(CLASSES) != 15:
        raise RuntimeError("MARIDA bands/classes changed: inspect wavelengths and label order before running")


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_config(mean, std, train_paths, val_paths):
    """Config contains sufficient data to reject stale/mixed cache entries."""
    check_bands()
    return {
        "schema": "marida_dofa_frozen_features_v1", "model": "DOFA ViT-Base patch16",
        "official_repo": MODEL_REPO, "weights_repo": WEIGHT_REPO,
        "weights_filename": WEIGHT_FILE, "bands": list(BANDS),
        "wavelength_um": WAVELENGTH_UM, "image_size": 256,
        "features_shape": [EMBED_DIM, PATCH_GRID, PATCH_GRID],
        "mean": list(map(float, mean)), "std": list(map(float, std)),
        "train_files": [p.name for p in train_paths],
        "val_files": [p.name for p in val_paths],
        "input_preprocessing": "v0.4 training-only mean/std, z-score clipped [-8,8]",
        "encoder_frozen": True,
        "caution": "Train statistics differ from official DOFA pretraining normalization",
    }


def save_cache_metadata(config, checkpoint_sha):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    content = dict(config, checkpoint_sha256=checkpoint_sha)
    META_FILE.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return content


def verify_cache_metadata(config):
    if not META_FILE.is_file():
        raise FileNotFoundError("DOFA cache metadata missing. Run 20_prepare_dofa.py first")
    content = json.loads(META_FILE.read_text(encoding="utf-8"))
    for key, value in config.items():
        if content.get(key) != value:
            raise RuntimeError(f"Stale DOFA cache metadata at {key}: rerun 20_prepare_dofa.py "
                               "after deleting ONLY baseline/cache/dofa_v06")
    if len(content.get("checkpoint_sha256", "")) != 64:
        raise RuntimeError("Pretrained checkpoint SHA256 missing from cache metadata")
    return content


def feature_path(split, path):
    return CACHE_ROOT / split / (Path(path).stem + ".npy")


def check_cached_feature(path):
    if not path.is_file():
        raise FileNotFoundError("Missing DOFA frozen features: " + str(path) + "\nRun 20_prepare_dofa.py")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.shape != (EMBED_DIM, PATCH_GRID, PATCH_GRID) or arr.dtype != np.float16:
        raise ValueError("Invalid DOFA feature cache shape/dtype: " + str(path))


def _interpolate_pos_embed(weight, target):
    """Official DOFA ViT-B checkpoint has 14x14 patches; MARIDA uses 16x16."""
    if weight.shape == target.shape:
        return weight
    if weight.ndim != 3 or target.ndim != 3 or weight.shape[2] != target.shape[2]:
        raise ValueError("Unexpected DOFA position embedding shape")
    in_n, out_n = weight.shape[1] - 1, target.shape[1] - 1
    in_side, out_side = math.isqrt(in_n), math.isqrt(out_n)
    if in_side ** 2 != in_n or out_side ** 2 != out_n:
        raise ValueError("DOFA positional tokens do not form a square grid")
    cls = weight[:, :1]
    grid = weight[:, 1:].transpose(1, 2).reshape(1, weight.shape[2], in_side, in_side)
    grid = F.interpolate(grid.float(), size=(out_side, out_side), mode="bicubic", align_corners=False)
    return torch.cat([cls, grid.flatten(2).transpose(1, 2)], dim=1)


def load_real_pretrained_dofa():
    """Load real upstream pretrained weights, fail rather than silently use random init.

    The external model repository is trusted as source code; weights are downloaded
    from the model owner's Hugging Face repository and verified via key coverage.
    """
    check_bands()
    try:
        from huggingface_hub import hf_hub_download
        import timm  # noqa: F401 - DOFA hubconf dependency
    except ImportError as exc:
        raise ImportError("Install timm==0.9.2 and huggingface_hub in the PyCharm dl env") from exc
    print("Loading official DOFA source via Torch Hub (first run needs internet)...")
    model = torch.hub.load(MODEL_REPO, "vit_base_dofa", pretrained=False,
                           trust_repo=True, img_size=256, global_pool=False,
                           num_classes=0)
    if not hasattr(model, "blocks") or not hasattr(model, "patch_embed"):
        raise RuntimeError("Unexpected DOFA backbone API in upstream GitHub repository")
    print("Downloading actual pretrained DOFA ViT-Base checkpoint if needed (~448 MB)...")
    weight_path = Path(hf_hub_download(repo_id=WEIGHT_REPO, filename=WEIGHT_FILE))
    if weight_path.stat().st_size < 100_000_000:
        raise RuntimeError("Checkpoint file too small; refusing random/unverified weights")
    raw = torch.load(weight_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise RuntimeError("Unexpected DOFA checkpoint format")
    state = {}
    for key, value in raw.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        state[key] = value
    target = model.state_dict()
    if "pos_embed" not in state or "pos_embed" not in target:
        raise RuntimeError("DOFA checkpoint missing positional embeddings")
    state["pos_embed"] = _interpolate_pos_embed(state["pos_embed"], target["pos_embed"])

    # Compare by tensor element count, excluding the discarded classification head.
    good = [key for key in target if key in state and state[key].shape == target[key].shape]
    total = sum(v.numel() for k, v in target.items() if not k.startswith("head."))
    loaded = sum(target[k].numel() for k in good if not k.startswith("head."))
    ratio = loaded / max(total, 1)
    if ratio < 0.98 or not any(k.startswith("patch_embed.") for k in good) or \
            not any(k.startswith("blocks.11.") for k in good):
        raise RuntimeError(f"DOFA pretrained weights mismatch: only {ratio:.2%} matched; STOP")
    cleaned = {k: state[k] for k in good}
    model.load_state_dict(cleaned, strict=False)
    print(f"VERIFIED pretrained tensor coverage: {ratio:.2%}; checkpoint: {weight_path.name}")
    sha = digest_file(weight_path)
    print("Checkpoint SHA256:", sha)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, sha


@torch.no_grad()
def dofa_dense_tokens(model, x):
    """Adapt DOFA's classification architecture to return dense patch tokens.

    Official forward_features pools tokens to one vector. We retain spatial tokens
    after the pretrained transformer and normalize with its pretrained LayerNorm.
    """
    if x.ndim != 4 or x.shape[1:] != (11, 256, 256):
        raise ValueError(f"DOFA expects [B,11,256,256], got {tuple(x.shape)}")
    waves = torch.tensor(WAVELENGTH_UM, dtype=torch.float32, device=x.device)
    tokens, _ = model.patch_embed(x, waves)
    if tokens.shape[1:] != (PATCH_GRID ** 2, EMBED_DIM):
        raise RuntimeError("DOFA patch embedding shape changed: " + str(tokens.shape))
    tokens = tokens + model.pos_embed[:, 1:, :]
    cls_token = (model.cls_token + model.pos_embed[:, :1, :]).expand(tokens.shape[0], -1, -1)
    tokens = torch.cat((cls_token, tokens), dim=1)
    for block in model.blocks:
        tokens = block(tokens)
    # Use pretrained norm from global_pool=False DOFA model.
    tokens = model.norm(tokens[:, 1:, :])
    return tokens.transpose(1, 2).reshape(-1, EMBED_DIM, PATCH_GRID, PATCH_GRID)


class FrozenDofaDataset(Dataset):
    """Pair cached frozen tokens with the original 11-band image and sparse mask."""
    def __init__(self, paths, split, mean, std, augment=False):
        self.paths = list(paths)
        self.split = split
        self.raw = MaridaSegmentationDataset(paths, mean, std, augment=False)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x, y, confidence, filename = self.raw[index]
        cached = feature_path(self.split, self.paths[index])
        feat = np.load(cached, allow_pickle=False)
        if feat.shape != (EMBED_DIM, PATCH_GRID, PATCH_GRID):
            raise ValueError("Malformed feature file: " + str(cached))
        feat = torch.tensor(feat.astype(np.float32))
        if self.augment:
            # Approximation: augment cached spatial tokens and matching raw image/mask.
            # Not equal to re-encoding an augmented image through absolute-pos ViT.
            if np.random.rand() < 0.5:
                x, y, confidence, feat = (a.flip(-1) for a in (x, y, confidence, feat))
            if np.random.rand() < 0.5:
                x, y, confidence, feat = (a.flip(-2) for a in (x, y, confidence, feat))
            k = int(np.random.randint(4))
            if k:
                x, y, confidence, feat = (torch.rot90(a, k, (-2, -1))
                                           for a in (x, y, confidence, feat))
        return x.contiguous(), feat.contiguous(), y.contiguous(), confidence.contiguous(), filename


class DofaDenseHead(nn.Module):
    """Lightweight segmentation decoder; ONLY these weights are optimized."""
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(EMBED_DIM, 64, 1, bias=False),
                                     nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                                 nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(64 + 11, 48, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, 48), nn.ReLU(inplace=True),
                                  nn.Conv2d(48, 32, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, 32), nn.ReLU(inplace=True))
        self.classify = nn.Conv2d(32, NUM_CLASSES, 1)

    def forward(self, features, raw_bands_normalized):
        if features.shape[1:] != (EMBED_DIM, 16, 16):
            raise ValueError("Cached DOFA features have incorrect shape")
        h = self.project(features)
        h = F.interpolate(h, size=(64, 64), mode="bilinear", align_corners=False)
        h = self.mid(h)
        h = F.interpolate(h, size=raw_bands_normalized.shape[-2:],
                          mode="bilinear", align_corners=False)
        h = torch.cat([h, raw_bands_normalized], dim=1)
        return self.classify(self.fuse(h))
