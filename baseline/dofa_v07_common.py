"""Shared utilities for v0.7 controlled DOFA source ablation and LoRA.

No official test access. Retains the v0.6 15-class decoder and masked loss.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from settings import PROJECT_DIR, RANDOM_SEED, CLASSES
from unet_core import (get_split_paths, training_statistics, confusion_from_batch,
                       metrics_from_cm, NUM_CLASSES, IGNORE)
from dofa_transfer import (FrozenDofaDataset, cache_config, verify_cache_metadata,
                           feature_path, check_cached_feature)

RESULTS = PROJECT_DIR / 'results' / 'dofa_v07'
FIGURES = PROJECT_DIR / 'assets' / 'figures' / 'dofa_v07'
MODELS = PROJECT_DIR / 'models' / 'dofa_v07'


def prepare_splits(check_all_cache=True):
    train_paths, train_names, lookup = get_split_paths('train')
    val_paths, val_names, _ = get_split_paths('val', lookup)
    assert not (train_names & val_names), 'TRAIN/VAL overlap'
    mean, std, counts, weights, debris = training_statistics(train_paths)
    meta = verify_cache_metadata(cache_config(mean, std, train_paths, val_paths))
    if check_all_cache:
        for split, paths in [('train', train_paths), ('val', val_paths)]:
            for p in paths:
                check_cached_feature(feature_path(split, p))
    for dest in [RESULTS, FIGURES, MODELS]:
        dest.mkdir(parents=True, exist_ok=True)
    return train_paths, val_paths, mean, std, counts, weights, debris, meta


def cached_loaders(train_paths, val_paths, mean, std, debris_flags, batch_size=4):
    # Use the same weighted replacement sampling mechanism as v0.6.
    train = FrozenDofaDataset(train_paths, 'train', mean, std, augment=False)
    val = FrozenDofaDataset(val_paths, 'val', mean, std, augment=False)
    patch_weights = torch.tensor([3.0 if flag else 1.0 for flag in debris_flags], dtype=torch.double)
    sampler = WeightedRandomSampler(patch_weights, num_samples=len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))
    return (DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True))


def apply_sources(head, feat, x, mode):
    """Same head capacity across all three experiments; zero a whole input source.

    DOFA-only: raw skip is zero. Spectral-only: DOFA grid is zero. Combined: both.
    Constants/normalization remain in the decoder, but no information from the
    ablated input source reaches its trainable weights.
    """
    if mode == 'dofa_only':
        x = torch.zeros_like(x)
    elif mode == 'spectral_only':
        feat = torch.zeros_like(feat)
    elif mode != 'combined':
        raise ValueError('Unknown source ablation mode: ' + str(mode))
    return head(feat, x)


def masked_weighted_ce(logits, truth, conf, class_weight):
    per_pixel = F.cross_entropy(logits.float(), truth, weight=class_weight,
                                ignore_index=IGNORE, reduction='none')
    numerator = (per_pixel * conf).sum()
    denominator = (class_weight[truth.clamp(0, NUM_CLASSES-1)] * conf).sum()
    if denominator.item() <= 0:
        return None, 0.0, 0.0
    return numerator / denominator.clamp_min(1e-8), numerator.detach().item(), denominator.detach().item()


def evaluate_head(head, val_loader, mode, device):
    head.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    ce_sum, pixels = 0.0, 0
    with torch.inference_mode():
        for x, feat, y, conf, paths in val_loader:
            x, feat, y = x.to(device), feat.to(device), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = apply_sources(head, feat, x, mode)
            ce = F.cross_entropy(logits.float(), y, ignore_index=IGNORE, reduction='none')
            valid = y != IGNORE
            ce_sum += float(ce[valid].sum().item())
            pixels += int(valid.sum().item())
            confusion_from_batch(cm, logits.argmax(1), y)
    return cm, ce_sum / max(pixels, 1)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def cm_to_rows(confusion, experiment):
    rows = []
    for i, label in enumerate(CLASSES):
        rows.append({'experiment': experiment, 'true_class': label,
                     'support': int(confusion[i].sum()),
                     'as_debris_fp': int(confusion[i, 0]) if i else 0,
                     'debris_as_this_fn': int(confusion[0, i]) if i else 0})
    return rows
