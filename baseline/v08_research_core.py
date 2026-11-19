"""MARIDA v0.8: shared train/score utilities; only the supplied path lists are accessed.

The outer geographic holdout is NEVER an early-stopping set. All scores are
computed on labeled pixels (1..15); mask=0 is unlabeled, not background.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from settings import PROJECT_DIR, CLASSES
from unet_core import (UNet11, MaridaSegmentationDataset, NUM_CLASSES,
                       IGNORE, seed_everything, confusion_from_batch, metrics_from_cm,
                       load_label)
from dofa_transfer import DofaDenseHead, EMBED_DIM, PATCH_GRID
from dofa_v07_common import masked_weighted_ce

ROOT = PROJECT_DIR / 'results' / 'v08'
FIGURES = PROJECT_DIR / 'assets' / 'figures' / 'v08'
MODELS = PROJECT_DIR / 'models' / 'v08'
CACHE = PROJECT_DIR / 'baseline' / 'cache' / 'dofa_v08'
SEEDS = (42, 7, 21, 123, 2026)


def ensure_folders():
    for folder in (ROOT, FIGURES, MODELS, CACHE):
        folder.mkdir(parents=True, exist_ok=True)


def save_json(target, value):
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def json_float(v):
    """Serialize undefined per-fold scores as JSON null, not nonstandard NaN."""
    if v is None:
        return None
    try:
        z = float(v)
        return z if math.isfinite(z) else None
    except (TypeError, ValueError):
        return None


def tile_id(path):
    parts = Path(path).stem.split('_')
    if len(parts) < 4 or parts[0].lower() != 's2' or not parts[2]:
        raise ValueError('Unexpected MARIDA patch name (expected S2_date_tile_index): ' + str(path))
    return parts[2].upper()


def acquisition_id(path):
    parts = Path(path).stem.split('_')
    if len(parts) < 4 or parts[0].lower() != 's2':
        raise ValueError('Unexpected MARIDA patch name: ' + str(path))
    return parts[1] + '/' + parts[2].upper()


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode('utf-8')).hexdigest()


def split_inner_by_acquisition(outer_fit_paths, heldout_tile, fraction=0.2):
    """Fixed inner Date+Tile holdout, entirely inside outer-fit Tile IDs.

    Inner checkpoint selection deliberately never sees outer-heldout patches.
    We ensure >=1 positive debris pixel in fit and inner validation; otherwise
    explicitly abort, instead of silently selecting on held-out outer data.
    """
    groups = {}
    positive = {}
    for path in outer_fit_paths:
        name = acquisition_id(path)
        groups.setdefault(name, []).append(path)
        positive[name] = positive.get(name, 0) + int(np.count_nonzero(load_label(path) == 1))
    keys = sorted(groups)
    positive_groups = [k for k in keys if positive[k] > 0]
    if len(positive_groups) < 2:
        raise RuntimeError('Cannot create group-held-out inner split with debris on both sides: ' + heldout_tile)
    # Deterministic, seed-independent split to compare models on identical folds.
    order = sorted(keys, key=lambda k: sha_json({'heldout': heldout_tile, 'group': k, 'version': 8}))
    inner_count = max(1, int(math.ceil(len(keys) * fraction)))
    chosen = next((k for k in order if positive[k] > 0 and
                   sum(positive[z] for z in keys if z != k) > 0), None)
    if chosen is None:
        raise RuntimeError('No suitable positive inner validation group for ' + heldout_tile)
    inner_groups = {chosen}
    for k in order:
        if len(inner_groups) >= inner_count:
            break
        if k not in inner_groups and sum(positive[z] for z in keys if z not in inner_groups | {k}) > 0:
            inner_groups.add(k)
    fit = [p for p in outer_fit_paths if acquisition_id(p) not in inner_groups]
    inner = [p for p in outer_fit_paths if acquisition_id(p) in inner_groups]
    if not fit or not inner or sum(int(np.count_nonzero(load_label(p) == 1)) for p in fit) == 0:
        raise AssertionError('Invalid inner fold: ' + heldout_tile)
    assert {tile_id(p) for p in fit + inner}.isdisjoint({heldout_tile})
    assert set(map(str, fit)).isdisjoint(map(str, inner))
    return fit, inner, sorted(inner_groups)


class FeatureDataset(Dataset):
    """Full-resolution normalized bands and optional precomputed frozen tokens."""
    def __init__(self, paths, mean, std, feature_root=None):
        self.raw = MaridaSegmentationDataset(paths, mean, std, augment=False)
        self.paths = list(paths)
        self.feature_root = Path(feature_root) if feature_root else None

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        x, y, conf, name = self.raw[idx]
        if self.feature_root is None:
            feat = torch.zeros((EMBED_DIM, PATCH_GRID, PATCH_GRID), dtype=torch.float32)
        else:
            location = self.feature_root / (self.paths[idx].stem + '.npy')
            if not location.is_file():
                raise FileNotFoundError('Feature cache missing: ' + str(location))
            array = np.load(location, allow_pickle=False)
            if array.shape != (EMBED_DIM, PATCH_GRID, PATCH_GRID) or array.dtype != np.float16:
                raise ValueError('Invalid cached DOFA tensor: ' + str(location))
            feat = torch.from_numpy(array.astype(np.float32))
        return x, feat, y, conf, name


def make_loaders(train_paths, valid_paths, mean, std, train_debris_flags,
                 batch_size, seed, feature_root=None):
    train_ds = FeatureDataset(train_paths, mean, std, feature_root)
    eval_ds = FeatureDataset(valid_paths, mean, std, feature_root)
    weights = torch.tensor([3.0 if flag else 1.0 for flag in train_debris_flags], dtype=torch.double)
    if len(weights) != len(train_ds):
        raise ValueError('Sampler flags must correspond to the exact training-patch order')
    sample = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True,
                                   generator=torch.Generator().manual_seed(seed))
    return (DataLoader(train_ds, batch_size=batch_size, sampler=sample,
                       num_workers=0, pin_memory=torch.cuda.is_available()),
            DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=torch.cuda.is_available()))


def build_model(model_name, device):
    if model_name in ('spectral_only', 'dofa_combined', 'dofa_only'):
        return DofaDenseHead().to(device)
    if model_name == 'unet':
        return UNet11(base=16).to(device)
    raise ValueError('Unknown v0.8 model name: ' + model_name)


def get_logits(model, x, feat, model_name):
    if model_name == 'unet':
        return model(x)
    if model_name == 'spectral_only':
        return model(torch.zeros_like(feat), x)
    if model_name == 'dofa_only':
        return model(feat, torch.zeros_like(x))
    if model_name == 'dofa_combined':
        return model(feat, x)
    raise ValueError(model_name)


def evaluate(model, loader, model_name, device, patch_details=False):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows = []
    with torch.inference_mode():
        for x, feat, y, confidence, names in loader:
            x, feat, y = x.to(device), feat.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                pred = get_logits(model, x, feat, model_name).argmax(dim=1)
            confusion_from_batch(cm, pred, y)
            if patch_details:
                for b, name in enumerate(names):
                    p = pred[b]; t = y[b]
                    valid = t != IGNORE
                    tp = int(((t == 0) & (p == 0) & valid).sum().item())
                    fp = int(((t != 0) & (p == 0) & valid).sum().item())
                    fn = int(((t == 0) & (p != 0) & valid).sum().item())
                    positive = tp + fn
                    den = 2 * tp + fp + fn
                    rows.append({'patch': Path(name).name, 'tile': tile_id(name),
                                 'acquisition': acquisition_id(name),
                                 'labeled_pixels': int(valid.sum().item()),
                                 'debris_support': positive, 'tp': tp, 'fp': fp, 'fn': fn,
                                 'debris_f1': 2 * tp / den if den else np.nan})
    return cm, rows


def record_metrics(cm, more=None):
    m = metrics_from_cm(cm)
    return dict(more or {}, tp=int(m['tp']), fp=int(m['fp']), fn=int(m['fn']),
                debris_f1=json_float(m['debris_f1']), debris_iou=json_float(m['debris_iou']),
                debris_precision=json_float(m['debris_precision']),
                debris_recall=json_float(m['debris_recall']),
                macro_f1_present=json_float(m['macro_f1_present']),
                evaluated_pixels=int(m['evaluated_pixels']), debris_support=int(cm[0].sum()))


def train_one(model_name, train_paths, inner_paths, outer_paths, mean, std,
              class_weights, debris_flags, seed, dest, feature_root=None,
              max_epochs=30, batch_size=4, patience=8, learning_rate=3e-4,
              weight_decay=1e-4):
    """Select best epoch only on inner_paths; evaluate outer_paths ONCE afterward.

    On multi-seed official protocol, inner_paths is the *official validation*
    and outer_paths is None. On geographic holdout, inner_paths must be a
    subset of outer-fit and outer_paths is the unseen Tile to report once.
    """
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    if set(map(str, train_paths)) & set(map(str, inner_paths)):
        raise AssertionError('Fit and checkpoint-validation data overlap')
    if outer_paths is not None and (set(map(str, outer_paths)) &
           (set(map(str, train_paths)) | set(map(str, inner_paths)))):
        raise AssertionError('OUTER HOLDOUT LEAKAGE: stop')
    if model_name.startswith('dofa_') and feature_root is None:
        raise ValueError('DOFA requires fold-specific feature caches')
    if model_name in ('spectral_only', 'unet') and feature_root is not None:
        raise ValueError('Spectral/UNet must not receive DOFA cached tokens')
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Strict resume signature: changing splits, normalization or hyperparameters
    # must NOT silently reuse an older finished run.
    run_signature = sha_json({
        'protocol_schema': 'v08_train_one_v2', 'model': model_name, 'seed': int(seed),
        'fit': [p.name for p in train_paths],
        'inner': [p.name for p in inner_paths],
        'outer': [p.name for p in outer_paths] if outer_paths is not None else None,
        'mean': [float(v) for v in mean], 'std': [float(v) for v in std],
        'class_weights': [float(v) for v in class_weights],
        'sampler_debris_flags': [bool(v) for v in debris_flags],
        'feature_root': str(Path(feature_root).resolve()) if feature_root else None,
        'max_epochs': max_epochs, 'batch_size': batch_size, 'patience': patience,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
    })
    summary_path = dest / 'metrics.json'
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding='utf-8'))
        if prior.get('run_signature') != run_signature:
            raise RuntimeError('Existing run metrics belong to a different protocol: '+str(dest))
        print('SKIP completed matching experiment:', dest)
        return prior
    model = build_model(model_name, device)
    train_loader, inner_loader = make_loaders(train_paths, inner_paths, mean, std,
                                             debris_flags, batch_size, seed, feature_root)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    class_t = torch.tensor(class_weights, device=device)
    history, best, stale = [], -1.0, 0
    best_ckpt = dest / 'best.pt'
    for epoch in range(1, max_epochs + 1):
        model.train()
        numerator, denominator = 0.0, 0.0
        for x, feat, y, conf, _ in train_loader:
            x, feat, y, conf = x.to(device), feat.to(device), y.to(device), conf.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                logits = get_logits(model, x, feat, model_name)
            loss, num, den = masked_weighted_ce(logits, y, conf, class_t)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss: ' + str(dest))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            numerator += num; denominator += den
        inner_cm, _ = evaluate(model, inner_loader, model_name, device)
        m = metrics_from_cm(inner_cm)
        score = json_float(m['debris_f1'])
        if score is None:
            raise RuntimeError('Inner validation has undefined debris F1: ' + str(dest))
        scheduler.step(score)
        history.append({'epoch': epoch, 'train_weighted_ce': numerator/max(denominator,1),
                        'inner_validation_f1': score, 'inner_tp': int(m['tp']),
                        'inner_fp': int(m['fp']), 'inner_fn': int(m['fn']),
                        'learning_rate': optimizer.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(dest/'history.csv', index=False)
        print(f'{dest.name} epoch={epoch:02d} innerF1={score:.4f}', flush=True)
        if score > best + 1e-5:
            best, stale = score, 0
            torch.save({'model': model.state_dict(), 'model_name': model_name,
                        'seed': seed, 'best_epoch': epoch, 'best_inner_f1': best,
                        'normalization_mean': mean.tolist(),
                        'normalization_std': std.tolist()}, best_ckpt)
        else:
            stale += 1
        if stale >= patience:
            break
    state = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    # Critical: outer Tile NOT scored until best epoch is frozen. Official
    # validation multi-seed scores below are selection-set results, not test.
    final_paths = outer_paths if outer_paths is not None else inner_paths
    final_loader = DataLoader(FeatureDataset(final_paths, mean, std, feature_root),
                              batch_size=batch_size, shuffle=False, num_workers=0,
                              pin_memory=(device.type == 'cuda'))
    final_cm, patch_rows = evaluate(model, final_loader, model_name, device, patch_details=True)
    pd.DataFrame(final_cm, index=CLASSES, columns=CLASSES).to_csv(dest/'confusion.csv')
    pd.DataFrame(patch_rows).to_csv(dest/'patch_errors.csv', index=False)
    result = record_metrics(final_cm, {'model': model_name, 'seed': seed,
        'selection_protocol': 'inner_train_only' if outer_paths is not None else 'official_validation',
        'best_epoch': state['best_epoch'], 'best_inner_f1': best,
        'outer_evaluated_once': outer_paths is not None,
        'fit_patches': len(train_paths), 'inner_validation_patches': len(inner_paths),
        'final_evaluation_patches': len(final_paths), 'test_accessed': False,
        'run_signature': run_signature})
    save_json(summary_path, result)
    del model, optimizer, train_loader, inner_loader, final_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def cache_features(model, paths, means, stds, cache_dir, identity, batch_size=2):
    """Compute DOFA tokens under current fold's FIT-only normalization.

    Metadata are checked before reuse to prevent accidentally using precomputed
    whole-official-train normalized features in a held-out Tile fold.
    """
    from dofa_transfer import dofa_dense_tokens
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(identity, mean=[float(x) for x in means],
                   std=[float(x) for x in stds],
                   patches=sorted(str(p.name) for p in paths),
                   schema='v08_dofa_fold_features_fp16_768x16x16')
    meta_path = cache_dir / 'meta.json'
    if meta_path.is_file():
        recorded = json.loads(meta_path.read_text(encoding='utf-8'))
        if recorded != payload:
            raise RuntimeError('Stale or cross-fold DOFA cache! ' + str(cache_dir))
    else:
        save_json(meta_path, payload)
    missing = [p for p in paths if not (cache_dir / (p.stem+'.npy')).is_file()]
    for path in paths:
        saved = cache_dir / (path.stem+'.npy')
        if saved.is_file():
            arr = np.load(saved, allow_pickle=False, mmap_mode='r')
            if arr.shape != (EMBED_DIM, PATCH_GRID, PATCH_GRID) or arr.dtype != np.float16:
                raise ValueError('Bad cached tensor: ' + str(saved))
    if not missing:
        print('Feature cache complete:', cache_dir)
        return cache_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    ds = MaridaSegmentationDataset(missing, means, stds, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print('Caching frozen features:', len(missing), 'patches,', cache_dir, flush=True)
    with torch.inference_mode():
        for index, (x, _, _, filenames) in enumerate(loader, 1):
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                feat = dofa_dense_tokens(model, x).float().cpu().numpy().astype(np.float16)
            if not np.isfinite(feat).all():
                raise RuntimeError('Non-finite DOFA tokens, refusing to cache')
            for tensor, name in zip(feat, filenames):
                target = cache_dir / (Path(name).stem+'.npy')
                temp = target.with_suffix('.partial.npy')
                np.save(temp, tensor, allow_pickle=False)
                temp.replace(target)
            if index % 25 == 0:
                print('Cached', min(index*batch_size, len(missing)), '/', len(missing), flush=True)
    return cache_dir
