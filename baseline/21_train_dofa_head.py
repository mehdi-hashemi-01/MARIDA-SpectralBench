"""v0.6 / step 21: train ONLY a lightweight decoder on cached frozen DOFA features.

Uses official TRAIN for gradients and official VAL for best checkpoint selection.
Official TEST is never opened. Click Run in PyCharm; no argparse or command line.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from settings import PROJECT_DIR, BANDS, CLASSES, RANDOM_SEED
from unet_core import (get_split_paths, training_statistics,
                       seed_everything, confusion_from_batch, metrics_from_cm,
                       NUM_CLASSES, IGNORE)
from dofa_transfer import (FrozenDofaDataset, DofaDenseHead, check_cached_feature,
                           feature_path, cache_config, verify_cache_metadata,
                           RESULTS_DIR, FIGURES_DIR, MODELS_DIR)

# ------------------ SETTINGS (NO CLI) ----------------------
EPOCHS = 30
BATCH_SIZE = 4            # Reduce to 2 if CUDA out of memory.
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
PATIENCE = 8
NUM_WORKERS = 0           # Windows/PyCharm stability
USE_DEBRIS_OVERSAMPLING = True  # Matches v0.5 training sampling convention.
USE_CLASS_WEIGHTS = True        # Matches v0.5 weighting convention.
AUGMENT_CACHED_FEATURES = False # Avoid flipping pretrained positional features.
FORCE_RETRAIN = False
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable: run 15_check_gpu_and_data.py first")
device = torch.device("cuda")
seed_everything(RANDOM_SEED)
print("Training frozen-DOFA segmentation HEAD on:", torch.cuda.get_device_name(0))

train_paths, train_names, lookup = get_split_paths("train")
val_paths, val_names, _ = get_split_paths("val", lookup)
if train_names & val_names:
    raise AssertionError("Train/val overlap")
means, stds, class_counts, class_weights, debris_flags = training_statistics(train_paths)
config = cache_config(means, stds, train_paths, val_paths)
meta = verify_cache_metadata(config)
for split, paths in (("train", train_paths), ("val", val_paths)):
    for path in paths:
        check_cached_feature(feature_path(split, path))
print("Pretrained checkpoint verified in stage 20:", meta["checkpoint_sha256"][:16])
print("Encoder is FROZEN. Optimizing segmentation decoder only.")

for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
checkpoint_path = MODELS_DIR / "dofa_frozen_head_best.pt"
history_path = RESULTS_DIR / "training_history.csv"
if checkpoint_path.exists() and not FORCE_RETRAIN:
    raise RuntimeError("A previous DOFA head checkpoint exists. Set FORCE_RETRAIN=True "
                       "only if you intentionally want to overwrite it, or run 22_analyze_dofa.py")
train_ds = FrozenDofaDataset(train_paths, "train", means, stds,
                             augment=AUGMENT_CACHED_FEATURES)
val_ds = FrozenDofaDataset(val_paths, "val", means, stds, augment=False)
patch_weights = ([3.0 if flag else 1.0 for flag in debris_flags]
                 if USE_DEBRIS_OVERSAMPLING else [1.0] * len(debris_flags))
sampler = WeightedRandomSampler(
    torch.tensor(patch_weights, dtype=torch.double), num_samples=len(train_ds),
    replacement=True, generator=torch.Generator().manual_seed(RANDOM_SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
head = DofaDenseHead().to(device)
class_weight_t = torch.tensor(class_weights if USE_CLASS_WEIGHTS else
                               np.ones(NUM_CLASSES, dtype=np.float32), device=device)
optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
scaler = torch.amp.GradScaler("cuda", enabled=True)

run_config = {
    "experiment": "DOFA-ViT-Base (pretrained, frozen encoder) + dense segmentation head",
    "encoder": "DOFA e100", "checkpoint_sha256": meta["checkpoint_sha256"],
    "stage20_cache": str(meta.get("schema")),
    "training": "segmentation head only; NO backbone fine-tuning",
    "epochs_max": EPOCHS, "batch_size": BATCH_SIZE, "seed": RANDOM_SEED,
    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
    "patience": PATIENCE, "train_patch_count": len(train_paths),
    "val_patch_count": len(val_paths), "test_accessed": False,
    "bands": list(BANDS), "classes": list(CLASSES),
    "training_class_counts": class_counts.tolist(), "class_weights": class_weights.tolist(),
    "uses_debris_oversampling": USE_DEBRIS_OVERSAMPLING,
    "uses_class_weights": USE_CLASS_WEIGHTS,
    "augment_cached_features": AUGMENT_CACHED_FEATURES,
    "normalization": "train-only mean/std (same as U-Net v0.4); possible DOFA pretrain domain mismatch",
    "selection": "pooled labeled-pixel MARIDA class-1 F1 on official validation",
    "decoder_has_raw_spectral_skip": True,
    "caution": "Frozen DOFA + trainable spectral skip; not an isolated measurement of pretrained features",
}
(RESULTS_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

best_f1, best_epoch, stale = -1.0, 0, 0
history = []
started = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    head.train()
    loss_num, loss_den = 0.0, 0.0
    for x, feature, y, confidence, filenames in train_loader:
        x = x.to(device, non_blocking=True)
        feature = feature.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        confidence = confidence.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = head(feature, x)
        per_pixel = F.cross_entropy(logits.float(), y, weight=class_weight_t,
                                    ignore_index=IGNORE, reduction="none")
        weighted_num = (per_pixel * confidence).sum()
        weighted_den = (class_weight_t[y.clamp(0, NUM_CLASSES-1)] * confidence).sum()
        if weighted_den.item() == 0:
            continue
        loss = weighted_num / weighted_den.clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; inspect DOFA features")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_num += float(weighted_num.detach().item())
        loss_den += float(weighted_den.detach().item())

    head.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    val_loss_sum, valid_count = 0., 0
    with torch.inference_mode():
        for x, feature, y, confidence, filenames in val_loader:
            x = x.to(device, non_blocking=True)
            feature = feature.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = head(feature, x)
            valid = y != IGNORE
            ce = F.cross_entropy(logits.float(), y, ignore_index=IGNORE, reduction="none")
            val_loss_sum += float(ce[valid].sum().item())
            valid_count += int(valid.sum().item())
            confusion_from_batch(cm, logits.argmax(1), y)
    m = metrics_from_cm(cm)
    score = float(m["debris_f1"])
    scheduler.step(score)
    record = {
        "epoch": epoch, "train_weighted_ce": loss_num / max(loss_den, 1),
        "val_unweighted_ce": val_loss_sum / max(valid_count, 1),
        "val_debris_f1": score, "val_debris_precision": m["debris_precision"],
        "val_debris_recall": m["debris_recall"], "val_debris_iou": m["debris_iou"],
        "val_tp": m["tp"], "val_fp": m["fp"], "val_fn": m["fn"],
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    history.append(record)
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Epoch {epoch:02d}/{EPOCHS}: trainCE={record['train_weighted_ce']:.4f} "
          f"valF1={score:.4f} valFP={m['fp']} valFN={m['fn']}")
    if not np.isfinite(score):
        raise RuntimeError("Validation Debris F1 missing; check masks")
    if score > best_f1 + 1e-5:
        best_f1, best_epoch, stale = score, epoch, 0
        torch.save({"head_state_dict": head.state_dict(), "best_epoch": epoch,
                    "best_val_f1": score, "run_config": run_config,
                    "mean": means.tolist(), "std": stds.tolist(),
                    "pretrained_sha256": meta["checkpoint_sha256"]}, checkpoint_path)
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
            RESULTS_DIR / "best_val_confusion.csv")
    else:
        stale += 1
    if stale >= PATIENCE:
        print("Early stop: no validation F1 improvement for", PATIENCE, "epochs")
        break

plt.figure(figsize=(9, 4.5))
xs = [r["epoch"] for r in history]
ax1 = plt.gca()
ax1.plot(xs, [r["train_weighted_ce"] for r in history], label="Train weighted CE")
ax1.plot(xs, [r["val_unweighted_ce"] for r in history], label="Val unweighted CE")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy (different weighting; do not compare levels directly)")
ax2 = ax1.twinx()
ax2.plot(xs, [r["val_debris_f1"] for r in history], linestyle="--", label="Val debris F1")
ax2.set_ylabel("Marine Debris F1")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
plt.title("DOFA frozen encoder / trainable dense head — Train and Validation")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "training_curves.png", dpi=160)
plt.close()
print("Best VAL F1:", round(best_f1, 4), "at epoch:", best_epoch)
print("Elapsed training seconds:", round(time.perf_counter() - started, 1))
print("Checkpoint:", checkpoint_path)
print("TEST was never accessed. Next: 22_analyze_dofa.py")
