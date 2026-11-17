"""v0.4 Step 16: train 11-band, 15-class U-Net on MARIDA. PyCharm Run.

Uses official TRAIN for fitting; validation used to select the best checkpoint.
Official TEST never read in this script. Sparse labels: class 0 ignored.
"""
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from settings import PROJECT_DIR, RANDOM_SEED, CLASSES, BANDS
from unet_core import (UNet11, MaridaSegmentationDataset, get_split_paths,
                       seed_everything, training_statistics, confusion_from_batch,
                       metrics_from_cm, NUM_CLASSES, IGNORE)

# ------------------ EASY SETTINGS TO CHANGE ------------------
EPOCHS = 35
BATCH_SIZE = 4                 # On RTX 5070, reduce to 2 if out of VRAM.
BASE_CHANNELS = 16             # Lightweight U-Net (11 input channels).
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
PATIENCE = 10                  # Early stopping if debris F1 stops improving.
DEBRIS_PATCH_SAMPLING_WEIGHT = 3.0   # Train only; increases debris-patch frequency.
SHOW_FIGURES = False
NUM_WORKERS = 0                # Reliable with Windows / PyCharm.
# --------------------------------------------------------------

OUT = PROJECT_DIR / "results" / "unet"
FIG = PROJECT_DIR / "assets" / "figures" / "unet"
MODEL_FILE = PROJECT_DIR / "models" / "unet_best.pt"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable. Run 15_check_gpu_and_data.py and install CUDA PyTorch first.")
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
seed_everything(RANDOM_SEED)

# Read ONLY the official train/val lists (no test data access during training).
train_paths, train_names, lookup = get_split_paths("train")
val_paths, val_names, _ = get_split_paths("val", lookup)
if train_names.intersection(val_names):
    raise AssertionError("TRAIN/VAL image filenames overlap")
print("Official TRAIN patches:", len(train_paths), "| VAL patches:", len(val_paths))

means, stds, train_class_counts, class_weights, debris_flags = training_statistics(train_paths)
print("Train debris-containing patches:", sum(debris_flags), "of", len(debris_flags))
print("TRAIN-ONLY normalized means:", means.round(5).tolist())
print("TRAIN class counts:", dict(zip(CLASSES, train_class_counts.tolist())))
print("TRAIN class weights:", class_weights.round(2).tolist())

train_dataset = MaridaSegmentationDataset(train_paths, means, stds, augment=True)
val_dataset = MaridaSegmentationDataset(val_paths, means, stds, augment=False)
patch_weights = torch.as_tensor(
    [DEBRIS_PATCH_SAMPLING_WEIGHT if flag else 1.0 for flag in debris_flags], dtype=torch.double)
sampler = WeightedRandomSampler(patch_weights, num_samples=len(train_paths), replacement=True,
                                generator=torch.Generator().manual_seed(RANDOM_SEED))
loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
loader_val = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

model = UNet11(base=BASE_CHANNELS).to(device)
weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
# FP16 on supported GPUs improves speed/VRAM; safely disabled on CPU (not used here).
scaler = torch.amp.GradScaler("cuda", enabled=True)

config = {
    "study": "v0.4 11-band, 15-class U-Net on official MARIDA train/val",
    "training_split": "train_X.txt", "model_selection_split": "val_X.txt",
    "test_used_during_training": False,
    "bands": BANDS, "classes": CLASSES, "seed": RANDOM_SEED,
    "epochs_max": EPOCHS, "batch_size": BATCH_SIZE, "base_channels": BASE_CHANNELS,
    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
    "debris_patch_sampling_weight": DEBRIS_PATCH_SAMPLING_WEIGHT,
    "class_weights": class_weights.tolist(), "train_class_counts": train_class_counts.tolist(),
    "train_means": means.tolist(), "train_stds": stds.tolist(),
    "loss": "confidence * inverse-sqrt-frequency weighted CE, ignore class 0",
    "validation_model_selection": "global pooled Marine Debris F1 (labeled pixels only)",
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
}
(OUT / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

best_f1 = -1.0
best_epoch = 0
bad_epochs = 0
history = []
started = time.perf_counter()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_numerator, train_denominator = 0.0, 0.0
    for x, y, confidence, paths in loader_train:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        confidence = confidence.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(x)
            loss_by_pixel = F.cross_entropy(logits, y, weight=weights,
                                            ignore_index=IGNORE, reduction="none")
            # Weighted average over annotated pixels. Unlabeled pixels have 0 weight.
            loss_n = (loss_by_pixel * confidence).sum()
            loss_d = (confidence * weights[y.clamp(0, NUM_CLASSES - 1)]).sum()
            if loss_d.item() == 0:
                continue
            loss = loss_n / loss_d.clamp_min(1e-8)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        train_numerator += float(loss_n.detach().item())
        train_denominator += float(loss_d.detach().item())

    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    val_numerator, val_count = 0.0, 0
    with torch.inference_mode():
        for x, y, confidence, paths in loader_val:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x)
                # Validation loss is unweighted CE on annotated pixels.
                loss_by_pixel = F.cross_entropy(logits, y, ignore_index=IGNORE, reduction="none")
            valid = y != IGNORE
            val_numerator += float(loss_by_pixel[valid].sum().item())
            val_count += int(valid.sum().item())
            confusion_from_batch(confusion, logits.argmax(1), y)

    score = metrics_from_cm(confusion)
    val_f1 = score["debris_f1"]
    if np.isnan(val_f1):
        raise RuntimeError("No valid debris F1 on validation; check masks/data.")
    scheduler.step(val_f1)
    row = dict(epoch=epoch, train_weighted_loss=train_numerator / max(train_denominator, 1),
               val_ce_loss=val_numerator / max(val_count, 1),
               val_debris_f1=val_f1, val_debris_iou=score["debris_iou"],
               val_debris_precision=score["debris_precision"],
               val_debris_recall=score["debris_recall"],
               val_tp=score["tp"], val_fp=score["fp"], val_fn=score["fn"],
               learning_rate=optimizer.param_groups[0]["lr"])
    history.append(row)
    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {row['train_weighted_loss']:.4f} "
          f"| val CE {row['val_ce_loss']:.4f} | debris F1 {val_f1:.4f} "
          f"| P {row['val_debris_precision']:.3f} R {row['val_debris_recall']:.3f}")

    if val_f1 > best_f1 + 1e-5:
        best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
        checkpoint = {
            "model_state_dict": model.state_dict(), "base_channels": BASE_CHANNELS,
            "bands": BANDS, "classes": CLASSES, "mean": means.tolist(), "std": stds.tolist(),
            "best_epoch": epoch, "val_debris_f1": float(val_f1),
            "seed": RANDOM_SEED,
        }
        torch.save(checkpoint, MODEL_FILE)
        print("   New best checkpoint:", MODEL_FILE)
    else:
        bad_epochs += 1
    if bad_epochs >= PATIENCE:
        print("Early stopping: no validation F1 improvement for", PATIENCE, "epochs.")
        break

if not MODEL_FILE.is_file():
    raise RuntimeError("No checkpoint saved")

frame = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
axes[0].plot(frame.epoch, frame.train_weighted_loss, label="Train weighted CE")
axes[0].plot(frame.epoch, frame.val_ce_loss, label="Validation unweighted CE")
axes[0].set_title("Masked cross-entropy during training")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (different weighting; trends, not direct levels)")
axes[0].legend()
axes[0].grid(alpha=.2)
axes[1].plot(frame.epoch, frame.val_debris_f1, marker="o", markersize=3)
axes[1].axvline(best_epoch, linestyle="--", label=f"Selected epoch {best_epoch}")
axes[1].set_title("Validation Marine Debris F1")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1 (annotated pixels)")
axes[1].legend()
axes[1].grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "loss_and_validation_f1.png", dpi=160)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)
summary = {"best_epoch": best_epoch, "best_val_f1": best_f1,
           "epochs_completed": len(history), "seconds": round(time.perf_counter() - started, 2),
           "checkpoint": str(MODEL_FILE), "test_evaluated": False}
(OUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nTraining complete. Best epoch:", best_epoch, "F1:", round(best_f1, 4))
print("Next step: run 17_evaluate_unet.py exactly once for a finalized test comparison.")
