"""v0.5: controlled U-Net training ablations. Run directly in PyCharm.

Only TRAIN is used to update weights, and only official VAL selects checkpoints.
Official TEST is NEVER opened by this script. Existing v0.4 files are unchanged.

Four controlled configurations (all retain the same 11-band/15-class U-Net):
 - v04_replay: reproduce the previous training recipe with a fresh run
 - no_class_weights: only disable inverse-frequency class weighting
 - no_debris_oversampling: only remove preference for debris-containing patches
 - extra_spectral_jitter: only add mild training-only spectral perturbations
"""
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from settings import PROJECT_DIR, RANDOM_SEED, BANDS, CLASSES
from unet_core import (UNet11, MaridaSegmentationDataset, get_split_paths,
                       seed_everything, training_statistics,
                       confusion_from_batch, metrics_from_cm, NUM_CLASSES, IGNORE)

# ----------------- SETTINGS (NO COMMAND-LINE ARGUMENTS) -----------------
EPOCHS = 35
BATCH_SIZE = 4              # Set 2 if GPU runs out of memory.
BASE_CHANNELS = 16
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
PATIENCE = 10
NUM_WORKERS = 0             # Most reliable on Windows / PyCharm.
SHOW_FIGURES = False
FORCE_RETRAIN_COMPLETED = False
EXPERIMENTS = ["v04_replay", "no_class_weights", "no_debris_oversampling",
               "extra_spectral_jitter"]

# The following two switches are for code testing, NOT research.
# Leave them False for a real MARIDA experiment.
SMOKE_MODE = os.environ.get("MARIDA_V05_SMOKE") == "1"
ALLOW_CPU_SMOKE = os.environ.get("MARIDA_V05_ALLOW_CPU") == "1"
# -----------------------------------------------------------------------

if SMOKE_MODE:
    EPOCHS = 2
    PATIENCE = 2
    print("*** SMOKE TEST: small synthetic subset; DO NOT REPORT AS MARIDA RESULT ***")
    torch.set_num_threads(1)

if not torch.cuda.is_available():
    if not (SMOKE_MODE and ALLOW_CPU_SMOKE):
        raise RuntimeError("CUDA unavailable. Run 15_check_gpu_and_data.py; RTX 5070 expected.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
print("Device:", device, "| PyTorch:", torch.__version__)

out = PROJECT_DIR / "results" / ("unet_v05_smoke" if SMOKE_MODE else "unet_v05")
fig_dir = PROJECT_DIR / "assets" / "figures" / ("unet_v05_smoke" if SMOKE_MODE else "unet_v05")
models_dir = PROJECT_DIR / "models" / ("unet_v05_smoke" if SMOKE_MODE else "unet_v05")
for directory in [out, fig_dir, models_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# No test split is read here. All class statistics come from the official TRAIN.
train_paths, train_names, lookup = get_split_paths("train")
val_paths, val_names, _ = get_split_paths("val", lookup)
if train_names & val_names:
    raise AssertionError("Train/val patch overlap; stopping")
if SMOKE_MODE:
    train_paths = train_paths[:4]
    val_paths = val_paths[:2]
print("TRAIN patches:", len(train_paths), "| VAL patches:", len(val_paths))

means, stds, counts, inverse_freq_weights, debris_flags = training_statistics(train_paths)
print("Train-only class counts:", counts.tolist())
print("Debris-positive training patches:", sum(debris_flags), "of", len(debris_flags))

# Baseline includes flip/rotate on TRAIN; validation is never augmented.
train_ds = MaridaSegmentationDataset(train_paths, means, stds, augment=True)
val_ds = MaridaSegmentationDataset(val_paths, means, stds, augment=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

all_rows = []
for experiment in EXPERIMENTS:
    experiment_out = out / experiment
    experiment_out.mkdir(parents=True, exist_ok=True)
    checkpoint_file = models_dir / f"{experiment}_best.pt"
    summary_file = experiment_out / "training_summary.json"
    if (not FORCE_RETRAIN_COMPLETED and checkpoint_file.is_file()
            and summary_file.is_file()):
        existing = json.loads(summary_file.read_text(encoding="utf-8"))
        print("SKIP completed:", experiment, "val F1:", existing["best_val_f1"])
        all_rows.append({"experiment": experiment, **existing})
        continue

    seed_everything(RANDOM_SEED)
    weighted_classes = experiment != "no_class_weights"
    debris_sampling = experiment != "no_debris_oversampling"
    use_jitter = experiment == "extra_spectral_jitter"
    class_weights = (inverse_freq_weights if weighted_classes
                     else np.ones(NUM_CLASSES, dtype=np.float32))
    patch_weights = ([3.0 if flag else 1.0 for flag in debris_flags]
                     if debris_sampling else [1.0] * len(train_paths))
    # Even the uniform condition uses replacement, so sampler mechanics/epoch size
    # stay identical; ONLY patch sampling probabilities change.
    sampler = WeightedRandomSampler(
        weights=torch.tensor(patch_weights, dtype=torch.double),
        num_samples=len(train_paths), replacement=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model = UNet11(base=BASE_CHANNELS).to(device)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    train_means = torch.as_tensor(means, dtype=torch.float32, device=device).view(1, 11, 1, 1)
    train_stds = torch.as_tensor(stds, dtype=torch.float32, device=device).view(1, 11, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    config = {
        "experiment": experiment, "based_on": "v0.4 U-Net recipe",
        "seed": RANDOM_SEED, "epochs_max": EPOCHS, "batch_size": BATCH_SIZE,
        "base_channels": BASE_CHANNELS, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "uses_class_frequency_weights": weighted_classes,
        "uses_debris_patch_oversampling": debris_sampling,
        "debris_patch_sampling_weight": 3.0 if debris_sampling else 1.0,
        "training_geometric_flips_rotations": True,
        "training_extra_spectral_jitter": use_jitter,
        "spectral_jitter": (
            "per-patch raw reflectance gain +/-3%; Gaussian noise 0.015 x train band std"
            if use_jitter else "disabled"),
        "train_class_weights": class_weights.tolist(),
        "train_class_counts": counts.tolist(),
        "train_means": means.tolist(), "train_stds": stds.tolist(),
        "input_bands": BANDS, "classes": CLASSES,
        "train_patches": len(train_paths), "validation_patches": len(val_paths),
        "test_accessed": False, "smoke_mode": SMOKE_MODE,
        "selection": "maximum pooled Marine Debris F1 on labeled VAL pixels",
        "validation_not_augmented": True,
        "unlabeled_mask_value": 0,
        "neural_target_ignore_value": IGNORE,
        "sampling": "weighted replacement sampling; same epoch length",
    }
    (experiment_out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n===", experiment, "===")
    best_f1, best_epoch, no_improve = -1.0, 0, 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_num, train_den = 0.0, 0.0
        for x, y, confidence, filenames in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            confidence = confidence.to(device, non_blocking=True)

            if use_jitter:
                # Reconstruct clipped reflectance, apply small physical-scale
                # perturbations, re-standardize; TRAIN ONLY, masks untouched.
                patch_gain = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=device) - .5) * .06
                raw = x * train_stds + train_means
                raw = raw * patch_gain + .015 * train_stds * torch.randn_like(raw)
                x = ((raw - train_means) / train_stds).clamp(-8, 8)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp_enabled):
                logits = model(x)
                loss_map = F.cross_entropy(logits.float(), y, weight=weights,
                                           ignore_index=IGNORE, reduction="none")
                numerator = (loss_map * confidence).sum()
                denominator = (confidence * weights[y.clamp(0, NUM_CLASSES - 1)]).sum()
                if denominator.item() == 0:
                    continue
                loss = numerator / denominator.clamp_min(1e-8)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_num += float(numerator.detach().item())
            train_den += float(denominator.detach().item())

        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        val_numerator, val_pixels = 0.0, 0
        with torch.inference_mode():
            for x, y, confidence, filenames in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp_enabled):
                    logits = model(x)
                ce = F.cross_entropy(logits.float(), y, ignore_index=IGNORE,
                                     reduction="none")
                valid = y != IGNORE
                val_numerator += float(ce[valid].sum().item())
                val_pixels += int(valid.sum().item())
                confusion_from_batch(cm, logits.argmax(1), y)

        metrics = metrics_from_cm(cm)
        val_f1 = float(metrics["debris_f1"])
        if not np.isfinite(val_f1):
            raise RuntimeError("No Marine Debris F1 in validation; check label masks")
        scheduler.step(val_f1)
        record = {"epoch": epoch, "train_weighted_ce": train_num / max(train_den, 1),
                  "val_unweighted_ce": val_numerator / max(val_pixels, 1),
                  "val_debris_f1": val_f1, "val_debris_precision": metrics["debris_precision"],
                  "val_debris_recall": metrics["debris_recall"],
                  "val_debris_iou": metrics["debris_iou"],
                  "val_tp": metrics["tp"], "val_fp": metrics["fp"],
                  "val_fn": metrics["fn"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        pd.DataFrame(history).to_csv(experiment_out / "training_history.csv", index=False)
        print(f"{experiment:25s} epoch {epoch:02d}/{EPOCHS}: "
              f"train CE={record['train_weighted_ce']:.4f}, "
              f"val CE={record['val_unweighted_ce']:.4f}, F1={val_f1:.4f}, "
              f"FP={metrics['fp']}, FN={metrics['fn']}")

        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "base_channels": BASE_CHANNELS, "bands": BANDS,
                        "classes": CLASSES, "mean": means.tolist(), "std": stds.tolist(),
                        "best_epoch": epoch, "val_debris_f1": val_f1,
                        "experiment": experiment, "seed": RANDOM_SEED}, checkpoint_file)
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
                experiment_out / "best_val_confusion.csv")
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stopping:", experiment)
            break

    seconds = round(time.perf_counter() - started, 2)
    summary = {"best_epoch": best_epoch, "best_val_f1": best_f1,
               "epochs_completed": len(history), "train_seconds": seconds,
               "checkpoint": str(checkpoint_file), "test_evaluated": False,
               "smoke_mode": SMOKE_MODE}
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    all_rows.append({"experiment": experiment, **summary})

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame.epoch, frame.train_weighted_ce, label="Train weighted CE")
    axes[0].plot(frame.epoch, frame.val_unweighted_ce, label="Val unweighted CE")
    axes[0].set_title(experiment + " | masked CE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (different weighting)")
    axes[0].legend()
    axes[1].plot(frame.epoch, frame.val_debris_f1, marker="o", markersize=3)
    axes[1].axvline(best_epoch, linestyle="--", label=f"Best epoch {best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Marine Debris F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{experiment}_learning_curves.png", dpi=160)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

pd.DataFrame(all_rows).to_csv(out / "training_runs.csv", index=False)
print("\nAll completed training runs:", out / "training_runs.csv")
print("No official test images or labels were loaded.")
print("NEXT: Run 19_analyze_unet_v05.py in PyCharm.")
