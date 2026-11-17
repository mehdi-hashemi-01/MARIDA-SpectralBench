"""v0.6 / step 22: evaluate DOFA frozen features on OFFICIAL VALIDATION ONLY.

Creates class-1 F1, 15-class confusion, patch error table, comparison with
ARCHIVED RF/U-Net validation, and 5-panel qualitative cases. TEST is untouched.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.utils.data import DataLoader

from settings import PROJECT_DIR, CLASSES, BANDS
from unet_core import (get_split_paths, MaridaSegmentationDataset, UNet11,
                       load_raw_image, make_rgb, training_statistics,
                       metrics_from_cm, confusion_from_batch, NUM_CLASSES, IGNORE)
from dofa_transfer import (DofaDenseHead, FrozenDofaDataset, cache_config,
                           verify_cache_metadata, check_cached_feature, feature_path,
                           RESULTS_DIR, FIGURES_DIR, MODELS_DIR)

# ---------------- PYCHARM SETTINGS -----------------
BATCH_SIZE = 4
MAX_PATCH_FIGURES = 4
SHOW_FIGURES = False
# ---------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError("RTX CUDA unavailable. Run 15_check_gpu_and_data.py first")
device = torch.device("cuda")
checkpoint_file = MODELS_DIR / "dofa_frozen_head_best.pt"
if not checkpoint_file.is_file():
    raise FileNotFoundError("Run 21_train_dofa_head.py first: " + str(checkpoint_file))
head_ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

train_paths, train_names, lookup = get_split_paths("train")
val_paths, val_names, _ = get_split_paths("val", lookup)
if train_names & val_names:
    raise AssertionError("Official train and val overlap")
means, stds, _, _, _ = training_statistics(train_paths)
meta = verify_cache_metadata(cache_config(means, stds, train_paths, val_paths))
if meta["checkpoint_sha256"] != head_ckpt["pretrained_sha256"]:
    raise RuntimeError("DOFA cache and saved decoder checkpoint come from different pretrained weights")
for p in val_paths:
    check_cached_feature(feature_path("val", p))

head = DofaDenseHead().to(device)
head.load_state_dict(head_ckpt["head_state_dict"])
head.eval()
val_ds = FrozenDofaDataset(val_paths, "val", means, stds, augment=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
patch_rows = []
cloud_idx, wave_idx = CLASSES.index("Clouds"), CLASSES.index("Waves")
with torch.inference_mode():
    for x, features, y, conf, filenames in val_loader:
        x = x.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            p = head(features, x).argmax(1)
        confusion_from_batch(cm, p, y)
        pred = p.cpu().numpy()
        y_np = y.numpy()
        for i, filename in enumerate(filenames):
            valid = y_np[i] != IGNORE
            local = np.bincount(
                NUM_CLASSES * y_np[i][valid] + pred[i][valid],
                minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            m = metrics_from_cm(local)
            patch_rows.append({
                "patch": Path(filename).name, "path": str(filename),
                "labeled_pixels": int(local.sum()),
                "true_debris": int(local[0, :].sum()),
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "debris_f1": m["debris_f1"],
                "clouds_as_debris_fp": int(local[cloud_idx, 0]),
                "waves_as_debris_fp": int(local[wave_idx, 0]),
                "debris_as_clouds_fn": int(local[0, cloud_idx]),
                "debris_as_waves_fn": int(local[0, wave_idx]),
            })

patch_df = pd.DataFrame(patch_rows)
metrics = metrics_from_cm(cm)
for k in ("tp", "fp", "fn"):
    if int(patch_df[k].sum()) != metrics[k]:
        raise AssertionError("Patch confusion totals do not match global confusion matrix: " + k)
if metrics["evaluated_pixels"] != 213102 or len(val_paths) != 328:
    raise RuntimeError("Validation label count/patch count differs from archived MARIDA split. "
                       "Check dataset, preprocessing and official manifest")
patch_df.to_csv(RESULTS_DIR / "validation_patch_errors.csv", index=False)
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(RESULTS_DIR / "validation_confusion.csv")
fp_source = pd.DataFrame([
    {"true_class": CLASSES[i], "labeled_support": int(cm[i, :].sum()),
     "false_positive_as_debris": int(cm[i, 0])}
    for i in range(1, NUM_CLASSES)
]).sort_values("false_positive_as_debris", ascending=False)
fp_source.to_csv(RESULTS_DIR / "validation_fp_by_true_class.csv", index=False)
summary = {"experiment": "DOFA ViT-B frozen encoder + spectral skip decoder",
           "split": "val", "marine_debris_f1": metrics["debris_f1"],
           "marine_debris_iou": metrics["debris_iou"],
           "marine_debris_precision": metrics["debris_precision"],
           "marine_debris_recall": metrics["debris_recall"],
           "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
           "evaluated_pixels": metrics["evaluated_pixels"],
           "best_epoch": int(head_ckpt["best_epoch"]),
           "pretrained_checkpoint_sha256": head_ckpt["pretrained_sha256"],
           "clouds_as_debris_fp": int(cm[cloud_idx, 0]),
           "waves_as_debris_fp": int(cm[wave_idx, 0])}
pd.DataFrame([summary]).to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)
print("DOFA frozen-head validation F1:", round(metrics["debris_f1"], 4),
      "FP:", metrics["fp"], "FN:", metrics["fn"])

# Comparison includes ONLY available historical validation CSV records.
rows = [{"model": "DOFA frozen + dense decoder (v0.6)", "split": "val",
         "marine_debris_f1": metrics["debris_f1"],
         "marine_debris_precision": metrics["debris_precision"],
         "marine_debris_recall": metrics["debris_recall"],
         "marine_debris_iou": metrics["debris_iou"],
         "evaluated_pixels": metrics["evaluated_pixels"],
         "provenance": "new pretrained DOFA experiment; frozen encoder"}]
rf_csv = PROJECT_DIR / "results" / "benchmark" / "validation_comparison.csv"
if rf_csv.is_file():
    for _, r in pd.read_csv(rf_csv).iterrows():
        rows.append({"model": "Archived RF: " + str(r["model"]),
                     "split": "val", "marine_debris_f1": float(r["marine_debris_f1"]),
                     "marine_debris_precision": float(r["marine_debris_precision"]),
                     "marine_debris_recall": float(r["marine_debris_recall"]),
                     "marine_debris_iou": float(r["marine_debris_iou"]),
                     "evaluated_pixels": int(r["evaluated_pixels"]),
                     "provenance": "v0.1 archived; separate historical training"})
v05_csv = PROJECT_DIR / "results" / "unet_v05" / "validation_model_comparison.csv"
if v05_csv.is_file():
    for _, r in pd.read_csv(v05_csv).iterrows():
        rows.append({"model": "U-Net v0.5: " + str(r["experiment"]),
                     "split": "val", "marine_debris_f1": float(r["debris_f1"]),
                     "marine_debris_precision": float(r["debris_precision"]),
                     "marine_debris_recall": float(r["debris_recall"]),
                     "marine_debris_iou": float(r["debris_iou"]),
                     "evaluated_pixels": int(r["evaluated_pixels"]),
                     "provenance": "v0.5 archived; separate historical training"})
compare = pd.DataFrame(rows)
if compare.evaluated_pixels.nunique() != 1:
    raise AssertionError("Comparison models used different evaluated pixel counts")
compare.sort_values("marine_debris_f1", ascending=False).to_csv(
    RESULTS_DIR / "validation_model_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 8))
normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASSES, rotation=80, ha="right", fontsize=7)
ax.set_yticklabels(CLASSES, fontsize=7)
ax.set_ylabel("True class"); ax.set_xlabel("Predicted class")
ax.set_title("DOFA frozen-head: normalized VAL confusion (labeled pixels only)")
fig.colorbar(image, ax=ax, fraction=0.02, pad=0.03)
fig.tight_layout(); fig.savefig(FIGURES_DIR / "validation_confusion_normalized.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.5))
subset = compare.loc[compare.model.str.contains("DOFA|U-Net v0.5: v04_replay|Archived RF: spectral_plus_indices")]
ax.bar(subset.model, subset.marine_debris_f1)
ax.set_ylim(0, 1); ax.set_ylabel("Labeled-pixel Marine Debris F1")
ax.set_title("Official MARIDA VAL — different training recipes / snapshots")
ax.tick_params(axis="x", labelrotation=14)
fig.tight_layout(); fig.savefig(FIGURES_DIR / "model_f1_comparison.png", dpi=160)
plt.close(fig)

# Optional U-Net v0.5 comparison on exactly the same qualitative validation patches.
unet_path = PROJECT_DIR / "models" / "unet_v05" / "v04_replay_best.pt"
unet = None
if unet_path.is_file():
    prev = torch.load(unet_path, map_location="cpu", weights_only=True)
    if list(prev.get("bands", [])) == list(BANDS) and list(prev.get("classes", [])) == list(CLASSES):
        unet = UNet11(base=int(prev["base_channels"])).to(device)
        unet.load_state_dict(prev["model_state_dict"]); unet.eval()
        prev_mean = np.asarray(prev["mean"], dtype=np.float32).reshape(11, 1, 1)
        prev_std = np.asarray(prev["std"], dtype=np.float32).reshape(11, 1, 1)
        print("U-Net v0.5 checkpoint available: creating paired visualizations")

candidate_names = ["S2_11-6-18_16PCC_26.tif"]
candidate_names += patch_df.sort_values("fp", ascending=False).head(3).patch.tolist()
candidate_names += patch_df.sort_values("fn", ascending=False).head(3).patch.tolist()
unique_names = list(dict.fromkeys(candidate_names))[:MAX_PATCH_FIGURES]
path_by_name = {p.name.lower(): p for p in val_paths}
binary_cmap = ListedColormap(["#e8eef2", "#00a6a6"])
binary_cmap.set_bad("#9fa9b3")
error_cmap = ListedColormap(["#edf2f5", "#269c68", "#df543d", "#377db7"])
error_cmap.set_bad("#9fa9b3")
with torch.inference_mode():
    for name in unique_names:
        path = path_by_name.get(name.lower())
        if path is None:
            continue
        one = FrozenDofaDataset([path], "val", means, stds, augment=False)
        x, feat, y, conf, filename = one[0]
        x_cuda = x[None].to(device); feat_cuda = feat[None].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            dofa_pred = head(feat_cuda, x_cuda).argmax(1)[0].cpu().numpy()
        y_np = y.numpy(); valid = y_np != IGNORE
        gt = y_np == 0; dp = dofa_pred == 0
        err = np.zeros_like(dofa_pred, dtype=np.int16)
        err[valid & gt & dp] = 1
        err[valid & ~gt & dp] = 2
        err[valid & gt & ~dp] = 3
        maps = [gt.astype(int), dp.astype(int)]
        titles = ["Ground Truth (annotated)", "DOFA predicted debris"]
        if unet is not None:
            raw = load_raw_image(path)
            norm = np.clip((np.where(np.isfinite(raw), raw, prev_mean) - prev_mean) /
                           np.maximum(prev_std, 1e-4), -8, 8).astype(np.float32)
            with torch.autocast("cuda", dtype=torch.float16):
                up = unet(torch.from_numpy(norm[None]).to(device)).argmax(1)[0].cpu().numpy()
            maps.append((up == 0).astype(int))
            titles.append("U-Net v0.5 predicted debris")
        fig, axes = plt.subplots(1, 3 + len(maps), figsize=(4 * (3 + len(maps)), 4))
        axes[0].imshow(make_rgb(load_raw_image(path)))
        axes[0].set_title("Sentinel-2 RGB")
        for i, (m, title) in enumerate(zip(maps, titles), 1):
            axes[i].imshow(np.ma.masked_where(~valid, m), cmap=binary_cmap,
                           vmin=-0.5, vmax=1.5, interpolation="nearest")
            axes[i].set_title(title)
        idx = len(maps) + 1
        axes[idx].imshow(np.ma.masked_where(~valid, err), cmap=error_cmap,
                         vmin=-0.5, vmax=3.5, interpolation="nearest")
        axes[idx].set_title("DOFA errors: green TP / red FP / blue FN")
        # Last panel is annotation coverage for correct interpretation of gray background.
        axes[idx + 1].imshow(valid, cmap="gray", vmin=0, vmax=1)
        axes[idx + 1].set_title("Annotation coverage (white=valid)")
        for axis in axes:
            axis.set_xticks([]); axis.set_yticks([])
        fig.suptitle(path.name + " | pixels WITHOUT annotations excluded from scoring", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / (path.stem + "_val_visual_comparison.png"), dpi=150)
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)

print("Results:", RESULTS_DIR)
print("Figures:", FIGURES_DIR)
print("Official TEST not used. DOFA encoder remained frozen.")
