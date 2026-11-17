"""v0.5: evaluate ALL trained variants on official VALIDATION only.

Outputs: per-model metrics, cloud/wave FP and missed-debris errors, 15-class
confusion matrices, per-patch errors (including patches with NO true debris),
learning curves and side-by-side segmentation/error maps.

IMPORTANT: No official test images or labels are read in this script.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from settings import PROJECT_DIR, BANDS, CLASSES
from unet_core import (UNet11, MaridaSegmentationDataset, get_split_paths,
                       load_raw_image, make_rgb, confusion_from_batch,
                       metrics_from_cm, NUM_CLASSES, IGNORE)

BATCH_SIZE = 4
SHOW_FIGURES = False
MAX_VISUALIZED_PATCHES = 3
SMOKE_MODE = os.environ.get("MARIDA_V05_SMOKE") == "1"
ALLOW_CPU_SMOKE = os.environ.get("MARIDA_V05_ALLOW_CPU") == "1"
VARIANTS = ["v04_replay", "no_class_weights", "no_debris_oversampling",
            "extra_spectral_jitter"]

if not torch.cuda.is_available() and not (SMOKE_MODE and ALLOW_CPU_SMOKE):
    raise RuntimeError("CUDA not found. Run 15_check_gpu_and_data.py first.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if SMOKE_MODE:
    torch.set_num_threads(1)
    print("*** SMOKE OUTPUT: not a real research result ***")

out = PROJECT_DIR / "results" / ("unet_v05_smoke" if SMOKE_MODE else "unet_v05")
fig_dir = PROJECT_DIR / "assets" / "figures" / ("unet_v05_smoke" if SMOKE_MODE else "unet_v05")
models_dir = PROJECT_DIR / "models" / ("unet_v05_smoke" if SMOKE_MODE else "unet_v05")
for folder in (out, fig_dir):
    folder.mkdir(parents=True, exist_ok=True)

# Validation only. Deliberately NOT get_all_split_paths(), which includes TEST.
val_paths, val_names, lookup = get_split_paths("val")
train_paths, train_names, _ = get_split_paths("train", lookup)
if train_names & val_names:
    raise AssertionError("TRAIN and VAL patches overlap")
if SMOKE_MODE:
    val_paths = val_paths[:2]
print("Evaluating", len(val_paths), "official validation patches; TEST is NOT accessed")

cloud_idx = CLASSES.index("Clouds")
wave_idx = CLASSES.index("Waves")

def evaluate_variant(name):
    checkpoint_file = models_dir / f"{name}_best.pt"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing {checkpoint_file}; run 18_train_unet_v05.py first")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if list(checkpoint["bands"]) != list(BANDS) or list(checkpoint["classes"]) != list(CLASSES):
        raise ValueError("Checkpoint class/band names do not match project settings")
    model = UNet11(base=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = MaridaSegmentationDataset(val_paths, checkpoint["mean"],
                                        checkpoint["std"], augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    patch_rows = []
    with torch.inference_mode():
        for x, y, confidence, filenames in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                pred = model(x).argmax(1)
            confusion_from_batch(cm, pred, y)
            pred_np = pred.cpu().numpy()
            truth_np = y.numpy()
            for i, filename in enumerate(filenames):
                valid = truth_np[i] != IGNORE
                t, p = truth_np[i][valid], pred_np[i][valid]
                patch_cm = np.bincount(NUM_CLASSES * t + p,
                                       minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
                info = metrics_from_cm(patch_cm)
                patch_rows.append({
                    "patch": Path(filename).name, "full_path": filename,
                    "labeled_pixels": int(patch_cm.sum()),
                    "true_debris": int(patch_cm[0, :].sum()),
                    "tp": info["tp"], "fp": info["fp"], "fn": info["fn"],
                    "debris_f1": info["debris_f1"],
                    "clouds_as_debris_fp": int(patch_cm[cloud_idx, 0]),
                    "waves_as_debris_fp": int(patch_cm[wave_idx, 0]),
                    "debris_as_clouds_fn": int(patch_cm[0, cloud_idx]),
                    "debris_as_waves_fn": int(patch_cm[0, wave_idx]),
                })

    # Verify every patch count contributes to global confusion.
    table = pd.DataFrame(patch_rows)
    totals = table[["tp", "fp", "fn"]].sum().astype(int)
    global_metrics = metrics_from_cm(cm)
    for key in ("tp", "fp", "fn"):
        if int(totals[key]) != global_metrics[key]:
            raise AssertionError("Patch/global confusion mismatch for " + key)
    output = out / name
    output.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(output / "val_confusion.csv")
    table.to_csv(output / "val_patch_errors.csv", index=False)

    fp_classes = []
    for i, true_class in enumerate(CLASSES):
        if i == 0:
            continue
        support = int(cm[i, :].sum())
        fp = int(cm[i, 0])
        fp_classes.append({"true_class": true_class, "true_class_labeled_pixels": support,
                           "false_positive_as_debris": fp,
                           "fp_rate_within_class": fp / support if support else np.nan})
    pd.DataFrame(fp_classes).sort_values("false_positive_as_debris", ascending=False).to_csv(
        output / "val_false_positives_by_class.csv", index=False)

    summary = {
        "experiment": name, "best_epoch": int(checkpoint["best_epoch"]),
        "debris_f1": global_metrics["debris_f1"],
        "debris_iou": global_metrics["debris_iou"],
        "debris_precision": global_metrics["debris_precision"],
        "debris_recall": global_metrics["debris_recall"],
        "macro_f1_present": global_metrics["macro_f1_present"],
        "tp": global_metrics["tp"], "fp": global_metrics["fp"],
        "fn": global_metrics["fn"],
        "evaluated_pixels": global_metrics["evaluated_pixels"],
        "clouds_as_debris_fp": int(cm[cloud_idx, 0]),
        "waves_as_debris_fp": int(cm[wave_idx, 0]),
        "debris_as_clouds_fn": int(cm[0, cloud_idx]),
        "debris_as_waves_fn": int(cm[0, wave_idx]),
        "cloud_labeled_support": int(cm[cloud_idx, :].sum()),
        "wave_labeled_support": int(cm[wave_idx, :].sum()),
    }
    print(f"{name:25s} | F1={summary['debris_f1']:.4f} "
          f"FP={summary['fp']} FN={summary['fn']} "
          f"cloud->debris={summary['clouds_as_debris_fp']} "
          f"wave->debris={summary['waves_as_debris_fp']}")
    return summary, cm, table, checkpoint, model, dataset

all_summaries = []
confusions = {}
all_patch_errors = {}
for variant in VARIANTS:
    summary, confusion, patch_errors, checkpoint, model, dataset = evaluate_variant(variant)
    all_summaries.append(summary)
    confusions[variant] = confusion
    all_patch_errors[variant] = patch_errors
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

result_table = pd.DataFrame(all_summaries).sort_values("debris_f1", ascending=False)
if result_table.evaluated_pixels.nunique() != 1:
    raise AssertionError("Models were not evaluated on identical labeled-pixel counts")
result_table.to_csv(out / "validation_model_comparison.csv", index=False)
reference = result_table.loc[result_table.experiment == "v04_replay"].iloc[0]
result_table["delta_f1_vs_reference"] = result_table.debris_f1 - reference.debris_f1
result_table["delta_fp_vs_reference"] = result_table.fp - reference.fp
result_table.to_csv(out / "validation_model_comparison.csv", index=False)

# Cloud/wave confusion is reported on VALIDATION, not on previously inspected TEST.
error_rows = []
for row in all_summaries:
    for cname, support_key, fp_key, fn_key in [
        ("Clouds", "cloud_labeled_support", "clouds_as_debris_fp", "debris_as_clouds_fn"),
        ("Waves", "wave_labeled_support", "waves_as_debris_fp", "debris_as_waves_fn")]:
        support = row[support_key]
        error_rows.append({
            "experiment": row["experiment"], "class_name": cname,
            "labeled_support": support, "class_predicted_as_debris_fp": row[fp_key],
            "fp_rate_per_labeled_class_pixel": row[fp_key] / support if support else np.nan,
            "true_debris_predicted_as_class_fn": row[fn_key],
        })
pd.DataFrame(error_rows).to_csv(out / "cloud_wave_validation_errors.csv", index=False)

# Compare to ARCHIVED classical RF and v0.4 U-Net on the SAME official VAL split.
comparison = []
for row in all_summaries:
    comparison.append({"model": "U-Net v0.5: " + row["experiment"], "split": "val",
                       "marine_debris_f1": row["debris_f1"],
                       "marine_debris_precision": row["debris_precision"],
                       "marine_debris_recall": row["debris_recall"],
                       "evaluated_pixels": row["evaluated_pixels"],
                       "provenance": "v0.5, new controlled training"})
rf_file = PROJECT_DIR / "results" / "benchmark" / "validation_comparison.csv"
if rf_file.is_file():
    rf_table = pd.read_csv(rf_file)
    for _, row in rf_table.iterrows():
        comparison.append({"model": "Archived RF: " + str(row["model"]), "split": "val",
                           "marine_debris_f1": float(row["marine_debris_f1"]),
                           "marine_debris_precision": float(row["marine_debris_precision"]),
                           "marine_debris_recall": float(row["marine_debris_recall"]),
                           "evaluated_pixels": int(row["evaluated_pixels"]),
                           "provenance": "archived RF, independently trained"})
v04_file = PROJECT_DIR / "results" / "unet" / "unet_metrics.csv"
if v04_file.is_file():
    previous = pd.read_csv(v04_file)
    for _, row in previous.loc[previous.split == "val"].iterrows():
        comparison.append({"model": "Archived U-Net v0.4", "split": "val",
                           "marine_debris_f1": float(row["debris_f1"]),
                           "marine_debris_precision": float(row["debris_precision"]),
                           "marine_debris_recall": float(row["debris_recall"]),
                           "evaluated_pixels": int(row["evaluated_pixels"]),
                           "provenance": "archived v0.4, separate checkpoint"})
comparison_table = pd.DataFrame(comparison)
comparison_table["same_evaluated_pixel_count"] = (
    comparison_table.evaluated_pixels == int(result_table.evaluated_pixels.iloc[0]))
comparison_table.to_csv(out / "validation_comparison_with_archives.csv", index=False)

# Charts: performance and cloud/wave false-positive counts.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
order = result_table.sort_values("debris_f1")
axes[0].barh(order.experiment, order.debris_f1)
axes[0].set_xlabel("Marine Debris F1 on official VAL")
axes[0].set_xlim(0, 1)
axes[0].set_title("Controlled U-Net ablations")
axes[0].grid(axis="x", alpha=.2)
order2 = result_table.set_index("experiment").loc[VARIANTS]
indices = np.arange(len(order2))
axes[1].bar(indices - .18, order2.clouds_as_debris_fp, width=.35, label="Clouds -> debris")
axes[1].bar(indices + .18, order2.waves_as_debris_fp, width=.35, label="Waves -> debris")
axes[1].set_xticks(indices, ["v04 replay", "no class weight", "no patch bias", "extra jitter"],
                   rotation=20, ha="right")
axes[1].set_ylabel("False-positive labeled pixels")
axes[1].set_title("Validation false-positive sources")
axes[1].legend()
axes[1].grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(fig_dir / "ablation_f1_cloud_wave_fp.png", dpi=170)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.7))
for name in VARIANTS:
    history_file = out / name / "training_history.csv"
    frame = pd.read_csv(history_file)
    ax.plot(frame.epoch, frame.val_debris_f1, label=name)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Marine Debris F1")
ax.set_title("Learning curves: one change per training run")
ax.set_ylim(0, 1)
ax.grid(alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(fig_dir / "validation_f1_learning_curves.png", dpi=170)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)

best_name = result_table.iloc[0].experiment
best_cm = confusions[best_name]
normalized = best_cm / np.maximum(best_cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(12.5, 10))
image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=65, ha="right", fontsize=8)
ax.set_yticks(range(NUM_CLASSES), CLASSES, fontsize=8)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title(f"Official VAL confusion matrix (row-normalized) | {best_name}")
fig.colorbar(image, ax=ax, fraction=.04, pad=.04)
fig.tight_layout()
fig.savefig(fig_dir / "best_validation_confusion_normalized.png", dpi=170)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)

# Display both cases: selected model may equal the reference.
other_name = (best_name if best_name != "v04_replay"
              else result_table.loc[result_table.experiment != "v04_replay"].iloc[0].experiment)

# Pick validation patches using ONLY validation error counts, including no-debris patches.
source = all_patch_errors[best_name][["patch", "full_path", "fp", "fn",
                                      "clouds_as_debris_fp", "waves_as_debris_fp"]].copy()
ref = all_patch_errors["v04_replay"][["patch", "fp"]].rename(columns={"fp": "ref_fp"})
source = source.merge(ref, on="patch", validate="one_to_one")
source["cloud_wave_fp"] = source.clouds_as_debris_fp + source.waves_as_debris_fp
candidates = source.sort_values(["cloud_wave_fp", "fp", "ref_fp", "fn"],
                                ascending=False).head(MAX_VISUALIZED_PATCHES)

def predict_one(name, path):
    checkpoint = torch.load(models_dir / f"{name}_best.pt", map_location="cpu",
                            weights_only=True)
    model = UNet11(base=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = MaridaSegmentationDataset([path], checkpoint["mean"], checkpoint["std"],
                                        augment=False)
    x, target, confidence, _ = dataset[0]
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            predicted = model(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
    return predicted, target.numpy()

for _, row in candidates.iterrows():
    patch_path = Path(row["full_path"])
    ref_pred, target = predict_one("v04_replay", patch_path)
    other_pred, other_target = predict_one(other_name, patch_path)
    if not np.array_equal(target, other_target):
        raise AssertionError("Validation labels changed across two model variants")
    valid = target != IGNORE
    rgb = make_rgb(load_raw_image(patch_path))
    binary_cmap = ListedColormap(["#c8cdd2", "#bad1df", "#e66a57"])
    err_cmap = ListedColormap(["#c8cdd2", "#bad1df", "#39a56b", "#dc3d3d", "#f0aa37"])
    gt_display = np.where(valid, np.where(target == 0, 2, 1), 0)

    def binary_prediction(pred):
        return np.where(valid, np.where(pred == 0, 2, 1), 0)

    def error_map(pred):
        error = np.where(valid, 1, 0)
        error[valid & (target == 0) & (pred == 0)] = 2
        error[valid & (target != 0) & (pred == 0)] = 3
        error[valid & (target == 0) & (pred != 0)] = 4
        return error

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 RGB")
    axes[0, 1].imshow(gt_display, cmap=binary_cmap, vmin=0, vmax=2)
    axes[0, 1].set_title("Ground Truth (labeled only)")
    axes[0, 2].imshow(binary_prediction(ref_pred), cmap=binary_cmap, vmin=0, vmax=2)
    axes[0, 2].set_title("v04_replay: predicted debris")
    axes[1, 0].imshow(binary_prediction(other_pred), cmap=binary_cmap, vmin=0, vmax=2)
    axes[1, 0].set_title(f"{other_name}: predicted debris")
    axes[1, 1].imshow(error_map(ref_pred), cmap=err_cmap, vmin=0, vmax=4)
    axes[1, 1].set_title("Reference errors (red FP / orange FN)")
    axes[1, 2].imshow(error_map(other_pred), cmap=err_cmap, vmin=0, vmax=4)
    axes[1, 2].set_title("Comparison errors (red FP / orange FN)")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Official VAL, labeled pixels only | {patch_path.name}")
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(fig_dir / f"val_{patch_path.stem}_v05_comparison.png", dpi=170)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

metadata = {"split": "official validation only", "test_read": False,
            "best_by_validation_f1": best_name, "best_f1": float(result_table.iloc[0].debris_f1),
            "reference": "v04_replay", "comparison_visualized_variant": other_name,
            "experiment_count": len(VARIANTS), "smoke_mode": SMOKE_MODE,
            "interpretation": "Exploratory validation comparisons, not independent test evidence",
            "note": "Archived v0.4 and RF scores come from separate runs. "
                    "Do not tune on previously inspected test outcomes."}
(out / "analysis_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# Auto-generated reproducible short report from observed values only.
report = ["# v0.5 — Controlled U-Net Training Ablations (Validation Only)", "",
          "This report contains real results only after the local scripts run on MARIDA.",
          "No official TEST data were read by scripts 18 or 19.", "",
          "## Validation models", "",
          "| Experiment | Debris F1 | Precision | Recall | IoU | TP | FP | FN | Clouds→Debris FP | Waves→Debris FP |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for row in result_table.itertuples():
    report.append(f"| {row.experiment} | {row.debris_f1:.4f} | {row.debris_precision:.4f} "
                  f"| {row.debris_recall:.4f} | {row.debris_iou:.4f} | {row.tp} | {row.fp} | {row.fn} "
                  f"| {row.clouds_as_debris_fp} | {row.waves_as_debris_fp} |")
report += ["", f"Best *validation* experiment: **{best_name}**; this is not a fresh test-set result.",
           "", "### Design", "", "- A fixed 11-band / 15-class U-Net and fixed seed.",
           "- The reference replays the original v0.4 class weights, debris patch sampling, and geometric augmentation.",
           "- Each ablation changes only one of those components.",
           "- All model selection is on official VAL, and all scoring ignores unlabeled pixels.",
           "- `cloud_wave_validation_errors.csv` includes class supports and per-class FP rates.",
           "", "### Limitations", "", "- One seed; comparisons are exploratory.",
           "- Official TEST had already been examined in earlier v0.4 work, and is intentionally untouched here.",
           "- Scene composition and sparse labels limit claims about geographic generalization.",
           "- A lower cloud/wave false-positive count must be interpreted alongside recall and total F1.", ""]
(out / "validation_findings.md").write_text("\n".join(report), encoding="utf-8")
print("\nBest validation run:", best_name, "F1:", result_table.iloc[0].debris_f1)
print("Outputs:", out)
print("Figures:", fig_dir)
print("Official TEST was NOT accessed. Do not select configurations based on v0.4 TEST failures.")
