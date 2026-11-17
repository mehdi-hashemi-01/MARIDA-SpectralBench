"""v0.4 Step 17: evaluate the saved best U-Net on official VAL and TEST.

Run AFTER 16_train_unet.py. Never use TEST metrics to change hyperparameters.
Writes labeled-pixel confusion matrices, plots, image-level errors, and comparison
with the previous archived RF benchmark (same official splits, separate runs).
"""
import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from settings import PROJECT_DIR, BANDS, CLASSES
from unet_core import (UNet11, MaridaSegmentationDataset, get_all_split_paths,
                       load_raw_image, load_label, make_rgb, confusion_from_batch,
                       metrics_from_cm, NUM_CLASSES, IGNORE)

BATCH_SIZE = 4
NUM_VISUALIZATIONS_PER_SPLIT = 5
EVALUATE_TEST = True     # Use only after your training settings are finalized.
SHOW_FIGURES = False

OUT = PROJECT_DIR / "results" / "unet"
FIG = PROJECT_DIR / "assets" / "figures" / "unet"
MODEL_FILE = PROJECT_DIR / "models" / "unet_best.pt"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
if not MODEL_FILE.is_file():
    raise FileNotFoundError("Missing saved U-Net. Run 16_train_unet.py first: " + str(MODEL_FILE))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Check PyTorch RTX 5070 setup with script 15 first.")
device = torch.device("cuda")
ckpt = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
if list(ckpt["bands"]) != list(BANDS) or list(ckpt["classes"]) != list(CLASSES):
    raise ValueError("Checkpoint bands/classes differ from project settings.")
model = UNet11(base=int(ckpt["base_channels"])).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
means, stds = ckpt["mean"], ckpt["std"]
splits = get_all_split_paths()
print("Selected epoch:", ckpt["best_epoch"], "| Validation F1 when saved:", ckpt["val_debris_f1"])
print("Official split sizes:", {k: len(v) for k, v in splits.items()})


def draw_confusion(cm, split):
    names = CLASSES
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    heat = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), names, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"U-Net 11-band — {split} | row-normalized confusion (labeled pixels)")
    fig.colorbar(heat, ax=ax, fraction=.045, pad=.04)
    fig.tight_layout()
    fig.savefig(FIG / f"{split}_confusion_normalized.png", dpi=170)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 10))
    heat = ax.imshow(np.log1p(cm), cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), names, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"U-Net 11-band — {split} | log(1 + class counts)")
    fig.colorbar(heat, ax=ax, fraction=.045, pad=.04, label="log(1 + pixels)")
    fig.tight_layout()
    fig.savefig(FIG / f"{split}_confusion_logcounts.png", dpi=170)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def draw_patch(path, truth, predicted, split, score):
    """All scores/colored predictions are restricted to labeled pixels."""
    labeled = truth != 0
    debris = truth == 1
    pred_debris = predicted == 1
    tp = labeled & debris & pred_debris
    fp = labeled & ~debris & pred_debris
    fn = labeled & debris & ~pred_debris
    true_binary = np.where(labeled, np.where(debris, 2, 1), 0)
    pred_binary = np.where(labeled, np.where(pred_debris, 2, 1), 0)
    error = np.where(labeled, 1, 0)
    error[tp], error[fp], error[fn] = 2, 3, 4
    colors = ListedColormap(["#c8cdd2", "#acc8d8", "#d75645"])
    errcolors = ListedColormap(["#c8cdd2", "#b0cee0", "#3aa66d", "#dc3d3d", "#f0a234"])
    rgb = make_rgb(load_raw_image(path))
    overlay = np.zeros((*truth.shape, 4), dtype=np.float32)
    overlay[tp] = [0.1, .8, .2, .7]
    overlay[fp] = [.95, .1, .1, .85]
    overlay[fn] = [1.0, .65, 0, .85]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[1].imshow(true_binary, cmap=colors, vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title("Truth (annotated only)")
    axes[2].imshow(pred_binary, cmap=colors, vmin=0, vmax=2, interpolation="nearest")
    axes[2].set_title("U-Net (annotated only)")
    axes[3].imshow(error, cmap=errcolors, vmin=0, vmax=4, interpolation="nearest")
    axes[3].set_title("Debris errors")
    axes[4].imshow(rgb)
    axes[4].imshow(overlay)
    axes[4].set_title("RGB + errors")
    for ax in axes:
        ax.axis("off")
    fig.legend(handles=[Patch(color="#3aa66d", label="TP"),
                        Patch(color="#dc3d3d", label="FP"),
                        Patch(color="#f0a234", label="FN"),
                        Patch(color="#c8cdd2", label="Unlabeled")],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"{split.upper()} | {path.name} | TP={score['tp']} FP={score['fp']} FN={score['fn']}")
    fig.tight_layout(rect=[0, .10, 1, .94])
    fig.savefig(FIG / f"{split}_{path.stem}_comparison.png", dpi=160)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    # Additional full 15-class semantic map. Class 0 stays unscored/unlabeled.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    cmap = ListedColormap(["#bfc3ca"] + [plt.get_cmap("tab20")(i) for i in range(15)])
    truth_display = np.where(labeled, truth, 0)
    pred_display = np.where(labeled, predicted, 0)
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[1].imshow(truth_display, cmap=cmap, vmin=0, vmax=15, interpolation="nearest")
    axes[1].set_title("15-class ground truth (labeled only)")
    im = axes[2].imshow(pred_display, cmap=cmap, vmin=0, vmax=15, interpolation="nearest")
    axes[2].set_title("15-class U-Net (labeled only)")
    for ax in axes:
        ax.axis("off")
    bar = fig.colorbar(im, ax=axes[1:], fraction=.025, pad=.02)
    bar.set_ticks([0, 1, 5, 7, 8, 15])
    bar.set_ticklabels(["Unlabeled", "Debris", "Ship", "Marine Water", "Sediment", "Mixed Water"])
    fig.suptitle(f"{split.upper()} | {path.name} | 15-class comparison")
    fig.subplots_adjust(top=.86, bottom=.09, right=.94)
    fig.savefig(FIG / f"{split}_{path.stem}_multiclass.png", dpi=160, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


all_metrics = []
for split in ("val", "test"):
    if split == "test" and not EVALUATE_TEST:
        continue
    print("\nEvaluating", split.upper(), "on", len(splits[split]), "official patches")
    ds = MaridaSegmentationDataset(splits[split], means, stds, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    patch_records = []
    with torch.inference_mode():
        for batch_no, (x, target, confidence, filenames) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(x).argmax(1)
            # Global confusion is constructed solely on labeled pixels.
            confusion_from_batch(cm, prediction, target)
            predicted_np = prediction.cpu().numpy()
            target_np = target.numpy()
            for j, filename in enumerate(filenames):
                local = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                truth_flat = target_np[j].reshape(-1)
                pred_flat = predicted_np[j].reshape(-1)
                valid = truth_flat != IGNORE
                if valid.any():
                    local += np.bincount(NUM_CLASSES * truth_flat[valid] + pred_flat[valid],
                                         minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
                score = metrics_from_cm(local)
                patch_records.append({"patch": str(filename), "split": split,
                                      "true_debris": int(local[0, :].sum()), **score})
            if batch_no % 40 == 0:
                print(f"    {batch_no} batches processed")

    result = metrics_from_cm(cm)
    result.update(model="unet_11band", split=split, best_epoch=int(ckpt["best_epoch"]))
    all_metrics.append(result)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(OUT / f"{split}_confusion.csv")
    draw_confusion(cm, split)

    per_class = []
    for i, name in enumerate(CLASSES):
        support = int(cm[i, :].sum())
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        per_class.append(dict(class_name=name, support=support, tp=tp, fp=fp, fn=fn,
                              precision=tp / (tp + fp) if tp + fp else 0.0,
                              recall=tp / (tp + fn) if tp + fn else 0.0,
                              f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan))
    pd.DataFrame(per_class).to_csv(OUT / f"{split}_per_class.csv", index=False)
    fp_sources = [{"true_class": CLASSES[i], "false_positive_debris": int(cm[i, 0])}
                  for i in range(1, NUM_CLASSES)]
    pd.DataFrame(fp_sources).sort_values("false_positive_debris", ascending=False).to_csv(
        OUT / f"{split}_false_positives_by_class.csv", index=False)

    frame = pd.DataFrame(patch_records)
    frame.to_csv(OUT / f"{split}_patch_errors.csv", index=False)
    # First prefer the patches with most labeled debris errors; also include patch
    # with the most true debris, even if it was not among the hard cases.
    candidates = frame.loc[frame.true_debris > 0].sort_values(["fp", "fn"], ascending=False)
    chosen = candidates.head(NUM_VISUALIZATIONS_PER_SPLIT)["patch"].tolist()
    if not candidates.empty:
        representative = candidates.sort_values("true_debris", ascending=False).iloc[0]["patch"]
        if representative not in chosen:
            chosen = chosen[:max(0, NUM_VISUALIZATIONS_PER_SPLIT - 1)] + [representative]
    for filename in chosen:
        path = next(p for p in splits[split] if str(p) == filename)
        dataset_index = splits[split].index(path)
        # Inference for a few selected figures; same checkpoint, no training.
        x, y, conf, _ = ds[dataset_index]
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy() + 1
        truth = load_label(path)
        selected_score = frame.loc[frame.patch == filename].iloc[0].to_dict()
        draw_patch(path, truth, pred, split, selected_score)

    print(f"{split.upper()} => F1={result['debris_f1']:.4f} "
          f"P={result['debris_precision']:.4f} R={result['debris_recall']:.4f} "
          f"IoU={result['debris_iou']:.4f} TP={result['tp']} FP={result['fp']} FN={result['fn']}")

# Record the actual U-Net results, never invent a new RF run.
pd.DataFrame(all_metrics).to_csv(OUT / "unet_metrics.csv", index=False)
comparison = []
bench_dir = PROJECT_DIR / "results" / "benchmark"
for filename in ["validation_comparison.csv", "test_result.csv"]:
    filepath = bench_dir / filename
    if not filepath.is_file():
        print("Archived RF benchmark not found:", filepath)
        continue
    rf = pd.read_csv(filepath)
    for _, row in rf.iterrows():
        comparison.append(dict(split=row["split"], model="Random Forest: " + row["model"],
                               input_features="11 bands" if row["model"] == "spectral_only" else "11 bands + 8 indices",
                               debris_f1=float(row["marine_debris_f1"]),
                               debris_iou=float(row["marine_debris_iou"]),
                               debris_precision=float(row["marine_debris_precision"]),
                               debris_recall=float(row["marine_debris_recall"]),
                               evaluated_pixels=int(row["evaluated_pixels"]),
                               source="archived v0.1 RF, independent training run"))
for row in all_metrics:
    comparison.append(dict(split=row["split"], model="U-Net 11-band",
                           input_features="11 bands + spatial neighborhood",
                           debris_f1=row["debris_f1"], debris_iou=row["debris_iou"],
                           debris_precision=row["debris_precision"],
                           debris_recall=row["debris_recall"],
                           evaluated_pixels=row["evaluated_pixels"],
                           source="v0.4 U-Net best-epoch checkpoint, same official split"))
result_table = pd.DataFrame(comparison)
result_table.to_csv(OUT / "model_comparison.csv", index=False)
if len(result_table):
    for split in result_table.split.unique():
        subset = result_table[result_table.split == split]
        if subset.evaluated_pixels.nunique() != 1:
            print("WARNING: Evaluated pixel counts differ for", split,
                  "- comparison may not be perfectly like-for-like!")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(subset.model, subset.debris_f1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Marine Debris F1 on labeled pixels")
        ax.set_title(f"MARIDA {split}: U-Net vs archived RF (separate runs)")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(FIG / f"{split}_unet_vs_rf_f1.png", dpi=160)
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)

(OUT / "evaluation_protocol.json").write_text(json.dumps({
    "checkpoint_selected_by": "validation Marine Debris F1",
    "checkpoint_epoch": int(ckpt["best_epoch"]),
    "test_touched_during_training": False,
    "test_evaluated_now": EVALUATE_TEST,
    "scored_pixels": "annotated class IDs 1..15 only; 0 ignored",
    "rf_provenance": "previous archived benchmark; not retrained alongside this U-Net",
    "caveat": "Do not tune or choose a new model using held-out test outcomes; "
              "original benchmark test results had already been inspected.",
}, indent=2), encoding="utf-8")
print("\nSaved results:", OUT)
print("Saved figures:", FIG)
print("NOTICE: Test results must not be used to tune training or select new models.")
