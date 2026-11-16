"""Plot validation-patch predictions and analyze Marine Debris errors.

Run directly in PyCharm after training the Random Forest models.
Only annotated pixels are evaluated; class 0 is unlabeled, not water.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import tifffile

from settings import DATA_FILE, MODELS_DIR, PROJECT_DIR, RESULTS_DIR, BANDS, CLASSES, RANDOM_SEED
from spectral_features import make_features

# Edit these three values if you want a larger or smaller inspection.
MAX_VALIDATION_PATCHES = 60
NUMBER_OF_FIGURES = 5
SHOW_FIGURES_IN_PYCHARM = True

DATA_DIR = DATA_FILE.parent
PATCHES_DIR = DATA_DIR / "patches"
SPLIT_FILE = DATA_DIR / "splits" / "val_X.txt"
TABLES_DIR = PROJECT_DIR / "results" / "visual_analysis"
FIGURES_DIR = PROJECT_DIR / "assets" / "figures" / "visual_errors"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Load the model chosen during validation in step 03.
selection_file = RESULTS_DIR / "selected_model.txt"
if not selection_file.is_file():
    selection_file = PROJECT_DIR / "results" / "benchmark" / "selected_model.txt"
if not selection_file.is_file():
    raise FileNotFoundError("Missing selected_model.txt. Run 03_evaluate_models.py first.")

model_name = selection_file.read_text(encoding="utf-8").strip()
model_file = MODELS_DIR / (model_name + ".joblib")
if not model_file.is_file():
    raise FileNotFoundError("Missing trained model: " + str(model_file) + ". Run 02_train_random_forest.py.")
saved_model = joblib.load(model_file)
model = saved_model["model"]
features = saved_model["features"]
medians = saved_model["medians"]

if not SPLIT_FILE.is_file():
    raise FileNotFoundError("Official validation list is missing: " + str(SPLIT_FILE))
if not PATCHES_DIR.is_dir():
    raise FileNotFoundError("Original MARIDA patches are missing: " + str(PATCHES_DIR))

# Build a lookup for raw images. Skip annotation masks, which are not inputs.
image_files = {}
for path in PATCHES_DIR.rglob("*.tif"):
    if not path.name.endswith(("_cl.tif", "_conf.tif")):
        image_files[path.name.lower()] = path

# Each line of val_X.txt identifies a validation patch (with or without .tif).
validation_names = []
for line in SPLIT_FILE.read_text(encoding="utf-8-sig").splitlines():
    clean_line = line.strip()
    if not clean_line or clean_line.startswith("#"):
        continue
    name = Path(clean_line.strip('"\'').replace("\\", "/")).name.lower()
    if not name.endswith(".tif"):
        name += ".tif"
    # MARIDA split lists omit the S2_ prefix used by extracted GeoTIFFs.
    if not name.startswith("s2_"):
        name = "s2_" + name
    validation_names.append(name)

validation_names = sorted(set(validation_names))
if not validation_names:
    raise ValueError("val_X.txt did not contain any image names.")
missing = [name for name in validation_names if name not in image_files]
if missing:
    raise FileNotFoundError(
        "Validation image names do not match the extracted MARIDA patches. "
        "Example from val_X.txt: " + missing[0] + ". Example extracted image: "
        + next(iter(image_files), "(none)")
    )

rng = np.random.default_rng(RANDOM_SEED)
if len(validation_names) > MAX_VALIDATION_PATCHES:
    chosen_ids = rng.choice(len(validation_names), size=MAX_VALIDATION_PATCHES, replace=False)
    validation_names = [validation_names[i] for i in sorted(chosen_ids)]


# For each selected patch: read the image, predict ONLY on labeled pixels,
# then keep a small binary (Marine Debris vs other) error map.
rows = []
error_pixels = []
confusions = {name: 0 for name in CLASSES if name != "Marine Debris"}
plots_to_save = []

def load_patch(image_path):
    image = np.asarray(tifffile.imread(image_path), dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D multispectral image: {image_path} -> {image.shape}")
    if image.shape[0] == len(BANDS):
        image = np.moveaxis(image, 0, -1)
    elif image.shape[-1] != len(BANDS):
        raise ValueError(f"Expected 11 bands in {image_path}, found {image.shape}")

    mask = np.squeeze(tifffile.imread(image_path.with_name(image_path.stem + "_cl.tif")))
    confidence = np.squeeze(tifffile.imread(image_path.with_name(image_path.stem + "_conf.tif")))
    if mask.shape != image.shape[:2] or confidence.shape != mask.shape:
        raise ValueError("Image, class mask and confidence mask dimensions differ: " + image_path.name)
    return image, mask.astype(int), confidence.astype(int)


def make_rgb(image):
    # Sentinel-2: Red=665nm, Green=560nm, Blue=490nm.
    rgb = image[:, :, [3, 2, 1]].astype(float).copy()
    for channel in range(3):
        part = rgb[:, :, channel]
        finite = part[np.isfinite(part)]
        if len(finite):
            low, high = np.percentile(finite, [2, 98])
            rgb[:, :, channel] = np.clip((part - low) / max(high - low, 1e-8), 0, 1)
    return np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)


def plot_patch(image, truth, predicted, filename, stats):
    labeled = (truth >= 1) & (truth <= len(CLASSES))
    true_debris = truth == 1
    pred_debris = predicted == 1
    tp = labeled & true_debris & pred_debris
    fp = labeled & ~true_debris & pred_debris
    fn = labeled & true_debris & ~pred_debris

    # 0 unlabeled, 1 annotated non-debris, 2 debris.
    truth_display = np.where(labeled, np.where(true_debris, 2, 1), 0)
    pred_display = np.where(labeled, np.where(pred_debris, 2, 1), 0)
    binary_colors = ListedColormap(["#c8cdd2", "#b6d1df", "#db544a"])

    # 0 unlabeled, 1 correct non-debris, 2 TP, 3 FP, 4 FN.
    errors = np.where(labeled, 1, 0).astype(int)
    errors[tp] = 2
    errors[fp] = 3
    errors[fn] = 4
    error_colors = ListedColormap(["#c8cdd2", "#b6d1df", "#53a66f", "#dc4545", "#f4b343"])

    rgb = make_rgb(image)
    overlay = np.zeros((*truth.shape, 4), dtype=float)
    overlay[tp] = [0.1, 0.7, 0.25, 0.70]
    overlay[fp] = [0.95, 0.10, 0.10, 0.85]
    overlay[fn] = [1.0, 0.65, 0.0, 0.85]

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[1].imshow(truth_display, cmap=binary_colors, vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title("Ground truth (labeled only)")
    axes[2].imshow(pred_display, cmap=binary_colors, vmin=0, vmax=2, interpolation="nearest")
    axes[2].set_title("RF prediction (labeled only)")
    axes[3].imshow(errors, cmap=error_colors, vmin=0, vmax=4, interpolation="nearest")
    axes[3].set_title("Marine Debris errors")
    axes[4].imshow(rgb)
    axes[4].imshow(overlay)
    axes[4].set_title("Errors on RGB")
    for ax in axes:
        ax.axis("off")
    handles = [
        Patch(color="#53a66f", label="TP: debris found"),
        Patch(color="#dc4545", label="FP: false alarm"),
        Patch(color="#f4b343", label="FN: missed debris"),
        Patch(color="#c8cdd2", label="Unlabeled (not evaluated)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"Validation | {filename}  |  TP={stats['tp']}  FP={stats['fp']}  FN={stats['fn']}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    saved_path = FIGURES_DIR / (Path(filename).stem + "_visual_errors.png")
    fig.savefig(saved_path, dpi=150, bbox_inches="tight")
    if SHOW_FIGURES_IN_PYCHARM:
        plt.show()
    plt.close(fig)
    print("Saved figure:", saved_path)


print("Model:", model_name)
print("Validation patches to inspect:", len(validation_names))
for index, patch_name in enumerate(validation_names, 1):
    image_path = image_files[patch_name]
    image, truth, confidence = load_patch(image_path)
    annotated = (truth >= 1) & (truth <= len(CLASSES))
    if not annotated.any():
        continue

    pixel_positions = np.flatnonzero(annotated)
    selected_spectra = image.reshape(-1, len(BANDS))[pixel_positions]
    if index == 1 and np.nanmedian(np.abs(selected_spectra)) > 3:
        print("WARNING: check GeoTIFF reflectance units against the training HDF5 values.")
    data = pd.DataFrame(selected_spectra, columns=BANDS)
    pixel_features = make_features(data)[features]
    pixel_features = pixel_features.fillna(medians).fillna(0)
    predicted_classes = model.predict(pixel_features)
    predicted_ids = pd.Series(predicted_classes).map({name: i + 1 for i, name in enumerate(CLASSES)})
    if predicted_ids.isna().any():
        raise ValueError("Model predicted an unknown class label.")

    predicted = np.zeros(truth.shape, dtype=np.int16)
    predicted.flat[pixel_positions] = predicted_ids.to_numpy(dtype=np.int16)
    true_debris = (truth == 1) & annotated
    pred_debris = (predicted == 1) & annotated
    tp_map = true_debris & pred_debris
    fp_map = (~true_debris) & pred_debris & annotated
    fn_map = true_debris & (~pred_debris)

    tp, fp, fn = int(tp_map.sum()), int(fp_map.sum()), int(fn_map.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    row = {"patch": image_path.name, "split": "val", "annotated_pixels": int(annotated.sum()),
           "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    rows.append(row)

    # Save the row/column location and true class of every false alarm or miss.
    for error_kind, mask in [("FP", fp_map), ("FN", fn_map)]:
        y_coords, x_coords = np.where(mask)
        for r, c in zip(y_coords, x_coords):
            true_name = CLASSES[truth[r, c] - 1]
            prediction_name = CLASSES[predicted[r, c] - 1]
            error_pixels.append({"patch": image_path.name, "split": "val",
                                 "row": int(r), "col": int(c), "error": error_kind,
                                 "true_class": true_name, "predicted_class": prediction_name,
                                 "confidence": int(confidence[r, c])})
            if error_kind == "FP":
                confusions[true_name] += 1

    # Store only a few patches for display at the end.
    plots_to_save.append((fp + fn, image_path.name, image, truth, predicted, row))
    print(f"{index:3d}  {image_path.name}: annotated={int(annotated.sum())}, TP={tp}, FP={fp}, FN={fn}")

if not rows:
    raise ValueError("No annotated validation pixels were found in the inspected patches.")

summary = pd.DataFrame(rows)
summary.to_csv(TABLES_DIR / "patch_error_summary.csv", index=False)
pd.DataFrame(error_pixels, columns=["patch", "split", "row", "col", "error",
                                    "true_class", "predicted_class", "confidence"]).to_csv(
    TABLES_DIR / "error_pixel_locations.csv", index=False)
pd.DataFrame([{"true_class": name, "false_positive_pixels": count}
              for name, count in confusions.items()]).sort_values(
    "false_positive_pixels", ascending=False).to_csv(
    TABLES_DIR / "false_positives_by_class.csv", index=False)

total_tp, total_fp, total_fn = int(summary.tp.sum()), int(summary.fp.sum()), int(summary.fn.sum())
precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
pd.DataFrame([{"split": "val", "model": model_name, "inspected_patches": len(rows),
               "annotated_pixels": int(summary.annotated_pixels.sum()),
               "tp": total_tp, "fp": total_fp, "fn": total_fn,
               "precision": precision, "recall": recall, "f1": f1}]).to_csv(
    TABLES_DIR / "validation_subset_summary.csv", index=False)

counts = pd.Series(confusions).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(counts.index[::-1], counts.values[::-1])
ax.set_xlabel("Annotated validation pixels incorrectly predicted as Marine Debris")
ax.set_title("Visual error analysis | False alarms by true class")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "false_positives_by_class.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

# Display the most informative patches (highest FP+FN), not randomly chosen success cases.
plots_to_save.sort(key=lambda item: item[0], reverse=True)
for _, name, image, truth, predicted, row in plots_to_save[:NUMBER_OF_FIGURES]:
    plot_patch(image, truth, predicted, name, row)

print("\nValidation subset (not full validation or test):")
print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}; Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
print("Tables:", TABLES_DIR)
print("Figures:", FIGURES_DIR)
