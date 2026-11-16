"""v0.3: compare six ablation models on ONE known validation GeoTIFF.

Run AFTER 09_feature_ablation.py.
This is a diagnostic for the known sediment error, NOT a test set evaluation.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile

from settings import DATA_FILE, PROJECT_DIR, MODELS_DIR, BANDS, CLASSES
from spectral_features import make_features

# The validation image with 112 FP in the previously inspected baseline.
PATCH_NAME = "S2_11-6-18_16PCC_26.tif"
SHOW_FIGURE_IN_PYCHARM = True

PATCHES_DIR = DATA_FILE.parent / "patches"
MODEL_DIR = MODELS_DIR / "ablation"
RESULT_DIR = PROJECT_DIR / "results" / "ablation"
FIGURE_DIR = PROJECT_DIR / "assets" / "figures" / "ablation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

names = ["full_19", "minus_ndmi", "minus_ndwi", "minus_fdi", "minus_three", "bands_only"]

# Look up the patch in the original MARIDA directory structure.
images = [p for p in PATCHES_DIR.rglob("*.tif") if p.name.lower() == PATCH_NAME.lower()]
if not images:
    raise FileNotFoundError("Cannot find patch: " + PATCH_NAME + " under " + str(PATCHES_DIR))
image_path = images[0]

# Read the multispectral image and its annotated class mask.
image = np.asarray(tifffile.imread(image_path), dtype=np.float32)
if image.ndim != 3:
    raise ValueError("Unexpected image shape: " + str(image.shape))
if image.shape[0] == len(BANDS):
    image = np.moveaxis(image, 0, -1)
elif image.shape[-1] != len(BANDS):
    raise ValueError("Expected 11 bands: " + str(image.shape))

mask_file = image_path.with_name(image_path.stem + "_cl.tif")
truth_mask = np.asarray(tifffile.imread(mask_file)).squeeze()
if truth_mask.shape != image.shape[:2]:
    raise ValueError("Image and annotation mask have different shapes.")

# Class 0 is UNLABELED and must never be counted as a negative.
valid_pixels = (truth_mask >= 1) & (truth_mask <= len(CLASSES))
if not valid_pixels.any():
    raise ValueError("No labeled pixels in this patch.")

spectra = image[valid_pixels].reshape(-1, len(BANDS))
features_table = make_features(pd.DataFrame(spectra, columns=BANDS))
true_ids = truth_mask[valid_pixels].astype(int)
true_debris = true_ids == 1

rows = []
for name in names:
    model_file = MODEL_DIR / (name + ".joblib")
    if not model_file.is_file():
        raise FileNotFoundError("Run step 09 first. Model is missing: " + str(model_file))

    saved = joblib.load(model_file)
    inputs = features_table[saved["features"]].fillna(saved["medians"]).fillna(0)
    predicted_names = saved["model"].predict(inputs)
    predicted_ids = pd.Series(predicted_names).map({item: i + 1 for i, item in enumerate(CLASSES)})
    if predicted_ids.isna().any():
        raise ValueError("Unknown class name in model predictions.")
    predicted_debris = predicted_ids.to_numpy(dtype=int) == 1

    tp = int(np.sum(true_debris & predicted_debris))
    fp = int(np.sum(~true_debris & predicted_debris))
    fn = int(np.sum(true_debris & ~predicted_debris))
    sediment_fp = int(np.sum((true_ids == CLASSES.index("Sediment-Laden Water") + 1) & predicted_debris))
    rows.append({
        "experiment": name,
        "patch": PATCH_NAME,
        "split": "val",
        "annotated_pixels": int(len(true_ids)),
        "tp": tp, "fp": fp, "fn": fn,
        "sediment_laden_water_fp": sediment_fp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    })

summary = pd.DataFrame(rows)
summary.to_csv(RESULT_DIR / "known_validation_patch_comparison.csv", index=False)
print("\nKNOWN VALIDATION PATCH:", PATCH_NAME)
print(summary[["experiment", "tp", "fp", "fn", "sediment_laden_water_fp"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.barh(summary["experiment"].str.replace("_", " "), summary["fp"])
ax.invert_yaxis()
ax.set_xlabel("False-positive Marine Debris predictions (labeled pixels only)")
ax.set_title("Known sediment-rich validation patch | Ablation diagnostic")
ax.grid(axis="x", alpha=0.15)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "known_validation_patch_false_positives.png", dpi=170)
if SHOW_FIGURE_IN_PYCHARM:
    plt.show()
plt.close(fig)
print("\nSaved:", RESULT_DIR / "known_validation_patch_comparison.csv")
print("No test pixels or unlabeled pixels were evaluated.")
