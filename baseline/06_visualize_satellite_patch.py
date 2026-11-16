"""Visualize one original MARIDA GeoTIFF patch and its label/confidence masks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from settings import DATA_FILE, RESULTS_DIR, FIGURES_DIR, CLASSES

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
data_root = DATA_FILE.parent
mapping_path = data_root / "labels_mapping.txt"
patch_root = data_root / "patches"
if not patch_root.exists():
    raise FileNotFoundError(f"Original MARIDA patches not found in {patch_root}")

# Prefer a patch with a labeled Marine Debris example.
chosen = None
if mapping_path.exists():
    labels_mapping = json.loads(mapping_path.read_text(encoding="utf8"))
    debris_images = [filename for filename, indicators in labels_mapping.items()
                     if len(indicators) == 15 and indicators[0] == 1]
    for filename in debris_images:
        candidates = list(patch_root.rglob(filename))
        if candidates:
            chosen = candidates[0]
            break
if chosen is None:
    chosen = next((p for p in patch_root.rglob("*.tif")
                   if not p.name.endswith(("_cl.tif", "_conf.tif"))), None)
if chosen is None:
    raise FileNotFoundError("No original patch GeoTIFF was found")

label_path = chosen.with_name(chosen.stem + "_cl.tif")
confidence_path = chosen.with_name(chosen.stem + "_conf.tif")
if not label_path.exists() or not confidence_path.exists():
    raise FileNotFoundError("Matching _cl.tif or _conf.tif is missing")

image = tifffile.imread(chosen)
labels = np.squeeze(tifffile.imread(label_path))
confidence = np.squeeze(tifffile.imread(confidence_path))

# MARIDA multispectral GeoTIFFs normally use bands-first storage.
if image.ndim != 3:
    raise ValueError(f"Expected multispectral 3D image, got {image.shape}")
if image.shape[0] == 11:
    image = np.moveaxis(image, 0, -1)
if image.shape[-1] < 4:
    raise ValueError("Expected the visible Sentinel-2 spectral bands")

# Red = nm665, Green = nm560, Blue = nm490 (band order in MARIDA).
rgb = image[:, :, [3, 2, 1]].astype(float)
for channel in range(3):
    plane = rgb[:, :, channel]
    finite = plane[np.isfinite(plane)]
    if len(finite):
        lo, hi = np.percentile(finite, [2, 98])
        rgb[:, :, channel] = np.clip((plane - lo) / max(hi - lo, 1e-8), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB composite")
mask = np.ma.masked_where(labels == 0, labels)
im = axes[1].imshow(mask, cmap="tab20", vmin=1, vmax=15, interpolation="nearest")
axes[1].set_title("Annotated classes (nonzero pixels)")
cbar = fig.colorbar(im, ax=axes[1], fraction=.046, pad=.04)
cbar.set_ticks([1, 5, 7, 10, 15])
cbar.set_ticklabels(["Debris", "Ship", "Water", "Turbid", "Mixed"])
conf = np.ma.masked_where(confidence == 0, confidence)
im2 = axes[2].imshow(conf, cmap="viridis", vmin=1, vmax=3, interpolation="nearest")
axes[2].set_title("Annotation confidence")
cbar2 = fig.colorbar(im2, ax=axes[2], fraction=.046, pad=.04)
cbar2.set_ticks([1, 2, 3])
cbar2.set_ticklabels(["High", "Moderate", "Low"])
for ax in axes:
    ax.axis("off")
fig.suptitle(chosen.name)
fig.tight_layout()
file_out = FIGURES_DIR / "marida_patch_example.png"
fig.savefig(file_out, dpi=170)
plt.show()

counts = pd.Series(labels.reshape(-1)).value_counts().sort_index()
pd.DataFrame({"class_id": counts.index, "annotated_pixels": counts.values}).to_csv(
    RESULTS_DIR / "visualized_patch_class_counts.csv", index=False)
print("Patch:", chosen)
print("Figure:", file_out)
