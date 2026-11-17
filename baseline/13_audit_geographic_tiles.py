"""v0.3.2 | Step 13: audit geographic Tile groups in MARIDA TRAIN.

Reads /train only. Does NOT inspect /val or /test.
"""

import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import DATA_FILE, PROJECT_DIR, CHUNK_SIZE, CLASSES
from spectral_features import clean_labels

# ---------------------- SETTINGS ----------------------
SHOW_FIGURES_IN_PYCHARM = False
MIN_DEBRIS_FOR_INFORMATIVE_F1 = 20  # descriptive warning, NOT a filtering rule

OUTPUT_DIR = PROJECT_DIR / "results" / "geographic_holdout"
FIGURE_DIR = PROJECT_DIR / "assets" / "figures" / "geographic_holdout"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.is_file():
    raise FileNotFoundError("dataset.h5 not found: " + str(DATA_FILE))

print("STEP 13 | Auditing class distribution by geographic Tile...")
print("Source: /train ONLY. Original /val and /test are not read.")

class_counts = Counter()
scene_ids = defaultdict(set)
patch_ids = defaultdict(set)
valid_total = 0

with pd.HDFStore(DATA_FILE, "r") as store:
    if "/train" not in store.keys():
        raise ValueError("The HDF5 file must have a /train table.")
    for number, chunk in enumerate(store.select("train", chunksize=CHUNK_SIZE), 1):
        needed = {"Class", "Date", "Tile", "Image"}
        if not needed.issubset(chunk.columns):
            raise ValueError("Missing HDF5 columns: " + str(sorted(needed - set(chunk.columns))))

        classes = clean_labels(chunk["Class"])
        mask = np.isin(classes, CLASSES)
        if not np.any(mask):
            continue
        part = chunk.loc[mask, ["Date", "Tile", "Image"]].copy()
        if part[["Date", "Tile", "Image"]].isna().any().any():
            raise ValueError("Found missing Date / Tile / Image metadata in TRAIN.")
        part["class"] = classes[mask]
        part["tile"] = part["Tile"].astype(str).str.strip().str.upper()
        part["date"] = part["Date"].astype(str).str.strip().str.lower()
        part["image"] = part["Image"].astype(str).str.strip().str.lower()
        if part["tile"].isin(["", "NAN", "NONE"]).any():
            raise ValueError("Some labeled pixels have missing Tile identifiers.")

        valid_total += len(part)
        by_class = part.groupby(["tile", "class"]).size()
        for (tile, class_name), amount in by_class.items():
            class_counts[(tile, class_name)] += int(amount)

        # Count unique acquisition scenes and patches, not individual pixels.
        for tile, date, image in part[["tile", "date", "image"]].drop_duplicates().itertuples(index=False, name=None):
            scene_ids[tile].add(date + "|" + tile)
            patch_ids[tile].add(date + "|" + tile + "|" + image)
        if number % 10 == 0:
            print("  Scanned TRAIN chunks:", number)

all_tiles = sorted({tile for (tile, _) in class_counts})
if not all_tiles:
    raise ValueError("No labeled training pixels with valid Tile identifiers.")

long_rows = []
for tile in all_tiles:
    for class_name in CLASSES:
        long_rows.append({"tile": tile, "class": class_name,
                          "labeled_pixels": class_counts.get((tile, class_name), 0)})
counts_df = pd.DataFrame(long_rows)
counts_df.to_csv(OUTPUT_DIR / "tile_class_counts.csv", index=False)

wide = counts_df.pivot(index="tile", columns="class", values="labeled_pixels").fillna(0).astype(int)
wide = wide.reindex(columns=CLASSES)
wide.to_csv(OUTPUT_DIR / "tile_class_counts_wide.csv")

all_debris = int(wide["Marine Debris"].sum())
summary_rows = []
for tile in all_tiles:
    counts = wide.loc[tile]
    labeled = int(counts.sum())
    debris = int(counts["Marine Debris"])
    other_debris = all_debris - debris
    # The remainder of the TRAIN split must be capable of learning >1 class.
    other_class_count = sum(int(wide[c].sum() - counts[c]) > 0 for c in CLASSES)
    can_train = other_debris > 0 and other_class_count >= 2
    note = "debris_support_ok"
    if debris == 0:
        note = "no_true_debris__f1_not_defined"
    elif debris < MIN_DEBRIS_FOR_INFORMATIVE_F1:
        note = "few_true_debris__interpret_with_care"
    if not can_train:
        note += "__training_support_insufficient"

    summary_rows.append({
        "tile": tile,
        "labeled_pixels": labeled,
        "marine_debris_pixels": debris,
        "non_debris_pixels": labeled - debris,
        "debris_percent": round(100 * debris / labeled, 5) if labeled else 0,
        "classes_present": int((counts > 0).sum()),
        "scene_count": len(scene_ids[tile]),
        "patch_count": len(patch_ids[tile]),
        "training_debris_pixels_excluding_tile": other_debris,
        "training_classes_excluding_tile": other_class_count,
        "run_fold": bool(can_train and labeled > 0),
        "interpretation_note": note,
    })

summary = pd.DataFrame(summary_rows).sort_values("tile").reset_index(drop=True)
summary.to_csv(OUTPUT_DIR / "tile_summary.csv", index=False)

# All eligible tiles are held out once; no tile is selected based on F1 or FP.
plan = summary[["tile", "labeled_pixels", "marine_debris_pixels", "run_fold",
                "interpretation_note"]].copy()
plan.to_csv(OUTPUT_DIR / "tile_holdout_plan.csv", index=False)

config = {
    "version": "v0.3.2",
    "source_hdf_key": "/train",
    "original_val_read": False,
    "original_test_read": False,
    "method": "Leave-One-Tile-Out (LOTO), geographic proxy = Tile metadata",
    "all_eligible_tiles_evaluated": True,
    "min_debris_for_informative_f1_warning_only": MIN_DEBRIS_FOR_INFORMATIVE_F1,
    "total_train_labeled_pixels": valid_total,
    "total_debris_pixels": all_debris,
    "total_tiles": len(all_tiles),
    "runnable_folds": int(plan["run_fold"].sum()),
}
(OUTPUT_DIR / "audit_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

# Figures are for reporting only: chart labels use Tile IDs, not geographic coordinates.
shown = summary.sort_values("marine_debris_pixels", ascending=True)
fig, ax = plt.subplots(figsize=(10, max(4.5, 0.37 * len(shown) + 1.5)))
ax.barh(shown["tile"], shown["marine_debris_pixels"])
ax.set_xlabel("Annotated Marine Debris pixels in official TRAIN")
ax.set_ylabel("Tile ID (not a precise polygon)")
ax.set_title("MARIDA | Marine Debris annotation support by Tile")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "train_debris_by_tile.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

# Heatmap: log-scaled counts reveal classes with tiny support as well as large ones.
heatmap_data = np.log10(1 + wide.to_numpy(dtype=float))
fig, ax = plt.subplots(figsize=(13, max(4.3, 0.45 * len(wide) + 1.5)))
im = ax.imshow(heatmap_data, aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(len(wide)), wide.index.tolist(), fontsize=9)
ax.set_title("MARIDA TRAIN | Annotated class support per geographic Tile")
fig.colorbar(im, ax=ax, shrink=0.75, label="log10(1 + labeled pixels)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "tile_class_support_heatmap.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

print("\n===== TRAIN TILE AUDIT =====")
print(summary[["tile", "labeled_pixels", "marine_debris_pixels", "scene_count",
               "run_fold", "interpretation_note"]].to_string(index=False))
print("\nLabeled TRAIN pixels:", valid_total)
print("Marine Debris TRAIN pixels:", all_debris)
print("Total Tile groups:", len(all_tiles))
print("Runnable held-out Tile folds:", int(plan["run_fold"].sum()))
print("\nSaved:", OUTPUT_DIR)
print("NEXT: Run 14_geographic_holdout.py")
