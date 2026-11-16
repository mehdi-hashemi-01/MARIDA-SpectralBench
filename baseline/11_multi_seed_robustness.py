"""v0.3.1: Repeat all six RF ablations with multiple training seeds.
TRAIN is used to fit models and VAL is used to compare them. TEST is never read.
All six feature sets share the same sampled training pixels within each seed.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from settings import (
    DATA_FILE, PROJECT_DIR, CHUNK_SIZE, TRAIN_SAMPLES_PER_CLASS,
    NUMBER_OF_TREES, CPU_JOBS, BANDS, INDEX_NAMES, CLASSES,
)
from spectral_features import clean_labels, make_features

# ==============================================================
# 1. EDITABLE OPTIONS
# ==============================================================
SEEDS = [42, 7, 21, 123, 2026]  # Five independent repetitions of the pipeline
SHOW_FIGURES_IN_PYCHARM = False  # False prevents plots from pausing the runs

ALL_FEATURES = BANDS + INDEX_NAMES
EXPERIMENTS = {
    "full_19": ALL_FEATURES,
    "minus_ndmi": [x for x in ALL_FEATURES if x != "NDMI"],
    "minus_ndwi": [x for x in ALL_FEATURES if x != "NDWI"],
    "minus_fdi": [x for x in ALL_FEATURES if x != "FDI"],
    "minus_three": [x for x in ALL_FEATURES if x not in ["NDMI", "NDWI", "FDI"]],
    "bands_only": BANDS,
}

RESULT_DIR = PROJECT_DIR / "results" / "robustness"
FIGURE_DIR = PROJECT_DIR / "assets" / "figures" / "robustness"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.is_file():
    raise FileNotFoundError("Cannot find HDF5: " + str(DATA_FILE))
if len(SEEDS) != len(set(SEEDS)):
    raise ValueError("SEEDS must not contain duplicates.")

# This configuration is saved for scientific reproducibility.
config = {
    "data_file": DATA_FILE.name,
    "train_split": "train", "evaluation_split": "val", "test_used": False,
    "seeds": SEEDS, "sampling_seed_and_forest_seed_are_both_changed": True,
    "train_pixels_per_class_limit": TRAIN_SAMPLES_PER_CLASS,
    "trees": NUMBER_OF_TREES, "min_samples_leaf": 2,
    "cpu_jobs": CPU_JOBS, "chunk_size": CHUNK_SIZE,
    "confidence_weights": {"High": 1.0, "Moderate": 0.5, "Low": 1 / 3},
    "experiments": EXPERIMENTS,
    "grouping": {"patch": "Date + Tile + Image",
                 "scene": "Date + Tile (acquisition proxy)",
                 "tile": "Tile (geographic proxy)"},
    "warning": "These groups are descriptive; this is not grouped cross-validation.",
}
(RESULT_DIR / "run_config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RESULT_DIR / "run_status.json").write_text(
    json.dumps({"state": "running", "seeds": SEEDS, "expected_runs": len(SEEDS) * len(EXPERIMENTS)}, indent=2),
    encoding="utf-8",
)


def sample_train_pixels(seed):
    """Keep at most N pixels per class using random-key sampling from step 09."""
    rng = np.random.default_rng(seed)
    samples = pd.DataFrame()
    with pd.HDFStore(DATA_FILE, mode="r") as store:
        for chunk in store.select("train", chunksize=CHUNK_SIZE):
            chunk = chunk.copy()
            chunk["Class"] = clean_labels(chunk["Class"])
            chunk = chunk.loc[chunk["Class"].isin(CLASSES)].copy()
            chunk["random_key"] = rng.random(len(chunk))
            small = (chunk.sort_values("random_key")
                     .groupby("Class", sort=False)
                     .head(TRAIN_SAMPLES_PER_CLASS))
            samples = pd.concat([samples, small], ignore_index=True)
            samples = (samples.sort_values("random_key")
                       .groupby("Class", sort=False)
                       .head(TRAIN_SAMPLES_PER_CLASS)
                       .reset_index(drop=True))
    if samples.empty:
        raise ValueError("No labeled TRAIN pixels found.")
    return samples


def counts_by_group(codes, group_info, true_ids, predicted_ids):
    """Compute TP/FP/FN independently for each patch, scene or tile."""
    group_count = len(group_info)
    debris = true_ids == 0  # CLASSES[0] == Marine Debris
    predicted_debris = predicted_ids == 0
    tp = debris & predicted_debris
    fp = ~debris & predicted_debris
    fn = debris & ~predicted_debris

    def count(mask):
        return np.bincount(codes, weights=mask.astype(np.int64),
                           minlength=group_count).astype(np.int64)

    output = group_info.copy()
    output["labeled_pixels"] = np.bincount(codes, minlength=group_count)
    output["true_debris_pixels"] = count(debris)
    output["predicted_debris_pixels"] = count(predicted_debris)
    output["tp"] = count(tp)
    output["fp"] = count(fp)
    output["fn"] = count(fn)
    for class_name, column in [
        ("Sediment-Laden Water", "sediment_fp"),
        ("Ship", "ship_fp"),
        ("Clouds", "cloud_fp"),
        ("Wakes", "wakes_fp"),
        ("Waves", "waves_fp"),
    ]:
        output[column] = count(fp & (true_ids == CLASSES.index(class_name)))

    output["sediment_labeled_pixels"] = count(
        true_ids == CLASSES.index("Sediment-Laden Water")
    )
    denominator = 2 * output["tp"] + output["fp"] + output["fn"]
    # No real debris AND no predicted debris => undefined, not F1=0.
    output["f1"] = np.where(denominator > 0,
                            2 * output["tp"] / np.maximum(denominator, 1), np.nan)
    output["f1_defined"] = denominator > 0
    output["precision"] = np.where(
        output["tp"] + output["fp"] > 0,
        output["tp"] / np.maximum(output["tp"] + output["fp"], 1), np.nan
    )
    output["recall"] = np.where(
        output["tp"] + output["fn"] > 0,
        output["tp"] / np.maximum(output["tp"] + output["fn"], 1), np.nan
    )
    return output


# ==============================================================
# 2. LOAD FULL VALIDATION ONCE (NO TEST ROWS)
# ==============================================================
print("STEP 1/3 | Reading full labeled VALIDATION once...")
with pd.HDFStore(DATA_FILE, mode="r") as store:
    keys = store.keys()
    if "/train" not in keys or "/val" not in keys:
        raise ValueError("HDF5 must contain /train and /val. Found: " + str(keys))

validation_feature_parts = []
validation_metadata_parts = []
validation_label_parts = []
required = set(BANDS + ["Class", "Date", "Tile", "Image"])

with pd.HDFStore(DATA_FILE, mode="r") as store:
    for chunk in store.select("val", chunksize=CHUNK_SIZE):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError("Missing columns for patch-level analysis: " + str(sorted(missing)))
        labels = clean_labels(chunk["Class"])
        valid = np.isin(labels, CLASSES)
        if not valid.any():
            continue
        chunk = chunk.loc[valid].copy()
        validation_feature_parts.append(make_features(chunk))
        validation_label_parts.append(labels[valid])
        validation_metadata_parts.append(chunk[["Date", "Tile", "Image"]])

if not validation_feature_parts:
    raise ValueError("No annotated validation pixels found.")

validation_features = pd.concat(validation_feature_parts, ignore_index=True)
true_labels = np.concatenate(validation_label_parts)
true_ids = pd.Categorical(true_labels, categories=CLASSES).codes
if (true_ids < 0).any():
    raise ValueError("Unknown validation class labels.")
metadata = pd.concat(validation_metadata_parts, ignore_index=True)
if metadata.isna().any().any():
    raise ValueError("Date / Tile / Image contains nulls; cannot group safely.")

metadata = pd.DataFrame({
    "date": metadata["Date"].astype(str).str.strip().str.lower(),
    "tile": metadata["Tile"].astype(str).str.strip().str.lower(),
    "image": metadata["Image"].astype(str).str.strip().str.lower(),
})
metadata["patch_id"] = metadata["date"] + "|" + metadata["tile"] + "|" + metadata["image"]
metadata["scene_id"] = metadata["date"] + "|" + metadata["tile"]

patch_codes, patch_names = pd.factorize(metadata["patch_id"], sort=False)
scene_codes, scene_names = pd.factorize(metadata["scene_id"], sort=False)
tile_codes, tile_names = pd.factorize(metadata["tile"], sort=False)
patch_info = metadata.drop_duplicates("patch_id").set_index("patch_id").loc[patch_names].rename_axis("patch_id").reset_index()
scene_info = metadata.drop_duplicates("scene_id")[["scene_id", "date", "tile"]].set_index("scene_id").loc[scene_names].rename_axis("scene_id").reset_index()
tile_info = pd.DataFrame({"tile": tile_names})
print("  Labeled VAL pixels:", len(true_ids))
print("  Distinct patches:", len(patch_names))
print("  Acquisition groups (Date + Tile):", len(scene_names))
print("  Geographic tile groups:", len(tile_names))

# Audit metadata overlap between official TRAIN and VAL splits. This is
# NOT a new split, but helps prevent overclaiming scene independence.
print("  Auditing TRAIN/VAL patch, acquisition, and tile group overlap...")
train_patch_ids, train_scene_ids, train_tile_ids = set(), set(), set()
with pd.HDFStore(DATA_FILE, mode="r") as store:
    for train_chunk in store.select("train", chunksize=CHUNK_SIZE):
        if not {"Date", "Tile", "Image"}.issubset(train_chunk.columns):
            raise ValueError("TRAIN metadata must include Date, Tile and Image.")
        d = train_chunk["Date"].astype(str).str.strip().str.lower()
        t = train_chunk["Tile"].astype(str).str.strip().str.lower()
        i = train_chunk["Image"].astype(str).str.strip().str.lower()
        sc = d + "|" + t
        train_patch_ids.update((sc + "|" + i).unique())
        train_scene_ids.update(sc.unique())
        train_tile_ids.update(t.unique())

val_patch_ids = set(metadata["patch_id"].unique())
val_scene_ids = set(metadata["scene_id"].unique())
val_tile_ids = set(metadata["tile"].unique())
shared = {
    "patch": sorted(train_patch_ids & val_patch_ids),
    "scene_date_tile": sorted(train_scene_ids & val_scene_ids),
    "tile": sorted(train_tile_ids & val_tile_ids),
}
train_all = {"patch": train_patch_ids, "scene_date_tile": train_scene_ids,
             "tile": train_tile_ids}
val_all = {"patch": val_patch_ids, "scene_date_tile": val_scene_ids,
           "tile": val_tile_ids}
overlap_rows = []
for level, overlapping in shared.items():
    overlap_rows.append({
        "grouping": level,
        "train_distinct_groups": len(train_all[level]),
        "validation_distinct_groups": len(val_all[level]),
        "shared_train_validation_groups": len(overlapping),
        "fraction_validation_groups_shared": (
            len(overlapping) / len(val_all[level]) if len(val_all[level]) else np.nan
        ),
    })
    print(f"    {level}: {len(overlapping)} shared groups")
    pd.DataFrame({"shared_group_id": overlapping}).to_csv(
        RESULT_DIR / ("train_val_shared_" + level + ".csv"), index=False
    )
pd.DataFrame(overlap_rows).to_csv(RESULT_DIR / "train_val_group_overlap.csv", index=False)

# ==============================================================
# 3. REPEAT ALL SIX MODELS FOR FIVE SEEDS
# ==============================================================
print("\nSTEP 2/3 | Starting", len(SEEDS), "seeds x", len(EXPERIMENTS), "feature sets...")
run_rows = []
patch_tables = []
scene_tables = []
tile_tables = []
false_positive_rows = []

for seed in SEEDS:
    print("\n========== SEED", seed, "==========", flush=True)
    samples = sample_train_pixels(seed)
    x_all_train = make_features(samples)
    y_train = samples["Class"].to_numpy()
    if "Confidence" in samples.columns:
        confidence = samples["Confidence"].astype(str)
        sample_weights = confidence.map({
            "High": 1.0, "Moderate": 0.5, "Low": 1 / 3,
            "1": 1.0, "2": 0.5, "3": 1 / 3,
        }).fillna(1.0).to_numpy()
    else:
        sample_weights = np.ones(len(samples))
    train_count = len(samples)
    del samples

    for experiment, columns in EXPERIMENTS.items():
        print("  Training", experiment, "...", flush=True)
        started = time.perf_counter()
        x_train = x_all_train[columns]
        medians = x_train.median().fillna(0)  # Train only: no validation leakage
        x_train = x_train.fillna(medians).fillna(0)

        rf = RandomForestClassifier(
            n_estimators=NUMBER_OF_TREES, min_samples_leaf=2,
            random_state=seed, n_jobs=CPU_JOBS,
        )
        rf.fit(x_train, y_train, sample_weight=sample_weights)
        seconds = time.perf_counter() - started
        x_val = validation_features[columns].fillna(medians).fillna(0)
        predicted = rf.predict(x_val)
        predicted_ids = pd.Categorical(predicted, categories=CLASSES).codes
        if (predicted_ids < 0).any():
            raise ValueError("Model returned an unknown class label.")

        actual_debris = true_ids == 0
        predicted_debris = predicted_ids == 0
        tp = int(np.sum(actual_debris & predicted_debris))
        fp = int(np.sum(~actual_debris & predicted_debris))
        fn = int(np.sum(actual_debris & ~predicted_debris))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
        sediment_fp = int(np.sum(
            (true_ids == CLASSES.index("Sediment-Laden Water")) & predicted_debris
        ))

        run_rows.append({
            "seed": seed, "experiment": experiment, "n_features": len(columns),
            "train_pixels": train_count, "validation_labeled_pixels": len(true_ids),
            "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "iou": iou,
            "sediment_fp": sediment_fp, "train_seconds": round(seconds, 3),
        })
        for true_class in CLASSES[1:]:
            count = int(np.sum(
                (true_ids == CLASSES.index(true_class)) & predicted_debris
            ))
            false_positive_rows.append({
                "seed": seed, "experiment": experiment,
                "true_class": true_class, "false_positive_pixels": count,
            })

        for codes, info, collection in [
            (patch_codes, patch_info, patch_tables),
            (scene_codes, scene_info, scene_tables),
            (tile_codes, tile_info, tile_tables),
        ]:
            group_result = counts_by_group(codes, info, true_ids, predicted_ids)
            group_result.insert(0, "experiment", experiment)
            group_result.insert(0, "seed", seed)
            collection.append(group_result)

        print(f"    VAL F1={f1:.4f} | FP={fp} | FN={fn} | Sediment FP={sediment_fp}", flush=True)
        del rf, x_train, x_val, predicted, predicted_ids
    del x_all_train

runs = pd.DataFrame(run_rows)
runs.to_csv(RESULT_DIR / "per_seed_validation.csv", index=False)
pd.concat(patch_tables, ignore_index=True).to_csv(RESULT_DIR / "per_patch_metrics.csv", index=False)
pd.concat(scene_tables, ignore_index=True).to_csv(RESULT_DIR / "per_scene_metrics.csv", index=False)
pd.concat(tile_tables, ignore_index=True).to_csv(RESULT_DIR / "per_tile_metrics.csv", index=False)
pd.DataFrame(false_positive_rows).to_csv(
    RESULT_DIR / "false_positives_by_class.csv", index=False
)

# ==============================================================
# 4. AGGREGATE MULTI-SEED RESULTS + PAIRED DIFFERENCES
# ==============================================================
print("\nSTEP 3/3 | Saving summaries and plots...")
summary = runs.groupby("experiment", sort=False).agg(
    n_seeds=("seed", "count"),
    mean_f1=("f1", "mean"), sd_f1=("f1", "std"),
    min_f1=("f1", "min"), max_f1=("f1", "max"),
    mean_precision=("precision", "mean"),
    mean_recall=("recall", "mean"),
    mean_iou=("iou", "mean"),
    mean_fp=("fp", "mean"), sd_fp=("fp", "std"),
    mean_fn=("fn", "mean"),
    mean_sediment_fp=("sediment_fp", "mean"),
).reset_index()
summary["sd_f1"] = summary["sd_f1"].fillna(0)
summary["sd_fp"] = summary["sd_fp"].fillna(0)
summary.to_csv(RESULT_DIR / "summary_by_model.csv", index=False)

reference = runs.loc[runs["experiment"] == "full_19", ["seed", "f1", "fp", "fn", "sediment_fp"]]
reference = reference.rename(columns={
    "f1": "full_f1", "fp": "full_fp", "fn": "full_fn",
    "sediment_fp": "full_sediment_fp",
})
paired = runs.merge(reference, on="seed", validate="many_to_one")
for metric in ["f1", "fp", "fn", "sediment_fp"]:
    paired["delta_" + metric + "_vs_full"] = paired[metric] - paired["full_" + metric]
paired[["seed", "experiment", "delta_f1_vs_full", "delta_fp_vs_full",
        "delta_fn_vs_full", "delta_sediment_fp_vs_full"]].to_csv(
    RESULT_DIR / "paired_deltas_vs_full.csv", index=False
)
paired_summary = paired.groupby("experiment", sort=False).agg(
    mean_delta_f1=("delta_f1_vs_full", "mean"),
    sd_delta_f1=("delta_f1_vs_full", "std"),
    mean_delta_fp=("delta_fp_vs_full", "mean"),
    mean_delta_fn=("delta_fn_vs_full", "mean"),
    mean_delta_sediment_fp=("delta_sediment_fp_vs_full", "mean"),
).reset_index()
paired_summary.to_csv(RESULT_DIR / "paired_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 5.6))
for name in EXPERIMENTS:
    subset = runs[runs.experiment == name].set_index("seed").loc[SEEDS]
    ax.plot(range(len(SEEDS)), subset["f1"], marker="o", label=name)
ax.set_xticks(range(len(SEEDS)), [str(seed) for seed in SEEDS])
ax.set_ylabel("Marine Debris F1 (full validation)")
ax.set_xlabel("Training and Random Forest seed")
ax.set_title("MARIDA | Feature ablation across repeated seeds")
ax.legend(ncol=3, fontsize=8)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "validation_f1_across_seeds.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.3))
positions = np.arange(len(summary))
ax.errorbar(summary["mean_f1"], positions, xerr=summary["sd_f1"],
            fmt="o", capsize=5)
ax.set_yticks(positions, summary["experiment"])
ax.invert_yaxis()
ax.set_xlabel("Mean validation Marine Debris F1 +/- 1 seed SD")
ax.set_title("MARIDA | Mean and standard deviation over seeds")
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "validation_f1_mean_sd.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

(RESULT_DIR / "run_status.json").write_text(
    json.dumps({"state": "complete", "seeds": SEEDS, "expected_runs": len(SEEDS) * len(EXPERIMENTS),
                "completed_runs": len(runs)}, indent=2),
    encoding="utf-8",
)

print("\n===== MULTI-SEED VALIDATION (NOT TEST) =====")
print(summary[["experiment", "mean_f1", "sd_f1", "mean_fp", "sd_fp",
               "mean_sediment_fp"]].round(4).to_string(index=False))
print("\nCSV outputs:", RESULT_DIR)
print("Figures:", FIGURE_DIR)
print("Next: Run 12_analyze_scene_errors.py (does not train any models).")
print("Important: varying seed tests training stability, NOT independent-scene generalization.")
