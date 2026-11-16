"""v0.3: compare six Random Forest feature sets on FULL VALIDATION.
Train is used only for fitting; validation is used only for comparisons.
The TEST split is never accessed by this script.
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from settings import (
    DATA_FILE, PROJECT_DIR, MODELS_DIR, CHUNK_SIZE, RANDOM_SEED,
    TRAIN_SAMPLES_PER_CLASS, NUMBER_OF_TREES, CPU_JOBS,
    BANDS, INDEX_NAMES, CLASSES,
)
from spectral_features import clean_labels, make_features

# ================================================================
# 1. Options: you can edit these without changing the algorithm.
# ================================================================
SHOW_FIGURES_IN_PYCHARM = True
SAVE_TRAINED_MODELS = True

# A new, separate folder keeps earlier benchmark outputs untouched.
OUTPUT_DIR = PROJECT_DIR / "results" / "ablation"
FIGURE_DIR = PROJECT_DIR / "assets" / "figures" / "ablation"
ABLATION_MODELS_DIR = MODELS_DIR / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Each experiment is trained FROM SCRATCH using the same training rows.
ALL_19 = BANDS + INDEX_NAMES
EXPERIMENTS = {
    "full_19": ALL_19,
    "minus_ndmi": [name for name in ALL_19 if name != "NDMI"],
    "minus_ndwi": [name for name in ALL_19 if name != "NDWI"],
    "minus_fdi": [name for name in ALL_19 if name != "FDI"],
    "minus_three": [name for name in ALL_19 if name not in ["NDMI", "NDWI", "FDI"]],
    "bands_only": BANDS,
}

if not DATA_FILE.is_file():
    raise FileNotFoundError("Missing dataset: " + str(DATA_FILE))

# Save the experiment settings so future readers know exactly what was run.
run_config = {
    "dataset": DATA_FILE.name,
    "fit_split": "train",
    "evaluation_split": "val",
    "test_split_used": False,
    "random_seed": RANDOM_SEED,
    "training_samples_per_class": TRAIN_SAMPLES_PER_CLASS,
    "number_of_trees": NUMBER_OF_TREES,
    "min_samples_leaf": 2,
    "cpu_jobs": CPU_JOBS,
    "confidence_weights": {"High": 1.0, "Moderate": 0.5, "Low": 1 / 3},
    "experiment_features": EXPERIMENTS,
}
(OUTPUT_DIR / "run_config.json").write_text(
    json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
)

with pd.HDFStore(DATA_FILE, mode="r") as store:
    missing = [name for name in ["train", "val"] if "/" + name not in store.keys()]
    if missing:
        raise ValueError("Missing HDF5 split(s): " + ", ".join(missing))

# ================================================================
# 2. Build one fixed stratified training sample.
#    This reproduces the sampling method in 02_train_random_forest.py.
# ================================================================
print("STEP 1/3: Sampling TRAIN pixels (same method as step 02)...")
rng = np.random.default_rng(RANDOM_SEED)
train_samples = pd.DataFrame()

with pd.HDFStore(DATA_FILE, mode="r") as store:
    for chunk_no, chunk in enumerate(store.select("train", chunksize=CHUNK_SIZE), 1):
        chunk = chunk.copy()
        chunk["Class"] = clean_labels(chunk["Class"])
        chunk = chunk[chunk["Class"].isin(CLASSES)].copy()

        # Random keys are generated in the same order as the existing baseline.
        chunk["random_key"] = rng.random(len(chunk))
        small = (
            chunk.sort_values("random_key")
            .groupby("Class", sort=False)
            .head(TRAIN_SAMPLES_PER_CLASS)
        )
        train_samples = pd.concat([train_samples, small], ignore_index=True)
        train_samples = (
            train_samples.sort_values("random_key")
            .groupby("Class", sort=False)
            .head(TRAIN_SAMPLES_PER_CLASS)
            .reset_index(drop=True)
        )
        if chunk_no % 10 == 0:
            print("  Processed TRAIN chunks:", chunk_no)

if train_samples.empty:
    raise ValueError("No labeled TRAIN pixels were found.")

sample_counts = train_samples["Class"].value_counts().reindex(CLASSES, fill_value=0)
sample_counts.rename_axis("class").reset_index(name="sampled_training_pixels").to_csv(
    OUTPUT_DIR / "training_sample_counts.csv", index=False
)
print("Training sample size:", len(train_samples))

# We calculate all 19 features once. Each model uses a different subset.
all_train_features = make_features(train_samples)
y_train = train_samples["Class"].to_numpy()

# Same confidence weighting as step 02, to keep the comparison fair.
if "Confidence" in train_samples.columns:
    confidence = train_samples["Confidence"].astype(str)
    weights = confidence.map({
        "High": 1.0, "Moderate": 0.5, "Low": 1 / 3,
        "1": 1.0, "2": 0.5, "3": 1 / 3,
    }).fillna(1.0).to_numpy()
else:
    weights = np.ones(len(train_samples))

# The old sampling table is no longer needed in memory.
del train_samples

# ================================================================
# 3. Train six RF models and evaluate the ENTIRE validation split.
# ================================================================
print("\nSTEP 2/3: Train from scratch and evaluate on FULL VALIDATION...")
comparison_rows = []
false_positive_rows = []

for experiment_name, features in EXPERIMENTS.items():
    print("\nTraining:", experiment_name, "| features:", len(features))
    started_at = time.perf_counter()

    x_train = all_train_features[features]
    # Medians are computed ONLY from the training sample (no validation leakage).
    medians = x_train.median().fillna(0)
    x_train = x_train.fillna(medians).fillna(0)

    model = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=CPU_JOBS,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    train_seconds = time.perf_counter() - started_at

    if SAVE_TRAINED_MODELS:
        joblib.dump(
            {"model": model, "features": features, "medians": medians.to_dict()},
            ABLATION_MODELS_DIR / (experiment_name + ".joblib"),
        )

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with pd.HDFStore(DATA_FILE, mode="r") as store:
        for chunk in store.select("val", chunksize=CHUNK_SIZE):
            truth = clean_labels(chunk["Class"])
            valid = np.isin(truth, CLASSES)
            if not valid.any():
                continue
            x_val = make_features(chunk.loc[valid])[features]
            x_val = x_val.fillna(medians).fillna(0)
            predicted = model.predict(x_val)
            confusion += confusion_matrix(truth[valid], predicted, labels=CLASSES)

    if confusion.sum() == 0:
        raise ValueError("Validation has no labeled pixels.")

    pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_csv(
        OUTPUT_DIR / (experiment_name + "_val_confusion.csv")
    )

    # Marine Debris is CLASSES[0]. All 15 classes are still used for training.
    tp = int(confusion[0, 0])
    fp = int(confusion[:, 0].sum() - tp)
    fn = int(confusion[0, :].sum() - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    iou = (tp / (tp + fp + fn)) if tp + fp + fn else 0.0

    comparison_rows.append({
        "experiment": experiment_name,
        "split": "val",
        "n_features": len(features),
        "train_pixels": len(y_train),
        "evaluated_labeled_pixels": int(confusion.sum()),
        "marine_debris_tp": tp,
        "marine_debris_fp": fp,
        "marine_debris_fn": fn,
        "marine_debris_precision": precision,
        "marine_debris_recall": recall,
        "marine_debris_f1": f1,
        "marine_debris_iou": iou,
        "sediment_laden_water_fp": int(confusion[CLASSES.index("Sediment-Laden Water"), 0]),
        "training_seconds": round(train_seconds, 2),
    })

    # A Marine Debris false positive means: true class is something else.
    for true_class in CLASSES[1:]:
        false_positive_rows.append({
            "experiment": experiment_name,
            "true_class": true_class,
            "predicted_class": "Marine Debris",
            "false_positive_pixels": int(confusion[CLASSES.index(true_class), 0]),
        })

    print(
        f"  VALIDATION: F1={f1:.4f} | Precision={precision:.4f} | "
        f"Recall={recall:.4f} | IoU={iou:.4f} | FP={fp} | "
        f"Sediment FP={comparison_rows[-1]['sediment_laden_water_fp']}"
    )

comparison = pd.DataFrame(comparison_rows)
full_f1 = float(comparison.loc[comparison.experiment == "full_19", "marine_debris_f1"].iloc[0])
full_fp = int(comparison.loc[comparison.experiment == "full_19", "marine_debris_fp"].iloc[0])
comparison["delta_f1_vs_full"] = comparison["marine_debris_f1"] - full_f1
comparison["delta_fp_vs_full"] = comparison["marine_debris_fp"] - full_fp
comparison.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)
pd.DataFrame(false_positive_rows).to_csv(
    OUTPUT_DIR / "false_positives_by_true_class.csv", index=False
)

# ================================================================
# 4. Save two clear, GitHub-friendly plots.
# ================================================================
print("\nSTEP 3/3: Saving figures...")
plot_labels = comparison["experiment"].str.replace("_", " ").tolist()
y_positions = np.arange(len(comparison))

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.barh(y_positions, comparison["marine_debris_f1"])
ax.axvline(full_f1, linestyle="--", color="black", linewidth=1.2, label="Full 19-feature baseline")
ax.set_yticks(y_positions, plot_labels)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Marine Debris F1 (full validation split)")
ax.set_title("MARIDA | Random Forest feature ablation")
ax.legend()
ax.grid(axis="x", alpha=0.15)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "validation_f1_by_feature_set.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.barh(y_positions, comparison["sediment_laden_water_fp"])
ax.set_yticks(y_positions, plot_labels)
ax.invert_yaxis()
ax.set_xlabel("Sediment-Laden Water pixels predicted as Marine Debris (FP)")
ax.set_title("MARIDA | Sediment false alarms on full validation")
ax.grid(axis="x", alpha=0.15)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "sediment_false_positives.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

print("\n=== FULL VALIDATION COMPARISON (NOT TEST) ===")
print(comparison[[
    "experiment", "n_features", "marine_debris_f1", "marine_debris_precision",
    "marine_debris_recall", "marine_debris_fp", "marine_debris_fn",
    "sediment_laden_water_fp", "delta_f1_vs_full",
]].round(4).to_string(index=False))
print("\nResult table:", OUTPUT_DIR / "validation_comparison.csv")
print("Figures:", FIGURE_DIR)
print("Trained models:", ABLATION_MODELS_DIR if SAVE_TRAINED_MODELS else "not saved")
print("TEST SPLIT NOT USED. Next: run 10_compare_ablation_patches.py.")
