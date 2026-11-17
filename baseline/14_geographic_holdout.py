"""v0.3.2 | Step 14: geographically disjoint, Leave-One-Tile-Out RF study.

Uses official MARIDA /train only, never /val or /test.

Each fold: keep one Tile completely OUT of model fitting; all other Tiles
form the fit pool. Comparison feature sets share one training pixel sample.
Pooled out-of-fold micro F1 is NOT the official validation/test benchmark.
"""

import json
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from settings import (DATA_FILE, PROJECT_DIR, CHUNK_SIZE, TRAIN_SAMPLES_PER_CLASS,
                      NUMBER_OF_TREES, CPU_JOBS, RANDOM_SEED, BANDS, INDEX_NAMES, CLASSES)
from spectral_features import clean_labels, make_features

# --------------------- EDITABLE SETTINGS ---------------------
SHOW_FIGURES_IN_PYCHARM = False
# Keep 0 for ALL eligible Tile groups. A small value is for a quick code check,
# and must NOT be reported as a complete geographic validation study.
MAX_FOLDS = 0
# Previously examined 19-feature model and simpler 11-band baseline, plus
# minus-NDWI to assess the scene-specific tradeoff seen in v0.3/v0.3.1.
ALL_19 = BANDS + INDEX_NAMES
EXPERIMENTS = {
    "full_19": ALL_19,
    "minus_ndwi": [x for x in ALL_19 if x != "NDWI"],
    "bands_only": BANDS,
}

OUTPUT_DIR = PROJECT_DIR / "results" / "geographic_holdout"
FIGURE_DIR = PROJECT_DIR / "assets" / "figures" / "geographic_holdout"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PLAN_FILE = OUTPUT_DIR / "tile_holdout_plan.csv"
if not DATA_FILE.exists():
    raise FileNotFoundError("Missing HDF5: " + str(DATA_FILE))
if not PLAN_FILE.is_file():
    raise FileNotFoundError("Run 13_audit_geographic_tiles.py first: " + str(PLAN_FILE))

plan = pd.read_csv(PLAN_FILE)
needed_plan = {"tile", "labeled_pixels", "marine_debris_pixels", "run_fold"}
if not needed_plan.issubset(plan.columns):
    raise ValueError("Invalid fold plan: " + str(sorted(needed_plan - set(plan.columns))))

# Parse user-editable plan safely (Pandas may read bool as strings).
plan["tile"] = plan["tile"].astype(str).str.strip().str.upper()
plan["run_fold"] = plan["run_fold"].astype(str).str.lower().isin(["true", "1", "yes"])
if plan["tile"].duplicated().any():
    raise ValueError("Duplicate Tile IDs in tile_holdout_plan.csv")
folds = plan.loc[plan.run_fold, "tile"].tolist()
if MAX_FOLDS > 0:
    folds = folds[:MAX_FOLDS]
if not folds:
    raise ValueError("No runnable folds in plan. Check the TRAIN Tile/class audit.")

with pd.HDFStore(DATA_FILE, "r") as store:
    if "/train" not in store.keys():
        raise ValueError("Expected /train in dataset.h5")

config = {
    "study": "Tile-grouped Leave-One-Tile-Out (LOTO)",
    "input_split": "/train only",
    "official_val_read": False,
    "official_test_read": False,
    "heldout_groups": folds,
    "num_holdout_groups": len(folds),
    "all_eligible_folds_executed": MAX_FOLDS == 0 and len(folds) == int(plan.run_fold.sum()),
    "group_proxy": "Tile string from HDF5 metadata; actual spatial footprints not audited",
    "random_seed": RANDOM_SEED,
    "training_sample_cap_per_class": TRAIN_SAMPLES_PER_CLASS,
    "n_trees": NUMBER_OF_TREES,
    "min_samples_leaf": 2,
    "models": EXPERIMENTS,
    "important": "All feature sets share a sampled fit set within each holdout fold.",
    "known_limitation": "The feature sets were previously explored on official validation: not pristine external confirmation.",
}
(OUTPUT_DIR / "holdout_run_config.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
(OUTPUT_DIR / "holdout_run_status.json").write_text(
    json.dumps({"status": "running", "folds_completed": 0, "folds_planned": len(folds)}, indent=2), encoding="utf-8")

# Function 1: sample from all TRAIN tiles except the geographic holdout.
def make_fold_training_sample(heldout_tile):
    rng = np.random.default_rng(RANDOM_SEED)
    samples = pd.DataFrame()
    fit_tiles = set()
    with pd.HDFStore(DATA_FILE, "r") as store:
        for chunk in store.select("train", chunksize=CHUNK_SIZE):
            tiles = chunk["Tile"].astype(str).str.strip().str.upper()
            truth = clean_labels(chunk["Class"])
            allowed = (tiles != heldout_tile) & np.isin(truth, CLASSES)
            if not np.any(allowed):
                continue
            part = chunk.loc[allowed].copy()
            part["Class"] = truth[allowed]
            fit_tiles.update(tiles[allowed].unique())
            part["random_key"] = rng.random(len(part))
            small = part.sort_values("random_key").groupby("Class", sort=False).head(TRAIN_SAMPLES_PER_CLASS)
            samples = pd.concat([samples, small], ignore_index=True)
            samples = (samples.sort_values("random_key")
                       .groupby("Class", sort=False)
                       .head(TRAIN_SAMPLES_PER_CLASS)
                       .reset_index(drop=True))

    if samples.empty or "Marine Debris" not in samples["Class"].values:
        raise ValueError("Training pool has no debris for held-out Tile " + heldout_tile)
    if heldout_tile in fit_tiles:
        raise AssertionError("GEOGRAPHIC LEAKAGE: heldout tile in training group set!")
    samples = samples.drop(columns=["random_key"])
    return samples, sorted(fit_tiles)


def scores_from_confusion(cm):
    """Marine Debris is at matrix position 0; F1 undefined for no true debris."""
    tp = int(cm[0, 0])
    fp = int(cm[:, 0].sum() - tp)
    fn = int(cm[0, :].sum() - tp)
    positives = int(cm[0, :].sum())
    negative_count = int(cm.sum()) - positives
    predicted_pos = int(cm[:, 0].sum())
    precision = tp / predicted_pos if predicted_pos else np.nan
    recall = tp / positives if positives else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if positives else np.nan
    iou = tp / (tp + fp + fn) if positives else np.nan
    return {"labeled_pixels": int(cm.sum()), "true_debris": positives,
            "non_debris": negative_count, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "fp_per_1000_non_debris": 1000 * fp / negative_count if negative_count else np.nan}


fold_rows = []
scene_rows = []
class_error_rows = []
all_confusions = {name: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
                  for name in EXPERIMENTS}
completed_tiles = []

print("===== GEOGRAPHIC HOLDOUT: OFFICIAL TRAIN ONLY =====")
print("Fold groups:", ", ".join(folds))
print("Models per fold:", ", ".join(EXPERIMENTS))
print("Original /val and /test: NOT READ")

for fold_number, heldout_tile in enumerate(folds, 1):
    started = time.perf_counter()
    print(f"\n[{fold_number}/{len(folds)}] Holding out Tile: {heldout_tile}")

    # ----------------------------------------------------------
    # 1) Fit data contains no labeled pixel from heldout_tile.
    # ----------------------------------------------------------
    sample, fit_tiles = make_fold_training_sample(heldout_tile)
    if heldout_tile in fit_tiles:
        raise AssertionError("Tile group appears in both train and evaluation")
    x_all_fit = make_features(sample)
    y_fit = sample["Class"].to_numpy()
    confidence = sample["Confidence"].astype(str) if "Confidence" in sample else pd.Series(["High"] * len(sample))
    weights = confidence.map({"High": 1.0, "Moderate": 0.5, "Low": 1/3,
                              "1": 1.0, "2": 0.5, "3": 1/3}).fillna(1.0).to_numpy()
    fit_class_count = len(np.unique(y_fit))
    if fit_class_count < 2:
        raise ValueError("Need at least two fit classes for Tile " + heldout_tile)

    models = {}
    for name, columns in EXPERIMENTS.items():
        x_fit = x_all_fit[columns]
        medians = x_fit.median().fillna(0)
        x_fit = x_fit.fillna(medians).fillna(0)
        model = RandomForestClassifier(n_estimators=NUMBER_OF_TREES, min_samples_leaf=2,
                                       random_state=RANDOM_SEED, n_jobs=CPU_JOBS)
        model.fit(x_fit, y_fit, sample_weight=weights)
        models[name] = (model, columns, medians)

    # ----------------------------------------------------------
    # 2) Only the excluded Tile is scored, on ALL labeled pixels.
    #    Models and training medians are frozen during evaluation.
    # ----------------------------------------------------------
    matrices = {name: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
                for name in EXPERIMENTS}
    scenes = {name: defaultdict(lambda: np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64))
              for name in EXPERIMENTS}
    eval_tile_names = set()

    with pd.HDFStore(DATA_FILE, "r") as store:
        for chunk in store.select("train", chunksize=CHUNK_SIZE):
            tile = chunk["Tile"].astype(str).str.strip().str.upper()
            labels = clean_labels(chunk["Class"])
            mask = (tile == heldout_tile) & np.isin(labels, CLASSES)
            if not np.any(mask):
                continue
            part = chunk.loc[mask].copy()
            y_true = labels[mask]
            eval_tile_names.update(tile[mask].unique())
            features = make_features(part)
            date = part["Date"].astype(str).str.strip().str.lower()
            scene_names = (date + "|" + heldout_tile).to_numpy()

            for name, (model, columns, medians) in models.items():
                x_eval = features[columns].fillna(medians).fillna(0)
                y_pred = model.predict(x_eval)
                matrices[name] += confusion_matrix(y_true, y_pred, labels=CLASSES)

                # Preserve within-Tile acquisition differences (Date + Tile).
                for scene_id in np.unique(scene_names):
                    which = scene_names == scene_id
                    scenes[name][scene_id] += confusion_matrix(
                        y_true[which], y_pred[which], labels=CLASSES)

    if eval_tile_names != {heldout_tile}:
        raise AssertionError("Unexpected Tile(s) appeared in excluded Tile evaluation")

    expected = plan.loc[plan.tile == heldout_tile].iloc[0]
    for name in EXPERIMENTS:
        cm = matrices[name]
        score = scores_from_confusion(cm)
        if score["labeled_pixels"] != int(expected["labeled_pixels"]):
            raise ValueError("Audit/data mismatch for " + heldout_tile + ": run 13 again.")
        if score["true_debris"] != int(expected["marine_debris_pixels"]):
            raise ValueError("Debris count mismatch for " + heldout_tile + ": run 13 again.")
        if (sum(scenes[name].values(), np.zeros_like(cm)) != cm).any():
            raise AssertionError("Scene counts did not add up to Tile confusion")
        all_confusions[name] += cm
        fold_rows.append({"fold": fold_number, "heldout_tile": heldout_tile,
                          "experiment": name, "features": len(EXPERIMENTS[name]),
                          "fit_tile_count": len(fit_tiles),
                          "fit_sampled_pixels": len(sample),
                          "fit_sampled_classes": fit_class_count,
                          "shared_train_eval_tile_count": len(set(fit_tiles) & eval_tile_names),
                          **score})
        for true_name in CLASSES[1:]:
            class_error_rows.append({"heldout_tile": heldout_tile, "experiment": name,
                                     "true_class": true_name,
                                     "false_positive_debris_pixels": int(cm[CLASSES.index(true_name), 0]),
                                     "true_class_support": int(cm[CLASSES.index(true_name), :].sum())})
        for scene_id, scene_cm in scenes[name].items():
            scene_rows.append({"heldout_tile": heldout_tile, "scene_id": scene_id,
                               "experiment": name, **scores_from_confusion(scene_cm)})

    completed_tiles.append(heldout_tile)
    print("  TRAIN Tiles:", len(fit_tiles), "| Sampled pixels:", len(sample),
          "| Holdout pixels:", scores_from_confusion(matrices["full_19"])["labeled_pixels"])
    for name in EXPERIMENTS:
        s = scores_from_confusion(matrices[name])
        f1_text = f"{s['f1']:.4f}" if np.isfinite(s["f1"]) else "N/A (zero true debris)"
        print("  ", name, "F1:", f1_text, "TP:", s["tp"], "FP:", s["fp"], "FN:", s["fn"])
    print("  Fold duration (seconds):", round(time.perf_counter() - started, 1))

    # Save an incremental checkpoint after each completed Fold.
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(OUTPUT_DIR / "scene_metrics.csv", index=False)
    pd.DataFrame(class_error_rows).to_csv(OUTPUT_DIR / "false_positives_by_tile_and_true_class.csv", index=False)
    (OUTPUT_DIR / "holdout_run_status.json").write_text(
        json.dumps({"status": "running", "folds_completed": len(completed_tiles),
                    "folds_planned": len(folds), "completed_tiles": completed_tiles}, indent=2), encoding="utf-8")

# ----------------------------------------------------------
# 3) Aggregate genuinely out-of-fold predictions across Tiles.
# ----------------------------------------------------------
fold_df = pd.DataFrame(fold_rows)
summary_rows = []
for name in EXPERIMENTS:
    cm = all_confusions[name]
    # Each pixel is evaluated only when its own Tile is held out.
    # This is pooled, out-of-Tile-fold (not official MARIDA /val or /test).
    pooled = scores_from_confusion(cm)
    model_folds = fold_df.loc[fold_df.experiment == name]
    defined = model_folds[model_folds.true_debris > 0]
    summary_rows.append({
        "experiment": name,
        "features": len(EXPERIMENTS[name]),
        "evaluated_tiles": len(model_folds),
        "tiles_with_true_debris": len(defined),
        "tiles_without_true_debris": int((model_folds.true_debris == 0).sum()),
        "pooled_out_of_tile_f1": pooled["f1"],
        "pooled_out_of_tile_precision": pooled["precision"],
        "pooled_out_of_tile_recall": pooled["recall"],
        "pooled_out_of_tile_iou": pooled["iou"],
        "pooled_tp": pooled["tp"], "pooled_fp": pooled["fp"], "pooled_fn": pooled["fn"],
        "pooled_labeled_pixels": pooled["labeled_pixels"],
        "pooled_fp_per_1000_non_debris": pooled["fp_per_1000_non_debris"],
        "tile_macro_f1_mean_debris_tiles_only": defined["f1"].mean(),
        "tile_macro_f1_sd_debris_tiles_only": defined["f1"].std(ddof=1),
        "tile_min_f1_debris_tiles_only": defined["f1"].min(),
        "tile_max_f1_debris_tiles_only": defined["f1"].max(),
    })
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        OUTPUT_DIR / (name + "_pooled_out_of_tile_confusion.csv"))
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "geographic_holdout_summary.csv", index=False)

# Compare to the same 19-feature study, without selecting on official test.
base_f1 = float(summary.loc[summary.experiment == "full_19", "pooled_out_of_tile_f1"].iloc[0])
summary["delta_pooled_f1_vs_full_19"] = summary["pooled_out_of_tile_f1"] - base_f1
summary.to_csv(OUTPUT_DIR / "geographic_holdout_summary.csv", index=False)

# Descriptive charts: missing debris tiles have no defined positive-class F1.
chart = fold_df.pivot(index="heldout_tile", columns="experiment", values="f1")
chart = chart.reindex(columns=list(EXPERIMENTS))
ax = chart.plot(kind="bar", figsize=(max(10, 0.8 * len(chart) + 2), 5))
ax.set_ylim(0, 1.02)
ax.set_ylabel("Marine Debris F1 (undefined if no true debris)")
ax.set_xlabel("Held-out Tile from official TRAIN")
ax.set_title("Leave-One-Tile-Out | Debris F1 by unseen Tile")
ax.legend(title="Feature set", fontsize=8)
ax.grid(axis="y", alpha=0.2)
ax.figure.tight_layout()
ax.figure.savefig(FIGURE_DIR / "f1_by_heldout_tile.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(ax.figure)

fp_chart = fold_df.pivot(index="heldout_tile", columns="experiment", values="fp")
fp_chart = fp_chart.reindex(columns=list(EXPERIMENTS))
ax = fp_chart.plot(kind="bar", figsize=(max(10, 0.8 * len(fp_chart) + 2), 5))
ax.set_ylabel("Marine Debris false positives (counts, not rates)")
ax.set_xlabel("Held-out Tile from official TRAIN")
ax.set_title("Leave-One-Tile-Out | False positives by unseen Tile")
ax.legend(title="Feature set", fontsize=8)
ax.grid(axis="y", alpha=0.2)
ax.figure.tight_layout()
ax.figure.savefig(FIGURE_DIR / "fp_by_heldout_tile.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(ax.figure)

with (OUTPUT_DIR / "geographic_holdout_findings.md").open("w", encoding="utf-8") as f:
    f.write("# Geographic Holdout — Leave-One-Tile-Out (v0.3.2)\n\n")
    f.write("**Source:** official MARIDA `train` HDF5 split only. No `val` or `test` rows used.\n\n")
    f.write("**Protocol:** one Tile excluded for each fold. Three predeclared feature sets; 100-tree RF, ")
    f.write("confidence weights, same fit pixels within a fold, training-only median imputation.\n\n")
    f.write("| Model | Pooled OOF F1 | Tile macro F1 (debris-support tiles) | TP | FP | FN |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for _, r in summary.iterrows():
        f.write(f"| {r['experiment']} | {r['pooled_out_of_tile_f1']:.4f} | "
                f"{r['tile_macro_f1_mean_debris_tiles_only']:.4f} | "
                f"{int(r['pooled_tp'])} | {int(r['pooled_fp'])} | {int(r['pooled_fn'])} |\n")
    f.write("\nTile F1 is undefined if no true debris is annotated in the Tile. Such Tiles still ")
    f.write("contribute FP to pooled metrics. All 15 class labels remain in the confusion matrix.\n\n")
    f.write("**Caution:** Tiles differ in sizes/class prevalence; pooled and tile-macro F1 answer ")
    f.write("different questions. This is within TRAIN grouped cross-validation, not an untouched ")
    f.write("external geographic test. The earlier feature-set exploration used official validation, ")
    f.write("so this analysis should not be presented as wholly independent model-selection evidence. ")
    f.write("Tile IDs are geographic proxies; spatial footprints have not been independently confirmed.\n")

(OUTPUT_DIR / "holdout_run_status.json").write_text(
    json.dumps({"status": "complete", "folds_completed": len(folds),
                "folds_planned": len(folds), "all_eligible_folds_executed": config["all_eligible_folds_executed"],
                "completed_tiles": completed_tiles}, indent=2), encoding="utf-8")

print("\n===== GEOGRAPHIC HOLDOUT SUMMARY =====")
print(summary[["experiment", "pooled_out_of_tile_f1", "tile_macro_f1_mean_debris_tiles_only",
               "pooled_tp", "pooled_fp", "pooled_fn", "evaluated_tiles"]].to_string(index=False))
print("\nAll reported predictions use models that never trained on the corresponding Tile.")
print("These are NOT official validation/test results and not a spatially independent external test.")
print("Saved:", OUTPUT_DIR)
print("Figures:", FIGURE_DIR)
