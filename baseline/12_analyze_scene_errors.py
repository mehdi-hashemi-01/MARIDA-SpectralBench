"""v0.3.1: inspect errors by acquisition scene, patch and geographic tile.

Run AFTER 11_multi_seed_robustness.py. This script trains no model.
No Test data, GeoTIFFs, or trained-model files are needed here.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import PROJECT_DIR

# ==============================================================
# 1. EDITABLE SETTINGS
# ==============================================================
REFERENCE_MODEL = "full_19"
COMPARE_MODEL = "minus_ndwi"
REFERENCE_SEED = 42
SCENE_BOOTSTRAP_REPEATS = 1000
SCENE_BOOTSTRAP_SEED = 2026
TOP_N = 12
SHOW_FIGURES_IN_PYCHARM = False

RESULT_DIR = PROJECT_DIR / "results" / "robustness"
FIGURE_DIR = PROJECT_DIR / "assets" / "figures" / "robustness"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
status_file = RESULT_DIR / "run_status.json"
if not status_file.is_file():
    raise FileNotFoundError("Run 11_multi_seed_robustness.py first (run_status.json is missing).")
status = json.loads(status_file.read_text(encoding="utf-8"))
if status.get("state") != "complete":
    raise RuntimeError("Step 11 has not completed successfully. Do not use incomplete or stale CSVs.")

needed = [
    "per_seed_validation.csv", "per_patch_metrics.csv",
    "per_scene_metrics.csv", "per_tile_metrics.csv",
]
for filename in needed:
    if not (RESULT_DIR / filename).is_file():
        raise FileNotFoundError("Missing " + filename + ". First run 11_multi_seed_robustness.py")

runs = pd.read_csv(RESULT_DIR / "per_seed_validation.csv")
patch = pd.read_csv(RESULT_DIR / "per_patch_metrics.csv")
scene = pd.read_csv(RESULT_DIR / "per_scene_metrics.csv")
tile = pd.read_csv(RESULT_DIR / "per_tile_metrics.csv")
if REFERENCE_MODEL not in runs["experiment"].unique() or COMPARE_MODEL not in runs["experiment"].unique():
    raise ValueError("Required experiment is missing. Re-run step 11 with all six feature sets.")
if REFERENCE_SEED not in runs["seed"].unique():
    raise ValueError("REFERENCE_SEED is missing from step 11: " + str(REFERENCE_SEED))

# Every validation pixel belongs to exactly one patch, scene and tile.
# Their TP/FP/FN sums must equal the full-validation totals.
for name, table in [("patch", patch), ("scene", scene), ("tile", tile)]:
    sums = table.groupby(["seed", "experiment"])[["tp", "fp", "fn"]].sum().reset_index()
    check = sums.merge(runs[["seed", "experiment", "tp", "fp", "fn"]],
                       on=["seed", "experiment"], suffixes=("_groups", "_full"),
                       validate="one_to_one")
    if len(check) != len(runs):
        raise ValueError("Missing per-" + name + " results: mismatch in number of runs.")
    for metric in ["tp", "fp", "fn"]:
        if not (check[metric + "_groups"] == check[metric + "_full"]).all():
            raise ValueError("Per-" + name + " " + metric + " totals do not match validation!")
print("Validated: patch, scene and tile TP/FP/FN totals all match full validation.")

overlap_file = RESULT_DIR / "train_val_group_overlap.csv"
if overlap_file.is_file():
    overlap = pd.read_csv(overlap_file)
    print("\nTRAIN/VAL GROUP OVERLAP (metadata proxy; not a leakage proof):")
    print(overlap.to_string(index=False))


def summarize_groups(source, id_columns):
    """Average each group's error counts over the repeated training seeds."""
    full = source[source["experiment"] == REFERENCE_MODEL]
    output = full.groupby(id_columns, as_index=False, dropna=False).agg(
        seeds=("seed", "nunique"),
        labeled_pixels=("labeled_pixels", "first"),
        true_debris_pixels=("true_debris_pixels", "first"),
        sediment_labeled_pixels=("sediment_labeled_pixels", "first"),
        mean_tp=("tp", "mean"),
        mean_fp=("fp", "mean"), sd_fp=("fp", "std"),
        mean_fn=("fn", "mean"),
        mean_sediment_fp=("sediment_fp", "mean"),
        mean_ship_fp=("ship_fp", "mean"),
        mean_cloud_fp=("cloud_fp", "mean"),
        mean_wakes_fp=("wakes_fp", "mean"),
        mean_f1_when_defined=("f1", "mean"),
        seeds_with_defined_f1=("f1", "count"),
    )
    output["sd_fp"] = output["sd_fp"].fillna(0)
    # In addition to raw counts, show normalized error rates for fairer comparisons
    # of groups that contain different numbers of labeled pixels.
    output["mean_fp_per_1000_labeled"] = 1000 * output["mean_fp"] / output["labeled_pixels"]
    output["mean_sediment_fp_rate"] = output["mean_sediment_fp"] / output["sediment_labeled_pixels"].replace(0, np.nan)
    output["mean_fn_rate_among_debris"] = output["mean_fn"] / output["true_debris_pixels"].replace(0, np.nan)
    return output.sort_values(["mean_fp", "mean_sediment_fp"], ascending=False)


print("STEP 1/3 | Finding problematic patches, acquisition groups, and tiles...")
patch_summary = summarize_groups(patch, ["patch_id", "scene_id", "date", "tile", "image"])
scene_summary = summarize_groups(scene, ["scene_id", "date", "tile"])
tile_summary = summarize_groups(tile, ["tile"])
patch_summary.to_csv(RESULT_DIR / "patch_error_summary.csv", index=False)
scene_summary.to_csv(RESULT_DIR / "scene_error_summary.csv", index=False)
tile_summary.to_csv(RESULT_DIR / "tile_error_summary.csv", index=False)


def compare_groups(source, group_key, detail_columns):
    """Compare full and minus-NDWI ON THE SAME seed and SAME group."""
    columns = ["seed", group_key, "tp", "fp", "fn", "sediment_fp"]
    a = source.loc[source["experiment"] == REFERENCE_MODEL, columns].copy()
    b = source.loc[source["experiment"] == COMPARE_MODEL, columns].copy()
    joined = a.merge(b, on=["seed", group_key], suffixes=("_full", "_minus_ndwi"),
                     validate="one_to_one")
    if len(joined) != len(a) or len(joined) != len(b):
        raise ValueError("Some groups are missing from one experiment.")
    for metric in ["tp", "fp", "fn", "sediment_fp"]:
        joined["delta_" + metric] = joined[metric + "_minus_ndwi"] - joined[metric + "_full"]

    info = source[source["experiment"] == REFERENCE_MODEL]
    info = info[[group_key] + detail_columns].drop_duplicates(subset=[group_key])
    changes = joined.groupby(group_key, as_index=False).agg(
        mean_delta_tp=("delta_tp", "mean"),
        mean_delta_fp=("delta_fp", "mean"),
        mean_delta_fn=("delta_fn", "mean"),
        mean_delta_sediment_fp=("delta_sediment_fp", "mean"),
        seeds=("seed", "count"),
    ).merge(info, on=group_key, validate="one_to_one")
    return changes.sort_values("mean_delta_sediment_fp"), joined


scene_changes, scene_pairs = compare_groups(scene, "scene_id", ["date", "tile"])
patch_changes, _ = compare_groups(patch, "patch_id", ["scene_id", "date", "tile", "image"])
scene_changes.to_csv(RESULT_DIR / "scene_ndwi_tradeoff.csv", index=False)
patch_changes.to_csv(RESULT_DIR / "patch_ndwi_tradeoff.csv", index=False)

print("\nMost false-positive acquisition groups (averaged over seeds):")
print(scene_summary[["scene_id", "labeled_pixels", "mean_fp", "mean_fn",
                     "mean_sediment_fp"]].head(10).round(2).to_string(index=False))
print("\nHighest FP rates PER 1000 LABELED pixels (only groups with >= 100 labeled pixels):")
large = scene_summary.loc[scene_summary["labeled_pixels"] >= 100]
print(large.sort_values("mean_fp_per_1000_labeled", ascending=False)[
    ["scene_id", "labeled_pixels", "mean_fp_per_1000_labeled", "mean_fp"]
].head(10).round(2).to_string(index=False))

print("\nLargest mean REDUCTIONS in sediment FP after removing NDWI:")
print(scene_changes[["scene_id", "mean_delta_sediment_fp", "mean_delta_fp",
                     "mean_delta_fn"]].head(10).round(2).to_string(index=False))
print("Negative change = fewer errors without NDWI; positive = more errors.")

# ==============================================================
# 2. PAIRED BOOTSTRAP BY ACQUISITION GROUP (Date + Tile)
# ==============================================================
print("\nSTEP 2/3 | Descriptive, paired SCENE bootstrap (seed", REFERENCE_SEED, ")...")
selected = scene_pairs.loc[scene_pairs["seed"] == REFERENCE_SEED].reset_index(drop=True)
if len(selected) < 2:
    raise ValueError("At least two acquisition groups are needed for scene bootstrap.")

# Resample WHOLE Date+Tile groups, never individual correlated pixels.
# The identical sampled group indices are used for both compared models.
# This reflects sensitivity to the composition of THIS validation split,
# NOT an independent out-of-region test or a formal population confidence interval.
rng = np.random.default_rng(SCENE_BOOTSTRAP_SEED)
n_scenes = len(selected)
indices = rng.integers(0, n_scenes, size=(SCENE_BOOTSTRAP_REPEATS, n_scenes))


def bootstrap_f1(suffix):
    tp = selected["tp_" + suffix].to_numpy(dtype=float)[indices].sum(axis=1)
    fp = selected["fp_" + suffix].to_numpy(dtype=float)[indices].sum(axis=1)
    fn = selected["fn_" + suffix].to_numpy(dtype=float)[indices].sum(axis=1)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.full_like(tp, np.nan),
                     where=denominator > 0)


f1_full = bootstrap_f1("full")
f1_no_ndwi = bootstrap_f1("minus_ndwi")
paired_difference = f1_no_ndwi - f1_full

# Report the actual (un-resampled) validation scores for seed 42, and
# descriptive percentiles of the group-resampling distribution.
seed_runs = runs.set_index(["seed", "experiment"])
actual_full = float(seed_runs.loc[(REFERENCE_SEED, REFERENCE_MODEL), "f1"])
actual_no_ndwi = float(seed_runs.loc[(REFERENCE_SEED, COMPARE_MODEL), "f1"])


def ci_values(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)), len(x)


bootstrap_rows = []
for label, actual, values in [
    (REFERENCE_MODEL, actual_full, f1_full),
    (COMPARE_MODEL, actual_no_ndwi, f1_no_ndwi),
    ("delta_minus_ndwi_minus_full", actual_no_ndwi - actual_full, paired_difference),
]:
    low, high, valid_count = ci_values(values)
    bootstrap_rows.append({
        "statistic": label, "actual_full_validation_value": actual,
        "bootstrap_mean": float(np.nanmean(values)),
        "bootstrap_2_5_percentile": low, "bootstrap_97_5_percentile": high,
        "valid_replicates": valid_count,
        "bootstrap_repeats": SCENE_BOOTSTRAP_REPEATS,
        "groups_resampled": n_scenes,
        "grouping": "Date+Tile", "seed": REFERENCE_SEED,
        "interpretation": "descriptive within-validation group resampling; not out-of-scene generalization",
    })
bootstrap_table = pd.DataFrame(bootstrap_rows)
bootstrap_table.to_csv(RESULT_DIR / "scene_bootstrap_descriptive.csv", index=False)
print("\nScene bootstrap descriptives (not a new held-out-test result):")
print(bootstrap_table[["statistic", "actual_full_validation_value",
                       "bootstrap_2_5_percentile", "bootstrap_97_5_percentile"]]
      .round(4).to_string(index=False))

# ==============================================================
# 3. FIGURES AND AN AUTO-GENERATED FACTUAL MARKDOWN SUMMARY
# ==============================================================
print("\nSTEP 3/3 | Saving graphs and research notes...")
top = scene_summary.head(TOP_N).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, max(5, 0.46 * len(top) + 1.3)))
ax.barh(top["scene_id"], top["mean_fp"], label="All false positives")
ax.barh(top["scene_id"], top["mean_sediment_fp"], label="From sediment-laden water")
ax.set_xlabel("Mean false-positive pixels across training seeds")
ax.set_title("MARIDA | Highest-error acquisition groups (Date + Tile)")
ax.legend()
ax.grid(axis="x", alpha=0.18)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "top_error_scenes.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

# Display the 12 groups where deleting NDWI changes sediment errors most.
selected_changes = scene_changes.copy()
selected_changes["abs_change"] = selected_changes["mean_delta_sediment_fp"].abs()
selected_changes = selected_changes.sort_values("abs_change", ascending=False).head(TOP_N)
selected_changes = selected_changes.sort_values("mean_delta_sediment_fp")
fig, ax = plt.subplots(figsize=(11, max(5, 0.46 * len(selected_changes) + 1.3)))
ax.barh(selected_changes["scene_id"], selected_changes["mean_delta_sediment_fp"])
ax.axvline(0, linewidth=1)
ax.set_xlabel("Delta sediment FP (without NDWI minus full 19 features)")
ax.set_title("MARIDA | Scene-level trade-offs when removing NDWI")
ax.grid(axis="x", alpha=0.18)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "scene_ndwi_sediment_tradeoff.png", dpi=170)
if SHOW_FIGURES_IN_PYCHARM:
    plt.show()
plt.close(fig)

valid_diff = paired_difference[np.isfinite(paired_difference)]
if len(valid_diff):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valid_diff, bins=30)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("F1 difference (without NDWI - full 19)")
    ax.set_ylabel("Number of resampled validation-group sets")
    ax.set_title("MARIDA | Descriptive paired acquisition-group bootstrap")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "scene_bootstrap_delta_f1.png", dpi=170)
    if SHOW_FIGURES_IN_PYCHARM:
        plt.show()
    plt.close(fig)

summary = pd.read_csv(RESULT_DIR / "summary_by_model.csv")
reference = summary.set_index("experiment").loc[REFERENCE_MODEL]
comparison = summary.set_index("experiment").loc[COMPARE_MODEL]
lo, hi, _ = ci_values(paired_difference)
lines = [
    "# Robustness and acquisition-group error analysis (v0.3.1)",
    "",
    "**Automatically generated from actual `dataset.h5` validation evaluations.**",
    "No results from the held-out test split are included.",
    "",
    "## Repeated-seed results",
    "",
    f"- Seeds evaluated: {sorted(runs['seed'].unique().tolist())}.",
    f"- Full 19-feature RF: mean Marine Debris F1 = {reference['mean_f1']:.4f} "
    f"(SD across seeds = {reference['sd_f1']:.4f}).",
    f"- Without NDWI: mean Marine Debris F1 = {comparison['mean_f1']:.4f} "
    f"(SD across seeds = {comparison['sd_f1']:.4f}).",
    f"- Full model mean FP = {reference['mean_fp']:.1f}; "
    f"without NDWI mean FP = {comparison['mean_fp']:.1f}.",
    "",
    "## Acquisition-group findings",
    "",
    f"- Number of validation Date+Tile acquisition groups = {len(scene_summary)}.",
    f"- Number of distinct validation patches = {len(patch_summary)}.",
    f"- Number of distinct geographic tile identifiers = {len(tile_summary)}.",
    "- Group-level F1 is undefined (NaN) when there is neither true nor predicted debris.",
    "- Mean group error counts are over training seeds, **not independent replications of scenes**.",
    "",
    "Highest average false-positive acquisition groups (full model):",
    "",
    "| Date+Tile group | Mean FP | Mean sediment FP | Mean FN |",
    "|:--|--:|--:|--:|",
]
for row in scene_summary.head(5).itertuples(index=False):
    lines.append(f"| {row.scene_id} | {row.mean_fp:.1f} | "
                 f"{row.mean_sediment_fp:.1f} | {row.mean_fn:.1f} |")
lines.extend([
    "", "## Descriptive scene-resampling check",
    "",
    f"With seed {REFERENCE_SEED}, resampling {n_scenes} complete Date+Tile groups "
    f"({SCENE_BOOTSTRAP_REPEATS} paired bootstrap replicates) gives a descriptive "
    f"95% percentile range [{lo:.4f}, {hi:.4f}] for the validation F1 difference "
    "(without NDWI minus full model).",
    "",
    "**Important limitations:** This percentile range describes sensitivity to the "
    "acquisition-group mix in the existing validation data; it is not a formal "
    "confidence interval for performance on new regions, dates, or the independent test set. "
    "Date+Tile and Tile are metadata-based grouping proxies; they do not demonstrate "
    "that the original train/validation split is free of spatial or temporal overlap. "
    "See `train_val_group_overlap.csv` to inspect exact shared Date+Tile and Tile IDs "
    "across the official splits; missing shared IDs would still not prove spatial independence. "
    "Seed variation jointly changes the training sample and Random Forest randomness. "
    "The validation split is fixed, and this is not grouped cross-validation.",
    "",
    "## Output tables",
    "",
    "- `summary_by_model.csv`: across-seed model comparison",
    "- `paired_summary.csv`: within-seed differences from the full model",
    "- `scene_error_summary.csv`, `patch_error_summary.csv`, `tile_error_summary.csv`: "
    "where the remaining errors concentrate",
    "- `scene_ndwi_tradeoff.csv`, `patch_ndwi_tradeoff.csv`: "
    "which groups improve or worsen after removing NDWI",
    "- `scene_bootstrap_descriptive.csv`: paired acquisition-group resampling",
    "",
])
(RESULT_DIR / "robustness_findings.md").write_text("\n".join(lines), encoding="utf-8")

print("\nSaved results:", RESULT_DIR)
print("Saved figures:", FIGURE_DIR)
print("Generated scientific report: robustness_findings.md")
print("TEST was never read. Grouping is descriptive; no independent new-scene evaluation was performed.")
