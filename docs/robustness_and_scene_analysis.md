# v0.3.1 — Multi-seed robustness and scene-level error analysis

This stage extends the finished **v0.3 feature ablation** experiments. It does not modify the models or CSVs saved in `results/ablation/` and does not evaluate the held-out test split.

## 1. Research questions

1. Are the feature-ablation conclusions stable when the training pixel sample and Random Forest random state change?
2. Do false-positive Marine Debris predictions concentrate in a small number of **patches** or **acquisition groups**, rather than being distributed evenly?
3. Does removing NDWI lower sediment-related false positives consistently, or only in one previously highlighted patch?
4. Are there identical patch, acquisition (`Date + Tile`), or geographic tile IDs shared between the official **train and validation** tables?

## 2. Repeated-seed protocol (`11_multi_seed_robustness.py`)

Open this file in **PyCharm** and click **Run**. No command-line arguments or custom classes are needed.

- Default seeds: `42, 7, 21, 123, 2026`.
- For **each seed**, construct a fresh class-stratified sample from the official `train` HDF5 table using the same random-key and per-class limit method as `02_train_random_forest.py` and `09_feature_ablation.py`.
- For that seed, train **all six** feature groups using **exactly the same sampled training rows**. This makes the ablation comparisons **paired within seed**.
- Use the same RF configuration as v0.3: `n_estimators=100`, `min_samples_leaf=2`, `sample_weight` from annotation confidence, and the training-only median imputation.
- Evaluate every fitted model on the *entire labeled* official validation HDF5 table. **No test rows are loaded.**
- No model files are saved; earlier model files under `models/` are untouched.
- Both training sampling and forest random state vary between seeds. Therefore the variability is for the **combined pipeline**, not a controlled isolation of RF randomness versus sampling randomness.

With the defaults, **5 seeds × 6 feature sets = 30 RF fits**. This may take significantly longer than step 09 on your computer. Random Forest uses CPU in this implementation; an NVIDIA GPU is not required. To run fewer repetitions for debugging, edit `SEEDS` at the top; restore all five for the final study.

### Group definitions

The script uses the HDF5 `Date`, `Tile`, and `Image` fields:

| Group | Identifier | What it represents |
|---|---|---|
| Patch | `Date + Tile + Image` | One image patch identifier |
| Acquisition group | `Date + Tile` | Patches from the same acquisition date and tile, a *proxy* for shared scene conditions |
| Geographic tile | `Tile` | A coarser proxy for geographic repeat observations across dates |

**These are metadata-based groupings, not guaranteed independent scenes.** They do not verify geolocation, patch overlap, or train/validation independence. Treat the latter as a separate question requiring spatially grouped holdout design.

### Output from step 11

All tables are written to `results/robustness/`; charts to `assets/figures/robustness/`.

| File | Interpretation |
|---|---|
| `per_seed_validation.csv` | Individual F1, precision, recall, IoU, TP, FP, FN and sediment FP for every seed/model |
| `summary_by_model.csv` | **Mean, standard deviation, minimum, maximum** F1 and mean/error variability over seeds |
| `paired_deltas_vs_full.csv` | Each feature group's differences from `full_19` using the **same seed** |
| `paired_summary.csv` | Mean paired differences across seeds |
| `per_patch_metrics.csv` | Prediction and error counts for each patch, seed, model |
| `per_scene_metrics.csv` | Same for acquisition groups (`Date + Tile`) |
| `per_tile_metrics.csv` | Same for geographic Tile identifiers |
| `false_positives_by_class.csv` | True class names of false Marine Debris predictions per seed/model |
| `train_val_group_overlap.csv` | How many group identifiers occur in both official Train and Validation |
| `train_val_shared_patch.csv`, `train_val_shared_scene_date_tile.csv`, `train_val_shared_tile.csv` | The corresponding shared IDs (if any) |
| `run_config.json`, `run_status.json` | Reproducibility settings and whether all requested fits completed |

For groups with neither actual nor predicted Marine Debris, F1 is **undefined** and saved as `NaN`, *not* artificially assigned zero.

### Plots

- `validation_f1_across_seeds.png`: whether model rankings stay stable for all five seeds.
- `validation_f1_mean_sd.png`: average F1 and ±1 standard deviation across seeds. **Seed SD is not a confidence interval** and does not establish statistical significance.

## 3. Scene and patch error analysis (`12_analyze_scene_errors.py`)

Run **after** step 11 finishes. It reads only saved CSVs: **no model retraining, no GeoTIFFs, no test data.** It first verifies that summing TP/FP/FN across patches, scenes, or tiles gives precisely the whole-validation counts for each seed/model. It then:

1. Ranks the patches, scenes, and tile identifiers producing the most errors for the full 19-feature model, averaging **counts over seeds**.
2. Reports `mean_fp_per_1000_labeled` alongside raw FP counts because scenes with more annotated pixels naturally contribute more errors.
3. Calculates `mean_sediment_fp_rate` (mean sediment FP divided by number of annotated sediment pixels) and `mean_fn_rate_among_debris` where the denominators are positive.
4. Compares `full_19` with `minus_ndwi` **for the same seed and same scene/patch**. A **negative** `mean_delta_sediment_fp` means the version *without NDWI* had fewer sediment false alarms.
5. Performs a **paired resampling of entire acquisition groups** for `seed=42`. This samples groups, rather than correlated individual pixels, and uses the *same sampled groups* for both models. It reports descriptive 2.5th and 97.5th percentiles for each validation F1 and the paired F1 difference.

That resampling describes how the scores vary when the mix of **the observed validation acquisition groups** changes. **It is not an independent geographic or temporal generalization test, nor a population-level 95% confidence interval.** Repeated `Tile` identifiers across dates may remain correlated even when groups are sampled by `Date + Tile`.

### Output from step 12

| File | Interpretation |
|---|---|
| `patch_error_summary.csv` | Patches ranked by average full-model false positives, with rates |
| `scene_error_summary.csv` | Acquisition groups ranked by average errors, with normalized rates |
| `tile_error_summary.csv` | Geographic tile error summary |
| `scene_ndwi_tradeoff.csv`, `patch_ndwi_tradeoff.csv` | How removing NDWI changes TP, FP, FN, and sediment FP per group |
| `scene_bootstrap_descriptive.csv` | Validation scene-resampling distributions for two paired models |
| `robustness_findings.md` | **Automatically generated from your actual run**, with key figures and limitations |

Saved figures:

- `top_error_scenes.png`
- `scene_ndwi_sediment_tradeoff.png`
- `scene_bootstrap_delta_f1.png`

## 4. Reading the results without misinterpreting them

**Question: Is the 19-feature model consistently better?** Compare `summary_by_model.csv` and `paired_summary.csv`. If paired F1 changes have different signs over seeds, avoid describing one feature's removal as consistently helpful or harmful.

**Question: Did NDWI cause sediment errors?** Compare `scene_ndwi_tradeoff.csv` (not only the single patch from v0.2). Improvement in one class or acquisition group can coexist with worse total validation F1. This is not a causal mechanism test.

**Question: Are errors clustered?** Check `scene_error_summary.csv`, the plot `top_error_scenes.png`, and normalized FP rates. Raw FP counts and rates answer different questions.

**Question: Is there split leakage?** `train_val_group_overlap.csv` reports whether the *same metadata identifiers* occur in both partitions. A shared patch ID is stronger evidence of potential nonindependence than a shared tile ID, but neither proves that training contains a particular validation pixel. Absence of shared IDs does not prove spatial independence. Confirming that would require geographic coordinates, overlap distances, and a dedicated scene/tile-disjoint evaluation protocol.

## 5. What this does NOT do

- It does **not** touch the held-out `test` split.
- It does **not** tune hyperparameters using test performance.
- It does **not** perform grouped K-fold cross-validation or create a genuinely geographically independent evaluation split.
- It does **not** fit U-Net or any other deep learning model.
- It does **not** transform five seed repetitions into independent validation scenes.

A natural next study is a spatially/temporally disjoint validation protocol followed by a spatial model such as U-Net, if the dataset grouping permits it.
