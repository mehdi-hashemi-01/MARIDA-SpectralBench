# v0.3 — Spectral Feature Ablation (Validation Only)

## Question

Do NDMI, NDWI, and FDI improve Marine Debris detection when the Random Forest is **retrained without each index**, and what happens to false positives caused by sediment-laden water?

This is a controlled **retraining** experiment, not permutation feature importance. The indices are computed deterministically from the 11 Sentinel-2 bands, so removing an index does not remove the underlying spectral information from the other features. Ablation changes how a fixed model family uses its inputs, not proof of a physical mechanism.

## Experimental design

Run the scripts directly in PyCharm. No `argparse`, command-line options, or custom classes are required.

1. `baseline/09_feature_ablation.py` selects a **single**, reproducible, per-class training sample from the `train` split using exactly the procedure in `02_train_random_forest.py` (`RANDOM_SEED`, `TRAIN_SAMPLES_PER_CLASS`, and HDF5 chunk processing).
2. Six new Random Forest models are fit from scratch with the same sampled pixels, confidence weights, tree count, `min_samples_leaf=2`, random state, and CPU configuration.
3. Feature-specific train medians fill non-finite values; validation uses those saved train medians without refitting the preprocessing.
4. **All labeled validation pixels** are evaluated in chunks for each model using the same 15-class prediction task. No test rows are accessed.
5. `baseline/10_compare_ablation_patches.py` compares the six saved models on the known sediment-rich **validation** patch `S2_11-6-18_16PCC_26.tif`. Only labeled pixels from the class mask are scored; class 0 is excluded.

| Experiment | Input features | Count |
|:--|:--|--:|
| `full_19` | 11 Sentinel-2 bands + 8 indices | 19 |
| `minus_ndmi` | All 19 except NDMI | 18 |
| `minus_ndwi` | All 19 except NDWI | 18 |
| `minus_fdi` | All 19 except FDI | 18 |
| `minus_three` | All 19 except NDMI, NDWI, and FDI | 16 |
| `bands_only` | 11 spectral bands only | 11 |

## Metrics and interpretation

The target metric is **Marine Debris F1**, computed from the pooled 15-class validation confusion matrix as a one-vs-rest score: `F1 = 2 TP / (2 TP + FP + FN)`. Precision, recall, IoU, true positives, false positives, and false negatives are recorded alongside it. The `sediment_laden_water_fp` field counts **true Sediment-Laden Water pixels predicted as Marine Debris** on the full validation split.

- `delta_f1_vs_full = model F1 - full_19 F1`. Positive means higher validation F1.
- `delta_fp_vs_full = model FP - full_19 FP`. Negative means fewer false-positive debris predictions.
- A lower false-positive count can come with lower recall. Compare **F1, precision, recall, FP, and FN together**.
- The `10` script is a **post hoc analysis of a known problematic patch**. It is illustrative and must not be used as the sole criterion to choose features.
- These scores are measured on the **full official validation split**, not the 60-patch diagnostic subset from v0.2, nor the previously inspected test split.

## Saved artifacts

`09_feature_ablation.py` produces:

- `results/ablation/run_config.json` (data split, random seed, RF settings and features)
- `results/ablation/training_sample_counts.csv`
- `results/ablation/validation_comparison.csv`
- `results/ablation/false_positives_by_true_class.csv`
- `results/ablation/<experiment>_val_confusion.csv` (six files)
- `assets/figures/ablation/validation_f1_by_feature_set.png`
- `assets/figures/ablation/sediment_false_positives.png`
- `models/ablation/<experiment>.joblib` (six local saved models; excluded from Git)

`10_compare_ablation_patches.py` produces:

- `results/ablation/known_validation_patch_comparison.csv`
- `assets/figures/ablation/known_validation_patch_false_positives.png`

The new output directories are independent of `results/benchmark/`, `results/runs/latest/`, and the old RF models.

## Reproduce

Before running: install the requirements, ensure `baseline/data/dataset.h5` exists, and retain matching parameters in `baseline/settings.py`. For step 10 also retain extracted MARIDA TIFF images under `baseline/data/patches/`.

1. Run `baseline/09_feature_ablation.py` in PyCharm. It trains six **new** models; the old `.joblib` files are not required.
2. Inspect `results/ablation/validation_comparison.csv` and the two saved figures. Note any discrepancy between the new `full_19` F1 and the archived reference (0.8157); discrepancies require checking data and configuration, not quietly replacing baseline results.
3. Run `baseline/10_compare_ablation_patches.py` to see what happens on the known sediment-rich validation patch.
4. Once actual results are generated, add measured numbers to the README with the validation-only split designation. Do **not** put synthetic test figures or unexecuted hypothetical numbers in the README.

## Important limitations

- The official test split was used in v0.1 and has already been inspected. **Do not iterate on it for feature selection.** Any later claim of final generalization should ideally use new independent, spatially or temporally separated evaluation data.
- This first ablation uses one fixed random seed. Improvements may not be stable across seeds, satellite scenes, or acquisition dates. Multiple-seed and scene-level uncertainty analysis should follow.
- Some spectral indices and source bands are correlated; ablation is **not** causal attribution.
- All numbers are for **annotated pixels**, not dense whole-image segmentation. Unlabeled pixels are ignored.
- The previous HDF5/GeoTIFF check covered 411 sampled records from three patches and did not verify their raster coordinates individually; it is not an archive-wide spatial audit.
