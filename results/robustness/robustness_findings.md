# Robustness and acquisition-group error analysis (v0.3.1)

**Automatically generated from actual `dataset.h5` validation evaluations.**
No results from the held-out test split are included.

## Repeated-seed results

- Seeds evaluated: [7, 21, 42, 123, 2026].
- Full 19-feature RF: mean Marine Debris F1 = 0.8232 (SD across seeds = 0.0107).
- Without NDWI: mean Marine Debris F1 = 0.8061 (SD across seeds = 0.0055).
- Full model mean FP = 300.4; without NDWI mean FP = 330.8.

## Acquisition-group findings

- Number of validation Date+Tile acquisition groups = 12.
- Number of distinct validation patches = 328.
- Number of distinct geographic tile identifiers = 7.
- Group-level F1 is undefined (NaN) when there is neither true nor predicted debris.
- Mean group error counts are over training seeds, **not independent replications of scenes**.

Highest average false-positive acquisition groups (full model):

| Date+Tile group | Mean FP | Mean sediment FP | Mean FN |
|:--|--:|--:|--:|
| 11-6-18|16pcc | 126.0 | 100.8 | 1.0 |
| 18-9-20|16pcc | 87.6 | 0.0 | 76.4 |
| 18-1-18|48pzc | 20.4 | 0.0 | 1.6 |
| 12-1-17|16pec | 16.6 | 0.0 | 0.0 |
| 29-12-20|18qyf | 13.8 | 0.0 | 1.4 |

## Descriptive scene-resampling check

With seed 42, resampling 12 complete Date+Tile groups (1000 paired bootstrap replicates) gives a descriptive 95% percentile range [-0.0372, 0.0380] for the validation F1 difference (without NDWI minus full model).

**Important limitations:** This percentile range describes sensitivity to the acquisition-group mix in the existing validation data; it is not a formal confidence interval for performance on new regions, dates, or the independent test set. Date+Tile and Tile are metadata-based grouping proxies; they do not demonstrate that the original train/validation split is free of spatial or temporal overlap. See `train_val_group_overlap.csv` to inspect exact shared Date+Tile and Tile IDs across the official splits; missing shared IDs would still not prove spatial independence. Seed variation jointly changes the training sample and Random Forest randomness. The validation split is fixed, and this is not grouped cross-validation.

## Output tables

- `summary_by_model.csv`: across-seed model comparison
- `paired_summary.csv`: within-seed differences from the full model
- `scene_error_summary.csv`, `patch_error_summary.csv`, `tile_error_summary.csv`: where the remaining errors concentrate
- `scene_ndwi_tradeoff.csv`, `patch_ndwi_tradeoff.csv`: which groups improve or worsen after removing NDWI
- `scene_bootstrap_descriptive.csv`: paired acquisition-group resampling
