# Geographic Holdout — Leave-One-Tile-Out (v0.3.2)

**Source:** official MARIDA `train` HDF5 split only. No `val` or `test` rows used.

**Protocol:** one Tile excluded for each fold. Three predeclared feature sets; 100-tree RF, confidence weights, same fit pixels within a fold, training-only median imputation.

| Model | Pooled OOF F1 | Tile macro F1 (debris-support tiles) | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| full_19 | 0.3018 | 0.5309 | 1639 | 7278 | 304 |
| minus_ndwi | 0.3504 | 0.5204 | 1625 | 5706 | 318 |
| bands_only | 0.2609 | 0.5527 | 1605 | 8755 | 338 |

Tile F1 is undefined if no true debris is annotated in the Tile. Such Tiles still contribute FP to pooled metrics. All 15 class labels remain in the confusion matrix.

**Caution:** Tiles differ in sizes/class prevalence; pooled and tile-macro F1 answer different questions. This is within TRAIN grouped cross-validation, not an untouched external geographic test. The earlier feature-set exploration used official validation, so this analysis should not be presented as wholly independent model-selection evidence. Tile IDs are geographic proxies; spatial footprints have not been independently confirmed.
