# Baseline Experiment Notes

## Completed experiments

| Run | Features | Validation Marine Debris F1 | Validation IoU | Held-out test F1 |
|:--|:--|--:|--:|--:|
| Spectral-only RF | 11 satellite bands | 0.795708 | 0.660727 | Not evaluated |
| Spectral + indices RF | 11 bands and 8 indices | 0.815723 | 0.688794 | 0.677871 |

The second model is the validation-selected candidate. The test set contains 194,863 labeled pixels (381 in the Marine Debris class). For that class, the test confusion matrix records 363 TP, 327 FP, and 18 FN. Test recall is 0.952756; precision is 0.526087.

## Feature dependence

Permutation analysis on a classwise sampled validation subset indicates the largest F1 decreases for NDMI (0.056685), NDWI (0.044725), and FDI (0.039194). These changes are not causal attributions, and shared source bands make feature importance estimates correlated.

## Error clusters

The largest sources of false-positive Marine Debris predictions are waves (103), clouds (72), and ships (66). This motivates a future investigation of spatial/contextual modeling. These error analyses are descriptive and are not additional model-selection criteria.

## Research questions for subsequent work

1. Can spatial context reduce debris false positives while retaining high recall?
2. How stable are the feature importance findings across multiple random seeds and geographically distinct scenes?
3. Do remote-sensing pretrained encoders offer reproducible gains over a strong classical spectral baseline?
4. Can retrieved evidence support better-calibrated and auditable predictions?

**Status:** questions are prospective and no outcomes for these extensions are claimed here.
