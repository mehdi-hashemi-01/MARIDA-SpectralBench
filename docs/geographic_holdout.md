# Geographic Holdout on MARIDA (v0.3.2)

## Research question

How well does a spectral Random Forest recognize annotated Marine Debris pixels when **all pixels from the evaluated Sentinel-2 Tile are absent from that fold's training set**?

This is a **geographic-proxy grouped cross-validation experiment** using Tile metadata, not an untouched external geographic test. Tile IDs are not independently verified as non-overlapping scene footprints. Because feature sets were already investigated using the official validation split in earlier work, this is an additional diagnostic assessment rather than independent confirmation of model selection.

## Data boundaries

- Source: official MARIDA `dataset.h5`, key `/train` only.
- Official `/val`: not accessed.
- Official `/test`: not accessed.
- Each record is evaluated once, when its own Tile is the held-out group, if its fold is eligible and all folds are run.
- Only labeled pixels in the 15 MARIDA classes contribute to metrics. Unlabeled pixels are not negatives.
- All models are retrained from scratch for every held-out Tile.

## Why use two steps?

### Script 13 — `baseline/13_audit_geographic_tiles.py`

Before training, the script inspects class support in each Tile, using `/train` exclusively. It creates:

| File | Contents |
|---|---|
| `results/geographic_holdout/tile_class_counts.csv` | One row for every Tile × class combination, including zero counts |
| `results/geographic_holdout/tile_class_counts_wide.csv` | Matrix of Tile × all 15 classes |
| `results/geographic_holdout/tile_summary.csv` | Total annotated pixels, debris pixels, scene and patch counts, fit feasibility |
| `results/geographic_holdout/tile_holdout_plan.csv` | One proposed held-out fold for every Tile with a trainable complement |
| `results/geographic_holdout/audit_config.json` | Audit protocol settings |
| `assets/figures/geographic_holdout/train_debris_by_tile.png` | Debris support across Tiles |
| `assets/figures/geographic_holdout/tile_class_support_heatmap.png` | Class support across Tiles (log color scale) |

A Tile with no Marine Debris annotations **remains in the experiment**, if the complementary training pool can fit a classifier: it is useful for measuring false positives but its debris F1 is undefined. Tiles with fewer than 20 debris pixels receive a warning for interpretation, not automatic exclusion. A fold is skipped only if its complementary training pool lacks Marine Debris or enough distinct classes.

### Script 14 — `baseline/14_geographic_holdout.py`

For each planned held-out Tile:

1. Load annotated training pixels from all **other** Tile IDs; verify the exclusion.
2. Select at most 2,000 fitting pixels per class, using the same random-key approach as the earlier baseline.
3. Calculate the same original 11 Sentinel-2 bands and eight spectral indices.
4. Train three predeclared 100-tree Random Forest variants with the same sampled fitting rows and annotation-confidence weights:
   - `full_19`: 11 bands + all eight indices.
   - `minus_ndwi`: full feature set excluding NDWI (18 features).
   - `bands_only`: 11 original bands.
5. Compute median imputation **from this fold's training sample only**. Use the same medians to transform its holdout Tile.
6. Predict on **all labeled pixels** from the held-out Tile. Aggregate 15-class confusion matrices, debris TP/FP/FN, precision, recall, F1 and IoU.
7. Record a secondary within-Tile grouping by `Date + Tile`, and false positives by true MARIDA class.
8. Repeat on every eligible Tile; never select a Tile after seeing its results.

No command-line parameters or class-based programming are used. Open each file in PyCharm and click **Run**. Run 13 before 14.

## Outputs

| Path | Description |
|---|---|
| `results/geographic_holdout/fold_metrics.csv` | F1, IoU, TP/FP/FN, held-out Tile, training group counts for each fold and feature set |
| `results/geographic_holdout/geographic_holdout_summary.csv` | Pooled out-of-fold and per-Tile macro metrics |
| `results/geographic_holdout/scene_metrics.csv` | Within-Tile (`Date + Tile`) error breakdown |
| `results/geographic_holdout/false_positives_by_tile_and_true_class.csv` | Sources of erroneous Marine Debris predictions |
| `results/geographic_holdout/*_pooled_out_of_tile_confusion.csv` | 15 × 15 combined out-of-Tile-fold confusion matrices |
| `results/geographic_holdout/holdout_run_config.json` | Exact settings and feature sets |
| `results/geographic_holdout/holdout_run_status.json` | `running` or `complete`, and fold counts |
| `results/geographic_holdout/geographic_holdout_findings.md` | Automatically produced initial Markdown summary |
| `assets/figures/geographic_holdout/f1_by_heldout_tile.png` | Positive-class F1 by unseen Tile; missing bars are undefined |
| `assets/figures/geographic_holdout/fp_by_heldout_tile.png` | Debris false positives in each unseen Tile |

`MAX_FOLDS = 0` in script 14 means all eligible Tile groups. Any positive value enables a quick partial check; **a partial run must not be published as full geographic cross-validation**.

## Interpreting scores responsibly

- **Pooled out-of-fold F1:** compute Marine Debris F1 from TP, FP and FN summed across held-out Tiles. Large Tiles and Tiles with many labeled pixels have more influence.
- **Tile-macro F1:** arithmetic mean of Tile-specific Marine Debris F1 **among Tiles with at least one annotated true debris pixel**. Each such Tile receives equal weight. Report the number of contributing Tiles.
- **Zero-debris Tile:** F1 and recall are *undefined*, not zero; any false positives still affect the pooled score. Report its false positives or FP per 1,000 labeled non-debris pixels.
- **Confidence:** annotation-confidence weights are applied during fitting; evaluation is an unweighted labeled-pixel count.
- **Group disjointness:** the training Tile set is compared with the held-out Tile ID in each fold and asserted disjoint.
- **Not an unseen external dataset:** this is cross-validation within the historical training data, not evidence that the detector is ready for new geographies.
- **Single seed:** the geographic study uses one fixed seed (42) by default. The previous v0.3.1 experiment assesses seed variability under the official split, not seed variability in geographic folds. These are complementary diagnostics.
- **Not directly comparable with the archived validation/test numbers:** data partitions, class prevalence, and training sample composition differ.

## Study status and Git

Actual MARIDA numerical findings should be inserted **only after running scripts 13 and 14 on the original dataset**. The implementation was smoke-tested with a small synthetic HDF5 table solely to verify logic, split disjointness, zero-debris Tile handling, and aggregation; synthetic metrics are **not MARIDA results**.

The original MARIDA data and `.joblib` files stay local and excluded from version control. Commit the source scripts, this document, the intended figures and selected lightweight CSV outputs once their paths are verified; check `.gitignore` before staging generated results.
