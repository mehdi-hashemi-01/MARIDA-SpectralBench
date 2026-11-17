## Geographic Holdout Validation (v0.3.2 — methodology)

To assess spatial generalization beyond the original pixel-level split, we implement a **Leave-One-Tile-Out (LOTO)** grouped validation experiment on the **official MARIDA training split only**. For each fold, an entire Sentinel-2 Tile ID is excluded from model fitting and used exclusively for evaluating that fold's trained classifier. The official validation and test splits are not accessed by these scripts.

The two-step workflow is documented in [Geographic Holdout Protocol](docs/geographic_holdout.md):

1. [`13_audit_geographic_tiles.py`](baseline/13_audit_geographic_tiles.py) — audit the distribution of the 15 annotated classes across Tiles, check per-Tile Marine Debris support, and prepare an outcome-independent plan.
2. [`14_geographic_holdout.py`](baseline/14_geographic_holdout.py) — retrain and score `full_19`, `minus_ndwi`, and `bands_only` Random Forest classifiers with no shared Tile identifiers between each fold's training and evaluation groups.

The reporting separates **pooled out-of-fold F1** from **mean F1 across debris-containing Tiles** and includes per-Tile false positives, per-acquisition errors, full 15-class confusion matrices, and class-specific error sources. Tiles with no annotated debris contribute false-positive evidence but have undefined positive-class F1.

**Status:** implementation ready; quantitative results from the user's original MARIDA data are **pending execution**. No synthetic smoke-test scores are research results. Tile identifiers are geographic grouping proxies, not independently verified spatial footprints. This grouped evaluation within the historical training split is a useful stress test, **not an untouched independent geographic test**, especially because feature families were explored in earlier experiments.

After a completed run, figures can be included as:

```markdown
![Debris pixels by Tile](assets/figures/geographic_holdout/train_debris_by_tile.png)
![Out-of-Tile-fold F1](assets/figures/geographic_holdout/f1_by_heldout_tile.png)
![False positives by held-out Tile](assets/figures/geographic_holdout/fp_by_heldout_tile.png)
```
