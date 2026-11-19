# v0.8 Protocol — Reproducibility, Geographic Generalization and Pretraining Control

**Research status:** Implementation provided; actual v0.8 MARIDA outputs must be generated on the local machine before any new scores are reported.

## Design choices

| Component | Locked v0.8 decision |
|---|---|
| Main label protocol | MARIDA raw 0 = unannotated/ignored; classes 1–15 become 0–14 |
| Primary metric | Pooled Marine Debris F1 from labeled-pixel 15×15 confusion |
| Reporting | Precision, recall, IoU, TP, FP, FN, patch support and class confusion |
| Multi-seed | Five independent Spectral Only fits on official Train/Validation |
| Geographic split | Outer leave-one-Tile-ID-out, **only official Train** |
| Inner selection | Held-out complete Date+Tile acquisition groups from outer-fit portion |
| Geographic checkpoint selection | **Inner validation only**, no outer Fold metrics used in selection |
| Geographic normalization/weights | **Inner-fitting only** for every held-out Tile |
| Geographic models | Spectral Only head, U-Net 11 bands, frozen DOFA + spectral-skip decoder |
| Geographic model seed | 42, same per fold/model |
| Geographic fit budget | 30 epochs maximum, patience 8, batch 4, AdamW 3e-4 |
| Spatial augmentation | Disabled for all three new geographic baselines; historical U-Net was different |
| Positive-patch sampling | 3× probability, replacement; only inner-fitting patches |
| DOFA geographic cache | Per outer-fold, generated with **inner-fitting normalization**, never reused from v0.6 Train-wide cache |
| Pretraining control | Frozen official pretrained versus frozen randomly initialized DOFA (DOFA tokens only, raw skip zeroed) |
| Pretraining decoder seeds | 42, 7, 21 (paired) |
| Official Test | **Never opened in v0.8** |

## Instructions for PyCharm

1. Extract the ZIP in `C:\Users\Mehti\Documents\GitHub\` so that the new files appear under `MARIDA-SpectralBench\baseline\`.
2. Run `27_multiseed_spectral_v08.py`; review `results/v08/multiseed/`.
3. Run `28_tile_holdout_v08.py` with `FOLDS_TO_RUN='all'` and all three models. A full geographic benchmark requires all Fold×Model cells. You can diagnose a single fold first, e.g. `FOLDS_TO_RUN=('16PDC',)`; the script will not pretend that a partial run is complete. Output resumes from completed fold/model results.
4. Run `29_random_dofa_control_v08.py` (optional but recommended for a research publication). The first run needs local Torch Hub code and a cached v0.6 pretrained DOFA feature store. It creates a distinct, fixed random-feature cache.
5. Run `30_research_report_v08.py` at any point to write/update `docs/MARIDA_v08_technical_research_report.md` from outputs that actually exist.

All scripts run with the existing `dl` Python environment and the RTX 5070 Laptop GPU (when available); no command-line flags required. Keep `timm==0.9.2`, `huggingface_hub`, `torch`, `tifffile`, `numpy`, `pandas`, and `matplotlib` from previous stages. **Do not update the CUDA/PyTorch installation just for v0.8.**

## Expected outputs

```text
baseline/
├── 27_multiseed_spectral_v08.py
├── 28_tile_holdout_v08.py
├── 29_random_dofa_control_v08.py
├── 30_research_report_v08.py
└── v08_research_core.py
results/v08/
├── multiseed/{per_seed_validation.csv,multiseed_summary.json,seed_*/...}
├── geographic/{per_fold_metrics.csv,pooled_metrics.csv,16PDC/...,...}
└── pretraining_control/{paired_per_seed.csv,paired_summary.json,...}
assets/figures/v08/
├── multiseed_spectral_f1.png
├── geographic_pooled_f1.png
└── random_vs_pretrained_f1.png
docs/
├── v08_reproducibility_and_geographic_protocol.md
└── MARIDA_v08_technical_research_report.md
```

The `geographic/pooled_metrics.csv` file is deliberately **not written** until all thirteen outer folds and all three models finish. Each outer Tile's metrics are written only once the model's best epoch has been selected from its inner validation set. Every existing fold output includes a frozen checkpoint under `results/v08/...` for local reproducibility. Check `.gitignore`; exclude binary `.pt` weights and large feature caches.

## Necessary caveats

The grouped folds are based on MARIDA Tile-ID **names** rather than audited georeferenced footprints; this is stronger than arbitrary random splitting by patch, but not a verified spatially buffered independent dataset. Sparse annotations and rare positive pixels make per-Tile F1 unstable. The old official Test split was inspected in v0.1/v0.4; it must not be marketed as untouched for v0.8 confirmatory testing. Several models use validation-selected best Epochs; performance on that validation split is exploratory. No v0.8 scores should appear in the GitHub README before generating and verifying actual results.

Random frozen DOFA control tests the utility of *frozen pretrained representations versus frozen random ones* with the same head and preprocessing. It does **not** test full pretraining benefit under LoRA, fine-tuning or geographic holdout. The three geographic models have different parameter counts and old versions used different augmentation and training budgets: avoid architecture-only causal conclusions.

## External research references

- MARIDA: https://doi.org/10.1371/journal.pone.0262247
- DOFA: https://github.com/zhu-xlab/DOFA and https://arxiv.org/abs/2403.15356
- Spatially structured validation: Roberts et al. (2017), https://doi.org/10.1111/ecog.02881
