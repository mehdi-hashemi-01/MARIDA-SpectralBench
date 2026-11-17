# U-Net v0.5: Controlled Training Ablations (Validation Only)

## Why this experiment?

The v0.4 11-band, 15-class U-Net achieved **validation Marine Debris F1 = 0.8009**, but its previously inspected test F1 was **0.3851** (315 TP, 940 FP, 66 FN). These values come from the previous completed v0.4 run, **not** from v0.5.

We cannot conclude that a larger network or more augmentation will solve the generalization gap. First, compare **one training change at a time** using the official **TRAIN** and **VALIDATION** splits. Neither v0.5 script opens official TEST.

**Important correction about v0.4:** it ALREADY used all three of the following:
- Train-frequency-based per-class weights (capped at 8).
- A 3× probability weight for training patches with labeled Marine Debris.
- Random horizontal/vertical flips and 90-degree rotations on TRAIN images/masks.

Therefore v0.5 starts with a re-run of the v0.4 recipe; it does not present these existing techniques as newly introduced.

## Experiments

| Name | What changes? | What stays fixed? |
|---|---|---|
| `v04_replay` | Nothing in the intended training recipe; new run | 11 bands, 15 classes, network, seed 42, loss masking, patch sampling, augmentation, optimizer, scheduler |
| `no_class_weights` | Change class-frequency weights to 1.0 | Annotation-confidence weights and all other settings |
| `no_debris_oversampling` | Set training patch sampling weights to 1.0 | Replacement sampler, epoch sample count, class weights and augmentation |
| `extra_spectral_jitter` | Add train-only reflectance jitter (global ±3% gain + independent small 0.015×training-band-std Gaussian noise) | Class weights, sampling, geometric transforms and model |

The extra augmentation is exploratory, not a physically validated simulation of Sentinel-2 calibration. Images, labels and confidence masks undergo the original identical geometric transformations; spectral jitter modifies **only** the image tensor. The fixed seed controls initialization and sampling, but a single seed is **not** a statistical robustness study.

## File locations (no previous script needs replacing)

```text
MARIDA-SpectralBench/
├── baseline/
│   ├── settings.py
│   ├── unet_core.py           # v0.4, remains unchanged
│   ├── 15_check_gpu_and_data.py
│   ├── 16_train_unet.py      # v0.4, remains unchanged
│   ├── 17_evaluate_unet.py   # v0.4, remains unchanged
│   ├── 18_train_unet_v05.py         # NEW
│   └── 19_analyze_unet_v05.py       # NEW
├── docs/
│   └── unet_v05_controlled_ablation.md # NEW
├── models/
│   ├── unet_best.pt          # v0.4, NOT overwritten
│   └── unet_v05/             # NEW model checkpoints
├── results/
│   ├── unet/                 # v0.4 results, NOT overwritten
│   └── unet_v05/             # NEW study outputs
└── assets/
    └── figures/
        └── unet_v05/        # NEW charts and comparison maps
```

### Before running

1. Open `MARIDA-SpectralBench` in PyCharm and select the SAME Python interpreter where v0.4 and RTX 5070 worked.
2. Ensure the original `baseline/data/patches/`, `baseline/data/splits/train_X.txt`, and `val_X.txt` are present.
3. Keep the existing `baseline/unet_core.py` and `baseline/settings.py` untouched.
4. The PyTorch CUDA setup is already working if script `15_check_gpu_and_data.py` passed. No additional deep-learning packages are needed beyond the v0.4 setup.

### Step 1: run `18_train_unet_v05.py` in PyCharm

- Default: 4 runs; each has up to 35 epochs, batch size 4, patience 10.
- The script **does not read test data**. It uses the full official training and validation manifests.
- Only the **TRAIN** data determine spectral normalization, class counts and loss weights.
- Unlabeled value `0` is converted to ignore index `255`; it never contributes to the loss or classification counts.
- Annotation-confidence weighting stays active in ALL configurations (even without class-frequency weighting).
- Each best checkpoint is chosen by pooled Marine Debris F1 on labeled validation pixels.
- Each variant has its own output and checkpoint: it cannot overwrite the previous `models/unet_best.pt`.
- Completed variants are skipped if the corresponding `.pt` checkpoint and `training_summary.json` both exist. To force a fresh full run, set `FORCE_RETRAIN_COMPLETED = True` at the top of script 18.
- If CUDA is out of memory: change `BATCH_SIZE = 4` to `BATCH_SIZE = 2` **before** starting a new controlled experiment. For fair comparisons, keep the same batch size for every variant; remove any partial outputs before a fresh experiment.

**Note:** the `no_debris_oversampling` condition uses uniform *replacement* sampling. It keeps the sampler type and epoch size constant, removing only the debris-dependent preference.

### Step 2: run `19_analyze_unet_v05.py` in PyCharm

This script opens the four best checkpoints and makes predictions on all labeled pixels of the **official validation split only**. It writes:

| Output file | Meaning |
|---|---|
| `results/unet_v05/validation_model_comparison.csv` | F1, precision, recall, IoU, TP/FP/FN and changes vs the fresh reference run |
| `results/unet_v05/cloud_wave_validation_errors.csv` | Clouds→Debris and Waves→Debris FP, with true-class support and FP rate; Debris→Cloud/Wave FN |
| `results/unet_v05/validation_comparison_with_archives.csv` | v0.5 compared with any available **archived** validation RF and v0.4 numbers; provenance identified |
| `results/unet_v05/validation_findings.md` | Generated short English report using locally observed results |
| `results/unet_v05/<variant>/val_confusion.csv` | 15×15 validation confusion matrices (rows=true, columns=predicted) |
| `results/unet_v05/<variant>/val_false_positives_by_class.csv` | Which true classes were confused with Marine Debris |
| `results/unet_v05/<variant>/val_patch_errors.csv` | Patch-level errors, including scenes **without** labeled debris |
| `assets/figures/unet_v05/ablation_f1_cloud_wave_fp.png` | Overall F1 and cloud/wave FP comparison |
| `assets/figures/unet_v05/validation_f1_learning_curves.png` | Validation F1 vs epoch for all variants |
| `assets/figures/unet_v05/best_validation_confusion_normalized.png` | Normalized confusion matrix of the best **validation** model |
| `assets/figures/unet_v05/val_*_v05_comparison.png` | RGB, labeled truth, two models' predictions and error maps |

### How to interpret the results

- **Marine Debris F1** is the **primary model-selection measure**.
- **Clouds→Debris FP** means a pixel *labeled Clouds* was incorrectly predicted as Marine Debris. **Waves→Debris FP** is analogous. These are NOT the same as **Debris→Clouds/Waves FN**.
- Check FP counts **alongside the labeled support per class**, FP rates, overall F1, and recall. A model can trivially reduce FP by failing to detect debris.
- Some classes (especially clouds or waves) may be less represented in validation than in the previously examined test split. Do not claim that validation cloud/wave improvements will transfer to test.
- The selected best variant is selected **on validation**, not independently validated; one seed and one validation split do not establish statistical superiority or robustness to unseen geography.
- Original test outcomes were previously examined. To make a clean final generalization claim, future comparisons need a fresh untouched evaluation protocol, preferably spatial/grouped and external.

### Important safeguards

- Do **not** run `17_evaluate_unet.py` to tune the v0.5 experiments: it reads official TEST.
- No stage here changes `models/unet_best.pt`, `results/unet/` or `results/benchmark/`.
- Source GeoTIFFs, HDF5 and trained checkpoints are **not** included in the distributed ZIP.
- This package contains **code and protocol only**. It does not assert v0.5 performance before execution on real MARIDA data.
