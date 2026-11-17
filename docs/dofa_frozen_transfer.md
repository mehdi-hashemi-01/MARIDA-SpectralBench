# v0.6 — DOFA Foundation-Model Feature Transfer for MARIDA

## Research question

Does a **pretrained remote-sensing feature extractor** help a lightweight dense segmentation head identify Marine Debris on the *same official, labeled-pixel MARIDA validation split* used for the classical RF and U-Net v0.5 baselines?

**Model:** DOFA ViT-Base patch-16 (official **pretrained** weights, not randomly initialized), frozen, with a small trainable decoder and an explicit 11-band spectral skip. This is **not** full backbone fine-tuning or a vision-language model.

**No v0.6 MARIDA results are claimed until the user runs the scripts on the real dataset.**

## Why DOFA?

- DOFA conditions its patch embedding on **central wavelengths**, allowing the original 11 MARIDA Sentinel-2 channels. It avoids introducing fake channels required by some 12-band architectures.
- The published official source is `https://github.com/zhu-xlab/DOFA`; the official ViT-B pretrained checkpoint `DOFA_ViT_base_e100.pth` is hosted in `https://huggingface.co/earthflow/DOFA` (large download, around 448 MB).
- We use the **official source and weights**, not a custom random ViT. The loader asserts that at least 98% of backbone tensor elements match the pretrained checkpoint. The exact SHA256 is recorded.
- DOFA's standard `forward_features()` gives an **image-level vector**. For segmentation, we retain the spatial tokens after the transformer, apply its pretrained normalization, reshape to a 16×16 grid and interpolate in a separate supervised decoder.
- The original DOFA checkpoint is built around 224×224 images (14×14 tokens). We bicubically interpolate the pretrained positional embeddings to 256×256 MARIDA inputs (16×16 tokens). This is an adaptation and should be documented.

## Installation in the *same* Conda / PyCharm environment

Use the existing `dl` Python interpreter with CUDA-supported PyTorch, verified previously on RTX 5070 Laptop GPU. If needed, install `timm==0.9.2` and `huggingface_hub` via PyCharm `Settings -> Python Interpreter -> Install Package`, or run:

```powershell
C:\Users\Mehti\anaconda3\envs\dl\python.exe -m pip install timm==0.9.2 huggingface_hub
```

Other packages already used by v0.4: `torch`, `numpy`, `pandas`, `matplotlib`, `tifffile`, `scikit-learn`, `tables`.

**First execution of script 20 requires Internet** so Torch Hub can download the official GitHub code and Hugging Face can download the actual checkpoint. Running upstream code through Torch Hub executes external code; only use the documented trusted upstream repository. Follow the official source/license terms; the source and checkpoints are NOT included in this repository or the downloadable project ZIP.

## Band order: never guess or swap

The existing `baseline/settings.py` has the following order. DOFA wavelengths are supplied in **micrometres**, not nm:

| MARIDA band | Sentinel-2 designation | DOFA wavelength (µm) |
|---|---|---:|
| nm440 | B01 coastal aerosol | 0.440 |
| nm490 | B02 blue | 0.490 |
| nm560 | B03 green | 0.560 |
| nm665 | B04 red | 0.665 |
| nm705 | B05 red edge 1 | 0.705 |
| nm740 | B06 red edge 2 | 0.740 |
| nm783 | B07 red edge 3 | 0.783 |
| nm842 | B08 NIR | 0.842 |
| nm865 | B8A narrow NIR | 0.865 |
| nm1600 | B11 SWIR 1 | 1.610 |
| nm2200 | B12 SWIR 2 | 2.190 |

The implementation validates band names before any model weights are loaded. Wavelength centre values are approximate and should not be mistaken for a full spectral-response function.

## Steps (PyCharm, click Run)

1. `baseline/20_prepare_dofa.py` — verifies CUDA, downloads official DOFA source/weights, validates pretrained tensor coverage, and extracts **frozen 768×16×16 feature maps** for every official TRAIN and VAL patch. Outputs about hundreds of MB under `baseline/cache/dofa_v06/`. Nothing is written to Git-tracked model locations. Repeated runs skip existing valid cache files.
2. `baseline/21_train_dofa_head.py` — trains the **segmentation head only**, based on frozen DOFA features **plus normalized 11-band input skip**; loads train-only band stats and confidence/class-weighted masked loss. Uses official TRAIN to optimize and official VAL to select the best epoch. Does NOT touch TEST. Saves model under `models/dofa_v06/dofa_frozen_head_best.pt` and training history under `results/dofa_v06/`.
3. `baseline/22_analyze_dofa.py` — reloads the best checkpoint and evaluates all **328 official VAL patches / 213,102 labeled pixels**, generates the 15-class confusion, per-patch error CSVs, 5-panel examples, and optional comparison against archived RF / U-Net v0.5 validation CSVs.

The number of patches/labels is explicitly checked. If data do not match the archived benchmark, evaluation stops, instead of producing an incomparable table.

## Inputs and outputs

```text
MARIDA-SpectralBench/
  baseline/
    data/patches/                 # Already downloaded; not tracked
    data/splits/train_X.txt      # Official split
    data/splits/val_X.txt        # Official split
    cache/dofa_v06/              # Frozen feature .npy files; DO NOT TRACK
    dofa_transfer.py             # Adapter + head + cache checks
    20_prepare_dofa.py
    21_train_dofa_head.py
    22_analyze_dofa.py
  models/dofa_v06/               # Decoder checkpoint; DO NOT TRACK
  results/dofa_v06/
    run_config.json
    training_history.csv
    best_val_confusion.csv
    validation_confusion.csv
    validation_metrics.csv
    validation_patch_errors.csv
    validation_fp_by_true_class.csv
    validation_model_comparison.csv
  assets/figures/dofa_v06/
    training_curves.png
    validation_confusion_normalized.png
    model_f1_comparison.png
    *_val_visual_comparison.png
```

Add `baseline/cache/` to `.gitignore`. The existing `.gitignore` typically already excludes `.npy`, raw images, and `models/*`, but the cache includes a JSON metadata file which **must not** be committed inadvertently.

## Methodology and interpretation

- **Target classes:** 15 MARIDA labels mapped 1..15 -> 0..14; raw label `0` -> `IGNORE=255`. Only annotated pixels contribute to loss and metrics.
- **Confidence:** label confidence 1/2/3 contributes weight 1 / 0.5 / 1/3 to the training loss.
- **Train sampling:** preference for debris-positive images (3× sampling weight); inverse-frequency class weights clipped as in the existing U-Net code.
- **Normalization:** training-only mean and standard deviation (same general pipeline as v0.4 U-Net), with standardized values clipped to [-8,8]. **This is not guaranteed to match the image normalization DOFA saw during original pretraining.** A future controlled pretraining-normalization ablation is recommended if results are weak.
- **Positional embedding:** 14×14 -> 16×16 interpolation; decoding uses pretrained patch tokens AND raw bands, so a positive result would **not isolate** the contribution of the pretrained tokens. A useful follow-up control is the same decoder with its DOFA input zeroed or a randomly initialized frozen encoder.
- **Frozen encoder:** DOFA weights remain fixed. The trainable decoder receives the low-resolution 768-dimensional map and a high-resolution 11-band skip connection, producing 15 logits per pixel. This is feature transfer, **not** foundation-model backbone fine-tuning.
- **Compute trade-off:** token extraction is cached once; repeated head epochs run without executing the large transformer again. Cache occupies hundreds of MB locally.
- **Validation-only iteration:** no Test reads in scripts 20–22. Historical v0.4 Test scores were already inspected, so final independent claims need a fresh held-out/geographic evaluation designed separately.
- **Comparison:** RF v0.1 and U-Net v0.5 are **separately trained**. Same official Validation scoring population helps compare observed scores, but not causal attribution or significance across seeds.
- **Geographic overlap:** official Train and Validation share several Tile IDs. Good official VAL performance alone cannot establish independent transfer to unseen geography.
- **No dense-scene accuracy claim:** unlabeled areas displayed in maps remain excluded from the quantitative evaluation.

## Troubleshooting

| Message | Action |
|---|---|
| `CUDA unavailable` | Run existing `15_check_gpu_and_data.py`; select the `dl` CUDA PyTorch environment. |
| `No module named timm` | Install `timm==0.9.2` in the same environment. |
| `No module named huggingface_hub` | Install `huggingface_hub` in the same environment. |
| GitHub/Hugging Face network error | First run requires accessible source and checkpoint downloads; configure your network. |
| `pretrained weights mismatch` | Upstream code/weight mismatch; **do not** bypass the check or use random weights. Report the full error. |
| `CUDA out of memory` | In script 20 set `BATCH_SIZE=1`; in scripts 21/22 reduce batch size to 2. |
| `Stale DOFA cache metadata` | Confirm settings/dataset unchanged. If intentionally changed, remove *only* `baseline/cache/dofa_v06/` and run 20 anew. |
| `DOFA frozen head checkpoint exists` | Run script 22 to analyze, or set `FORCE_RETRAIN=True` in script 21 intentionally. |
| Small/poor F1 | First inspect train-normalization/pretrain domain shift, class counts, scene-wise FP, and visual maps; do not tune using official Test. |

## Sources and attribution

- Official DOFA code and paper: https://github.com/zhu-xlab/DOFA and https://arxiv.org/abs/2403.15356
- Official pretrained weights: https://huggingface.co/earthflow/DOFA
- MARIDA dataset/paper: https://doi.org/10.1371/journal.pone.0262247

The project uses DOFA as an **independent downstream experiment** and is not affiliated with the DOFA or MARIDA research teams. DOFA code/weights are fetched from their original hosts and are not redistributed.
