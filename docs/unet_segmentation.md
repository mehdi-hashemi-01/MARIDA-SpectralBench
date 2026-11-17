# v0.4 — 11-band U-Net on MARIDA: implementation & evaluation protocol

## Research question

Does a spatial convolutional segmentation model improve marine debris detection relative to the archived pixel-wise Random Forest baselines, especially on difficult marine scenes?

**Status:** Scripts implemented; **real results require running the official MARIDA dataset**. No U-Net scores are claimed in advance.

## Required files

Use the ORIGINAL MARIDA GeoTIFFs, never a derived RGB/JPEG copy or the pixel-only HDF5 to train U-Net:

```
MARIDA-SpectralBench/
  baseline/
    settings.py
    unet_core.py
    15_check_gpu_and_data.py
    16_train_unet.py
    17_evaluate_unet.py
    data/
      patches/S2_DATE_TILE/S2_DATE_TILE_N.tif
      patches/S2_DATE_TILE/S2_DATE_TILE_N_cl.tif
      patches/S2_DATE_TILE/S2_DATE_TILE_N_conf.tif
      splits/train_X.txt
      splits/val_X.txt
      splits/test_X.txt
  models/          # local model checkpoint, excluded from Git
  results/unet/    # locally written output tables
  assets/figures/unet/
```

The official MARIDA masks use **1–15** for classes and **0** for **unlabeled** pixels. Labels 1–15 are converted to 0–14 for PyTorch; 0 is converted to ignore index 255. Unlabeled pixels are never treated as negative observations. The model outputs 15 classes and sees all 11 spectral channels. It still receives full images, so unlabeled neighborhoods may influence the model representations, but **only labeled target pixels contribute to the loss/metrics**.

## RTX 5070 and PyCharm (Windows)

1. Select the same `dl` interpreter already used by the project in PyCharm (`Settings -> Project -> Python Interpreter`).
2. Install `numpy`, `pandas`, `matplotlib`, `tifffile` in that interpreter. For the RTX 5070, install a **CUDA-enabled** PyTorch build supporting **Blackwell / compute capability sm_120**. Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) (OS Windows, Package Pip, CUDA 12.8 or another explicitly supported current CUDA build). One official example is:

   ```powershell
   C:\Users\Mehti\anaconda3\envs\dl\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   ```

   Install this via PyCharm's Terminal (not necessary for actually running scripts). If torch is already CUDA-capable, do not reinstall without need. Updating the NVIDIA graphics driver may be necessary. Never infer GPU availability just from `torch.cuda.is_available`: script 15 runs an **actual convolution on CUDA**.
3. Run `baseline/15_check_gpu_and_data.py` using the green Run button. It must show the GPU, CUDA support, 11-band U-Net forward PASS, and official splits (usually train 694, val 328, test 359).
4. Run `baseline/16_train_unet.py`. It calculates band normalization statistics **only from train patches**, scans train masks for class counts, trains a small 11-input, 15-class U-Net, and saves `models/unet_best.pt` when pooled validation Marine Debris F1 improves. No official test data is read during training. It writes `training_history.csv`, `run_config.json`, `training_summary.json`, and `loss_and_validation_f1.png`.
5. Once training settings are frozen, run `baseline/17_evaluate_unet.py` **once** to evaluate the best checkpoint on official val and test. It creates confusion matrices, per-class tables, patch-level errors, visual RGB/ground-truth/prediction/error comparisons, and a comparison with the **archived** Random Forest numbers. The evaluation does not train anything.

### Default model and compute

* Input: `[B,11,256,256]`; output: `[B,15,256,256]` logits; four encoder levels, U-Net skip connections, GroupNorm and ReLU, `BASE_CHANNELS=16`.
* Train: `EPOCHS=35`, `BATCH_SIZE=4`, AdamW (`LR=0.0003`, weight decay `0.0001`), random horizontal/vertical flips and 90° rotations, FP16 automatic mixed precision, gradient clipping 1.0, patience 10. These are **starting parameters**, not optimized values.
* Train-only normalization: channel mean and standard deviation estimated from spatial subsamples of all train patches; NaNs imputed with train means; standardized inputs clipped to [-8,8]. Never fit normalization on val/test.
* Class weights: inverse square root of training class frequencies, clipped to [1,8]. Annotator-confidence weights: 1.0 / 0.5 / 1/3 for high/moderate/low. The masked loss is a weighted mean cross-entropy; the **validation** cross-entropy is unweighted and uses only labeled pixels.
* Sampling: patches with any Marine Debris label are 3× as likely to be drawn within train epochs. This changes the observed training prevalence; results need classwise and calibration scrutiny. A sample is a complete image, never a pixel. No RGB-only input is used.
* Windows `DataLoader(num_workers=0)` is intentional; avoids multiprocessing spawn problems when clicking PyCharm Run.
* RTX 5070 out-of-memory? Change `BATCH_SIZE=2`, then `BASE_CHANNELS=8` if necessary. Because such changes alter the experiment, record them alongside the results.

### What the outputs mean

| File | Content |
|:--|:--|
| `models/unet_best.pt` | Best checkpoint based solely on validation debris F1; **excluded from Git** |
| `results/unet/training_history.csv` | Per-epoch train weighted loss, validation CE, F1, P, R, TP, FP, FN |
| `assets/figures/unet/loss_and_validation_f1.png` | Learning curves (the train and val CE use different weighting; compare trends, not raw magnitudes) |
| `results/unet/{val,test}_confusion.csv` | Full 15-class confusion matrix for annotated pixels |
| `assets/figures/unet/{val,test}_confusion_normalized.png` | Row-normalized confusion matrix |
| `assets/figures/unet/{val,test}_confusion_logcounts.png` | Log-count confusion matrix |
| `results/unet/{val,test}_per_class.csv` | Per-class Precision, Recall, F1, support |
| `results/unet/{val,test}_patch_errors.csv` | Image-level TP, FP, FN |
| `assets/figures/unet/{val,test}_*_comparison.png` | Satellite RGB, debris truth, prediction, error map, overlay |
| `assets/figures/unet/{val,test}_*_multiclass.png` | Multiclass prediction map vs annotated ground truth |
| `results/unet/model_comparison.csv` | U-Net v0.4 vs **archived v0.1** RF on same official val/test splits |

## Fairness and limitations

* The baseline 11-band Random Forest uses only the pixel spectrum; the U-Net uses the 11 bands plus **spatial context**. The archived 19-feature RF also gets engineered indices. The comparison is an **architectural benchmark**, not an equal-compute or identically tuned experiment.
* RF values are read from `results/benchmark/validation_comparison.csv` and `test_result.csv` and **not retrained**. U-Net scores and RF scores can be compared only when evaluated-pixel counts and data version match; the script warns if counts differ. Relative changes still require care because training procedures differ.
* Training data splits are by official patch; there is possible overlap of *Tile IDs* between official train/validation. This experiment is **not** a new geographically disjoint holdout; group generalization was previously evaluated for RF and would need a separate U-Net protocol for a fair geographic comparison.
* Sparse annotations can lead to misleading dense visual predictions. Predictions on unlabeled pixels are not scored or labeled as correct/incorrect; the visualization masks them out.
* Test data was already viewed for the RF research phase; use test scores for **reporting** after the U-Net configuration is frozen, not for trial-and-error changes. Avoid claiming a pristine never-seen test benchmark.
* A single-seed initial U-Net run and hyperparameter defaults do not establish statistical significance. Additional seeds/geographic holdouts are valid future work, not results of this code release.

## References

* [MARIDA official repository / dataset structure](https://github.com/marine-debris/marine-debris.github.io)
* [MARIDA Zenodo dataset](https://zenodo.org/records/5151941)
* [Official PyTorch install selector](https://pytorch.org/get-started/locally/)
