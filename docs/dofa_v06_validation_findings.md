# DOFA v0.6 — Frozen Remote-Sensing Foundation Model: Validation Findings

## Experiment and provenance

The experiment uses **DOFA ViT-Base pretrained weights**, verified to 100% tensor coverage, SHA256 `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`. The encoder is **frozen**; a **trainable dense segmentation decoder with an additional raw-spectral skip connection** is fitted on **694 MARIDA training patches**. The best checkpoint was selected by labeled-pixel Marine Debris F1 on **328 validation patches** (213,102 labeled pixels). The official test set **was not accessed**. This is frozen feature transfer, not backbone fine-tuning.

## Primary result (validation)

| Metric | Value |
|---|---:|
| Best epoch | 10 of 18 run epochs |
| Marine Debris F1 | 0.7813 (78.13%) |
| Precision | 0.6889 |
| Recall | 0.9023 |
| IoU | 0.6411 |
| True positives (TP) | 970 |
| False positives (FP) | 438 |
| False negatives (FN) | 105 |

Source: `results/dofa_v06/validation_metrics.csv` and the 15-class `validation_confusion.csv`. Pixel sums and per-patch totals were cross-checked.

## Historical validation comparison (separate training runs)

| Model | Marine Debris F1 |
|---|---:|
| U-Net v0.5 reference | **0.8286** |
| Archived 19-feature Random Forest | **0.8157** |
| Archived 11-band Random Forest | 0.7957 |
| DOFA v0.6 frozen + spectral-skip decoder | **0.7813** |

The frozen DOFA configuration is **4.73 percentage points below** the reference U-Net v0.5 and **3.44 points below** the archived 19-feature RF on validation. These are descriptive comparisons of independently trained models on the same official validation labeled-pixel split; no controlled variance/statistical significance is established.

## What are false positives mistaken for?

| True annotation class | Pixels predicted as Marine Debris |
|---|---:|
| Ship | 88 |
| Sparse Sargassum | 68 |
| Mixed Water | 59 |
| Natural Organic Material | 51 |
| Waves | 51 |
| Dense Sargassum | 43 |
| Marine Water | 40 |
| Turbid Water | 16 |
| Foam | 15 |
| Wakes | 7 |

Notably, **Clouds → Marine Debris = 0** and **Sediment-Laden Water → Marine Debris = 0** in this validation run. This does *not* establish that such confusions are absent in test or geographically unseen areas. Errors are instead concentrated in Ship, Sargassum, Mixed Water, organic material, and Waves.

## High-error patches

| Patch | TP | FP | FN | Actual Marine Debris pixels |
|---|---:|---:|---:|---:|
| `S2_12-1-17_16PEC_15.tif` | 0 | 52 | 0 | 0 |
| `S2_18-1-18_48PZC_23.tif` | 0 | 46 | 0 | 0 |
| `S2_14-3-20_18QYF_3.tif` | 44 | 44 | 1 | 45 |
| `S2_11-6-18_16PCC_51.tif` | 0 | 39 | 0 | 0 |
| `S2_18-9-20_16PCC_48.tif` | 2 | 19 | 0 | 2 |

`S2_11-6-18_16PCC_26.tif`, the earlier RF sediment false-positive case, has **zero FP** under DOFA v0.6. A different location with 52 FP (`S2_12-1-17_16PEC_15.tif`) has no labeled Marine Debris. Spatial error concentration is still present, but not primarily from the previously highlighted sediment patch.

## Learning behavior

The training run records **18 epochs**, but the best validation Marine Debris F1 occurs at **epoch 10**. After this epoch, validation F1 fluctuates strongly while training weighted cross-entropy generally decreases, indicating that loss minimization and the selected Marine Debris F1 objective are not identical. Training cross-entropy is weighted whereas validation cross-entropy is unweighted, so their absolute values should not be compared directly.

## Limits and next research decision

- The pretrained encoder was frozen; this result does not evaluate full fine-tuning, adapters or LoRA.
- The decoder also receives raw 11-band spectral information. A **spectral-skip-only control** and a **DOFA-feature-only control** are needed before attributing predictive value to pretraining.
- Validation has been repeatedly inspected during project development; avoid presenting it as a new independent test set.
- The official test split was not evaluated here; no v0.6 test F1 is available.
- Sparse annotations mean all metrics apply only to labeled pixels.
- A sound next comparison would hold the evaluation protocol fixed and compare DOFA features + spectral skip against matched controls; subsequently consider parameter-efficient tuning only if the controls justify it.

**Conclusion:** pretrained DOFA features plus a dense spectral-skip decoder are technically operational but currently **do not outperform** the project's U-Net v0.5 or 19-feature Random Forest on official labeled-pixel validation F1. Despite lower aggregate F1, the model has a different error profile, notably avoiding the labeled sediment/cloud false positives seen in other experiments.
