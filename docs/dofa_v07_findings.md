# MARIDA-SpectralBench — DOFA v0.7 Findings

**Experiment:** DOFA input-source controls and parameter-efficient adaptation (LoRA)  
**Evaluation:** Official MARIDA validation split only (328 image patches; 213,102 annotated pixels; 1,075 annotated Marine Debris pixels)  
**Selection:** Best Marine Debris validation F1 within each run; seed 42  
**Official Test:** Not accessed in these v0.7 experiments.

## 1. Research questions

1. Does frozen DOFA information improve segmentation when combined with a normalized 11-band spectral skip, under the implemented decoder and training recipe?
2. How much performance does the decoder achieve using only the spectral skip, with DOFA tokens zeroed?
3. Does LoRA adaptation of the last two DOFA attention blocks improve the combined system, and how carefully can the comparison be interpreted?

## 2. Exact experimental controls

The three source controls used **the same `DofaDenseHead` architecture**, identical train/validation patch identities, seed, masking, training weights, and debris-positive replacement sampling. All three source variants have the same total decoder parameter count. The selected input was disabled **by setting its feature tensor to zero**, not by changing the decoder architecture:

- `combined`: real pretrained DOFA feature grid + normalized 11-band skip.
- `dofa_only`: pretrained DOFA grid + zeroed raw spectral skip.
- `spectral_only`: **zeroed pretrained DOFA grid** + normalized 11-band skip.

The `spectral_only` variant is **not a non-spatial pixel-level classifier**: its decoder uses 3×3 convolutions over the 11-band image. Its high-resolution skip allows preservation of information that may be lost in the 16×16 DOFA grid.

The separate LoRA run uses the same pretrained DOFA checkpoint (SHA256 `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`), and trains **36,864 low-rank parameters** on the final two Transformer attention blocks plus **133,151 decoder parameters**. Its batch size (2) differs from source-control training (4), and its encoder is updated via LoRA rather than frozen; thus it is not an exact one-variable ablation against the `combined` run.

## 3. Validation results

| Configuration | Best epoch | Marine Debris F1 | Precision | Recall | IoU | TP | FP | FN |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **Spectral only** | 23 | **0.9008** | **0.9106** | **0.8912** | **0.8195** | 958 | **94** | **117** |
| Frozen DOFA + spectral skip (`combined`) | 17 | 0.7497 | 0.6862 | 0.8260 | 0.5996 | 888 | 406 | 187 |
| Frozen DOFA only | 10 | 0.6923 | 0.6080 | 0.8037 | 0.5294 | 864 | 557 | 211 |
| DOFA + LoRA + spectral skip | 10 | 0.8077 | 0.7617 | 0.8595 | 0.6774 | 924 | 289 | 151 |

All TP/FP/FN values above reproduce the values from the saved 15-class confusion matrices and the 328-patch error tables. The positive class is Marine Debris. Missing/unannotated pixels are excluded from scoring.

### Primary observation

Among the three matched source-control runs, **the high-resolution spectral skip without frozen DOFA tokens has the best validation result**. Its F1 exceeds `combined` by **0.1511** (15.11 percentage points), with **312 fewer false-positive predictions** and **70 fewer false-negative predictions**. Compared with `dofa_only`, it has **463 fewer false positives** and **94 fewer false negatives**.

This result **does not prove that remote-sensing pretraining is generally harmful**. It shows that injecting the particular frozen DOFA representation into this shallow fusion decoder under this single training recipe did not help. The token grid is coarse (16×16) relative to 256×256 sparse pixel labels. Architecture/fusion design, normalization, training dynamics, and local spatial detail are potential explanations, not established causes.

### LoRA result

LoRA improves validation F1 compared with the newly trained frozen `combined` arm by **0.0580** (+5.80 percentage points), reducing FP **406 → 289** and FN **187 → 151**. Against the separate historical DOFA v0.6 run (F1 = 0.7813), LoRA F1 is higher by **0.0264** (+2.64 percentage points). The runs differ in training details, so these are *observed numerical differences*, not a causal estimate of the benefit of LoRA alone.

Crucially, **LoRA remains below `spectral_only`** by **0.0931** (9.31 percentage points) on the same validation partition.

## 4. Error analysis

| Ground-truth class incorrectly predicted as debris | Spectral only | Combined | DOFA only | LoRA |
|:--|--:|--:|--:|--:|
| Natural Organic Material | 51 | 48 | 40 | 52 |
| Ship | 8 | 78 | 77 | 45 |
| Waves | **0** | 98 | **0** | 67 |
| Clouds | **0** | 2 | **0** | **0** |
| Sediment-Laden Water | **0** | **0** | **0** | 18 |
| Sparse Sargassum | 17 | 56 | 60 | 23 |
| Marine Water | 1 | 25 | 189 | 17 |
| Mixed Water | 12 | 34 | 70 | 37 |

`Spectral_only` is substantially more precise overall, but **Natural Organic Material** remains its main false-positive source: 51 of its 94 false alarms. The largest `spectral_only` false-negative hotspot is patch `S2_18-9-20_16PCC_17.tif`: **31 missed debris pixels of 38 annotated debris pixels**. Other remaining errors are worth local visual inspection rather than interpreting the high aggregate F1 as error-free segmentation.

Patch-level examples:

- `combined` generates 45 FP in `S2_11-6-18_16PCC_51.tif`, vs 0 FP for `spectral_only`.
- `combined` generates 43 FP in `S2_12-1-17_16PEC_15.tif`, vs 0 FP for `spectral_only`.
- `dofa_only` generates 71 FP in `S2_18-1-18_48PZC_23.tif`, vs 0 FP for `spectral_only`.
- LoRA generates 41 FP in `S2_11-6-18_16PCC_51.tif`; it does not uniformly resolve the difficult combined-branch cases.

## 5. Comparison with earlier project stages (descriptive only)

| System | Evaluation | Marine Debris F1 |
|:--|:--|--:|
| **v0.7 spectral-only decoder** | Official validation | **0.9008** |
| U-Net v0.5 `v04_replay` | Official validation | 0.8286 |
| RF with 19 inputs (archived) | Official validation | 0.8157 |
| DOFA + LoRA v0.7 | Official validation | 0.8077 |
| DOFA frozen v0.6 | Official validation | 0.7813 |

The same labeled validation split was used for these headline scores, but **the training recipes, architectures, feature paths, and checkpoints differ**. The U-Net, RF, and v0.7 control results do not constitute a controlled one-factor architecture comparison. The v0.7 spectral-only score is a promising *validation* result, **not** a test/generalization claim.

## 6. Limitations and next controls

- **Only one seed** was used in each v0.7 run. Repeatability has not been established.
- **Model selection occurred on the official validation split**, which has already supported many prior experimental decisions. The official Test was previously inspected in the project, so it must not be treated as a pristine confirmatory evaluation for new model choices.
- **Tile IDs overlap between official Train and Validation** (five shared IDs in a prior audit). There is no group-disjoint, independent geographic performance result for v0.7.
- **Identical decoder size is not identical information or optimization difficulty.** Zeroing one input branch yields different feature statistics and can alter training dynamics.
- **Pretrained vs randomly initialized encoder** has not been tested under a controlled, matched architecture. Therefore the *causal contribution of pretraining itself* is still unmeasured.
- **LoRA vs frozen control is not strictly matched** for batch size and trainable parameter count. No causal LoRA gain can be claimed from the observed score difference alone.
- **Sparse annotations** make these labeled-pixel scores different from complete-image segmentation accuracy.

Recommended focused next steps:

1. Retain the `spectral_only` best checkpoint as the current **validation leader**, without an independent-performance claim.
2. Audit normalization, label masking, image split membership and cache handling; particularly verify that `spectral_only` does not access validation labels or cached training-only statistics derived from validation.
3. Repeat the leading lightweight spectral model and a tightly matched `combined` control across several seeds, then evaluate both under exactly the same group-disjoint protocol.
4. Only if a Foundation Model-specific scientific question remains, compare **pretrained DOFA vs a randomly initialized DOFA encoder** using a matched decoder, channel inputs, and training budget, and refine fusion independently of pretraining.

## 7. Provenance

Uploaded results analyzed: `dofa_v07.zip` (three figures) and `dofa_v07(1).zip` (confusion matrices, training logs, run configurations, patch/class metrics). No source files, raw satellite data, or model checkpoints were included in these two result archives; all observations here refer to logged v0.7 evaluation artifacts.
