# v0.8 — Frozen DOFA Pretraining Control: Recorded Findings

## Experimental question

Does a **frozen, genuinely pretrained DOFA ViT-Base** encoder provide more useful features for MARIDA semantic segmentation than the **same architecture with random initial weights**?

## Paired protocol

The experiment used 694 official Train and 328 official Validation patches. Both DOFA encoders were **frozen**. The original 11-band raw-spectral skip was zeroed, leaving only the DOFA token pathway available to the identical trainable segmentation decoder. The two treatments used the same Train-derived normalization, training and validation identities, decoder architecture, training schedule, loss, and paired decoder seeds **7, 21 and 42**. Best checkpoints were selected on official Validation.

One single random DOFA encoder was initialized using seed `20261119` and **reused for all three decoder seeds**. This is not an experiment over three independently randomized backbones. The pretrained encoder matches the previously verified official DOFA weights (100% tensor coverage).

## Recorded validation results

| Decoder seed | Pretrained F1 | Random F1 | Paired delta |
|--:|--:|--:|--:|
| 7 | 0.6276 | 0.0363 | +0.5914 |
| 21 | 0.6745 | 0.1249 | +0.5496 |
| 42 | 0.6643 | 0.1306 | +0.5337 |
| **Mean** | **0.6555** | **0.0973** | **+0.5582** |

The paired delta has sample standard deviation **0.0298** across the three decoder seeds. All six runs use the same **213,102 annotated official-Validation pixels** containing **1,075** positive pixels. The official Test was not accessed.

## Reproducibility hashes

- Pretrained checkpoint SHA256: `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`
- Fixed random backbone initialization fingerprint: `1255b8cfda921928586f48b67aeb0f5ea5a8706394c672420b0143e05d8418ad`

## Scientific interpretation and caveats

- **Supported:** under the reported frozen DOFA-only setting, the pretrained representations substantially outperform the features of the randomly initialized DOFA encoder across these three decoder seeds.
- **Not established:** independent unseen-geography transfer, a foundation-model advantage over the higher-resolution spectral-only convolutional decoder, or the causal value of an alternative pretraining task or fusion architecture.
- The random encoder has only **one initialization**; the three seed repetitions change the **decoder**. The selected checkpoints use Validation for selection and scoring, so these are exploratory validation results rather than confirmatory test estimates.
- Original DOFA pretraining normalization and Sentinel-2 band preprocessing may not be identical; a normalization and multiple-random-backbone study remains worthwhile.

## Evidence

`results/v08/pretraining_control/paired_per_seed.csv`, `paired_summary.json`, `per_run.csv`, per-seed `metrics.json`, `confusion.csv`, `history.csv`, and `patch_errors.csv` are retained. Model `.pt` checkpoints and cached spectral features are **not** included in the GitHub documentation overlay.
