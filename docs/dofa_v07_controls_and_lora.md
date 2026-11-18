# v0.7 — DOFA Input-Source Controls and Low-Rank Fine-Tuning

**Status:** implementation ready; research outcomes pending execution on local MARIDA data. No new F1 values are claimed here.

## Why this experiment?

The completed v0.6 model combined **frozen pretrained DOFA tokens** and a **raw 11-band spectral skip** inside a shared segmentation decoder. Its Validation Marine Debris F1 was **0.7813** (970 TP / 438 FP / 105 FN). That score cannot isolate the contribution of pretrained features because the decoder also sees the raw bands.

The v0.7 experiments ask two different questions:

1. **Input-source ablation:** which input source(s) contribute useful predictive information to this particular decoder trained from scratch?
2. **Parameter-efficient adaptation:** can updating a small subset of pretrained encoder transformations (via LoRA) improve the same official Validation metric compared with the frozen configuration?

These are **exploratory** studies. The historical official Test has already been inspected in earlier project stages and is **not used** by any v0.7 script.

## Part A — Matched source controls (scripts 23–24)

All three variants use the exact `DofaDenseHead` architecture from v0.6. The same 694 training patches, 328 validation patches, 15-class sparse-label masked cross-entropy, sampling weights, class/confidence weighting, random seed, optimizer/schedule and maximum epochs are used. Each decoder is initialized and fitted **from scratch**; frozen DOFA features are read from the validated stage-20 cache.

| Variant | Frozen pretrained DOFA tokens | Normalized 11-band spectral skip | Architecture |
|---|---|---|---|
| `combined` | Present | Present | Same DofaDenseHead |
| `dofa_only` | Present | Replaced by zeros | Same DofaDenseHead |
| `spectral_only` | Replaced by zeros | Present | Same DofaDenseHead |

We zero **entire input channels**, instead of changing the decoder's module sizes. This is a matched-input control with the same decoder parameter count, but it is **not** a random/unpretrained-DOFA control: even if the pretrained token branch improves scores, the causal value of pretraining relative to a random encoder is not independently measured.

The decoder has several normalization and bias parameters, so a zeroed branch can still produce learned constants. It receives **no information from the excluded source**, which is the intended control.

Run in PyCharm:

1. `baseline/23_train_dofa_source_controls.py`
2. `baseline/24_analyze_dofa_source_controls.py`

All three models save separate best-validation checkpoints in `models/dofa_v07/`, which is intended to remain local. Comparison CSVs are written to `results/dofa_v07/`; figures appear under `assets/figures/dofa_v07/`. Main output: `source_controls_validation_comparison.csv`.

**Interpretation:**

- `combined > spectral_only` suggests DOFA tokens provide incremental predictive information given this decoder and this training protocol.
- `combined > dofa_only` suggests the raw spectral skip provides incremental information.
- `spectral_only >= combined` weakens the case for using the frozen pretrained features in the current architecture.
- These observations are **one-seed, in-validation** differences, not independent regional generalization or definitive statistical significance.

## Part B — LoRA on the last two DOFA attention blocks (scripts 25–26)

LoRA (low-rank adaptation) freezes the original pretrained linear transformations and learns a small correction through two low-rank matrices:

`y = frozen_linear(x) + (alpha / rank) * B(A(x))`

Default settings: rank 4, alpha 8, inject into the **attention `qkv` and attention `proj` linear layers of the final two Transformer blocks**. Earlier blocks, wavelength-sensitive patch embedding, original attention weights, and other pretrained parameters remain frozen.

- The original DOFA checkpoint is obtained via the existing `load_real_pretrained_dofa` function and checked against the SHA256 recorded in the stage-20 cache metadata.
- The **last two blocks** must be evaluated on live inputs every batch; the cached v0.6 embeddings cannot be reused for trainable LoRA. Therefore LoRA training uses more GPU compute/memory than frozen feature-head training.
- The decoder is the same `DofaDenseHead`, using **both** DOFA tokens and spectral skip.
- Unlabeled pixels (raw label 0; neural ignore 255) do not contribute to loss or metrics.
- Only the trainable LoRA tensor values and decoder weights are stored in `models/dofa_v07/lora_last2_best.pt` (plus configuration and pretrained checkpoint SHA256); **DOFA base weights are not duplicated**.

Run in PyCharm **after reviewing Part A results**:

3. `baseline/25_train_dofa_lora.py`
4. `baseline/26_analyze_dofa_lora.py`

The default LoRA training batch size is 2. If a CUDA OOM occurs, set `BATCH_SIZE=1` at the top of file 25. The training/validation processes use the existing RTX 5070 CUDA environment. `timm==0.9.2` and `huggingface_hub` are already prerequisites for the v0.6 model-loader workflow; do not reinstall PyTorch if stage 20 worked.

Primary new outputs:

- `results/dofa_v07/lora_training_history.csv`
- `results/dofa_v07/lora_validation_metrics.csv`
- `results/dofa_v07/lora_validation_confusion.csv`
- `results/dofa_v07/lora_validation_patch_errors.csv`
- `results/dofa_v07/lora_vs_frozen_validation.csv`
- `assets/figures/dofa_v07/lora_training.png`
- `assets/figures/dofa_v07/lora_validation_summary.png`

## Important comparability limitations

- `combined` (cached frozen ViT feature maps) and LoRA (live encoder forward) have different computational pathways; frozen caches were stored in FP16. Small numerical differences can occur.
- The LoRA experiment trains a new decoder from scratch. The added encoder adapters and differences in batch size/learning rate mean **any score difference is not an isolated causal treatment effect**. A stronger follow-up would match batch size, optimization budget and random seeds and compare live frozen encoder vs LoRA directly.
- **Never select LoRA settings on the official Test.** Only official Train/Validation are opened in v0.7.
- These splits are not fully independent by geographic Tile ID; later group-disjoint evaluation is required.
- A better Validation F1 in one run is not proof of improvement on unseen regions. Report FP/FN, per-class and scene-level errors, seeds and training configuration alongside F1.

## Reproducibility and Git hygiene

The pretrained DOFA checksum recorded in prior successful extraction was:

`4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`

The scripts read and check the **actual** SHA256 from cache/model; the hexadecimal string above is historical provenance, **not** a hardcoded replacement for verification.

Keep `baseline/cache/`, `models/`, raw MARIDA GeoTIFFs, and full Hugging Face caches out of Git. Commit scripts, documentation, figures and small CSV summaries when they have been created by the actual MARIDA run. No synthetic smoke-test metrics belong in README.

## Sources

- [DOFA official code](https://github.com/zhu-xlab/DOFA)
- [DOFA official pretrained weight repository](https://huggingface.co/earthflow/DOFA)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
