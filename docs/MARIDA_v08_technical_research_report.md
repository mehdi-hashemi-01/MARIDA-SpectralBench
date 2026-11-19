# MARIDA-SpectralBench: Reproducibility, Spatial Generalization and the Contribution of Remote-Sensing Pretraining

**Technical research report — v0.8**  
**Research code:** [MARIDA-SpectralBench](https://github.com/mehdi-hashemi-01/MARIDA-SpectralBench)  
**Data:** MARIDA Sentinel-2 benchmark  
**Scope:** 11 input bands, 15-class supervised segmentation on annotated pixels  
**Status:** A reproducible research record, **not a peer-reviewed publication**.

## Abstract

Floating marine debris is difficult to separate from spectrally related sea-surface structures in moderate-resolution satellite imagery. We study MARIDA with a hierarchy of classical spectral baselines, an 11-band U-Net, and wavelength-aware pretrained DOFA representations. Previously recorded official-validation experiments showed Marine Debris F1 scores of **0.8157** for a spectral-plus-index Random Forest, **0.8286** for the best v0.5 U-Net run, **0.7813** for a frozen-DOFA-plus-spectral decoder, and **0.9008** for a compact *spectral-only decoder* in v0.7. The apparently strong v0.7 result is an internally selected validation result; the official Test has been inspected in earlier project stages and is **not** treated as fresh confirmatory evidence. The present phase evaluates seed stability, Tile-ID-held-out generalization with nested checkpoint selection, and a matched frozen pretrained-versus-random DOFA control. These questions are addressed only when the corresponding experiment output files exist; unexecuted analyses are explicitly marked pending.

## 1. Motivation and questions

The key scientific question is not whether increasing model complexity can improve the same familiar validation set, but whether higher measured performance is stable and transferable. Three hypotheses are investigated: **H1:** the reported spectral-only model retains useful performance over different random seeds; **H2:** model rankings and error types change under unseen Tile-ID evaluation; **H3:** frozen DOFA pretrained features provide a measurable advantage over equally frozen features from an untrained architecture under matched decoder training.

## 2. Data and annotation interpretation

MARIDA contains 1,381 Sentinel-2 patches of 256 × 256 pixels, each with 11 spectral bands. Its published official split consists of **694 training, 328 validation and 359 test patches**. This project uses **15 classes** in its per-pixel prediction head, with raw mask class `0` marking **unlabeled** pixels. Such pixels do not contribute to loss or confusion matrices; consequently reported F1 values represent performance on *annotated pixels*, not detection prevalence or false-alarm rates over whole images. The 11-band input preprocessing uses finite-value handling and per-band z-scores clipped to [−8, 8]. Normalization values and class weights are calculated from the fitting subset only. The official Test was used in historical v0.1/v0.4 analyses, so it should not be represented as untouched evidence for the newly selected architectures.

## 3. Methods

### 3.1 Spectral-only reproducibility: five independent decoder fits

File `baseline/27_multiseed_spectral_v08.py` retrains the **v0.7 Spectral Only** architecture with five seeds: 42, 7, 21, 123 and 2026. The network is the v0.7 `DofaDenseHead` with its token input zeroed; it learns only from the 11 normalized raw spectral bands and small spatial convolutions. Each seed uses its own initial decoder parameters and training-sampling random stream. All checkpoints are selected on **official Validation** and evaluated on the same labeled Validation pixels. The arithmetic mean and sample standard deviation summarize *training stochasticity* on this evaluation population; they are **not** confidence limits for unseen locations. The v0.7 single-seed F1 of 0.9008 is a recorded reference, not a guarantee of bit-exact reproduction.

### 3.2 Geographic generalization: nested leave-one-Tile-ID-out on official Train

File `baseline/28_tile_holdout_v08.py` partitions the **official Train only** by the metadata Tile ID parsed from each patch filename. For each outer fold, all patches having the selected Tile ID are reserved. The remaining official-Train patches are divided into **inner fitting** and **inner checkpoint-validation** sets using complete Date+Tile acquisition groups. The inner split is deterministic, consistent across models, and requires some labeled debris in both parts. Normalization statistics and class weights come **only from inner fitting**, and sampling is based only on those fitting patches. Each candidate model is trained to a best-inner-validation checkpoint. **Only after that selection** is the outer Tile evaluated once. The outer Tile is never used for early stopping, optimizer updates, or learning-rate scheduling.

Three models use identical outer and inner partitions: **Spectral Only** (v0.7 decoder with tokens zeroed), **U-Net 11-band** (base width 16), and **frozen pretrained DOFA + raw-spectral skip** (`dofa_combined`). Each uses masked class-/confidence-weighted cross-entropy, AdamW, the same training seed, nominal batch size and epoch budget; the models differ in architecture and capacity, so this is **not** a controlled architecture-only causal comparison. To prevent normalization leakage, DOFA frozen features are **recomputed and cached separately for each fold** from images normalized by that fold's *inner-fitting* statistics. The old whole-Train-normalized DOFA v0.6 cache is not reused for geographic folds.

We report a confusion matrix pooled over all outer fold predictions (each original Train patch scored exactly once per model), as well as per-Tile debris metrics and Tile-macro F1 over Tiles containing annotated debris. Tiles without debris still contribute false positives to pooled F1; an undefined debris F1 is not silently replaced by zero in the Tile-macro average. **Tile identifiers are only a geographic proxy. Exact geographic footprint disjointness is not independently verified.**

### 3.3 Does DOFA pretraining help compared with random initialization?

File `baseline/29_random_dofa_control_v08.py` evaluates paired frozen-feature decoders under two encoder initializations: (i) real pretrained DOFA ViT-Base weights verified in v0.6, and (ii) the identical DOFA architecture constructed with `pretrained=False` using a fixed random initialization seed. Both encoders are frozen. Their generated features use the same official-Train normalization, and both decoders are in **DOFA-only mode**, with the raw-spectral skip forced to zero. For decoder seeds 42, 7 and 21, the two branches use identical optimization protocols and new decoder initializations. The signed difference `F1(pretrained) − F1(random)` is computed within each paired seed. This tests frozen representation value under the existing preprocessing/decoder, not the value of other foundation-model adaptation strategies.

### 3.4 Reproducibility controls and output provenance

All experiments fix the feature channel order, use the same 15-class annotated-pixel evaluation definition and write machine-readable metrics, patch-level errors, confusion matrices and best-checkpoint metadata. Fold manifests record Train/inner/outer filenames and per-fold train statistics. The pretrained DOFA checkpoint in the previous validated run has SHA256 `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`. Large model weights, raw images and cached tensors are stored locally and excluded from ordinary source commits.

## 4. Results

### 4.1 Archived historical results (different training/evaluation protocols)

| Model | Evaluation scope | Debris F1 | Evidence boundary |
|:--|:--|--:|:--|
| RF, 19 spectral/derived features | Official validation, v0.1 | 0.8157 | Historical one-run baseline |
| RF, 19 features | Official test, v0.1 | 0.6779 | Previously inspected Test |
| U-Net, 11 spectral bands | Official validation, v0.4 | 0.8009 | Single fitted model |
| U-Net, 11 spectral bands | Official test, v0.4 | 0.3851 | Previously inspected Test |
| U-Net v0.5 replay | Official validation | 0.8286 | Best of four validation-selected fits |
| Frozen pretrained DOFA + raw skip, v0.6 | Official validation | 0.7813 | Not isolated pretraining |
| Spectral Only decoder, v0.7 | Official validation | **0.9008** | Single-seed, validation-selected |
| DOFA-only frozen, v0.7 | Official validation | 0.6923 | Not a paired random-control comparison |
| DOFA + raw skip, v0.7 | Official validation | 0.7497 | Different standalone training run |
| DOFA LoRA + raw skip, v0.7 | Official validation | 0.8077 | Different training budget/batch size |

These numbers must not be sorted into a common independent generalization leaderboard.

### 4.2 Five-seed spectral-only reproducibility

**Status:** COMPLETED: repeated official-validation runs; not independent Test scores.

| Seeds | Mean validation debris F1 | Sample SD | Min F1 | Max F1 |
|--:|--:|--:|--:|--:|
| 5 | 0.8785 | 0.0151 | 0.8642 | 0.9008 |

### 4.3 Nested Tile-ID holdout generalization

**Status:** COMPLETED: 13-fold pooled out-of-Tile results from official Train only.

| Model | Pooled F1 | Precision | Recall | TP | FP | FN | Tile-macro F1 (positive tiles) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| spectral_only | 0.6089 | 0.5565 | 0.6722 | 1306 | 1041 | 637 | 0.5529 |
| unet | 0.2716 | 0.1949 | 0.4478 | 870 | 3594 | 1073 | 0.2105 |
| dofa_combined | 0.4232 | 0.2951 | 0.7478 | 1453 | 3470 | 490 | 0.4213 |


**Interpretation rule:** A lower out-of-Tile F1 than the earlier official-validation score would indicate sensitivity to the group-change protocol, but cannot by itself be ascribed solely to geography because training populations and inner-fit sizes also differ.

### 4.4 Frozen DOFA: pretrained versus random

**Status:** COMPLETED: matched paired decoder-seed experiment.

| Frozen DOFA initialization | Mean validation F1 |
|:--|--:|
| Official pretrained | 0.6555 |
| Randomly initialized | 0.0973 |
| Pretrained minus random, paired mean | 0.5582 |

**Interpretation rule:** A paired advantage for pretrained tokens under the fixed-encoder setting supports representation usefulness *for this protocol*; it does not imply that DOFA outperforms the spectral decoder, fully fine-tuned variants, or unfamiliar locations.

## 5. Discussion and limitations

**Selection-induced optimism.** Multiple decisions were made on official Validation across stages v0.1–v0.7. A strong Validation F1, particularly 0.9008, is therefore exploratory and possibly optimistic. The old Test has already been inspected; v0.8 does not re-use it to justify new feature or model choices. Future confirmatory claims require newly labeled, appropriately grouped data or a prospectively locked protocol.

**Geographic support and dependence.** The Tile-ID grouped folds exclude the same Tile identifier from fitting and scoring, but distinct identifiers are not necessarily nonoverlapping spatial footprints. Some Tile groups may have very few or no labeled debris pixels. Pooled pixel F1 and macro-Tile F1 answer different questions and must both be reported with supports.

**Sparse labels and error interpretation.** Unlabeled pixels are not verified negatives, so confusion matrices cannot establish full-scene false-alarm rates. Error concentration in ships, sediment-rich water, waves or natural organic material warrants targeted qualitative and spectral checks, not post-hoc physical explanations.

**Control limitations.** The spectral-only and DOFA branches share decoder capacity in v0.7 controls but differ in the scale and spatial resolution of inputs; zeroed feature branches retain parameters in the head. The randomly initialized DOFA test controls the pretrained encoder initialization in the frozen-token setting, but not training compute, backbone architecture alternatives or pretraining domain mismatch. DOFA normalization here may differ from its original pretraining preprocessing.

**Uncertainty limits.** Five training seeds measure algorithmic randomness on a fixed validation population, and three paired initialization runs measure conditional variation in the comparison. Neither replaces scene-independent confidence assessment; nearby annotated pixels should not be bootstrapped as independent observations.

## 6. Conclusions and recommended next action

**Confirmed prior finding:** under the official Validation protocol, the compact spectral-only decoder achieved the highest observed F1 (0.9008) among reported v0.1–v0.7 experiments. **Not yet established by that finding:** superior geographic generalization, true pretraining benefit, or reliable performance on completely new marine debris scenes. Interpret any v0.8 findings according to the completed outputs above. A PhD application can truthfully present the implementation and evaluation design even if geographic performance is modest; negative controls and analysis of domain shift are part of the research contribution.

## 7. Reproduction and source availability

- Train+Val seed stability: `baseline/27_multiseed_spectral_v08.py`.
- Official-Train nested geographic evaluation: `baseline/28_tile_holdout_v08.py`.
- Paired pretrained-vs-random frozen DOFA control: `baseline/29_random_dofa_control_v08.py`.
- This report generator: `baseline/30_research_report_v08.py`.
- Configurable settings are at the top of each file for direct **PyCharm Run**; no command-line arguments are needed.
- Results: `results/v08/multiseed/`, `results/v08/geographic/`, `results/v08/pretraining_control/`.
- Detailed protocol: `docs/v08_reproducibility_and_geographic_protocol.md`.

## References

1. Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., & Karantzalos, K. (2022). MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. *PLOS ONE*, 17(1), e0262247. https://doi.org/10.1371/journal.pone.0262247
2. Xiong, Z., et al. (2024). Neural Plasticity-Inspired Multimodal Foundation Model for Observing the Earth Crossing Modalities. https://arxiv.org/abs/2403.15356
3. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. https://doi.org/10.1111/ecog.02881

---

*All v0.8 numerical cells above are populated only by completed experiment artifacts. This report is generated locally from recorded files, not from an invented simulation of the MARIDA dataset.*
