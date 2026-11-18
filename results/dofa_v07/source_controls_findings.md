# DOFA source controls — Validation findings

All three variants reuse the v0.6 decoder architecture; one input source is zeroed where needed. Three independent fits (one seed) on identical official Train/Validation splits. **No TEST evaluation.** DOFA pretrained backbone is frozen.

| Mode | F1 | Precision | Recall | IoU | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| spectral_only | 0.9008 | 0.9106 | 0.8912 | 0.8195 | 958 | 94 | 117 |
| combined | 0.7497 | 0.6862 | 0.8260 | 0.5996 | 888 | 406 | 187 |
| dofa_only | 0.6923 | 0.6080 | 0.8037 | 0.5294 | 864 | 557 | 211 |


These controls measure the contribution of supplied inputs to this decoder in one training run. They are not a random-initialized-encoder control and do not alone establish a causal advantage from pretraining.
