<!-- OPTIONAL README SNIPPET. Insert after the Feature Ablation v0.3 section.
     Do NOT insert any real performance numbers until you run steps 11 and 12. -->

## Robustness and Scene-Level Error Analysis (v0.3.1)

To check whether feature ablation findings depend on a single training seed or a single challenging satellite patch, this extension repeats all six Random Forest feature sets using five different seeds. **All feature sets within a given seed share the same sampled training pixels.** Evaluation always uses the complete labeled validation split; the held-out test partition remains untouched.

- `baseline/11_multi_seed_robustness.py`: 5 seeds × 6 feature sets, per-seed metrics, mean/SD summaries, paired comparisons against the 19-feature baseline, and error totals by patch, acquisition group (`Date + Tile`), and geographic tile (`Tile`).
- `baseline/12_analyze_scene_errors.py`: checks that grouped counts match validation totals, ranks scene-level errors, compares `full_19` vs `minus_ndwi` across all available validation groups, and performs a *descriptive*, paired acquisition-group bootstrap.

[Detailed method and interpretation](docs/robustness_and_scene_analysis.md) · [Experiment results directory](results/robustness/)

**Limitations:** Changing the seed varies training sampling and forest randomness jointly. The Date+Tile groups are proxies for acquisition conditions, not guaranteed independent examples. Scene-group resampling is descriptive, not an out-of-region validation. Shared Train/Validation metadata groups are audited and listed, but this does not establish that the existing official split is spatially disjoint.

*Measured findings should be inserted only after both scripts complete on the real MARIDA data.*
