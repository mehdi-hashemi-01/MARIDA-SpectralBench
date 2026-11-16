# Dataset and Evaluation Protocol

## Data source

Kikaki et al., MARIDA, PLOS ONE (2022), DOI: [10.1371/journal.pone.0262247](https://doi.org/10.1371/journal.pone.0262247). Dataset archive: <https://zenodo.org/records/5151941>.

MARIDA provides 1,381 multispectral Sentinel-2 patches of 256 × 256 pixels. The original patch directory contains multispectral GeoTIFFs, label rasters ending `_cl.tif`, and confidence rasters ending `_conf.tif`. The separate `shapefiles/` directory stores vector annotations and geographical metadata; `splits/` holds official training/validation/test partition manifests. `labels_mapping.txt` is a mapping of image filenames to 15-element multi-label presence vectors.

The pixel-level HDF5 export is used here; it must contain table-format `train`, `val`, and `test` keys, including 11 spectral band columns and `Class`. The HDF5 is not part of this repository. The official reference scripts can be consulted via <https://github.com/marine-debris/marine-debris.github.io>.

## Experimental setup

- Features: 11 spectral reflectance columns; optionally eight engineered spectral indices.
- Learner: RandomForestClassifier, 100 trees, min_samples_leaf=2, seed=42.
- Training: classwise uniform random sampling, maximum 2,000 pixels per class, same sample across both feature sets.
- Optional training sample weights: High = 1.0, Moderate = 0.5, Low = 1/3; otherwise 1.0.
- Missing values: filled with medians fitted exclusively to the sampled training data.
- Model selection: maximize Marine Debris F1 on validation.
- Final evaluation: run the validation-selected model once on the held-out test split.
- Evaluation units: HDF5 annotated pixels, not dense segmentation pixels over entire image patches.
- Results: archived CSVs under `results/benchmark/`; PNGs under `assets/figures/`.

## Interpretation of the results

Because data are weakly labeled, performance scores describe agreement on the annotated evaluation pixels. Unlabeled pixels are not automatically negative. The original class distribution is not preserved by per-class training sampling. These results should not be treated as a strict reproduction of the MARIDA paper without aligning all preprocessing, evaluation, and sampling details.

Validation and test should not be pooled. Subsequent model development should be decided on validation or new held-out geography, not by repeatedly optimizing against the reported test set.
