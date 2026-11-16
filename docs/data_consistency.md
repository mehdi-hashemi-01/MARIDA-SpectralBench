# HDF5–GeoTIFF consistency check

`baseline/08_verify_hdf_geotiff.py` verifies whether the spectral signatures stored in MARIDA's `dataset.h5` agree with the original multispectral GeoTIFFs and their pixel-level class masks. The script works on selected validation patches and does not train or tune a model.

## Run

Open the repository in PyCharm, verify that `baseline/data/dataset.h5` and `baseline/data/patches/` exist, and run `baseline/08_verify_hdf_geotiff.py` directly. The script uses the existing project `settings.py`. It requires pandas, PyTables (`tables`), NumPy, scikit-learn, tifffile and matplotlib.

Edit `PATCHES_TO_CHECK` at the top of the script to inspect additional validation patches.

## Method

For each patch, HDF5 rows are located by `Date`, `Tile` and `Image`. Up to 200 valid records are compared against annotated, finite-spectra pixels from the corresponding 11-band GeoTIFF using nearest-neighbor matching in spectral space. In addition to spectral agreement, the class label is checked against the GeoTIFF class mask at the matched pixel. Multipliers of 1, 0.0001 and 10000 are tested to detect common unit errors. HDF5 pixel coordinates are checked against candidate image-coordinate conventions only when their range suggests pixel indices; map coordinates may not be comparable directly.

A match means **all 11 bands** agree to the configured numerical tolerances. A `PASS` result requires at least 98% agreement in both the spectral signature and mask class, with no rescaling. A `SCALE_DIFFERENCE` means a simple scale factor fits but the inference code should not use the original values unchanged. `CHECK_REQUIRED` means the inputs must be examined before interpreting the visual error analysis.

## Outputs

- `results/data_consistency/data_consistency_summary.csv`: per-patch match rates, units, metadata and optional coordinate checks.
- `results/data_consistency/band_differences.csv`: per-band numerical differences.
- `assets/figures/data_consistency/band_differences.png`: chart of mean absolute spectral differences.

## Interpretation and limitations

This test verifies spectral equality and associated mask labels for a sample of annotated pixels. It does not prove geospatial projection correctness, exhaustively verify every patch, or establish that the trained model is scientifically valid. The nearest-neighbor check does not rely on HDF5 `XCoords`/`YCoords` being raster indices. If the HDF5 lacks matching patch metadata, the script stops that patch with `NO_HDF_ROWS` and prints example metadata for diagnosis.

Only after representative patches pass should the visual error analysis be reported as verified. If scaling or ordering differs, fix the input pipeline in `07_visual_error_analysis.py`, rerun the analysis, and regenerate the related figures and tables before publishing.
