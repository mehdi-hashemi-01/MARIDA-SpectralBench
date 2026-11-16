"""Check that MARIDA HDF5 pixel spectra agree with original GeoTIFF patches.

Run directly in PyCharm. Uses validation data only, does not retrain models.
No command-line arguments or custom classes are needed.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from sklearn.neighbors import NearestNeighbors

from settings import DATA_FILE, PROJECT_DIR, BANDS, CLASSES

# Two important failure cases identified in the visual error analysis.
PATCHES_TO_CHECK = [
    "S2_11-6-18_16PCC_26.tif",
    "S2_18-9-20_16PCC_39.tif",
    "S2_18-9-20_16PCC_29.tif",
]
SPLIT = "val"
MAX_ROWS_PER_PATCH = 200
HDF_CHUNK_SIZE = 50_000
# Small differences due to numeric precision are acceptable.
ABS_TOLERANCE = 2e-5
REL_TOLERANCE = 1e-4
PASS_RATE = 0.98

DATA_DIR = DATA_FILE.parent
PATCHES_DIR = DATA_DIR / "patches"
RESULT_DIR = PROJECT_DIR / "results" / "data_consistency"
FIGURE_DIR = PROJECT_DIR / "assets" / "figures" / "data_consistency"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_date(value):
    """Accept DD-MM-YY and YYYY-MM-DD, preserving day/month identity."""
    value = str(value).strip().lower().replace("s2_", "")
    value = value.split(" ")[0]
    parts = value.replace("/", "-").split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        a, b, c = (int(p) for p in parts)
        if len(parts[0]) == 4:  # ISO YYYY-MM-DD
            return f"{c}-{b}-{a % 100}"
        return f"{a}-{b}-{c % 100}"
    return value


def normalize_image_number(value):
    """Image can be 26, '26', '26.0', or an entire patch filename."""
    s = str(value).strip().lower().replace("\\", "/").split("/")[-1]
    s = s.removesuffix(".tif")
    s = s.split("_")[-1]
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        return str(int(s))
    return s


def normalize_class(value):
    """Convert a MARIDA class name or class ID into an integer from 1 to 15."""
    s = str(value).strip()
    if s in CLASSES:
        return CLASSES.index(s) + 1
    try:
        v = int(float(s))
        return v if 1 <= v <= len(CLASSES) else 0
    except ValueError:
        return 0


def load_tiff(path):
    array = np.asarray(tifffile.imread(path), dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"Not an 11-band image: {path.name}, shape={array.shape}")
    if array.shape[0] == len(BANDS):
        array = np.moveaxis(array, 0, -1)
    elif array.shape[2] != len(BANDS):
        raise ValueError(f"Unexpected band count: {path.name}, shape={array.shape}")
    label_file = path.with_name(path.stem + "_cl.tif")
    mask = np.asarray(tifffile.imread(label_file)).squeeze()
    if mask.shape != array.shape[:2]:
        raise ValueError(f"Different image and mask shapes: {path.name}")
    return array, mask


def find_hdf_rows(patch_name, hdf):
    """Find one patch in the official HDF5 val table using Date/Tile/Image."""
    stem = Path(patch_name).stem
    date, tile, number = stem.split("_")[1:]
    date = normalize_date(date)
    tile = tile.lower()
    number = normalize_image_number(number)
    matches = []
    with pd.HDFStore(hdf, mode="r") as store:
        if "/" + SPLIT not in store.keys():
            raise ValueError(f"HDF5 has no /{SPLIT} key: {store.keys()}")
        for chunk in store.select(SPLIT, chunksize=HDF_CHUNK_SIZE):
            required = set(BANDS + ["Class", "Date", "Tile", "Image"])
            if not required.issubset(chunk.columns):
                raise ValueError("Missing HDF5 columns: " + str(sorted(required - set(chunk.columns))))
            ok = (
                chunk["Date"].map(normalize_date).eq(date)
                & chunk["Tile"].astype(str).str.strip().str.lower().eq(tile)
                & chunk["Image"].map(normalize_image_number).eq(number)
            )
            if ok.any():
                matches.append(chunk.loc[ok].copy())
    if not matches:
        return pd.DataFrame()
    return pd.concat(matches, ignore_index=True)


def pixel_coordinate_check(hdf_rows, array, label_mask):
    """Try common pixel coordinate conventions, if HDF5 XCoords/YCoords are pixel indices.

    This is an optional, stricter check. If HDF5 holds map/geographic coordinates,
    it will return 'not_pixel_coordinates' rather than claiming a mismatch.
    """
    if not {"XCoords", "YCoords"}.issubset(hdf_rows.columns):
        return "not_available", 0, np.nan
    x = pd.to_numeric(hdf_rows["XCoords"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(hdf_rows["YCoords"], errors="coerce").to_numpy(dtype=float)
    if len(x) == 0 or not np.isfinite(x).any() or not np.isfinite(y).any():
        return "not_available", 0, np.nan
    h, w = label_mask.shape
    if not ((np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w + 1) & (y >= 0) & (y <= h + 1)).mean() > 0.9):
        return "not_pixel_coordinates", 0, np.nan
    # Also test common one-based indexing and reversed coordinate conventions.
    values = hdf_rows[BANDS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    classes = hdf_rows["Class"].map(normalize_class).to_numpy(dtype=int)
    best = ("no_match", 0, np.inf)
    for base in [0, 1]:
        for swapped in [False, True]:
            row = x - base if swapped else y - base
            col = y - base if swapped else x - base
            for flip_row in [False, True]:
                for flip_col in [False, True]:
                    r = (h - 1 - row) if flip_row else row
                    c = (w - 1 - col) if flip_col else col
                    valid = (np.isfinite(r) & np.isfinite(c)
                             & (np.abs(r - np.rint(r)) < 1e-7)
                             & (np.abs(c - np.rint(c)) < 1e-7)
                             & (r >= 0) & (r < h) & (c >= 0) & (c < w))
                    if valid.sum() < max(3, len(values) // 2):
                        continue
                    rr = r[valid].astype(int)
                    cc = c[valid].astype(int)
                    label_match = label_mask[rr, cc] == classes[valid]
                    spectrum_match = np.all(
                        np.isclose(array[rr, cc], values[valid], atol=ABS_TOLERANCE, rtol=REL_TOLERANCE),
                        axis=1,
                    )
                    matches = int(np.count_nonzero(label_match & spectrum_match))
                    label_rate = float(np.mean(label_match))
                    key = f"base={base},swapped={swapped},flip_row={flip_row},flip_col={flip_col}"
                    if matches > best[1]:
                        best = (key, matches, label_rate)
    return best


if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Missing HDF5: {DATA_FILE}")
if not PATCHES_DIR.is_dir():
    raise FileNotFoundError(f"Missing patches directory: {PATCHES_DIR}")

image_paths = {}
for file in PATCHES_DIR.rglob("*.tif"):
    if not file.name.lower().endswith(("_cl.tif", "_conf.tif")):
        image_paths[file.name.lower()] = file

print("HDF5:", DATA_FILE)
print("Checking", len(PATCHES_TO_CHECK), "validation patches")

summary_rows = []
band_rows = []
for patch_name in PATCHES_TO_CHECK:
    print("\n---", patch_name, "---")
    path = image_paths.get(patch_name.lower())
    if path is None:
        print("FAIL: TIFF not found. Check the patch name in PATCHES_TO_CHECK.")
        summary_rows.append({"patch": patch_name, "status": "MISSING_TIFF"})
        continue

    raw, truth = load_tiff(path)
    hdf_rows = find_hdf_rows(patch_name, DATA_FILE)
    if hdf_rows.empty:
        print("FAIL: no matching HDF5 rows. Inspect the Date, Tile and Image columns below:")
        with pd.HDFStore(DATA_FILE, mode="r") as store:
            example = store.select(SPLIT, start=0, stop=3)
            print(example[["Date", "Tile", "Image"]].to_string(index=False))
        summary_rows.append({"patch": patch_name, "status": "NO_HDF_ROWS"})
        continue

    if len(hdf_rows) > MAX_ROWS_PER_PATCH:
        hdf_rows = hdf_rows.sample(n=MAX_ROWS_PER_PATCH, random_state=42).reset_index(drop=True)

    hdf_values = hdf_rows[BANDS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    hdf_classes = hdf_rows["Class"].map(normalize_class).to_numpy(dtype=int)
    valid_hdf = np.isfinite(hdf_values).all(axis=1) & (hdf_classes >= 1)
    hdf_rows = hdf_rows.loc[valid_hdf].reset_index(drop=True)
    hdf_values = hdf_values[valid_hdf]
    hdf_classes = hdf_classes[valid_hdf]
    if not len(hdf_values):
        print("FAIL: no valid HDF5 spectral rows")
        summary_rows.append({"patch": patch_name, "status": "NO_VALID_HDF_SPECTRA"})
        continue

    annotated = (truth >= 1) & (truth <= len(CLASSES)) & np.isfinite(raw).all(axis=2)
    pixel_positions = np.flatnonzero(annotated)
    tiff_values = raw.reshape(-1, len(BANDS))[pixel_positions]
    tiff_classes = truth.ravel()[pixel_positions].astype(int)
    if not len(tiff_values):
        print("FAIL: no annotated finite pixels in GeoTIFF")
        summary_rows.append({"patch": patch_name, "status": "NO_VALID_TIFF_SPECTRA"})
        continue

    best = None
    for scale in [1.0, 0.0001, 10000.0]:
        scaled_tiff = tiff_values * scale
        nearest = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(scaled_tiff)
        _, indices = nearest.kneighbors(hdf_values)
        indices = indices[:, 0]
        paired = scaled_tiff[indices]
        diff = np.abs(hdf_values - paired)
        match = np.all(np.isclose(hdf_values, paired, atol=ABS_TOLERANCE, rtol=REL_TOLERANCE), axis=1)
        class_match = tiff_classes[indices] == hdf_classes
        candidate = {
            "scale": scale, "indices": indices, "paired": paired,
            "abs_diff": diff, "match": match, "class_match": class_match,
            "match_rate": float(match.mean()),
            "class_match_rate": float(class_match[match].mean()) if match.any() else 0.0,
            "median_max_diff": float(np.median(diff.max(axis=1))),
        }
        if best is None or (candidate["match_rate"], -candidate["median_max_diff"]) > (
            best["match_rate"], -best["median_max_diff"]
        ):
            best = candidate

    coord_mode, coord_matches, coord_label_rate = pixel_coordinate_check(hdf_rows, raw, truth)
    matched = best["match"]
    class_match = best["class_match"]
    aligned_matches = matched & class_match
    agreement = float(aligned_matches.mean())
    if best["scale"] == 1 and agreement >= PASS_RATE:
        status = "PASS"
    elif best["scale"] != 1 and agreement >= PASS_RATE:
        status = "SCALE_DIFFERENCE"
    else:
        status = "CHECK_REQUIRED"

    print("HDF5 rows checked:", len(hdf_values))
    print("GeoTIFF labeled pixels:", len(tiff_values))
    print("Best GeoTIFF multiplier:", best["scale"])
    print("Exact/near spectral match:", f'{best["match_rate"]:.1%}')
    print("Spectra + label agreement:", f"{agreement:.1%}")
    print("Coordinate check:", coord_mode, "matches:", coord_matches)
    print("STATUS:", status)
    summary_rows.append({
        "patch": patch_name, "status": status, "hdf_rows_checked": len(hdf_values),
        "tiff_annotated_finite_pixels": len(tiff_values),
        "best_tiff_multiplier": best["scale"],
        "spectral_match_rate": best["match_rate"],
        "spectral_and_label_agreement": agreement,
        "median_max_abs_band_difference": best["median_max_diff"],
        "coordinate_check": coord_mode, "coordinate_exact_matches": coord_matches,
        "coordinate_label_agreement": coord_label_rate,
    })
    for j, band in enumerate(BANDS):
        band_rows.append({
            "patch": patch_name, "band": band,
            "mean_absolute_difference": float(best["abs_diff"][:, j].mean()),
            "median_absolute_difference": float(np.median(best["abs_diff"][:, j])),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(RESULT_DIR / "data_consistency_summary.csv", index=False)
band_table = pd.DataFrame(band_rows)
band_table.to_csv(RESULT_DIR / "band_differences.csv", index=False)

if not band_table.empty:
    average = band_table.groupby("band")["mean_absolute_difference"].mean().reindex(BANDS)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(average.index, average.values)
    ax.set_title("HDF5 vs GeoTIFF — Mean absolute spectral difference")
    ax.set_xlabel("Spectral band")
    ax.set_ylabel("Mean absolute difference")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "band_differences.png", dpi=160)
    plt.show()
    plt.close(fig)

print("\n===== SUMMARY =====")
if len(summary):
    print(summary[["patch", "status", "spectral_match_rate", "spectral_and_label_agreement"]].to_string(index=False))
print("\nOutput:", RESULT_DIR)
print("Figure:", FIGURE_DIR)
print("Note: this check compares pixel spectra and labels; it does not retrain the model.")
