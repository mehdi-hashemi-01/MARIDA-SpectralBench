"""Feature calculations shared by training, evaluation and interpretation."""
import numpy as np
import pandas as pd
from settings import BANDS, CLASSES


def clean_labels(values):
    """Accept the official text class labels or numeric IDs 1..15."""
    labels = pd.Series(values).astype(str)
    mapping = {str(i + 1): name for i, name in enumerate(CLASSES)}
    return labels.replace(mapping).to_numpy()


def safe_divide(top, bottom):
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    return np.divide(
        top, bottom, out=np.full(len(top), np.nan),
        where=np.abs(bottom) > 1e-6,
    )


def make_features(df):
    """Return 11 spectral bands plus 8 hand-crafted spectral indices."""
    x = df[BANDS].apply(pd.to_numeric, errors="coerce").copy()
    blue = x["nm490"].to_numpy()
    green = x["nm560"].to_numpy()
    red = x["nm665"].to_numpy()
    red_edge = x["nm740"].to_numpy()
    nir = x["nm842"].to_numpy()
    swir = x["nm1600"].to_numpy()

    x["NDVI"] = safe_divide(nir - red, nir + red)
    x["FAI"] = nir - (red + (swir - red) * ((833 - 665) / (1614 - 665)))
    x["FDI"] = nir - (red_edge + 10 * (swir - red_edge) * ((833 - 740) / (1614 - 740)))
    x["SI"] = np.cbrt((1 - blue) * (1 - green) * (1 - red))
    x["NDWI"] = safe_divide(green - nir, green + nir)
    x["NRD"] = nir - red
    x["NDMI"] = safe_divide(nir - swir, nir + swir)
    x["BSI"] = safe_divide((swir + red) - (nir + blue), (swir + red) + (nir + blue))

    return x.replace([np.inf, -np.inf], np.nan)
