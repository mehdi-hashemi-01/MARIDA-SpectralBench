"""Quick sanity check for spectral feature functions (run in PyCharm)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from spectral_features import make_features, safe_divide
from settings import BANDS, INDEX_NAMES

sample = pd.DataFrame([{band: 0.2 for band in BANDS}])
sample["nm665"] = .1
sample["nm842"] = .3
features = make_features(sample)
assert set(BANDS + INDEX_NAMES).issubset(features.columns)
assert abs(features.loc[0, "NDVI"] - .5) < 1e-10
assert np.isnan(safe_divide(np.array([1]), np.array([0]))[0])
print("Spectral feature checks passed.")
