"""Project settings: edit numbers here and run scripts directly in PyCharm."""
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASELINE_DIR.parent

DATA_FILE = BASELINE_DIR / "data" / "dataset.h5"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results" / "runs" / "latest"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_SEED = 42
CHUNK_SIZE = 50_000
TRAIN_SAMPLES_PER_CLASS = 2_000
IMPORTANCE_SAMPLES_PER_CLASS = 100
NUMBER_OF_TREES = 100
CPU_JOBS = -1

BANDS = [
    "nm440", "nm490", "nm560", "nm665", "nm705", "nm740",
    "nm783", "nm842", "nm865", "nm1600", "nm2200",
]

CLASSES = [
    "Marine Debris", "Dense Sargassum", "Sparse Sargassum",
    "Natural Organic Material", "Ship", "Clouds", "Marine Water",
    "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water",
    "Waves", "Cloud Shadows", "Wakes", "Mixed Water",
]

INDEX_NAMES = ["NDVI", "FAI", "FDI", "SI", "NDWI", "NRD", "NDMI", "BSI"]
