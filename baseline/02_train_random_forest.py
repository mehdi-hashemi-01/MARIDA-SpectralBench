"""Step 2: train two simple Random Forest baselines on MARIDA."""
import joblib
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from settings import (
    DATA_FILE, MODELS_DIR, RESULTS_DIR, CHUNK_SIZE, RANDOM_SEED,
    TRAIN_SAMPLES_PER_CLASS, NUMBER_OF_TREES, CPU_JOBS,
    BANDS, CLASSES, INDEX_NAMES, FIGURES_DIR,
)
from spectral_features import clean_labels, make_features

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Uniform random sample within each class, obtained by streaming the full train split.
# Keeping the smallest random keys is an unbiased, reproducible sample.
rng = np.random.default_rng(RANDOM_SEED)
samples = pd.DataFrame()
counts = {name: 0 for name in CLASSES}

print("Reading training data in chunks...")
with pd.HDFStore(DATA_FILE, mode="r") as store:
    for chunk_number, chunk in enumerate(store.select("train", chunksize=CHUNK_SIZE), 1):
        chunk = chunk.copy()
        chunk["Class"] = clean_labels(chunk["Class"])
        chunk = chunk[chunk["Class"].isin(CLASSES)].copy()
        for name, total in chunk["Class"].value_counts().items():
            counts[name] += int(total)

        chunk["random_key"] = rng.random(len(chunk))
        small = chunk.sort_values("random_key").groupby("Class", sort=False).head(TRAIN_SAMPLES_PER_CLASS)
        samples = pd.concat([samples, small], ignore_index=True)
        samples = samples.sort_values("random_key").groupby("Class", sort=False).head(TRAIN_SAMPLES_PER_CLASS)
        samples = samples.reset_index(drop=True)

        if chunk_number % 10 == 0:
            print("Processed chunks:", chunk_number)

if samples.empty:
    raise ValueError("No valid labeled training pixels were found.")

sample_counts = samples["Class"].value_counts()
print("\nOriginal train class counts:")
print(pd.Series(counts))
print("\nSelected training sample counts:")
print(sample_counts)

pd.DataFrame({"all_train_pixels": pd.Series(counts), "sampled_train_pixels": sample_counts}).fillna(0).to_csv(
    RESULTS_DIR / "training_class_counts.csv"
)

counts_figure = pd.DataFrame({"all": pd.Series(counts), "sampled": sample_counts}).fillna(0).reindex(CLASSES)
fig, ax = plt.subplots(figsize=(11, 8))
y_pos = np.arange(len(CLASSES))
ax.barh(y_pos - .2, counts_figure["all"], height=.38, label="Original train")
ax.barh(y_pos + .2, counts_figure["sampled"], height=.38, label="Sampled for training")
ax.set_yticks(y_pos, CLASSES)
ax.invert_yaxis()
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("Pixels (symmetric log scale)")
ax.set_title("MARIDA | Training pixel sampling by class")
ax.legend()
ax.grid(axis="x", alpha=.18)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "training_sampling.png", dpi=170)
plt.show()

all_features = make_features(samples)
y_train = samples["Class"].to_numpy()

# Optional confidence weights: uncertain annotations contribute less to training.
if "Confidence" in samples.columns:
    confidence = samples["Confidence"].astype(str)
    weights = confidence.map({"High": 1.0, "Moderate": 0.5, "Low": 1 / 3,
                              "1": 1.0, "2": 0.5, "3": 1 / 3}).fillna(1.0).to_numpy()
else:
    weights = np.ones(len(samples))

experiments = {
    "spectral_only": BANDS,
    "spectral_plus_indices": BANDS + INDEX_NAMES,
}

training_rows = []
for experiment_name, features in experiments.items():
    start_time = time.perf_counter()
    print("\nTraining:", experiment_name, "| Features:", len(features))
    x_train = all_features[features]

    # The median is learned ONLY from train, and saved for future predictions.
    medians = x_train.median().fillna(0)
    x_train = x_train.fillna(medians).fillna(0)

    model = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=CPU_JOBS,
    )
    model.fit(x_train, y_train, sample_weight=weights)

    output_file = MODELS_DIR / f"{experiment_name}.joblib"
    joblib.dump({"model": model, "features": features, "medians": medians.to_dict()}, output_file)
    duration = time.perf_counter() - start_time
    training_rows.append({"model": experiment_name, "features": len(features),
                          "training_pixels": len(samples), "seconds": round(duration, 3)})
    print("Saved:", output_file)

times = pd.DataFrame(training_rows)
times.to_csv(RESULTS_DIR / "training_runtime.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(times["model"], times["seconds"])
ax.set_ylabel("Training time (seconds)")
ax.set_title("Random Forest | Training runtime")
ax.tick_params(axis="x", rotation=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "training_runtime.png", dpi=170)
plt.show()
print("\nTraining finished. Run 03_evaluate_models.py next.")
