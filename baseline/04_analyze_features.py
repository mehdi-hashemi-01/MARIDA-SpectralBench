"""Step 4: simple permutation feature importance on VALIDATION only."""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from settings import (
    DATA_FILE, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, CHUNK_SIZE,
    RANDOM_SEED, IMPORTANCE_SAMPLES_PER_CLASS, CLASSES,
)
from spectral_features import clean_labels, make_features

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
selected_file = RESULTS_DIR / "selected_model.txt"
if not selected_file.is_file():
    raise FileNotFoundError("Run 03_evaluate_models.py first.")

name = selected_file.read_text(encoding="utf-8").strip()
saved = joblib.load(MODELS_DIR / f"{name}.joblib")
model = saved["model"]
features = saved["features"]
medians = saved["medians"]

# Keep a small, reproducible random validation sample PER CLASS.
rng = np.random.default_rng(RANDOM_SEED)
samples = pd.DataFrame()
with pd.HDFStore(DATA_FILE, "r") as store:
    for chunk in store.select("val", chunksize=CHUNK_SIZE):
        chunk = chunk.copy()
        chunk["Class"] = clean_labels(chunk["Class"])
        chunk = chunk[chunk["Class"].isin(CLASSES)].copy()
        chunk["random_key"] = rng.random(len(chunk))
        small = chunk.sort_values("random_key").groupby("Class", sort=False).head(IMPORTANCE_SAMPLES_PER_CLASS)
        samples = pd.concat([samples, small], ignore_index=True)
        samples = samples.sort_values("random_key").groupby("Class", sort=False).head(IMPORTANCE_SAMPLES_PER_CLASS)
        samples = samples.reset_index(drop=True)

if samples.empty or not (samples["Class"] == "Marine Debris").any():
    raise ValueError("No labeled Marine Debris pixels in validation sample.")

y_true = samples["Class"].to_numpy()
x = make_features(samples)[features].fillna(medians).fillna(0)
original_prediction = model.predict(x)
original_f1 = f1_score(y_true == "Marine Debris", original_prediction == "Marine Debris", zero_division=0)

rows = []
for feature in features:
    changed = x.copy()
    changed[feature] = rng.permutation(changed[feature].to_numpy())
    changed_prediction = model.predict(changed)
    new_f1 = f1_score(y_true == "Marine Debris", changed_prediction == "Marine Debris", zero_division=0)
    rows.append({"feature": feature, "f1_drop_after_permutation": original_f1 - new_f1})

results = pd.DataFrame(rows).sort_values("f1_drop_after_permutation", ascending=False)
results.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
print("Chosen model:", name)
print("Original validation-sample Marine Debris F1:", round(original_f1, 4))
print(results.to_string(index=False))

plt.figure(figsize=(9, 6))
plt.barh(results["feature"], results["f1_drop_after_permutation"])
plt.gca().invert_yaxis()
plt.xlabel("Change in Marine Debris F1 when feature is shuffled")
plt.title("Permutation feature importance (validation sample)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
plt.show()

# Top ten strongest absolute perturbations (signed effect retained).
top = results.assign(magnitude=results["f1_drop_after_permutation"].abs()).nlargest(10, "magnitude")
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top["feature"], top["f1_drop_after_permutation"])
ax.axvline(0, linewidth=.8)
ax.invert_yaxis()
ax.set_xlabel("F1 change after feature permutation")
ax.set_title("Top 10 | Largest absolute permutation effects")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "feature_effects_top10.png", dpi=170)
plt.show()

print("\nNote: these are model-reliance diagnostics, NOT causal or mechanistic explanations.")
print("Correlated spectral bands can make this measure misleading.")
