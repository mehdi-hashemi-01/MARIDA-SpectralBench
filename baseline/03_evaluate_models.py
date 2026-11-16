"""Step 3: compare models on validation, then test ONLY the selected model."""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from settings import DATA_FILE, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, CLASSES, CHUNK_SIZE
from spectral_features import clean_labels, make_features

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(split, saved_model, experiment_name):
    model = saved_model["model"]
    columns = saved_model["features"]
    medians = saved_model["medians"]
    total_cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)

    with pd.HDFStore(DATA_FILE, mode="r") as store:
        for chunk in store.select(split, chunksize=CHUNK_SIZE):
            truth = clean_labels(chunk["Class"])
            valid = np.isin(truth, CLASSES)
            if not valid.any():
                continue
            x = make_features(chunk.loc[valid])[columns]
            x = x.fillna(medians).fillna(0)
            prediction = model.predict(x)
            total_cm += confusion_matrix(truth[valid], prediction, labels=CLASSES)

    if total_cm.sum() == 0:
        raise ValueError("No valid pixels in " + split)

    cm_df = pd.DataFrame(total_cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(RESULTS_DIR / f"{experiment_name}_{split}_confusion.csv")

    tp = np.diag(total_cm).astype(float)
    fp = total_cm.sum(axis=0) - tp
    fn = total_cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)

    class_table = pd.DataFrame({
        "class": CLASSES, "precision": precision, "recall": recall,
        "f1": f1, "iou": iou, "support": total_cm.sum(axis=1),
    })
    class_table.to_csv(RESULTS_DIR / f"{experiment_name}_{split}_per_class.csv", index=False)
    debris = class_table.iloc[0]
    return {
        "model": experiment_name,
        "split": split,
        "marine_debris_f1": float(debris["f1"]),
        "marine_debris_iou": float(debris["iou"]),
        "marine_debris_precision": float(debris["precision"]),
        "marine_debris_recall": float(debris["recall"]),
        "overall_accuracy": float(tp.sum() / total_cm.sum()),
        "macro_f1_present_classes": float(f1[total_cm.sum(axis=1) > 0].mean()),
        "evaluated_pixels": int(total_cm.sum()),
    }


names = ["spectral_only", "spectral_plus_indices"]
validation_rows = []

for name in names:
    model_file = MODELS_DIR / f"{name}.joblib"
    if not model_file.is_file():
        raise FileNotFoundError("Model missing. Run 02_train_random_forest.py: " + str(model_file))
    print("Evaluating on validation:", name)
    saved_model = joblib.load(model_file)
    validation_rows.append(evaluate("val", saved_model, name))

validation = pd.DataFrame(validation_rows)
validation.to_csv(RESULTS_DIR / "validation_comparison.csv", index=False)
print("\nVALIDATION COMPARISON")
print(validation.to_string(index=False))

# We choose once using validation F1, and then evaluate the untouched test data.
best_name = validation.sort_values("marine_debris_f1", ascending=False).iloc[0]["model"]
(RESULTS_DIR / "selected_model.txt").write_text(best_name, encoding="utf-8")
print("\nSelected on validation:", best_name)

best_model = joblib.load(MODELS_DIR / f"{best_name}.joblib")
print("Evaluating the selected model on held-out TEST split...")
test_metrics = evaluate("test", best_model, best_name)
pd.DataFrame([test_metrics]).to_csv(RESULTS_DIR / "test_result.csv", index=False)
print("\nTEST RESULT")
print(pd.DataFrame([test_metrics]).to_string(index=False))
# Validation models and the selected held-out test are visually separated.
all_results = pd.concat([validation, pd.DataFrame([test_metrics])], ignore_index=True)
labels = [f"{row.model}\n({row.split})" for row in all_results.itertuples()]
fig, ax = plt.subplots(figsize=(10, 5))
pos = np.arange(len(all_results))
width = .38
ax.bar(pos - width / 2, all_results["marine_debris_f1"], width, label="F1")
ax.bar(pos + width / 2, all_results["marine_debris_iou"], width, label="IoU")
ax.set_xticks(pos, labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("MARIDA | Marine Debris performance")
ax.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "benchmark_scores.png", dpi=170)
plt.show()

# Selected model: 15-class normalized confusion matrix on the untouched test set.
cm = pd.read_csv(RESULTS_DIR / f"{best_name}_test_confusion.csv", index_col=0)
cm_values = cm.to_numpy(dtype=float)
cm_pct = np.divide(cm_values, cm_values.sum(axis=1, keepdims=True),
                   out=np.zeros_like(cm_values), where=cm_values.sum(axis=1, keepdims=True) > 0)
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm_pct, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(np.arange(len(CLASSES)), CLASSES, rotation=65, ha="right")
ax.set_yticks(np.arange(len(CLASSES)), CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Selected Random Forest | Test confusion (row-normalized)")
fig.colorbar(im, ax=ax, label="Row fraction")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "test_confusion_normalized.png", dpi=170)
plt.show()

per_class = pd.read_csv(RESULTS_DIR / f"{best_name}_test_per_class.csv")
fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(per_class["class"], per_class["f1"])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("F1")
ax.set_title("Selected Random Forest | Test F1 by class")
ax.grid(axis="x", alpha=.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "test_f1_by_class.png", dpi=170)
plt.show()
print("\nAll tables and figures saved in:", RESULTS_DIR)
