"""Summarize errors from the selected test confusion matrix, without refitting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from settings import RESULTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
selected = (RESULTS_DIR / "selected_model.txt").read_text(encoding="utf8").strip()
cm = pd.read_csv(RESULTS_DIR / f"{selected}_test_confusion.csv", index_col=0)

# Each row: ground truth; each column: predicted label.
labels = cm.index.tolist()
debris = "Marine Debris"
false_positives = cm.loc[cm.index != debris, debris].sort_values(ascending=False)
false_negatives = cm.loc[debris, cm.columns != debris].sort_values(ascending=False)

tp = int(cm.loc[debris, debris])
fp = int(false_positives.sum())
fn = int(false_negatives.sum())
pd.DataFrame([{"tp": tp, "fp": fp, "fn": fn,
               "precision": tp / (tp + fp) if tp + fp else 0,
               "recall": tp / (tp + fn) if tp + fn else 0}]).to_csv(
    RESULTS_DIR / "marine_debris_error_summary.csv", index=False)

false_positives.rename_axis("true_class").reset_index(name="count").to_csv(
    RESULTS_DIR / "false_positives_by_class.csv", index=False)
false_negatives.rename_axis("predicted_class").reset_index(name="count").to_csv(
    RESULTS_DIR / "false_negatives_by_class.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 7))
false_positives[false_positives > 0].head(10).sort_values().plot.barh(ax=ax)
ax.set_xlabel("False-positive pixels predicted as Marine Debris")
ax.set_ylabel("True class")
ax.set_title("Test error analysis | Marine Debris false positives")
ax.grid(axis="x", alpha=.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "test_false_positives.png", dpi=170)
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
false_negatives[false_negatives > 0].head(10).sort_values().plot.barh(ax=ax)
ax.set_xlabel("Missed Marine Debris pixels")
ax.set_ylabel("Predicted class")
ax.set_title("Test error analysis | Marine Debris false negatives")
ax.grid(axis="x", alpha=.2)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "test_false_negatives.png", dpi=170)
plt.show()
print("Marine Debris: TP =", tp, "FP =", fp, "FN =", fn)
print("Saved error tables and figures in:", RESULTS_DIR)
