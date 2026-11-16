"""Inspect the three HDF5 splits and save class-distribution figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from settings import DATA_FILE, RESULTS_DIR, FIGURES_DIR, BANDS, CLASSES, CHUNK_SIZE
from spectral_features import clean_labels

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")

rows = []
summary = []
with pd.HDFStore(DATA_FILE, "r") as store:
    for split in ["train", "val", "test"]:
        if "/" + split not in store.keys():
            raise ValueError(f"Missing HDF5 split: {split}")
        count = {name: 0 for name in CLASSES}
        total = 0
        for chunk in store.select(split, chunksize=CHUNK_SIZE):
            missing = set(BANDS + ["Class"]) - set(chunk.columns)
            if missing:
                raise ValueError(f"Missing columns in {split}: {missing}")
            labels = pd.Series(clean_labels(chunk["Class"]))
            total += len(chunk)
            for name, n in labels.value_counts().items():
                if name in count:
                    count[name] += int(n)
        summary.append({"split": split, "rows": total, "annotated_known_classes": sum(count.values())})
        for name in CLASSES:
            rows.append({"split": split, "class": name, "pixels": count[name]})
        print(f"{split}: {total:,} rows; {sum(count.values()):,} recognized labels")
        print(store.select(split, start=0, stop=3).to_string(index=False))

summary_df = pd.DataFrame(summary)
counts_df = pd.DataFrame(rows)
summary_df.to_csv(RESULTS_DIR / "dataset_splits.csv", index=False)
counts_df.to_csv(RESULTS_DIR / "dataset_class_counts.csv", index=False)

pivot = counts_df.pivot(index="class", columns="split", values="pixels").reindex(CLASSES)
fig, ax = plt.subplots(figsize=(12, 8))
y = np.arange(len(CLASSES))
for offset, split in zip([-0.26, 0, 0.26], ["train", "val", "test"]):
    ax.barh(y + offset, pivot[split].to_numpy(), height=0.25, label=split)
ax.set_yticks(y, CLASSES)
ax.invert_yaxis()
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("Annotated pixels (symmetric log scale)")
ax.set_title("MARIDA | Annotated pixel distribution by split")
ax.legend()
ax.grid(axis="x", alpha=.18)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "dataset_class_distribution.png", dpi=170)
plt.show()
print("Saved:", RESULTS_DIR / "dataset_class_counts.csv")
