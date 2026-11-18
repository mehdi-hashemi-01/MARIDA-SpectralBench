"""v0.7A: evaluate all three DOFA input-source controls on identical VAL pixels.

Use after 23_train_dofa_source_controls.py; official TEST is not accessed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from settings import CLASSES
from unet_core import (NUM_CLASSES, IGNORE, metrics_from_cm, confusion_from_batch,
                       training_statistics, get_split_paths)
from dofa_transfer import DofaDenseHead, FrozenDofaDataset
from dofa_v07_common import (prepare_splits, apply_sources, RESULTS,
                             FIGURES, MODELS, cm_to_rows)

MODES = ['combined', 'dofa_only', 'spectral_only']
BATCH_SIZE = 4

if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; use the dl environment on the RTX 5070')
device = torch.device('cuda')
train_paths, val_paths, mean, std, counts, class_weights, flags, meta = prepare_splits()
val_ds = FrozenDofaDataset(val_paths, 'val', mean, std, augment=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
rows, class_rows, patch_rows = [], [], []
for mode in MODES:
    file = MODELS / f'{mode}_best.pt'
    if not file.exists():
        raise FileNotFoundError('Missing '+str(file)+'; run 23 first')
    ckpt = torch.load(file, map_location='cpu', weights_only=True)
    if ckpt.get('pretrained_sha256') != meta['checkpoint_sha256'] or ckpt.get('mode') != mode:
        raise RuntimeError('Checkpoint/cached DOFA mismatch: ' + mode)
    head = DofaDenseHead().to(device)
    head.load_state_dict(ckpt['head_state_dict']); head.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    with torch.inference_mode():
        for x, feat, y, conf, paths in val_loader:
            with torch.autocast('cuda', dtype=torch.float16):
                predicted = apply_sources(head, feat.to(device), x.to(device), mode).argmax(1)
            confusion_from_batch(cm, predicted, y)
            pred_np = predicted.cpu().numpy(); true_np = y.numpy()
            for i, path in enumerate(paths):
                valid = true_np[i] != IGNORE
                local = np.bincount(NUM_CLASSES*true_np[i][valid] + pred_np[i][valid],
                                    minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
                metric = metrics_from_cm(local)
                patch_rows.append({'mode': mode, 'patch': Path(path).name,
                                   'labeled_pixels': int(local.sum()),
                                   'true_debris': int(local[0].sum()),
                                   'tp': metric['tp'], 'fp': metric['fp'], 'fn': metric['fn'],
                                   'debris_f1': metric['debris_f1']})
    metrics = metrics_from_cm(cm)
    if len(val_paths) != 328 or metrics['evaluated_pixels'] != 213102:
        raise RuntimeError('Unexpected MARIDA validation coverage: '+str(metrics['evaluated_pixels']))
    per_patch = pd.DataFrame([r for r in patch_rows if r['mode']==mode])
    for k in ['tp','fp','fn']:
        assert int(per_patch[k].sum()) == metrics[k]
    rows.append({'mode': mode, 'best_epoch': ckpt['best_epoch'],
                 'debris_f1': metrics['debris_f1'], 'debris_precision': metrics['debris_precision'],
                 'debris_recall': metrics['debris_recall'], 'debris_iou': metrics['debris_iou'],
                 'macro_f1_present': metrics['macro_f1_present'],
                 'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
                 'evaluated_pixels': metrics['evaluated_pixels'],
                 'clouds_as_debris_fp': int(cm[CLASSES.index('Clouds'),0]),
                 'waves_as_debris_fp': int(cm[CLASSES.index('Waves'),0]),
                 'sediment_as_debris_fp': int(cm[CLASSES.index('Sediment-Laden Water'),0]),
                 'ship_as_debris_fp': int(cm[CLASSES.index('Ship'),0])})
    class_rows.extend(cm_to_rows(cm, mode))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(RESULTS/f'{mode}_validation_confusion.csv')
    print(mode, 'F1',round(metrics['debris_f1'],4),'FP',metrics['fp'],'FN',metrics['fn'])
    del head
summary = pd.DataFrame(rows).sort_values('debris_f1', ascending=False)
summary.to_csv(RESULTS/'source_controls_validation_comparison.csv', index=False)
pd.DataFrame(class_rows).to_csv(RESULTS/'source_controls_by_class.csv', index=False)
pd.DataFrame(patch_rows).to_csv(RESULTS/'source_controls_patch_errors.csv', index=False)

fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].bar(summary['mode'], summary['debris_f1'], color=['#476a86','#7c9f9e','#d6a34b'])
axes[0].set_ylim(0,1); axes[0].set_title('DOFA input-source controls / official VAL')
axes[0].set_ylabel('Marine Debris F1 (labeled pixels only)')
width=.34; x=np.arange(len(summary))
axes[1].bar(x-width/2, summary['fp'], width, label='FP')
axes[1].bar(x+width/2, summary['fn'], width, label='FN')
axes[1].set_xticks(x,summary['mode'],rotation=15); axes[1].legend()
axes[1].set_title('FP and FN on same 213,102 annotated pixels')
fig.tight_layout(); fig.savefig(FIGURES/'source_controls_summary.png',dpi=170)
plt.close(fig)

with open(RESULTS/'source_controls_findings.md','w',encoding='utf-8') as out:
    out.write('# DOFA source controls — Validation findings\n\n')
    out.write('All three variants reuse the v0.6 decoder architecture; one input source is zeroed where needed. '
              'Three independent fits (one seed) on identical official Train/Validation splits. '
              '**No TEST evaluation.** DOFA pretrained backbone is frozen.\n\n')
    out.write('| Mode | F1 | Precision | Recall | IoU | TP | FP | FN |\n')
    out.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for _, r in summary.iterrows():
        out.write(f'| {r["mode"]} | {r["debris_f1"]:.4f} | {r["debris_precision"]:.4f} | {r["debris_recall"]:.4f} | {r["debris_iou"]:.4f} | {int(r["tp"])} | {int(r["fp"])} | {int(r["fn"])} |\n')
    out.write('\n\nThese controls measure the contribution of supplied inputs to this decoder in one training run. '
              'They are not a random-initialized-encoder control and do not alone establish a causal advantage from pretraining.\n')
print('Saved:',RESULTS,'Figures:',FIGURES)
print('NEXT: review results BEFORE investing GPU time in LoRA')
