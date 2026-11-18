"""v0.7A: controlled source ablation for DOFA v0.6's exact segmentation head.

uses only TRAIN/VAL, never TEST. Three separate fits:
  combined       = pretrained DOFA features + normalized 11-band spectral skip
  dofa_only      = pretrained features, raw skip explicitly zeroed
  spectral_only  = raw skip, pretrained features explicitly zeroed

Using the same DofaDenseHead retains parameter count/architecture in all arms.
"""
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from settings import BANDS, CLASSES, RANDOM_SEED
from unet_core import seed_everything, metrics_from_cm
from dofa_transfer import DofaDenseHead
from dofa_v07_common import (prepare_splits, cached_loaders, apply_sources,
                             masked_weighted_ce, evaluate_head, RESULTS, FIGURES,
                             MODELS, save_json)

# ----- Easy-to-edit PyCharm settings -----
MODES = ['combined', 'dofa_only', 'spectral_only']
EPOCHS = 30
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 8
FORCE_RETRAIN = False
# -----------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError('A CUDA GPU is required. Run 15_check_gpu_and_data.py')
device = torch.device('cuda')
train_paths, val_paths, mean, std, counts, class_weights, debris, meta = prepare_splits()
print('DOFA pretrained SHA256:', meta['checkpoint_sha256'])
print('Three source controls; official TEST never read')

for mode in MODES:
    checkpoint = MODELS / (mode + '_best.pt')
    if checkpoint.exists() and not FORCE_RETRAIN:
        prior = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if prior.get('mode') != mode or prior.get('pretrained_sha256') != meta['checkpoint_sha256']:
            raise RuntimeError('Existing control checkpoint has mismatching mode/DOFA SHA: ' + str(checkpoint))
        print('SKIP completed mode:', mode, '| set FORCE_RETRAIN=True to replace')
        continue
    seed_everything(RANDOM_SEED)  # Same random head initialization in all three arms.
    train_loader, val_loader = cached_loaders(train_paths, val_paths, mean, std, debris, BATCH_SIZE)
    head = DofaDenseHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)
    scaler = torch.amp.GradScaler('cuda')
    weight_t = torch.tensor(class_weights, device=device)
    best, best_epoch, stale, history = -1, 0, 0, []
    started = time.perf_counter()
    print('\n===== Training', mode, '=====', flush=True)
    for epoch in range(1, EPOCHS+1):
        head.train()
        num, den = 0., 0.
        for x, feat, y, conf, paths in train_loader:
            x, feat = x.to(device), feat.to(device)
            y, conf = y.to(device), conf.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = apply_sources(head, feat, x, mode)
            loss, local_num, local_den = masked_weighted_ce(logits, y, conf, weight_t)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss in mode ' + mode)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
            num += local_num; den += local_den
        cm, val_ce = evaluate_head(head, val_loader, mode, device)
        m = metrics_from_cm(cm)
        score = float(m['debris_f1'])
        if not np.isfinite(score):
            raise ValueError('No positive-class validation score for ' + mode)
        scheduler.step(score)
        record = {'mode': mode, 'epoch': epoch, 'train_weighted_ce': num/max(den,1),
                  'val_unweighted_ce': val_ce, 'val_debris_f1': score,
                  'val_precision': m['debris_precision'], 'val_recall': m['debris_recall'],
                  'val_fp': m['fp'], 'val_fn': m['fn'], 'lr': opt.param_groups[0]['lr']}
        history.append(record)
        pd.DataFrame(history).to_csv(RESULTS / f'{mode}_history.csv', index=False)
        print(f'{mode} epoch={epoch:02d} F1={score:.4f} FP={m["fp"]} FN={m["fn"]}', flush=True)
        if score > best + 1e-5:
            best, best_epoch, stale = score, epoch, 0
            torch.save({'head_state_dict': head.state_dict(), 'mode': mode,
                        'best_epoch': epoch, 'best_val_f1': score,
                        'pretrained_sha256': meta['checkpoint_sha256'],
                        'train_mean': mean.tolist(), 'train_std': std.tolist(),
                        'seed': RANDOM_SEED}, checkpoint)
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(RESULTS / f'{mode}_best_confusion.csv')
        else:
            stale += 1
        if stale >= PATIENCE:
            print('Early stopping:', mode)
            break
    print(f'Finished {mode} best F1={best:.4f} epoch={best_epoch}; elapsed={time.perf_counter()-started:.1f}s')
    del head, opt, train_loader, val_loader
    torch.cuda.empty_cache()

save_json(RESULTS / 'source_controls_config.json', {
    'purpose': 'Isolate predictive contribution of DOFA frozen tokens vs spectral skip',
    'modes': MODES, 'seed': RANDOM_SEED, 'epochs_max': EPOCHS,
    'batch_size': BATCH_SIZE, 'learning_rate': LR, 'weight_decay': WEIGHT_DECAY,
    'patience': PATIENCE, 'train_patches': len(train_paths), 'validation_patches': len(val_paths),
    'pretrained_sha256': meta['checkpoint_sha256'], 'loss': 'class- and confidence-weighted masked CE',
    'sampler': 'replacement, debris-positive weight=3; same generator seed each arm',
    'equal_head_parameter_count': True, 'test_accessed': False,
})
print('NEXT: run 24_analyze_dofa_source_controls.py')
