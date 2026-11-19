"""v0.8A - 5-seed repeat of the v0.7 Spectral-Only decoder on official TRAIN/VAL.

Every seed retrains the
model and picks its own checkpoint by official validation F1 (not a test score).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import PROJECT_DIR
from unet_core import get_split_paths, training_statistics
from v08_research_core import (SEEDS, ROOT, FIGURES, ensure_folders, train_one,
                               record_metrics, save_json)

SEEDS_TO_RUN = (42, 7, 21, 123, 2026)
MAX_EPOCHS = 30
PATIENCE = 8
BATCH_SIZE = 4
LEARNING_RATE = 3e-4


def main():
    ensure_folders()
    train_paths, train_names, lookup = get_split_paths('train')
    val_paths, val_names, _ = get_split_paths('val', lookup)
    if train_names & val_names:
        raise AssertionError('TRAIN/VAL overlap')
    mean, std, counts, class_weights, debris_flags = training_statistics(train_paths)
    folder = ROOT/'multiseed'
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder/'protocol.json', {
        'stage':'v0.8 multi-seed spectral-only', 'seeds':list(SEEDS_TO_RUN),
        'train_patches':len(train_paths), 'validation_patches':len(val_paths),
        'checkpoint_selection':'official validation F1 per seed; selection-set metric only',
        'model':'DofaDenseHead with zero DOFA tokens; raw spectral skip only',
        'epochs_max':MAX_EPOCHS,'batch_size':BATCH_SIZE,'patience':PATIENCE,
        'lr':LEARNING_RATE,'augment':False,'sampler':'debris patches x3, with replacement',
        'train_only_normalization':True, 'test_accessed':False,
        'historical_seed42_v07_f1':0.9008,
        'warning':'The historical v0.7 score is a comparison target, not a lock-step reproduction guarantee.'})
    rows=[]
    for seed in SEEDS_TO_RUN:
        print('\n=== SPECTRAL ONLY, SEED',seed,'===',flush=True)
        result=train_one('spectral_only',train_paths,val_paths,None,mean,std,
                         class_weights,debris_flags,seed,folder/f'seed_{seed}',
                         max_epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,patience=PATIENCE,
                         learning_rate=LEARNING_RATE)
        rows.append(result)
        pd.DataFrame(rows).to_csv(folder/'per_seed_validation.csv',index=False)
    frame=pd.DataFrame(rows).sort_values('seed')
    mean_f1=float(frame['debris_f1'].mean())
    sd_f1=float(frame['debris_f1'].std(ddof=1)) if len(frame)>1 else float('nan')
    summary={'model':'spectral_only','number_of_seeds':len(frame),
             'mean_validation_f1':mean_f1,
             'sample_sd_validation_f1':sd_f1 if np.isfinite(sd_f1) else None,
             'min_validation_f1':float(frame['debris_f1'].min()),
             'max_validation_f1':float(frame['debris_f1'].max()),
             'total_test_accesses':0,
             'caution':'SD across seeds is not a geographic confidence interval; best checkpoints are validation-selected.'}
    save_json(folder/'multiseed_summary.json',summary)
    plt.figure(figsize=(8.5,4.5))
    plt.scatter(frame['seed'].astype(str),frame['debris_f1'],s=90)
    plt.axhline(mean_f1,linestyle='--',label=f'Mean={mean_f1:.4f}; SD={sd_f1:.4f}')
    plt.axhline(0.9008,linestyle=':',label='v0.7 archived spectral-only (0.9008)')
    plt.ylim(0,1);plt.ylabel('Marine Debris F1 (official validation)')
    plt.xlabel('Random seed');plt.title('v0.8 Spectral-Only: Five-Seed Stability')
    plt.legend();plt.tight_layout()
    plt.savefig(FIGURES/'multiseed_spectral_f1.png',dpi=170);plt.close()
    print('PASS: 5-seed VALIDATION report written:',folder)
    print('Mean F1=',round(mean_f1,4),'sample SD=',round(sd_f1,4))
    print('NEXT: 28_tile_holdout_v08.py')

if __name__ == '__main__':
    main()
