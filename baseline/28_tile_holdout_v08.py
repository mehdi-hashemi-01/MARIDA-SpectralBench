"""v0.8B - nested leave-one-Tile-ID-out comparison, ONLY official Train.

Per outer Tile: reserve its patches, then form a Date+Tile grouped inner
validation strictly from remaining Train patches. Fit normalizers on the
remaining inner-FIT patches. Epoch selection occurs on INNER VALIDATION ONLY.
The unseen OUTER TILE is predicted ONCE with the selected checkpoint.

PyCharm Run. 13 folds x 3 models may be computationally expensive. The script
resumes completed fold/model outputs. Full run is required for pooled results.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from settings import CLASSES
from unet_core import get_split_paths, training_statistics, NUM_CLASSES
from dofa_transfer import load_real_pretrained_dofa
from v08_research_core import (ROOT, FIGURES, CACHE, ensure_folders,
    tile_id, acquisition_id, split_inner_by_acquisition, cache_features,
    train_one, save_json, record_metrics)

SEED = 42
MODELS_TO_RUN = ('spectral_only','unet','dofa_combined')
FOLDS_TO_RUN = 'all'             # e.g. ('16PDC',) for diagnostic; not complete benchmark.
MAX_EPOCHS = 30
BATCH_SIZE = 4
DOFA_CACHE_BATCH_SIZE = 2
PATIENCE = 8
LEARNING_RATE = 3e-4


def audit_fold(paths, outer, fit, inner):
    outer_set=set(map(str,outer)); fit_set=set(map(str,fit)); inner_set=set(map(str,inner))
    if fit_set & inner_set or outer_set & (fit_set | inner_set):
        raise AssertionError('Fold patch leakage')
    if outer_set | fit_set | inner_set != set(map(str,paths)):
        raise AssertionError('Fold does not cover exact official training manifest')
    if {tile_id(p) for p in outer} & {tile_id(p) for p in fit+inner}:
        raise AssertionError('Outer Tile ID shared with fitting or inner validation')


def main():
    ensure_folders()
    # Do not invoke get_split_paths('val') or get_split_paths('test').
    paths, names, _ = get_split_paths('train')
    grouped=defaultdict(list)
    for p in paths:
        grouped[tile_id(p)].append(p)
    tile_ids=sorted(grouped)
    selected=tile_ids if FOLDS_TO_RUN=='all' else list(FOLDS_TO_RUN)
    unknown=set(selected)-set(tile_ids)
    if unknown:raise ValueError('Unknown Train Tile IDs: '+str(sorted(unknown)))
    folder=ROOT/'geographic';folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'protocol.json',{
        'benchmark':'v0.8 grouped outer Tile ID from official TRAIN only',
        'all_tile_ids':tile_ids,'requested_tile_ids':selected,
        'seed':SEED,'models':MODELS_TO_RUN,'total_train_patches':len(paths),
        'inner_selection':'Date+Tile group-disjoint selection from remaining Train patches',
        'preprocessing':'inner-fit mean/std and class weights, never outer labels',
        'DOFA':'real frozen ViT-B + raw skip; inner-fit-only normalization; per-fold cache',
        'model_selection':'inner group validation F1 only',
        'outer_scoring':'once after checkpoint selection for each model',
        'do_not_use_official_val_or_test':True,
        'not_verified':'Tile ID separation is not an independent geospatial footprint audit',
        'epochs_max':MAX_EPOCHS,'batch_size':BATCH_SIZE,
        'patience':PATIENCE,'learning_rate':LEARNING_RATE,
        'augment':False,'sampler':'debris-positive train patches x3 with replacement',
        'interpretation':'Original v0.1 RF grouped results have different training protocol: descriptive only.'})
    encoder=None; sha=None
    all_results=[]
    for fold_index, heldout in enumerate(selected,1):
        print(f'\n=== OUTER TILE {heldout}, {fold_index}/{len(selected)} ===',flush=True)
        outer=list(grouped[heldout]);outer_fit=[p for p in paths if tile_id(p)!=heldout]
        fit,inner,inner_group_ids=split_inner_by_acquisition(outer_fit,heldout)
        audit_fold(paths,outer,fit,inner)
        mean,std,counts,class_weights,flags=training_statistics(fit)
        fold_dir=folder/heldout;fold_dir.mkdir(parents=True,exist_ok=True)
        save_json(fold_dir/'fold_manifest.json',{
            'outer_tile':heldout, 'outer_files':[p.name for p in outer],
            'fit_files':[p.name for p in fit],
            'inner_files':[p.name for p in inner],
            'inner_groups':inner_group_ids,
            'fit_positive_pixel_count':int(counts[0]),
            'outer_positive_pixel_count':'not_read_until_final_outer_scoring',
            'fit_only_mean':mean.tolist(),'fit_only_std':std.tolist(),
            'test_accessed':False})
        for model_name in MODELS_TO_RUN:
            print('  MODEL:',model_name,flush=True)
            run_dir=fold_dir/model_name
            features=None
            if model_name=='dofa_combined':
                if encoder is None:
                    encoder,sha=load_real_pretrained_dofa()
                all_fold_paths=fit+inner+outer
                features=cache_features(encoder,all_fold_paths,mean,std,
                    CACHE/'geographic'/heldout,{
                        'source':'DOFA real pretrained','sha256':sha,
                        'outer_tile':heldout,
                        'fit_names':[p.name for p in fit],
                        'inner_names':[p.name for p in inner]},
                    batch_size=DOFA_CACHE_BATCH_SIZE)
                # Avoid keeping an unused 86M-param model on GPU during decoder fitting.
                encoder.cpu()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            result=train_one(model_name,fit,inner,outer,mean,std,class_weights,flags,
                SEED,run_dir,feature_root=features,max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,patience=PATIENCE,learning_rate=LEARNING_RATE)
            result['outer_tile']=heldout
            result['outer_debris_support']=result['debris_support']
            all_results.append(result)
            pd.DataFrame(all_results).to_csv(folder/'per_fold_partial.csv',index=False)
    # Never present subset folds as complete pooled benchmark.
    complete=set(selected)==set(tile_ids) and all(
        (folder/tile/m/'metrics.json').is_file()
        for tile in tile_ids for m in MODELS_TO_RUN)
    if not complete:
        print('Partial fold run. Full pooled comparison intentionally not produced.')
        return
    rows=[]; summary=[]
    for m in MODELS_TO_RUN:
        pooled=np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.int64)
        model_rows=[]
        for tile in tile_ids:
            cm=pd.read_csv(folder/tile/m/'confusion.csv',index_col=0).to_numpy(dtype=np.int64)
            if cm.shape!=(NUM_CLASSES,NUM_CLASSES):raise ValueError('Malformed confusion for '+tile+'/'+m)
            pooled+=cm
            metrics=json.loads((folder/tile/m/'metrics.json').read_text(encoding='utf-8'))
            metrics['outer_tile']=tile
            model_rows.append(metrics);rows.append(metrics)
        pd.DataFrame(pooled,index=CLASSES,columns=CLASSES).to_csv(folder/f'{m}_pooled_confusion.csv')
        if int(pooled.sum()) != 429412 or int(pooled[0, :].sum()) != 1943:
            raise AssertionError('Unexpected official-Train annotated-pixel or debris support; '
                                 'expected 429,412 labeled pixels / 1,943 debris. '
                                 'Check split and class-mask mapping before publication.')
        pooled_metrics=record_metrics(pooled,{'model':m,'fold_count':len(tile_ids),
            'scope':'only official Train; leave-one-Tile-ID-out',
            'outer_selection_free':True,'test_accessed':False})
        positive=[float(r['debris_f1']) for r in model_rows
                  if r['debris_support']>0 and r['debris_f1'] is not None]
        pooled_metrics['macro_tile_f1_positive_tiles']=float(np.mean(positive)) if positive else None
        pooled_metrics['positive_support_tile_count']=len(positive)
        pooled_metrics['debris_absent_tile_count']=len(tile_ids)-len(positive)
        summary.append(pooled_metrics)
    pd.DataFrame(rows).to_csv(folder/'per_fold_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(folder/'pooled_metrics.csv',index=False)
    save_json(folder/'complete.json',{'all_folds_finished':True,'folds':tile_ids,
        'each_patch_scored_exactly_once_per_model':True,
        'outer_tile_not_used_for_checkpoint_selection':True,
        'warning':'Tile IDs used as geographic proxy, footprints not externally audited.'})
    plt.figure(figsize=(9,4.8))
    for m in MODELS_TO_RUN:
        r=next(z for z in summary if z['model']==m)
        plt.bar(m,float(r['debris_f1']),label=m)
    plt.ylim(0,1);plt.ylabel('Pooled Marine Debris F1')
    plt.title('v0.8 Leave-One-Tile-ID-Out — Official Train Only')
    plt.tight_layout();plt.savefig(FIGURES/'geographic_pooled_f1.png',dpi=170);plt.close()
    print('PASS: all geographic folds complete:',folder/'pooled_metrics.csv')
    print('NEXT: 29_random_dofa_control_v08.py or 30_research_report_v08.py')

if __name__=='__main__':
    main()
