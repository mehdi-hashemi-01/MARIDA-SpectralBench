"""v0.8C: paired pretrained-vs-random frozen DOFA token control (official Train/Val).

This is a clean *pretraining* contrast for the DOFA-only head:
- Same 11-band normalization, patch sample order, head architecture and seeds.
- Encoder weights fixed in BOTH conditions.
- Spectral skip is ZEROED in BOTH conditions.
- Random branch uses the same upstream DOFA architecture with pretrained=False
  and NEVER loads the official checkpoint into that branch.

This uses official validation for checkpoint selection, NEVER official test.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from unet_core import get_split_paths, training_statistics, seed_everything
from dofa_transfer import (cache_config, verify_cache_metadata, feature_path,
    check_cached_feature, load_real_pretrained_dofa, MODEL_REPO, WEIGHT_FILE)
from v08_research_core import (ROOT, FIGURES, CACHE, ensure_folders, train_one,
    cache_features, save_json)

SEEDS_TO_RUN=(42,7,21)      # Matched decoder seeds; increase after first results if needed.
RANDOM_ENCODER_SEED=20261119  # Fixed once across all decoder seeds.
EPOCHS=30
PATIENCE=8
BATCH_SIZE=4
CACHE_BATCH_SIZE=2
LEARNING_RATE=3e-4


def random_encoder_fingerprint(model):
    """Diagnostic fingerprint of a handful of actual initialized tensors."""
    sha=hashlib.sha256()
    selected=0
    for name,p in model.named_parameters():
        if any(key in name for key in ('pos_embed','cls_token','blocks.0.attn.qkv.weight',
                                       'blocks.11.attn.qkv.weight','patch_embed')):
            sha.update(name.encode('utf-8'))
            sha.update(p.detach().cpu().contiguous().numpy().tobytes())
            selected+=1
    if selected<3:
        raise RuntimeError('DOFA architecture does not expose expected tensors')
    return sha.hexdigest()


def main():
    ensure_folders()
    train_paths,train_names,lookup=get_split_paths('train')
    val_paths,val_names,_=get_split_paths('val',lookup)
    if train_names & val_names:raise AssertionError('TRAIN/VAL duplicate patches')
    mean,std,counts,class_weights,debris_flags=training_statistics(train_paths)
    meta=verify_cache_metadata(cache_config(mean,std,train_paths,val_paths))
    for split, paths in [('train',train_paths),('val',val_paths)]:
        for p in paths:check_cached_feature(feature_path(split,p))
    # In v0.8 the cached official pretrained weights are NOT downloaded again
    # solely to build the random encoder; source code is imported via Torch Hub.
    seed_everything(RANDOM_ENCODER_SEED)
    print('Constructing DOFA ViT-B from upstream architecture (pretrained=False)...')
    random_encoder=torch.hub.load(MODEL_REPO,'vit_base_dofa',pretrained=False,
        trust_repo=True,img_size=256,global_pool=False,num_classes=0)
    if not hasattr(random_encoder,'blocks') or not hasattr(random_encoder,'patch_embed'):
        raise RuntimeError('Unexpected DOFA architecture')
    fp=random_encoder_fingerprint(random_encoder)
    print('Frozen random DOFA initialization fingerprint:',fp)
    random_encoder.eval()
    for p in random_encoder.parameters():p.requires_grad_(False)
    # The random-token folder is separate from v0.6 pretrained cache; checked
    # fold identity includes exact random init and identical Train normalization.
    random_root=CACHE/'random_v08'
    all_paths=train_paths+val_paths
    cache_features(random_encoder,all_paths,mean,std,random_root,{
        'type':'random_untrained_DOFA','seed':RANDOM_ENCODER_SEED,
        'init_sha256':fp,'train_names':[p.name for p in train_paths],
        'val_names':[p.name for p in val_paths],
        'pretrained_loaded_into_random_encoder':False},batch_size=CACHE_BATCH_SIZE)
    del random_encoder
    if torch.cuda.is_available():torch.cuda.empty_cache()
    # Pretrained features live in split-specific v0.6 locations. Create a
    # lightweight hardlink-or-copy unified view for use by v08's dataset.
    pretrained_root=CACHE/'pretrained_v08_view'
    pretrained_root.mkdir(parents=True,exist_ok=True)
    save_json(pretrained_root/'source_metadata.json',{'original_checkpoint_sha256':meta['checkpoint_sha256'],
        'train_normalization':mean.tolist(),'train_std':std.tolist(),
        'no_test':True,'original_cache':'baseline/cache/dofa_v06/'})
    import shutil
    for split, paths in [('train',train_paths),('val',val_paths)]:
        for p in paths:
            target=pretrained_root/(p.stem+'.npy')
            if target.is_file():continue
            source=feature_path(split,p)
            # Windows: if NTFS hardlink permissions/filesystems differ, copy.
            try:
                import os
                os.link(source,target)
            except OSError:
                shutil.copy2(source,target)
    # Check a real feature is not identical between pretrained and random.
    sample=train_paths[0].stem+'.npy'
    v1=np.load(pretrained_root/sample,allow_pickle=False)
    v2=np.load(random_root/sample,allow_pickle=False)
    if v1.shape!=v2.shape or np.allclose(v1,v2,rtol=1e-4,atol=1e-4):
        raise RuntimeError('Random and pretrained DOFA features unexpectedly identical')
    print('PASS: random and pretrained tokens differ; both encoders frozen')
    results=[]
    folder=ROOT/'pretraining_control';folder.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS_TO_RUN:
        for source,root in [('pretrained',pretrained_root),('random',random_root)]:
            print('\n=== DOFA ONLY:',source,'seed',seed,'===',flush=True)
            row=train_one('dofa_only',train_paths,val_paths,None,mean,std,
                class_weights,debris_flags,seed,folder/f'seed_{seed}'/source,
                feature_root=root,max_epochs=EPOCHS,batch_size=BATCH_SIZE,
                patience=PATIENCE,learning_rate=LEARNING_RATE)
            row=dict(row,encoder_initialization=source,
                pretrained_checkpoint_sha256=meta['checkpoint_sha256'] if source=='pretrained' else None,
                random_encoder_init_sha256=fp if source=='random' else None)
            results.append(row)
            pd.DataFrame(results).to_csv(folder/'per_run.csv',index=False)
    frame=pd.DataFrame(results)
    piv=frame.pivot(index='seed',columns='encoder_initialization',values='debris_f1').dropna()
    if set(piv.columns)!={'random','pretrained'} or len(piv)!=len(SEEDS_TO_RUN):
        raise RuntimeError('Incomplete paired control')
    delta=piv['pretrained']-piv['random']
    table=piv.reset_index()
    table['pretrained_minus_random_f1']=delta.to_numpy()
    table.to_csv(folder/'paired_per_seed.csv',index=False)
    summary={'number_of_pairs':len(delta),
       'pretrained_mean_f1':float(piv['pretrained'].mean()),
       'random_mean_f1':float(piv['random'].mean()),
       'paired_delta_mean':float(delta.mean()),
       'paired_delta_sample_sd':float(delta.std(ddof=1)) if len(delta)>1 else None,
       'decoder_seed_pairs':list(SEEDS_TO_RUN), 'random_encoder_seed':RANDOM_ENCODER_SEED,
       'random_encoder_init_sha256':fp, 'real_pretrained_sha256':meta['checkpoint_sha256'],
       'same_head_architecture':True,'same_head_seeds':True,'raw_skip_zeroed':True,
       'encoders_frozen':True,'test_accessed':False,
       'caution':'Comparison is conditional on this preprocessing, model architecture and official validation.'}
    save_json(folder/'paired_summary.json',summary)
    plt.figure(figsize=(8,4.5))
    for seed,r in piv.iterrows():
        plt.plot(['random','pretrained'],[r['random'],r['pretrained']],marker='o',label=f'Seed {seed}')
    plt.ylabel('Marine Debris F1 (official validation)');plt.ylim(0,1)
    plt.title('v0.8 Frozen DOFA: Random vs Pretrained Encoder')
    plt.legend();plt.tight_layout()
    plt.savefig(FIGURES/'random_vs_pretrained_f1.png',dpi=170);plt.close()
    print('PASS: matched pretrained-vs-random comparison:',folder)

if __name__=='__main__':
    main()
