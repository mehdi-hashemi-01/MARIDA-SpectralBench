"""v0.7B: verify saved LoRA model and evaluate on complete official Validation only.

This loads real pretrained DOFA plus the small LoRA and decoder checkpoint,
then compares with v0.7 source controls, archived DOFA v0.6 and U-Net/RF.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from settings import PROJECT_DIR, CLASSES
from unet_core import (MaridaSegmentationDataset, NUM_CLASSES, IGNORE,
                       metrics_from_cm, confusion_from_batch)
from dofa_transfer import DofaDenseHead, load_real_pretrained_dofa
from dofa_lora_v07 import (attach_lora_to_last_blocks, dense_tokens_with_lora,
                           restore_lora_state)
from dofa_v07_common import (prepare_splits, RESULTS, FIGURES, MODELS, cm_to_rows)

BATCH_SIZE=2
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU unavailable')
device=torch.device('cuda')
file=MODELS/'lora_last2_best.pt'
if not file.exists():
    raise FileNotFoundError('Run 25_train_dofa_lora.py first: '+str(file))
checkpoint=torch.load(file,map_location='cpu',weights_only=True)
train_paths,val_paths,mean,std,counts,weights,flags,meta=prepare_splits(check_all_cache=False)
if checkpoint['pretrained_sha256']!=meta['checkpoint_sha256']:
    raise RuntimeError('DOFA cache/LoRA checkpoint SHA mismatch')
config=checkpoint['config']
encoder,sha=load_real_pretrained_dofa()
if sha!=checkpoint['pretrained_sha256']:
    raise RuntimeError('Original pretrained weights changed. Cannot evaluate.')
encoder=encoder.to(device).eval()
from_block,names=attach_lora_to_last_blocks(encoder,config['lora_blocks'],config['rank'],config['alpha'])
restore_lora_state(encoder,checkpoint['lora_state_dict'])
head=DofaDenseHead().to(device)
head.load_state_dict(checkpoint['head_state_dict']);head.eval()
val_ds=MaridaSegmentationDataset(val_paths,mean,std,augment=False)
loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)
cm=np.zeros((NUM_CLASSES,NUM_CLASSES),np.int64);patch_rows=[]
with torch.inference_mode():
    for x,y,conf,paths in loader:
        x=x.to(device)
        with torch.autocast('cuda',dtype=torch.float16):
            features=dense_tokens_with_lora(encoder,x,from_block)
            pred=head(features,x).argmax(1)
        confusion_from_batch(cm,pred,y)
        np_pred=pred.cpu().numpy();np_y=y.numpy()
        for i,path in enumerate(paths):
            valid=np_y[i]!=IGNORE
            local=np.bincount(NUM_CLASSES*np_y[i][valid]+np_pred[i][valid],
                              minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES,NUM_CLASSES)
            m=metrics_from_cm(local)
            patch_rows.append({'patch':Path(path).name,'labeled_pixels':int(local.sum()),
                               'true_debris':int(local[0].sum()),'tp':m['tp'],
                               'fp':m['fp'],'fn':m['fn'],'f1':m['debris_f1']})
metrics=metrics_from_cm(cm)
if metrics['evaluated_pixels']!=213102 or len(val_paths)!=328:
    raise RuntimeError('Validation coverage unexpectedly changed')
patch_df=pd.DataFrame(patch_rows)
for k in ('tp','fp','fn'):
    assert int(patch_df[k].sum())==metrics[k]
patch_df.to_csv(RESULTS/'lora_validation_patch_errors.csv',index=False)
pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(RESULTS/'lora_validation_confusion.csv')
pd.DataFrame(cm_to_rows(cm,'lora_last2')).to_csv(RESULTS/'lora_validation_by_class.csv',index=False)
row={'model':'DOFA pretrained + LoRA last2 + spectral skip (v0.7)',
     'source':'new v0.7 LoRA', 'best_epoch':checkpoint['best_epoch'],
     'debris_f1':metrics['debris_f1'],'debris_precision':metrics['debris_precision'],
     'debris_recall':metrics['debris_recall'],'debris_iou':metrics['debris_iou'],
     'tp':metrics['tp'],'fp':metrics['fp'],'fn':metrics['fn'],
     'evaluated_pixels':metrics['evaluated_pixels'],'pretrained_sha256':sha,
     'test_evaluated':False,'clouds_as_debris_fp':int(cm[CLASSES.index('Clouds'),0]),
     'waves_as_debris_fp':int(cm[CLASSES.index('Waves'),0]),
     'sediment_as_debris_fp':int(cm[CLASSES.index('Sediment-Laden Water'),0])}
pd.DataFrame([row]).to_csv(RESULTS/'lora_validation_metrics.csv',index=False)

rows=[{'model':row['model'],'debris_f1':row['debris_f1'],'source':'LoRA v0.7'}]
controls=RESULTS/'source_controls_validation_comparison.csv'
if controls.exists():
    c=pd.read_csv(controls)
    for _,v in c.iterrows():
        rows.append({'model':'Source control '+str(v['mode']),'debris_f1':float(v['debris_f1']),
                     'source':'fresh v0.7 controlled'} )
old=PROJECT_DIR/'results'/'dofa_v06'/'validation_metrics.csv'
if old.exists():
    s=pd.read_csv(old).iloc[0]
    rows.append({'model':'DOFA v0.6 (archived frozen + spectral skip)',
                 'debris_f1':float(s['marine_debris_f1']),'source':'historical separate run'})
compare=pd.DataFrame(rows).sort_values('debris_f1',ascending=False)
compare.to_csv(RESULTS/'lora_vs_frozen_validation.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,5))
axes[0].barh(compare['model'],compare['debris_f1']);axes[0].invert_yaxis()
axes[0].set_xlim(0,1);axes[0].set_xlabel('Marine Debris F1 / official VAL (labeled pixels)')
axes[0].set_title('LoRA vs source controls / distinct training protocols')
norm=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
img=axes[1].imshow(norm,vmin=0,vmax=1,cmap='Blues')
axes[1].set_xticks(range(NUM_CLASSES),CLASSES,rotation=90,fontsize=6)
axes[1].set_yticks(range(NUM_CLASSES),CLASSES,fontsize=6)
axes[1].set_title('LoRA: row-normalized 15-class confusion')
fig.colorbar(img,ax=axes[1],fraction=.046,pad=.04)
fig.tight_layout();fig.savefig(FIGURES/'lora_validation_summary.png',dpi=160)
plt.close(fig)

print('LoRA VAL F1:',round(metrics['debris_f1'],4),'TP',metrics['tp'],'FP',metrics['fp'],'FN',metrics['fn'])
print('Outputs:',RESULTS, FIGURES)
print('Only official VALIDATION evaluated. TEST not used.')
