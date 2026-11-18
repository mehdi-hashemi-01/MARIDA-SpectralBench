"""v0.7B: parameter-efficient DOFA LoRA fine-tuning (last 2 attention blocks).

Run after source-control stage 23/24 and only if worth the extra training.
Uses real pretrained checkpoint already in Hugging Face cache. No TEST use.
Trainable parameters = last-block LoRA factors + original v0.6 decoder.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler

from settings import BANDS, CLASSES, RANDOM_SEED
from unet_core import (MaridaSegmentationDataset, seed_everything, NUM_CLASSES,
                       IGNORE, confusion_from_batch, metrics_from_cm)
from dofa_transfer import load_real_pretrained_dofa, DofaDenseHead
from dofa_v07_common import (prepare_splits, masked_weighted_ce, RESULTS,
                             FIGURES, MODELS, save_json)
from dofa_lora_v07 import (attach_lora_to_last_blocks, dense_tokens_with_lora,
                           trainable_lora_state)

# ------ PyCharm settings --------
EPOCHS = 30
BATCH_SIZE = 2          # On an 8 GB laptop GPU reduce to 1 if out of memory.
PATIENCE = 8
LR_HEAD = 3e-4
LR_LORA = 8e-5
WEIGHT_DECAY = 1e-4
LAST_N_BLOCKS = 2
RANK = 4
ALPHA = 8
FORCE_RETRAIN = False
# --------------------------------

if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU required; use the RTX 5070 dl environment')
device=torch.device('cuda')
seed_everything(RANDOM_SEED)
train_paths,val_paths,mean,std,counts,class_weights,debris,meta=prepare_splits(check_all_cache=False)
ckpt_file=MODELS/'lora_last2_best.pt'
if ckpt_file.exists() and not FORCE_RETRAIN:
    raise RuntimeError('LoRA checkpoint already exists. Run 26_analyze_dofa_lora.py, '
                       'or set FORCE_RETRAIN=True to replace intentionally.')

print('Loading actual official pretrained DOFA checkpoint (cached after stage 20)...')
encoder, sha = load_real_pretrained_dofa()
if sha!=meta['checkpoint_sha256']:
    raise RuntimeError('Pretrained SHA differs from stage20 feature-cache SHA; abort')
encoder=encoder.to(device).eval()  # Keep base dropout/DropPath inactive.
from_block,names=attach_lora_to_last_blocks(encoder,LAST_N_BLOCKS,RANK,ALPHA)
head=DofaDenseHead().to(device)
trainable_encoder=[p for p in encoder.parameters() if p.requires_grad]
trainable_n=sum(p.numel() for p in trainable_encoder)
print('Trainable LoRA encoder parameters:',f'{trainable_n:,}', '| head:',f'{sum(p.numel() for p in head.parameters()):,}')
assert trainable_n>0 and all(not p.requires_grad for n,p in encoder.named_parameters() if n not in names)

train_ds=MaridaSegmentationDataset(train_paths,mean,std,augment=False)
val_ds=MaridaSegmentationDataset(val_paths,mean,std,augment=False)
sampler=WeightedRandomSampler(torch.tensor([3.0 if f else 1.0 for f in debris],dtype=torch.double),
    num_samples=len(train_ds),replacement=True,generator=torch.Generator().manual_seed(RANDOM_SEED))
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,sampler=sampler,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)
weight_t=torch.tensor(class_weights,device=device)
optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':LR_HEAD},
                             {'params':trainable_encoder,'lr':LR_LORA}],weight_decay=WEIGHT_DECAY)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4)
scaler=torch.amp.GradScaler('cuda')
config={ 'experiment':'DOFA pretrained frozen base + LoRA on last 2 attention blocks + spectral skip',
    'test_accessed':False, 'checkpoint_sha256':sha,'lora_blocks':LAST_N_BLOCKS,
    'rank':RANK,'alpha':ALPHA,'trainable_encoder_params':trainable_n,
    'trainable_encoder_names':names,'epochs_max':EPOCHS,'batch_size':BATCH_SIZE,
    'patience':PATIENCE,'lr_head':LR_HEAD,'lr_lora':LR_LORA,'seed':RANDOM_SEED,
    'train_patches':len(train_paths),'validation_patches':len(val_paths),
    'sampler':'debris-positive x3 with replacement','augmentation':'none, same as v0.6',
    'loss':'class- and confidence-weighted sparse-label CE',
    'caution':'LoRA training forwards through pretrained encoder every step, unlike frozen cache', }
save_json(RESULTS/'lora_run_config.json',config)

history=[];best=-1.;best_epoch=0;stale=0;started=time.perf_counter()
for epoch in range(1,EPOCHS+1):
    encoder.eval()    # LoRA params still receive gradients under eval mode.
    head.train()
    tot_num=0.;tot_den=0.
    for x,y,conf,paths in train_loader:
        x,y,conf=x.to(device),y.to(device),conf.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            feat=dense_tokens_with_lora(encoder,x,from_block)
            logits=head(feat,x)
        loss,num,den=masked_weighted_ce(logits,y,conf,weight_t)
        if loss is None: continue
        if not torch.isfinite(loss): raise RuntimeError('Non-finite LoRA loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(head.parameters())+trainable_encoder,1.0)
        scaler.step(optimizer);scaler.update()
        tot_num+=num;tot_den+=den
    encoder.eval();head.eval()
    cm=np.zeros((NUM_CLASSES,NUM_CLASSES),np.int64)
    val_ce_sum=0.;pixel_count=0
    with torch.inference_mode():
        for x,y,conf,paths in val_loader:
            x,y=x.to(device),y.to(device)
            with torch.autocast('cuda',dtype=torch.float16):
                feat=dense_tokens_with_lora(encoder,x,from_block)
                logits=head(feat,x)
            valid=y!=IGNORE
            ce=F.cross_entropy(logits.float(),y,ignore_index=IGNORE,reduction='none')
            val_ce_sum+=float(ce[valid].sum().item());pixel_count+=int(valid.sum().item())
            confusion_from_batch(cm,logits.argmax(1),y)
    metric=metrics_from_cm(cm);score=metric['debris_f1']
    if not np.isfinite(score):raise RuntimeError('Invalid debris F1 from Validation')
    scheduler.step(score)
    row={'epoch':epoch,'train_weighted_ce':tot_num/max(tot_den,1),
         'val_unweighted_ce':val_ce_sum/max(pixel_count,1),'val_debris_f1':score,
         'val_precision':metric['debris_precision'],'val_recall':metric['debris_recall'],
         'val_fp':metric['fp'],'val_fn':metric['fn'], 'lr_head':optimizer.param_groups[0]['lr'],
         'lr_lora':optimizer.param_groups[1]['lr']}
    history.append(row);pd.DataFrame(history).to_csv(RESULTS/'lora_training_history.csv',index=False)
    print(f'LoRA epoch {epoch:02d} F1={score:.4f} FP={metric["fp"]} FN={metric["fn"]}',flush=True)
    if score>best+1e-5:
        best=score;best_epoch=epoch;stale=0
        torch.save({'head_state_dict':head.state_dict(),
                    'lora_state_dict':trainable_lora_state(encoder),
                    'pretrained_sha256':sha,'best_epoch':epoch,'best_val_f1':score,
                    'mean':mean.tolist(),'std':std.tolist(),
                    'config':config},ckpt_file)
        pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(RESULTS/'lora_best_val_confusion.csv')
    else:stale+=1
    if stale>=PATIENCE:
        print('LoRA early stopping');break
print('Best LoRA VAL F1:',round(best,4),'epoch',best_epoch,'elapsed secs',round(time.perf_counter()-started,1))
plt.figure(figsize=(10,4.5))
plt.plot([r['epoch'] for r in history],[r['train_weighted_ce'] for r in history],label='Train weighted CE')
plt.plot([r['epoch'] for r in history],[r['val_unweighted_ce'] for r in history],label='Val CE (unweighted)')
plt.xlabel('Epoch');plt.ylabel('CE (different weightings)')
plt.twinx()
plt.plot([r['epoch'] for r in history],[r['val_debris_f1'] for r in history],linestyle='--',label='Val debris F1',color='#d18b2c')
plt.ylabel('Marine Debris F1');plt.title('DOFA LoRA last 2 ViT blocks — Validation only')
plt.tight_layout();plt.savefig(FIGURES/'lora_training.png',dpi=160);plt.close()
print('NEXT: 26_analyze_dofa_lora.py; no official TEST used.')
