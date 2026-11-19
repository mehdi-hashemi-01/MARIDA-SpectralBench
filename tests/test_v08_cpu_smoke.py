"""Quick CPU-only v0.8 software smoke tests; NOT MARIDA research results.

Run by opening this file in PyCharm. Requires only existing project packages;
uses synthetic 32x32 tensors, NEVER opens the dataset or downloads DOFA.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))

import numpy as np
import torch
from settings import CLASSES
from unet_core import NUM_CLASSES, IGNORE, metrics_from_cm
from v08_research_core import (build_model, get_logits, record_metrics, tile_id,
                               acquisition_id, json_float, sha_json)
from dofa_v07_common import masked_weighted_ce


def main():
    torch.set_num_threads(2)
    assert tile_id('S2_11-6-18_16PCC_26.tif') == '16PCC'
    assert acquisition_id('S2_11-6-18_16PCC_26.tif') == '11-6-18/16PCC'
    assert sha_json({'a':1,'b':2})==sha_json({'b':2,'a':1})
    assert json_float(float('nan')) is None
    for name in ('spectral_only','unet','dofa_only','dofa_combined'):
        model=build_model(name,torch.device('cpu'))
        x=torch.randn(1,11,32,32)
        feat=torch.randn(1,768,16,16)
        truth=torch.full((1,32,32),IGNORE,dtype=torch.long)
        truth[:,4:9,4:9]=0      # 25 positive/debris labels
        truth[:,12:15,12:15]=4  # 9 labeled ship pixels
        conf=torch.zeros((1,32,32),dtype=torch.float32)
        conf[truth != IGNORE]=1.0
        logits=get_logits(model,x,feat,name)
        assert logits.shape == (1,NUM_CLASSES,32,32),name
        loss,n,d=masked_weighted_ce(logits,truth,conf,torch.ones(NUM_CLASSES))
        assert torch.isfinite(loss) and d==34, (name,d)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print('PASS:',name,'forward/backward, shape and masked loss')
        del model
    cm=np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.int64)
    cm[0,0]=15;cm[0,4]=10;cm[4,0]=3
    output=record_metrics(cm,{'test_accessed':False})
    assert output['tp']==15 and output['fp']==3 and output['fn']==10
    assert abs(output['debris_f1']-(30/43))<1e-12
    print('PASS: TP/FP/FN/confusion, no undefined values leaked into JSON')
    # Tiny end-to-end nested-geographic smoke with synthetic 32x32 GeoTIFFs.
    import tempfile
    import tifffile
    from v08_research_core import split_inner_by_acquisition, train_one
    with tempfile.TemporaryDirectory(prefix='marida_v08_smoke_') as temp:
        root=Path(temp)
        names=['S2_1-1-20_16PCC_1.tif','S2_1-2-20_16PEC_2.tif',
               'S2_1-3-20_18QYF_3.tif','S2_1-4-20_16PDC_4.tif']
        paths=[]
        for idx,name in enumerate(names):
            path=root/name
            signal=np.random.default_rng(idx).normal(0,0.1,(11,32,32)).astype(np.float32)
            label=np.zeros((32,32),dtype=np.uint8)
            label[3:9,3:9]=1   # 36 debris pixels
            label[12:16,12:16]=5  # ship labels
            tifffile.imwrite(path,signal)
            tifffile.imwrite(root/(path.stem+'_cl.tif'),label)
            tifffile.imwrite(root/(path.stem+'_conf.tif'),np.ones((32,32),dtype=np.uint8))
            paths.append(path)
        fit,inner,inner_groups=split_inner_by_acquisition(paths[:3],'16PDC',fraction=0.34)
        assert len(fit)>0 and len(inner)>0
        assert not(set(map(str,fit))&set(map(str,inner)))
        assert all(tile_id(p)!='16PDC' for p in fit+inner)
        r=train_one('spectral_only',fit,inner,paths[3:],
                    np.zeros(11,dtype=np.float32),np.ones(11,dtype=np.float32),
                    np.ones(15,dtype=np.float32),[True]*len(fit),
                    seed=42,dest=root/'output',max_epochs=1,patience=1,batch_size=1)
        assert r['outer_evaluated_once'] and r['test_accessed'] is False
        assert r['final_evaluation_patches']==1
        assert (root/'output'/'metrics.json').is_file()
        print('PASS: nested split, one-epoch training, inner selection, final outer scoring')
    print('ALL v0.8 CPU SMOKE TESTS PASS (synthetic only; not real MARIDA validation)')

if __name__=='__main__':main()
