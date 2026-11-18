"""CPU-only smoke test: verifies math/code wiring, NOT real DOFA pretrained weights.

Run as Python file inside PyCharm or use pytest. No internet/dataset needed.
"""
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'baseline'))
from dofa_transfer import DofaDenseHead
from dofa_v07_common import apply_sources, masked_weighted_ce
from dofa_lora_v07 import (LoraLinear, attach_lora_to_last_blocks,
                           dense_tokens_with_lora, trainable_lora_state,
                           restore_lora_state)


def test_sources_and_mask():
    torch.manual_seed(42)
    head=DofaDenseHead()
    head.eval()
    feat=torch.randn(1,768,16,16)
    x=torch.randn(1,11,256,256)
    # All modes produce 15 dense pixel logits.
    for mode in ('combined','dofa_only','spectral_only'):
        out=apply_sources(head,feat,x,mode)
        assert out.shape==(1,15,256,256)
        assert torch.isfinite(out).all()
    y=torch.full((1,256,256),255,dtype=torch.long)
    conf=torch.zeros((1,256,256))
    y[0,1,1]=0;conf[0,1,1]=1.0
    y[0,1,2]=5;conf[0,1,2]=0.5
    out=apply_sources(head,feat,x,'combined')
    loss,num,den=masked_weighted_ce(out,y,conf,torch.ones(15))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())


class FakePatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Conv2d(11,768,16,stride=16)
    def forward(self,x,wavelengths):
        z=self.project(x)
        return z.flatten(2).transpose(1,2),None

class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv=nn.Linear(768,768*3)
        self.proj=nn.Linear(768,768)
    def forward(self,x):
        return self.proj(self.qkv(x)[...,:768])*0.01

class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=FakeAttention()
    def forward(self,x):
        return x + self.attn(x)

class FakeDofa(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed=FakePatchEmbed()
        self.cls_token=nn.Parameter(torch.zeros(1,1,768))
        self.pos_embed=nn.Parameter(torch.zeros(1,257,768))
        self.blocks=nn.ModuleList([FakeBlock() for _ in range(3)])
        self.norm=nn.LayerNorm(768)


def test_lora_gradients_and_portable_state():
    torch.manual_seed(42)
    lin=nn.Linear(8,12)
    wrapper=LoraLinear(lin,rank=2,alpha=4)
    probe=torch.randn(2,8)
    assert torch.allclose(lin(probe),wrapper(probe),atol=1e-6) # B starts at zero.
    fake=FakeDofa()
    from_block,names=attach_lora_to_last_blocks(fake,last_n=2,rank=2,alpha=4)
    assert len(names)==8
    x=torch.randn(1,11,256,256)
    features=dense_tokens_with_lora(fake,x,from_block)
    assert tuple(features.shape)==(1,768,16,16)
    features.square().mean().backward()
    trainable={n:p for n,p in fake.named_parameters() if p.requires_grad}
    assert len(trainable)==8
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    assert all(p.grad is None for n,p in fake.named_parameters() if n not in trainable)
    saved=trainable_lora_state(fake)
    with torch.no_grad():
        for p in trainable.values():
            p.zero_()
    restore_lora_state(fake,saved)
    for n,p in trainable.items():
        assert torch.equal(p,saved[n])

if __name__=='__main__':
    torch.set_num_threads(2)
    test_sources_and_mask()
    print('PASS: 3 source controls, 15-class logits and masked loss')
    test_lora_gradients_and_portable_state()
    print('PASS: low-rank layers, trainable-only gradient and saved state')
    print('Smoke tests use synthetic tensors / fake encoder, not real MARIDA or DOFA.')
