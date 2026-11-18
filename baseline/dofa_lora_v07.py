"""Small, dependency-free LoRA implementation for DOFA ViT attention.

Only the last two ViT blocks receive LoRA; original DOFA weights stay frozen.
Uses the SAME patch-token extraction as v0.6, but permits gradients in LoRA.
"""
import math
import torch
from torch import nn

from dofa_transfer import WAVELENGTH_UM, PATCH_GRID, EMBED_DIM


class LoraLinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank correction B(A(x))*alpha/r."""
    def __init__(self, linear, rank=4, alpha=8.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError('Expected nn.Linear in DOFA attention, got '+repr(type(linear)))
        self.base = linear
        for param in self.base.parameters():
            param.requires_grad_(False)
        # The pretrained DOFA encoder is moved to CUDA BEFORE LoRA is attached.
        # Construct new LoRA parameters on the SAME device/dtype as qkv/proj;
        # otherwise PyTorch creates them on CPU and CUDA matmul fails.
        base_device = linear.weight.device
        base_dtype = linear.weight.dtype
        self.a = nn.Linear(linear.in_features, rank, bias=False,
                           device=base_device, dtype=base_dtype)
        self.b = nn.Linear(rank, linear.out_features, bias=False,
                           device=base_device, dtype=base_dtype)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


def attach_lora_to_last_blocks(model, last_n=2, rank=4, alpha=8):
    """Modify two existing attention modules per final ViT block; no PEFT needed."""
    if not hasattr(model, 'blocks') or len(model.blocks)<last_n:
        raise RuntimeError('Unexpected DOFA blocks / invalid last_n')
    for p in model.parameters():
        p.requires_grad_(False)
    start = len(model.blocks)-last_n
    for index in range(start,len(model.blocks)):
        block = model.blocks[index]
        if not hasattr(block,'attn') or not hasattr(block.attn,'qkv') or not hasattr(block.attn,'proj'):
            raise RuntimeError(f'DOFA block {index} lacks expected timm Attention qkv/proj')
        block.attn.qkv = LoraLinear(block.attn.qkv,rank=rank,alpha=alpha)
        block.attn.proj = LoraLinear(block.attn.proj,rank=rank,alpha=alpha)
    names = [name for name,p in model.named_parameters() if p.requires_grad]
    if len(names)!=last_n*2*2 or any(not ('.a.weight' in n or '.b.weight' in n) for n in names):
        raise RuntimeError('Unexpected LoRA trainable parameter list: '+repr(names))
    return start, names


def verify_lora_device(model, expected_device):
    """Fail before training if newly attached LoRA parameters are off-device."""
    expected_device = torch.device(expected_device)
    for name, p in model.named_parameters():
        same_type = p.device.type == expected_device.type
        same_index = (expected_device.index is None or
                      p.device.index == expected_device.index)
        if p.requires_grad and not (same_type and same_index):
            raise RuntimeError(
                f'LoRA parameter {name} is on {p.device}, expected {expected_device}. '
                'Ensure LoRA modules are attached on the correct device.'
            )


def dense_tokens_with_lora(model, x, from_block):
    """Early transformer blocks frozen/no-grad; last blocks allow LoRA gradients."""
    if x.ndim!=4 or tuple(x.shape[1:])!=(11,256,256):
        raise ValueError('DOFA expects B,11,256,256')
    waves = torch.tensor(WAVELENGTH_UM,dtype=torch.float32,device=x.device)
    with torch.no_grad():
        token, _ = model.patch_embed(x, waves)
        if tuple(token.shape[1:])!=(PATCH_GRID*PATCH_GRID, EMBED_DIM):
            raise RuntimeError('Unexpected DOFA tokens: '+repr(tuple(token.shape)))
        token = token + model.pos_embed[:, 1:, :]
        cls = (model.cls_token + model.pos_embed[:, :1,:]).expand(x.shape[0],-1,-1)
        token = torch.cat([cls,token],dim=1)
        for block in model.blocks[:from_block]:
            token = block(token)
    for block in model.blocks[from_block:]:
        token = block(token)
    token = model.norm(token[:,1:,:])
    return token.transpose(1,2).reshape(-1,EMBED_DIM,PATCH_GRID,PATCH_GRID)


def trainable_lora_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def restore_lora_state(model, saved):
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(saved)!=set(trainable):
        raise RuntimeError('Saved LoRA keys do not match installed modules')
    with torch.no_grad():
        for name,p in trainable.items():
            if tuple(saved[name].shape)!=tuple(p.shape):
                raise RuntimeError('LoRA weight shape mismatch for '+name)
            p.copy_(saved[name].to(p.device))
