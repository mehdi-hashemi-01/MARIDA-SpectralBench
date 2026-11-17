"""Small LOCAL CPU smoke test for adapter/head/mask. Does NOT download real DOFA.

This is only a code test; synthetic values must never be reported as research results.
To run: Run in PyCharm after installing PyTorch/Numpy/tifffile.
"""
from pathlib import Path
import sys
import tempfile

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import dofa_transfer as d
from unet_core import IGNORE, metrics_from_cm, confusion_from_batch

assert d.WAVELENGTH_UM == [0.440, 0.490, 0.560, 0.665, 0.705, 0.740,
                           0.783, 0.842, 0.865, 1.610, 2.190]
d.check_bands()
assert d._interpolate_pos_embed(torch.randn(1, 197, 768), torch.randn(1, 257, 768)).shape == (1, 257, 768)

class MockBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 768))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.norm = torch.nn.LayerNorm(768)

    def patch_embed(self, x, wavelengths):
        assert len(wavelengths) == 11
        pooled = F.avg_pool2d(x, kernel_size=16, stride=16)
        z = pooled.mean(dim=1, keepdim=True).repeat(1, 768, 1, 1)
        return z.flatten(2).transpose(1, 2), None

model = MockBackbone().eval()
features = d.dofa_dense_tokens(model, torch.ones((1, 11, 256, 256)))
assert features.shape == (1, 768, 16, 16)
head = d.DofaDenseHead()
raw = torch.randn(1, 11, 256, 256)
logits = head(features, raw)
assert logits.shape == (1, 15, 256, 256)
y = torch.full((1, 256, 256), IGNORE, dtype=torch.long)
y[:, 8:25, 12:20] = 0
y[:, 30:60, 40:90] = 2
weight = torch.ones(15)
ce = F.cross_entropy(logits, y, weight=weight, ignore_index=IGNORE, reduction="none")
assert ce[y == IGNORE].abs().sum().item() == 0
loss = ce[y != IGNORE].mean()
loss.backward()
assert any(p.grad is not None for p in head.parameters())
cm = np.zeros((15, 15), np.int64)
confusion_from_batch(cm, logits.detach().argmax(1), y)
assert cm.sum() == (y != IGNORE).sum().item()

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    d.CACHE_ROOT = root / "cache"
    p = root / "S2_1-1-20_16PCC_0.tif"
    arr = np.random.default_rng(42).random((11, 256, 256), dtype=np.float32)
    label = np.zeros((256, 256), dtype=np.uint8)
    label[1:8, 2:10] = 1
    label[12:22, 5:30] = 8
    confidence = np.zeros_like(label)
    confidence[label > 0] = 1
    tifffile.imwrite(str(p), arr)
    tifffile.imwrite(str(p.with_name(p.stem + '_cl.tif')), label)
    tifffile.imwrite(str(p.with_name(p.stem + '_conf.tif')), confidence)
    feat_file = d.feature_path("train", p)
    feat_file.parent.mkdir(parents=True)
    np.save(feat_file, np.ones((768, 16, 16), dtype=np.float16), allow_pickle=False)
    ds = d.FrozenDofaDataset([p], "train", np.zeros(11), np.ones(11), augment=True)
    sample = ds[0]
    assert sample[0].shape == (11, 256, 256)
    assert sample[1].shape == (768, 16, 16)
    assert (sample[2] == 0).sum() == (label == 1).sum()
    assert (sample[2] == IGNORE).sum() == (label == 0).sum()

print("PASS: wavelength ordering, positional interpolation, dense features, "
      "masked CE, gradients, cached data & paired geometry")
print("Synthetic local data ONLY; NO pretrained DOFA weights downloaded/tested")
