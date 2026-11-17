"""Tiny CPU-only structural check; does not use MARIDA data or assert research results.
Run directly in PyCharm (may require selecting project root as working directory).
"""
import sys
from pathlib import Path
import tempfile
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from unet_core import (UNet11, MaridaSegmentationDataset, IGNORE, NUM_CLASSES,
                       confusion_from_batch, metrics_from_cm)

with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "S2_11-6-18_16PCC_0.tif"
    raw = np.random.default_rng(42).uniform(.01, .25, (11, 32, 32)).astype(np.float32)
    raw[0, 0, 0] = np.nan  # sample no-data value
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[3:9, 3:9] = 1   # Marine Debris
    labels[16:30, 16:30] = 8  # Sediment-laden Water
    confidence = np.zeros((32, 32), dtype=np.uint8)
    confidence[labels > 0] = 1
    confidence[4:8, 4:8] = 2
    tifffile.imwrite(path, raw)
    tifffile.imwrite(path.with_name(path.stem + "_cl.tif"), labels)
    tifffile.imwrite(path.with_name(path.stem + "_conf.tif"), confidence)
    ds = MaridaSegmentationDataset([path], means=np.ones(11) * .1,
                                   stds=np.ones(11) * .1, augment=True)
    x, y, weights, _ = ds[0]
    assert x.shape == (11, 32, 32) and torch.isfinite(x).all()
    assert y.shape == (32, 32) and torch.all((y == IGNORE) | ((y >= 0) & (y < 15)))
    assert int((y != IGNORE).sum()) == int((labels > 0).sum())
    assert int((weights == 0).sum()) == int((labels == 0).sum())

    model = UNet11(base=8)
    logits = model(x.unsqueeze(0))
    assert logits.shape == (1, 15, 32, 32)
    loss = F.cross_entropy(logits, y.unsqueeze(0), ignore_index=IGNORE)
    assert torch.isfinite(loss)
    loss.backward()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    confusion_from_batch(cm, logits.argmax(1), y.unsqueeze(0))
    assert int(cm.sum()) == int((labels != 0).sum())
    scores = metrics_from_cm(cm)
    print("PASS: input, augmentation, sparse-label mask, U-Net output, loss, backprop, confusion")
    print("NOTE: Synthetic smoke test only; its accuracy must never be reported as MARIDA results.")
