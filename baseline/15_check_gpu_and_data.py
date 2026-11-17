"""v0.4 Step 15: verify RTX/CUDA, official MARIDA splits and image/masks.
"""
from collections import Counter
import numpy as np
import torch
from unet_core import (NUM_BANDS, NUM_CLASSES, get_all_split_paths,
                       load_raw_image, load_label, load_mask, UNet11)

print("=== DEVICE CHECK ===")
print("PyTorch:", torch.__version__)
print("CUDA runtime in PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot use CUDA. Install a CUDA-enabled PyTorch build in "
        "the SAME PyCharm interpreter. See docs/unet_segmentation.md."
    )
print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
print("Torch compiled architectures:", torch.cuda.get_arch_list())
try:
    x = torch.randn(1, NUM_BANDS, 64, 64, device="cuda")
    network = UNet11(base=16).to("cuda")
    with torch.no_grad():
        y = network(x)
    torch.cuda.synchronize()
    assert tuple(y.shape) == (1, NUM_CLASSES, 64, 64)
    print("GPU 11-band U-Net forward PASS:", tuple(y.shape))
except Exception as exc:
    raise RuntimeError("CUDA was detected but an actual U-Net operation failed. "
                       "Check PyTorch sm_120/Blackwell support and NVIDIA driver.") from exc
finally:
    del x, network
    if "y" in locals():
        del y
    torch.cuda.empty_cache()

print("\n=== OFFICIAL MARIDA SPLITS ===")
splits = get_all_split_paths()
for split, paths in splits.items():
    print(f"{split:>5}: {len(paths)} patches | first: {paths[0].name}")
    for sample_path in [paths[0], paths[-1]]:
        raw = load_raw_image(sample_path)
        label = load_label(sample_path)
        confidence = load_mask(sample_path, "_conf")
        if raw.shape[0] != NUM_BANDS or raw.shape[1:] != label.shape or label.shape != confidence.shape:
            raise ValueError("Image/mask mismatch: " + sample_path.name)
        print("   ", sample_path.name, "image:", raw.shape,
              "labeled:", int(np.count_nonzero(label)))
print("\nPASS: GPU forward, split separation, sampled masks and spectral dimensions")
