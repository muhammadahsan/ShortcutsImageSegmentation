"""Build TransUNet (R50-ViT-B/16) for ISIC multiclass segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TRANSUNET_ROOT = Path(__file__).resolve().parent
if str(TRANSUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSUNET_ROOT))

from networks.vit_seg_modeling import CONFIGS, VisionTransformer  # noqa: E402

DEFAULT_VIT_NAME = "R50-ViT-B_16"
DEFAULT_N_SKIP = 3
DEFAULT_VIT_PATCHES_SIZE = 16
DEFAULT_PRETRAINED = (
    TRANSUNET_ROOT / "model" / "vit_checkpoint" / "imagenet21k" / "R50+ViT-B_16.npz"
)


def build_transunet(
    num_classes: int = 3,
    img_size: int = 256,
    vit_name: str = DEFAULT_VIT_NAME,
    n_skip: int = DEFAULT_N_SKIP,
    vit_patches_size: int = DEFAULT_VIT_PATCHES_SIZE,
    pretrained_path: str | Path | None = DEFAULT_PRETRAINED,
    device: torch.device | str | None = None,
) -> VisionTransformer:
    """Instantiate TransUNet and optionally load ImageNet-21k ViT weights."""
    config = CONFIGS[vit_name]
    config.n_classes = num_classes
    config.n_skip = n_skip
    if vit_name.find("R50") != -1:
        grid = int(img_size / vit_patches_size)
        config.patches.grid = (grid, grid)

    model = VisionTransformer(
        config,
        img_size=img_size,
        num_classes=num_classes,
    )
    if device is not None:
        model = model.to(device)

    ckpt = Path(pretrained_path) if pretrained_path else None
    if ckpt and ckpt.exists():
        print(f"Loading TransUNet pretrained weights from {ckpt}")
        model.load_from(weights=np.load(ckpt, allow_pickle=True))
    elif ckpt:
        print(
            f"Pretrained weights not found at {ckpt} — training from scratch.\n"
            "Download R50+ViT-B_16.npz from https://github.com/Beckschen/TransUNet"
        )
    return model
