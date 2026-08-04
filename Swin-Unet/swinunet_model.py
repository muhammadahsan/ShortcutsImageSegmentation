
from __future__ import annotations

import sys
from pathlib import Path

import torch

SWINUNET_ROOT = Path(__file__).resolve().parent / "vendor" / "Swin-Unet"
if str(SWINUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(SWINUNET_ROOT))

from config import _C, _update_config_from_file  # noqa: E402
from networks.vision_transformer import SwinUnet  # noqa: E402

DEFAULT_CFG = SWINUNET_ROOT / "configs" / "swin_tiny_patch4_window7_224_lite.yaml"
DEFAULT_PRETRAINED = SWINUNET_ROOT / "pretrained_ckpt" / "swin_tiny_patch4_window7_224.pth"


def load_config(cfg_path: str | Path = DEFAULT_CFG, img_size: int = 224):
    """Load Swin-Unet yacs config and set input resolution."""
    cfg_path = Path(cfg_path)
    config = _C.clone()
    _update_config_from_file(config, str(cfg_path))
    config.defrost()
    config.DATA.IMG_SIZE = img_size
    if config.MODEL.PRETRAIN_CKPT and not Path(config.MODEL.PRETRAIN_CKPT).is_absolute():
        config.MODEL.PRETRAIN_CKPT = str(
            (SWINUNET_ROOT / config.MODEL.PRETRAIN_CKPT).resolve()
        )
    config.freeze()
    return config


def build_swinunet(
    num_classes: int = 3,
    img_size: int = 256,
    cfg_path: str | Path = DEFAULT_CFG,
    pretrained_path: str | Path | None = DEFAULT_PRETRAINED,
    device: torch.device | str | None = None,
) -> SwinUnet:
    """Instantiate Swin-Unet and optionally load Swin-T pretrained weights."""
    config = load_config(cfg_path, img_size=img_size)

    ckpt = Path(pretrained_path) if pretrained_path is not None else None
    config.defrost()
    if ckpt and ckpt.exists():
        config.MODEL.PRETRAIN_CKPT = str(ckpt)
        print(f"Loading Swin-Unet pretrained weights from {ckpt}")
    else:
        config.MODEL.PRETRAIN_CKPT = None
        if ckpt:
            print(
                f"Pretrained weights not found at {ckpt} — training from scratch.\n"
                "Download swin_tiny_patch4_window7_224.pth from "
                "https://github.com/HuCaoFighting/Swin-Unet"
            )
    config.freeze()

    model = SwinUnet(config, img_size=img_size, num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    model.load_from(config)
    return model