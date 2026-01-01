"""
Unified checkpoint loading for multiple model architectures.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from models.dccrn import DCCRN, DCCRNConfig
from models.unet import UNetDenoiser, convert_old_checkpoint, detect_encoder_channels_from_checkpoint


def _get_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    # raw state dict
    return ckpt  # type: ignore[return-value]

def _infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Best-effort architecture inference when checkpoint has no config.

    We support two families in this repo:
    - UNetDenoiser (real Conv2d, keys like: init_conv.block.0.weight, encoders.0.conv1.block.0.weight, ...)
    - DCCRN (complex conv implemented via wr/wi, plus an LSTM bottleneck, keys like:
      encoders.0.conv.wr.weight, encoders.0.conv.wi.weight, rnn.weight_ih_l0, out_conv_r.weight, ...)
    """
    keys = list(state_dict.keys())
    # Strong DCCRN signals
    if any(".wr." in k or ".wi." in k for k in keys):
        return "DCCRN"
    if any(k.startswith("rnn.") or k.startswith("rnn_proj.") for k in keys):
        return "DCCRN"
    if any(k.startswith("out_conv_r.") or k.startswith("out_conv_i.") for k in keys):
        return "DCCRN"
    # Default to UNet
    return "UNetDenoiser"


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
    strict: bool = False,
    n_fft: int | None = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PERF: always load checkpoint tensors on CPU first.
    #
    # Loading directly to CUDA (map_location=cuda) can be noticeably slower and
    # may temporarily spike GPU memory during deserialization. We instead load
    # on CPU, then copy parameters into the model on the target device.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    model_name = str(model_cfg.get("name", "")).strip()

    state_dict = _get_state_dict(ckpt) if isinstance(ckpt, dict) else ckpt

    if not model_name:
        model_name = _infer_arch_from_state_dict(state_dict)

    if model_name.lower() in {"dccrn", "dccrn_denoiser"}:
        # freq bins = n_fft//2 + 1
        stft_cfg = config.get("stft", {}) if isinstance(config, dict) else {}
        effective_n_fft = int(stft_cfg.get("n_fft", n_fft if n_fft is not None else 512))
        freq_bins = effective_n_fft // 2 + 1

        # Minimal config; allow overriding encoder channels + rnn size from yaml
        dcfg = DCCRNConfig(
            encoder_channels=list(model_cfg.get("encoder_channels", [16, 32, 64, 128, 256])),
            rnn_layers=int(model_cfg.get("rnn_layers", 2)),
            rnn_hidden=int(model_cfg.get("rnn_hidden", 256)),
        )
        model = DCCRN(freq_bins=freq_bins, cfg=dcfg).to(device)
        model.load_state_dict(state_dict, strict=strict)
        return model, config

    # Default: UNet
    converted = convert_old_checkpoint(state_dict)
    detected_channels = detect_encoder_channels_from_checkpoint(converted)
    encoder_channels = model_cfg.get("encoder_channels", detected_channels)
    model = UNetDenoiser(
        in_channels=model_cfg.get("in_channels", 2),
        out_channels=model_cfg.get("out_channels", 2),
        encoder_channels=encoder_channels,
        use_attention=model_cfg.get("use_attention", True),
        dropout=0.0,
        mask_type=model_cfg.get("mask_type", "CRM"),
    ).to(device)

    try:
        model.load_state_dict(converted, strict=strict)
    except RuntimeError:
        if strict:
            raise
        # partial load
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in converted.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)

    return model, config

