"""
Unified checkpoint loading for multiple model architectures.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Tuple

import os
import time
import torch

from models.dccrn import DCCRN, DCCRNConfig
from models.unet import UNetDenoiser, convert_old_checkpoint, detect_encoder_channels_from_checkpoint


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)

def _assign_enabled() -> bool:
    """
    Whether to attempt PyTorch's load_state_dict(..., assign=True) fast-path.

    Some PyTorch/Windows builds have been reported to hang in the assign=True path.
    Allow users to force-disable it via env var:
      SPEECH_DENOISING_DISABLE_ASSIGN=1
    """
    v = os.getenv("SPEECH_DENOISING_DISABLE_ASSIGN", "").strip().lower()
    return v not in {"1", "true", "yes", "y", "on"}

def _load_state_dict_with_optional_assign(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool,
) -> bool:
    """
    Try to use PyTorch's faster path: load_state_dict(..., assign=True) when available.

    - On supported PyTorch versions, assign=True can be significantly faster and reduce RAM copies.
    - On older versions, it will raise TypeError (unexpected keyword), and we fallback safely.

    Returns:
        True if assign=True was used, False otherwise.
    """
    if not _assign_enabled():
        model.load_state_dict(state_dict, strict=strict)
        return False

    # Prefer signature detection so the caller can log correctly before loading.
    try:
        sig = inspect.signature(model.load_state_dict)
        if "assign" not in sig.parameters:
            model.load_state_dict(state_dict, strict=strict)
            return False
    except (TypeError, ValueError):
        # If we cannot inspect the signature, fall back to try/except.
        pass

    try:
        model.load_state_dict(state_dict, strict=strict, assign=True)  # type: ignore[call-arg]
        return True
    except TypeError as e:
        # Only swallow the "assign not supported" case; re-raise other TypeErrors.
        if "assign" not in str(e):
            raise
        model.load_state_dict(state_dict, strict=strict)
        return False


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
    progress: Callable[[str], None] | None = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PERF: always load checkpoint tensors on CPU first.
    #
    # Loading directly to CUDA (map_location=cuda) can be noticeably slower and
    # may temporarily spike GPU memory during deserialization. We instead load
    # on CPU, then copy parameters into the model on the target device.
    try:
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        _emit(progress, f"[1/4] Reading checkpoint from disk (~{size_mb:.1f} MB)...")
    except OSError:
        _emit(progress, "[1/4] Reading checkpoint from disk...")

    t0 = time.perf_counter()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _emit(progress, f"[1/4] Checkpoint loaded in {time.perf_counter() - t0:.2f}s")

    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    model_name = str(model_cfg.get("name", "")).strip()

    state_dict = _get_state_dict(ckpt) if isinstance(ckpt, dict) else ckpt

    if not model_name:
        model_name = _infer_arch_from_state_dict(state_dict)

    _emit(progress, f"[2/4] Building model architecture ({model_name})...")
    t1 = time.perf_counter()

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
        model = DCCRN(freq_bins=freq_bins, cfg=dcfg)  # build on CPU first
        _emit(progress, f"[2/4] Model built in {time.perf_counter() - t1:.2f}s")

        # Log the expected path before doing the heavy load.
        try:
            supports_assign = _assign_enabled() and ("assign" in inspect.signature(model.load_state_dict).parameters)
        except (TypeError, ValueError):
            supports_assign = False
        _emit(
            progress,
            "[3/4] Loading weights (load_state_dict, assign=True)..." if supports_assign else "[3/4] Loading weights (load_state_dict)...",
        )
        t2 = time.perf_counter()
        _load_state_dict_with_optional_assign(model, state_dict, strict=strict)
        _emit(progress, f"[3/4] Weights loaded in {time.perf_counter() - t2:.2f}s")

        if device.type != "cpu":
            _emit(progress, f"[4/4] Moving model to {device}...")
            t3 = time.perf_counter()
            model = model.to(device)
            _emit(progress, f"[4/4] Model moved in {time.perf_counter() - t3:.2f}s")
        else:
            _emit(progress, "[4/4] Using CPU (no device transfer)")
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
    )  # build on CPU first

    _emit(progress, f"[2/4] Model built in {time.perf_counter() - t1:.2f}s")

    try:
        try:
            supports_assign = _assign_enabled() and ("assign" in inspect.signature(model.load_state_dict).parameters)
        except (TypeError, ValueError):
            supports_assign = False
        _emit(
            progress,
            "[3/4] Loading weights (load_state_dict, assign=True)..." if supports_assign else "[3/4] Loading weights (load_state_dict)...",
        )
        t2 = time.perf_counter()
        _load_state_dict_with_optional_assign(model, converted, strict=strict)
        _emit(progress, f"[3/4] Weights loaded in {time.perf_counter() - t2:.2f}s")
    except RuntimeError:
        if strict:
            raise
        # partial load
        _emit(progress, "[3/4] Weights mismatch; doing partial load...")
        t2 = time.perf_counter()
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in converted.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        try:
            supports_assign = _assign_enabled() and ("assign" in inspect.signature(model.load_state_dict).parameters)
        except (TypeError, ValueError):
            supports_assign = False
        if supports_assign:
            _emit(progress, "[3/4] Partial load (load_state_dict, assign=True)...")
        _load_state_dict_with_optional_assign(model, model_dict, strict=False)
        _emit(progress, f"[3/4] Partial load done in {time.perf_counter() - t2:.2f}s")

    if device.type != "cpu":
        _emit(progress, f"[4/4] Moving model to {device}...")
        t3 = time.perf_counter()
        model = model.to(device)
        _emit(progress, f"[4/4] Model moved in {time.perf_counter() - t3:.2f}s")
    else:
        _emit(progress, "[4/4] Using CPU (no device transfer)")

    return model, config

