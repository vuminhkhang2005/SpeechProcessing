"""
Convert a training checkpoint into a smaller "weights-only" checkpoint.

Motivation:
  - Training checkpoints often include optimizer/scheduler states that can be huge.
  - The GUI/app only needs model weights (and optionally the training config).
  - Loading large checkpoints can appear to "hang" if the machine is RAM-starved and swapping.

Example:
  python convert_checkpoint.py --input best_model.pt --output best_model_weights.pt --keep-config
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _format_size(num_bytes: int) -> str:
    if num_bytes < 0:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.1f} {u}"
        size /= 1024.0
    return f"{size:.1f} B"


def _guess_state_dict_key(ckpt: Dict[str, Any]) -> Tuple[str, Dict[str, torch.Tensor]]:
    """
    Return (key_name, state_dict) where key_name is preferred top-level key.

    The repo loader supports:
      - ckpt["model_state_dict"]
      - ckpt["state_dict"]
      - raw state_dict (ckpt itself)
    """
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return "model_state_dict", ckpt["model_state_dict"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return "state_dict", ckpt["state_dict"]
    return "model_state_dict", ckpt  # treat as raw state_dict


def convert_checkpoint(input_path: Path, output_path: Path, *, keep_config: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    try:
        in_size = os.path.getsize(input_path)
    except OSError:
        in_size = -1

    print(f"Reading: {input_path} ({_format_size(in_size)})")
    ckpt = torch.load(str(input_path), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        state_key, state_dict = _guess_state_dict_key(ckpt)
        out: Dict[str, Any] = {state_key: state_dict}
        if keep_config and "config" in ckpt:
            out["config"] = ckpt["config"]
    else:
        # Rare case: checkpoint is directly a state_dict (or a module).
        # We only support state_dict-like objects here.
        if not hasattr(ckpt, "keys"):
            raise TypeError(
                "Unsupported checkpoint format (expected a dict-like state_dict). "
                "If this is a full model object, re-save as a state_dict checkpoint first."
            )
        out = {"model_state_dict": ckpt}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(output_path))

    try:
        out_size = os.path.getsize(output_path)
    except OSError:
        out_size = -1

    removed_hint = ""
    if isinstance(ckpt, dict):
        removed_keys = [
            k
            for k in ("optimizer_state_dict", "optimizer", "scheduler_state_dict", "scheduler", "scaler", "ema")
            if k in ckpt
        ]
        if removed_keys:
            removed_hint = f" (dropped: {', '.join(removed_keys)})"

    print(f"Wrote:   {output_path} ({_format_size(out_size)}){removed_hint}")
    if in_size > 0 and out_size > 0:
        print(f"Shrink:  {in_size / out_size:.2f}x smaller")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a checkpoint into a smaller weights-only checkpoint.")
    parser.add_argument("--input", required=True, help="Input checkpoint path (e.g. best_model.pt)")
    parser.add_argument("--output", required=True, help="Output checkpoint path (e.g. best_model_weights.pt)")
    parser.add_argument(
        "--keep-config",
        action="store_true",
        help="Keep ckpt['config'] if present (recommended for correct model reconstruction).",
    )
    args = parser.parse_args()

    convert_checkpoint(Path(args.input), Path(args.output), keep_config=bool(args.keep_config))


if __name__ == "__main__":
    main()

