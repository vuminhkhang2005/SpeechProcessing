"""
Utility to convert a training checkpoint into a lightweight "weights-only" checkpoint.

Why:
- Many training checkpoints include optimizer/scheduler/scaler states that can be huge.
- For inference, we only need the model weights (state_dict) + optional config.
- Smaller checkpoints load faster and reduce RAM pressure (less swapping / thrashing).

Usage:
  python3 convert_checkpoint.py --input best_model.pt --output best_model_weights.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch


def _get_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        # raw state dict
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            # best-effort: assume it's already a state_dict-like mapping
            return ckpt  # type: ignore[return-value]
    raise TypeError("Unsupported checkpoint format; expected a dict-like checkpoint or state_dict.")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert PyTorch checkpoint to weights-only format")
    p.add_argument("--input", required=True, help="Path to input checkpoint (.pt/.pth)")
    p.add_argument("--output", required=True, help="Path to output weights-only checkpoint (.pt/.pth)")
    p.add_argument(
        "--keep-config",
        action="store_true",
        help="Keep ckpt['config'] (if present) in the output for auto model reconstruction",
    )
    args = p.parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        raise SystemExit(f"Input checkpoint not found: {in_path}")

    in_size_mb = os.path.getsize(in_path) / (1024 * 1024)
    print(f"[1/3] Loading checkpoint on CPU (~{in_size_mb:.1f} MB): {in_path}")
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)

    print("[2/3] Extracting model state_dict")
    state_dict = _get_state_dict(ckpt)

    out_obj: Dict[str, Any] = {"model_state_dict": state_dict}
    if args.keep_config and isinstance(ckpt, dict) and "config" in ckpt:
        out_obj["config"] = ckpt["config"]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    print(f"[3/3] Saving weights-only checkpoint: {out_path}")
    torch.save(out_obj, out_path)

    out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done. Size: {in_size_mb:.1f} MB -> {out_size_mb:.1f} MB")


if __name__ == "__main__":
    main()

