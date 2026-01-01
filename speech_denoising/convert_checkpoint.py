"""
Wrapper to run the repo-level convert script from within speech_denoising/.

Usage:
  python convert_checkpoint.py --input best_model.pt --output best_model_weights.pt --keep-config
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from convert_checkpoint import main  # noqa: E402


if __name__ == "__main__":
    main()

