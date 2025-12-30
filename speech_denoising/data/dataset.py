"""
Dataset utilities for VoiceBank + DEMAND speech denoising.

The training / evaluation scripts in this repo import:
    from data.dataset import VoiceBankDEMANDDataset, create_dataloaders, setup_gdrive_dataset

Implementation notes
--------------------
- Uses `librosa` (not torchaudio) for maximum compatibility (e.g. Colab).
- Returns both waveform pairs (clean/noisy) and their complex STFTs
  as real tensors with last-dim size 2: [..., 2] = (real, imag).
- For training/validation, `segment_length` should be an int so DataLoader
  can collate fixed-size tensors. For test/eval you may pass `segment_length=None`
  to keep full utterances (variable length).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import torch
from torch.utils.data import Dataset, DataLoader

from utils.audio_utils import AudioProcessor


@dataclass(frozen=True)
class DatasetPaths:
    train_clean_dir: str
    train_noisy_dir: str
    test_clean_dir: str
    test_noisy_dir: str


def setup_gdrive_dataset(
    gdrive_path: Optional[str] = None,
    gdrive_folder_id: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Convenience helper for Google Colab users who mounted Google Drive.

    This function does NOT download anything. It only validates and returns
    directory paths under `gdrive_path`.

    Args:
        gdrive_path: Path to the folder that contains the 4 dataset directories.
        gdrive_folder_id: Accepted for compatibility with older notebooks/configs.
            (Not used here; downloading by folder_id is intentionally not implemented.)
    """
    if gdrive_path is None:
        return None

    root = Path(gdrive_path)
    if not root.exists():
        raise FileNotFoundError(f"Google Drive dataset path does not exist: {root}")

    paths = DatasetPaths(
        train_clean_dir=str(root / "clean_trainset_28spk_wav"),
        train_noisy_dir=str(root / "noisy_trainset_28spk_wav"),
        test_clean_dir=str(root / "clean_testset_wav"),
        test_noisy_dir=str(root / "noisy_testset_wav"),
    )

    # Validate
    missing = [p for p in paths.__dict__.values() if not Path(p).exists()]
    if missing:
        msg = (
            "Missing expected dataset folders under gdrive_path.\n"
            f"gdrive_path={root}\n"
            "Expected:\n"
            f"- {paths.train_clean_dir}\n"
            f"- {paths.train_noisy_dir}\n"
            f"- {paths.test_clean_dir}\n"
            f"- {paths.test_noisy_dir}\n"
        )
        if gdrive_folder_id:
            msg += (
                "\nNote: `gdrive_folder_id` was provided but automatic downloading "
                "is not implemented in this repo."
            )
        raise FileNotFoundError(msg)

    return paths.__dict__.copy()


class VoiceBankDEMANDDataset(Dataset):
    """
    VoiceBank + DEMAND Dataset for Speech Denoising.

    Each item returns:
      - clean:      Tensor [T]
      - noisy:      Tensor [T]
      - clean_stft: Tensor [F, TT, 2]
      - noisy_stft: Tensor [F, TT, 2]
      - filename:   str
    """

    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        sample_rate: int = 16000,
        segment_length: Optional[int] = 32000,
        mode: str = "train",  # "train" | "val" | "test"
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
    ):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.mode = mode
        self.is_train = mode == "train"

        self.audio_processor = AudioProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sample_rate=sample_rate,
        )

        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean dir not found: {self.clean_dir}")
        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"Noisy dir not found: {self.noisy_dir}")

        clean_files = sorted(self.clean_dir.glob("*.wav"))
        noisy_files = sorted(self.noisy_dir.glob("*.wav"))

        clean_by_stem = {p.stem: p for p in clean_files}
        noisy_by_stem = {p.stem: p for p in noisy_files}
        common = sorted(set(clean_by_stem.keys()) & set(noisy_by_stem.keys()))

        self.file_pairs = [(clean_by_stem[s], noisy_by_stem[s]) for s in common]
        if not self.file_pairs:
            raise RuntimeError(
                "No matching clean/noisy pairs found. "
                f"clean_dir={self.clean_dir} noisy_dir={self.noisy_dir}"
            )

    def __len__(self) -> int:
        return len(self.file_pairs)

    def _load_wav(self, path: Path) -> torch.Tensor:
        wav_np, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return torch.from_numpy(wav_np).float()

    def _crop_or_pad_pair(self, clean: torch.Tensor, noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep full utterance if segment_length is None
        if self.segment_length is None:
            min_len = min(clean.numel(), noisy.numel())
            return clean[:min_len], noisy[:min_len]

        seg = int(self.segment_length)
        if clean.numel() != noisy.numel():
            min_len = min(clean.numel(), noisy.numel())
            clean = clean[:min_len]
            noisy = noisy[:min_len]

        if self.is_train and clean.numel() > seg:
            start = random.randint(0, clean.numel() - seg)
            return clean[start : start + seg], noisy[start : start + seg]

        # For val/test with fixed segments: pad/truncate deterministically
        if clean.numel() < seg:
            pad = seg - clean.numel()
            clean = torch.nn.functional.pad(clean, (0, pad))
            noisy = torch.nn.functional.pad(noisy, (0, pad))
        else:
            clean = clean[:seg]
            noisy = noisy[:seg]

        return clean, noisy

    def __getitem__(self, idx: int) -> Dict[str, object]:
        clean_path, noisy_path = self.file_pairs[idx]

        clean = self._load_wav(clean_path)
        noisy = self._load_wav(noisy_path)

        clean, noisy = self._crop_or_pad_pair(clean, noisy)

        clean_stft = self.audio_processor.stft(clean).squeeze(0)  # [F, TT, 2]
        noisy_stft = self.audio_processor.stft(noisy).squeeze(0)

        return {
            "clean": clean,
            "noisy": noisy,
            "clean_stft": clean_stft,
            "noisy_stft": noisy_stft,
            "filename": clean_path.name,
        }


def create_dataloaders(
    train_clean_dir: str,
    train_noisy_dir: str,
    test_clean_dir: str,
    test_noisy_dir: str,
    sample_rate: int = 16000,
    segment_length: int = 32000,
    batch_size: int = 16,
    num_workers: int = 4,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders.

    Note: many repos use the official "test" split as validation during training.
    That's what this helper does (to match the existing scripts).
    """
    train_ds = VoiceBankDEMANDDataset(
        clean_dir=train_clean_dir,
        noisy_dir=train_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        mode="train",
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    val_ds = VoiceBankDEMANDDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        mode="val",
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader

