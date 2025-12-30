"""
Dataset classes and utilities for VoiceBank + DEMAND speech denoising dataset

This module provides:
- VoiceBankDEMANDDataset: PyTorch Dataset for loading audio pairs
- create_dataloaders: Factory function for creating train/val dataloaders
- Google Colab/Drive integration utilities
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa


def is_colab() -> bool:
    """
    Check if running in Google Colab environment
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_google_drive(mount_path: str = '/content/drive') -> bool:
    """
    Mount Google Drive in Colab
    
    Args:
        mount_path: Path where Drive will be mounted
    
    Returns:
        True if successfully mounted, False otherwise
    """
    if not is_colab():
        print("Not running in Google Colab. Google Drive mount not available.")
        return False
    
    try:
        from google.colab import drive
        drive.mount(mount_path)
        print(f"✅ Google Drive mounted at {mount_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False


def setup_gdrive_dataset(
    gdrive_path: Optional[str] = None,
    gdrive_folder_id: Optional[str] = None,
    mount_path: str = '/content/drive'
) -> Optional[Dict[str, str]]:
    """
    Setup dataset paths from Google Drive
    
    Args:
        gdrive_path: Path to dataset folder in Google Drive 
                     (e.g., '/content/drive/MyDrive/datasets')
        gdrive_folder_id: Google Drive folder ID (alternative to path)
        mount_path: Where to mount Google Drive
    
    Returns:
        Dictionary with dataset paths, or None if setup failed
    
    Example:
        >>> paths = setup_gdrive_dataset(gdrive_path='/content/drive/MyDrive/datasets')
        >>> train_loader, val_loader = create_dataloaders(**paths)
    """
    # Mount Google Drive if in Colab
    if is_colab():
        if not os.path.exists(mount_path):
            if not mount_google_drive(mount_path):
                return None
    
    # Determine dataset base path
    if gdrive_path:
        base_path = Path(gdrive_path)
    elif gdrive_folder_id:
        # For folder ID, user needs to have already mounted and shared the folder
        base_path = Path(mount_path) / 'MyDrive' / 'datasets'
        print(f"Using default path: {base_path}")
    else:
        # Default path
        base_path = Path(mount_path) / 'MyDrive' / 'datasets'
        print(f"Using default path: {base_path}")
    
    # Expected subdirectories
    expected_dirs = {
        'train_clean_dir': 'clean_trainset_28spk_wav',
        'train_noisy_dir': 'noisy_trainset_28spk_wav',
        'test_clean_dir': 'clean_testset_wav',
        'test_noisy_dir': 'noisy_testset_wav'
    }
    
    paths = {}
    all_found = True
    
    print(f"\n📂 Checking dataset at: {base_path}")
    print("-" * 50)
    
    for key, dirname in expected_dirs.items():
        dir_path = base_path / dirname
        if dir_path.exists():
            wav_count = len(list(dir_path.glob('*.wav')))
            print(f"  ✅ {dirname}: {wav_count} files")
            paths[key] = str(dir_path)
        else:
            print(f"  ❌ {dirname}: Not found")
            all_found = False
    
    print("-" * 50)
    
    if not all_found:
        print("\n⚠️  Some dataset directories are missing!")
        print("Expected structure:")
        print(f"  {base_path}/")
        print("  ├── clean_trainset_28spk_wav/")
        print("  ├── noisy_trainset_28spk_wav/")
        print("  ├── clean_testset_wav/")
        print("  └── noisy_testset_wav/")
        return None
    
    print("\n✅ Dataset ready!")
    return paths


class VoiceBankDEMANDDataset(Dataset):
    """
    PyTorch Dataset for VoiceBank + DEMAND speech denoising dataset
    
    Each sample contains:
    - noisy: Noisy waveform
    - clean: Clean waveform  
    - noisy_stft: STFT of noisy waveform [freq_bins, time_frames, 2]
    - clean_stft: STFT of clean waveform [freq_bins, time_frames, 2]
    
    Args:
        clean_dir: Path to directory containing clean audio files
        noisy_dir: Path to directory containing noisy audio files
        sample_rate: Target sample rate (default: 16000)
        segment_length: Length of audio segments in samples (default: 32000 = 2 seconds)
        n_fft: FFT size for STFT (default: 512)
        hop_length: Hop length for STFT (default: 128)
        win_length: Window length for STFT (default: 512)
        train: Whether this is training set (enables random cropping)
    """
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        sample_rate: int = 16000,
        segment_length: int = 32000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        train: bool = True
    ):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.train = train
        
        # Get list of audio files
        self.clean_files = sorted(list(self.clean_dir.glob('*.wav')))
        self.noisy_files = sorted(list(self.noisy_dir.glob('*.wav')))
        
        # Create mapping from filename to path for matching
        self.clean_map = {f.name: f for f in self.clean_files}
        self.noisy_map = {f.name: f for f in self.noisy_files}
        
        # Find matching pairs (same filename in both directories)
        self.file_pairs = []
        for noisy_file in self.noisy_files:
            if noisy_file.name in self.clean_map:
                self.file_pairs.append({
                    'clean': self.clean_map[noisy_file.name],
                    'noisy': noisy_file
                })
        
        if len(self.file_pairs) == 0:
            raise ValueError(
                f"No matching file pairs found between:\n"
                f"  Clean: {clean_dir} ({len(self.clean_files)} files)\n"
                f"  Noisy: {noisy_dir} ({len(self.noisy_files)} files)"
            )
        
        # STFT window
        self.window = torch.hann_window(win_length)
        
        print(f"Dataset loaded: {len(self.file_pairs)} audio pairs")
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def _load_audio(self, filepath: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        # Use librosa for maximum portability (no torchaudio dependency).
        wav, _sr = librosa.load(str(filepath), sr=self.sample_rate, mono=True)
        return torch.from_numpy(wav).float()
    
    def _random_crop(self, clean: torch.Tensor, noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly crop audio to segment_length"""
        length = clean.shape[-1]
        
        if length < self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - length
            clean = F.pad(clean, (0, pad_length))
            noisy = F.pad(noisy, (0, pad_length))
        elif length > self.segment_length:
            # Random crop if too long
            if self.train:
                start = random.randint(0, length - self.segment_length)
            else:
                start = 0  # Use beginning for validation
            clean = clean[start:start + self.segment_length]
            noisy = noisy[start:start + self.segment_length]
        
        return clean, noisy
    
    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute STFT of waveform"""
        stft_out = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to real representation [freq, time, 2]
        stft_real = torch.stack([stft_out.real, stft_out.imag], dim=-1)
        
        return stft_real
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        pair = self.file_pairs[idx]
        
        # Load audio files
        clean = self._load_audio(pair['clean'])
        noisy = self._load_audio(pair['noisy'])
        
        # Ensure same length
        min_len = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # Random crop to segment length
        clean, noisy = self._random_crop(clean, noisy)
        
        # Compute STFT
        clean_stft = self._compute_stft(clean)
        noisy_stft = self._compute_stft(noisy)
        
        return {
            'clean': clean,
            'noisy': noisy,
            'clean_stft': clean_stft,
            'noisy_stft': noisy_stft
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
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_clean_dir: Path to clean training audio
        train_noisy_dir: Path to noisy training audio
        test_clean_dir: Path to clean test audio
        test_noisy_dir: Path to noisy test audio
        sample_rate: Audio sample rate
        segment_length: Length of audio segments in samples
        batch_size: Batch size
        num_workers: Number of data loading workers
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Create datasets
    train_dataset = VoiceBankDEMANDDataset(
        clean_dir=train_clean_dir,
        noisy_dir=train_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        train=True
    )
    
    val_dataset = VoiceBankDEMANDDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"\nDataloaders created:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader
