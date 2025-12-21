"""
Dataset loader for VoiceBank + DEMAND dataset
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np


class VoiceBankDEMANDDataset(Dataset):
    """
    Dataset loader cho VoiceBank + DEMAND dataset
    
    Dataset có thể download từ:
    https://datashare.ed.ac.uk/handle/10283/2791
    
    Cấu trúc thư mục:
    data/
    ├── clean_trainset_28spk_wav/    # Clean training audio
    ├── noisy_trainset_28spk_wav/    # Noisy training audio
    ├── clean_testset_wav/           # Clean test audio
    └── noisy_testset_wav/           # Noisy test audio
    """
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        sample_rate: int = 16000,
        segment_length: Optional[int] = 32000,
        mode: str = 'train',
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512
    ):
        """
        Args:
            clean_dir: Path to clean audio directory
            noisy_dir: Path to noisy audio directory
            sample_rate: Target sample rate
            segment_length: Length of audio segment (None for full audio)
            mode: 'train' or 'test'
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            win_length: Window length for STFT
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.mode = mode
        
        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        
        # Get list of audio files
        self.file_list = self._get_file_list()
        
        print(f"Loaded {len(self.file_list)} audio pairs for {mode}")
    
    def _get_file_list(self) -> List[str]:
        """Get list of matching clean-noisy file pairs"""
        # Get all wav files from noisy directory
        noisy_files = sorted([f.name for f in self.noisy_dir.glob("*.wav")])
        
        # Filter to only include files that exist in both directories
        file_list = []
        for filename in noisy_files:
            clean_path = self.clean_dir / filename
            noisy_path = self.noisy_dir / filename
            
            if clean_path.exists() and noisy_path.exists():
                file_list.append(filename)
        
        return file_list
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def _load_audio(self, filepath: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(filepath)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze(0)  # [samples]
    
    def _random_crop(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random crop audio to segment_length"""
        if self.segment_length is None:
            return clean, noisy
        
        audio_length = clean.shape[0]
        
        if audio_length < self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - audio_length
            clean = torch.nn.functional.pad(clean, (0, pad_length))
            noisy = torch.nn.functional.pad(noisy, (0, pad_length))
        elif audio_length > self.segment_length:
            # Random crop if too long
            if self.mode == 'train':
                start = random.randint(0, audio_length - self.segment_length)
            else:
                start = 0  # Use beginning for test
            
            clean = clean[start:start + self.segment_length]
            noisy = noisy[start:start + self.segment_length]
        
        return clean, noisy
    
    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT of waveform
        
        Returns:
            STFT tensor [freq_bins, time_frames, 2] (real, imag)
        """
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to real representation
        return torch.stack([stft.real, stft.imag], dim=-1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample
        
        Returns:
            Dictionary containing:
                - noisy: Noisy waveform [samples]
                - clean: Clean waveform [samples]
                - noisy_stft: STFT of noisy [freq, time, 2]
                - clean_stft: STFT of clean [freq, time, 2]
                - filename: Audio filename
        """
        filename = self.file_list[idx]
        
        # Load audio
        clean = self._load_audio(self.clean_dir / filename)
        noisy = self._load_audio(self.noisy_dir / filename)
        
        # Ensure same length
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # Random crop
        clean, noisy = self._random_crop(clean, noisy)
        
        # Compute STFT
        clean_stft = self._compute_stft(clean)
        noisy_stft = self._compute_stft(noisy)
        
        return {
            'noisy': noisy,
            'clean': clean,
            'noisy_stft': noisy_stft,
            'clean_stft': clean_stft,
            'filename': filename
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
    win_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = VoiceBankDEMANDDataset(
        clean_dir=train_clean_dir,
        noisy_dir=train_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        mode='train',
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    
    test_dataset = VoiceBankDEMANDDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        segment_length=None,  # Use full audio for testing
        mode='test',
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Variable length audio
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def download_voicebank_demand(output_dir: str = "./data"):
    """
    Hướng dẫn download VoiceBank + DEMAND dataset
    
    Dataset không thể tự động download do yêu cầu đăng ký.
    Vui lòng download thủ công từ:
    https://datashare.ed.ac.uk/handle/10283/2791
    """
    print("=" * 60)
    print("HƯỚNG DẪN DOWNLOAD VOICEBANK + DEMAND DATASET")
    print("=" * 60)
    print()
    print("1. Truy cập: https://datashare.ed.ac.uk/handle/10283/2791")
    print()
    print("2. Download các file sau:")
    print("   - clean_trainset_28spk_wav.zip")
    print("   - noisy_trainset_28spk_wav.zip")
    print("   - clean_testset_wav.zip")
    print("   - noisy_testset_wav.zip")
    print()
    print("3. Giải nén vào thư mục data/:")
    print(f"   {output_dir}/clean_trainset_28spk_wav/")
    print(f"   {output_dir}/noisy_trainset_28spk_wav/")
    print(f"   {output_dir}/clean_testset_wav/")
    print(f"   {output_dir}/noisy_testset_wav/")
    print()
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir
