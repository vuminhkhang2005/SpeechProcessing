"""
VoiceBank + DEMAND Dataset for Speech Denoising

Cải tiến chuẩn hóa dữ liệu:
1. Global normalization (mean=0, std=1) thay vì per-file normalization
2. Tính mean/std trên toàn bộ training set và áp dụng cho tất cả
3. Tránh khuếch đại các file có nhiễu yếu

References:
- LeCun: Normalize using global statistics from training set
- https://bioacoustics.stackexchange.com/questions/516/
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf


class AudioNormalizer:
    """
    Global Audio Normalizer
    
    Chuẩn hóa audio theo global statistics (mean, std) tính trên training set.
    QUAN TRỌNG: Không chuẩn hóa theo từng file riêng lẻ!
    
    Theo khuyến cáo của LeCun và các nghiên cứu về audio preprocessing:
    - Chuẩn hóa về mean=0, std=1 sử dụng statistics từ training set
    - Áp dụng cùng một transformation cho tất cả files
    """
    
    def __init__(
        self,
        global_mean: float = 0.0,
        global_std: float = 0.1,  # Typical std for speech audio
        eps: float = 1e-8
    ):
        """
        Args:
            global_mean: Global mean computed from training set
            global_std: Global std computed from training set
            eps: Small constant to prevent division by zero
        """
        self.global_mean = global_mean
        self.global_std = global_std
        self.eps = eps
    
    def normalize(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio using global statistics"""
        return (audio - self.global_mean) / (self.global_std + self.eps)
    
    def denormalize(self, audio: torch.Tensor) -> torch.Tensor:
        """Denormalize audio back to original scale"""
        return audio * (self.global_std + self.eps) + self.global_mean
    
    @classmethod
    def compute_global_statistics(
        cls,
        audio_dir: str,
        sample_rate: int = 16000,
        max_files: int = 1000,
        seed: int = 42
    ) -> Tuple[float, float]:
        """
        Tính global mean và std từ training set
        
        Args:
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            max_files: Maximum number of files to process (for speed)
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (global_mean, global_std)
        """
        audio_dir = Path(audio_dir)
        wav_files = list(audio_dir.glob('*.wav'))
        
        # Sample subset for efficiency
        if len(wav_files) > max_files:
            random.seed(seed)
            wav_files = random.sample(wav_files, max_files)
        
        print(f"Computing global statistics from {len(wav_files)} files...")
        
        all_samples = []
        for wav_file in wav_files:
            try:
                audio, sr = librosa.load(wav_file, sr=sample_rate)
                all_samples.append(audio)
            except Exception as e:
                print(f"Warning: Could not load {wav_file}: {e}")
        
        # Concatenate all samples
        all_audio = np.concatenate(all_samples)
        
        global_mean = float(np.mean(all_audio))
        global_std = float(np.std(all_audio))
        
        print(f"Global statistics: mean={global_mean:.6f}, std={global_std:.6f}")
        
        return global_mean, global_std
    
    def save(self, path: str):
        """Save normalizer statistics to file"""
        stats = {
            'global_mean': self.global_mean,
            'global_std': self.global_std
        }
        with open(path, 'w') as f:
            json.dump(stats, f)
    
    @classmethod
    def load(cls, path: str) -> 'AudioNormalizer':
        """Load normalizer statistics from file"""
        with open(path, 'r') as f:
            stats = json.load(f)
        return cls(
            global_mean=stats['global_mean'],
            global_std=stats['global_std']
        )


class VoiceBankDEMANDDataset(Dataset):
    """
    VoiceBank + DEMAND Dataset for Speech Denoising
    
    Cải tiến:
    1. Global normalization thay vì per-file normalization
    2. Consistent sample rate and length
    3. Proper pairing of clean/noisy files
    4. On-the-fly STFT computation
    """
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        sample_rate: int = 16000,
        segment_length: Optional[int] = 32000,  # 2 seconds at 16kHz
        mode: str = 'train',  # 'train', 'val', 'test'
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        normalizer: Optional[AudioNormalizer] = None,
        augment: bool = False
    ):
        """
        Args:
            clean_dir: Directory containing clean audio files
            noisy_dir: Directory containing noisy audio files
            sample_rate: Target sample rate
            segment_length: Length of audio segments (None for full audio)
            mode: Dataset mode ('train', 'val', 'test')
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            normalizer: AudioNormalizer instance (uses default if None)
            augment: Whether to apply data augmentation
        """
        super().__init__()
        
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.augment = augment and mode == 'train'
        
        # Setup normalizer - QUAN TRỌNG: dùng global statistics
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            # Default values based on typical speech audio
            # Should be computed from actual training data for best results
            self.normalizer = AudioNormalizer(global_mean=0.0, global_std=0.05)
        
        # Hann window for STFT
        self.window = torch.hann_window(win_length)
        
        # Load file list
        self.file_list = self._get_file_list()
        
        if len(self.file_list) == 0:
            raise ValueError(
                f"No matching audio files found.\n"
                f"Clean dir: {self.clean_dir}\n"
                f"Noisy dir: {self.noisy_dir}"
            )
        
        print(f"Loaded {len(self.file_list)} audio pairs for {mode}")
    
    def _get_file_list(self) -> List[Tuple[Path, Path]]:
        """Get list of matching clean/noisy file pairs"""
        clean_files = {f.name: f for f in self.clean_dir.glob('*.wav')}
        noisy_files = {f.name: f for f in self.noisy_dir.glob('*.wav')}
        
        # Find matching pairs
        common_names = set(clean_files.keys()) & set(noisy_files.keys())
        
        file_list = [(clean_files[name], noisy_files[name]) 
                     for name in sorted(common_names)]
        
        return file_list
    
    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load audio file and convert to tensor"""
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        return torch.from_numpy(audio).float()
    
    def _get_segment(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract random segment from audio pair"""
        if self.segment_length is None or len(clean) <= self.segment_length:
            # Use full audio
            if self.segment_length is not None:
                # Pad if necessary
                if len(clean) < self.segment_length:
                    pad_len = self.segment_length - len(clean)
                    clean = F.pad(clean, (0, pad_len))
                    noisy = F.pad(noisy, (0, pad_len))
            return clean, noisy
        
        # Random segment
        max_start = len(clean) - self.segment_length
        if self.mode == 'train':
            start = random.randint(0, max_start)
        else:
            # Deterministic for validation/test
            start = max_start // 2
        
        clean_seg = clean[start:start + self.segment_length]
        noisy_seg = noisy[start:start + self.segment_length]
        
        return clean_seg, noisy_seg
    
    def _augment(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        # Random gain (only for training)
        if random.random() < 0.3:
            gain = random.uniform(0.8, 1.2)
            clean = clean * gain
            noisy = noisy * gain
        
        return clean, noisy
    
    def _compute_stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT of audio
        
        Returns:
            STFT tensor [freq_bins, time_frames, 2] (real, imag)
        """
        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to real representation [batch, freq, time, 2]
        stft_real = torch.stack([stft.real, stft.imag], dim=-1)
        
        return stft_real.squeeze(0)  # [freq, time, 2]
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean_path, noisy_path = self.file_list[idx]
        
        # Load audio
        clean = self._load_audio(clean_path)
        noisy = self._load_audio(noisy_path)
        
        # Ensure same length
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # Get segment
        clean, noisy = self._get_segment(clean, noisy)
        
        # Apply augmentation (training only)
        if self.augment:
            clean, noisy = self._augment(clean, noisy)
        
        # QUAN TRỌNG: Normalize using global statistics
        # Không chuẩn hóa theo từng file riêng lẻ!
        clean_norm = self.normalizer.normalize(clean)
        noisy_norm = self.normalizer.normalize(noisy)
        
        # Compute STFT
        clean_stft = self._compute_stft(clean_norm)
        noisy_stft = self._compute_stft(noisy_norm)
        
        return {
            'clean': clean,  # Original scale (for metrics)
            'noisy': noisy,  # Original scale (for metrics)
            'clean_norm': clean_norm,  # Normalized
            'noisy_norm': noisy_norm,  # Normalized
            'clean_stft': clean_stft,
            'noisy_stft': noisy_stft,
            'filename': clean_path.name
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
    compute_normalizer: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_clean_dir: Directory with clean training audio
        train_noisy_dir: Directory with noisy training audio  
        test_clean_dir: Directory with clean test audio
        test_noisy_dir: Directory with noisy test audio
        sample_rate: Target sample rate
        segment_length: Audio segment length
        batch_size: Batch size
        num_workers: Number of data loading workers
        n_fft: FFT size
        hop_length: STFT hop length
        win_length: STFT window length
        compute_normalizer: Whether to compute global normalization statistics
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Setup normalizer - compute from training data
    stats_file = Path(train_clean_dir).parent / 'normalizer_stats.json'
    
    if compute_normalizer and not stats_file.exists():
        print("Computing global normalization statistics from training data...")
        global_mean, global_std = AudioNormalizer.compute_global_statistics(
            train_clean_dir, sample_rate=sample_rate, max_files=500
        )
        normalizer = AudioNormalizer(global_mean=global_mean, global_std=global_std)
        normalizer.save(str(stats_file))
        print(f"Saved normalizer statistics to {stats_file}")
    elif stats_file.exists():
        print(f"Loading normalizer statistics from {stats_file}")
        normalizer = AudioNormalizer.load(str(stats_file))
        print(f"Global stats: mean={normalizer.global_mean:.6f}, std={normalizer.global_std:.6f}")
    else:
        # Use default values
        normalizer = AudioNormalizer(global_mean=0.0, global_std=0.05)
        print("Using default normalizer statistics")
    
    # Create datasets
    train_dataset = VoiceBankDEMANDDataset(
        clean_dir=train_clean_dir,
        noisy_dir=train_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        mode='train',
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        normalizer=normalizer,
        augment=True
    )
    
    val_dataset = VoiceBankDEMANDDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        segment_length=None,  # Full audio for validation
        mode='val',
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        normalizer=normalizer,  # Same normalizer!
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Smaller batch size for validation (full length audio)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full length audio can vary
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def setup_gdrive_dataset(
    gdrive_path: Optional[str] = None,
    gdrive_folder_id: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    Setup dataset from Google Drive (for Google Colab)
    
    Args:
        gdrive_path: Path to dataset folder in Google Drive
        gdrive_folder_id: Google Drive folder ID
    
    Returns:
        Dictionary with dataset paths or None if setup fails
    """
    try:
        from google.colab import drive
        
        # Mount Google Drive
        mount_point = '/content/drive'
        if not os.path.ismount(mount_point):
            print("Mounting Google Drive...")
            drive.mount(mount_point)
        
        # Try different paths
        possible_paths = []
        
        if gdrive_path:
            possible_paths.append(gdrive_path)
        
        # Common locations
        possible_paths.extend([
            '/content/drive/MyDrive/speech_denoising_data',
            '/content/drive/MyDrive/datasets/speech_denoising_data',
            '/content/drive/MyDrive/VoiceBank_DEMAND'
        ])
        
        data_root = None
        for path in possible_paths:
            if os.path.exists(path):
                data_root = path
                print(f"Found dataset at: {data_root}")
                break
        
        if data_root is None:
            print("Could not find dataset in Google Drive.")
            print("Tried paths:", possible_paths)
            return None
        
        # Setup paths
        paths = {
            'train_clean_dir': os.path.join(data_root, 'clean_trainset_28spk_wav'),
            'train_noisy_dir': os.path.join(data_root, 'noisy_trainset_28spk_wav'),
            'test_clean_dir': os.path.join(data_root, 'clean_testset_wav'),
            'test_noisy_dir': os.path.join(data_root, 'noisy_testset_wav')
        }
        
        # Verify paths exist
        for name, path in paths.items():
            if not os.path.exists(path):
                print(f"Warning: {name} not found at {path}")
        
        return paths
        
    except ImportError:
        print("Not running in Google Colab environment.")
        return None
    except Exception as e:
        print(f"Error setting up Google Drive dataset: {e}")
        return None


if __name__ == '__main__':
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, default='./data/clean_trainset_28spk_wav')
    parser.add_argument('--noisy_dir', type=str, default='./data/noisy_trainset_28spk_wav')
    args = parser.parse_args()
    
    # Compute global statistics
    if os.path.exists(args.clean_dir):
        mean, std = AudioNormalizer.compute_global_statistics(args.clean_dir)
        print(f"Global mean: {mean:.6f}")
        print(f"Global std: {std:.6f}")
        
        # Test dataset
        normalizer = AudioNormalizer(global_mean=mean, global_std=std)
        dataset = VoiceBankDEMANDDataset(
            clean_dir=args.clean_dir,
            noisy_dir=args.noisy_dir,
            normalizer=normalizer
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Clean shape: {sample['clean'].shape}")
        print(f"Noisy shape: {sample['noisy'].shape}")
        print(f"Clean STFT shape: {sample['clean_stft'].shape}")
        print(f"Noisy STFT shape: {sample['noisy_stft'].shape}")
    else:
        print(f"Directory not found: {args.clean_dir}")
        print("Please download the dataset first: python download_dataset.py")
