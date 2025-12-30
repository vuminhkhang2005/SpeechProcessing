"""
Dataset class for VoiceBank + DEMAND speech denoising dataset

CẢI TIẾN QUAN TRỌNG:
1. Global normalization thay vì per-file normalization
2. Tránh peak normalization để không khuếch đại file yên tĩnh
3. Chuẩn hóa dữ liệu về mean=0, std=1 trên toàn bộ tập training
4. Đảm bảo pairs (clean, noisy) được map chính xác
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')


class AudioNormalizer:
    """
    Global Audio Normalizer - Chuẩn hóa theo mean/std của toàn bộ dataset
    
    QUAN TRỌNG: Tránh per-file peak normalization vì:
    1. File yên tĩnh sẽ bị khuếch đại không hợp lý
    2. Model sẽ học không ổn định
    3. Output sẽ có amplitude không nhất quán
    
    Thay vào đó: Dùng global statistics từ training set
    """
    
    def __init__(self):
        self.global_mean: float = 0.0
        self.global_std: float = 1.0
        self.is_fitted: bool = False
        self.stats_path: Optional[str] = None
    
    def fit(self, audio_files: List[str], sample_rate: int = 16000, 
            max_files: int = 1000) -> None:
        """
        Tính global mean và std từ một tập files mẫu
        
        Args:
            audio_files: List đường dẫn tới audio files
            sample_rate: Sample rate
            max_files: Số file tối đa để tính statistics (để tiết kiệm thời gian)
        """
        print(f"Computing global normalization statistics from {min(len(audio_files), max_files)} files...")
        
        # Sample files nếu quá nhiều
        if len(audio_files) > max_files:
            sample_files = random.sample(audio_files, max_files)
        else:
            sample_files = audio_files
        
        all_samples = []
        for filepath in sample_files:
            try:
                audio, _ = librosa.load(filepath, sr=sample_rate, mono=True)
                # Lấy sample ngẫu nhiên để tiết kiệm memory
                if len(audio) > sample_rate * 10:  # > 10 seconds
                    start = random.randint(0, len(audio) - sample_rate * 10)
                    audio = audio[start:start + sample_rate * 10]
                all_samples.append(audio)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        
        if not all_samples:
            print("Warning: No files could be loaded, using default statistics")
            self.global_mean = 0.0
            self.global_std = 1.0
            self.is_fitted = True
            return
        
        # Concatenate và tính statistics
        all_audio = np.concatenate(all_samples)
        self.global_mean = float(np.mean(all_audio))
        self.global_std = float(np.std(all_audio))
        
        # Avoid division by zero
        if self.global_std < 1e-6:
            self.global_std = 1.0
        
        self.is_fitted = True
        print(f"Global statistics: mean={self.global_mean:.6f}, std={self.global_std:.6f}")
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa audio về mean=0, std=1 sử dụng global statistics
        """
        if not self.is_fitted:
            # Default: just zero-mean
            return audio - np.mean(audio)
        
        return (audio - self.global_mean) / self.global_std
    
    def denormalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Khôi phục audio về scale gốc
        """
        if not self.is_fitted:
            return audio
        
        return audio * self.global_std + self.global_mean
    
    def save(self, path: str) -> None:
        """Lưu statistics"""
        stats = {
            'global_mean': self.global_mean,
            'global_std': self.global_std,
            'is_fitted': self.is_fitted
        }
        with open(path, 'w') as f:
            json.dump(stats, f)
        self.stats_path = path
    
    def load(self, path: str) -> None:
        """Load statistics"""
        with open(path, 'r') as f:
            stats = json.load(f)
        self.global_mean = stats['global_mean']
        self.global_std = stats['global_std']
        self.is_fitted = stats['is_fitted']
        self.stats_path = path


class VoiceBankDEMANDDataset(Dataset):
    """
    VoiceBank + DEMAND dataset for speech denoising
    
    CẢI TIẾN:
    1. Global normalization thay vì per-file
    2. Proper pair matching giữa clean và noisy
    3. Segment với overlap để augmentation
    4. STFT preprocessing tích hợp
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
            clean_dir: Path to clean audio directory
            noisy_dir: Path to noisy audio directory
            sample_rate: Target sample rate
            segment_length: Length of audio segments (None for full audio)
            mode: Dataset mode ('train', 'val', 'test')
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            normalizer: AudioNormalizer instance (sẽ được tạo nếu None)
            augment: Enable augmentation (chỉ cho training)
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.augment = augment and (mode == 'train')
        
        # STFT window
        self.window = torch.hann_window(win_length)
        
        # Normalizer - QUAN TRỌNG!
        self.normalizer = normalizer if normalizer else AudioNormalizer()
        
        # Collect file pairs
        self.file_pairs = self._collect_files()
        
        if len(self.file_pairs) == 0:
            raise ValueError(f"No audio files found in {clean_dir} or {noisy_dir}")
        
        print(f"[{mode.upper()}] Found {len(self.file_pairs)} audio pairs")
        
        # Fit normalizer trên training data
        if mode == 'train' and not self.normalizer.is_fitted:
            clean_files = [pair[0] for pair in self.file_pairs]
            self.normalizer.fit(clean_files, sample_rate)
    
    def _collect_files(self) -> List[Tuple[str, str]]:
        """
        Collect matching clean-noisy pairs
        
        VoiceBank+DEMAND có cấu trúc: cùng tên file trong clean và noisy dirs
        """
        extensions = {'.wav', '.flac', '.mp3', '.ogg'}
        
        # Get all clean files
        clean_files = {}
        for f in self.clean_dir.iterdir():
            if f.suffix.lower() in extensions:
                clean_files[f.stem] = f
        
        # Match với noisy files
        pairs = []
        for f in self.noisy_dir.iterdir():
            if f.suffix.lower() in extensions:
                name = f.stem
                if name in clean_files:
                    pairs.append((str(clean_files[name]), str(f)))
        
        # Sort để reproducibility
        pairs.sort(key=lambda x: x[0])
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def _load_audio(self, filepath: str) -> np.ndarray:
        """Load và resample audio"""
        audio, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
        return audio
    
    def _get_segment(
        self, 
        clean: np.ndarray, 
        noisy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy segment từ audio với cùng vị trí cho clean và noisy
        """
        # Đảm bảo cùng độ dài
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        if self.segment_length is None or min_len <= self.segment_length:
            # Pad nếu cần
            if self.segment_length is not None and min_len < self.segment_length:
                pad_len = self.segment_length - min_len
                clean = np.pad(clean, (0, pad_len), mode='constant')
                noisy = np.pad(noisy, (0, pad_len), mode='constant')
            return clean, noisy
        
        # Random segment (cùng vị trí cho clean và noisy!)
        start = random.randint(0, min_len - self.segment_length)
        return clean[start:start + self.segment_length], noisy[start:start + self.segment_length]
    
    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT
        
        Returns:
            STFT tensor [freq_bins, time_frames, 2] (real, imag)
        """
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
        
        # [freq, time, 2]
        return torch.stack([stft_out.real, stft_out.imag], dim=-1)
    
    def _augment(
        self, 
        clean: np.ndarray, 
        noisy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Data augmentation cho training
        
        Techniques:
        1. Random gain (nhẹ để không ảnh hưởng normalization)
        2. Random noise level adjustment
        """
        if not self.augment:
            return clean, noisy
        
        # Random gain (±3dB)
        if random.random() < 0.3:
            gain_db = random.uniform(-3, 3)
            gain = 10 ** (gain_db / 20)
            clean = clean * gain
            noisy = noisy * gain
        
        # Đôi khi điều chỉnh tỷ lệ noise
        if random.random() < 0.2:
            noise = noisy - clean
            noise_scale = random.uniform(0.8, 1.2)
            noisy = clean + noise * noise_scale
        
        return clean, noisy
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item
        
        Returns:
            Dictionary với:
            - 'clean': Clean waveform tensor
            - 'noisy': Noisy waveform tensor
            - 'clean_stft': Clean STFT [freq, time, 2]
            - 'noisy_stft': Noisy STFT [freq, time, 2]
            - 'filename': Filename
        """
        clean_path, noisy_path = self.file_pairs[idx]
        
        # Load audio
        clean = self._load_audio(clean_path)
        noisy = self._load_audio(noisy_path)
        
        # Get segment
        clean, noisy = self._get_segment(clean, noisy)
        
        # Augmentation
        clean, noisy = self._augment(clean, noisy)
        
        # Global normalization - QUAN TRỌNG!
        clean = self.normalizer.normalize(clean)
        noisy = self.normalizer.normalize(noisy)
        
        # Convert to tensor
        clean_tensor = torch.from_numpy(clean).float()
        noisy_tensor = torch.from_numpy(noisy).float()
        
        # Compute STFT
        clean_stft = self._compute_stft(clean_tensor)
        noisy_stft = self._compute_stft(noisy_tensor)
        
        return {
            'clean': clean_tensor,
            'noisy': noisy_tensor,
            'clean_stft': clean_stft,
            'noisy_stft': noisy_stft,
            'filename': Path(clean_path).name
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
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_clean_dir: Path to training clean audio
        train_noisy_dir: Path to training noisy audio
        test_clean_dir: Path to test clean audio  
        test_noisy_dir: Path to test noisy audio
        sample_rate: Target sample rate
        segment_length: Length of audio segments
        batch_size: Batch size
        num_workers: Number of workers
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        val_split: Fraction of training data for validation
    
    Returns:
        train_loader, val_loader
    """
    # Tạo normalizer và fit trên training data
    normalizer = AudioNormalizer()
    
    # Tạo training dataset (sẽ fit normalizer)
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
    
    # Sử dụng test set làm validation (như VoiceBank+DEMAND standard)
    # hoặc split từ training nếu cần
    val_dataset = VoiceBankDEMANDDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,  # Vẫn dùng segment cho validation
        mode='val',
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        normalizer=normalizer,  # Dùng cùng normalizer!
        augment=False
    )
    
    # Lưu normalizer statistics
    stats_dir = Path(train_clean_dir).parent
    normalizer_path = stats_dir / 'normalizer_stats.json'
    try:
        normalizer.save(str(normalizer_path))
        print(f"Saved normalizer statistics to {normalizer_path}")
    except Exception as e:
        print(f"Warning: Could not save normalizer stats: {e}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
    Setup dataset from Google Drive
    
    Args:
        gdrive_path: Path to dataset folder in Google Drive
        gdrive_folder_id: Google Drive folder ID
    
    Returns:
        Dictionary with paths or None if failed
    """
    try:
        # Check if running in Colab
        from google.colab import drive
        
        # Mount Drive
        mount_path = '/content/drive'
        if not os.path.exists(mount_path):
            drive.mount(mount_path)
        
        # Determine dataset path
        if gdrive_path:
            dataset_path = Path(gdrive_path)
        else:
            # Default paths to check
            possible_paths = [
                '/content/drive/MyDrive/speech_denoising_data',
                '/content/drive/MyDrive/VoiceBank-DEMAND',
                '/content/drive/MyDrive/datasets/VoiceBank-DEMAND'
            ]
            
            dataset_path = None
            for p in possible_paths:
                if Path(p).exists():
                    dataset_path = Path(p)
                    break
            
            if dataset_path is None:
                print("Could not find dataset in Google Drive")
                return None
        
        # Expected subdirectories
        paths = {
            'train_clean_dir': str(dataset_path / 'clean_trainset_28spk_wav'),
            'train_noisy_dir': str(dataset_path / 'noisy_trainset_28spk_wav'),
            'test_clean_dir': str(dataset_path / 'clean_testset_wav'),
            'test_noisy_dir': str(dataset_path / 'noisy_testset_wav')
        }
        
        # Verify all directories exist
        for name, path in paths.items():
            if not Path(path).exists():
                print(f"Warning: {name} not found at {path}")
                # Try alternative naming
                alt_paths = {
                    'train_clean_dir': ['clean_trainset_wav', 'trainset_clean'],
                    'train_noisy_dir': ['noisy_trainset_wav', 'trainset_noisy'],
                    'test_clean_dir': ['clean_testset', 'testset_clean'],
                    'test_noisy_dir': ['noisy_testset', 'testset_noisy']
                }
                for alt in alt_paths.get(name, []):
                    alt_path = str(dataset_path / alt)
                    if Path(alt_path).exists():
                        paths[name] = alt_path
                        print(f"Found alternative: {alt_path}")
                        break
        
        print(f"Dataset paths configured:")
        for name, path in paths.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name}: {path}")
        
        return paths
    
    except ImportError:
        print("Not running in Google Colab")
        return None
    except Exception as e:
        print(f"Error setting up Google Drive dataset: {e}")
        return None


if __name__ == '__main__':
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--noisy_dir', type=str, required=True)
    args = parser.parse_args()
    
    dataset = VoiceBankDEMANDDataset(
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        mode='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Clean shape: {sample['clean'].shape}")
    print(f"Noisy shape: {sample['noisy'].shape}")
    print(f"Clean STFT shape: {sample['clean_stft'].shape}")
    print(f"Noisy STFT shape: {sample['noisy_stft'].shape}")
