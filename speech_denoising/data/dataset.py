"""
Dataset module for Speech Denoising

Supports loading VoiceBank + DEMAND dataset from:
- Local directories
- Google Drive (for Google Colab training)

Usage:
    # Local dataset
    dataset = VoiceBankDEMANDDataset(
        clean_dir='./data/clean_trainset_28spk_wav',
        noisy_dir='./data/noisy_trainset_28spk_wav'
    )
    
    # Google Drive dataset (in Colab)
    from data.dataset import setup_gdrive_dataset, create_dataloaders
    
    paths = setup_gdrive_dataset(
        gdrive_folder_id='1mDHfxtzvC-7kw0YXF0dFAcYlh7GAb2-',  # from Drive URL
        # OR
        gdrive_path='/content/drive/MyDrive/datasets'
    )
    
    train_loader, val_loader = create_dataloaders(**paths)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np


def is_colab() -> bool:
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_google_drive(mount_path: str = '/content/drive') -> bool:
    """
    Mount Google Drive in Colab
    
    Args:
        mount_path: Path to mount Google Drive
        
    Returns:
        True if successful, False otherwise
    """
    if not is_colab():
        print("⚠️ Not running in Google Colab. Google Drive mount skipped.")
        return False
    
    try:
        from google.colab import drive
        
        if not os.path.ismount(mount_path):
            print("📂 Mounting Google Drive...")
            drive.mount(mount_path)
            print("✅ Google Drive mounted successfully!")
        else:
            print("✅ Google Drive already mounted.")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False


def get_gdrive_path_from_id(folder_id: str, mount_path: str = '/content/drive') -> Optional[str]:
    """
    Get the local path for a Google Drive folder by its ID
    
    Note: This searches common locations. For shared folders, 
    you may need to add them to "My Drive" first.
    
    Args:
        folder_id: Google Drive folder ID (from URL)
        mount_path: Google Drive mount path
        
    Returns:
        Local path to the folder or None if not found
    """
    if not is_colab():
        return None
    
    # Common locations to search
    search_paths = [
        f"{mount_path}/MyDrive",
        f"{mount_path}/My Drive",
        f"{mount_path}/Shareddrives",
    ]
    
    # This is a simplified approach - for shared folders,
    # users should use the direct path instead
    print(f"ℹ️ For Google Drive folders, please use the direct path method.")
    print(f"   Example: /content/drive/MyDrive/datasets")
    
    return None


def setup_gdrive_dataset(
    gdrive_path: Optional[str] = None,
    gdrive_folder_id: Optional[str] = None,
    mount_path: str = '/content/drive',
    auto_mount: bool = True
) -> Dict[str, str]:
    """
    Setup dataset paths for Google Drive in Colab
    
    The dataset should have this structure on Google Drive:
    <your_folder>/
    ├── clean_trainset_28spk_wav/   (11,572 .wav files)
    ├── noisy_trainset_28spk_wav/   (11,572 .wav files)
    ├── clean_testset_wav/          (824 .wav files)
    └── noisy_testset_wav/          (824 .wav files)
    
    Args:
        gdrive_path: Direct path to dataset folder in Google Drive
                    Example: '/content/drive/MyDrive/datasets'
        gdrive_folder_id: Google Drive folder ID (from URL)
                         Example: '1mDHfxtzvC-7kw0YXF0dFAcYlh7GAb2-'
        mount_path: Google Drive mount path
        auto_mount: Automatically mount Google Drive
        
    Returns:
        Dictionary with dataset paths ready for create_dataloaders()
        
    Example:
        # Option 1: Using direct path (recommended)
        paths = setup_gdrive_dataset(
            gdrive_path='/content/drive/MyDrive/datasets'
        )
        
        # Option 2: Using folder ID from URL
        # URL: https://drive.google.com/drive/folders/1mDHfxtzvC-7kw0YXF0dFAcYlh7GAb2-
        paths = setup_gdrive_dataset(
            gdrive_folder_id='1mDHfxtzvC-7kw0YXF0dFAcYlh7GAb2-'
        )
        
        # Then create dataloaders
        train_loader, val_loader = create_dataloaders(**paths)
    """
    if not is_colab():
        print("⚠️ Not running in Google Colab.")
        print("   For local usage, specify paths directly in config.yaml")
        return {}
    
    # Mount Google Drive if needed
    if auto_mount:
        mount_google_drive(mount_path)
    
    # Determine the dataset path
    data_path = None
    
    if gdrive_path:
        data_path = Path(gdrive_path)
    elif gdrive_folder_id:
        # Try to find the folder
        found_path = get_gdrive_path_from_id(gdrive_folder_id, mount_path)
        if found_path:
            data_path = Path(found_path)
        else:
            print(f"❌ Could not locate folder with ID: {gdrive_folder_id}")
            print("   Please use gdrive_path parameter with the direct path instead.")
            print(f"   Example: gdrive_path='/content/drive/MyDrive/datasets'")
            return {}
    else:
        print("❌ Please provide either gdrive_path or gdrive_folder_id")
        return {}
    
    # Verify the path exists
    if not data_path.exists():
        print(f"❌ Path does not exist: {data_path}")
        print("   Make sure Google Drive is mounted and the path is correct.")
        return {}
    
    # Check for required subdirectories
    required_dirs = [
        "clean_trainset_28spk_wav",
        "noisy_trainset_28spk_wav", 
        "clean_testset_wav",
        "noisy_testset_wav"
    ]
    
    paths = {}
    all_found = True
    
    print(f"\n📂 Checking dataset at: {data_path}")
    print("-" * 50)
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if dir_path.exists():
            wav_count = len(list(dir_path.glob("*.wav")))
            print(f"  ✅ {dir_name}: {wav_count} files")
            
            # Map to the expected path keys
            if "train_clean" in dir_name or "clean_trainset" in dir_name:
                paths['train_clean_dir'] = str(dir_path)
            elif "train_noisy" in dir_name or "noisy_trainset" in dir_name:
                paths['train_noisy_dir'] = str(dir_path)
            elif "test_clean" in dir_name or "clean_testset" in dir_name:
                paths['test_clean_dir'] = str(dir_path)
            elif "test_noisy" in dir_name or "noisy_testset" in dir_name:
                paths['test_noisy_dir'] = str(dir_path)
        else:
            print(f"  ❌ {dir_name}: NOT FOUND")
            all_found = False
    
    print("-" * 50)
    
    if all_found:
        print("✅ Dataset is ready for training!")
    else:
        print("⚠️ Some directories are missing.")
        print("   Please upload the complete dataset to Google Drive.")
    
    return paths


class VoiceBankDEMANDDataset(Dataset):
    """
    VoiceBank + DEMAND Dataset for speech denoising
    
    Supports loading from local directories or Google Drive paths.
    
    Args:
        clean_dir: Path to clean audio directory
        noisy_dir: Path to noisy audio directory
        sample_rate: Target sample rate (default: 16000)
        segment_length: Length of audio segments in samples (None for full audio)
        mode: 'train' or 'test'
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        augment: Apply data augmentation (train mode only)
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
        win_length: int = 512,
        augment: bool = False
    ):
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
        
        # Get file list
        self.filenames = self._get_filenames()
        
        if len(self.filenames) == 0:
            raise ValueError(
                f"No matching files found!\n"
                f"Clean dir: {self.clean_dir}\n"
                f"Noisy dir: {self.noisy_dir}"
            )
        
        print(f"📊 Dataset loaded: {len(self.filenames)} samples ({mode} mode)")
    
    def _get_filenames(self) -> List[str]:
        """Get list of matching filenames in both directories"""
        # Get files from both directories
        clean_files = set(f.name for f in self.clean_dir.glob("*.wav"))
        noisy_files = set(f.name for f in self.noisy_dir.glob("*.wav"))
        
        # Find common files
        common_files = sorted(clean_files & noisy_files)
        
        if len(common_files) < len(clean_files):
            missing = len(clean_files) - len(common_files)
            print(f"⚠️ {missing} files in clean_dir have no matching noisy files")
        
        return common_files
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def _load_audio(self, filepath: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(str(filepath))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def _segment_audio(
        self, 
        clean: torch.Tensor, 
        noisy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract random segment from audio"""
        if self.segment_length is None:
            return clean, noisy
        
        length = clean.shape[0]
        
        if length <= self.segment_length:
            # Pad if too short
            pad_length = self.segment_length - length
            clean = torch.nn.functional.pad(clean, (0, pad_length))
            noisy = torch.nn.functional.pad(noisy, (0, pad_length))
        else:
            # Random crop
            start = random.randint(0, length - self.segment_length)
            clean = clean[start:start + self.segment_length]
            noisy = noisy[start:start + self.segment_length]
        
        return clean, noisy
    
    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute STFT and return real/imag representation"""
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
        
        # Convert to real representation [freq, time, 2]
        stft_real = torch.stack([stft.real, stft.imag], dim=-1)
        
        return stft_real
    
    def _augment(
        self, 
        clean: torch.Tensor, 
        noisy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        # Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.5, 1.5)
            clean = clean * gain
            noisy = noisy * gain
        
        # Random flip
        if random.random() < 0.5:
            clean = torch.flip(clean, dims=[0])
            noisy = torch.flip(noisy, dims=[0])
        
        return clean, noisy
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.filenames[idx]
        
        # Load audio
        clean = self._load_audio(self.clean_dir / filename)
        noisy = self._load_audio(self.noisy_dir / filename)
        
        # Ensure same length
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # Segment audio
        clean, noisy = self._segment_audio(clean, noisy)
        
        # Augmentation
        if self.augment:
            clean, noisy = self._augment(clean, noisy)
        
        # Compute STFT
        clean_stft = self._compute_stft(clean)
        noisy_stft = self._compute_stft(noisy)
        
        return {
            'clean': clean,
            'noisy': noisy,
            'clean_stft': clean_stft,
            'noisy_stft': noisy_stft,
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
    win_length: int = 512,
    augment: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        train_clean_dir: Path to training clean audio
        train_noisy_dir: Path to training noisy audio
        test_clean_dir: Path to test clean audio
        test_noisy_dir: Path to test noisy audio
        sample_rate: Audio sample rate
        segment_length: Audio segment length in samples
        batch_size: Batch size
        num_workers: Number of data loading workers
        n_fft: FFT size
        hop_length: STFT hop length
        win_length: STFT window length
        augment: Apply data augmentation
        
    Returns:
        train_loader, val_loader
    """
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
        augment=augment
    )
    
    val_dataset = VoiceBankDEMANDDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        mode='test',
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
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
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Convenience function for quick setup in Colab
def setup_colab_training(
    gdrive_path: str = '/content/drive/MyDrive/datasets',
    batch_size: int = 8,
    num_workers: int = 2,
    sample_rate: int = 16000,
    segment_length: int = 32000,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512
) -> Tuple[DataLoader, DataLoader]:
    """
    Quick setup for training in Google Colab
    
    This function handles:
    1. Mounting Google Drive
    2. Verifying dataset
    3. Creating dataloaders
    
    Args:
        gdrive_path: Path to dataset folder in Google Drive
        batch_size: Batch size (default: 8 for Colab)
        num_workers: Data loading workers (default: 2 for Colab)
        Other args: Audio/STFT parameters
        
    Returns:
        train_loader, val_loader
        
    Example:
        train_loader, val_loader = setup_colab_training(
            gdrive_path='/content/drive/MyDrive/datasets'
        )
    """
    # Setup paths
    paths = setup_gdrive_dataset(gdrive_path=gdrive_path)
    
    if not paths:
        raise ValueError("Failed to setup Google Drive dataset. Check the path.")
    
    # Add other parameters
    paths.update({
        'sample_rate': sample_rate,
        'segment_length': segment_length,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'win_length': win_length
    })
    
    # Create dataloaders
    return create_dataloaders(**paths)
