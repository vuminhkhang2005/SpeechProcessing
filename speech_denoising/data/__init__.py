"""
Data module for Speech Denoising

Supports loading datasets from:
- Local directories
- Google Drive (for Google Colab)
"""

from .dataset import (
    VoiceBankDEMANDDataset,
    create_dataloaders,
    mount_google_drive,
    setup_gdrive_dataset,
    is_colab
)

__all__ = [
    'VoiceBankDEMANDDataset',
    'create_dataloaders',
    'mount_google_drive',
    'setup_gdrive_dataset',
    'is_colab'
]
