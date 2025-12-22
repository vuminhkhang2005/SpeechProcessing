"""
Evaluation metrics for speech enhancement
"""

import torch
import numpy as np
from typing import Optional, Dict
import warnings

# Suppress warnings from PESQ
warnings.filterwarnings('ignore')

# Check if PESQ is available
try:
    from pesq import pesq as pesq_func
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: PESQ not installed. PESQ metric will not be available.")
    print("To install PESQ on Windows:")
    print("  1. Install Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("  2. pip install pesq")
    print("Training and inference will still work without PESQ.\n")


def calculate_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio improvement
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
    
    Returns:
        SNR in dB
    """
    noise = clean - enhanced
    clean_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2) + 1e-8
    
    snr = 10 * np.log10(clean_power / noise_power)
    return snr


def calculate_pesq(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int = 16000,
    mode: str = 'wb'
) -> float:
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality)
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        sample_rate: Sample rate (8000 for nb, 16000 for wb)
        mode: 'wb' for wideband, 'nb' for narrowband
    
    Returns:
        PESQ score (-0.5 to 4.5), or 0.0 if PESQ not available
    """
    if not PESQ_AVAILABLE:
        return 0.0
    
    try:
        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        score = pesq_func(sample_rate, clean, enhanced, mode)
        return score
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        return 0.0


def calculate_stoi(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int = 16000,
    extended: bool = False
) -> float:
    """
    Calculate STOI (Short-Time Objective Intelligibility)
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        sample_rate: Sample rate
        extended: Use extended STOI
    
    Returns:
        STOI score (0 to 1)
    """
    try:
        from pystoi import stoi
        
        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        score = stoi(clean, enhanced, sample_rate, extended=extended)
        return score
    except Exception as e:
        print(f"STOI calculation failed: {e}")
        return 0.0


def calculate_si_sdr(
    clean: np.ndarray,
    enhanced: np.ndarray
) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
    
    Returns:
        SI-SDR in dB
    """
    # Ensure same length
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    # Zero mean
    clean = clean - np.mean(clean)
    enhanced = enhanced - np.mean(enhanced)
    
    # SI-SDR calculation
    s_target = np.sum(clean * enhanced) / (np.sum(clean ** 2) + 1e-8) * clean
    e_noise = enhanced - s_target
    
    si_sdr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8))
    
    return si_sdr


def evaluate_batch(
    clean_batch: torch.Tensor,
    enhanced_batch: torch.Tensor,
    sample_rate: int = 16000,
    compute_pesq: bool = True,
    compute_stoi: bool = True
) -> Dict[str, float]:
    """
    Evaluate a batch of enhanced audio
    
    Args:
        clean_batch: Batch of clean audio [batch, samples]
        enhanced_batch: Batch of enhanced audio [batch, samples]
        sample_rate: Sample rate
        compute_pesq: Calculate PESQ (only if PESQ is installed)
        compute_stoi: Calculate STOI
    
    Returns:
        Dictionary of average metrics
    """
    if isinstance(clean_batch, torch.Tensor):
        clean_batch = clean_batch.cpu().numpy()
    if isinstance(enhanced_batch, torch.Tensor):
        enhanced_batch = enhanced_batch.cpu().numpy()
    
    if clean_batch.ndim == 1:
        clean_batch = clean_batch[np.newaxis, :]
        enhanced_batch = enhanced_batch[np.newaxis, :]
    
    batch_size = clean_batch.shape[0]
    
    metrics = {
        'snr': [],
        'si_sdr': []
    }
    
    # Only compute PESQ if available and requested
    if compute_pesq and PESQ_AVAILABLE:
        metrics['pesq'] = []
    if compute_stoi:
        metrics['stoi'] = []
    
    for i in range(batch_size):
        clean = clean_batch[i]
        enhanced = enhanced_batch[i]
        
        metrics['snr'].append(calculate_snr(clean, enhanced))
        metrics['si_sdr'].append(calculate_si_sdr(clean, enhanced))
        
        if compute_pesq and PESQ_AVAILABLE:
            metrics['pesq'].append(calculate_pesq(clean, enhanced, sample_rate))
        if compute_stoi:
            metrics['stoi'].append(calculate_stoi(clean, enhanced, sample_rate))
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return avg_metrics


def is_pesq_available() -> bool:
    """Check if PESQ is available"""
    return PESQ_AVAILABLE
