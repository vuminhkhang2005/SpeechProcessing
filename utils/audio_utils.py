"""
Audio processing utilities for speech denoising
Compatible with Google Colab (no torchaudio dependency)

Key improvements:
1. Better output normalization (preserve amplitude, not peak normalize)
2. Energy-based gain matching for output
3. Proper STFT/iSTFT handling to prevent amplitude loss

Reference:
- bioacoustics.stackexchange.com: Don't normalize per file, use global constants
- ijsrem.com: Proper iSTFT reconstruction with phase
"""

import torch
import numpy as np
from typing import Tuple, Optional
import librosa
import soundfile as sf


class AudioProcessor:
    """
    Xử lý âm thanh: STFT, iSTFT, và các hàm tiện ích
    
    Cải tiến:
    - Bảo toàn năng lượng trong quá trình STFT/iSTFT
    - Hỗ trợ gain matching với tín hiệu gốc
    """
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        sample_rate: int = 16000
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        
        # Hann window
        self.window = torch.hann_window(win_length)
    
    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện Short-Time Fourier Transform
        
        Args:
            waveform: Audio waveform [batch, samples] or [samples]
        
        Returns:
            Complex STFT [batch, freq_bins, time_frames, 2] (real, imag)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Move window to same device as waveform
        window = self.window.to(waveform.device)
        
        # Compute STFT
        stft_out = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to real representation [batch, freq, time, 2]
        stft_real = torch.stack([stft_out.real, stft_out.imag], dim=-1)
        
        return stft_real
    
    def istft(self, stft_tensor: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện Inverse STFT
        
        Args:
            stft_tensor: STFT tensor [batch, freq_bins, time_frames, 2]
        
        Returns:
            Waveform [batch, samples]
        """
        window = self.window.to(stft_tensor.device)
        
        # Convert to complex
        stft_complex = torch.complex(stft_tensor[..., 0], stft_tensor[..., 1])
        
        # Compute iSTFT
        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=False
        )
        
        return waveform
    
    def magnitude_phase(self, stft_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính magnitude và phase từ STFT
        
        Args:
            stft_tensor: STFT tensor [batch, freq, time, 2]
        
        Returns:
            magnitude: [batch, freq, time]
            phase: [batch, freq, time]
        """
        real = stft_tensor[..., 0]
        imag = stft_tensor[..., 1]
        
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)
        
        return magnitude, phase
    
    def polar_to_rect(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Chuyển từ polar (magnitude, phase) sang rectangular (real, imag)
        
        Args:
            magnitude: [batch, freq, time]
            phase: [batch, freq, time]
        
        Returns:
            STFT tensor [batch, freq, time, 2]
        """
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        
        return torch.stack([real, imag], dim=-1)


def load_audio(
    filepath: str,
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using librosa (Google Colab compatible)
    
    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
    
    Returns:
        waveform: Audio tensor
        sr: Sample rate
    """
    # Use librosa to load audio (handles resampling automatically)
    waveform, sr = librosa.load(filepath, sr=sample_rate, mono=mono)
    
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform).float()
    
    return waveform, sr


def save_audio(
    filepath: str,
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    normalize_mode: str = 'safe',
    reference_waveform: Optional[torch.Tensor] = None
):
    """
    Save audio to file with proper amplitude handling
    
    IMPORTANT: Avoid per-file peak normalization as it can amplify quiet files
    inappropriately. Instead use energy-based matching or safe clipping.
    
    Args:
        filepath: Output path
        waveform: Audio tensor
        sample_rate: Sample rate
        normalize_mode: 
            - 'none': No normalization (may clip)
            - 'safe': Only scale down if would clip (default, preserves quiet files)
            - 'peak': Peak normalize to [-1, 1] (NOT RECOMMENDED for speech)
            - 'energy': Match energy to reference (requires reference_waveform)
            - 'rms': Normalize to target RMS level
        reference_waveform: Reference for energy matching
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    if isinstance(reference_waveform, torch.Tensor):
        reference_waveform = reference_waveform.cpu().numpy()
    
    if normalize_mode == 'none':
        # No normalization - may clip
        pass
    
    elif normalize_mode == 'safe':
        # Safe mode: Only scale down if max exceeds 1.0
        # This preserves the relative amplitude of quiet signals
        max_val = np.abs(waveform).max()
        if max_val > 0.99:
            waveform = waveform * (0.99 / max_val)
    
    elif normalize_mode == 'peak':
        # Peak normalize (NOT recommended for speech enhancement)
        # This can make quiet denoised output louder than intended
        max_val = np.abs(waveform).max()
        if max_val > 1e-8:
            waveform = waveform / max_val * 0.99
    
    elif normalize_mode == 'energy' and reference_waveform is not None:
        # Energy matching: Match output energy to reference
        # This is the RECOMMENDED method for speech denoising output
        waveform = match_energy(waveform, reference_waveform)
        # Then apply safe clipping
        max_val = np.abs(waveform).max()
        if max_val > 0.99:
            waveform = waveform * (0.99 / max_val)
    
    elif normalize_mode == 'rms':
        # RMS normalization to target level
        waveform = normalize_rms(waveform, target_rms=0.1)
        max_val = np.abs(waveform).max()
        if max_val > 0.99:
            waveform = waveform * (0.99 / max_val)
    
    # Ensure float32 for soundfile
    waveform = waveform.astype(np.float32)
    
    sf.write(filepath, waveform, sample_rate)


def normalize_audio(waveform: torch.Tensor, target_db: float = -25.0) -> torch.Tensor:
    """
    Normalize audio to target dB level
    
    Args:
        waveform: Audio tensor
        target_db: Target dB level
    
    Returns:
        Normalized waveform
    """
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    target_rms = 10 ** (target_db / 20)
    
    return waveform * (target_rms / rms)


def add_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float
) -> torch.Tensor:
    """
    Add noise to clean audio at specified SNR
    
    Args:
        clean: Clean audio
        noise: Noise audio
        snr_db: Target SNR in dB
    
    Returns:
        Noisy audio
    """
    # Calculate power
    clean_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Calculate scaling factor for noise
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(clean_power / (snr_linear * noise_power + 1e-8))
    
    # Scale noise and add to clean
    noisy = clean + scale * noise
    
    return noisy


def match_energy(
    waveform: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Match energy of waveform to reference.
    
    This is the RECOMMENDED way to handle output amplitude in speech denoising.
    It ensures the denoised output has similar loudness to the input/reference.
    
    Args:
        waveform: Audio to scale
        reference: Reference audio
        eps: Small constant to prevent division by zero
    
    Returns:
        Energy-matched waveform
    """
    # Calculate RMS energy
    waveform_rms = np.sqrt(np.mean(waveform ** 2) + eps)
    reference_rms = np.sqrt(np.mean(reference ** 2) + eps)
    
    # Scale to match reference energy
    scale = reference_rms / waveform_rms
    
    # Limit scaling factor to prevent extreme changes
    scale = np.clip(scale, 0.5, 2.0)
    
    return waveform * scale


def normalize_rms(
    waveform: np.ndarray,
    target_rms: float = 0.1,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize audio to target RMS level.
    
    Args:
        waveform: Audio to normalize
        target_rms: Target RMS value
        eps: Small constant to prevent division by zero
    
    Returns:
        Normalized waveform
    """
    current_rms = np.sqrt(np.mean(waveform ** 2) + eps)
    scale = target_rms / current_rms
    return waveform * scale


def compute_energy(waveform: np.ndarray) -> float:
    """
    Compute energy of audio signal.
    
    Args:
        waveform: Audio signal
    
    Returns:
        Energy value
    """
    return np.sum(waveform ** 2)


def compute_rms(waveform: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute RMS of audio signal.
    
    Args:
        waveform: Audio signal
        eps: Small constant to prevent division by zero
    
    Returns:
        RMS value
    """
    return np.sqrt(np.mean(waveform ** 2) + eps)


def apply_gain(
    waveform: np.ndarray,
    gain_db: float
) -> np.ndarray:
    """
    Apply gain in dB to audio.
    
    Args:
        waveform: Audio signal
        gain_db: Gain in dB
    
    Returns:
        Gained audio
    """
    gain_linear = 10 ** (gain_db / 20)
    return waveform * gain_linear


def check_clipping(
    waveform: np.ndarray,
    threshold: float = 0.99
) -> Tuple[bool, float]:
    """
    Check if audio would clip.
    
    Args:
        waveform: Audio signal
        threshold: Clipping threshold
    
    Returns:
        Tuple of (is_clipping, max_value)
    """
    max_val = np.abs(waveform).max()
    return max_val > threshold, max_val


class AmplitudePreserver:
    """
    Helper class to preserve amplitude across denoising process.
    
    Usage:
        preserver = AmplitudePreserver()
        preserver.store_input_stats(noisy_waveform)
        
        # ... perform denoising ...
        
        denoised = preserver.restore_amplitude(denoised_waveform)
    """
    
    def __init__(self, method: str = 'rms'):
        """
        Args:
            method: 'rms', 'peak', or 'energy'
        """
        self.method = method
        self.input_rms = None
        self.input_peak = None
        self.input_energy = None
    
    def store_input_stats(self, waveform: np.ndarray):
        """Store statistics from input waveform"""
        self.input_rms = compute_rms(waveform)
        self.input_peak = np.abs(waveform).max()
        self.input_energy = compute_energy(waveform)
    
    def restore_amplitude(self, waveform: np.ndarray) -> np.ndarray:
        """Restore amplitude to match input"""
        if self.method == 'rms':
            output_rms = compute_rms(waveform)
            if output_rms > 1e-8:
                scale = self.input_rms / output_rms
                # Limit scaling
                scale = np.clip(scale, 0.5, 2.0)
                waveform = waveform * scale
        
        elif self.method == 'peak':
            output_peak = np.abs(waveform).max()
            if output_peak > 1e-8:
                scale = self.input_peak / output_peak
                scale = np.clip(scale, 0.5, 2.0)
                waveform = waveform * scale
        
        elif self.method == 'energy':
            output_energy = compute_energy(waveform)
            if output_energy > 1e-8:
                scale = np.sqrt(self.input_energy / output_energy)
                scale = np.clip(scale, 0.5, 2.0)
                waveform = waveform * scale
        
        return waveform
