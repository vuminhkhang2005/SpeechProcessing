"""
Audio processing utilities for speech denoising
Compatible with Google Colab (no torchaudio dependency)
"""

import torch
import numpy as np
from typing import Tuple, Optional
import librosa
import soundfile as sf


class AudioProcessor:
    """
    Xử lý âm thanh: STFT, iSTFT, và các hàm tiện ích
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
    normalize: bool = True,
    target_db: Optional[float] = None,
    prevent_clipping: bool = True,
    reference_waveform: Optional[np.ndarray] = None
):
    """
    Save audio to file với xử lý amplitude tốt hơn
    
    CẢI TIẾN:
    - Có thể normalize theo target dB level
    - Có thể match amplitude với reference (input signal)
    - Tránh clipping một cách thông minh
    
    Args:
        filepath: Output path
        waveform: Audio tensor
        sample_rate: Sample rate
        normalize: Whether to normalize audio
        target_db: Target dB level (e.g., -25 dB). If None, uses peak normalization
        prevent_clipping: Prevent clipping by limiting to [-1, 1]
        reference_waveform: Reference waveform to match RMS level (e.g., input signal)
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    if normalize:
        if reference_waveform is not None:
            # Match RMS level với reference signal
            # Đây là cách tốt nhất để giữ volume consistent
            waveform = match_rms_level(waveform, reference_waveform)
        elif target_db is not None:
            # Normalize theo target dB level
            waveform = normalize_to_db(waveform, target_db)
        else:
            # Peak normalize to prevent clipping
            peak = np.abs(waveform).max()
            if peak > 1.0:
                waveform = waveform / peak * 0.95  # Leave some headroom
    
    if prevent_clipping:
        # Soft clipping để tránh harsh distortion
        waveform = soft_clip(waveform, threshold=0.95)
    
    sf.write(filepath, waveform, sample_rate)


def match_rms_level(
    audio: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Match RMS level của audio với reference
    
    QUAN TRỌNG cho inference: Đảm bảo output có volume tương tự input
    
    Args:
        audio: Audio to adjust
        reference: Reference audio
        eps: Small constant to prevent division by zero
    
    Returns:
        Audio with matched RMS level
    """
    audio_rms = np.sqrt(np.mean(audio ** 2) + eps)
    ref_rms = np.sqrt(np.mean(reference ** 2) + eps)
    
    # Scale audio to match reference RMS
    gain = ref_rms / audio_rms
    
    return audio * gain


def normalize_to_db(
    audio: np.ndarray,
    target_db: float = -25.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize audio to target dB level (RMS-based)
    
    Args:
        audio: Audio array
        target_db: Target dB level
        eps: Small constant to prevent log of zero
    
    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    current_db = 20 * np.log10(rms + eps)
    gain = 10 ** ((target_db - current_db) / 20)
    
    return audio * gain


def soft_clip(
    audio: np.ndarray,
    threshold: float = 0.95
) -> np.ndarray:
    """
    Soft clipping để tránh harsh distortion
    
    Sử dụng tanh-based soft clipping thay vì hard clipping
    
    Args:
        audio: Audio array
        threshold: Threshold where soft clipping starts
    
    Returns:
        Soft-clipped audio
    """
    peak = np.abs(audio).max()
    
    if peak <= threshold:
        return audio
    
    # Apply tanh soft clipping for values above threshold
    # This creates a smoother transition than hard clipping
    scale = threshold / np.tanh(1.0)
    clipped = np.where(
        np.abs(audio) > threshold,
        np.sign(audio) * scale * np.tanh(np.abs(audio) / scale),
        audio
    )
    
    return clipped


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
