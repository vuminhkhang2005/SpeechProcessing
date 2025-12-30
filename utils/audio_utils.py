"""
Audio processing utilities for speech denoising
Compatible with Google Colab (no torchaudio dependency)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
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
    match_rms_to: Optional[Union[torch.Tensor, np.ndarray]] = None,
    target_db: Optional[float] = None,
    prevent_clipping: bool = True,
):
    """
    Save audio to file
    
    Args:
        filepath: Output path
        waveform: Audio tensor
        sample_rate: Sample rate
    """
    def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _rms(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        return float(np.sqrt(np.mean(x * x) + 1e-8))

    wav = _to_numpy(waveform).astype(np.float32, copy=False)

    # Optional loudness handling:
    # - match_rms_to: scale output to have the same RMS as reference (often the noisy input)
    # - target_db: scale output to a fixed RMS level (dBFS) for convenient listening
    if match_rms_to is not None:
        ref = _to_numpy(match_rms_to).astype(np.float32, copy=False)
        ref_rms = _rms(ref)
        out_rms = _rms(wav)
        if out_rms > 0:
            wav = wav * (ref_rms / out_rms)
    elif target_db is not None:
        out_rms = _rms(wav)
        target_rms = float(10 ** (target_db / 20))
        if out_rms > 0:
            wav = wav * (target_rms / out_rms)

    # Prevent clipping, but do NOT auto-amplify quiet files.
    if prevent_clipping:
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak

    sf.write(filepath, wav, sample_rate)


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
