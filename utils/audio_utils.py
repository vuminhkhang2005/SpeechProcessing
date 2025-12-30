"""
Audio processing utilities for speech denoising
Compatible with Google Colab (no torchaudio dependency)

Cải tiến:
1. Global normalization support (mean=0, std=1)
2. Proper denormalization để khôi phục amplitude
3. Post-processing để tránh volume reduction
4. Amplitude matching với reference signal
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import librosa
import soundfile as sf
import json
from pathlib import Path


class GlobalNormalizer:
    """
    Global normalizer để chuẩn hóa audio theo mean và std của toàn bộ training set.
    
    Đây là approach đúng đắn thay vì per-file peak normalization.
    Reference: LeCun - "Efficient BackProp"
    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def normalize(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalize audio"""
        return (audio - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, audio: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Denormalize audio để khôi phục về giá trị gốc"""
        return audio * self.std + self.mean
    
    def save(self, filepath: str) -> None:
        """Lưu statistics"""
        with open(filepath, 'w') as f:
            json.dump({'mean': self.mean, 'std': self.std}, f)
    
    def load(self, filepath: str) -> 'GlobalNormalizer':
        """Load statistics đã lưu"""
        with open(filepath, 'r') as f:
            stats = json.load(f)
        self.mean = stats['mean']
        self.std = stats['std']
        return self
    
    @classmethod
    def from_file(cls, filepath: str) -> 'GlobalNormalizer':
        """Tạo normalizer từ file"""
        normalizer = cls()
        return normalizer.load(filepath)


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
    target_peak: float = 0.95
):
    """
    Save audio to file với proper amplitude handling.
    
    Cải tiến:
    - Không sử dụng peak normalization mặc định (có thể gây inconsistent)
    - Option để normalize với target_peak cụ thể
    - Clip to [-1, 1] để tránh distortion
    
    Args:
        filepath: Output path
        waveform: Audio tensor
        sample_rate: Sample rate
        normalize: Whether to normalize output
        target_peak: Target peak level (default 0.95 để tránh clipping)
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Flatten if needed
    if waveform.ndim > 1:
        waveform = waveform.flatten()
    
    current_peak = np.abs(waveform).max()
    
    if normalize and current_peak > 0:
        # Scale to target peak
        waveform = waveform * (target_peak / current_peak)
    elif current_peak > 1.0:
        # Just clip if not normalizing but exceeds range
        waveform = np.clip(waveform, -1.0, 1.0)
    
    sf.write(filepath, waveform, sample_rate)


def save_audio_with_reference(
    filepath: str,
    waveform: torch.Tensor,
    reference: torch.Tensor,
    sample_rate: int = 16000
):
    """
    Save audio với amplitude matching từ reference signal.
    
    QUAN TRỌNG: Điều này giúp output có cùng loudness với input/reference,
    tránh vấn đề "âm lượng giảm" sau khi khử nhiễu.
    
    Args:
        filepath: Output path
        waveform: Audio tensor (enhanced/denoised)
        reference: Reference audio tensor (original noisy or clean)
        sample_rate: Sample rate
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    
    # Flatten
    if waveform.ndim > 1:
        waveform = waveform.flatten()
    if reference.ndim > 1:
        reference = reference.flatten()
    
    # Match RMS của reference
    waveform_rms = np.sqrt(np.mean(waveform ** 2) + 1e-8)
    reference_rms = np.sqrt(np.mean(reference ** 2) + 1e-8)
    
    if waveform_rms > 0:
        gain = reference_rms / waveform_rms
        waveform = waveform * gain
    
    # Soft clip để tránh harsh distortion
    waveform = np.tanh(waveform)  # Soft clipping
    
    # Scale back nếu cần
    current_peak = np.abs(waveform).max()
    if current_peak > 0.95:
        waveform = waveform * (0.95 / current_peak)
    
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


def match_amplitude(
    waveform: Union[torch.Tensor, np.ndarray],
    reference: Union[torch.Tensor, np.ndarray],
    method: str = 'rms'
) -> Union[torch.Tensor, np.ndarray]:
    """
    Match amplitude của waveform với reference signal.
    
    ĐÂY LÀ FUNCTION QUAN TRỌNG để tránh vấn đề "âm lượng giảm" sau khử nhiễu!
    
    Args:
        waveform: Audio cần điều chỉnh amplitude
        reference: Reference audio để match amplitude
        method: 'rms' hoặc 'peak'
            - 'rms': Match RMS energy (khuyến nghị cho speech)
            - 'peak': Match peak amplitude
    
    Returns:
        Waveform với amplitude đã được match
    """
    is_tensor = isinstance(waveform, torch.Tensor)
    
    if is_tensor:
        waveform_np = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()
        reference_np = reference.cpu().numpy() if reference.is_cuda else reference.numpy()
    else:
        waveform_np = waveform
        reference_np = reference
    
    if method == 'rms':
        waveform_rms = np.sqrt(np.mean(waveform_np ** 2) + 1e-8)
        reference_rms = np.sqrt(np.mean(reference_np ** 2) + 1e-8)
        gain = reference_rms / waveform_rms
    elif method == 'peak':
        waveform_peak = np.abs(waveform_np).max() + 1e-8
        reference_peak = np.abs(reference_np).max() + 1e-8
        gain = reference_peak / waveform_peak
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rms' or 'peak'.")
    
    result = waveform_np * gain
    
    if is_tensor:
        return torch.from_numpy(result).to(waveform.device)
    return result


def loudness_normalize(
    waveform: Union[torch.Tensor, np.ndarray],
    target_lufs: float = -23.0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize loudness theo LUFS (Loudness Units Full Scale).
    
    LUFS là chuẩn broadcast và được sử dụng rộng rãi để đảm bảo
    consistent loudness giữa các audio files.
    
    Args:
        waveform: Input audio
        target_lufs: Target loudness in LUFS (default -23 LUFS - EBU R128)
    
    Returns:
        Loudness normalized waveform
    """
    is_tensor = isinstance(waveform, torch.Tensor)
    
    if is_tensor:
        audio = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()
    else:
        audio = waveform
    
    # Simple LUFS approximation using RMS
    # For accurate LUFS, use pyloudnorm library
    rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    current_lufs = 20 * np.log10(rms + 1e-8)
    
    # Calculate gain
    gain_db = target_lufs - current_lufs
    gain = 10 ** (gain_db / 20)
    
    result = audio * gain
    
    # Prevent clipping
    peak = np.abs(result).max()
    if peak > 0.99:
        result = result * (0.99 / peak)
    
    if is_tensor:
        return torch.from_numpy(result).to(waveform.device)
    return result


def post_process_denoised(
    denoised: Union[torch.Tensor, np.ndarray],
    noisy_input: Union[torch.Tensor, np.ndarray],
    clean_target: Optional[Union[torch.Tensor, np.ndarray]] = None,
    normalizer: Optional[GlobalNormalizer] = None,
    match_input_loudness: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Post-process audio sau khi khử nhiễu.
    
    QUAN TRỌNG: Đây là bước cần thiết để:
    1. Denormalize nếu đã sử dụng global normalization trong training
    2. Match amplitude với input để tránh volume reduction
    3. Đảm bảo output trong range hợp lệ
    
    Args:
        denoised: Denoised audio từ model
        noisy_input: Original noisy input (để match amplitude)
        clean_target: Optional clean target (để reference)
        normalizer: GlobalNormalizer đã dùng trong training
        match_input_loudness: Match loudness với input
    
    Returns:
        Post-processed audio
    """
    is_tensor = isinstance(denoised, torch.Tensor)
    
    # Step 1: Denormalize nếu có normalizer
    if normalizer is not None:
        denoised = normalizer.denormalize(denoised)
        if noisy_input is not None:
            noisy_input = normalizer.denormalize(noisy_input)
    
    # Step 2: Match amplitude với input
    if match_input_loudness and noisy_input is not None:
        denoised = match_amplitude(denoised, noisy_input, method='rms')
    
    # Step 3: Soft clipping để tránh harsh distortion
    if is_tensor:
        # Soft clip using tanh for values outside [-1, 1]
        mask = torch.abs(denoised) > 1.0
        if mask.any():
            denoised = torch.where(mask, torch.tanh(denoised), denoised)
    else:
        mask = np.abs(denoised) > 1.0
        if mask.any():
            denoised = np.where(mask, np.tanh(denoised), denoised)
    
    # Step 4: Final scale to safe range
    if is_tensor:
        peak = torch.abs(denoised).max()
        if peak > 0.95:
            denoised = denoised * (0.95 / peak)
    else:
        peak = np.abs(denoised).max()
        if peak > 0.95:
            denoised = denoised * (0.95 / peak)
    
    return denoised
