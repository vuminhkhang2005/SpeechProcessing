"""
Audio processing utilities for speech denoising
Compatible with Google Colab (no torchaudio dependency)

CẢI TIẾN QUAN TRỌNG:
1. Global normalization thay vì per-file peak normalization
2. Proper denormalization để khôi phục amplitude gốc
3. Phase reconstruction cho STFT-based models
4. Anti-clipping processing
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
import librosa
import soundfile as sf
import json
from pathlib import Path


class GlobalNormalizer:
    """
    Global Audio Normalizer
    
    QUAN TRỌNG: Thay vì normalize từng file riêng lẻ (peak normalization),
    chúng ta sử dụng statistics từ toàn bộ training set.
    
    Lý do:
    1. Peak normalization làm file yên tĩnh bị khuếch đại không hợp lý
    2. Model học không ổn định khi mỗi file có scale khác nhau
    3. Output amplitude sẽ không nhất quán
    
    Reference: LeCun et al. - Standardization is crucial for training stability
    """
    
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0
    ):
        self.mean = mean
        self.std = std
        self._is_fitted = False
    
    def fit_from_files(
        self, 
        audio_files: list, 
        sample_rate: int = 16000,
        max_samples: int = 1000
    ) -> None:
        """Tính global statistics từ list audio files"""
        import random
        
        if len(audio_files) > max_samples:
            audio_files = random.sample(audio_files, max_samples)
        
        all_audio = []
        for f in audio_files:
            try:
                audio, _ = librosa.load(f, sr=sample_rate, mono=True)
                all_audio.append(audio)
            except Exception:
                pass
        
        if all_audio:
            concatenated = np.concatenate(all_audio)
            self.mean = float(np.mean(concatenated))
            self.std = float(np.std(concatenated))
            if self.std < 1e-6:
                self.std = 1.0
            self._is_fitted = True
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize về mean=0, std=1"""
        return (audio - self.mean) / self.std
    
    def denormalize(self, audio: np.ndarray) -> np.ndarray:
        """Khôi phục về scale gốc"""
        return audio * self.std + self.mean
    
    def save(self, path: str) -> None:
        """Lưu statistics"""
        with open(path, 'w') as f:
            json.dump({'mean': self.mean, 'std': self.std}, f)
    
    def load(self, path: str) -> None:
        """Load statistics"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.mean = data['mean']
        self.std = data['std']
        self._is_fitted = True
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class AudioProcessor:
    """
    Xử lý âm thanh: STFT, iSTFT, và các hàm tiện ích
    
    CẢI TIẾN:
    - Hỗ trợ global normalization
    - Proper phase handling cho STFT
    - Output amplitude matching
    """
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        sample_rate: int = 16000,
        normalizer: Optional[GlobalNormalizer] = None
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        
        # Hann window
        self.window = torch.hann_window(win_length)
        
        # Global normalizer
        self.normalizer = normalizer
    
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
    sample_rate: int = 16000
):
    """
    Save audio to file
    
    Args:
        filepath: Output path
        waveform: Audio tensor
        sample_rate: Sample rate
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Normalize to prevent clipping
    if np.abs(waveform).max() > 1.0:
        waveform = waveform / np.abs(waveform).max()
    
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
    enhanced: np.ndarray,
    reference: np.ndarray,
    method: str = 'rms'
) -> np.ndarray:
    """
    Match amplitude của enhanced audio với reference
    
    QUAN TRỌNG: Sau khi denoise, amplitude có thể thay đổi.
    Hàm này đảm bảo output có amplitude tương tự input/reference.
    
    Args:
        enhanced: Enhanced audio (output từ model)
        reference: Reference audio (input hoặc target)
        method: 'rms' (root mean square) hoặc 'peak'
    
    Returns:
        Enhanced audio với amplitude matched
    """
    if method == 'rms':
        # RMS matching (recommended - more robust)
        enhanced_rms = np.sqrt(np.mean(enhanced ** 2) + 1e-8)
        reference_rms = np.sqrt(np.mean(reference ** 2) + 1e-8)
        scale = reference_rms / enhanced_rms
    elif method == 'peak':
        # Peak matching
        enhanced_peak = np.max(np.abs(enhanced)) + 1e-8
        reference_peak = np.max(np.abs(reference)) + 1e-8
        scale = reference_peak / enhanced_peak
    else:
        scale = 1.0
    
    return enhanced * scale


def prevent_clipping(
    audio: np.ndarray,
    threshold: float = 0.99
) -> np.ndarray:
    """
    Ngăn clipping bằng cách scale xuống nếu amplitude vượt ngưỡng
    
    Args:
        audio: Audio waveform
        threshold: Maximum allowed amplitude (default 0.99)
    
    Returns:
        Audio với amplitude trong giới hạn an toàn
    """
    max_amp = np.max(np.abs(audio))
    if max_amp > threshold:
        audio = audio * (threshold / max_amp)
    return audio


def reconstruct_with_phase(
    magnitude: np.ndarray,
    phase: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512
) -> np.ndarray:
    """
    Reconstruct waveform từ magnitude và phase
    
    QUAN TRỌNG: Nếu model chỉ predict magnitude, cần dùng phase từ input
    để reconstruct waveform. Bỏ qua bước này sẽ gây méo tiếng nghiêm trọng.
    
    Args:
        magnitude: Predicted magnitude spectrogram [freq, time]
        phase: Phase from input signal [freq, time]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
    
    Returns:
        Reconstructed waveform
    """
    # Combine magnitude và phase
    stft_complex = magnitude * np.exp(1j * phase)
    
    # iSTFT
    waveform = librosa.istft(
        stft_complex,
        hop_length=hop_length,
        win_length=win_length,
        center=True
    )
    
    return waveform


def griffin_lim_reconstruct(
    magnitude: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    n_iter: int = 32
) -> np.ndarray:
    """
    Griffin-Lim algorithm để reconstruct phase từ magnitude
    
    Sử dụng khi không có phase từ input (e.g., TTS applications)
    
    Args:
        magnitude: Magnitude spectrogram
        n_fft: FFT size
        hop_length: Hop length  
        win_length: Window length
        n_iter: Number of iterations
    
    Returns:
        Reconstructed waveform
    """
    return librosa.griffinlim(
        magnitude,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length
    )


def compute_signal_stats(audio: np.ndarray) -> Dict[str, float]:
    """
    Tính các thống kê của tín hiệu audio
    
    Hữu ích để debug và kiểm tra chất lượng output
    """
    return {
        'mean': float(np.mean(audio)),
        'std': float(np.std(audio)),
        'min': float(np.min(audio)),
        'max': float(np.max(audio)),
        'peak': float(np.max(np.abs(audio))),
        'rms': float(np.sqrt(np.mean(audio ** 2))),
        'dynamic_range_db': float(20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-8) + 1e-8))
    }


def ensure_proper_length(
    audio: np.ndarray,
    target_length: int,
    mode: str = 'pad'
) -> np.ndarray:
    """
    Đảm bảo audio có độ dài đúng
    
    Args:
        audio: Input audio
        target_length: Target length
        mode: 'pad' (zero-padding) hoặc 'trim' (cắt)
    
    Returns:
        Audio với độ dài target_length
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
    elif current_length < target_length:
        # Pad
        pad_length = target_length - current_length
        return np.pad(audio, (0, pad_length), mode='constant')
    else:
        # Trim
        return audio[:target_length]


class OutputProcessor:
    """
    Xử lý output từ model để đảm bảo chất lượng âm thanh
    
    Bao gồm:
    1. Denormalization (khôi phục scale gốc)
    2. Amplitude matching (match với input)
    3. Anti-clipping
    4. Format conversion (float -> int16 WAV)
    """
    
    def __init__(
        self,
        normalizer: Optional[GlobalNormalizer] = None,
        match_amplitude_method: str = 'rms',
        prevent_clipping: bool = True,
        clip_threshold: float = 0.99
    ):
        self.normalizer = normalizer
        self.match_amplitude_method = match_amplitude_method
        self._prevent_clipping = prevent_clipping
        self.clip_threshold = clip_threshold
    
    def process(
        self,
        enhanced: np.ndarray,
        input_audio: Optional[np.ndarray] = None,
        denormalize: bool = True
    ) -> np.ndarray:
        """
        Process output từ model
        
        Args:
            enhanced: Output từ model
            input_audio: Input audio gốc (để match amplitude)
            denormalize: Có denormalize hay không
        
        Returns:
            Processed audio sẵn sàng để save
        """
        output = enhanced.copy()
        
        # 1. Denormalize
        if denormalize and self.normalizer is not None:
            output = self.normalizer.denormalize(output)
        
        # 2. Match amplitude với input
        if input_audio is not None:
            # Nếu input đã được normalize, denormalize trước
            if denormalize and self.normalizer is not None:
                input_denorm = self.normalizer.denormalize(input_audio)
            else:
                input_denorm = input_audio
            
            output = match_amplitude(output, input_denorm, self.match_amplitude_method)
        
        # 3. Prevent clipping
        if self._prevent_clipping:
            output = prevent_clipping(output, self.clip_threshold)
        
        return output
    
    def save(
        self,
        filepath: str,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> None:
        """Save processed audio"""
        # Final clipping check
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        sf.write(filepath, audio, sample_rate)
