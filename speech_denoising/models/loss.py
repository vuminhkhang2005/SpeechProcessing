"""
Loss functions for speech denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class STFTLoss(nn.Module):
    """
    STFT-based loss function
    Combines spectral convergence and magnitude loss
    """
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))
    
    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude"""
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        return torch.abs(stft)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple:
        """
        Args:
            pred: Predicted waveform [batch, samples]
            target: Target waveform [batch, samples]
        
        Returns:
            spectral_convergence_loss, magnitude_loss
        """
        pred_mag = self._stft(pred)
        target_mag = self._stft(target)
        
        # Spectral convergence loss
        sc_loss = torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
        
        # Log magnitude loss
        mag_loss = F.l1_loss(torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8))
        
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss
    Uses multiple STFT configurations for better spectral coverage
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [50, 120, 240],
        win_sizes: List[int] = [240, 600, 1200],
        sc_weight: float = 1.0,
        mag_weight: float = 1.0
    ):
        super().__init__()
        
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
        
        self.stft_losses = nn.ModuleList()
        for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses.append(STFTLoss(n_fft, hop, win))
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform [batch, samples]
            target: Target waveform [batch, samples]
        
        Returns:
            Combined multi-resolution STFT loss
        """
        sc_loss = 0.0
        mag_loss = 0.0
        
        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(pred, target)
            sc_loss += sc
            mag_loss += mag
        
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        
        return self.sc_weight * sc_loss + self.mag_weight * mag_loss


class ComplexMSELoss(nn.Module):
    """
    MSE loss on complex STFT representation
    """
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted STFT [batch, 2, freq, time]
            target: Target STFT [batch, 2, freq, time]
        """
        return F.mse_loss(pred, target)


class ComplexL1Loss(nn.Module):
    """
    L1 loss on complex STFT representation
    """
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted STFT [batch, 2, freq, time]
            target: Target STFT [batch, 2, freq, time]
        """
        return F.l1_loss(pred, target)


class MagnitudeLoss(nn.Module):
    """
    Loss on magnitude spectrum only
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def _get_magnitude(self, stft: torch.Tensor) -> torch.Tensor:
        """Get magnitude from complex STFT [batch, 2, freq, time]"""
        return torch.sqrt(stft[:, 0] ** 2 + stft[:, 1] ** 2 + 1e-8)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred_mag = self._get_magnitude(pred)
        target_mag = self._get_magnitude(target)
        
        if self.loss_type == 'l1':
            return F.l1_loss(pred_mag, target_mag)
        else:
            return F.mse_loss(pred_mag, target_mag)


class PhaseLoss(nn.Module):
    """
    Loss on phase spectrum
    Uses instantaneous frequency representation for stability
    """
    
    def _get_phase(self, stft: torch.Tensor) -> torch.Tensor:
        """Get phase from complex STFT [batch, 2, freq, time]"""
        return torch.atan2(stft[:, 1], stft[:, 0])
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred_phase = self._get_phase(pred)
        target_phase = self._get_phase(target)
        
        # Phase difference (wrapped)
        phase_diff = pred_phase - target_phase
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        return torch.mean(torch.abs(phase_diff))


class DenoiserLoss(nn.Module):
    """
    Combined loss function for speech denoising
    
    Combines:
    1. Complex L1 loss on STFT
    2. Magnitude L1 loss
    3. Multi-resolution STFT loss (optional, on waveform)
    """
    
    def __init__(
        self,
        complex_weight: float = 1.0,
        magnitude_weight: float = 1.0,
        stft_weight: float = 0.5,
        use_mr_stft: bool = True,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512
    ):
        super().__init__()
        
        self.complex_weight = complex_weight
        self.magnitude_weight = magnitude_weight
        self.stft_weight = stft_weight
        self.use_mr_stft = use_mr_stft
        
        self.complex_loss = ComplexL1Loss()
        self.magnitude_loss = MagnitudeLoss('l1')
        
        if use_mr_stft:
            self.mr_stft_loss = MultiResolutionSTFTLoss()
        
        # STFT parameters for waveform reconstruction
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))
    
    def _istft(self, stft: torch.Tensor) -> torch.Tensor:
        """Convert STFT to waveform"""
        # stft: [batch, 2, freq, time]
        stft_complex = torch.complex(stft[:, 0], stft[:, 1])
        
        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=False
        )
        
        return waveform
    
    def forward(
        self,
        pred_stft: torch.Tensor,
        target_stft: torch.Tensor,
        pred_waveform: Optional[torch.Tensor] = None,
        target_waveform: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            pred_stft: Predicted STFT [batch, 2, freq, time]
            target_stft: Target STFT [batch, 2, freq, time]
            pred_waveform: Predicted waveform (optional)
            target_waveform: Target waveform (optional)
        
        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}
        
        # Complex L1 loss
        complex_l = self.complex_loss(pred_stft, target_stft)
        losses['complex_loss'] = complex_l
        
        # Magnitude loss
        mag_l = self.magnitude_loss(pred_stft, target_stft)
        losses['magnitude_loss'] = mag_l
        
        # Total loss
        total = self.complex_weight * complex_l + self.magnitude_weight * mag_l
        
        # Multi-resolution STFT loss on waveform
        if self.use_mr_stft and (pred_waveform is not None and target_waveform is not None):
            mr_stft_l = self.mr_stft_loss(pred_waveform, target_waveform)
            losses['mr_stft_loss'] = mr_stft_l
            total += self.stft_weight * mr_stft_l
        
        losses['total_loss'] = total
        
        return losses


class SDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio loss
    Commonly used in source separation tasks
    """
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform [batch, samples]
            target: Target waveform [batch, samples]
        
        Returns:
            Negative SI-SDR (to minimize)
        """
        # Zero mean
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # SI-SDR
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
        proj = dot * target / s_target_energy
        
        noise = pred - proj
        
        si_sdr = 10 * torch.log10(
            torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8
        )
        
        return -si_sdr.mean()  # Negative because we want to maximize SI-SDR
