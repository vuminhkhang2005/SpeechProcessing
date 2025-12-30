"""
Loss functions for speech denoising

Cải tiến để ngăn "lazy learning" (model chỉ giảm âm lượng thay vì lọc ồn):
1. SI-SDR Loss: Scale-invariant, không quan tâm âm lượng chỉ quan tâm chất lượng
2. Time-domain L1 Loss: Đảm bảo waveform được tái tạo chính xác  
3. Energy Conservation Loss: Phạt nếu model giảm năng lượng quá nhiều
4. Multi-resolution STFT Loss: Cải tiến với nhiều resolution hơn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


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
    Combined loss function for speech denoising - PHIÊN BẢN CẢI TIẾN
    
    Combines (để ngăn lazy learning):
    1. Complex L1 loss on STFT
    2. Magnitude L1 loss  
    3. Multi-resolution STFT loss (on waveform)
    4. SI-SDR loss (QUAN TRỌNG! - scale-invariant)
    5. Time-domain L1 loss
    6. Energy conservation loss (ngăn giảm volume)
    
    Mục đích: Ép model phải thực sự lọc noise, không chỉ giảm âm lượng
    """
    
    def __init__(
        self,
        # STFT domain losses
        complex_weight: float = 1.0,
        magnitude_weight: float = 1.0,
        stft_weight: float = 0.5,
        
        # Time domain losses (QUAN TRỌNG)
        si_sdr_weight: float = 0.5,      # SI-SDR loss weight
        time_l1_weight: float = 0.3,     # Time domain L1
        
        # Regularization losses
        energy_weight: float = 0.1,       # Energy conservation
        
        use_mr_stft: bool = True,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512
    ):
        super().__init__()
        
        # Weights
        self.complex_weight = complex_weight
        self.magnitude_weight = magnitude_weight
        self.stft_weight = stft_weight
        self.si_sdr_weight = si_sdr_weight
        self.time_l1_weight = time_l1_weight
        self.energy_weight = energy_weight
        self.use_mr_stft = use_mr_stft
        
        # STFT domain losses
        self.complex_loss = ComplexL1Loss()
        self.magnitude_loss = MagnitudeLoss('l1')
        
        if use_mr_stft:
            # Cải tiến: Thêm nhiều resolution hơn
            self.mr_stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=[256, 512, 1024, 2048],
                hop_sizes=[64, 128, 256, 512],
                win_sizes=[256, 512, 1024, 2048]
            )
        
        # Time domain losses (QUAN TRỌNG cho anti-lazy-learning)
        self.sdr_loss = SDRLoss(clip_value=30.0)
        self.time_l1_loss = TimeDomainL1Loss()
        self.energy_loss = EnergyConservationLoss(min_ratio=0.6, max_ratio=1.4)
        
        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))
    
    def _istft(self, stft: torch.Tensor) -> torch.Tensor:
        """Convert STFT to waveform"""
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
        target_waveform: Optional[torch.Tensor] = None,
        noisy_waveform: Optional[torch.Tensor] = None  # Thêm để tính noise awareness
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_stft: Predicted STFT [batch, 2, freq, time]
            target_stft: Target STFT [batch, 2, freq, time]
            pred_waveform: Predicted waveform (optional)
            target_waveform: Target/clean waveform (optional)
            noisy_waveform: Original noisy waveform (optional)
        
        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}
        
        # ===== STFT Domain Losses =====
        # Complex L1 loss
        complex_l = self.complex_loss(pred_stft, target_stft)
        losses['complex_loss'] = complex_l
        
        # Magnitude loss
        mag_l = self.magnitude_loss(pred_stft, target_stft)
        losses['magnitude_loss'] = mag_l
        
        # Total starts with STFT domain losses
        total = self.complex_weight * complex_l + self.magnitude_weight * mag_l
        
        # ===== Time Domain Losses (QUAN TRỌNG!) =====
        if pred_waveform is not None and target_waveform is not None:
            # Multi-resolution STFT loss
            if self.use_mr_stft:
                mr_stft_l = self.mr_stft_loss(pred_waveform, target_waveform)
                losses['mr_stft_loss'] = mr_stft_l
                total += self.stft_weight * mr_stft_l
            
            # SI-SDR loss - ĐÂY LÀ KEY ĐỂ NGĂN LAZY LEARNING!
            if self.si_sdr_weight > 0:
                sdr_l = self.sdr_loss(pred_waveform, target_waveform)
                losses['si_sdr_loss'] = sdr_l
                total += self.si_sdr_weight * sdr_l
            
            # Time domain L1 loss
            if self.time_l1_weight > 0:
                time_l1 = self.time_l1_loss(pred_waveform, target_waveform)
                losses['time_l1_loss'] = time_l1
                total += self.time_l1_weight * time_l1
            
            # Energy conservation loss - Ngăn model giảm volume quá nhiều
            if self.energy_weight > 0:
                energy_l = self.energy_loss(pred_waveform, target_waveform)
                losses['energy_loss'] = energy_l
                total += self.energy_weight * energy_l
        
        losses['total_loss'] = total
        
        return losses


class DenoiserLossLegacy(nn.Module):
    """
    Legacy loss function (giữ lại để backward compatibility)
    Dùng DenoiserLoss mới thay thế để có kết quả tốt hơn
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
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))
    
    def forward(
        self,
        pred_stft: torch.Tensor,
        target_stft: torch.Tensor,
        pred_waveform: Optional[torch.Tensor] = None,
        target_waveform: Optional[torch.Tensor] = None
    ) -> dict:
        losses = {}
        
        complex_l = self.complex_loss(pred_stft, target_stft)
        losses['complex_loss'] = complex_l
        
        mag_l = self.magnitude_loss(pred_stft, target_stft)
        losses['magnitude_loss'] = mag_l
        
        total = self.complex_weight * complex_l + self.magnitude_weight * mag_l
        
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
    
    ĐÂY LÀ LOSS QUAN TRỌNG NHẤT để ngăn "lazy learning"!
    - Scale-invariant: Không quan tâm âm lượng, chỉ quan tâm chất lượng
    - Nếu model chỉ giảm volume, SI-SDR sẽ không cải thiện
    """
    
    def __init__(self, clip_value: float = 30.0):
        super().__init__()
        self.clip_value = clip_value  # Clip để tránh gradient explosion
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform [batch, samples] or [batch, 1, samples]
            target: Target waveform [batch, samples] or [batch, 1, samples]
        
        Returns:
            Negative SI-SDR (to minimize)
        """
        # Flatten if needed
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
            
        # Zero mean (remove DC offset)
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # SI-SDR calculation
        # s_target = <pred, target> * target / ||target||^2
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
        proj = dot * target / s_target_energy
        
        # e_noise = pred - s_target
        noise = pred - proj
        
        # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        si_sdr = 10 * torch.log10(
            torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8
        )
        
        # Clip to prevent gradient explosion
        si_sdr = torch.clamp(si_sdr, -self.clip_value, self.clip_value)
        
        return -si_sdr.mean()  # Negative because we want to maximize SI-SDR


class TimeDomainL1Loss(nn.Module):
    """
    L1 loss trên waveform domain
    Giúp đảm bảo waveform được tái tạo chính xác
    """
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        return F.l1_loss(pred, target)


class EnergyConservationLoss(nn.Module):
    """
    Loss để ngăn model giảm năng lượng/amplitude quá nhiều (chống lazy learning)
    
    Vấn đề: Model có thể "lazy learn" bằng cách chỉ giảm volume để giảm loss,
    thay vì học cách loại bỏ noise thực sự.
    
    Giải pháp: Phạt nếu năng lượng output khác quá nhiều so với clean signal
    
    Theo ai.stackexchange.com, đây là một pattern phổ biến trong speech enhancement
    khi model không được train đúng cách.
    """
    
    def __init__(
        self, 
        min_ratio: float = 0.6,  # Cho phép giảm tối đa 40%
        max_ratio: float = 1.4,  # Cho phép tăng tối đa 40%
        use_rms: bool = True     # Dùng RMS thay vì energy tổng
    ):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.use_rms = use_rms
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Phạt nếu tỷ lệ năng lượng pred/target nằm ngoài [min_ratio, max_ratio]
        
        Args:
            pred: Predicted waveform
            target: Target/clean waveform
        
        Returns:
            Energy conservation loss
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        if self.use_rms:
            # RMS-based ratio (more stable)
            pred_rms = torch.sqrt(torch.mean(pred ** 2, dim=-1) + 1e-8)
            target_rms = torch.sqrt(torch.mean(target ** 2, dim=-1) + 1e-8)
            ratio = pred_rms / target_rms
        else:
            # Energy-based ratio
            pred_energy = torch.sum(pred ** 2, dim=-1)
            target_energy = torch.sum(target ** 2, dim=-1) + 1e-8
            ratio = torch.sqrt(pred_energy / target_energy)  # sqrt for amplitude ratio
        
        # Phạt nếu ratio < min_ratio (output quá nhỏ - lazy learning!)
        low_penalty = F.relu(self.min_ratio - ratio) ** 2  # Squared for stronger penalty
        
        # Phạt nếu ratio > max_ratio (output quá lớn - có thể amplify noise)  
        high_penalty = F.relu(ratio - self.max_ratio) ** 2
        
        return (low_penalty + high_penalty).mean()


class NoiseAwarenessLoss(nn.Module):
    """
    Loss để kiểm tra model có thực sự loại bỏ noise hay không
    
    Ý tưởng: So sánh residual (pred - clean) với noise gốc (noisy - clean)
    Residual nên nhỏ hơn noise gốc
    """
    
    def forward(
        self,
        pred: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor
    ) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if clean.dim() == 3:
            clean = clean.squeeze(1)
        if noisy.dim() == 3:
            noisy = noisy.squeeze(1)
            
        # Residual noise trong prediction
        pred_residual = pred - clean
        
        # Noise gốc
        original_noise = noisy - clean
        
        # Năng lượng của residual so với noise gốc
        pred_residual_energy = torch.sum(pred_residual ** 2, dim=-1) + 1e-8
        original_noise_energy = torch.sum(original_noise ** 2, dim=-1) + 1e-8
        
        # Tỷ lệ noise reduction (càng nhỏ càng tốt)
        noise_ratio = pred_residual_energy / original_noise_energy
        
        # Phạt nếu không giảm được noise
        return F.relu(noise_ratio - 0.5).mean()  # Yêu cầu giảm ít nhất 50% noise
