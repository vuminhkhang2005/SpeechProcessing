"""
U-Net based Speech Denoising Model

Kiến trúc U-Net được sử dụng rộng rãi trong speech enhancement vì:
1. Skip connections giúp bảo toàn thông tin chi tiết
2. Encoder-decoder structure cho phép học features ở nhiều scale
3. Hoạt động tốt trên spectrogram domain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """
    Convolutional block với BatchNorm và activation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        use_bn: bool = True,
        activation: str = 'leaky_relu'
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'prelu':
            layers.append(nn.PReLU(out_channels))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block: Conv -> Conv -> Downsample
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, dropout=dropout
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size,
            padding=kernel_size // 2, dropout=dropout
        )
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (downsampled, skip_connection)"""
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample -> Concat skip -> Conv -> Conv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        
        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, dropout=dropout
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size,
            padding=kernel_size // 2, dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.shape
        
        # Query, Key, Value
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)
        
        # Attention weights
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        
        return self.gamma * out + x


class UNetDenoiser(nn.Module):
    """
    U-Net based Speech Denoiser
    
    Input: STFT spectrogram [batch, 2, freq, time] (real + imag)
    Output: Mask hoặc Enhanced STFT [batch, 2, freq, time]
    
    Architecture:
    - Encoder: series of ConvBlocks with downsampling
    - Bottleneck: optional attention
    - Decoder: series of ConvBlocks with upsampling and skip connections
    - Output: Magnitude mask for speech enhancement
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # Real + Imaginary parts of STFT
        out_channels: int = 2,
        encoder_channels: List[int] = [32, 64, 128, 256, 512],
        use_attention: bool = True,
        dropout: float = 0.1,
        mask_type: str = 'CRM'  # 'IRM', 'CRM', or 'direct'
    ):
        """
        Args:
            in_channels: Number of input channels (2 for complex STFT)
            out_channels: Number of output channels
            encoder_channels: List of channel sizes for encoder layers
            use_attention: Use attention in bottleneck
            dropout: Dropout rate
            mask_type: Type of mask ('IRM', 'CRM', 'direct')
        """
        super().__init__()
        
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.init_conv = ConvBlock(in_channels, encoder_channels[0], 3, padding=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                EncoderBlock(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    dropout=dropout
                )
            )
        
        # Bottleneck with attention
        bottleneck_channels = encoder_channels[-1]
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_channels, bottleneck_channels * 2, 3, padding=1),
            ConvBlock(bottleneck_channels * 2, bottleneck_channels, 3, padding=1)
        )
        
        if use_attention:
            self.attention = AttentionBlock(bottleneck_channels)
        else:
            self.attention = None
        
        # Decoder
        self.decoders = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]  # Reverse
        for i in range(len(decoder_channels) - 1):
            self.decoders.append(
                DecoderBlock(
                    decoder_channels[i] * 2,  # Skip connection doubles channels
                    decoder_channels[i + 1],
                    dropout=dropout
                )
            )
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(encoder_channels[0], out_channels, 1)
        )
        
        # Mask activation based on mask type
        if mask_type == 'IRM':
            self.mask_activation = nn.Sigmoid()
        elif mask_type == 'CRM':
            self.mask_activation = nn.Tanh()
        else:
            self.mask_activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input STFT [batch, 2, freq, time]
        
        Returns:
            Enhanced STFT or mask [batch, 2, freq, time]
        """
        # Store input for residual
        input_stft = x
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.attention is not None:
            x = self.attention(x)
        
        # Decoder with skip connections
        skips = skips[::-1]  # Reverse for decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])
        
        # Output
        output = self.output_conv(x)
        
        # Apply mask if applicable
        if self.mask_type == 'IRM':
            # Ideal Ratio Mask - applied to magnitude only
            mask = self.mask_activation(output)
            # Apply mask to both real and imaginary parts
            output = input_stft * mask
        elif self.mask_type == 'CRM':
            # Complex Ratio Mask
            mask = self.mask_activation(output)
            output = input_stft * mask
        # else: direct output
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetDenoiserLite(nn.Module):
    """
    Lightweight version of UNet for faster inference
    Suitable for real-time applications
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        depth: int = 4
    ):
        super().__init__()
        
        # Build encoder
        self.encoders = nn.ModuleList()
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(depth)]
        
        for i in range(depth):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(0.2)
            ))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.2)
        )
        
        # Build decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = channels[i+1] * 2 if i < depth - 1 else channels[i+1]
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, channels[i] if i > 0 else base_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i] if i > 0 else base_channels),
                nn.LeakyReLU(0.2)
            ))
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_stft = x
        
        # Encoder
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skips = skips[::-1]
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                x = torch.cat([x, skips[i]], dim=1)
            x = decoder(x)
            # Handle size mismatch
            if i < len(skips) - 1 and x.shape[2:] != skips[i+1].shape[2:]:
                x = F.interpolate(x, size=skips[i+1].shape[2:], mode='bilinear', align_corners=True)
        
        # Output mask
        mask = self.output(x)
        
        # Handle final size mismatch with input
        if mask.shape[2:] != input_stft.shape[2:]:
            mask = F.interpolate(mask, size=input_stft.shape[2:], mode='bilinear', align_corners=True)
        
        return input_stft * mask


if __name__ == '__main__':
    # Test model
    model = UNetDenoiser()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 2, 257, 251)  # [batch, channels, freq, time]
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
