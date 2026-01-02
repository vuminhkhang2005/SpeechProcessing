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
from typing import List, Optional, Dict
import re


def _apply_complex_mask_2ch(input_stft: torch.Tensor, mask_2ch: torch.Tensor) -> torch.Tensor:
    """
    Apply a complex mask to a complex STFT represented as 2 channels.

    Both tensors are shaped [B, 2, F, T] where channel 0 is real and channel 1 is imag.
    Complex multiplication:
      (xr + j xi) * (mr + j mi) = (xr*mr - xi*mi) + j(xr*mi + xi*mr)
    """
    xr, xi = input_stft[:, 0], input_stft[:, 1]
    mr, mi = mask_2ch[:, 0], mask_2ch[:, 1]
    yr = xr * mr - xi * mi
    yi = xr * mi + xi * mr
    return torch.stack([yr, yi], dim=1)


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
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Upsample to out_channels
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        
        # After concat: out_channels + skip_channels
        self.conv1 = ConvBlock(
            out_channels + skip_channels, out_channels, kernel_size,
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
        self.bottleneck_out_channels = bottleneck_channels * 2
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_channels, self.bottleneck_out_channels, 3, padding=1),
            ConvBlock(self.bottleneck_out_channels, self.bottleneck_out_channels, 3, padding=1)
        )
        
        if use_attention:
            self.attention = AttentionBlock(self.bottleneck_out_channels)
        else:
            self.attention = None
        
        # Decoder
        self.decoders = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]  # Reverse: [512, 256, 128, 64, 32]
        decoder_in_channels = self.bottleneck_out_channels  # Start with bottleneck output
        
        for i in range(len(decoder_channels) - 1):
            skip_ch = decoder_channels[i]  # Skip connection channels
            dec_out_ch = decoder_channels[i + 1]  # Output channels for this decoder
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_in_channels,
                    skip_channels=skip_ch,
                    out_channels=dec_out_ch,
                    dropout=dropout
                )
            )
            decoder_in_channels = dec_out_ch  # Next decoder input = current output
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(encoder_channels[0], self.out_channels, 1)
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
            # Ensure mask matches input STFT resolution (freq, time)
            if mask.shape[2:] != input_stft.shape[2:]:
                mask = F.interpolate(mask, size=input_stft.shape[2:], mode="bilinear", align_corners=False)
            # IRM is a magnitude (real-valued) gain; use a single mask shared by real/imag.
            if mask.shape[1] == 1:
                mag_mask = mask
            else:
                mag_mask = mask.mean(dim=1, keepdim=True)
            output = input_stft * mag_mask
        elif self.mask_type == 'CRM':
            # Complex Ratio Mask
            mask = self.mask_activation(output)
            # Ensure mask matches input STFT resolution (freq, time)
            if mask.shape[2:] != input_stft.shape[2:]:
                mask = F.interpolate(mask, size=input_stft.shape[2:], mode="bilinear", align_corners=False)
            # IMPORTANT: CRM is complex-valued; must use complex multiplication (not per-channel multiply).
            if mask.shape[1] == 2 and input_stft.shape[1] == 2:
                output = _apply_complex_mask_2ch(input_stft, mask)
            else:
                # Fallback for atypical checkpoints: keep legacy behavior.
                output = input_stft * mask
        # else: direct output
        
        # Safety: always return same spatial size as input (legacy checkpoints / odd lengths)
        if output.shape[2:] != input_stft.shape[2:]:
            output = F.interpolate(output, size=input_stft.shape[2:], mode="bilinear", align_corners=False)
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetDenoiserLegacy(nn.Module):
    """
    Legacy U-Net variant kept for backward-compatible checkpoint loading.

    Key differences vs `UNetDenoiser`:
    - No separate `init_conv`; the first encoder consumes `in_channels` directly.
    - Optional attention in bottleneck (only enabled if checkpoint/config expects it).
    - Output head can be either a simple `output` 1x1 conv (common in older checkpoints)
      or the newer `output_conv` stack.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        encoder_channels: List[int] = [32, 64, 128, 256, 512],
        use_attention: bool = False,
        dropout: float = 0.0,
        mask_type: str = "CRM",
        output_head: str = "output",
    ):
        super().__init__()

        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder: first block consumes raw input channels
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in encoder_channels:
            self.encoders.append(EncoderBlock(prev, ch, dropout=dropout))
            prev = ch

        # Bottleneck
        bottleneck_channels = encoder_channels[-1]
        self.bottleneck_out_channels = bottleneck_channels * 2
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_channels, self.bottleneck_out_channels, 3, padding=1),
            ConvBlock(self.bottleneck_out_channels, self.bottleneck_out_channels, 3, padding=1),
        )

        self.attention = AttentionBlock(self.bottleneck_out_channels) if use_attention else None

        # Decoder
        self.decoders = nn.ModuleList()
        decoder_in_channels = self.bottleneck_out_channels
        for i in range(len(encoder_channels) - 1, 0, -1):
            skip_ch = encoder_channels[i]      # skip at same level
            out_ch = encoder_channels[i - 1]   # decode to previous level channels
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_in_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout=dropout,
                )
            )
            decoder_in_channels = out_ch

        # Output head
        self.output_head = output_head
        if output_head == "output_conv":
            self.output_conv = nn.Sequential(
                nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(encoder_channels[0], self.out_channels, 1),
            )
            self.output = None
        else:
            # Common legacy head: a single 1x1 conv named `output`.
            self.output = nn.Conv2d(encoder_channels[0], self.out_channels, 1)
            self.output_conv = None

        # Mask activation based on mask type
        if mask_type == "IRM":
            self.mask_activation = nn.Sigmoid()
        elif mask_type == "CRM":
            self.mask_activation = nn.Tanh()
        else:
            self.mask_activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_stft = x

        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)
        if self.attention is not None:
            x = self.attention(x)

        skips = skips[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])

        if self.output_conv is not None:
            output = self.output_conv(x)
        else:
            output = self.output(x)

        if self.mask_type in {"IRM", "CRM"}:
            mask = self.mask_activation(output)  # type: ignore[misc]
            # Legacy layouts can end up with smaller (freq,time); always match input STFT.
            if mask.shape[2:] != input_stft.shape[2:]:
                mask = F.interpolate(mask, size=input_stft.shape[2:], mode="bilinear", align_corners=False)
            if self.mask_type == "IRM":
                mag_mask = mask if mask.shape[1] == 1 else mask.mean(dim=1, keepdim=True)
                output = input_stft * mag_mask
            else:
                if mask.shape[1] == 2 and input_stft.shape[1] == 2:
                    output = _apply_complex_mask_2ch(input_stft, mask)
                else:
                    output = input_stft * mask
        else:
            # direct output: ensure shape match
            if output.shape[2:] != input_stft.shape[2:]:
                output = F.interpolate(output, size=input_stft.shape[2:], mode="bilinear", align_corners=False)

        return output


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

        # Treat 2-channel output as a complex mask (CRM-style).
        if mask.shape[1] == 2 and input_stft.shape[1] == 2:
            return _apply_complex_mask_2ch(input_stft, mask)
        return input_stft * mask


def convert_old_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert old checkpoint format to new format.
    
    Changes:
    - .conv -> .block (in ConvBlock)
    - Handle different encoder channel configurations
    
    Args:
        state_dict: Old state dict
    
    Returns:
        Converted state dict compatible with new model
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert .conv to .block in ConvBlock
        # Pattern: xxx.conv1.conv.0.weight -> xxx.conv1.block.0.weight
        # Pattern: xxx.conv2.conv.0.weight -> xxx.conv2.block.0.weight
        new_key = re.sub(r'\.conv(\d?)\.conv\.', r'.conv\1.block.', new_key)
        
        # Convert init_conv pattern if needed
        # init_conv.conv.0.weight -> init_conv.block.0.weight (if old format)
        if new_key.startswith('init_conv.conv.'):
            new_key = new_key.replace('init_conv.conv.', 'init_conv.block.')
        
        # Convert output_conv patterns if needed
        new_key = new_key.replace('.output_conv.conv.', '.output_conv.')

        # Common older naming: `output.weight`/`output.bias` for the 1x1 head
        # (newer code typically uses `output_conv.2.*`).
        if new_key.startswith("output.") and not new_key.startswith("output_conv."):
            # Keep as-is; loader may choose a legacy model with `output` head.
            pass
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def detect_encoder_channels_from_checkpoint(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    """
    Detect encoder channel configuration from checkpoint.
    
    Args:
        state_dict: Checkpoint state dict
    
    Returns:
        List of encoder channels
    """
    encoder_indices = set()
    encoder_channels = {}
    
    for key in state_dict.keys():
        # Look for encoder patterns
        match = re.match(r'encoders\.(\d+)\.conv1\.(conv|block)\.0\.weight', key)
        if match:
            idx = int(match.group(1))
            encoder_indices.add(idx)
            # Get output channels from first conv weight shape
            encoder_channels[idx] = state_dict[key].shape[0]
    
    if not encoder_indices:
        # Fallback to default
        return [32, 64, 128, 256, 512]
    
    # Build channel list for model construction.
    #
    # Two historical layouts exist:
    # - Newer UNet: separate `init_conv` then `encoders.0` consumes init_conv output.
    #   In this case return: [init_out, enc0_out, enc1_out, ...]
    # - Legacy UNet: no `init_conv`; `encoders.0` consumes raw in_channels (often 2).
    #   In this case return: [enc0_out, enc1_out, ...]
    max_idx = max(encoder_indices)
    
    # Get init_conv output channels (first encoder input)
    init_conv_key = None
    for key in state_dict.keys():
        if 'init_conv' in key and 'weight' in key and '.0.' in key:
            init_conv_key = key
            break
    
    if init_conv_key:
        # Newer layout: include init_conv output as the first entry.
        first_channel = int(state_dict[init_conv_key].shape[0])
        channels: List[int] = [first_channel]
        for i in range(max_idx + 1):
            if i in encoder_channels:
                channels.append(int(encoder_channels[i]))
            else:
                channels.append(int(channels[-1] * 2))
        return channels

    # Legacy layout: return encoder outputs only, in order.
    channels_legacy: List[int] = []
    for i in range(max_idx + 1):
        if i in encoder_channels:
            channels_legacy.append(int(encoder_channels[i]))
        else:
            channels_legacy.append(int(channels_legacy[-1] * 2) if channels_legacy else 32)
    return channels_legacy


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device = None,
    strict: bool = False
) -> tuple:
    """
    Load model from checkpoint with automatic format conversion.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        strict: Whether to strictly enforce state_dict matching
    
    Returns:
        Tuple of (model, config_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Get config from checkpoint if available
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    
    # Try to detect encoder channels from checkpoint
    detected_channels = detect_encoder_channels_from_checkpoint(state_dict)
    encoder_channels = model_cfg.get('encoder_channels', detected_channels)
    
    # Convert old checkpoint format if needed
    converted_state_dict = convert_old_checkpoint(state_dict)
    
    # Create model with detected configuration
    model = UNetDenoiser(
        in_channels=model_cfg.get('in_channels', 2),
        out_channels=model_cfg.get('out_channels', 2),
        encoder_channels=encoder_channels,
        use_attention=model_cfg.get('use_attention', True),
        dropout=0.0,  # No dropout during inference
        mask_type=model_cfg.get('mask_type', 'CRM')
    )
    
    # Try loading with converted state dict first
    try:
        model.load_state_dict(converted_state_dict, strict=strict)
        print(f"Successfully loaded checkpoint with {'strict' if strict else 'non-strict'} mode")
    except RuntimeError as e:
        if not strict:
            # Try partial load - load matching keys only
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in converted_state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
            unexpected_keys = set(converted_state_dict.keys()) - set(model_dict.keys())
            
            if pretrained_dict:
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"Partial load: {len(pretrained_dict)}/{len(model_dict)} keys loaded")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
            else:
                raise RuntimeError(f"Could not load any weights from checkpoint: {e}")
        else:
            raise
    
    model = model.to(device)
    return model, config


if __name__ == '__main__':
    # Test model
    model = UNetDenoiser()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 2, 257, 251)  # [batch, channels, freq, time]
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
