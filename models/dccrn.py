"""
DCCRN (Deep Complex Convolution Recurrent Network) - TF-domain speech enhancement.

Implementation notes (pragmatic / "deadline-safe"):
- Operates on complex STFT represented as 2 channels: [B, 2, F, T] (real, imag).
- Uses complex convolutions implemented via two real convs (Wr/Wi) and algebra.
- Uses an LSTM on the bottleneck features (concatenated real+imag) to model temporal context.
- Predicts a complex residual (delta STFT) and adds it to the input (residual mapping)
  to reduce the risk of volume collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_ri(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [B, 2*C, F, T] -> (real, imag) each [B, C, F, T]
    c2 = x.shape[1]
    if c2 % 2 != 0:
        raise ValueError(f"Expected even channel dim for complex tensor, got {c2}")
    c = c2 // 2
    return x[:, :c], x[:, c:]


def _merge_ri(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    return torch.cat([real, imag], dim=1)


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (5, 2),
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 1),
        bias: bool = True,
    ):
        super().__init__()
        self.wr = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.wi = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return _merge_ri(yr, yi)


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (5, 2),
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 1),
        output_padding: Tuple[int, int] = (1, 0),
        bias: bool = True,
    ):
        super().__init__()
        self.wr = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.wi = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return _merge_ri(yr, yi)


class ComplexBatchNorm2d(nn.Module):
    """
    Simple complex BN: apply BN to real and imag separately.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)
        return _merge_ri(self.bn_r(xr), self.bn_i(xi))


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters: int):
        super().__init__()
        self.act_r = nn.PReLU(num_parameters)
        self.act_i = nn.PReLU(num_parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)
        return _merge_ri(self.act_r(xr), self.act_i(xi))


class ComplexEncoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (5, 2),
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 1),
    ):
        super().__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = ComplexBatchNorm2d(out_ch)
        self.act = ComplexPReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ComplexDecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (5, 2),
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 1),
        output_padding: Tuple[int, int] = (1, 0),
    ):
        super().__init__()
        self.deconv = ComplexConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bn = ComplexBatchNorm2d(out_ch)
        self.act = ComplexPReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))

class ComplexUpBlock(nn.Module):
    """
    Upsample block with skip connection:
      deconv(in_ch -> out_ch) + concat(skip) + conv(out_ch+skip_ch -> out_ch)
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int] = (5, 2),
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 1),
        output_padding: Tuple[int, int] = (1, 0),
    ):
        super().__init__()
        self.up = ComplexDecoderBlock(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.fuse = nn.Sequential(
            ComplexConv2d(
                in_channels=out_ch + skip_ch,
                out_channels=out_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            ComplexBatchNorm2d(out_ch),
            ComplexPReLU(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


@dataclass(frozen=True)
class DCCRNConfig:
    """
    Minimal DCCRN configuration.

    encoder_channels are "real" channel counts; internally we keep 2x for (r,i) packing.
    """

    encoder_channels: List[int] = (16, 32, 64, 128, 256)  # type: ignore[assignment]
    kernel_size: Tuple[int, int] = (5, 2)
    stride: Tuple[int, int] = (2, 1)  # downsample freq only
    padding: Tuple[int, int] = (2, 1)
    rnn_layers: int = 2
    rnn_hidden: int = 256


class DCCRN(nn.Module):
    """
    Input:  [B, 2, F, T] complex STFT (real, imag)
    Output: [B, 2, F, T] enhanced complex STFT (residual mapping)
    """

    def __init__(self, freq_bins: int = 257, cfg: DCCRNConfig | None = None):
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.cfg = cfg or DCCRNConfig()

        # Encoder
        enc_ch = list(self.cfg.encoder_channels)
        self.encoders = nn.ModuleList()
        in_ch = 1  # per real/imag branch; packed input has 2 channels => 1 complex channel
        # We pack complex as [2*C,...] where C is "complex channels" count. Input C=1.
        for out_ch in enc_ch:
            self.encoders.append(
                ComplexEncoderBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=self.cfg.kernel_size,
                    stride=self.cfg.stride,
                    padding=self.cfg.padding,
                )
            )
            in_ch = out_ch

        # Bottleneck (no further downsample)
        self.bottleneck = nn.Sequential(
            ComplexEncoderBlock(
                in_ch=in_ch,
                out_ch=in_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            ComplexEncoderBlock(
                in_ch=in_ch,
                out_ch=in_ch,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )

        # Infer bottleneck F and build RNN projections (deadline-safe: infer by dummy forward)
        with torch.no_grad():
            dummy = torch.zeros(1, 2, self.freq_bins, 10)
            z = self._encode_features(dummy)
            _, _, f_b, _ = z.shape
            self._bottleneck_freq = int(f_b)
            self._bottleneck_ch = int(z.shape[1] // 2)  # complex channels

        rnn_in = 2 * self._bottleneck_ch * self._bottleneck_freq
        self.rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=self.cfg.rnn_hidden,
            num_layers=self.cfg.rnn_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.rnn_proj = nn.Linear(self.cfg.rnn_hidden, rnn_in)

        # Decoder (mirror)
        self.decoders = nn.ModuleList()
        # Example enc_ch: [16,32,64,128,256]
        # Decoder targets: [128,64,32,16] with matching skips.
        dec_targets = enc_ch[-2::-1]
        in_c = enc_ch[-1]
        for out_c in dec_targets:
            self.decoders.append(
                ComplexUpBlock(
                    in_ch=in_c,
                    skip_ch=out_c,
                    out_ch=out_c,
                    kernel_size=self.cfg.kernel_size,
                    stride=self.cfg.stride,
                    padding=self.cfg.padding,
                    output_padding=(1, 0),
                )
            )
            in_c = out_c

        # Final projection to complex residual (1 complex channel => 2 real channels)
        last_c = enc_ch[0]
        self.out_conv_r = nn.Conv2d(last_c, 1, kernel_size=1)
        self.out_conv_i = nn.Conv2d(last_c, 1, kernel_size=1)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, F, T] -> convert to packed complex channels [B, 2*1, F, T]
        # Here "complex channels" C=1, so packed already.
        z = x
        for enc in self.encoders:
            z = enc(z)
        z = self.bottleneck(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        skips: List[torch.Tensor] = []

        z = x
        for enc in self.encoders:
            z = enc(z)
            skips.append(z)

        z = self.bottleneck(z)

        # RNN on bottleneck features over time
        zr, zi = _split_ri(z)
        zcat = torch.cat([zr, zi], dim=1)  # [B, 2C, F, T]
        b, ch2, f, t = zcat.shape
        seq = zcat.permute(0, 3, 1, 2).contiguous().view(b, t, ch2 * f)  # [B, T, 2C*F]
        seq_out, _ = self.rnn(seq)
        seq_out = self.rnn_proj(seq_out)
        zcat2 = seq_out.view(b, t, ch2, f).permute(0, 2, 3, 1).contiguous()  # [B, 2C, F, T]
        zr2, zi2 = torch.split(zcat2, zr.shape[1], dim=1)
        z = _merge_ri(zr2, zi2)

        # Decoder with skip connections (mirror encoder levels)
        # skips: [enc16, enc32, enc64, enc128, enc256] (in order)
        # We start from bottleneck at enc256 and go to 128->64->32->16
        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            z = dec(z, skip)

        # Output residual
        zr, zi = _split_ri(z)
        dr = self.out_conv_r(zr)
        di = self.out_conv_i(zi)
        delta = torch.cat([dr, di], dim=1)  # [B, 2, F, T]

        # Residual mapping + final shape match
        if delta.shape[2:] != inp.shape[2:]:
            delta = F.interpolate(delta, size=inp.shape[2:], mode="nearest")
        return inp + delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

