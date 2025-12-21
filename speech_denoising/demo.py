"""
Demo script để kiểm tra model hoạt động

Tạo audio noise giả lập và test model inference

Usage:
    python demo.py --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from models.unet import UNetDenoiser
from utils.audio_utils import AudioProcessor, save_audio


def generate_test_signal(
    duration: float = 3.0,
    sample_rate: int = 16000,
    speech_freq: float = 300.0,
    noise_level: float = 0.3
):
    """
    Tạo tín hiệu test giả lập (không cần dataset thật)
    
    Tạo một tín hiệu "giọng nói" đơn giản (sóng sin với harmonics)
    và thêm noise trắng
    """
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    
    # Tạo "giọng nói" giả lập với harmonics
    speech = np.zeros_like(t)
    for i, harmonic in enumerate([1, 2, 3, 4, 5]):
        amplitude = 1.0 / (i + 1)
        speech += amplitude * np.sin(2 * np.pi * speech_freq * harmonic * t)
    
    # Thêm envelope để giống giọng nói hơn
    envelope = np.abs(np.sin(2 * np.pi * 3 * t)) ** 0.3
    speech = speech * envelope
    
    # Normalize
    speech = speech / np.max(np.abs(speech)) * 0.7
    
    # Thêm noise
    noise = np.random.randn(len(t)).astype(np.float32) * noise_level
    noisy = speech + noise
    
    return torch.from_numpy(speech), torch.from_numpy(noisy)


def plot_spectrograms(clean, noisy, enhanced, sample_rate=16000, save_path=None):
    """Vẽ spectrogram của clean, noisy, và enhanced audio"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    signals = [
        (noisy, 'Noisy Speech'),
        (enhanced, 'Enhanced Speech'),
        (clean, 'Clean Speech (Reference)')
    ]
    
    for ax, (signal, title) in zip(axes, signals):
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()
        
        # Compute spectrogram
        D = np.abs(np.fft.rfft(signal.reshape(-1, 512), axis=-1))
        D = 20 * np.log10(D + 1e-8)
        
        ax.imshow(D.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title)
        ax.set_ylabel('Frequency Bin')
        ax.set_xlabel('Time Frame')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Spectrogram saved to: {save_path}")
    else:
        plt.show()


def demo_model(checkpoint_path: str = None):
    """Demo model với tín hiệu test"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Tạo tín hiệu test
    print("\n1. Generating test signals...")
    clean, noisy = generate_test_signal(duration=2.0)
    print(f"   Signal length: {len(clean)} samples ({len(clean)/16000:.2f}s)")
    
    # Tạo model
    print("\n2. Creating model...")
    model = UNetDenoiser(
        in_channels=2,
        out_channels=2,
        encoder_channels=[32, 64, 128, 256, 512],
        use_attention=True,
        dropout=0.0
    ).to(device)
    
    # Load checkpoint nếu có
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded from epoch {checkpoint['epoch']}")
    else:
        print("   Using random weights (no checkpoint)")
    
    model.eval()
    print(f"   Model parameters: {model.count_parameters():,}")
    
    # Audio processor
    audio_processor = AudioProcessor(
        n_fft=512,
        hop_length=128,
        win_length=512
    )
    
    # Inference
    print("\n3. Running inference...")
    with torch.no_grad():
        # STFT
        noisy_stft = audio_processor.stft(noisy.to(device))
        noisy_stft = noisy_stft.permute(0, 3, 1, 2)  # [1, 2, freq, time]
        
        # Model
        enhanced_stft = model(noisy_stft)
        
        # iSTFT
        enhanced_stft = enhanced_stft.permute(0, 2, 3, 1)
        enhanced = audio_processor.istft(enhanced_stft).squeeze(0)
    
    print(f"   Enhanced signal length: {len(enhanced)} samples")
    
    # Lưu output
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    
    print("\n4. Saving audio files...")
    save_audio(output_dir / 'demo_clean.wav', clean, 16000)
    save_audio(output_dir / 'demo_noisy.wav', noisy, 16000)
    save_audio(output_dir / 'demo_enhanced.wav', enhanced.cpu(), 16000)
    
    print(f"   Saved to: {output_dir}/")
    print("   - demo_clean.wav")
    print("   - demo_noisy.wav")
    print("   - demo_enhanced.wav")
    
    # Plot
    print("\n5. Generating spectrogram plot...")
    try:
        plot_spectrograms(
            clean, noisy, enhanced.cpu(),
            save_path=output_dir / 'demo_spectrograms.png'
        )
    except Exception as e:
        print(f"   Could not generate plot: {e}")
    
    print("\n✅ Demo completed!")
    print("\nNote: Without a trained model, the enhanced audio may not be meaningful.")
    print("Train the model with: python train.py")


def test_model_architecture():
    """Test kiến trúc model"""
    print("Testing U-Net architecture...")
    
    model = UNetDenoiser()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test với input size khác nhau
    test_sizes = [
        (1, 2, 257, 63),   # ~0.5 second
        (1, 2, 257, 126),  # ~1 second
        (1, 2, 257, 251),  # ~2 seconds
        (4, 2, 257, 251),  # batch size 4
    ]
    
    for size in test_sizes:
        x = torch.randn(size)
        y = model(x)
        print(f"Input: {size} -> Output: {y.shape}")
        assert x.shape == y.shape, "Shape mismatch!"
    
    print("✅ All tests passed!")


def main():
    parser = argparse.ArgumentParser(description='Demo Speech Denoising')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--test_architecture', action='store_true',
                        help='Test model architecture only')
    
    args = parser.parse_args()
    
    if args.test_architecture:
        test_model_architecture()
    else:
        demo_model(args.checkpoint)


if __name__ == '__main__':
    main()
