"""
Inference script for Speech Denoising

CẢI TIẾN QUAN TRỌNG:
1. Proper denormalization để khôi phục amplitude gốc
2. Amplitude matching với input audio
3. Anti-clipping processing
4. Phase reconstruction từ input (cho STFT-based models)

Sử dụng để khử nhiễu cho file audio đơn lẻ hoặc thư mục chứa nhiều file

Usage:
    # Khử nhiễu một file
    python inference.py --input noisy.wav --output clean.wav --checkpoint best_model.pt
    
    # Khử nhiễu nhiều file
    python inference.py --input_dir noisy_folder/ --output_dir clean_folder/ --checkpoint best_model.pt
    
    # Với amplitude matching
    python inference.py --input noisy.wav --output clean.wav --checkpoint best_model.pt --match_amplitude
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Callable
import time
import json
import subprocess

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.load import load_model_checkpoint
from utils.audio_utils import (
    AudioProcessor, load_audio, save_audio,
    GlobalNormalizer, OutputProcessor,
    match_amplitude, prevent_clipping, compute_signal_stats
)


class SpeechDenoiser:
    """
    Speech Denoiser for inference
    
    CẢI TIẾN:
    1. Hỗ trợ global normalization (load từ training stats)
    2. Output amplitude matching với input
    3. Anti-clipping processing
    4. Debug statistics output
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        sample_rate: int = 16000,
        normalizer_path: Optional[str] = None,
        match_amplitude: bool = True,
        prevent_clipping: bool = True,
        strict_load: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
        cuda_probe_timeout_s: float = 10.0,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            sample_rate: Target sample rate
            normalizer_path: Path to normalizer stats JSON (for denormalization)
            match_amplitude: Match output amplitude with input
            prevent_clipping: Prevent output from clipping
        """
        self._progress = progress_callback
        self._cuda_probe_timeout_s = float(cuda_probe_timeout_s)

        # Set device (with a safe CUDA probe to avoid "hang forever")
        requested = device
        if requested is None:
            requested = 'cuda' if torch.cuda.is_available() else 'cpu'

        if str(requested).startswith("cuda"):
            if not self._probe_cuda_ok(timeout_s=self._cuda_probe_timeout_s):
                if self._progress:
                    self._progress(
                        f"CUDA init seems stuck (>{self._cuda_probe_timeout_s:.0f}s). Falling back to CPU."
                    )
                requested = "cpu"

        self.device = torch.device(requested)
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.strict_load = strict_load
        
        # Global normalizer - QUAN TRỌNG để denormalize output đúng cách
        self.normalizer = None
        if normalizer_path and Path(normalizer_path).exists():
            self.normalizer = GlobalNormalizer()
            self.normalizer.load(normalizer_path)
            print(f"Loaded normalizer stats: mean={self.normalizer.mean:.6f}, std={self.normalizer.std:.6f}")
        else:
            # Try to find normalizer in common locations
            possible_paths = [
                Path(checkpoint_path).parent / 'normalizer_stats.json',
                Path(checkpoint_path).parent.parent / 'data' / 'normalizer_stats.json',
                Path('./data/normalizer_stats.json')
            ]
            for p in possible_paths:
                if p.exists():
                    self.normalizer = GlobalNormalizer()
                    self.normalizer.load(str(p))
                    print(f"Found and loaded normalizer stats from: {p}")
                    break
        
        # Output processing settings
        self.match_amplitude_enabled = match_amplitude
        self.prevent_clipping_enabled = prevent_clipping
        
        # Output processor
        self.output_processor = OutputProcessor(
            normalizer=self.normalizer,
            match_amplitude_method='rms',
            prevent_clipping=prevent_clipping
        )
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sample_rate=sample_rate,
            normalizer=self.normalizer
        )
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Output processing: match_amplitude={match_amplitude}, prevent_clipping={prevent_clipping}")
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint with automatic format conversion"""
        model, _config = load_model_checkpoint(
            checkpoint_path,
            device=self.device,
            strict=self.strict_load,
            n_fft=self.n_fft,
            progress=self._progress,
        )
        return model

    @staticmethod
    def _probe_cuda_ok(timeout_s: float = 10.0) -> bool:
        """
        Probe CUDA init in a subprocess to avoid hanging the main process.
        If CUDA init is broken/hangs, subprocess will timeout and we fallback to CPU.
        """
        try:
            cmd = [
                sys.executable,
                "-c",
                "import torch; "
                "assert torch.cuda.is_available(); "
                "x=torch.zeros(1, device='cuda'); "
                "torch.cuda.synchronize(); "
                "print('ok')",
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_s)
            return True
        except Exception:
            return False
    
    @torch.no_grad()
    def denoise(
        self, 
        waveform: torch.Tensor,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Denoise audio waveform
        
        CẢI TIẾN:
        - Lưu input stats để amplitude matching
        - Proper denormalization
        
        Args:
            waveform: Input waveform [samples] or [batch, samples]
            return_stats: Return statistics dictionary
        
        Returns:
            Denoised waveform
        """
        # Ensure batch dimension
        single_input = waveform.dim() == 1
        if single_input:
            waveform = waveform.unsqueeze(0)
        
        # Lưu input stats trước khi normalize
        input_numpy = waveform.cpu().numpy()
        input_rms = np.sqrt(np.mean(input_numpy ** 2))
        input_peak = np.max(np.abs(input_numpy))
        
        waveform = waveform.to(self.device)
        
        # Compute STFT
        stft = self.audio_processor.stft(waveform)  # [batch, freq, time, 2]
        stft = stft.permute(0, 3, 1, 2)  # [batch, 2, freq, time]
        
        # Run model
        enhanced_stft = self.model(stft)
        
        # Convert back to waveform
        enhanced_stft = enhanced_stft.permute(0, 2, 3, 1)  # [batch, freq, time, 2]
        enhanced_wav = self.audio_processor.istft(enhanced_stft)
        
        # Squeeze if single input
        if single_input:
            enhanced_wav = enhanced_wav.squeeze(0)
        
        if return_stats:
            output_numpy = enhanced_wav.cpu().numpy()
            stats = {
                'input_rms': input_rms,
                'input_peak': input_peak,
                'output_rms': np.sqrt(np.mean(output_numpy ** 2)),
                'output_peak': np.max(np.abs(output_numpy)),
            }
            return enhanced_wav, stats
        
        return enhanced_wav
    
    def denoise_file(
        self,
        input_path: str,
        output_path: str,
        chunk_size: Optional[int] = None,
        verbose: bool = False
    ) -> dict:
        """
        Denoise an audio file
        
        CẢI TIẾN:
        1. Lưu input stats trước khi normalize
        2. Denormalize output đúng cách
        3. Match amplitude với input
        4. Anti-clipping
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save denoised audio
            chunk_size: Process audio in chunks (for long files)
            verbose: Print detailed statistics
        
        Returns:
            Dictionary with processing info
        """
        start_time = time.time()
        
        # Load audio (raw, chưa normalize)
        waveform_raw, sr = load_audio(input_path, self.sample_rate)
        original_length = len(waveform_raw)
        
        # Compute input stats TRƯỚC KHI normalize
        input_stats = compute_signal_stats(waveform_raw.numpy())
        
        if verbose:
            print(f"\nInput stats:")
            print(f"  RMS: {input_stats['rms']:.6f}")
            print(f"  Peak: {input_stats['peak']:.6f}")
        
        # Normalize input nếu có normalizer
        if self.normalizer is not None:
            waveform = torch.from_numpy(
                self.normalizer.normalize(waveform_raw.numpy())
            ).float()
        else:
            waveform = waveform_raw
        
        # Process
        if chunk_size is None or len(waveform) <= chunk_size:
            enhanced = self.denoise(waveform)
        else:
            # Process in chunks for long audio
            enhanced = self._denoise_chunks(waveform, chunk_size)
        
        # Trim to original length
        enhanced = enhanced[:original_length]
        
        # Convert to numpy
        enhanced_numpy = enhanced.cpu().numpy()
        
        # ===== POST-PROCESSING - QUAN TRỌNG! =====
        
        # 1. Denormalize nếu đã normalize
        if self.normalizer is not None:
            enhanced_numpy = self.normalizer.denormalize(enhanced_numpy)
        
        # 2. Match amplitude với input
        if self.match_amplitude_enabled:
            enhanced_numpy = match_amplitude(
                enhanced_numpy,
                waveform_raw.numpy(),
                method='rms'
            )
        
        # 3. Prevent clipping
        if self.prevent_clipping_enabled:
            enhanced_numpy = prevent_clipping(enhanced_numpy, threshold=0.99)
        
        # Compute output stats
        output_stats = compute_signal_stats(enhanced_numpy)
        
        if verbose:
            print(f"\nOutput stats:")
            print(f"  RMS: {output_stats['rms']:.6f}")
            print(f"  Peak: {output_stats['peak']:.6f}")
            print(f"  RMS ratio (out/in): {output_stats['rms'] / (input_stats['rms'] + 1e-8):.3f}")
        
        # Save
        save_audio(output_path, enhanced_numpy, self.sample_rate)
        
        processing_time = time.time() - start_time
        rtf = processing_time / (original_length / self.sample_rate)
        
        return {
            'input_path': input_path,
            'output_path': output_path,
            'duration': original_length / self.sample_rate,
            'processing_time': processing_time,
            'rtf': rtf,  # Real-time factor
            'input_rms': input_stats['rms'],
            'output_rms': output_stats['rms'],
            'amplitude_ratio': output_stats['rms'] / (input_stats['rms'] + 1e-8)
        }
    
    def _denoise_chunks(
        self,
        waveform: torch.Tensor,
        chunk_size: int,
        overlap: int = 4000
    ) -> torch.Tensor:
        """Process long audio in overlapping chunks"""
        total_length = len(waveform)
        enhanced_chunks = []
        
        # Process with overlap
        start = 0
        while start < total_length:
            end = min(start + chunk_size, total_length)
            chunk = waveform[start:end]
            
            # Pad if necessary
            if len(chunk) < chunk_size:
                pad_length = chunk_size - len(chunk)
                chunk = torch.nn.functional.pad(chunk, (0, pad_length))
            
            # Denoise
            enhanced_chunk = self.denoise(chunk)
            
            # Remove padding
            if end == total_length:
                enhanced_chunk = enhanced_chunk[:total_length - start]
            
            enhanced_chunks.append(enhanced_chunk)
            start += chunk_size - overlap
        
        # Combine chunks with crossfade
        return self._combine_chunks(enhanced_chunks, chunk_size, overlap)
    
    def _combine_chunks(
        self,
        chunks: list,
        chunk_size: int,
        overlap: int
    ) -> torch.Tensor:
        """Combine overlapping chunks with crossfade"""
        if len(chunks) == 1:
            return chunks[0]
        
        # Create output buffer
        total_length = (len(chunks) - 1) * (chunk_size - overlap) + len(chunks[-1])
        output = torch.zeros(total_length, device=chunks[0].device)
        weights = torch.zeros(total_length, device=chunks[0].device)
        
        # Crossfade window
        fade_in = torch.linspace(0, 1, overlap, device=chunks[0].device)
        fade_out = torch.linspace(1, 0, overlap, device=chunks[0].device)
        
        # Combine
        pos = 0
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk)
            
            # Apply fading
            if i > 0:
                chunk[:overlap] *= fade_in[:chunk_len] if chunk_len < overlap else fade_in
            if i < len(chunks) - 1:
                chunk[-overlap:] *= fade_out
            
            output[pos:pos + chunk_len] += chunk
            weights[pos:pos + chunk_len] += 1
            
            pos += chunk_size - overlap
        
        # Normalize by weights
        output /= weights.clamp(min=1)
        
        return output
    
    def denoise_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: list = ['.wav', '.mp3', '.flac', '.ogg']
    ) -> list:
        """
        Denoise all audio files in a directory
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: Audio file extensions to process
        
        Returns:
            List of processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_dir.glob(f'*{ext}'))
            audio_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(audio_files)} audio files")
        
        results = []
        for audio_file in tqdm(audio_files, desc="Processing"):
            output_path = output_dir / audio_file.name
            
            try:
                result = self.denoise_file(str(audio_file), str(output_path))
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results.append({
                    'input_path': str(audio_file),
                    'error': str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Speech Denoising Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input audio file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output audio file')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Path to input directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--chunk_size', type=int, default=160000,
                        help='Chunk size for processing (10 seconds at 16kHz)')
    
    # Output processing options
    parser.add_argument('--normalizer_path', type=str, default=None,
                        help='Path to normalizer stats JSON (for proper denormalization)')
    parser.add_argument('--match_amplitude', action='store_true', default=True,
                        help='Match output amplitude with input (recommended)')
    parser.add_argument('--no_match_amplitude', action='store_false', dest='match_amplitude',
                        help='Disable amplitude matching')
    parser.add_argument('--prevent_clipping', action='store_true', default=True,
                        help='Prevent output from clipping')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed statistics')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")
    
    if args.input is not None and args.output is None:
        args.output = args.input.replace('.wav', '_denoised.wav')
    
    if args.input_dir is not None and args.output_dir is None:
        args.output_dir = args.input_dir + '_denoised'
    
    print("=" * 60)
    print("SPEECH DENOISING INFERENCE")
    print("=" * 60)
    
    # Create denoiser
    denoiser = SpeechDenoiser(
        checkpoint_path=args.checkpoint,
        device=args.device,
        normalizer_path=args.normalizer_path,
        match_amplitude=args.match_amplitude,
        prevent_clipping=args.prevent_clipping
    )
    
    # Process
    if args.input is not None:
        # Single file
        print(f"\nProcessing: {args.input}")
        result = denoiser.denoise_file(
            args.input, args.output,
            chunk_size=args.chunk_size,
            verbose=args.verbose
        )
        
        print(f"\n{'='*40}")
        print(f"Output saved to: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Real-time factor: {result['rtf']:.2f}x")
        print(f"\nAmplitude ratio (output/input): {result['amplitude_ratio']:.3f}")
        if result['amplitude_ratio'] < 0.5:
            print("⚠️  Warning: Output amplitude is significantly lower than input!")
            print("   This may indicate a problem with the model or normalization.")
        elif result['amplitude_ratio'] > 1.5:
            print("⚠️  Warning: Output amplitude is significantly higher than input!")
        else:
            print("✓  Amplitude preserved correctly")
    else:
        # Directory
        print(f"\nProcessing directory: {args.input_dir}")
        results = denoiser.denoise_directory(args.input_dir, args.output_dir)
        
        # Summary
        successful = [r for r in results if 'error' not in r]
        print(f"\n{'='*40}")
        print(f"Processed {len(successful)}/{len(results)} files successfully")
        
        if successful:
            avg_rtf = np.mean([r['rtf'] for r in successful])
            avg_amp_ratio = np.mean([r.get('amplitude_ratio', 1.0) for r in successful])
            
            print(f"Average real-time factor: {avg_rtf:.2f}x")
            print(f"Average amplitude ratio: {avg_amp_ratio:.3f}")
            
            if avg_amp_ratio < 0.7:
                print("\n⚠️  Warning: Average output amplitude is low!")
                print("   Consider checking model training or normalization settings.")


if __name__ == '__main__':
    main()
