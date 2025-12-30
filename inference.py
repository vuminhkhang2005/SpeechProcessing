"""
Inference script for Speech Denoising

Sử dụng để khử nhiễu cho file audio đơn lẻ hoặc thư mục chứa nhiều file

Usage:
    # Khử nhiễu một file
    python inference.py --input noisy.wav --output clean.wav --checkpoint best_model.pt
    
    # Khử nhiễu nhiều file
    python inference.py --input_dir noisy_folder/ --output_dir clean_folder/ --checkpoint best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import time

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.unet import UNetDenoiser, load_model_checkpoint, convert_old_checkpoint, detect_encoder_channels_from_checkpoint
from utils.audio_utils import (
    AudioProcessor, load_audio, save_audio, 
    match_energy, compute_rms, AmplitudePreserver
)


class SpeechDenoiser:
    """
    Speech Denoiser for inference
    
    Cải tiến:
    - Preserve amplitude: Đảm bảo output có âm lượng tương tự input
    - Energy matching: Khớp năng lượng output với input để tránh "quá nhỏ tiếng"
    - Safe clipping: Tránh clipping khi save file
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        sample_rate: int = 16000,
        preserve_amplitude: bool = True,
        amplitude_method: str = 'rms'
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            sample_rate: Target sample rate
            preserve_amplitude: Whether to preserve input amplitude in output
            amplitude_method: Method for amplitude preservation ('rms', 'peak', 'energy')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Amplitude preservation settings
        self.preserve_amplitude = preserve_amplitude
        self.amplitude_method = amplitude_method
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            sample_rate=sample_rate
        )
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Amplitude preservation: {preserve_amplitude} (method: {amplitude_method})")
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint with automatic format conversion"""
        try:
            # Try using the new smart loading function
            model, config = load_model_checkpoint(
                checkpoint_path, 
                device=self.device,
                strict=False
            )
            return model
        except Exception as e:
            print(f"Smart loading failed: {e}")
            print("Trying fallback loading method...")
            
            # Fallback to manual loading with conversion
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
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
            
            # Detect encoder channels from checkpoint
            detected_channels = detect_encoder_channels_from_checkpoint(state_dict)
            encoder_channels = model_cfg.get('encoder_channels', detected_channels)
            
            # Convert old checkpoint format
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
            
            # Try to load weights with non-strict mode
            try:
                model.load_state_dict(converted_state_dict, strict=True)
            except RuntimeError:
                # Partial loading
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in converted_state_dict.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                
                if not pretrained_dict:
                    raise RuntimeError("Cannot load any weights from checkpoint. Architecture mismatch.")
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} weights (partial load)")
            
            model = model.to(self.device)
            return model
    
    @torch.no_grad()
    def denoise(
        self, 
        waveform: torch.Tensor,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Denoise audio waveform
        
        Args:
            waveform: Input waveform [samples] or [batch, samples]
            return_stats: If True, also return processing statistics
        
        Returns:
            Denoised waveform (and stats if requested)
        """
        # Ensure batch dimension
        single_input = waveform.dim() == 1
        if single_input:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        
        # Store input amplitude for preservation
        if self.preserve_amplitude:
            input_waveform_np = waveform.cpu().numpy()
            amplitude_preserver = AmplitudePreserver(method=self.amplitude_method)
            amplitude_preserver.store_input_stats(input_waveform_np[0] if single_input else input_waveform_np)
        
        # Compute STFT
        stft = self.audio_processor.stft(waveform)  # [batch, freq, time, 2]
        stft = stft.permute(0, 3, 1, 2)  # [batch, 2, freq, time]
        
        # Run model
        enhanced_stft = self.model(stft)
        
        # Convert back to waveform
        enhanced_stft = enhanced_stft.permute(0, 2, 3, 1)  # [batch, freq, time, 2]
        enhanced_wav = self.audio_processor.istft(enhanced_stft)
        
        # Amplitude preservation
        if self.preserve_amplitude:
            enhanced_np = enhanced_wav.squeeze(0).cpu().numpy()
            enhanced_np = amplitude_preserver.restore_amplitude(enhanced_np)
            enhanced_wav = torch.from_numpy(enhanced_np).float()
            if not single_input:
                enhanced_wav = enhanced_wav.unsqueeze(0)
        else:
            enhanced_wav = enhanced_wav.squeeze(0) if single_input else enhanced_wav
        
        if return_stats:
            stats = {
                'input_rms': amplitude_preserver.input_rms if self.preserve_amplitude else None,
                'output_rms': compute_rms(enhanced_wav.cpu().numpy())
            }
            return enhanced_wav, stats
        
        return enhanced_wav
    
    def denoise_file(
        self,
        input_path: str,
        output_path: str,
        chunk_size: Optional[int] = None,
        normalize_mode: str = 'safe'
    ) -> dict:
        """
        Denoise an audio file
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save denoised audio
            chunk_size: Process audio in chunks (for long files)
            normalize_mode: How to normalize output ('safe', 'energy', 'peak', 'none')
                - 'safe': Only scale down if would clip (default, recommended)
                - 'energy': Match energy to input (good for preserving loudness)
                - 'peak': Peak normalize (not recommended for speech)
                - 'none': No normalization
        
        Returns:
            Dictionary with processing info
        """
        start_time = time.time()
        
        # Load audio
        waveform, sr = load_audio(input_path, self.sample_rate)
        original_length = len(waveform)
        
        # Store input for reference
        input_waveform = waveform.clone()
        input_rms = compute_rms(waveform.numpy())
        
        # Process
        if chunk_size is None or len(waveform) <= chunk_size:
            enhanced = self.denoise(waveform)
        else:
            # Process in chunks for long audio
            enhanced = self._denoise_chunks(waveform, chunk_size)
        
        # Ensure tensor
        if not isinstance(enhanced, torch.Tensor):
            enhanced = torch.from_numpy(enhanced).float()
        
        # Trim to original length
        enhanced = enhanced[:original_length]
        
        # Calculate output stats before saving
        output_rms = compute_rms(enhanced.numpy())
        amplitude_ratio = output_rms / (input_rms + 1e-8)
        
        # Save with proper normalization
        # Use input waveform as reference for energy matching
        reference = input_waveform.numpy() if normalize_mode == 'energy' else None
        save_audio(
            output_path, 
            enhanced, 
            self.sample_rate,
            normalize_mode=normalize_mode,
            reference_waveform=reference
        )
        
        processing_time = time.time() - start_time
        rtf = processing_time / (original_length / self.sample_rate)
        
        return {
            'input_path': input_path,
            'output_path': output_path,
            'duration': original_length / self.sample_rate,
            'processing_time': processing_time,
            'rtf': rtf,  # Real-time factor
            'input_rms': input_rms,
            'output_rms': output_rms,
            'amplitude_ratio': amplitude_ratio
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
    
    # Amplitude preservation options
    parser.add_argument('--preserve_amplitude', type=bool, default=True,
                        help='Preserve input amplitude in output')
    parser.add_argument('--amplitude_method', type=str, default='rms',
                        choices=['rms', 'peak', 'energy'],
                        help='Method for amplitude preservation')
    parser.add_argument('--normalize_mode', type=str, default='safe',
                        choices=['safe', 'energy', 'peak', 'none'],
                        help='How to normalize output when saving')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")
    
    if args.input is not None and args.output is None:
        args.output = args.input.replace('.wav', '_denoised.wav')
    
    if args.input_dir is not None and args.output_dir is None:
        args.output_dir = args.input_dir + '_denoised'
    
    # Create denoiser with amplitude preservation
    denoiser = SpeechDenoiser(
        checkpoint_path=args.checkpoint,
        device=args.device,
        preserve_amplitude=args.preserve_amplitude,
        amplitude_method=args.amplitude_method
    )
    
    # Process
    if args.input is not None:
        # Single file
        print(f"\nProcessing: {args.input}")
        result = denoiser.denoise_file(
            args.input, args.output,
            chunk_size=args.chunk_size,
            normalize_mode=args.normalize_mode
        )
        print(f"Output saved to: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Real-time factor: {result['rtf']:.2f}x")
        print(f"Amplitude ratio (output/input): {result.get('amplitude_ratio', 1.0):.3f}")
    else:
        # Directory
        print(f"\nProcessing directory: {args.input_dir}")
        results = denoiser.denoise_directory(args.input_dir, args.output_dir)
        
        # Summary
        successful = [r for r in results if 'error' not in r]
        print(f"\nProcessed {len(successful)}/{len(results)} files successfully")
        if successful:
            avg_rtf = np.mean([r['rtf'] for r in successful])
            print(f"Average real-time factor: {avg_rtf:.2f}x")


if __name__ == '__main__':
    main()
