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
from utils.audio_utils import AudioProcessor, load_audio, save_audio, match_rms_level


class SpeechDenoiser:
    """
    Speech Denoiser for inference
    
    CẢI TIẾN:
    - Hỗ trợ match amplitude với input
    - Có thể normalize output theo target dB
    - Tránh volume quá nhỏ sau khi denoise
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        sample_rate: int = 16000,
        match_input_volume: bool = True,  # CẢI TIẾN: match volume với input
        target_db: Optional[float] = None  # CẢI TIẾN: target dB level
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            sample_rate: Target sample rate
            match_input_volume: Match output volume with input (recommended!)
            target_db: Target dB level for output (e.g., -25). Ignored if match_input_volume=True
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
        self.match_input_volume = match_input_volume
        self.target_db = target_db
        
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
        print(f"  Match input volume: {self.match_input_volume}")
        if self.target_db:
            print(f"  Target dB: {self.target_db}")
    
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
        match_volume: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Denoise audio waveform
        
        CẢI TIẾN: Match amplitude với input để tránh volume quá nhỏ
        
        Args:
            waveform: Input waveform [samples] or [batch, samples]
            match_volume: Override match_input_volume setting
        
        Returns:
            Denoised waveform
        """
        # Ensure batch dimension
        single_sample = waveform.dim() == 1
        if single_sample:
            waveform = waveform.unsqueeze(0)
        
        # Lưu lại input để match volume sau
        input_waveform = waveform.clone()
        
        waveform = waveform.to(self.device)
        
        # Compute STFT
        stft = self.audio_processor.stft(waveform)  # [batch, freq, time, 2]
        stft = stft.permute(0, 3, 1, 2)  # [batch, 2, freq, time]
        
        # Run model
        enhanced_stft = self.model(stft)
        
        # Convert back to waveform
        enhanced_stft = enhanced_stft.permute(0, 2, 3, 1)  # [batch, freq, time, 2]
        enhanced_wav = self.audio_processor.istft(enhanced_stft)
        
        # CẢI TIẾN: Match volume với input để tránh output quá nhỏ
        should_match = match_volume if match_volume is not None else self.match_input_volume
        
        if should_match:
            # Match RMS level với input
            enhanced_wav = enhanced_wav.cpu()
            input_np = input_waveform.numpy()
            enhanced_np = enhanced_wav.numpy()
            
            for i in range(enhanced_np.shape[0]):
                enhanced_np[i] = match_rms_level(enhanced_np[i], input_np[i])
            
            enhanced_wav = torch.from_numpy(enhanced_np)
        
        if single_sample:
            enhanced_wav = enhanced_wav.squeeze(0)
        
        return enhanced_wav
    
    def denoise_file(
        self,
        input_path: str,
        output_path: str,
        chunk_size: Optional[int] = None,
        preserve_volume: bool = True
    ) -> dict:
        """
        Denoise an audio file
        
        CẢI TIẾN: Giữ nguyên volume của input
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save denoised audio
            chunk_size: Process audio in chunks (for long files)
            preserve_volume: Match output volume with input (recommended!)
        
        Returns:
            Dictionary with processing info
        """
        start_time = time.time()
        
        # Load audio
        waveform, sr = load_audio(input_path, self.sample_rate)
        original_length = len(waveform)
        
        # Lưu reference để match volume
        input_numpy = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        
        # Process
        if chunk_size is None or len(waveform) <= chunk_size:
            enhanced = self.denoise(waveform, match_volume=preserve_volume)
        else:
            # Process in chunks for long audio
            enhanced = self._denoise_chunks(waveform, chunk_size)
            
            # Match volume với input sau khi xử lý chunks
            if preserve_volume:
                enhanced_np = enhanced.numpy() if isinstance(enhanced, torch.Tensor) else enhanced
                enhanced_np = match_rms_level(enhanced_np, input_numpy[:len(enhanced_np)])
                enhanced = torch.from_numpy(enhanced_np) if isinstance(enhanced, torch.Tensor) else enhanced_np
        
        # Trim to original length
        enhanced = enhanced[:original_length]
        
        # Save với reference để đảm bảo volume
        enhanced_np = enhanced.numpy() if isinstance(enhanced, torch.Tensor) else enhanced
        save_audio(
            output_path, 
            enhanced_np, 
            self.sample_rate,
            normalize=True,
            target_db=self.target_db,
            prevent_clipping=True,
            reference_waveform=input_numpy if preserve_volume else None
        )
        
        processing_time = time.time() - start_time
        rtf = processing_time / (original_length / self.sample_rate)
        
        return {
            'input_path': input_path,
            'output_path': output_path,
            'duration': original_length / self.sample_rate,
            'processing_time': processing_time,
            'rtf': rtf,  # Real-time factor
            'volume_preserved': preserve_volume
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
    
    # CẢI TIẾN: Options cho volume handling
    parser.add_argument('--preserve_volume', action='store_true', default=True,
                        help='Match output volume with input (default: True)')
    parser.add_argument('--no_preserve_volume', action='store_true',
                        help='Do not match output volume with input')
    parser.add_argument('--target_db', type=float, default=None,
                        help='Target dB level for output (e.g., -25)')
    
    args = parser.parse_args()
    
    # Handle preserve_volume flag
    preserve_volume = not args.no_preserve_volume
    
    # Validate arguments
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")
    
    if args.input is not None and args.output is None:
        args.output = args.input.replace('.wav', '_denoised.wav')
    
    if args.input_dir is not None and args.output_dir is None:
        args.output_dir = args.input_dir + '_denoised'
    
    # Create denoiser
    denoiser = SpeechDenoiser(
        checkpoint_path=args.checkpoint,
        device=args.device,
        match_input_volume=preserve_volume,
        target_db=args.target_db
    )
    
    # Process
    if args.input is not None:
        # Single file
        print(f"\nProcessing: {args.input}")
        result = denoiser.denoise_file(
            args.input, args.output,
            chunk_size=args.chunk_size,
            preserve_volume=preserve_volume
        )
        print(f"Output saved to: {result['output_path']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Real-time factor: {result['rtf']:.2f}x")
        print(f"Volume preserved: {result['volume_preserved']}")
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
