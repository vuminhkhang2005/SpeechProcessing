"""
Real-time Speech Denoising Demo

Demo khử nhiễu tiếng nói theo thời gian thực từ microphone

Yêu cầu thêm:
    pip install pyaudio

Usage:
    python realtime_demo.py --checkpoint checkpoints/best_model.pt
"""

import sys
import argparse
from pathlib import Path
import threading
import queue
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from models.unet import UNetDenoiser
from utils.audio_utils import AudioProcessor

# Try to import PyAudio
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class RealtimeDenoiser:
    """
    Real-time speech denoiser using streaming audio
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,  # 500ms chunks
        device: str = None
    ):
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            n_fft=512,
            hop_length=128,
            win_length=512,
            sample_rate=sample_rate
        )
        
        # Buffers
        self.input_buffer = np.zeros(self.chunk_size, dtype=np.float32)
        self.output_buffer = np.zeros(self.chunk_size, dtype=np.float32)
        
        # Threading
        self.is_running = False
        self.audio_queue = queue.Queue()
        
        print(f"Realtime Denoiser initialized on {self.device}")
        print(f"Chunk size: {self.chunk_size} samples ({chunk_duration*1000:.0f}ms)")
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        config = checkpoint.get('config', {})
        model_cfg = config.get('model', {})
        
        model = UNetDenoiser(
            in_channels=2,
            out_channels=2,
            encoder_channels=model_cfg.get('encoder_channels', [32, 64, 128, 256, 512]),
            use_attention=model_cfg.get('use_attention', True),
            dropout=0.0
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process a single audio chunk"""
        # Convert to tensor
        waveform = torch.from_numpy(audio_chunk).float().to(self.device)
        
        # STFT
        stft = self.audio_processor.stft(waveform)
        stft = stft.permute(0, 3, 1, 2)
        
        # Model inference
        enhanced_stft = self.model(stft)
        
        # iSTFT
        enhanced_stft = enhanced_stft.permute(0, 2, 3, 1)
        enhanced = self.audio_processor.istft(enhanced_stft)
        
        return enhanced.squeeze().cpu().numpy()
    
    def start_stream(self):
        """Start audio stream (requires PyAudio)"""
        if not PYAUDIO_AVAILABLE:
            print("Error: PyAudio not installed. Install with: pip install pyaudio")
            return
        
        self.is_running = True
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        def audio_callback(in_data, frame_count, time_info, status):
            # Convert bytes to numpy
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Process
            try:
                enhanced = self.process_chunk(audio_data)
                out_data = enhanced.astype(np.float32).tobytes()
            except Exception as e:
                print(f"Processing error: {e}")
                out_data = in_data
            
            return (out_data, pyaudio.paContinue if self.is_running else pyaudio.paComplete)
        
        # Open stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )
        
        print("\n" + "=" * 50)
        print("Real-time Speech Denoising Active")
        print("=" * 50)
        print("Speak into the microphone...")
        print("Press Ctrl+C to stop")
        print("=" * 50 + "\n")
        
        stream.start_stream()
        
        try:
            while stream.is_active() and self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        
        self.is_running = False
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("Stream stopped.")
    
    def process_file_realtime_simulation(
        self,
        input_path: str,
        output_path: str = None
    ):
        """
        Simulate real-time processing on a file
        (useful for testing without microphone)
        """
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(input_path)
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Process in chunks
        num_chunks = len(audio) // self.chunk_size
        enhanced_audio = []
        
        print(f"Processing {num_chunks} chunks...")
        start_time = time.time()
        
        for i in range(num_chunks):
            chunk = audio[i * self.chunk_size:(i + 1) * self.chunk_size]
            enhanced_chunk = self.process_chunk(chunk)
            enhanced_audio.append(enhanced_chunk)
        
        elapsed = time.time() - start_time
        rtf = elapsed / (num_chunks * self.chunk_size / self.sample_rate)
        
        print(f"Processing time: {elapsed:.2f}s")
        print(f"Real-time factor: {rtf:.2f}x")
        
        # Save
        if output_path:
            enhanced = np.concatenate(enhanced_audio)
            sf.write(output_path, enhanced, self.sample_rate)
            print(f"Saved to: {output_path}")
        
        return np.concatenate(enhanced_audio)


def main():
    parser = argparse.ArgumentParser(description='Real-time Speech Denoising')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='stream',
                        choices=['stream', 'file'],
                        help='Mode: stream (microphone) or file')
    parser.add_argument('--input', type=str, default=None,
                        help='Input file for file mode')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for file mode')
    parser.add_argument('--chunk_duration', type=float, default=0.5,
                        help='Chunk duration in seconds')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create denoiser
    denoiser = RealtimeDenoiser(
        checkpoint_path=args.checkpoint,
        chunk_duration=args.chunk_duration,
        device=args.device
    )
    
    if args.mode == 'stream':
        if not PYAUDIO_AVAILABLE:
            print("\n" + "=" * 60)
            print("PyAudio không được cài đặt!")
            print("Để sử dụng real-time streaming, hãy cài đặt PyAudio:")
            print("  pip install pyaudio")
            print()
            print("Trên Ubuntu/Debian:")
            print("  sudo apt-get install portaudio19-dev")
            print("  pip install pyaudio")
            print()
            print("Trên macOS:")
            print("  brew install portaudio")
            print("  pip install pyaudio")
            print("=" * 60)
            return
        
        denoiser.start_stream()
    else:
        if args.input is None:
            print("Error: --input required for file mode")
            return
        
        output = args.output or args.input.replace('.wav', '_realtime_denoised.wav')
        denoiser.process_file_realtime_simulation(args.input, output)


if __name__ == '__main__':
    main()
