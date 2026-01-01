# Speech Denoising

A deep learning-based speech denoising system using U-Net architecture for removing background noise from audio recordings.

## Overview

This project implements a **speech enhancement/denoising** system that removes background noise while preserving speech quality. It uses a U-Net convolutional neural network operating on STFT spectrograms.

### Key Features

- **U-Net Architecture**: Encoder-decoder network with skip connections for preserving fine details
- **Complex Ratio Mask (CRM)**: Applies learned masks to both real and imaginary STFT components
- **Self-Attention**: Optional attention mechanism in the bottleneck for capturing long-range dependencies
- **Multi-Resolution STFT Loss**: Combined L1 and spectral loss for better perceptual quality
- **GUI Application**: User-friendly interface built with tkinter
- **Real-time Demo**: Live microphone denoising capability

## Architecture

```
Audio (noisy) → STFT → U-Net (Encoder → Bottleneck → Decoder) → Mask → iSTFT → Audio (clean)
```

The model processes complex STFT spectrograms (real + imaginary parts) and predicts a mask that is applied to enhance the speech signal.

### Model Details

- **Input**: Complex STFT [batch, 2, freq, time]
- **Encoder**: 5 stages with channels [32, 64, 128, 256, 512]
- **Bottleneck**: 1024 channels with optional self-attention
- **Decoder**: Mirrors encoder with skip connections
- **Output**: Enhanced complex STFT (same shape as input)
- **Parameters**: ~26M trainable parameters

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd speech_denoising

# Install dependencies
pip install -r requirements.txt
```

### Notebook/Jupyter import note

If you see `ModuleNotFoundError: No module named 'models'` in a notebook, add the project root to `sys.path` (the folder that contains `models/`), or run notebooks from the project root directory.

## Dataset

This project uses the **VoiceBank + DEMAND** dataset, widely used in speech enhancement research.

- **Clean speech**: VoiceBank corpus
- **Noise**: DEMAND database
- **Sample rate**: 16 kHz

### Download

Download from: https://datashare.ed.ac.uk/handle/10283/2791

### Directory Structure

After downloading, organize the data as follows:

```
speech_denoising/
└── data/
    ├── clean_trainset_28spk_wav/
    ├── noisy_trainset_28spk_wav/
    ├── clean_testset_wav/
    └── noisy_testset_wav/
```

## Usage

### Training

Train the model with default configuration:

```bash
python train.py --config config.yaml
```

Resume training from a checkpoint:

```bash
python train.py --config config.yaml --resume checkpoints/model_epoch_20.pt
```

### Inference

Denoise a single audio file:

```bash
python inference.py --input noisy_audio.wav --output clean_audio.wav --checkpoint checkpoints/best_model.pt
```

### Evaluation

Evaluate model performance on the test set:

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

**Metrics:**
| Metric | Description |
|--------|-------------|
| SNR | Signal-to-Noise Ratio |
| STOI | Short-Time Objective Intelligibility |
| PESQ | Perceptual Evaluation of Speech Quality (optional) |

### GUI Application

Launch the graphical interface:

```bash
python app.py
# or
python run_app.py
```

Features:
- Load and process audio files
- Visualize waveforms and spectrograms
- Compare before/after denoising
- Batch processing support

### Real-time Demo

Run real-time denoising with microphone input:

```bash
python realtime_demo.py
```

## Configuration

Edit `config.yaml` to customize training parameters:

```yaml
data:
  sample_rate: 16000
  segment_length: 32000  # 2 seconds

stft:
  n_fft: 512
  hop_length: 128
  win_length: 512

model:
  name: "UNetDenoiser"
  encoder_channels: [32, 64, 128, 256, 512]
  use_attention: true
  dropout: 0.1

training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
```

## Project Structure

```
speech_denoising/
├── app.py              # GUI application
├── run_app.py          # GUI launcher
├── train.py            # Training script
├── inference.py        # Single-file inference
├── evaluate.py         # Model evaluation
├── demo.py             # Quick demo script
├── realtime_demo.py    # Real-time microphone demo
├── config.yaml         # Configuration file
├── models/
│   ├── __init__.py
│   ├── unet.py         # U-Net model architecture
│   └── loss.py         # Loss functions
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py  # Audio processing utilities
│   └── metrics.py      # Evaluation metrics
├── requirements.txt
└── README.md
```

## Notes

- **Checkpoint loading appears stuck at `[3/4]`**: This is usually RAM swap/thrash while `load_state_dict` materializes tensors. Convert to a smaller weights-only checkpoint first:
  - Linux/macOS: `python3 convert_checkpoint.py --input best_model.pt --output best_model_weights.pt --keep-config`
  - Windows: `python convert_checkpoint.py --input best_model.pt --output best_model_weights.pt --keep-config` (or `py -3 ...`)
- **Avoid network drives** for checkpoints (e.g. `K:/`). Prefer local SSD.
- **PESQ Installation**: PESQ requires C compilation. On Windows, install Microsoft Visual C++ Build Tools. The system works without PESQ if unavailable.
- **GPU Training**: Recommended for faster training. Enable with CUDA-compatible GPU.
- **Training Time**: ~2-4 hours on CPU, significantly faster on GPU.

## License

MIT License
