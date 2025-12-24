# Speech Denoising

A deep learning-based speech denoising system using U-Net architecture for removing background noise from audio recordings.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/speech_denoising/blob/main/train_colab.ipynb)

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

## Dataset

This project uses the **VoiceBank + DEMAND** dataset, widely used in speech enhancement research.

- **Clean speech**: VoiceBank corpus
- **Noise**: DEMAND database
- **Sample rate**: 16 kHz

### Option 1: Local Download

Download from: https://datashare.ed.ac.uk/handle/10283/2791

After downloading, organize the data as follows:

```
speech_denoising/
└── data/
    ├── clean_trainset_28spk_wav/
    ├── noisy_trainset_28spk_wav/
    ├── clean_testset_wav/
    └── noisy_testset_wav/
```

### Option 2: Google Drive (for Colab)

Upload your dataset to Google Drive with this structure:

```
My Drive/
└── datasets/                          # Your dataset folder
    ├── clean_trainset_28spk_wav/      # 11,572 .wav files
    ├── noisy_trainset_28spk_wav/      # 11,572 .wav files
    ├── clean_testset_wav/             # 824 .wav files
    └── noisy_testset_wav/             # 824 .wav files
```

Then use in Colab:
```python
from data.dataset import setup_gdrive_dataset, create_dataloaders

# Setup dataset from Google Drive
paths = setup_gdrive_dataset(gdrive_path='/content/drive/MyDrive/datasets')

# Create dataloaders
train_loader, val_loader = create_dataloaders(**paths)
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

### 🚀 Train on Google Colab with Google Drive Dataset (Recommended)

Train the model for free on Google Colab with GPU acceleration, using your dataset from Google Drive:

1. **Upload dataset to Google Drive**: Upload the VoiceBank + DEMAND dataset to your Google Drive (see Dataset section above)

2. **Open the notebook**: Click the "Open in Colab" badge at the top of this README, or upload `train_colab.ipynb` to Google Colab

3. **Enable GPU**: Go to `Runtime` → `Change runtime type` → Select `GPU`

4. **Configure dataset path**: In the notebook, set your Google Drive dataset path:
   ```python
   GDRIVE_DATASET_PATH = "/content/drive/MyDrive/datasets"  # Your path
   ```

5. **Run all cells**: The notebook will:
   - Mount Google Drive automatically
   - Load dataset directly from Drive (no download needed!)
   - Train the model for 50 epochs (~1-2 hours)
   - Save the best model

6. **Save model to Drive**: The trained model can be saved back to Google Drive for persistent storage

**Benefits of Google Drive Dataset:**
- ✅ No need to re-download dataset each session
- ✅ Dataset persists across Colab sessions
- ✅ Faster startup time
- ✅ Save trained models to Drive

**Colab Tips:**
- Batch size is reduced to 8 for Colab GPU memory constraints
- Training for 50 epochs is a good starting point; increase for better results

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
├── train_colab.ipynb   # Google Colab training notebook
├── inference.py        # Single-file inference
├── evaluate.py         # Model evaluation
├── demo.py             # Quick demo script
├── realtime_demo.py    # Real-time microphone demo
├── config.yaml         # Configuration file
├── download_dataset.py # Dataset download helper
├── data/
│   ├── __init__.py
│   └── dataset.py      # Dataset classes and dataloaders
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

- **PESQ Installation**: PESQ requires C compilation. On Windows, install Microsoft Visual C++ Build Tools. The system works without PESQ if unavailable.
- **GPU Training**: Recommended for faster training. Enable with CUDA-compatible GPU.
- **Training Time**: ~2-4 hours on CPU, significantly faster on GPU.

## License

MIT License
