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

### Anti "Lazy Learning" Features

This project includes several features to prevent the model from "lazy learning" (just reducing volume instead of actually denoising):

- **SI-SDR Loss**: Scale-Invariant SDR loss ensures the model focuses on quality, not just amplitude reduction
- **Energy Conservation Loss**: Penalizes the model if output energy differs too much from target
- **Global Normalization**: Uses dataset-wide mean/std instead of per-file normalization to avoid inconsistent training
- **Amplitude Matching**: Post-processing to ensure output has similar amplitude to input

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
  normalization:
    type: "global"  # Use dataset-wide statistics

stft:
  n_fft: 512
  hop_length: 128
  win_length: 512

model:
  name: "UNetDenoiser"
  encoder_channels: [32, 64, 128, 256, 512]
  use_attention: true
  dropout: 0.1
  mask_type: "CRM"  # Complex Ratio Mask

training:
  batch_size: 16
  num_epochs: 150  # Train longer for better results
  learning_rate: 0.0001
  early_stopping:
    patience: 25
    restore_best_weights: true

# Loss weights - Important for preventing lazy learning
loss:
  l1_weight: 1.0
  magnitude_weight: 1.0
  si_sdr_weight: 0.5      # Scale-Invariant SDR (anti lazy-learning)
  energy_weight: 0.1      # Energy conservation
  perceptual_weight: 0.2  # Mel-spectrogram loss
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

## Troubleshooting

### Problem: Stuck at `[3/4]` after `assign=True` while loading a checkpoint

If the app appears to freeze at the weights loading step (often shown as `[3/4]`), it's almost always **RAM swap/thrash**: `load_state_dict` must materialize many tensors, and on low-RAM machines it can look like it hangs forever.

**Do this first (most effective): convert to a smaller "weights-only" checkpoint** (drops optimizer state, keeps config if available), then load the new file in the app:

```bash
# Linux/macOS
python3 convert_checkpoint.py --input best_model.pt --output best_model_weights.pt --keep-config
```

On Windows (CMD/PowerShell) you typically don't have `python3`, use:

```bash
python convert_checkpoint.py --input best_model.pt --output best_model_weights.pt --keep-config
```

If `python` is not recognized, try:

```bash
py -3 convert_checkpoint.py --input best_model.pt --output best_model_weights.pt --keep-config
```

Quick check which one works:

```bash
py --version
python --version
```

**If it's still stuck:**

- **Check RAM and Disk/Swap** while it's stuck (Task Manager / Resource Monitor). If RAM is ~90–100% and Disk/Swap is high, it's swap-thrash: close other apps, use a machine with more RAM, or reduce checkpoint size.
- **Use local SSD storage** for the checkpoint (avoid network drives like `K:/`). Network drives can make checkpoint reads and tensor materialization behave "unreasonably" slow.

### Problem: Output audio is too quiet / volume is reduced

This is a common issue with denoising models ("lazy learning"). Solutions:

1. **Check SI-SDR loss weight** in `config.yaml`:
   ```yaml
   loss:
     si_sdr_weight: 0.5  # Increase if volume is still reduced
     energy_weight: 0.1  # Helps preserve energy
   ```

2. **Enable amplitude matching** during inference:
   ```bash
   python inference.py --input noisy.wav --output clean.wav --checkpoint best_model.pt --match_amplitude
   ```

3. **Check normalization**: The model uses global normalization (dataset-wide mean/std). If using a custom dataset, ensure the normalizer stats are saved and loaded correctly.

### Problem: Training stops too early (EarlyStopping)

Increase the patience in `config.yaml`:
```yaml
training:
  early_stopping:
    patience: 25  # Increase from default 15
    restore_best_weights: true  # Important!
```

### Problem: Output audio sounds distorted/muffled

1. **Check phase reconstruction**: The model preserves phase from input STFT. Ensure STFT parameters match between training and inference.

2. **Try different mask type**:
   ```yaml
   model:
     mask_type: "CRM"  # Try "IRM" or "direct" if CRM doesn't work well
   ```

3. **Train for more epochs**: The model may need more training for better quality.

### Problem: Inconsistent results between files

This is often due to per-file normalization. The project now uses global normalization to avoid this issue. Make sure you're using the latest version.

## Notes

- **PESQ Installation**: PESQ requires C compilation. On Windows, install Microsoft Visual C++ Build Tools. The system works without PESQ if unavailable.
- **GPU Training**: Recommended for faster training. Enable with CUDA-compatible GPU.
- **Training Time**: ~2-4 hours on CPU, significantly faster on GPU.

## Technical Improvements

This implementation includes several improvements over basic denoising models:

1. **Global Normalization**: Uses dataset-wide statistics (mean=0, std=1) instead of per-file normalization, following LeCun's recommendations for training stability.

2. **SI-SDR Loss**: Scale-Invariant SDR loss that focuses on signal quality regardless of amplitude. This prevents the model from "cheating" by just reducing volume.

3. **Energy Conservation Loss**: Ensures output energy is within 60-140% of target energy, preventing excessive volume changes.

4. **Improved EarlyStopping**: 
   - Higher patience (25 epochs instead of 15)
   - Minimum delta threshold for improvement detection
   - `restore_best_weights=True` to restore the best model when training stops

5. **Proper Output Post-processing**:
   - Denormalization to restore original amplitude scale
   - Amplitude matching with input
   - Anti-clipping processing

## License

MIT License
