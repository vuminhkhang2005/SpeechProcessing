# Speech Denoising - Khử Nhiễu Tiếng Nói

## 🎵 Giới Thiệu

Đây là dự án khử nhiễu tiếng nói (Speech Denoising) sử dụng Deep Learning. Mô hình được thiết kế để loại bỏ tiếng ồn nền và giữ lại giọng nói, tương tự như tính năng khử nhiễu của Discord.

### ✨ Đặc điểm:
- **🖥️ Giao diện đồ họa (GUI)**: Ứng dụng hoàn chỉnh với tkinter, dễ sử dụng
- **🎧 Phát audio trực tiếp**: Nghe và so sánh trước/sau khử nhiễu
- **📊 Visualization**: Hiển thị spectrogram để so sánh
- **📁 Batch Processing**: Xử lý nhiều file cùng lúc
- **Kiến trúc U-Net**: Sử dụng encoder-decoder với skip connections để bảo toàn chi tiết
- **Complex Ratio Mask (CRM)**: Xử lý cả magnitude và phase của tín hiệu
- **Multi-Resolution STFT Loss**: Đảm bảo chất lượng âm thanh ở nhiều tần số
- **Attention Mechanism**: Tăng khả năng xử lý dependencies dài

## Yêu Cầu Hệ Thống

- Python 3.8+
- PyTorch 2.0+
- CUDA (khuyến nghị cho training)
- RAM: 8GB+
- GPU VRAM: 4GB+ (cho training)

## Cài Đặt

### 1. Clone repository và cài đặt dependencies

```bash
cd speech_denoising
pip install -r requirements.txt
```

### 2. (Tùy chọn) Cài đặt PESQ metric

PESQ là một metric đánh giá chất lượng giọng nói, nhưng yêu cầu C compiler để cài đặt.

**Linux/macOS:**
```bash
pip install pesq
```

**Windows:**
PESQ yêu cầu Microsoft Visual C++ Build Tools. Nếu gặp lỗi khi cài đặt:
1. Tải và cài đặt [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Trong Visual Studio Installer, chọn "Desktop development with C++"
3. Sau đó chạy: `pip install pesq`

> **Lưu ý**: Dự án vẫn hoạt động bình thường mà không cần PESQ. Các metrics khác (STOI, SI-SDR, SNR) sẽ vẫn được tính toán.

### 3. Download VoiceBank + DEMAND Dataset

Dataset có thể download từ: https://datashare.ed.ac.uk/handle/10283/2791

Download các file sau:
- `clean_trainset_28spk_wav.zip`
- `noisy_trainset_28spk_wav.zip`
- `clean_testset_wav.zip`
- `noisy_testset_wav.zip`

Giải nén vào thư mục `data/`:

```
speech_denoising/
├── data/
│   ├── clean_trainset_28spk_wav/   # ~11,572 files
│   ├── noisy_trainset_28spk_wav/   # ~11,572 files
│   ├── clean_testset_wav/          # ~824 files
│   └── noisy_testset_wav/          # ~824 files
```

## Cấu Trúc Dự Án

```
speech_denoising/
├── app.py                   # 🖥️ GUI Application (CHÍNH)
├── run_app.py               # Script launcher cho GUI
├── config.yaml              # Cấu hình training
├── requirements.txt         # Dependencies
├── train.py                 # Script training
├── inference.py             # Script inference
├── evaluate.py              # Script đánh giá
├── demo.py                  # Demo script
├── data/
│   ├── __init__.py
│   └── dataset.py           # Dataset loader
├── models/
│   ├── __init__.py
│   ├── unet.py              # Kiến trúc U-Net
│   └── loss.py              # Loss functions
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py       # Xử lý audio
│   └── metrics.py           # Metrics (PESQ, STOI, etc.)
├── checkpoints/             # Model checkpoints
├── logs/                    # TensorBoard logs
└── outputs/                 # Output audio files
```

## 🖥️ Sử Dụng Ứng Dụng GUI (Khuyến Nghị)

### Khởi động ứng dụng

```bash
cd speech_denoising
python app.py
```

Hoặc sử dụng launcher script:

```bash
python run_app.py
```

### Giao diện ứng dụng

Ứng dụng có 4 tab chính:

#### 1. 🎤 Tab Khử Nhiễu (Denoise)
- Chọn file audio đầu vào (WAV, MP3, FLAC, OGG)
- Chọn đường dẫn file output
- Bấm **"Khử Nhiễu"** để xử lý
- Nghe và so sánh audio trước/sau
- Xem spectrogram để so sánh trực quan

#### 2. 📁 Tab Xử Lý Hàng Loạt (Batch Processing)
- Chọn thư mục chứa các file audio nhiễu
- Chọn thư mục đầu ra
- Bấm **"Quét File"** để tìm tất cả audio files
- Bấm **"Xử Lý Tất Cả"** để khử nhiễu hàng loạt

#### 3. 🎓 Tab Huấn Luyện (Training)
- Cấu hình đường dẫn dataset
- Điều chỉnh tham số training
- Theo dõi tiến trình training

#### 4. ⚙️ Tab Cài Đặt (Settings)
- **Quan trọng**: Tải model trước khi sử dụng!
- Chọn file checkpoint (.pt)
- Chọn device (CPU/CUDA)
- Xem thông tin hệ thống

### Yêu cầu thêm cho GUI

```bash
# Cài đặt sounddevice để phát audio
pip install sounddevice

# Linux: Cần cài PortAudio
sudo apt-get install libportaudio2

# Cài tkinter nếu chưa có
# Ubuntu/Debian:
sudo apt-get install python3-tk
# Fedora:
sudo dnf install python3-tkinter
```

### Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│ 🎵 Speech Denoising Application                              │
├─────────────────────────────────────────────────────────────┤
│ [🎤 Khử Nhiễu] [📁 Xử Lý Hàng Loạt] [🎓 Huấn Luyện] [⚙️ Cài Đặt] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📂 File Đầu Vào                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ /path/to/noisy_audio.wav              [Chọn...]      │  │
│  └──────────────────────────────────────────────────────┘  │
│  [▶ Phát Input] [⏹ Dừng]                                    │
│                                                             │
│  💾 File Đầu Ra                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ /path/to/clean_audio.wav              [Chọn...]      │  │
│  └──────────────────────────────────────────────────────┘  │
│  [▶ Phát Output]                                            │
│                                                             │
│  ⚡ Xử Lý                                                    │
│  [🎵 Khử Nhiễu] ████████████████████ Sẵn sàng               │
│                                                             │
│  📊 So Sánh Spectrogram                                      │
│  ┌────────────────────────┐ ┌────────────────────────┐     │
│  │                        │ │                        │     │
│  │   Input (Nhiễu)        │ │  Output (Đã khử nhiễu) │     │
│  │                        │ │                        │     │
│  └────────────────────────┘ └────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Sử Dụng Command Line

Nếu bạn muốn sử dụng từ dòng lệnh thay vì GUI:

## Training

### Cách 1: Sử dụng config mặc định

```bash
python train.py --config config.yaml
```

### Cách 2: Tùy chỉnh tham số

```bash
python train.py \
    --config config.yaml \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.0001
```

### Cách 3: Resume training từ checkpoint

```bash
python train.py \
    --config config.yaml \
    --resume checkpoints/checkpoint_epoch_50.pt
```

### Theo dõi training với TensorBoard

```bash
tensorboard --logdir logs/
```

## Inference (Khử Nhiễu)

### Khử nhiễu một file audio

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input path/to/noisy_audio.wav \
    --output path/to/clean_audio.wav
```

### Khử nhiễu nhiều file

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input_dir path/to/noisy_folder/ \
    --output_dir path/to/clean_folder/
```

## Đánh Giá (Evaluation)

### Đánh giá trên test set

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.yaml \
    --save_results results.csv
```

### Lưu audio đã khử nhiễu

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --save_samples \
    --output_dir outputs/
```

## Metrics Đánh Giá

| Metric | Mô tả | Phạm vi |
|--------|-------|---------|
| **PESQ** | Perceptual Evaluation of Speech Quality | -0.5 đến 4.5 (cao = tốt) |
| **STOI** | Short-Time Objective Intelligibility | 0 đến 1 (cao = tốt) |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | dB (cao = tốt) |
| **SNR** | Signal-to-Noise Ratio | dB (cao = tốt) |

## Kiến Trúc Model

### U-Net Denoiser

```
Input (Noisy STFT) → Encoder → Bottleneck (+ Attention) → Decoder → Output (Mask)
        ↓                                                    ↑
        └──────────────── Skip Connections ──────────────────┘
```

- **Input**: Complex STFT [batch, 2, freq, time]
- **Encoder**: 5 blocks với channels [32, 64, 128, 256, 512]
- **Attention**: Self-attention ở bottleneck
- **Decoder**: 5 blocks với skip connections
- **Output**: Complex Ratio Mask (CRM)

### Loss Function

```
Total Loss = λ₁ × Complex L1 + λ₂ × Magnitude L1 + λ₃ × Multi-Resolution STFT
```

## Cấu Hình (config.yaml)

```yaml
# Tham số audio
data:
  sample_rate: 16000
  segment_length: 32000  # 2 giây

# STFT parameters
stft:
  n_fft: 512
  hop_length: 128
  win_length: 512

# Training
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
```

## Tips & Tricks

### 1. Tăng tốc training
- Sử dụng `use_amp: true` (Mixed Precision Training)
- Tăng `num_workers` nếu có nhiều CPU cores
- Sử dụng SSD để load data nhanh hơn

### 2. Xử lý audio dài
- Sử dụng `--chunk_size` trong inference để xử lý audio dài
- Mặc định: 160000 samples (10 giây ở 16kHz)

### 3. Cải thiện kết quả
- Train lâu hơn (100+ epochs)
- Augmentation data (thêm noise types)
- Fine-tune learning rate

### 4. Debug
- Kiểm tra data với `python -c "from data.dataset import VoiceBankDEMANDDataset; ..."`
- Xem TensorBoard để theo dõi loss

## Kết Quả Tham Khảo

Trên VoiceBank + DEMAND test set (với config mặc định, 100 epochs):

| Metric | Noisy (Baseline) | Enhanced | Improvement |
|--------|------------------|----------|-------------|
| PESQ | ~1.97 | ~2.80+ | +0.83 |
| STOI | ~0.92 | ~0.95+ | +0.03 |
| SI-SDR | ~8.4 dB | ~17+ dB | +8.6 dB |

*Kết quả có thể thay đổi tùy thuộc vào hyperparameters và thời gian training.*

## Tài Liệu Tham Khảo

1. [VoiceBank+DEMAND Dataset](https://datashare.ed.ac.uk/handle/10283/2791)
2. [U-Net for Speech Enhancement](https://arxiv.org/abs/1806.07853)
3. [Complex Ratio Mask](https://ieeexplore.ieee.org/document/8462375)
4. [Multi-Resolution STFT Loss](https://arxiv.org/abs/1910.11480)

## License

MIT License

## Liên Hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue hoặc pull request.
