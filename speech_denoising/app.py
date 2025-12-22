"""
Speech Denoising Application - GUI với Tkinter

Ứng dụng hoàn chỉnh để khử nhiễu giọng nói với giao diện đồ họa.

Tính năng:
- Khử nhiễu file audio đơn lẻ hoặc nhiều file
- So sánh trước/sau khi khử nhiễu
- Xem spectrogram
- Phát audio trực tiếp
- Huấn luyện model
- Cài đặt tham số

Usage:
    python app.py
"""

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import time
from typing import Optional, Callable
import json

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
try:
    from models.unet import UNetDenoiser
    from utils.audio_utils import AudioProcessor, load_audio, save_audio
    from inference import SpeechDenoiser
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the speech_denoising directory")
    sys.exit(1)

# Optional imports for audio playback
try:
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    print("Warning: sounddevice not installed. Audio playback will be disabled.")
    print("Install with: pip install sounddevice")

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import librosa
    import librosa.display
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/librosa not fully available. Visualization may be limited.")


class ModernStyle:
    """Modern styling constants for the application"""
    
    # Colors
    BG_PRIMARY = "#1e1e2e"  # Dark background
    BG_SECONDARY = "#2d2d44"  # Slightly lighter
    BG_TERTIARY = "#3d3d5c"  # Accent background
    
    FG_PRIMARY = "#ffffff"  # White text
    FG_SECONDARY = "#b4b4b4"  # Gray text
    
    ACCENT = "#7c3aed"  # Purple accent
    ACCENT_HOVER = "#8b5cf6"
    ACCENT_LIGHT = "#a78bfa"
    
    SUCCESS = "#22c55e"  # Green
    WARNING = "#f59e0b"  # Orange
    ERROR = "#ef4444"  # Red
    INFO = "#3b82f6"  # Blue
    
    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_LARGE = 14
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_SMALL = 9
    
    # Padding
    PAD_SMALL = 5
    PAD_MEDIUM = 10
    PAD_LARGE = 20


class AudioPlayer:
    """Simple audio player using sounddevice"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.current_audio = None
        self.play_thread = None
        
    def play(self, audio: np.ndarray, callback: Optional[Callable] = None):
        """Play audio asynchronously"""
        if not AUDIO_PLAYBACK_AVAILABLE:
            messagebox.showwarning("Cảnh báo", "Không thể phát audio. Cài đặt sounddevice: pip install sounddevice")
            return
        
        self.stop()
        self.current_audio = audio
        self.is_playing = True
        
        def _play():
            try:
                sd.play(audio, self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self.is_playing = False
                if callback:
                    callback()
        
        self.play_thread = threading.Thread(target=_play, daemon=True)
        self.play_thread.start()
    
    def stop(self):
        """Stop current playback"""
        if AUDIO_PLAYBACK_AVAILABLE:
            sd.stop()
        self.is_playing = False


class DenoiseTab(ttk.Frame):
    """Tab for single file denoising"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.audio_player = AudioPlayer()
        
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Sẵn sàng")
        
        self.input_audio = None
        self.output_audio = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Input Section ===
        input_frame = ttk.LabelFrame(main_frame, text="📂 File Đầu Vào", padding=ModernStyle.PAD_MEDIUM)
        input_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Input file selection
        input_row = ttk.Frame(input_frame)
        input_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(input_row, text="File audio nhiễu:").pack(side=tk.LEFT)
        ttk.Entry(input_row, textvariable=self.input_path, width=50).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(input_row, text="Chọn...", command=self._browse_input).pack(side=tk.LEFT)
        
        # Input playback controls
        input_controls = ttk.Frame(input_frame)
        input_controls.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        self.play_input_btn = ttk.Button(input_controls, text="▶ Phát Input", command=self._play_input)
        self.play_input_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        ttk.Button(input_controls, text="⏹ Dừng", command=self._stop_audio).pack(side=tk.LEFT)
        
        self.input_info_label = ttk.Label(input_controls, text="")
        self.input_info_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        # === Output Section ===
        output_frame = ttk.LabelFrame(main_frame, text="💾 File Đầu Ra", padding=ModernStyle.PAD_MEDIUM)
        output_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Output file selection
        output_row = ttk.Frame(output_frame)
        output_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(output_row, text="File audio sạch:").pack(side=tk.LEFT)
        ttk.Entry(output_row, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(output_row, text="Chọn...", command=self._browse_output).pack(side=tk.LEFT)
        
        # Output playback controls
        output_controls = ttk.Frame(output_frame)
        output_controls.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        self.play_output_btn = ttk.Button(output_controls, text="▶ Phát Output", command=self._play_output, state=tk.DISABLED)
        self.play_output_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.output_info_label = ttk.Label(output_controls, text="")
        self.output_info_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        # === Process Section ===
        process_frame = ttk.LabelFrame(main_frame, text="⚡ Xử Lý", padding=ModernStyle.PAD_MEDIUM)
        process_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Process button
        btn_frame = ttk.Frame(process_frame)
        btn_frame.pack(fill=tk.X)
        
        self.process_btn = ttk.Button(btn_frame, text="🎵 Khử Nhiễu", command=self._process, style="Accent.TButton")
        self.process_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.progress = ttk.Progressbar(btn_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        ttk.Label(btn_frame, textvariable=self.status_text).pack(side=tk.LEFT)
        
        # === Visualization Section ===
        if VISUALIZATION_AVAILABLE:
            viz_frame = ttk.LabelFrame(main_frame, text="📊 So Sánh Spectrogram", padding=ModernStyle.PAD_MEDIUM)
            viz_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create matplotlib figure
            self.fig = Figure(figsize=(10, 4), dpi=100, facecolor=ModernStyle.BG_SECONDARY)
            self.fig.patch.set_facecolor('#2d2d44')
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Toolbar
            toolbar_frame = ttk.Frame(viz_frame)
            toolbar_frame.pack(fill=tk.X)
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            self.toolbar.update()
    
    def _browse_input(self):
        """Browse for input file"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
            ("WAV files", "*.wav"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.input_path.set(path)
            self._load_input(path)
            
            # Auto-generate output path
            base = Path(path)
            output = base.parent / f"{base.stem}_denoised{base.suffix}"
            self.output_path.set(str(output))
    
    def _browse_output(self):
        """Browse for output file"""
        filetypes = [("WAV files", "*.wav")]
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".wav")
        if path:
            self.output_path.set(path)
    
    def _load_input(self, path: str):
        """Load input audio file"""
        try:
            self.input_audio, sr = load_audio(path, sample_rate=16000)
            if isinstance(self.input_audio, torch.Tensor):
                self.input_audio = self.input_audio.numpy()
            
            duration = len(self.input_audio) / 16000
            self.input_info_label.config(text=f"Thời lượng: {duration:.2f}s | Sample rate: 16kHz")
            self.status_text.set("Đã tải file input")
            
            # Update visualization
            self._update_visualization(input_only=True)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải file audio: {e}")
    
    def _play_input(self):
        """Play input audio"""
        if self.input_audio is not None:
            self.audio_player.play(self.input_audio, callback=lambda: self.play_input_btn.config(text="▶ Phát Input"))
            self.play_input_btn.config(text="🔊 Đang phát...")
    
    def _play_output(self):
        """Play output audio"""
        if self.output_audio is not None:
            self.audio_player.play(self.output_audio, callback=lambda: self.play_output_btn.config(text="▶ Phát Output"))
            self.play_output_btn.config(text="🔊 Đang phát...")
    
    def _stop_audio(self):
        """Stop audio playback"""
        self.audio_player.stop()
        self.play_input_btn.config(text="▶ Phát Input")
        self.play_output_btn.config(text="▶ Phát Output")
    
    def _process(self):
        """Run denoising"""
        if not self.input_path.get():
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file audio đầu vào")
            return
        
        if not self.app.denoiser:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải model trước (tab Cài Đặt)")
            return
        
        # Disable button and start progress
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_text.set("Đang xử lý...")
        
        # Run in thread
        def _run():
            try:
                result = self.app.denoiser.denoise_file(
                    self.input_path.get(),
                    self.output_path.get()
                )
                
                # Load output for playback
                self.output_audio, _ = load_audio(self.output_path.get(), sample_rate=16000)
                if isinstance(self.output_audio, torch.Tensor):
                    self.output_audio = self.output_audio.numpy()
                
                # Update UI in main thread
                self.after(0, lambda: self._process_complete(result))
                
            except Exception as e:
                self.after(0, lambda: self._process_error(str(e)))
        
        threading.Thread(target=_run, daemon=True).start()
    
    def _process_complete(self, result: dict):
        """Handle process completion"""
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        self.play_output_btn.config(state=tk.NORMAL)
        
        self.status_text.set(f"✅ Hoàn thành! Thời gian: {result['processing_time']:.2f}s | RTF: {result['rtf']:.2f}x")
        self.output_info_label.config(text=f"Đã lưu: {result['output_path']}")
        
        # Update visualization
        self._update_visualization()
        
        messagebox.showinfo("Thành công", f"Đã khử nhiễu thành công!\n\nFile output: {result['output_path']}\nThời gian xử lý: {result['processing_time']:.2f}s")
    
    def _process_error(self, error: str):
        """Handle process error"""
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        self.status_text.set(f"❌ Lỗi: {error}")
        messagebox.showerror("Lỗi", f"Không thể xử lý: {error}")
    
    def _update_visualization(self, input_only: bool = False):
        """Update spectrogram visualization"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        self.fig.clear()
        
        if input_only and self.input_audio is not None:
            ax = self.fig.add_subplot(111)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(self.input_audio)), ref=np.max)
            librosa.display.specshow(D, sr=16000, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
            ax.set_title('Input (Nhiễu)', color='white', fontsize=12)
            ax.tick_params(colors='white')
            
        elif self.input_audio is not None and self.output_audio is not None:
            # Two spectrograms side by side
            ax1 = self.fig.add_subplot(121)
            D1 = librosa.amplitude_to_db(np.abs(librosa.stft(self.input_audio)), ref=np.max)
            librosa.display.specshow(D1, sr=16000, x_axis='time', y_axis='hz', ax=ax1, cmap='magma')
            ax1.set_title('Input (Nhiễu)', color='white', fontsize=12)
            ax1.tick_params(colors='white')
            
            ax2 = self.fig.add_subplot(122)
            D2 = librosa.amplitude_to_db(np.abs(librosa.stft(self.output_audio)), ref=np.max)
            librosa.display.specshow(D2, sr=16000, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
            ax2.set_title('Output (Đã khử nhiễu)', color='white', fontsize=12)
            ax2.tick_params(colors='white')
        
        self.fig.tight_layout()
        self.canvas.draw()


class BatchTab(ttk.Frame):
    """Tab for batch processing multiple files"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.file_list = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Directory Selection ===
        dir_frame = ttk.LabelFrame(main_frame, text="📁 Thư Mục", padding=ModernStyle.PAD_MEDIUM)
        dir_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Input directory
        input_row = ttk.Frame(dir_frame)
        input_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(input_row, text="Thư mục input:", width=15).pack(side=tk.LEFT)
        ttk.Entry(input_row, textvariable=self.input_dir, width=50).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(input_row, text="Chọn...", command=self._browse_input_dir).pack(side=tk.LEFT)
        
        # Output directory
        output_row = ttk.Frame(dir_frame)
        output_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(output_row, text="Thư mục output:", width=15).pack(side=tk.LEFT)
        ttk.Entry(output_row, textvariable=self.output_dir, width=50).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(output_row, text="Chọn...", command=self._browse_output_dir).pack(side=tk.LEFT)
        
        # === File List ===
        list_frame = ttk.LabelFrame(main_frame, text="📋 Danh Sách File", padding=ModernStyle.PAD_MEDIUM)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Treeview for file list
        columns = ("filename", "duration", "status")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        self.tree.heading("filename", text="Tên File")
        self.tree.heading("duration", text="Thời Lượng")
        self.tree.heading("status", text="Trạng Thái")
        
        self.tree.column("filename", width=300)
        self.tree.column("duration", width=100)
        self.tree.column("status", width=150)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === Controls ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)
        
        self.scan_btn = ttk.Button(control_frame, text="🔍 Quét File", command=self._scan_files)
        self.scan_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.process_btn = ttk.Button(control_frame, text="🚀 Xử Lý Tất Cả", command=self._process_all, style="Accent.TButton")
        self.process_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.progress = ttk.Progressbar(control_frame, mode='determinate', length=300)
        self.progress.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        self.status_label = ttk.Label(control_frame, text="")
        self.status_label.pack(side=tk.LEFT)
    
    def _browse_input_dir(self):
        """Browse for input directory"""
        path = filedialog.askdirectory()
        if path:
            self.input_dir.set(path)
            # Auto-generate output path
            self.output_dir.set(path + "_denoised")
            self._scan_files()
    
    def _browse_output_dir(self):
        """Browse for output directory"""
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)
    
    def _scan_files(self):
        """Scan input directory for audio files"""
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục input hợp lệ")
            return
        
        # Clear current list
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.file_list.clear()
        
        # Find audio files
        extensions = ['.wav', '.mp3', '.flac', '.ogg']
        for file in Path(input_dir).iterdir():
            if file.suffix.lower() in extensions:
                self.file_list.append(file)
                
                # Get duration
                try:
                    audio, _ = load_audio(str(file), sample_rate=16000)
                    if isinstance(audio, torch.Tensor):
                        duration = len(audio) / 16000
                    else:
                        duration = len(audio) / 16000
                    duration_str = f"{duration:.2f}s"
                except:
                    duration_str = "N/A"
                
                self.tree.insert("", tk.END, values=(file.name, duration_str, "Chờ xử lý"))
        
        self.status_label.config(text=f"Tìm thấy {len(self.file_list)} file")
    
    def _process_all(self):
        """Process all files"""
        if not self.file_list:
            messagebox.showwarning("Cảnh báo", "Không có file nào để xử lý")
            return
        
        if not self.app.denoiser:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải model trước (tab Cài Đặt)")
            return
        
        output_dir = Path(self.output_dir.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.process_btn.config(state=tk.DISABLED)
        self.progress['maximum'] = len(self.file_list)
        self.progress['value'] = 0
        
        def _run():
            for i, file in enumerate(self.file_list):
                try:
                    # Update status
                    self.after(0, lambda f=file, idx=i: self._update_file_status(idx, "Đang xử lý..."))
                    
                    # Process
                    output_path = output_dir / file.name
                    self.app.denoiser.denoise_file(str(file), str(output_path))
                    
                    # Update status
                    self.after(0, lambda idx=i: self._update_file_status(idx, "✅ Hoàn thành"))
                    
                except Exception as e:
                    self.after(0, lambda idx=i, err=str(e): self._update_file_status(idx, f"❌ {err}"))
                
                # Update progress
                self.after(0, lambda v=i+1: self.progress.configure(value=v))
            
            # Done
            self.after(0, self._process_complete)
        
        threading.Thread(target=_run, daemon=True).start()
    
    def _update_file_status(self, index: int, status: str):
        """Update status of a file in the tree"""
        items = self.tree.get_children()
        if index < len(items):
            values = self.tree.item(items[index])['values']
            self.tree.item(items[index], values=(values[0], values[1], status))
    
    def _process_complete(self):
        """Handle batch processing complete"""
        self.process_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Đã xử lý xong {len(self.file_list)} file")
        messagebox.showinfo("Thành công", f"Đã xử lý xong {len(self.file_list)} file!\n\nOutput: {self.output_dir.get()}")


class TrainTab(ttk.Frame):
    """Tab for model training"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.is_training = False
        self.training_thread = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Dataset Section ===
        dataset_frame = ttk.LabelFrame(main_frame, text="📚 Dataset", padding=ModernStyle.PAD_MEDIUM)
        dataset_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        self.train_clean_dir = tk.StringVar(value="./data/clean_trainset_28spk_wav")
        self.train_noisy_dir = tk.StringVar(value="./data/noisy_trainset_28spk_wav")
        self.test_clean_dir = tk.StringVar(value="./data/clean_testset_wav")
        self.test_noisy_dir = tk.StringVar(value="./data/noisy_testset_wav")
        
        dirs = [
            ("Train Clean:", self.train_clean_dir),
            ("Train Noisy:", self.train_noisy_dir),
            ("Test Clean:", self.test_clean_dir),
            ("Test Noisy:", self.test_noisy_dir),
        ]
        
        for label, var in dirs:
            row = ttk.Frame(dataset_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=50).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
            ttk.Button(row, text="...", width=3, command=lambda v=var: self._browse_dir(v)).pack(side=tk.LEFT)
        
        # === Training Parameters ===
        param_frame = ttk.LabelFrame(main_frame, text="⚙️ Tham Số Training", padding=ModernStyle.PAD_MEDIUM)
        param_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        params_grid = ttk.Frame(param_frame)
        params_grid.pack(fill=tk.X)
        
        # Row 1
        row1 = ttk.Frame(params_grid)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size = tk.StringVar(value="16")
        ttk.Entry(row1, textvariable=self.batch_size, width=10).pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1, text="Epochs:").pack(side=tk.LEFT)
        self.epochs = tk.StringVar(value="100")
        ttk.Entry(row1, textvariable=self.epochs, width=10).pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(row1, text="Learning Rate:").pack(side=tk.LEFT)
        self.learning_rate = tk.StringVar(value="0.0001")
        ttk.Entry(row1, textvariable=self.learning_rate, width=10).pack(side=tk.LEFT, padx=5)
        
        # === Training Controls ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=ModernStyle.PAD_MEDIUM)
        
        self.train_btn = ttk.Button(control_frame, text="🚀 Bắt Đầu Training", command=self._toggle_training, style="Accent.TButton")
        self.train_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.progress = ttk.Progressbar(control_frame, mode='determinate', length=300)
        self.progress.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        # === Log Output ===
        log_frame = ttk.LabelFrame(main_frame, text="📝 Log", padding=ModernStyle.PAD_MEDIUM)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=15, bg='#1e1e2e', fg='white', insertbackground='white')
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _browse_dir(self, var: tk.StringVar):
        """Browse for directory"""
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    def _log(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def _toggle_training(self):
        """Start or stop training"""
        if self.is_training:
            self.is_training = False
            self.train_btn.config(text="🚀 Bắt Đầu Training")
            self._log("⏹ Đã dừng training")
        else:
            self._start_training()
    
    def _start_training(self):
        """Start training"""
        self.is_training = True
        self.train_btn.config(text="⏹ Dừng Training")
        self._log("=" * 50)
        self._log("🚀 Bắt đầu training...")
        
        # Note: Full training implementation would go here
        # For now, show a message that training should be done via command line
        self._log("")
        self._log("⚠️ Training đầy đủ nên được chạy từ command line:")
        self._log("   python train.py --config config.yaml")
        self._log("")
        self._log("Để training từ GUI, bạn cần:")
        self._log("1. Tải dataset VoiceBank + DEMAND")
        self._log("2. Cấu hình đường dẫn dataset")
        self._log("3. Có GPU với đủ VRAM (khuyến nghị >= 8GB)")
        self._log("")
        
        self.is_training = False
        self.train_btn.config(text="🚀 Bắt Đầu Training")


class SettingsTab(ttk.Frame):
    """Tab for application settings"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        self.checkpoint_path = tk.StringVar()
        self.device = tk.StringVar(value="auto")
        
        self._create_widgets()
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Model Section ===
        model_frame = ttk.LabelFrame(main_frame, text="🧠 Model", padding=ModernStyle.PAD_MEDIUM)
        model_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Checkpoint selection
        ckpt_row = ttk.Frame(model_frame)
        ckpt_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(ckpt_row, text="Checkpoint:").pack(side=tk.LEFT)
        ttk.Entry(ckpt_row, textvariable=self.checkpoint_path, width=50).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(ckpt_row, text="Chọn...", command=self._browse_checkpoint).pack(side=tk.LEFT)
        
        # Device selection
        device_row = ttk.Frame(model_frame)
        device_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(device_row, text="Device:").pack(side=tk.LEFT)
        device_combo = ttk.Combobox(device_row, textvariable=self.device, values=["auto", "cuda", "cpu"], width=15)
        device_combo.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        # Load button
        btn_row = ttk.Frame(model_frame)
        btn_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        self.load_btn = ttk.Button(btn_row, text="📥 Tải Model", command=self._load_model, style="Accent.TButton")
        self.load_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.model_status = ttk.Label(btn_row, text="Chưa tải model")
        self.model_status.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        # === System Info ===
        info_frame = ttk.LabelFrame(main_frame, text="💻 Thông Tin Hệ Thống", padding=ModernStyle.PAD_MEDIUM)
        info_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # System info
        cuda_available = torch.cuda.is_available()
        cuda_text = f"✅ CUDA khả dụng - GPU: {torch.cuda.get_device_name(0)}" if cuda_available else "❌ CUDA không khả dụng"
        
        ttk.Label(info_frame, text=f"PyTorch: {torch.__version__}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=cuda_text).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Audio playback: {'✅ Khả dụng' if AUDIO_PLAYBACK_AVAILABLE else '❌ Không khả dụng'}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Visualization: {'✅ Khả dụng' if VISUALIZATION_AVAILABLE else '❌ Không khả dụng'}").pack(anchor=tk.W)
        
        # === About ===
        about_frame = ttk.LabelFrame(main_frame, text="ℹ️ Về Ứng Dụng", padding=ModernStyle.PAD_MEDIUM)
        about_frame.pack(fill=tk.X)
        
        about_text = """Speech Denoising Application
        
Ứng dụng khử nhiễu giọng nói sử dụng Deep Learning.

Tính năng:
• Khử nhiễu file audio đơn lẻ hoặc nhiều file
• So sánh trước/sau khi khử nhiễu
• Xem spectrogram
• Phát audio trực tiếp

Developed with PyTorch and Tkinter"""
        
        ttk.Label(about_frame, text=about_text, justify=tk.LEFT).pack(anchor=tk.W)
    
    def _browse_checkpoint(self):
        """Browse for checkpoint file"""
        filetypes = [
            ("PyTorch checkpoint", "*.pt *.pth"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.checkpoint_path.set(path)
    
    def _load_model(self):
        """Load model from checkpoint"""
        ckpt_path = self.checkpoint_path.get()
        
        if not ckpt_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file checkpoint")
            return
        
        if not os.path.exists(ckpt_path):
            messagebox.showerror("Lỗi", f"File không tồn tại: {ckpt_path}")
            return
        
        self.load_btn.config(state=tk.DISABLED)
        self.model_status.config(text="Đang tải...")
        
        def _load():
            try:
                device = None if self.device.get() == "auto" else self.device.get()
                self.app.denoiser = SpeechDenoiser(
                    checkpoint_path=ckpt_path,
                    device=device
                )
                self.after(0, lambda: self._load_complete())
            except Exception as e:
                self.after(0, lambda: self._load_error(str(e)))
        
        threading.Thread(target=_load, daemon=True).start()
    
    def _load_complete(self):
        """Handle model load complete"""
        self.load_btn.config(state=tk.NORMAL)
        device = self.app.denoiser.device
        self.model_status.config(text=f"✅ Model đã tải ({device})")
        messagebox.showinfo("Thành công", f"Đã tải model thành công!\nDevice: {device}")
    
    def _load_error(self, error: str):
        """Handle model load error"""
        self.load_btn.config(state=tk.NORMAL)
        self.model_status.config(text=f"❌ Lỗi: {error}")
        messagebox.showerror("Lỗi", f"Không thể tải model: {error}")


class SpeechDenoisingApp:
    """Main application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 Speech Denoising - Khử Nhiễu Giọng Nói")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Initialize denoiser
        self.denoiser = None
        
        # Configure style
        self._configure_style()
        
        # Create UI
        self._create_ui()
        
        # Center window
        self._center_window()
    
    def _configure_style(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        
        # Try to use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        
        # Configure colors
        style.configure(".", 
            background=ModernStyle.BG_SECONDARY,
            foreground=ModernStyle.FG_PRIMARY,
            font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_NORMAL)
        )
        
        style.configure("TFrame", background=ModernStyle.BG_SECONDARY)
        style.configure("TLabel", background=ModernStyle.BG_SECONDARY, foreground=ModernStyle.FG_PRIMARY)
        style.configure("TLabelframe", background=ModernStyle.BG_SECONDARY)
        style.configure("TLabelframe.Label", background=ModernStyle.BG_SECONDARY, foreground=ModernStyle.FG_PRIMARY, font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_NORMAL, 'bold'))
        
        style.configure("TButton",
            background=ModernStyle.BG_TERTIARY,
            foreground=ModernStyle.FG_PRIMARY,
            padding=(10, 5)
        )
        
        style.configure("Accent.TButton",
            background=ModernStyle.ACCENT,
            foreground=ModernStyle.FG_PRIMARY,
            padding=(15, 8),
            font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_NORMAL, 'bold')
        )
        
        style.configure("TNotebook", background=ModernStyle.BG_PRIMARY)
        style.configure("TNotebook.Tab", 
            background=ModernStyle.BG_TERTIARY,
            foreground=ModernStyle.FG_PRIMARY,
            padding=(20, 10),
            font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_NORMAL)
        )
        
        style.configure("Treeview",
            background=ModernStyle.BG_PRIMARY,
            foreground=ModernStyle.FG_PRIMARY,
            fieldbackground=ModernStyle.BG_PRIMARY,
            font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_SMALL)
        )
        style.configure("Treeview.Heading",
            background=ModernStyle.BG_TERTIARY,
            foreground=ModernStyle.FG_PRIMARY,
            font=(ModernStyle.FONT_FAMILY, ModernStyle.FONT_SIZE_SMALL, 'bold')
        )
        
        style.configure("TProgressbar",
            background=ModernStyle.ACCENT,
            troughcolor=ModernStyle.BG_PRIMARY
        )
        
        # Set root background
        self.root.configure(bg=ModernStyle.BG_PRIMARY)
    
    def _create_ui(self):
        """Create main UI"""
        # Header
        header = ttk.Frame(self.root, padding=ModernStyle.PAD_MEDIUM)
        header.pack(fill=tk.X)
        
        title_label = ttk.Label(header, 
            text="🎵 Speech Denoising Application",
            font=(ModernStyle.FONT_FAMILY, 18, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(header,
            text="Khử nhiễu giọng nói bằng Deep Learning",
            font=(ModernStyle.FONT_FAMILY, 10),
            foreground=ModernStyle.FG_SECONDARY
        )
        subtitle_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=ModernStyle.PAD_SMALL, pady=ModernStyle.PAD_SMALL)
        
        # Create tabs
        self.denoise_tab = DenoiseTab(self.notebook, self)
        self.batch_tab = BatchTab(self.notebook, self)
        self.train_tab = TrainTab(self.notebook, self)
        self.settings_tab = SettingsTab(self.notebook, self)
        
        self.notebook.add(self.denoise_tab, text="  🎤 Khử Nhiễu  ")
        self.notebook.add(self.batch_tab, text="  📁 Xử Lý Hàng Loạt  ")
        self.notebook.add(self.train_tab, text="  🎓 Huấn Luyện  ")
        self.notebook.add(self.settings_tab, text="  ⚙️ Cài Đặt  ")
        
        # Status bar
        status_bar = ttk.Frame(self.root, padding=ModernStyle.PAD_SMALL)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_bar, 
            text="Sẵn sàng | Tải model từ tab Cài Đặt để bắt đầu",
            foreground=ModernStyle.FG_SECONDARY
        )
        self.status_label.pack(side=tk.LEFT)
    
    def _center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    print("=" * 60)
    print("🎵 Speech Denoising Application")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Audio playback: {AUDIO_PLAYBACK_AVAILABLE}")
    print(f"  Visualization: {VISUALIZATION_AVAILABLE}")
    print()
    
    # Run app
    app = SpeechDenoisingApp()
    app.run()


if __name__ == '__main__':
    main()
