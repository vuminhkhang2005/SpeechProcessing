"""
Speech Denoising Application - Modern GUI với Tkinter

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
    """Modern styling constants for the application - Clean Dark Theme"""
    
    # Colors - Modern Dark Theme (easier on eyes)
    BG_DARK = "#0f0f0f"        # Darkest - window bg
    BG_PRIMARY = "#1a1a1a"      # Main panels
    BG_SECONDARY = "#242424"    # Cards/frames
    BG_TERTIARY = "#2e2e2e"     # Hover/accent bg
    BG_INPUT = "#1e1e1e"        # Input fields
    
    # Text colors
    FG_PRIMARY = "#ffffff"      # Main text
    FG_SECONDARY = "#a0a0a0"    # Secondary text
    FG_MUTED = "#666666"        # Muted/disabled text
    
    # Accent colors
    ACCENT = "#3b82f6"          # Blue primary accent
    ACCENT_HOVER = "#60a5fa"    # Blue hover
    ACCENT_DARK = "#1d4ed8"     # Blue pressed
    
    # Status colors
    SUCCESS = "#22c55e"         # Green
    SUCCESS_BG = "#052e16"      # Green background
    WARNING = "#f59e0b"         # Orange
    WARNING_BG = "#451a03"      # Orange background
    ERROR = "#ef4444"           # Red
    ERROR_BG = "#450a0a"        # Red background
    INFO = "#3b82f6"            # Blue
    INFO_BG = "#172554"         # Blue background
    
    # Borders
    BORDER = "#333333"
    BORDER_FOCUS = "#3b82f6"
    
    # Fonts - cross-platform
    FONT_FAMILY = ("Segoe UI", "SF Pro Display", "Helvetica Neue", "Arial")
    FONT_SIZE_XL = 18
    FONT_SIZE_LARGE = 14
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_SMALL = 10
    
    # Padding
    PAD_XS = 2
    PAD_SMALL = 5
    PAD_MEDIUM = 10
    PAD_LARGE = 15
    PAD_XL = 20
    
    # Border radius (simulated with relief)
    RELIEF = "flat"
    
    @staticmethod
    def get_font(size="normal", bold=False):
        """Get font tuple for cross-platform compatibility"""
        sizes = {
            "xl": ModernStyle.FONT_SIZE_XL,
            "large": ModernStyle.FONT_SIZE_LARGE, 
            "normal": ModernStyle.FONT_SIZE_NORMAL,
            "small": ModernStyle.FONT_SIZE_SMALL
        }
        weight = "bold" if bold else "normal"
        return (ModernStyle.FONT_FAMILY[0], sizes.get(size, ModernStyle.FONT_SIZE_NORMAL), weight)


class ThreadSafeCallback:
    """Helper class for thread-safe GUI updates"""
    
    def __init__(self, widget):
        self.widget = widget
        self.queue = queue.Queue()
        self._check_queue()
    
    def _check_queue(self):
        """Check queue and execute callbacks"""
        try:
            while True:
                callback = self.queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        self.widget.after(50, self._check_queue)
    
    def call(self, callback):
        """Thread-safe callback execution"""
        self.queue.put(callback)


class AudioPlayer:
    """Improved audio player with better state management"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_playing = False
        self.current_audio = None
        self.play_thread = None
        self._lock = threading.Lock()
        
    def play(self, audio: np.ndarray, on_start: Optional[Callable] = None, 
             on_finish: Optional[Callable] = None, on_error: Optional[Callable] = None):
        """Play audio asynchronously with callbacks"""
        if not AUDIO_PLAYBACK_AVAILABLE:
            if on_error:
                on_error("sounddevice not installed")
            return False
        
        self.stop()
        
        with self._lock:
            self.current_audio = audio
            self.is_playing = True
        
        def _play():
            try:
                if on_start:
                    on_start()
                sd.play(audio, self.sample_rate)
                sd.wait()
            except Exception as e:
                if on_error:
                    on_error(str(e))
            finally:
                with self._lock:
                    self.is_playing = False
                if on_finish:
                    on_finish()
        
        self.play_thread = threading.Thread(target=_play, daemon=True)
        self.play_thread.start()
        return True
    
    def stop(self):
        """Stop current playback"""
        with self._lock:
            if AUDIO_PLAYBACK_AVAILABLE:
                try:
                    sd.stop()
                except Exception:
                    pass
            self.is_playing = False
    
    @property
    def playing(self):
        """Check if currently playing"""
        with self._lock:
            return self.is_playing


class StatusBar(ttk.Frame):
    """Modern status bar with model indicator"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        
        # Left side - status text
        self.status_var = tk.StringVar(value="San sang")
        self.status_label = ttk.Label(
            self, 
            textvariable=self.status_var,
            font=ModernStyle.get_font("small")
        )
        self.status_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        # Right side - model status indicator
        self.model_frame = ttk.Frame(self)
        self.model_frame.pack(side=tk.RIGHT, padx=ModernStyle.PAD_MEDIUM)
        
        self.model_indicator = tk.Canvas(
            self.model_frame, 
            width=10, height=10, 
            bg=ModernStyle.BG_PRIMARY,
            highlightthickness=0
        )
        self.model_indicator.pack(side=tk.LEFT, padx=(0, 5))
        self._draw_indicator(False)
        
        self.model_label = ttk.Label(
            self.model_frame,
            text="Chua tai model",
            font=ModernStyle.get_font("small")
        )
        self.model_label.pack(side=tk.LEFT)
    
    def _draw_indicator(self, loaded: bool):
        """Draw status indicator circle"""
        self.model_indicator.delete("all")
        color = ModernStyle.SUCCESS if loaded else ModernStyle.FG_MUTED
        self.model_indicator.create_oval(2, 2, 8, 8, fill=color, outline="")
    
    def set_status(self, text: str):
        """Set status text"""
        self.status_var.set(text)
    
    def set_model_status(self, loaded: bool, device: str = ""):
        """Set model status"""
        self._draw_indicator(loaded)
        if loaded:
            self.model_label.config(text=f"Model: {device}")
        else:
            self.model_label.config(text="Chua tai model")


class DenoiseTab(ttk.Frame):
    """Tab for single file denoising - improved version"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.audio_player = AudioPlayer()
        self.callback_handler = ThreadSafeCallback(self)
        
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.status_text = tk.StringVar(value="San sang xu ly")
        
        self.input_audio = None
        self.output_audio = None
        self.is_processing = False
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Top Section - File I/O ===
        io_frame = ttk.Frame(main_frame)
        io_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Input Section
        self._create_input_section(io_frame)
        
        # Output Section
        self._create_output_section(io_frame)
        
        # === Process Section ===
        self._create_process_section(main_frame)
        
        # === Visualization Section ===
        if VISUALIZATION_AVAILABLE:
            self._create_visualization_section(main_frame)
    
    def _create_input_section(self, parent):
        """Create input file section"""
        input_frame = ttk.LabelFrame(parent, text="  File Dau Vao  ", padding=ModernStyle.PAD_MEDIUM)
        input_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_SMALL))
        
        # File selection row
        file_row = ttk.Frame(input_frame)
        file_row.pack(fill=tk.X, pady=(0, ModernStyle.PAD_SMALL))
        
        ttk.Label(file_row, text="File audio nhieu:", width=15, anchor="w").pack(side=tk.LEFT)
        
        entry = ttk.Entry(file_row, textvariable=self.input_path)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=ModernStyle.PAD_SMALL)
        
        browse_btn = ttk.Button(file_row, text="Chon File...", command=self._browse_input, width=12)
        browse_btn.pack(side=tk.LEFT)
        
        # Playback controls
        control_row = ttk.Frame(input_frame)
        control_row.pack(fill=tk.X)
        
        self.play_input_btn = ttk.Button(
            control_row, 
            text="Phat Audio", 
            command=self._play_input,
            width=12
        )
        self.play_input_btn.pack(side=tk.LEFT)
        
        self.stop_btn = ttk.Button(
            control_row, 
            text="Dung", 
            command=self._stop_audio,
            width=8
        )
        self.stop_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.input_info_label = ttk.Label(control_row, text="", foreground=ModernStyle.FG_SECONDARY)
        self.input_info_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
    
    def _create_output_section(self, parent):
        """Create output file section"""
        output_frame = ttk.LabelFrame(parent, text="  File Dau Ra  ", padding=ModernStyle.PAD_MEDIUM)
        output_frame.pack(fill=tk.X)
        
        # File selection row
        file_row = ttk.Frame(output_frame)
        file_row.pack(fill=tk.X, pady=(0, ModernStyle.PAD_SMALL))
        
        ttk.Label(file_row, text="Luu file sach:", width=15, anchor="w").pack(side=tk.LEFT)
        
        entry = ttk.Entry(file_row, textvariable=self.output_path)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=ModernStyle.PAD_SMALL)
        
        browse_btn = ttk.Button(file_row, text="Chon...", command=self._browse_output, width=12)
        browse_btn.pack(side=tk.LEFT)
        
        # Playback controls
        control_row = ttk.Frame(output_frame)
        control_row.pack(fill=tk.X)
        
        self.play_output_btn = ttk.Button(
            control_row, 
            text="Phat Output", 
            command=self._play_output,
            state=tk.DISABLED,
            width=12
        )
        self.play_output_btn.pack(side=tk.LEFT)
        
        self.output_info_label = ttk.Label(control_row, text="", foreground=ModernStyle.FG_SECONDARY)
        self.output_info_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
    
    def _create_process_section(self, parent):
        """Create processing controls section"""
        process_frame = ttk.LabelFrame(parent, text="  Xu Ly  ", padding=ModernStyle.PAD_MEDIUM)
        process_frame.pack(fill=tk.X, pady=ModernStyle.PAD_MEDIUM)
        
        # Button row
        btn_row = ttk.Frame(process_frame)
        btn_row.pack(fill=tk.X)
        
        self.process_btn = ttk.Button(
            btn_row, 
            text="KHU NHIEU",
            command=self._process,
            style="Accent.TButton",
            width=15
        )
        self.process_btn.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(btn_row, mode='indeterminate', length=200)
        self.progress.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        # Status label
        self.status_label = ttk.Label(
            btn_row, 
            textvariable=self.status_text,
            foreground=ModernStyle.FG_SECONDARY
        )
        self.status_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
    
    def _create_visualization_section(self, parent):
        """Create spectrogram visualization section"""
        viz_frame = ttk.LabelFrame(parent, text="  So Sanh Spectrogram  ", padding=ModernStyle.PAD_MEDIUM)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(10, 3.5), dpi=100, facecolor=ModernStyle.BG_SECONDARY)
        self.fig.patch.set_facecolor(ModernStyle.BG_SECONDARY)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Simple toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X, pady=(ModernStyle.PAD_SMALL, 0))
        
        ttk.Button(toolbar_frame, text="Zoom Reset", command=self._reset_zoom, width=12).pack(side=tk.LEFT)
        ttk.Label(toolbar_frame, text="  |  Left: Input (Nhieu)  |  Right: Output (Sach)", 
                  foreground=ModernStyle.FG_SECONDARY).pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
    
    def _reset_zoom(self):
        """Reset zoom on spectrogram"""
        if hasattr(self, 'fig'):
            for ax in self.fig.axes:
                ax.autoscale()
            self.canvas.draw()
    
    def _browse_input(self):
        """Browse for input file"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
            ("WAV files", "*.wav"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes, title="Chon file audio")
        if path:
            self.input_path.set(path)
            self._load_input(path)
            
            # Auto-generate output path
            base = Path(path)
            output = base.parent / f"{base.stem}_denoised.wav"
            self.output_path.set(str(output))
    
    def _browse_output(self):
        """Browse for output file"""
        filetypes = [("WAV files", "*.wav")]
        path = filedialog.asksaveasfilename(
            filetypes=filetypes, 
            defaultextension=".wav",
            title="Luu file output"
        )
        if path:
            self.output_path.set(path)
    
    def _load_input(self, path: str):
        """Load input audio file"""
        try:
            self.input_audio, sr = load_audio(path, sample_rate=16000)
            if isinstance(self.input_audio, torch.Tensor):
                self.input_audio = self.input_audio.numpy()
            
            duration = len(self.input_audio) / 16000
            self.input_info_label.config(text=f"Thoi luong: {duration:.2f}s | 16kHz")
            self.status_text.set("Da tai file input")
            self.play_input_btn.config(state=tk.NORMAL)
            
            # Update visualization
            self._update_visualization(input_only=True)
            
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the tai file audio:\n{e}")
            self.status_text.set("Loi khi tai file")
    
    def _play_input(self):
        """Play input audio"""
        if self.input_audio is None:
            messagebox.showwarning("Canh bao", "Chua tai file input")
            return
        
        if not AUDIO_PLAYBACK_AVAILABLE:
            messagebox.showwarning("Canh bao", "Khong the phat audio.\nCai dat: pip install sounddevice")
            return
        
        def on_start():
            self.callback_handler.call(lambda: self.play_input_btn.config(text="Dang phat..."))
        
        def on_finish():
            self.callback_handler.call(lambda: self.play_input_btn.config(text="Phat Audio"))
        
        def on_error(err):
            self.callback_handler.call(lambda: messagebox.showerror("Loi", f"Loi phat audio: {err}"))
            on_finish()
        
        self.audio_player.play(self.input_audio, on_start, on_finish, on_error)
    
    def _play_output(self):
        """Play output audio"""
        if self.output_audio is None:
            return
        
        if not AUDIO_PLAYBACK_AVAILABLE:
            messagebox.showwarning("Canh bao", "Khong the phat audio.\nCai dat: pip install sounddevice")
            return
        
        def on_start():
            self.callback_handler.call(lambda: self.play_output_btn.config(text="Dang phat..."))
        
        def on_finish():
            self.callback_handler.call(lambda: self.play_output_btn.config(text="Phat Output"))
        
        def on_error(err):
            self.callback_handler.call(lambda: messagebox.showerror("Loi", f"Loi phat audio: {err}"))
            on_finish()
        
        self.audio_player.play(self.output_audio, on_start, on_finish, on_error)
    
    def _stop_audio(self):
        """Stop audio playback"""
        self.audio_player.stop()
        self.play_input_btn.config(text="Phat Audio")
        if self.output_audio is not None:
            self.play_output_btn.config(text="Phat Output")
    
    def _process(self):
        """Run denoising"""
        if not self.input_path.get():
            messagebox.showwarning("Canh bao", "Vui long chon file audio dau vao")
            return
        
        if not self.app.denoiser:
            messagebox.showwarning("Canh bao", "Vui long tai model truoc!\n\nVao tab [Cai Dat] de tai model.")
            return
        
        if self.is_processing:
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_text.set("Dang xu ly...")
        
        def _run():
            try:
                result = self.app.denoiser.denoise_file(
                    self.input_path.get(),
                    self.output_path.get()
                )
                
                # Load output for playback
                output_audio, _ = load_audio(self.output_path.get(), sample_rate=16000)
                if isinstance(output_audio, torch.Tensor):
                    output_audio = output_audio.numpy()
                
                self.callback_handler.call(lambda: self._process_complete(result, output_audio))
                
            except Exception as e:
                self.callback_handler.call(lambda: self._process_error(str(e)))
        
        threading.Thread(target=_run, daemon=True).start()
    
    def _process_complete(self, result: dict, output_audio: np.ndarray):
        """Handle process completion"""
        self.is_processing = False
        self.output_audio = output_audio
        
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        self.play_output_btn.config(state=tk.NORMAL)
        
        time_str = f"{result['processing_time']:.2f}s"
        rtf_str = f"{result['rtf']:.2f}x"
        self.status_text.set(f"Hoan thanh! Thoi gian: {time_str} | RTF: {rtf_str}")
        self.output_info_label.config(text=f"Da luu: {Path(result['output_path']).name}")
        
        # Update visualization
        self._update_visualization()
        
        # Update app status
        self.app.status_bar.set_status(f"Da khu nhieu thanh cong: {Path(result['output_path']).name}")
        
        messagebox.showinfo(
            "Thanh cong", 
            f"Da khu nhieu thanh cong!\n\n"
            f"File output: {result['output_path']}\n"
            f"Thoi gian xu ly: {time_str}"
        )
    
    def _process_error(self, error: str):
        """Handle process error"""
        self.is_processing = False
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        self.status_text.set(f"Loi: {error[:50]}...")
        
        self.app.status_bar.set_status(f"Loi xu ly")
        messagebox.showerror("Loi", f"Khong the xu ly:\n{error}")
    
    def _update_visualization(self, input_only: bool = False):
        """Update spectrogram visualization"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        self.fig.clear()
        
        # Configure style for dark theme
        plt.style.use('dark_background')
        
        if input_only and self.input_audio is not None:
            ax = self.fig.add_subplot(111)
            ax.set_facecolor(ModernStyle.BG_PRIMARY)
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(self.input_audio)), ref=np.max)
            img = librosa.display.specshow(D, sr=16000, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
            ax.set_title('Input (Co nhieu)', color=ModernStyle.FG_PRIMARY, fontsize=12, pad=10)
            ax.tick_params(colors=ModernStyle.FG_SECONDARY)
            ax.set_xlabel('Thoi gian (s)', color=ModernStyle.FG_SECONDARY)
            ax.set_ylabel('Tan so (Hz)', color=ModernStyle.FG_SECONDARY)
            
        elif self.input_audio is not None and self.output_audio is not None:
            # Two spectrograms side by side
            ax1 = self.fig.add_subplot(121)
            ax1.set_facecolor(ModernStyle.BG_PRIMARY)
            D1 = librosa.amplitude_to_db(np.abs(librosa.stft(self.input_audio)), ref=np.max)
            librosa.display.specshow(D1, sr=16000, x_axis='time', y_axis='hz', ax=ax1, cmap='magma')
            ax1.set_title('Input (Co nhieu)', color=ModernStyle.FG_PRIMARY, fontsize=11, pad=8)
            ax1.tick_params(colors=ModernStyle.FG_SECONDARY, labelsize=9)
            ax1.set_xlabel('Thoi gian (s)', color=ModernStyle.FG_SECONDARY, fontsize=9)
            ax1.set_ylabel('Tan so (Hz)', color=ModernStyle.FG_SECONDARY, fontsize=9)
            
            ax2 = self.fig.add_subplot(122)
            ax2.set_facecolor(ModernStyle.BG_PRIMARY)
            D2 = librosa.amplitude_to_db(np.abs(librosa.stft(self.output_audio)), ref=np.max)
            librosa.display.specshow(D2, sr=16000, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis')
            ax2.set_title('Output (Da khu nhieu)', color=ModernStyle.SUCCESS, fontsize=11, pad=8)
            ax2.tick_params(colors=ModernStyle.FG_SECONDARY, labelsize=9)
            ax2.set_xlabel('Thoi gian (s)', color=ModernStyle.FG_SECONDARY, fontsize=9)
            ax2.set_ylabel('', color=ModernStyle.FG_SECONDARY, fontsize=9)
        
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


class BatchTab(ttk.Frame):
    """Tab for batch processing multiple files - improved version"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.callback_handler = ThreadSafeCallback(self)
        
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.file_list = []
        self.is_processing = False
        
        self._create_widgets()
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Directory Selection ===
        dir_frame = ttk.LabelFrame(main_frame, text="  Thu Muc  ", padding=ModernStyle.PAD_MEDIUM)
        dir_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Input directory
        input_row = ttk.Frame(dir_frame)
        input_row.pack(fill=tk.X, pady=ModernStyle.PAD_XS)
        
        ttk.Label(input_row, text="Thu muc input:", width=15, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(input_row, textvariable=self.input_dir).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(input_row, text="Chon...", command=self._browse_input_dir, width=10).pack(side=tk.LEFT)
        
        # Output directory
        output_row = ttk.Frame(dir_frame)
        output_row.pack(fill=tk.X, pady=ModernStyle.PAD_XS)
        
        ttk.Label(output_row, text="Thu muc output:", width=15, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(output_row, textvariable=self.output_dir).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(output_row, text="Chon...", command=self._browse_output_dir, width=10).pack(side=tk.LEFT)
        
        # === File List ===
        list_frame = ttk.LabelFrame(main_frame, text="  Danh Sach File  ", padding=ModernStyle.PAD_MEDIUM)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Treeview with scrollbar
        tree_container = ttk.Frame(list_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)
        
        columns = ("filename", "duration", "status")
        self.tree = ttk.Treeview(tree_container, columns=columns, show="headings", height=12)
        
        self.tree.heading("filename", text="Ten File")
        self.tree.heading("duration", text="Thoi Luong")
        self.tree.heading("status", text="Trang Thai")
        
        self.tree.column("filename", width=350, minwidth=200)
        self.tree.column("duration", width=100, minwidth=80)
        self.tree.column("status", width=150, minwidth=100)
        
        scrollbar_y = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar_y.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === Controls ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)
        
        self.scan_btn = ttk.Button(control_frame, text="Quet File", command=self._scan_files, width=12)
        self.scan_btn.pack(side=tk.LEFT, padx=(0, ModernStyle.PAD_SMALL))
        
        self.process_btn = ttk.Button(
            control_frame, 
            text="XU LY TAT CA", 
            command=self._process_all, 
            style="Accent.TButton",
            width=15
        )
        self.process_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        self.stop_btn = ttk.Button(control_frame, text="Dung", command=self._stop_processing, width=8, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        # Progress
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=ModernStyle.PAD_MEDIUM)
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', length=250)
        self.progress.pack(side=tk.LEFT)
        
        self.progress_label = ttk.Label(progress_frame, text="", foreground=ModernStyle.FG_SECONDARY)
        self.progress_label.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
    
    def _browse_input_dir(self):
        """Browse for input directory"""
        path = filedialog.askdirectory(title="Chon thu muc chua file audio")
        if path:
            self.input_dir.set(path)
            self.output_dir.set(path + "_denoised")
            self._scan_files()
    
    def _browse_output_dir(self):
        """Browse for output directory"""
        path = filedialog.askdirectory(title="Chon thu muc luu output")
        if path:
            self.output_dir.set(path)
    
    def _scan_files(self):
        """Scan input directory for audio files"""
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showwarning("Canh bao", "Vui long chon thu muc input hop le")
            return
        
        # Clear current list
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.file_list.clear()
        
        # Find audio files
        extensions = ['.wav', '.mp3', '.flac', '.ogg']
        for file in sorted(Path(input_dir).iterdir()):
            if file.suffix.lower() in extensions:
                self.file_list.append(file)
                
                # Get duration
                try:
                    audio, _ = load_audio(str(file), sample_rate=16000)
                    if isinstance(audio, torch.Tensor):
                        audio = audio.numpy()
                    duration = len(audio) / 16000
                    duration_str = f"{duration:.1f}s"
                except Exception:
                    duration_str = "--"
                
                self.tree.insert("", tk.END, values=(file.name, duration_str, "Cho xu ly"))
        
        self.progress_label.config(text=f"Tim thay {len(self.file_list)} file")
        self.app.status_bar.set_status(f"Da quet {len(self.file_list)} file audio")
    
    def _stop_processing(self):
        """Stop batch processing"""
        self.is_processing = False
        self.stop_btn.config(state=tk.DISABLED)
    
    def _process_all(self):
        """Process all files"""
        if not self.file_list:
            messagebox.showwarning("Canh bao", "Khong co file nao de xu ly.\nHay quet file truoc.")
            return
        
        if not self.app.denoiser:
            messagebox.showwarning("Canh bao", "Vui long tai model truoc!\n\nVao tab [Cai Dat] de tai model.")
            return
        
        if self.is_processing:
            return
        
        output_dir = Path(self.output_dir.get())
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the tao thu muc output:\n{e}")
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress['maximum'] = len(self.file_list)
        self.progress['value'] = 0
        
        def _run():
            success_count = 0
            for i, file in enumerate(self.file_list):
                if not self.is_processing:
                    break
                
                try:
                    self.callback_handler.call(lambda idx=i: self._update_file_status(idx, "Dang xu ly..."))
                    
                    output_path = output_dir / f"{file.stem}_denoised.wav"
                    self.app.denoiser.denoise_file(str(file), str(output_path))
                    
                    success_count += 1
                    self.callback_handler.call(lambda idx=i: self._update_file_status(idx, "Hoan thanh"))
                    
                except Exception as e:
                    error_msg = str(e)[:30]
                    self.callback_handler.call(lambda idx=i, err=error_msg: self._update_file_status(idx, f"Loi: {err}"))
                
                # Update progress
                self.callback_handler.call(lambda v=i+1, total=len(self.file_list): self._update_progress(v, total))
            
            self.callback_handler.call(lambda: self._process_complete(success_count))
        
        threading.Thread(target=_run, daemon=True).start()
    
    def _update_file_status(self, index: int, status: str):
        """Update status of a file in the tree"""
        items = self.tree.get_children()
        if index < len(items):
            values = self.tree.item(items[index])['values']
            self.tree.item(items[index], values=(values[0], values[1], status))
            self.tree.see(items[index])
    
    def _update_progress(self, current: int, total: int):
        """Update progress bar and label"""
        self.progress['value'] = current
        self.progress_label.config(text=f"{current}/{total} file")
    
    def _process_complete(self, success_count: int):
        """Handle batch processing complete"""
        self.is_processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        total = len(self.file_list)
        self.progress_label.config(text=f"Xong: {success_count}/{total} file")
        self.app.status_bar.set_status(f"Da xu ly xong {success_count}/{total} file")
        
        messagebox.showinfo(
            "Hoan thanh", 
            f"Da xu ly xong!\n\n"
            f"Thanh cong: {success_count}/{total} file\n"
            f"Output: {self.output_dir.get()}"
        )


class TrainTab(ttk.Frame):
    """Tab for model training information"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._create_widgets()
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Info Section ===
        info_frame = ttk.LabelFrame(main_frame, text="  Huong Dan Training  ", padding=ModernStyle.PAD_LARGE)
        info_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        info_text = """De training model, ban nen su dung command line de co hieu suat tot nhat:

    python train.py --config config.yaml

Cac buoc chuan bi:
  1. Tai dataset VoiceBank + DEMAND (chay: python download_dataset.py)
  2. Cau hinh cac tham so trong file config.yaml
  3. Chay lenh training o tren

Yeu cau phan cung:
  - GPU voi >= 8GB VRAM (khuyen nghi)
  - Hoac su dung Google Colab (mien phi)

Xem file README.md de biet them chi tiet."""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                  font=ModernStyle.get_font("normal")).pack(anchor=tk.W)
        
        # === Dataset Section ===
        dataset_frame = ttk.LabelFrame(main_frame, text="  Dataset  ", padding=ModernStyle.PAD_MEDIUM)
        dataset_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        self.train_clean = tk.StringVar(value="./data/clean_trainset_28spk_wav")
        self.train_noisy = tk.StringVar(value="./data/noisy_trainset_28spk_wav")
        
        dirs = [
            ("Train Clean:", self.train_clean),
            ("Train Noisy:", self.train_noisy),
        ]
        
        for label, var in dirs:
            row = ttk.Frame(dataset_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=12, anchor="w").pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
            ttk.Button(row, text="...", width=3, command=lambda v=var: self._browse_dir(v)).pack(side=tk.LEFT)
        
        # === Parameters ===
        param_frame = ttk.LabelFrame(main_frame, text="  Tham So  ", padding=ModernStyle.PAD_MEDIUM)
        param_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        param_row = ttk.Frame(param_frame)
        param_row.pack(fill=tk.X)
        
        params = [
            ("Batch Size:", "16", 8),
            ("Epochs:", "100", 8),
            ("Learning Rate:", "0.0001", 10),
        ]
        
        for label, default, width in params:
            ttk.Label(param_row, text=label).pack(side=tk.LEFT)
            entry = ttk.Entry(param_row, width=width)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, padx=(5, 20))
        
        # === Actions ===
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X)
        
        ttk.Button(
            action_frame, 
            text="Mo Command Prompt", 
            command=self._open_terminal,
            width=20
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            action_frame, 
            text="  Khuyen nghi chay training tu terminal de theo doi progress tot hon",
            foreground=ModernStyle.FG_SECONDARY
        ).pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
    
    def _browse_dir(self, var: tk.StringVar):
        """Browse for directory"""
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    def _open_terminal(self):
        """Open terminal/command prompt"""
        import subprocess
        import platform
        
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.Popen("cmd", shell=True)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", "-a", "Terminal"])
            else:  # Linux
                # Try common terminal emulators
                terminals = ["gnome-terminal", "konsole", "xfce4-terminal", "xterm"]
                for term in terminals:
                    try:
                        subprocess.Popen([term])
                        break
                    except FileNotFoundError:
                        continue
        except Exception as e:
            messagebox.showerror("Loi", f"Khong the mo terminal:\n{e}")


class SettingsTab(ttk.Frame):
    """Tab for application settings - improved version"""
    
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.callback_handler = ThreadSafeCallback(self)
        
        self.checkpoint_path = tk.StringVar()
        self.normalizer_path = tk.StringVar()  # Path to normalizer_stats.json
        self.device = tk.StringVar(value="auto")
        
        self._create_widgets()
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=ModernStyle.PAD_LARGE)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Model Section ===
        model_frame = ttk.LabelFrame(main_frame, text="  Tai Model  ", padding=ModernStyle.PAD_MEDIUM)
        model_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # Checkpoint selection
        ckpt_row = ttk.Frame(model_frame)
        ckpt_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(ckpt_row, text="Checkpoint:", width=12, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(ckpt_row, textvariable=self.checkpoint_path).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(ckpt_row, text="Chon File...", command=self._browse_checkpoint, width=12).pack(side=tk.LEFT)
        
        # Normalizer stats selection (QUAN TRONG!)
        norm_row = ttk.Frame(model_frame)
        norm_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(norm_row, text="Normalizer:", width=12, anchor="w").pack(side=tk.LEFT)
        ttk.Entry(norm_row, textvariable=self.normalizer_path).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL, expand=True, fill=tk.X)
        ttk.Button(norm_row, text="Chon File...", command=self._browse_normalizer, width=12).pack(side=tk.LEFT)
        
        # Normalizer help text
        norm_help = ttk.Frame(model_frame)
        norm_help.pack(fill=tk.X)
        ttk.Label(
            norm_help,
            text="  (Tuy chon: file normalizer_stats.json tu training - QUAN TRONG de denoise tot!)",
            foreground=ModernStyle.WARNING,
            font=ModernStyle.get_font("small")
        ).pack(side=tk.LEFT, padx=(0, 0))
        
        # Device selection
        device_row = ttk.Frame(model_frame)
        device_row.pack(fill=tk.X, pady=ModernStyle.PAD_SMALL)
        
        ttk.Label(device_row, text="Device:", width=12, anchor="w").pack(side=tk.LEFT)
        device_combo = ttk.Combobox(
            device_row, 
            textvariable=self.device, 
            values=["auto", "cuda", "cpu"], 
            width=15,
            state="readonly"
        )
        device_combo.pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        ttk.Label(
            device_row, 
            text="(auto = tu dong chon GPU neu co)",
            foreground=ModernStyle.FG_SECONDARY
        ).pack(side=tk.LEFT, padx=ModernStyle.PAD_SMALL)
        
        # Load button row
        btn_row = ttk.Frame(model_frame)
        btn_row.pack(fill=tk.X, pady=(ModernStyle.PAD_MEDIUM, ModernStyle.PAD_SMALL))
        
        self.load_btn = ttk.Button(
            btn_row, 
            text="TAI MODEL", 
            command=self._load_model, 
            style="Accent.TButton",
            width=15
        )
        self.load_btn.pack(side=tk.LEFT)
        
        self.load_progress = ttk.Progressbar(btn_row, mode='indeterminate', length=150)
        self.load_progress.pack(side=tk.LEFT, padx=ModernStyle.PAD_MEDIUM)
        
        self.model_status = ttk.Label(btn_row, text="Chua tai model", foreground=ModernStyle.FG_SECONDARY)
        self.model_status.pack(side=tk.LEFT)
        
        # === System Info ===
        info_frame = ttk.LabelFrame(main_frame, text="  Thong Tin He Thong  ", padding=ModernStyle.PAD_MEDIUM)
        info_frame.pack(fill=tk.X, pady=(0, ModernStyle.PAD_MEDIUM))
        
        # System info items
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                cuda_text = f"CUDA: Co (GPU: {gpu_name})"
                cuda_color = ModernStyle.SUCCESS
            except Exception:
                cuda_text = "CUDA: Co"
                cuda_color = ModernStyle.SUCCESS
        else:
            cuda_text = "CUDA: Khong"
            cuda_color = ModernStyle.WARNING
        
        info_items = [
            (f"PyTorch: {torch.__version__}", ModernStyle.FG_PRIMARY),
            (cuda_text, cuda_color),
            (f"Audio Playback: {'Co' if AUDIO_PLAYBACK_AVAILABLE else 'Khong (cai sounddevice)'}", 
             ModernStyle.SUCCESS if AUDIO_PLAYBACK_AVAILABLE else ModernStyle.WARNING),
            (f"Visualization: {'Co' if VISUALIZATION_AVAILABLE else 'Khong (cai matplotlib, librosa)'}", 
             ModernStyle.SUCCESS if VISUALIZATION_AVAILABLE else ModernStyle.WARNING),
        ]
        
        for text, color in info_items:
            row = ttk.Frame(info_frame)
            row.pack(fill=tk.X, pady=1)
            
            # Status indicator
            indicator = tk.Canvas(row, width=8, height=8, bg=ModernStyle.BG_SECONDARY, highlightthickness=0)
            indicator.pack(side=tk.LEFT, padx=(0, 8))
            indicator.create_oval(0, 0, 8, 8, fill=color, outline="")
            
            ttk.Label(row, text=text).pack(side=tk.LEFT)
        
        # === About ===
        about_frame = ttk.LabelFrame(main_frame, text="  Ve Ung Dung  ", padding=ModernStyle.PAD_MEDIUM)
        about_frame.pack(fill=tk.X)
        
        about_text = """Speech Denoising Application

Ung dung khu nhieu giong noi su dung Deep Learning (U-Net).

Tinh nang:
  - Khu nhieu file audio don le hoac hang loat
  - So sanh spectrogram truoc/sau khi xu ly
  - Phat audio truc tiep trong ung dung
  - Ho tro nhieu dinh dang: WAV, MP3, FLAC, OGG

Phat trien voi PyTorch va Tkinter
Version 1.0"""
        
        ttk.Label(about_frame, text=about_text, justify=tk.LEFT).pack(anchor=tk.W)
    
    def _browse_checkpoint(self):
        """Browse for checkpoint file"""
        filetypes = [
            ("PyTorch checkpoint", "*.pt *.pth"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes, title="Chon file checkpoint model")
        if path:
            self.checkpoint_path.set(path)
            # Auto-detect normalizer_stats.json in same directory
            normalizer_candidates = [
                Path(path).parent / 'normalizer_stats.json',
                Path(path).parent.parent / 'normalizer_stats.json',
                Path(path).parent / 'data' / 'normalizer_stats.json',
            ]
            for norm_path in normalizer_candidates:
                if norm_path.exists():
                    self.normalizer_path.set(str(norm_path))
                    break
    
    def _browse_normalizer(self):
        """Browse for normalizer stats file"""
        filetypes = [
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes, title="Chon file normalizer_stats.json")
        if path:
            self.normalizer_path.set(path)
    
    def _load_model(self):
        """Load model from checkpoint"""
        ckpt_path = self.checkpoint_path.get()
        norm_path = self.normalizer_path.get()
        
        if not ckpt_path:
            messagebox.showwarning("Canh bao", "Vui long chon file checkpoint truoc")
            return
        
        if not os.path.exists(ckpt_path):
            messagebox.showerror("Loi", f"File khong ton tai:\n{ckpt_path}")
            return
        
        # Warn if normalizer not specified
        if not norm_path:
            result = messagebox.askyesno(
                "Canh bao",
                "Ban chua chon file normalizer_stats.json!\n\n"
                "File nay rat quan trong de denoise dung cach.\n"
                "Neu model duoc train voi normalization, ket qua se bi sai.\n\n"
                "Ban co muon tiep tuc khong?"
            )
            if not result:
                return
        
        self.load_btn.config(state=tk.DISABLED)
        self.load_progress.start(10)
        self.model_status.config(text="Dang tai...", foreground=ModernStyle.INFO)
        
        def _load():
            try:
                device = None if self.device.get() == "auto" else self.device.get()
                # Truyen normalizer_path de denormalize output dung cach
                normalizer = norm_path if norm_path and os.path.exists(norm_path) else None
                denoiser = SpeechDenoiser(
                    checkpoint_path=ckpt_path,
                    device=device,
                    normalizer_path=normalizer,
                    match_amplitude=True,
                    prevent_clipping=True
                )
                has_normalizer = denoiser.normalizer is not None
                self.callback_handler.call(lambda: self._load_complete(denoiser, has_normalizer))
            except Exception as e:
                self.callback_handler.call(lambda: self._load_error(str(e)))
        
        threading.Thread(target=_load, daemon=True).start()
    
    def _load_complete(self, denoiser, has_normalizer=False):
        """Handle model load complete"""
        self.app.denoiser = denoiser
        
        self.load_btn.config(state=tk.NORMAL)
        self.load_progress.stop()
        
        device_str = str(denoiser.device)
        norm_status = "co normalizer" if has_normalizer else "KHONG co normalizer"
        self.model_status.config(text=f"Da tai ({device_str}, {norm_status})", foreground=ModernStyle.SUCCESS)
        
        # Update status bar
        self.app.status_bar.set_model_status(True, device_str)
        self.app.status_bar.set_status("Model da san sang su dung")
        
        # Show warning if no normalizer
        norm_msg = ""
        if not has_normalizer:
            norm_msg = "\n\n⚠️ CANH BAO: Khong tim thay normalizer_stats.json!\nNeu model duoc train voi normalization, ket qua co the bi sai."
        
        messagebox.showinfo(
            "Thanh cong", 
            f"Da tai model thanh cong!\n\nDevice: {device_str}\nNormalizer: {'Co' if has_normalizer else 'Khong'}{norm_msg}\n\nBay gio ban co the su dung tab [Khu Nhieu] de xu ly audio."
        )
    
    def _load_error(self, error: str):
        """Handle model load error"""
        self.load_btn.config(state=tk.NORMAL)
        self.load_progress.stop()
        self.model_status.config(text="Loi!", foreground=ModernStyle.ERROR)
        
        self.app.status_bar.set_status("Loi khi tai model")
        messagebox.showerror("Loi", f"Khong the tai model:\n\n{error}")


class SpeechDenoisingApp:
    """Main application class - improved version"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Speech Denoising - Khu Nhieu Giong Noi")
        self.root.geometry("1100x750")
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
        """Configure ttk styles for modern dark theme"""
        style = ttk.Style()
        
        # Use clam theme as base (most customizable)
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        
        # Configure base styles
        style.configure(".", 
            background=ModernStyle.BG_SECONDARY,
            foreground=ModernStyle.FG_PRIMARY,
            font=ModernStyle.get_font("normal"),
            borderwidth=0
        )
        
        # Frame styles
        style.configure("TFrame", background=ModernStyle.BG_SECONDARY)
        
        # Label styles
        style.configure("TLabel", 
            background=ModernStyle.BG_SECONDARY, 
            foreground=ModernStyle.FG_PRIMARY
        )
        
        # LabelFrame styles
        style.configure("TLabelframe", 
            background=ModernStyle.BG_SECONDARY,
            bordercolor=ModernStyle.BORDER,
            relief="solid",
            borderwidth=1
        )
        style.configure("TLabelframe.Label", 
            background=ModernStyle.BG_SECONDARY, 
            foreground=ModernStyle.FG_PRIMARY,
            font=ModernStyle.get_font("normal", bold=True)
        )
        
        # Button styles
        style.configure("TButton",
            background=ModernStyle.BG_TERTIARY,
            foreground=ModernStyle.FG_PRIMARY,
            padding=(12, 6),
            font=ModernStyle.get_font("normal")
        )
        style.map("TButton",
            background=[("active", ModernStyle.BG_INPUT), ("pressed", ModernStyle.BG_PRIMARY)],
            foreground=[("disabled", ModernStyle.FG_MUTED)]
        )
        
        # Accent button style
        style.configure("Accent.TButton",
            background=ModernStyle.ACCENT,
            foreground=ModernStyle.FG_PRIMARY,
            padding=(15, 8),
            font=ModernStyle.get_font("normal", bold=True)
        )
        style.map("Accent.TButton",
            background=[("active", ModernStyle.ACCENT_HOVER), ("pressed", ModernStyle.ACCENT_DARK)],
            foreground=[("disabled", ModernStyle.FG_MUTED)]
        )
        
        # Entry styles
        style.configure("TEntry",
            fieldbackground=ModernStyle.BG_INPUT,
            foreground=ModernStyle.FG_PRIMARY,
            insertcolor=ModernStyle.FG_PRIMARY,
            padding=5
        )
        
        # Combobox styles
        style.configure("TCombobox",
            fieldbackground=ModernStyle.BG_INPUT,
            background=ModernStyle.BG_TERTIARY,
            foreground=ModernStyle.FG_PRIMARY,
            arrowcolor=ModernStyle.FG_PRIMARY
        )
        style.map("TCombobox",
            fieldbackground=[("readonly", ModernStyle.BG_INPUT)],
            selectbackground=[("readonly", ModernStyle.ACCENT)]
        )
        
        # Notebook (tabs) styles
        style.configure("TNotebook", 
            background=ModernStyle.BG_DARK,
            borderwidth=0
        )
        style.configure("TNotebook.Tab", 
            background=ModernStyle.BG_TERTIARY,
            foreground=ModernStyle.FG_SECONDARY,
            padding=(20, 8),
            font=ModernStyle.get_font("normal")
        )
        style.map("TNotebook.Tab",
            background=[("selected", ModernStyle.BG_SECONDARY)],
            foreground=[("selected", ModernStyle.FG_PRIMARY)],
            expand=[("selected", [1, 1, 1, 0])]
        )
        
        # Treeview styles
        style.configure("Treeview",
            background=ModernStyle.BG_PRIMARY,
            foreground=ModernStyle.FG_PRIMARY,
            fieldbackground=ModernStyle.BG_PRIMARY,
            font=ModernStyle.get_font("small"),
            rowheight=25
        )
        style.configure("Treeview.Heading",
            background=ModernStyle.BG_TERTIARY,
            foreground=ModernStyle.FG_PRIMARY,
            font=ModernStyle.get_font("small", bold=True)
        )
        style.map("Treeview",
            background=[("selected", ModernStyle.ACCENT)],
            foreground=[("selected", ModernStyle.FG_PRIMARY)]
        )
        
        # Progressbar styles
        style.configure("TProgressbar",
            background=ModernStyle.ACCENT,
            troughcolor=ModernStyle.BG_PRIMARY,
            borderwidth=0,
            thickness=8
        )
        
        # Scrollbar styles
        style.configure("TScrollbar",
            background=ModernStyle.BG_TERTIARY,
            troughcolor=ModernStyle.BG_PRIMARY,
            borderwidth=0,
            arrowcolor=ModernStyle.FG_SECONDARY
        )
        
        # Set root background
        self.root.configure(bg=ModernStyle.BG_DARK)
    
    def _create_ui(self):
        """Create main UI"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Frame(main_container, padding=(ModernStyle.PAD_LARGE, ModernStyle.PAD_MEDIUM))
        header.pack(fill=tk.X)
        
        # Title
        title_frame = ttk.Frame(header)
        title_frame.pack(side=tk.LEFT)
        
        title_label = ttk.Label(
            title_frame, 
            text="Speech Denoising",
            font=ModernStyle.get_font("xl", bold=True)
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(
            title_frame,
            text="   Khu nhieu giong noi bang Deep Learning",
            font=ModernStyle.get_font("normal"),
            foreground=ModernStyle.FG_SECONDARY
        )
        subtitle_label.pack(side=tk.LEFT)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=ModernStyle.PAD_SMALL, pady=(0, ModernStyle.PAD_SMALL))
        
        # Create tabs
        self.denoise_tab = DenoiseTab(self.notebook, self)
        self.batch_tab = BatchTab(self.notebook, self)
        self.train_tab = TrainTab(self.notebook, self)
        self.settings_tab = SettingsTab(self.notebook, self)
        
        # Add tabs with cleaner names (no emojis for better cross-platform support)
        self.notebook.add(self.denoise_tab, text="  Khu Nhieu  ")
        self.notebook.add(self.batch_tab, text="  Xu Ly Hang Loat  ")
        self.notebook.add(self.train_tab, text="  Huan Luyen  ")
        self.notebook.add(self.settings_tab, text="  Cai Dat  ")
        
        # Status bar
        self.status_bar = StatusBar(main_container, self)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, ModernStyle.PAD_SMALL), padx=ModernStyle.PAD_MEDIUM)
    
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
    print("=" * 50)
    print("  Speech Denoising Application")
    print("=" * 50)
    print()
    
    # Check dependencies
    print("Kiem tra dependencies...")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {'Co' if torch.cuda.is_available() else 'Khong'}")
    print(f"  Audio playback: {'Co' if AUDIO_PLAYBACK_AVAILABLE else 'Khong'}")
    print(f"  Visualization: {'Co' if VISUALIZATION_AVAILABLE else 'Khong'}")
    print()
    print("Khoi dong ung dung...")
    print()
    
    # Run app
    app = SpeechDenoisingApp()
    app.run()


if __name__ == '__main__':
    main()
