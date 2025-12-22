#!/usr/bin/env python3
"""
Script launcher để chạy ứng dụng Speech Denoising GUI

Usage:
    python run_app.py
    
Hoặc trên Linux/macOS:
    chmod +x run_app.py
    ./run_app.py
"""

import sys
import os

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Kiểm tra dependencies
def check_dependencies():
    """Kiểm tra và thông báo về các dependencies cần thiết"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import torchaudio
    except ImportError:
        missing.append("torchaudio")
    
    try:
        import tkinter
    except ImportError:
        missing.append("tkinter (python3-tk)")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    # Optional but recommended
    optional_missing = []
    
    try:
        import sounddevice
    except ImportError:
        optional_missing.append("sounddevice (để phát audio)")
    
    try:
        import librosa
    except ImportError:
        optional_missing.append("librosa (để visualize)")
    
    try:
        import matplotlib
    except ImportError:
        optional_missing.append("matplotlib (để visualize)")
    
    if missing:
        print("=" * 60)
        print("❌ LỖI: Thiếu các thư viện bắt buộc:")
        print("=" * 60)
        for m in missing:
            print(f"  - {m}")
        print()
        print("Cài đặt với:")
        print("  pip install -r requirements.txt")
        print()
        
        if "tkinter" in str(missing):
            print("Đối với tkinter:")
            print("  Ubuntu/Debian: sudo apt-get install python3-tk")
            print("  Fedora: sudo dnf install python3-tkinter")
            print("  macOS: brew install python-tk")
        
        return False
    
    if optional_missing:
        print("=" * 60)
        print("⚠️ CẢNH BÁO: Thiếu một số thư viện tùy chọn:")
        print("=" * 60)
        for m in optional_missing:
            print(f"  - {m}")
        print()
        print("Ứng dụng vẫn chạy được nhưng một số tính năng sẽ bị vô hiệu.")
        print("Cài đặt với: pip install sounddevice librosa matplotlib")
        print()
    
    return True

def main():
    """Main entry point"""
    print()
    print("🎵 Speech Denoising Application")
    print("=" * 40)
    print()
    
    # Kiểm tra dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import và chạy app
    try:
        from app import SpeechDenoisingApp
        
        print("Khởi động ứng dụng...")
        print()
        
        app = SpeechDenoisingApp()
        app.run()
        
    except Exception as e:
        print(f"❌ Lỗi khi khởi động ứng dụng: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
