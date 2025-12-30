#!/usr/bin/env python3
"""
Script launcher de chay ung dung Speech Denoising GUI

Usage:
    python run_app.py
    
Hoac tren Linux/macOS:
    chmod +x run_app.py
    ./run_app.py
"""

import sys
import os

# Them thu muc hien tai vao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Kiem tra va thong bao ve cac dependencies can thiet"""
    print("Kiem tra dependencies...")
    print()
    
    missing_required = []
    missing_optional = []
    
    # Required dependencies
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"       CUDA: Co - {torch.cuda.get_device_name(0)}")
        else:
            print("       CUDA: Khong (se chay tren CPU)")
    except ImportError:
        missing_required.append("torch")
        print("  [X] PyTorch - THIEU")
    
    try:
        import numpy
        print(f"  [OK] NumPy {numpy.__version__}")
    except ImportError:
        missing_required.append("numpy")
        print("  [X] NumPy - THIEU")
    
    try:
        import librosa
        print(f"  [OK] Librosa {librosa.__version__}")
    except ImportError:
        missing_required.append("librosa")
        print("  [X] Librosa - THIEU")
    
    try:
        import tkinter
        print("  [OK] Tkinter")
    except ImportError:
        missing_required.append("tkinter (python3-tk)")
        print("  [X] Tkinter - THIEU")
    
    # Optional dependencies
    try:
        import sounddevice
        print(f"  [OK] Sounddevice {sounddevice.__version__} (phat audio)")
    except ImportError:
        missing_optional.append("sounddevice")
        print("  [!] Sounddevice - khong co (phat audio se bi tat)")
    
    try:
        import matplotlib
        print(f"  [OK] Matplotlib {matplotlib.__version__} (visualization)")
    except ImportError:
        missing_optional.append("matplotlib")
        print("  [!] Matplotlib - khong co (visualization se bi tat)")
    
    try:
        import soundfile
        print(f"  [OK] Soundfile {soundfile.__version__} (luu file)")
    except ImportError:
        missing_required.append("soundfile")
        print("  [X] Soundfile - THIEU")
    
    print()
    
    # Report missing required
    if missing_required:
        print("=" * 55)
        print("  LOI: Thieu cac thu vien bat buoc!")
        print("=" * 55)
        print()
        for m in missing_required:
            print(f"  - {m}")
        print()
        print("Cai dat voi:")
        print("  pip install -r requirements.txt")
        print()
        
        if "tkinter" in str(missing_required):
            print("Doi voi Tkinter:")
            print("  Ubuntu/Debian: sudo apt-get install python3-tk")
            print("  Fedora: sudo dnf install python3-tkinter")
            print("  macOS: brew install python-tk")
            print()
        
        return False
    
    # Report missing optional
    if missing_optional:
        print("-" * 55)
        print("  Canh bao: Thieu mot so thu vien tuy chon")
        print("-" * 55)
        print()
        for m in missing_optional:
            print(f"  - {m}")
        print()
        print("Ung dung van chay duoc nhung mot so tinh nang se bi tat.")
        print("Cai dat them: pip install sounddevice matplotlib")
        print()
    
    return True


def main():
    """Main entry point"""
    print()
    print("=" * 55)
    print("  Speech Denoising Application")
    print("  Khu nhieu giong noi bang Deep Learning")
    print("=" * 55)
    print()
    
    # Kiem tra dependencies
    if not check_dependencies():
        print("Vui long cai dat cac thu vien con thieu va thu lai.")
        sys.exit(1)
    
    # Import va chay app
    try:
        from app import SpeechDenoisingApp
        
        print("Khoi dong ung dung...")
        print()
        
        app = SpeechDenoisingApp()
        app.run()
        
    except ImportError as e:
        print(f"Loi import module: {e}")
        print()
        print("Dam bao ban dang chay tu thu muc chinh cua project.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Loi khi khoi dong ung dung: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
