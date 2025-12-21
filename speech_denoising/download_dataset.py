#!/usr/bin/env python3
"""
Script hướng dẫn và hỗ trợ download VoiceBank + DEMAND dataset

VoiceBank + DEMAND là dataset chuẩn cho speech enhancement với:
- 11,572 training utterances từ 28 speakers
- 824 test utterances từ 2 speakers
- SNR levels: 0, 5, 10, 15 dB
- Noise types: domestic, office, transportation, etc.

Nguồn: https://datashare.ed.ac.uk/handle/10283/2791
"""

import os
import sys
from pathlib import Path


def print_download_instructions():
    """In hướng dẫn download dataset"""
    print("=" * 70)
    print("HƯỚNG DẪN DOWNLOAD VOICEBANK + DEMAND DATASET")
    print("=" * 70)
    print()
    print("VoiceBank + DEMAND là dataset chuẩn cho speech enhancement/denoising.")
    print()
    print("📥 BƯỚC 1: Truy cập website")
    print("-" * 40)
    print("   https://datashare.ed.ac.uk/handle/10283/2791")
    print()
    print("📦 BƯỚC 2: Download các file sau")
    print("-" * 40)
    print("   1. clean_trainset_28spk_wav.zip     (~1.5 GB)")
    print("   2. noisy_trainset_28spk_wav.zip     (~1.5 GB)")
    print("   3. clean_testset_wav.zip            (~150 MB)")
    print("   4. noisy_testset_wav.zip            (~150 MB)")
    print()
    print("   Tổng dung lượng: ~3.3 GB")
    print()
    print("📂 BƯỚC 3: Giải nén vào thư mục data/")
    print("-" * 40)
    
    base_path = Path(__file__).parent / "data"
    print(f"   Đường dẫn: {base_path.absolute()}")
    print()
    print("   Cấu trúc thư mục sau khi giải nén:")
    print(f"   {base_path}/")
    print("   ├── clean_trainset_28spk_wav/   # 11,572 files")
    print("   ├── noisy_trainset_28spk_wav/   # 11,572 files")
    print("   ├── clean_testset_wav/          # 824 files")
    print("   └── noisy_testset_wav/          # 824 files")
    print()
    print("💻 BƯỚC 4: (Linux/Mac) Lệnh giải nén")
    print("-" * 40)
    print("   cd data/")
    print("   unzip clean_trainset_28spk_wav.zip")
    print("   unzip noisy_trainset_28spk_wav.zip")
    print("   unzip clean_testset_wav.zip")
    print("   unzip noisy_testset_wav.zip")
    print()
    print("🔍 BƯỚC 5: Kiểm tra dataset")
    print("-" * 40)
    print("   python download_dataset.py --check")
    print()
    print("=" * 70)
    print()


def check_dataset():
    """Kiểm tra xem dataset đã được download chưa"""
    data_dir = Path(__file__).parent / "data"
    
    required_dirs = [
        "clean_trainset_28spk_wav",
        "noisy_trainset_28spk_wav",
        "clean_testset_wav",
        "noisy_testset_wav"
    ]
    
    expected_counts = {
        "clean_trainset_28spk_wav": 11572,
        "noisy_trainset_28spk_wav": 11572,
        "clean_testset_wav": 824,
        "noisy_testset_wav": 824
    }
    
    print("=" * 50)
    print("KIỂM TRA DATASET")
    print("=" * 50)
    print(f"Thư mục data: {data_dir.absolute()}")
    print()
    
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        
        if not dir_path.exists():
            print(f"❌ {dir_name}: Không tìm thấy")
            all_ok = False
            continue
        
        # Đếm số file .wav
        wav_files = list(dir_path.glob("*.wav"))
        count = len(wav_files)
        expected = expected_counts[dir_name]
        
        if count == 0:
            print(f"❌ {dir_name}: Thư mục trống")
            all_ok = False
        elif count < expected:
            print(f"⚠️  {dir_name}: {count}/{expected} files (thiếu {expected - count})")
            all_ok = False
        else:
            print(f"✅ {dir_name}: {count} files")
    
    print()
    
    if all_ok:
        print("=" * 50)
        print("✅ Dataset đã sẵn sàng!")
        print("Bạn có thể bắt đầu training:")
        print("   python train.py --config config.yaml")
        print("=" * 50)
    else:
        print("=" * 50)
        print("❌ Dataset chưa đầy đủ")
        print("Vui lòng download theo hướng dẫn:")
        print("   python download_dataset.py")
        print("=" * 50)
    
    return all_ok


def create_data_directory():
    """Tạo thư mục data nếu chưa tồn tại"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"✅ Created directory: {data_dir.absolute()}")
    return data_dir


def get_dataset_info():
    """Thông tin về VoiceBank + DEMAND dataset"""
    info = """
╔══════════════════════════════════════════════════════════════════════╗
║                     VOICEBANK + DEMAND DATASET                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📊 THỐNG KÊ:                                                        ║
║  ├─ Training: 11,572 utterances từ 28 speakers                      ║
║  └─ Testing:  824 utterances từ 2 speakers                          ║
║                                                                      ║
║  🔊 LOẠI NHIỄU (DEMAND):                                            ║
║  ├─ Domestic: tivi, máy giặt, bếp...                               ║
║  ├─ Office: bàn phím, máy in, điện thoại...                        ║
║  ├─ Transportation: tàu, xe, máy bay...                            ║
║  └─ Public: nhà hàng, công viên, đường phố...                      ║
║                                                                      ║
║  📈 SNR LEVELS: 0, 5, 10, 15 dB                                      ║
║                                                                      ║
║  🎵 AUDIO FORMAT:                                                    ║
║  ├─ Sample rate: 48 kHz (sẽ được resample về 16 kHz)               ║
║  ├─ Bit depth: 16-bit                                               ║
║  └─ Channels: Mono                                                   ║
║                                                                      ║
║  📚 CITATION:                                                        ║
║  Valentini-Botinhao et al., "Noisy speech database for training    ║
║  speech enhancement algorithms", 2016                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(info)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VoiceBank + DEMAND Dataset Helper')
    parser.add_argument('--check', action='store_true',
                        help='Kiểm tra xem dataset đã được download chưa')
    parser.add_argument('--info', action='store_true',
                        help='Hiển thị thông tin về dataset')
    parser.add_argument('--create_dir', action='store_true',
                        help='Tạo thư mục data')
    
    args = parser.parse_args()
    
    if args.check:
        check_dataset()
    elif args.info:
        get_dataset_info()
    elif args.create_dir:
        create_data_directory()
    else:
        print_download_instructions()
        get_dataset_info()


if __name__ == '__main__':
    main()
