#!/usr/bin/env python3
"""
Script kiểm tra chất lượng model - Phát hiện "Lazy Learning"

Lazy Learning là khi model chỉ giảm âm lượng thay vì thực sự lọc ồn.

Các tiêu chí kiểm tra:
1. Energy Ratio: Năng lượng output so với clean (nên gần 1.0)
2. Noise Reduction: Lượng noise được loại bỏ (nên > 50%)
3. SI-SDR: Chất lượng tín hiệu (càng cao càng tốt)
4. STOI: Độ hiểu của giọng nói (càng cao càng tốt)
5. PESQ: Perceptual quality (nếu có cài đặt)

Cải tiến:
- Sử dụng librosa thay vì torchaudio (Google Colab compatible)
- Thêm STOI và PESQ metrics
- Thêm post-processing detection
- Thêm volume matching analysis

Cách sử dụng:
    python test_model_quality.py --checkpoint checkpoints/best_model.pt
    python test_model_quality.py --checkpoint checkpoints/best_model.pt --noisy test.wav --clean clean.wav
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.unet import load_model_checkpoint
from utils.audio_utils import AudioProcessor, post_process_denoised, match_amplitude
from utils.metrics import calculate_si_sdr, calculate_stoi, is_pesq_available, calculate_pesq


def calculate_energy(signal: np.ndarray) -> float:
    """Calculate signal energy"""
    return np.sum(signal ** 2)


def calculate_rms(signal: np.ndarray) -> float:
    """Calculate RMS amplitude"""
    return np.sqrt(np.mean(signal ** 2))


def diagnose_lazy_learning(
    noisy: np.ndarray,
    clean: np.ndarray,
    output: np.ndarray,
    sample_rate: int = 16000
) -> dict:
    """
    Chẩn đoán xem model có đang "lazy learning" không
    
    Args:
        noisy: Noisy input signal
        clean: Clean reference signal
        output: Model output signal
        sample_rate: Sample rate
    
    Returns:
        Dictionary with diagnostic metrics
    """
    results = {}
    
    # 1. Energy ratio (output vs clean)
    # Nếu model chỉ giảm volume, energy_ratio sẽ < 1.0 nhiều
    output_energy = calculate_energy(output)
    clean_energy = calculate_energy(clean)
    noisy_energy = calculate_energy(noisy)
    
    results['energy_ratio_vs_clean'] = output_energy / (clean_energy + 1e-8)
    results['energy_ratio_vs_noisy'] = output_energy / (noisy_energy + 1e-8)
    
    # 2. RMS ratio
    output_rms = calculate_rms(output)
    clean_rms = calculate_rms(clean)
    noisy_rms = calculate_rms(noisy)
    
    results['rms_ratio_vs_clean'] = output_rms / (clean_rms + 1e-8)
    results['rms_ratio_vs_noisy'] = output_rms / (noisy_rms + 1e-8)
    
    # 3. Noise reduction ratio
    # noise_in = noisy - clean
    # noise_out = output - clean
    noise_in = noisy - clean
    noise_out = output - clean
    
    noise_in_energy = calculate_energy(noise_in)
    noise_out_energy = calculate_energy(noise_out)
    
    results['noise_reduction_ratio'] = 1 - (noise_out_energy / (noise_in_energy + 1e-8))
    
    # 4. SI-SDR
    results['si_sdr_input'] = calculate_si_sdr(clean, noisy)
    results['si_sdr_output'] = calculate_si_sdr(clean, output)
    results['si_sdr_improvement'] = results['si_sdr_output'] - results['si_sdr_input']
    
    # 5. Correlation với clean và noisy
    results['correlation_with_clean'] = np.corrcoef(output.flatten(), clean.flatten())[0, 1]
    results['correlation_with_noisy'] = np.corrcoef(output.flatten(), noisy.flatten())[0, 1]
    
    # 6. STOI (Short-Time Objective Intelligibility)
    try:
        results['stoi_input'] = calculate_stoi(clean, noisy, sample_rate)
        results['stoi_output'] = calculate_stoi(clean, output, sample_rate)
        results['stoi_improvement'] = results['stoi_output'] - results['stoi_input']
    except Exception as e:
        print(f"   STOI calculation failed: {e}")
        results['stoi_input'] = 0.0
        results['stoi_output'] = 0.0
        results['stoi_improvement'] = 0.0
    
    # 7. PESQ (if available)
    if is_pesq_available():
        try:
            results['pesq_input'] = calculate_pesq(clean, noisy, sample_rate)
            results['pesq_output'] = calculate_pesq(clean, output, sample_rate)
            if results['pesq_input'] and results['pesq_output']:
                results['pesq_improvement'] = results['pesq_output'] - results['pesq_input']
        except Exception as e:
            print(f"   PESQ calculation failed: {e}")
    
    return results


def interpret_results(results: dict) -> str:
    """Diễn giải kết quả chẩn đoán"""
    
    issues = []
    good_signs = []
    
    # 1. Check energy ratio
    energy_ratio = results['energy_ratio_vs_clean']
    if energy_ratio < 0.5:
        issues.append(f"⚠️ LAZY LEARNING: Output energy quá nhỏ ({energy_ratio:.2f}x so với clean)")
        issues.append("   → Model đang chỉ giảm volume thay vì lọc noise!")
    elif energy_ratio < 0.7:
        issues.append(f"⚠️ Output energy hơi thấp ({energy_ratio:.2f}x so với clean)")
    elif 0.8 <= energy_ratio <= 1.2:
        good_signs.append(f"✅ Energy ratio tốt: {energy_ratio:.2f}x")
    elif energy_ratio > 1.5:
        issues.append(f"⚠️ Output energy quá cao ({energy_ratio:.2f}x)")
    
    # 2. Check noise reduction
    noise_reduction = results['noise_reduction_ratio']
    if noise_reduction < 0.2:
        issues.append(f"⚠️ Noise reduction rất kém: chỉ giảm {noise_reduction*100:.1f}% noise")
    elif noise_reduction < 0.5:
        issues.append(f"⚠️ Noise reduction thấp: giảm {noise_reduction*100:.1f}% noise")
    else:
        good_signs.append(f"✅ Noise reduction: {noise_reduction*100:.1f}%")
    
    # 3. Check SI-SDR improvement
    sdr_improvement = results['si_sdr_improvement']
    if sdr_improvement < 0:
        issues.append(f"⚠️ SI-SDR giảm {abs(sdr_improvement):.2f} dB! Output tệ hơn input!")
    elif sdr_improvement < 3:
        issues.append(f"⚠️ SI-SDR cải thiện ít: chỉ +{sdr_improvement:.2f} dB")
    else:
        good_signs.append(f"✅ SI-SDR cải thiện: +{sdr_improvement:.2f} dB")
    
    # 4. Check STOI improvement (if available)
    if 'stoi_improvement' in results and results.get('stoi_output', 0) > 0:
        stoi_improvement = results['stoi_improvement']
        stoi_output = results['stoi_output']
        if stoi_improvement < -0.05:
            issues.append(f"⚠️ STOI giảm {abs(stoi_improvement):.3f}! Độ hiểu giảm!")
        elif stoi_output > 0.9:
            good_signs.append(f"✅ STOI rất tốt: {stoi_output:.3f} (cải thiện +{stoi_improvement:.3f})")
        elif stoi_output > 0.7:
            good_signs.append(f"✅ STOI khá: {stoi_output:.3f}")
        else:
            issues.append(f"⚠️ STOI thấp: {stoi_output:.3f}")
    
    # 5. Check PESQ improvement (if available)
    if 'pesq_improvement' in results:
        pesq_improvement = results['pesq_improvement']
        pesq_output = results.get('pesq_output', 0)
        if pesq_improvement < -0.2:
            issues.append(f"⚠️ PESQ giảm {abs(pesq_improvement):.2f}!")
        elif pesq_output > 3.5:
            good_signs.append(f"✅ PESQ tốt: {pesq_output:.2f} (cải thiện +{pesq_improvement:.2f})")
        elif pesq_output > 2.5:
            good_signs.append(f"✅ PESQ khá: {pesq_output:.2f}")
    
    # 6. Check correlation pattern
    corr_clean = results['correlation_with_clean']
    corr_noisy = results['correlation_with_noisy']
    
    if corr_noisy > corr_clean:
        issues.append(f"⚠️ Output giống noisy ({corr_noisy:.3f}) hơn clean ({corr_clean:.3f})")
        issues.append("   → Model chưa học được cách lọc noise!")
    else:
        good_signs.append(f"✅ Output giống clean ({corr_clean:.3f}) hơn noisy ({corr_noisy:.3f})")
    
    # Build interpretation
    interpretation = "\n" + "="*60 + "\n"
    interpretation += "CHẨN ĐOÁN MODEL\n"
    interpretation += "="*60 + "\n\n"
    
    if good_signs:
        interpretation += "🟢 DẤU HIỆU TỐT:\n"
        for sign in good_signs:
            interpretation += f"   {sign}\n"
        interpretation += "\n"
    
    if issues:
        interpretation += "🔴 VẤN ĐỀ:\n"
        for issue in issues:
            interpretation += f"   {issue}\n"
        interpretation += "\n"
        
        interpretation += "💡 GỢI Ý SỬA:\n"
        if any("LAZY LEARNING" in i for i in issues):
            interpretation += """
   1. Sử dụng SI-SDR loss (đã được thêm vào models/loss.py)
   2. Tăng si_sdr_weight trong config.yaml (0.5 → 1.0)
   3. Train lâu hơn (100-150 epochs)
   4. Giảm learning rate (0.0001 → 0.00005)
   5. Kiểm tra dataset có đúng không (clean phải thực sự sạch)
   6. Tăng energy_weight trong config.yaml (0.1 → 0.2)
"""
        elif any("cải thiện ít" in i for i in issues):
            interpretation += """
   1. Train thêm epochs (100-150 epochs)
   2. Tăng model capacity (encoder_channels lớn hơn)
   3. Kiểm tra xem val_loss có đang giảm không
   4. Thử tăng learning rate một chút
"""
        elif any("STOI" in i or "PESQ" in i for i in issues):
            interpretation += """
   1. Model có thể đang over-smooth tín hiệu
   2. Thử giảm magnitude_weight, tăng si_sdr_weight
   3. Kiểm tra STFT parameters (n_fft, hop_length)
"""
    else:
        interpretation += "🎉 Model hoạt động tốt!\n"
    
    return interpretation


def process_file(
    model,
    audio_processor: AudioProcessor,
    input_path: str,
    clean_path: str = None,
    device: torch.device = None,
    apply_postprocess: bool = True
) -> dict:
    """
    Process a single file and run diagnostics
    
    Args:
        model: UNet model
        audio_processor: AudioProcessor instance
        input_path: Path to noisy audio
        clean_path: Path to clean reference (optional)
        device: Torch device
        apply_postprocess: Apply post-processing to match input loudness
    
    Returns:
        Dictionary with audio arrays and diagnostics
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load noisy audio using librosa (Google Colab compatible)
    noisy_np, sr = librosa.load(input_path, sr=16000, mono=True)
    noisy_wav = torch.from_numpy(noisy_np).float().unsqueeze(0)  # [1, samples]
    
    # Load clean audio if available
    if clean_path and Path(clean_path).exists():
        clean_np, _ = librosa.load(clean_path, sr=16000, mono=True)
        clean_wav = torch.from_numpy(clean_np).float().unsqueeze(0)
    else:
        clean_wav = None
        clean_np = None
        print("⚠️ Không có clean reference, một số metrics sẽ không tính được")
    
    # Process with model
    model.eval()
    with torch.no_grad():
        noisy_stft = audio_processor.stft(noisy_wav)
        noisy_stft = noisy_stft.permute(0, 3, 1, 2).to(device)
        
        pred_stft = model(noisy_stft)
        
        pred_stft = pred_stft.permute(0, 2, 3, 1).cpu()
        output_wav = audio_processor.istft(pred_stft)
    
    # Convert to numpy
    output_np = output_wav.numpy().flatten()
    
    # Ensure same length
    min_len = min(len(noisy_np), len(output_np))
    noisy_np = noisy_np[:min_len]
    output_np = output_np[:min_len]
    
    # Apply post-processing to match input loudness
    # Điều này giúp tránh vấn đề "âm lượng giảm" sau khử nhiễu
    if apply_postprocess:
        output_np_original = output_np.copy()
        output_np = match_amplitude(output_np, noisy_np, method='rms')
        
        # Report loudness change
        rms_before = calculate_rms(output_np_original)
        rms_after = calculate_rms(output_np)
        if abs(rms_after - rms_before) / (rms_before + 1e-8) > 0.1:
            print(f"   📊 Post-processing: amplitude adjusted by {(rms_after/rms_before - 1)*100:+.1f}%")
    
    results = {
        'noisy': noisy_np,
        'output': output_np,
        'output_raw': output_np_original if apply_postprocess else output_np
    }
    
    if clean_np is not None:
        clean_np = clean_np[:min_len]
        results['clean'] = clean_np
        
        # Run diagnostics on BOTH raw and post-processed output
        results['diagnostics'] = diagnose_lazy_learning(noisy_np, clean_np, output_np)
        
        if apply_postprocess:
            results['diagnostics_raw'] = diagnose_lazy_learning(
                noisy_np, clean_np, output_np_original if apply_postprocess else output_np
            )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test model quality - Detect lazy learning')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--noisy', type=str, default=None,
                        help='Path to noisy audio file')
    parser.add_argument('--clean', type=str, default=None,
                        help='Path to clean reference audio file')
    parser.add_argument('--test_dir', type=str, default='./data/noisy_testset_wav',
                        help='Directory with test files (if no --noisy specified)')
    parser.add_argument('--clean_dir', type=str, default='./data/clean_testset_wav',
                        help='Directory with clean reference files')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--save_output', type=str, default=None,
                        help='Path to save output audio')
    
    args = parser.parse_args()
    
    print("="*60)
    print("KIỂM TRA CHẤT LƯỢNG MODEL - PHÁT HIỆN LAZY LEARNING")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model_checkpoint(args.checkpoint, device)
    model.eval()
    
    # Setup audio processor
    stft_cfg = config.get('stft', {})
    audio_processor = AudioProcessor(
        n_fft=stft_cfg.get('n_fft', 512),
        hop_length=stft_cfg.get('hop_length', 128),
        win_length=stft_cfg.get('win_length', 512)
    )
    
    # Process files
    if args.noisy:
        # Single file mode
        print(f"\nProcessing: {args.noisy}")
        results = process_file(model, audio_processor, args.noisy, args.clean, device)
        
        if 'diagnostics' in results:
            print("\n📊 KẾT QUẢ CHẨN ĐOÁN:")
            diag = results['diagnostics']
            print(f"   Energy ratio (vs clean): {diag['energy_ratio_vs_clean']:.3f}")
            print(f"   RMS ratio (vs clean): {diag['rms_ratio_vs_clean']:.3f}")
            print(f"   Noise reduction: {diag['noise_reduction_ratio']*100:.1f}%")
            print(f"   SI-SDR input: {diag['si_sdr_input']:.2f} dB")
            print(f"   SI-SDR output: {diag['si_sdr_output']:.2f} dB")
            print(f"   SI-SDR improvement: {diag['si_sdr_improvement']:+.2f} dB")
            print(f"   Correlation with clean: {diag['correlation_with_clean']:.3f}")
            print(f"   Correlation with noisy: {diag['correlation_with_noisy']:.3f}")
            
            print(interpret_results(diag))
        
        # Save output if requested
        if args.save_output:
            # Use soundfile for saving (Google Colab compatible)
            sf.write(args.save_output, results['output'], 16000)
            print(f"\n💾 Output saved to: {args.save_output}")
            
            # Also save raw (before post-processing) for comparison
            if 'output_raw' in results:
                raw_path = args.save_output.replace('.wav', '_raw.wav')
                sf.write(raw_path, results['output_raw'], 16000)
                print(f"💾 Raw output saved to: {raw_path}")
    
    else:
        # Batch mode - test multiple files
        test_dir = Path(args.test_dir)
        clean_dir = Path(args.clean_dir)
        
        if not test_dir.exists():
            print(f"⚠️ Test directory not found: {test_dir}")
            print("Vui lòng chỉ định --noisy hoặc download dataset trước")
            return
        
        test_files = list(test_dir.glob('*.wav'))[:args.num_samples]
        
        all_diagnostics = []
        
        for test_file in test_files:
            print(f"\nProcessing: {test_file.name}")
            
            # Find matching clean file
            clean_file = clean_dir / test_file.name
            clean_path = str(clean_file) if clean_file.exists() else None
            
            try:
                results = process_file(
                    model, audio_processor, 
                    str(test_file), clean_path, device
                )
                
                if 'diagnostics' in results:
                    diag = results['diagnostics']
                    all_diagnostics.append(diag)
                    
                    print(f"   Energy ratio: {diag['energy_ratio_vs_clean']:.3f}")
                    print(f"   Noise reduction: {diag['noise_reduction_ratio']*100:.1f}%")
                    print(f"   SI-SDR improvement: {diag['si_sdr_improvement']:+.2f} dB")
                    
            except Exception as e:
                print(f"   Error: {e}")
        
        # Summary
        if all_diagnostics:
            print("\n" + "="*60)
            print("TỔNG KẾT")
            print("="*60)
            
            avg_energy = np.mean([d['energy_ratio_vs_clean'] for d in all_diagnostics])
            avg_noise_reduction = np.mean([d['noise_reduction_ratio'] for d in all_diagnostics])
            avg_sdr_improvement = np.mean([d['si_sdr_improvement'] for d in all_diagnostics])
            
            print(f"\n📊 Trung bình trên {len(all_diagnostics)} files:")
            print(f"   Energy ratio: {avg_energy:.3f}")
            print(f"   Noise reduction: {avg_noise_reduction*100:.1f}%")
            print(f"   SI-SDR improvement: {avg_sdr_improvement:+.2f} dB")
            
            # Overall interpretation
            fake_diag = {
                'energy_ratio_vs_clean': avg_energy,
                'noise_reduction_ratio': avg_noise_reduction,
                'si_sdr_improvement': avg_sdr_improvement,
                'correlation_with_clean': np.mean([d['correlation_with_clean'] for d in all_diagnostics]),
                'correlation_with_noisy': np.mean([d['correlation_with_noisy'] for d in all_diagnostics])
            }
            print(interpret_results(fake_diag))


if __name__ == '__main__':
    main()
