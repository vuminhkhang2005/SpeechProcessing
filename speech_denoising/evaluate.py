"""
Evaluation script for Speech Denoising Model

Đánh giá model trên test set với các metrics: PESQ, STOI, SI-SDR

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --config config.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import VoiceBankDEMANDDataset
from models.unet import UNetDenoiser
from utils.audio_utils import AudioProcessor, save_audio
from utils.metrics import calculate_pesq, calculate_stoi, calculate_si_sdr, calculate_snr


class Evaluator:
    """
    Evaluator for speech denoising model
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: torch.device
    ):
        self.config = config
        self.device = device
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Audio processor
        stft_cfg = config.get('stft', {})
        self.audio_processor = AudioProcessor(
            n_fft=stft_cfg.get('n_fft', 512),
            hop_length=stft_cfg.get('hop_length', 128),
            win_length=stft_cfg.get('win_length', 512),
            sample_rate=config['data']['sample_rate']
        )
        
        self.sample_rate = config['data']['sample_rate']
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_cfg = self.config.get('model', {})
        
        model = UNetDenoiser(
            in_channels=2,
            out_channels=2,
            encoder_channels=model_cfg.get('encoder_channels', [32, 64, 128, 256, 512]),
            use_attention=model_cfg.get('use_attention', True),
            dropout=0.0,
            mask_type='CRM'
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
    
    @torch.no_grad()
    def evaluate_sample(
        self,
        noisy_stft: torch.Tensor,
        clean_wav: torch.Tensor,
        noisy_wav: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate a single sample
        
        Args:
            noisy_stft: Noisy STFT [freq, time, 2]
            clean_wav: Clean waveform [samples]
            noisy_wav: Noisy waveform [samples]
        
        Returns:
            Dictionary of metrics
        """
        # Add batch dimension
        noisy_stft = noisy_stft.unsqueeze(0).to(self.device)  # [1, freq, time, 2]
        noisy_stft = noisy_stft.permute(0, 3, 1, 2)  # [1, 2, freq, time]
        
        # Run model
        enhanced_stft = self.model(noisy_stft)
        
        # Convert to waveform
        enhanced_stft = enhanced_stft.permute(0, 2, 3, 1)  # [1, freq, time, 2]
        enhanced_wav = self.audio_processor.istft(enhanced_stft)
        enhanced_wav = enhanced_wav.squeeze(0).cpu().numpy()
        
        # Convert reference signals
        clean_wav = clean_wav.numpy()
        noisy_wav = noisy_wav.numpy()
        
        # Ensure same length
        min_len = min(len(enhanced_wav), len(clean_wav), len(noisy_wav))
        enhanced_wav = enhanced_wav[:min_len]
        clean_wav = clean_wav[:min_len]
        noisy_wav = noisy_wav[:min_len]
        
        # Calculate metrics
        metrics = {}
        
        # Enhanced vs Clean
        metrics['pesq'] = calculate_pesq(clean_wav, enhanced_wav, self.sample_rate)
        metrics['stoi'] = calculate_stoi(clean_wav, enhanced_wav, self.sample_rate)
        metrics['si_sdr'] = calculate_si_sdr(clean_wav, enhanced_wav)
        metrics['snr'] = calculate_snr(clean_wav, enhanced_wav)
        
        # Noisy vs Clean (baseline)
        metrics['pesq_noisy'] = calculate_pesq(clean_wav, noisy_wav, self.sample_rate)
        metrics['stoi_noisy'] = calculate_stoi(clean_wav, noisy_wav, self.sample_rate)
        metrics['si_sdr_noisy'] = calculate_si_sdr(clean_wav, noisy_wav)
        
        # Improvement
        metrics['pesq_improvement'] = metrics['pesq'] - metrics['pesq_noisy']
        metrics['stoi_improvement'] = metrics['stoi'] - metrics['stoi_noisy']
        metrics['si_sdr_improvement'] = metrics['si_sdr'] - metrics['si_sdr_noisy']
        
        return metrics, enhanced_wav
    
    def evaluate_dataset(
        self,
        test_dataset: VoiceBankDEMANDDataset,
        save_samples: bool = False,
        output_dir: str = './outputs'
    ) -> pd.DataFrame:
        """
        Evaluate on entire test dataset
        
        Args:
            test_dataset: Test dataset
            save_samples: Save enhanced audio samples
            output_dir: Directory to save samples
        
        Returns:
            DataFrame with per-sample metrics
        """
        if save_samples:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset[idx]
            
            noisy_stft = sample['noisy_stft']
            clean_wav = sample['clean']
            noisy_wav = sample['noisy']
            filename = sample['filename']
            
            # Evaluate
            metrics, enhanced_wav = self.evaluate_sample(
                noisy_stft, clean_wav, noisy_wav
            )
            metrics['filename'] = filename
            all_metrics.append(metrics)
            
            # Save sample
            if save_samples:
                save_audio(
                    str(output_dir / f'enhanced_{filename}'),
                    enhanced_wav,
                    self.sample_rate
                )
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        metrics = ['pesq', 'stoi', 'si_sdr', 'snr']
        
        print("\nEnhanced Speech Metrics:")
        print("-" * 40)
        for metric in metrics:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"  {metric.upper():>10}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\nNoisy Speech Metrics (Baseline):")
        print("-" * 40)
        for metric in ['pesq_noisy', 'stoi_noisy', 'si_sdr_noisy']:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            name = metric.replace('_noisy', '').upper()
            print(f"  {name:>10}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\nImprovement:")
        print("-" * 40)
        for metric in ['pesq_improvement', 'stoi_improvement', 'si_sdr_improvement']:
            mean_val = df[metric].mean()
            name = metric.replace('_improvement', '').upper()
            print(f"  Δ{name:>9}: {mean_val:+.4f}")
        
        print("\n" + "=" * 60)
        print(f"Total samples evaluated: {len(df)}")
        print("=" * 60)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Speech Denoising Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save enhanced audio')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save enhanced audio samples')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save results CSV')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    
    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Create test dataset
    data_cfg = config['data']
    stft_cfg = config['stft']
    
    test_dataset = VoiceBankDEMANDDataset(
        clean_dir=data_cfg['test_clean_dir'],
        noisy_dir=data_cfg['test_noisy_dir'],
        sample_rate=data_cfg['sample_rate'],
        segment_length=None,  # Full audio
        mode='test',
        n_fft=stft_cfg['n_fft'],
        hop_length=stft_cfg['hop_length'],
        win_length=stft_cfg['win_length']
    )
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device
    )
    
    # Evaluate
    results_df = evaluator.evaluate_dataset(
        test_dataset,
        save_samples=args.save_samples,
        output_dir=args.output_dir
    )
    
    # Print summary
    evaluator.print_summary(results_df)
    
    # Save results
    if args.save_results:
        results_df.to_csv(args.save_results, index=False)
        print(f"\nResults saved to: {args.save_results}")
    
    if args.save_samples:
        print(f"Enhanced audio saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
