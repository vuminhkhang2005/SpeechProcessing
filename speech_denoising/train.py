"""
Training script for Speech Denoising Model

Usage:
    python train.py --config config.yaml
    
    hoặc với các tham số tùy chỉnh:
    python train.py --batch_size 16 --epochs 100 --lr 0.0001
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import VoiceBankDEMANDDataset, create_dataloaders
from models.unet import UNetDenoiser
from models.loss import DenoiserLoss
from utils.metrics import evaluate_batch
from utils.audio_utils import AudioProcessor


class Trainer:
    """
    Trainer class for speech denoising model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training parameters
        train_cfg = config['training']
        self.num_epochs = train_cfg['num_epochs']
        self.grad_clip = train_cfg.get('grad_clip', 5.0)
        self.use_amp = train_cfg.get('use_amp', True)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        scheduler_cfg = train_cfg.get('scheduler', {})
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_cfg.get('factor', 0.5),
            patience=scheduler_cfg.get('patience', 5),
            min_lr=scheduler_cfg.get('min_lr', 1e-6)
        )
        
        # Loss function
        loss_cfg = config.get('loss', {})
        stft_cfg = config.get('stft', {})
        self.criterion = DenoiserLoss(
            complex_weight=loss_cfg.get('l1_weight', 1.0),
            magnitude_weight=1.0,
            stft_weight=loss_cfg.get('stft_weight', 0.5),
            use_mr_stft=True,
            n_fft=stft_cfg.get('n_fft', 512),
            hop_length=stft_cfg.get('hop_length', 128),
            win_length=stft_cfg.get('win_length', 512)
        ).to(device)
        
        # Audio processor for iSTFT
        self.audio_processor = AudioProcessor(
            n_fft=stft_cfg.get('n_fft', 512),
            hop_length=stft_cfg.get('hop_length', 128),
            win_length=stft_cfg.get('win_length', 512)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        log_dir = config['logging']['log_dir']
        self.log_dir = Path(log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.log_every = config['logging'].get('log_every', 100)
        
        # Checkpoints
        ckpt_cfg = config['checkpoint']
        self.ckpt_dir = Path(ckpt_cfg['save_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = ckpt_cfg.get('save_every', 5)
        self.keep_last = ckpt_cfg.get('keep_last', 3)
        
        # Early stopping
        self.early_stopping_patience = train_cfg.get('early_stopping_patience', 15)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total_loss': 0.0, 'complex_loss': 0.0, 'magnitude_loss': 0.0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Move to device
            noisy_stft = batch['noisy_stft'].to(self.device)
            clean_stft = batch['clean_stft'].to(self.device)
            noisy_wav = batch['noisy'].to(self.device)
            clean_wav = batch['clean'].to(self.device)
            
            # Reshape STFT: [batch, freq, time, 2] -> [batch, 2, freq, time]
            noisy_stft = noisy_stft.permute(0, 3, 1, 2)
            clean_stft = clean_stft.permute(0, 3, 1, 2)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    pred_stft = self.model(noisy_stft)
                    
                    # Reconstruct waveform for multi-resolution STFT loss
                    pred_stft_for_istft = pred_stft.permute(0, 2, 3, 1)  # [batch, freq, time, 2]
                    pred_wav = self.audio_processor.istft(pred_stft_for_istft)
                    
                    # Ensure same length
                    min_len = min(pred_wav.shape[-1], clean_wav.shape[-1])
                    pred_wav = pred_wav[..., :min_len]
                    clean_wav_trimmed = clean_wav[..., :min_len]
                    
                    losses = self.criterion(
                        pred_stft, clean_stft,
                        pred_wav, clean_wav_trimmed
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(losses['total_loss']).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_stft = self.model(noisy_stft)
                
                # Reconstruct waveform
                pred_stft_for_istft = pred_stft.permute(0, 2, 3, 1)
                pred_wav = self.audio_processor.istft(pred_stft_for_istft)
                
                min_len = min(pred_wav.shape[-1], clean_wav.shape[-1])
                pred_wav = pred_wav[..., :min_len]
                clean_wav_trimmed = clean_wav[..., :min_len]
                
                losses = self.criterion(
                    pred_stft, clean_stft,
                    pred_wav, clean_wav_trimmed
                )
                
                losses['total_loss'].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if self.global_step % self.log_every == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        val_losses = {'total_loss': 0.0, 'complex_loss': 0.0, 'magnitude_loss': 0.0}
        metrics = {'pesq': 0.0, 'stoi': 0.0, 'si_sdr': 0.0}
        num_batches = 0
        
        eval_cfg = self.config.get('eval', {})
        compute_pesq = eval_cfg.get('compute_pesq', True)
        compute_stoi = eval_cfg.get('compute_stoi', True)
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            noisy_stft = batch['noisy_stft'].to(self.device)
            clean_stft = batch['clean_stft'].to(self.device)
            clean_wav = batch['clean'].to(self.device)
            
            # Reshape
            noisy_stft = noisy_stft.permute(0, 3, 1, 2)
            clean_stft = clean_stft.permute(0, 3, 1, 2)
            
            # Forward
            pred_stft = self.model(noisy_stft)
            
            # Reconstruct waveform
            pred_stft_for_istft = pred_stft.permute(0, 2, 3, 1)
            pred_wav = self.audio_processor.istft(pred_stft_for_istft)
            
            min_len = min(pred_wav.shape[-1], clean_wav.shape[-1])
            pred_wav = pred_wav[..., :min_len]
            clean_wav_trimmed = clean_wav[..., :min_len]
            
            # Calculate losses
            losses = self.criterion(pred_stft, clean_stft)
            
            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()
            
            # Calculate metrics
            try:
                batch_metrics = evaluate_batch(
                    clean_wav_trimmed, pred_wav,
                    sample_rate=self.config['data']['sample_rate'],
                    compute_pesq=compute_pesq,
                    compute_stoi=compute_stoi
                )
                for key in metrics:
                    if key in batch_metrics:
                        metrics[key] += batch_metrics[key]
            except Exception as e:
                print(f"Metrics calculation error: {e}")
            
            num_batches += 1
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
        for key in metrics:
            metrics[key] /= num_batches
        
        return {**val_losses, **metrics}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        ckpt_path = self.ckpt_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, ckpt_path)
        
        # Save best model
        if is_best:
            best_path = self.ckpt_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        # Remove old checkpoints
        checkpoints = sorted(self.ckpt_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoints) > self.keep_last:
            for ckpt in checkpoints[:-self.keep_last]:
                ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, resume_from: Optional[str] = None):
        """Full training loop"""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        print(f"\nStarting training from epoch {self.current_epoch}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_results = self.validate()
            
            # Update scheduler
            val_loss = val_results['total_loss']
            self.scheduler.step(val_loss)
            
            # Log epoch results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  PESQ: {val_results.get('pesq', 0):.3f}")
            print(f"  STOI: {val_results.get('stoi', 0):.3f}")
            print(f"  SI-SDR: {val_results.get('si_sdr', 0):.2f} dB")
            
            # TensorBoard logging
            for key, value in train_losses.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            for key, value in val_results.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.ckpt_dir}")
        print(f"Logs saved to: {self.log_dir}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Speech Denoising Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Override config options
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    
    # Override with command line arguments
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("SPEECH DENOISING TRAINING")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print()
    
    # Create dataloaders
    data_cfg = config['data']
    stft_cfg = config['stft']
    
    try:
        train_loader, val_loader = create_dataloaders(
            train_clean_dir=data_cfg['train_clean_dir'],
            train_noisy_dir=data_cfg['train_noisy_dir'],
            test_clean_dir=data_cfg['test_clean_dir'],
            test_noisy_dir=data_cfg['test_noisy_dir'],
            sample_rate=data_cfg['sample_rate'],
            segment_length=data_cfg['segment_length'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            n_fft=stft_cfg['n_fft'],
            hop_length=stft_cfg['hop_length'],
            win_length=stft_cfg['win_length']
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease make sure you have downloaded the VoiceBank + DEMAND dataset.")
        print("Run: python -c \"from data.dataset import download_voicebank_demand; download_voicebank_demand()\"")
        return
    
    # Create model
    model_cfg = config['model']
    model = UNetDenoiser(
        in_channels=2,
        out_channels=2,
        encoder_channels=model_cfg.get('encoder_channels', [32, 64, 128, 256, 512]),
        use_attention=model_cfg.get('use_attention', True),
        dropout=model_cfg.get('dropout', 0.1),
        mask_type='CRM'
    )
    
    print(f"Model: {model_cfg['name']}")
    print(f"Parameters: {model.count_parameters():,}")
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
