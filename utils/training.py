"""
Training Utilities for Surgical QA Model

Contains training loops, loss functions, and utility classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm.auto import tqdm


class AverageMeter:
    """
    Compute and store average and current value.
    """
    def __init__(self, name='', fmt=':f'):
        """
        Args:
            name: Name of the metric
            fmt: Format string for printing
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val, n=1):
        """
        Update statistics with new value.

        Args:
            val: New value to add
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
        self.history.append(val)

    def get_average(self):
        """Get current average."""
        return self.avg

    def get_std(self):
        """Get standard deviation of history."""
        if len(self.history) < 2:
            return 0.0
        return np.std(self.history)

    def __str__(self):
        """String representation."""
        if self.count > 0:
            return f"{self.name} {self.avg:{self.fmt}} (n={self.count})"
        return f"{self.name} N/A"


def train_epoch(model, dataloader, optimizer, criterion, device,
                epoch, grad_scaler=None, accumulation_steps=1,
                clip_grad_norm=1.0, use_amp=False,
                log_frequency=10, verbose=True):
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        grad_scaler: Optional gradient scaler for AMP
        accumulation_steps: Gradient accumulation steps
        clip_grad_norm: Max gradient norm (0 = no clipping)
        use_amp: Use automatic mixed precision
        log_frequency: Print every N steps
        verbose: Print progress

    Returns:
        train_loss: Average training loss
        metrics_dict: Dict of training metrics
    """
    model.train()
    # Handle Subset wrapper: access underlying dataset's is_train attribute
    if hasattr(dataloader.dataset, 'dataset'):
        dataloader.dataset.dataset.is_train = True
    else:
        dataloader.dataset.is_train = True

    # Reset meters
    loss_meter = AverageMeter(name='Loss')
    score_loss_meter = AverageMeter(name='Score Loss')
    mask_loss_meter = AverageMeter(name='Mask Loss')

    # Time tracking
    epoch_start = time.time()
    data_start = time.time()
    total_steps = len(dataloader)

    if verbose:
        print(f"\nEpoch {epoch} - Training")
        print(f"Steps: {total_steps}")
        print("-" * 60)

    optimizer.zero_grad()

    with tqdm(dataloader, leave=False, disable=not verbose) as pbar:
        for step_idx, batch in enumerate(pbar):
            data_time = time.time() - data_start
            iter_start = time.time()

            # Move to device
            video = batch['frames'].to(device)  # (B, C, T, H, W)
            score_gt = batch['score'].to(device)  # (B,)

            # Load masks
            if 'masks' in batch and batch['masks'] is not None:
                masks = batch['masks'].to(device)  # (B, T, H, W)
            else:
                masks = None

            # Forward pass
            if use_amp and grad_scaler is not None:
                with autocast():
                    score_pred, mask_loss = model(video, masks, return_features=False, return_attention=False)

                    # Compute loss
                    if mask_loss is not None:
                        total_loss, loss_dict = model.compute_loss(score_pred, score_gt, mask_loss)
                    else:
                        total_loss, loss_dict = model.compute_loss(score_pred, score_gt)

                # Backward pass with AMP
                total_loss = total_loss / accumulation_steps
                grad_scaler.scale(total_loss).backward()
            else:
                score_pred, mask_loss = model(video, masks, return_features=False, return_attention=False)

                # Compute loss
                if mask_loss is not None:
                    total_loss, loss_dict = model.compute_loss(score_pred, score_gt, mask_loss)
                else:
                    total_loss, loss_dict = model.compute_loss(score_pred, score_gt)

                # Backward pass
                total_loss = total_loss / accumulation_steps
                total_loss.backward()

            # Update meters
            loss_meter.update(loss_dict['total_loss'])
            score_loss_meter.update(loss_dict['score_loss'])
            if 'mask_loss' in loss_dict:
                mask_loss_meter.update(loss_dict['mask_loss'])

            # Gradient accumulation
            if (step_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if clip_grad_norm > 0:
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                # Optimizer step
                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Timing
            iter_time = time.time() - iter_start
            data_start = time.time()

            # Logging
            if verbose and (step_idx + 1) % log_frequency == 0:
                pbar.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}",
                    's_loss': f"{score_loss_meter.avg:.4f}",
                    'm_loss': f"{mask_loss_meter.avg:.4f}",
                    'time': f"{iter_time:.3f}s"
                })

    # Final step for gradient accumulation
    if total_steps % accumulation_steps != 0:
        if clip_grad_norm > 0:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

    # Epoch summary
    epoch_time = time.time() - epoch_start

    metrics_dict = {
        'epoch': epoch,
        'train_loss': loss_meter.get_average(),
        'train_score_loss': score_loss_meter.get_average(),
        'train_mask_loss': mask_loss_meter.get_average(),
        'epoch_time': epoch_time
    }

    if verbose:
        print("-" * 60)
        print(f"Epoch {epoch} - Training Summary")
        print("-" * 60)
        print(f"  Loss: {loss_meter.get_average():.4f}")
        print(f"  Score Loss: {score_loss_meter.get_average():.4f}")
        print(f"  Mask Loss: {mask_loss_meter.get_average():.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 60)

    return loss_meter.get_average(), metrics_dict


@torch.inference_mode()
def validate(model, dataloader, criterion, device,
             epoch=None, verbose=True, return_predictions=False):
    """
    Validate model.

    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number (optional)
        verbose: Print progress
        return_predictions: Return predictions for metrics

    Returns:
        val_loss: Average validation loss
        metrics_dict: Dict of validation metrics
    """
    model.eval()

    # Reset meters
    loss_meter = AverageMeter(name='Val Loss')
    score_loss_meter = AverageMeter(name='Val Score Loss')
    mask_loss_meter = AverageMeter(name='Val Mask Loss')

    # Store predictions
    all_predictions = []
    all_targets = []

    # Time tracking
    val_start = time.time()

    if verbose:
        print(f"\nValidation")
        print("-" * 60)

    with torch.no_grad():
        with tqdm(dataloader, leave=False, disable=not verbose) as pbar:
            for batch in pbar:
                # Move to device
                video = batch['frames'].to(device)  # (B, C, T, H, W)
                score_gt = batch['score'].to(device)  # (B,)

                # Load masks
                if 'masks' in batch and batch['masks'] is not None:
                    masks = batch['masks'].to(device)  # (B, T, H, W)
                else:
                    masks = None

                # Forward pass
                score_pred, mask_loss = model(video, masks, return_features=False, return_attention=False)

                # Compute loss
                if mask_loss is not None:
                    total_loss, loss_dict = model.compute_loss(score_pred, score_gt, mask_loss)
                else:
                    total_loss, loss_dict = model.compute_loss(score_pred, score_gt)

                # Update meters
                loss_meter.update(loss_dict['total_loss'])
                score_loss_meter.update(loss_dict['score_loss'])
                if 'mask_loss' in loss_dict:
                    mask_loss_meter.update(loss_dict['mask_loss'])

                # Store predictions
                all_predictions.append(score_pred.detach().cpu())
                all_targets.append(score_gt.detach().cpu())

                # Logging
                if verbose:
                    pbar.set_postfix({
                        'loss': f"{loss_meter.avg:.4f}",
                        's_loss': f"{score_loss_meter.avg:.4f}"
                    })

    # Combine predictions
    all_predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.tensor([])
    all_targets = torch.cat(all_targets, dim=0) if all_targets else torch.tensor([])

    # Compute detailed metrics
    metrics_dict = {
        'val_loss': loss_meter.get_average(),
        'val_score_loss': score_loss_meter.get_average(),
        'val_mask_loss': mask_loss_meter.get_average(),
        'predictions': all_predictions.numpy() if return_predictions else None,
        'targets': all_targets.numpy() if return_predictions else None
    }

    # Validation
    val_time = time.time() - val_start

    if verbose:
        print("-" * 60)
        print(f"Validation Summary")
        print("-" * 60)
        print(f"  Loss: {loss_meter.get_average():.4f}")
        print(f"  Score Loss: {score_loss_meter.get_average():.4f}")
        print(f"  Mask Loss: {mask_loss_meter.get_average():.4f}")
        print(f"  Time: {val_time:.2f}s")
        print("-" * 60)

    return loss_meter.get_average(), metrics_dict


class TrainingLogger:
    """
    Log training progress to file and console.
    """
    def __init__(self, log_dir='logs', log_file='training.log'):
        """
        Args:
            log_dir: Directory to save logs
            log_file: Name of log file
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)

        # Initialize log file
        with open(self.log_path, 'w') as f:
            f.write("Training Log\n")
            f.write("="*60 + "\n")

        self.metrics_history = []

    def log_epoch(self, epoch, train_metrics, val_metrics=None):
        """
        Log epoch summary.

        Args:
            epoch: Epoch number
            train_metrics: Dict of training metrics
            val_metrics: Dict of validation metrics (optional)
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        }

        self.metrics_history.append(log_entry)

        # Write to file
        with open(self.log_path, 'a') as f:
            f.write(f"\n[{timestamp}] Epoch {epoch}\n")

            # Training metrics
            f.write("Training:\n")
            for key, val in train_metrics.items():
                if isinstance(val, float):
                    f.write(f"  {key}: {val:.4f}\n")
                else:
                    f.write(f"  {key}: {val}\n")

            # Validation metrics
            if val_metrics is not None:
                f.write("\nValidation:\n")
                for key, val in val_metrics.items():
                    if isinstance(val, float):
                        f.write(f"  {key}: {val:.4f}\n")
                    else:
                        f.write(f"  {key}: {val}\n")

            f.write("-"*60 + "\n")

    def save_history(self, save_path='training_history.json'):
        """Save complete training history."""
        with open(os.path.join(os.path.dirname(self.log_path), save_path), 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"Training history saved to {save_path}")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.compare = lambda current, best: current < best - min_delta
        elif mode == 'max':
            self.compare = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __call__(self, score):
        """
        Check if should stop training.

        Args:
            score: Current metric value

        Returns:
            stop: True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.compare(score, self.best_score):
            # Improvement found
            self.best_score = score
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def save_checkpoint(model, optimizer, epoch, metrics, save_dir='checkpoints',
                 filename='checkpoint.pth', is_best=False):
    """
    Save model checkpoint.

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dict of metrics
        save_dir: Directory to save checkpoint
        filename: Name of checkpoint file
        is_best: Whether this is the best model
    """
    os.makedirs(save_dir, exist_ok=True)

    if is_best:
        filename = 'best_' + filename
        print(f"Saving best model at epoch {epoch}...")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Neural network model
        optimizer: Optimizer (optional)
        device: Device to load on

    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics dict from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, metrics


if __name__ == '__main__':
    # Test training utilities
    print("Training utilities test passed successfully!")
