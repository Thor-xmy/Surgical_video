"""
Training Script for Multi-Clip Bounded Surgical QA Model

完整支持：
1. Video-level DataLoader（返回整个视频）
2. Multi-clip特征提取（静态+动态）
3. Per-clip特征融合
4. 有界回归（Sigmoid，输出[0, 1））

Usage:
    python train_multiclip_bounded.py --data_root /path/to/data
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded, build_model_multiclip_bounded
# Modified to use frame sequence loader with proper data splitting
from utils.data_loader_video_level_frames import create_dataloader_with_split
from utils.training import AverageMeter, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Clip Bounded Surgical QA Model')
    parser.add_argument('--config', type=str, default='configs/bounded.yaml')
    parser.add_argument('--data_root', type=str, default=None, help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='output_multiclip_bounded')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--static_dim', type=int, default=512)
    parser.add_argument('--dynamic_dim', type=int, default=1024)
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--clip_stride', type=int, default=10)
    parser.add_argument('--max_clips', type=int, default=None)
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--use_mixed_conv', action='store_true', default=True)
    
    # 🌟 修改点 1：将 score_min 和 score_max 的默认值设为 None
    # 这样它们就不会强行覆盖你 bounded.yaml 里设置的 5.0 和 25.0 了
    parser.add_argument('--score_min', type=float, default=None)
    parser.add_argument('--score_max', type=float, default=None)
    
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=20)
    return parser.parse_args()


def load_config(config_path, args):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config_updates = {
        'data_root': args.data_root,
        #'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'static_dim': args.static_dim,
        'dynamic_dim': args.dynamic_dim,
        'clip_length': args.clip_length,
        'clip_stride': args.clip_stride,
        'max_clips': args.max_clips,
        #'freeze_backbone': args.freeze_backbone,
        'use_mixed_conv': args.use_mixed_conv,
        'score_min': args.score_min,
        'score_max': args.score_max,
        'use_pretrained': args.pretrained,
        'epochs': args.epochs,
        #'print_freq': args.print_freq,
        'save_freq': args.save_freq
    }

    for key, value in config_updates.items():
        if value is not None:
            config[key] = value

    return config


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

# 这里的双学习率代码你已经加得很完美了，完全保留
def build_optimizer(model, config):
    base_lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 假设动态特征提取器（I3D）的名字包含 'dynamic_extractor'
        if 'dynamic_extractor' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    # I3D 使用 base_lr (1e-4)，其他层（静态+回归头）使用 base_lr * 10 (1e-3)
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': base_lr},
        {'params': other_params, 'lr': base_lr * 10.0}
    ], weight_decay=weight_decay)

    return optimizer

def train_epoch(model, dataloader, optimizer, device, epoch, config, scaler=None):
    model.train()

    loss_meter = AverageMeter()
    pred_scores = []
    gt_scores = []

    print_freq = config.get('print_freq', 10)

    for batch_idx, batch in enumerate(dataloader):
        video = batch['video'].to(device)
        masks = batch.get('masks', None)
        if masks is not None:
            masks = masks.to(device)
        score_gt = batch['score'].to(device)

        batch_size = video.size(0)

        #score_pred = model(video, masks)
        #loss, loss_dict = model.compute_loss(score_pred, score_gt)

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        optimizer.zero_grad()
        
        if scaler is not None:
            # 🌟 开启混合精度：让模型在 float16 (半精度) 下跑，显存砍半！
            with autocast():
                score_pred = model(video, masks)
                loss, loss_dict = model.compute_loss(score_pred, score_gt)
            
            # 混合精度专用的反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 正常的 float32 全精度跑法
            score_pred = model(video, masks)
            loss, loss_dict = model.compute_loss(score_pred, score_gt)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), batch_size)
        pred_scores.extend(score_pred.detach().squeeze(-1).cpu().numpy())
        gt_scores.extend(score_gt.detach().cpu().numpy())

        if (batch_idx + 1) % print_freq == 0:
            print(f"Epoch [{epoch}] [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    metrics = compute_metrics(pred_scores, gt_scores)

    return loss_meter.avg, metrics


def validate(model, dataloader, device, config):
    model.eval()

    loss_meter = AverageMeter()
    pred_scores = []
    gt_scores = []

    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            masks = batch.get('masks', None)
            if masks is not None:
                masks = masks.to(device)
            score_gt = batch['score'].to(device)

            batch_size = video.size(0)

            score_pred = model(video, masks)
            loss, _ = model.compute_loss(score_pred, score_gt)

            loss_meter.update(loss.item(), batch_size)
            pred_scores.extend(score_pred.detach().squeeze(-1).cpu().numpy())
            gt_scores.extend(score_gt.detach().cpu().numpy())

    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    metrics = compute_metrics(pred_scores, gt_scores)

    return loss_meter.avg, metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, output_dir, filename='best.pth'):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': {
            'static_dim': model.static_dim,
            'dynamic_dim': model.dynamic_dim,
            'clip_length': model.clip_length,
            'clip_stride': model.clip_stride,
            'score_min': model.score_min,
            'score_max': model.score_max
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    args = parse_args()
    config = load_config(args.config, args)

    setup_seed(args.seed)
    device = setup_device(args.gpu)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    config_path = os.path.join(log_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("\n" + "="*70)
    print("Multi-Clip Bounded Surgical QA Model Training")
    print("="*70)
    print(f"Data root: {config['data_root']}")
    print(f"Output dir: {log_dir}")
    print("="*70 + "\n")

    model = SurgicalQAModelMultiClipBounded(config)
    model = model.to(device)
    model.count_parameters()

    optimizer = build_optimizer(model, config)

    start_epoch = 0
    best_metric = -float('inf')

    if args.resume is not None:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)  ################################################################
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # 🌟 修改点 2：将 YAML 里的 use_mask 开关透传给 DataLoader
    # 默认 True 表示如果你没在 YAML 里写这个参数，它就会像以前一样加载 Mask
    use_mask = config.get('use_mask', True)

    # Create dataloaders with proper data splitting (70/15/15 train/val/test)
    train_loader = create_dataloader_with_split(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        spatial_size=config.get('spatial_size', 112),
        clip_length=config['clip_length'],
        clip_stride=config['clip_stride'],
        score_min=config['score_min'],
        score_max=config['score_max'],
        # Data splitting parameters
        subset='train',
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15),
        split_seed=config.get('split_seed', 42),
        is_train=True,
        use_mask=use_mask  # 🌟 传入开关
    )

    val_loader = create_dataloader_with_split(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        spatial_size=config.get('spatial_size', 112),
        clip_length=config['clip_length'],
        clip_stride=config['clip_stride'],
        score_min=config['score_min'],
        score_max=config['score_max'],
        # Data splitting parameters
        subset='val',
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15),
        split_seed=config.get('split_seed', 42),
        is_train=False,
        use_mask=use_mask  # 🌟 传入开关
    )
    use_amp = config.get('use_amp', False)
    scaler = GradScaler() if use_amp else None
    print(f"AMP (Mixed Precision) enabled: {use_amp}")
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 70)

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config, scaler
        )

        val_loss, val_metrics = validate(
            model, val_loader, device, config
        )

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Train Spearman: {train_metrics['spearman']:.4f}")
        print(f"  Val Spearman: {val_metrics['spearman']:.4f}")
        print(f"  Train L2: {train_metrics['l2']:.4f}")
        print(f"  Val L2: {val_metrics['l2']:.4f}")
        print(f"  Train RL2: {train_metrics['rl2']:.4f}")
        print(f"  Val RL2: {val_metrics['rl2']:.4f}")

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/spearman', train_metrics['spearman'], epoch)
        writer.add_scalar('train/l2', train_metrics['l2'], epoch)
        writer.add_scalar('train/rl2', train_metrics['rl2'], epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/spearman', val_metrics['spearman'], epoch)
        writer.add_scalar('val/l2', val_metrics['l2'], epoch)
        writer.add_scalar('val/rl2', val_metrics['rl2'], epoch)

        if (epoch + 1) % config['save_freq'] == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, val_metrics,
                log_dir, f'checkpoint_epoch_{epoch + 1}.pth'
            )

        if val_metrics['spearman'] > best_metric:
            best_metric = val_metrics['spearman']
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, val_metrics,
                log_dir, 'best_model.pth'
            )

    print("\nTraining completed!")
    print(f"Best Spearman: {best_metric:.4f}")
    writer.close()


if __name__ == '__main__':
    main()