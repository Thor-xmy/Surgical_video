"""
LOSO-CV Training Script for Surgical QA Model (Bounded Version)

留一受试者组交叉验证 (Leave-One-Subject-Group-Out Cross-Validation)

数据集结构:
    JIGSAWS有9个受试者组: B, C, D, E, F, G, H, I (J组数据不完整，不使用)

划分策略:
    - 测试集: 1个受试者组的所有视频 (例如: B001-B005)
    - 验证集: 1个受试者组的所有视频 (例如: C001-C005)
    - 训练集: 剩余6个受试者组的所有视频

    共8折交叉验证 (B, C, D, E, F, G, H, I)

Usage:
    # 运行完整的LOSO-CV（8折）
    python train_loso_cv.py --data_root /path/to/jigsawas_frames --output_dir output_loso_cv

    # 仅运行特定折
    python train_loso_cv.py --data_root /path/to/jigsawas_frames --fold B
    python train_loso_cv.py --data_root /path/to/jigsawas_frames --fold C

    # 仅评估（不训练）
    python train_loso_cv.py --data_root /path/to/jigsawas_frames --fold B --evaluate --checkpoint checkpoints/losocv_B/best_model.pth
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from datetime import datetime
import json
import glob
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.surgical_qa_model_bounded import SurgicalQAModelBounded, build_model_bounded
from utils.training import (
    AverageMeter,
    train_epoch,
    validate,
    TrainingLogger,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint
)
from utils.metrics import compute_metrics
from utils.data_loader_jigsawas import JIGSAWSLOSODataLoader

# JIGSAWS受试者组 (标准LOSO-CV设置)
JIGSAWS_SUBJECT_GROUPS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Surgical QA Model with LOSO-CV (Bounded Version)')

    # Data and paths
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of JIGSAWS frame data (jigsawas_frames)')
    parser.add_argument('--labels_root', type=str, default=None,
                       help='Root directory of JIGSAWS labels (jigsawas/labels)')
    parser.add_argument('--output_dir', type=str, default='output_loso_cv',
                       help='Output directory for logs and checkpoints')
    parser.add_argument('--fold', type=str, default=None,
                       choices=JIGSAWS_SUBJECT_GROUPS + ['all'],
                       help='Subject group to leave out for testing (or "all" for all folds)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
            help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'adamw'], help='Optimizer')

    # Model parameters
    parser.add_argument('--static_dim', type=int, default=512,
                       help='Static feature dimension')
    parser.add_argument('--dynamic_dim', type=int, default=1024,
                       help='Dynamic feature dimension')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze backbone weights')
    parser.add_argument('--use_mask_loss', action='store_true', default=True,
                       help='Use mask supervision loss')

    # AMP and mixed precision
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Use automatic mixed precision')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum delta for improvement')

    # Checkpoint and evaluation
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--resume_optim', action='store_true', default=False,
                       help='Resume optimizer state')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='Evaluate only (requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbones')

    # Device
    parser.add_argument('--gpus', type=str, default='0',
                       help='GPU IDs to use (comma-separated)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Logging
    parser.add_argument('--print_freq', type=int, default=10,
                       help='Print every N steps')
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='TensorBoard log every N steps')

    # Dataset parameters
    parser.add_argument('--clip_length', type=int, default=16,
                       help='Number of frames per clip')
    parser.add_argument('--clip_stride', type=int, default=10,
                       help='Stride for clip extraction')
    parser.add_argument('--spatial_size', type=int, default=112,
                       help='Spatial size for resizing')
    parser.add_argument('--score_min', type=float, default=6.0,
                       help='Original score minimum (GRS: 6)')
    parser.add_argument('--score_max', type=float, default=30.0,
                       help='Original score maximum (GRS: 30)')

    return parser.parse_args()


def extract_subject_from_video_id(video_id):
    """
    从视频ID中提取受试者组

    Examples:
        Suturing_B001_capture1_stride3_fps10 -> B
        Needle_Passing_C002_capture1_stride3_fps10 -> C
    """
    # 查找模式: _[A-Z][0-9]{3}_
    import re
    match = re.search(r'_([A-Z])[0-9]{3}_', video_id)
    if match:
        return match.group(1)
    return None


def get_subject_group_videos(data_root, subject_group):
    """
    获取指定受试者组的所有视频

    Args:
        data_root: 数据集根目录
        subject_group: 受试者组 (如 'B', 'C', ...)

    Returns:
        List of video IDs
    """
    video_ids = []

    # 搜索所有任务目录
    task_dirs = ['Knot_tying', 'Needle_Passing', 'Suturing']

    for task in task_dirs:
        task_path = os.path.join(data_root, task)
        if not os.path.exists(task_path):
            continue

        # 搜索该受试者组的视频
        # 模式: TaskName_[Group]001_capture1_stride3_fps10
        pattern = f"*_{subject_group}[0-9][0-9][0-9]_*"
        for video_dir in glob.glob(os.path.join(task_path, pattern)):
            video_id = os.path.basename(video_dir)
            video_ids.append(video_id)

    return sorted(video_ids)


def analyze_dataset_by_subject(data_root):
    """
    分析数据集按受试者组的分布

    Args:
        data_root: 数据集根目录

    Returns:
        Dict: {subject_group: [video_ids1, video_ids2, ...]}
    """
    # 获取所有视频
    task_dirs = ['Knot_tying', 'Needle_Passing', 'Suturing']
    all_video_ids = []

    for task in task_dirs:
        task_path = os.path.join(data_root, task)
        if not os.path.exists(task_path):
            continue
        for video_dir in glob.glob(os.path.join(task_path, "*")):
            video_id = os.path.basename(video_dir)
            all_video_ids.append(video_id)

    # 按受试者组分组
    subject_groups = defaultdict(list)
    for video_id in all_video_ids:
        subject = extract_subject_from_video_id(video_id)
        if subject:
            subject_groups[subject].append(video_id)

    # 按受试者组排序
    sorted_groups = {}
    for group in sorted(subject_groups.keys()):
        sorted_groups[group] = sorted(subject_groups[group])

    return sorted_groups


def setup_device(gpus):
    """Setup device and multi-GPU."""
    gpu_list = [int(x) for x in gpus.split(',')]

    if torch.cuda.is_available():
        if len(gpu_list) == 1:
            device = torch.device(f'cuda:{gpu_list[0]}')
            print(f"Using single GPU: {gpu_list[0]}")
        else:
            device = torch.device(f'cuda:{gpu_list[0]}')
            print(f"Using multi-GPU: {gpu_list}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def build_optimizer(model, config):
    """Build optimizer with differential learning rates."""
    backbone_lr = config.get('backbone_lr', 1e-4)
    static_lr = config.get('static_lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'dynamic_extractor' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

    param_groups = []

    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': static_lr,
            'name': 'static_module'
        })

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found!")

    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            param_groups,
            weight_decay=weight_decay
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    return optimizer


def build_scheduler(optimizer, config):
    """Build learning rate scheduler."""
    T_max = config.get('epochs', 100)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=1e-6
    )

    return scheduler


def train_single_fold(model, dataloaders, optimizer, scheduler, criterion, device,
                      config, logger, tensorboard_writer=None, fold_name=''):
    """Train for a single fold."""
    start_epoch = 0
    best_val_loss = float('inf')
    grad_scaler = GradScaler() if config['use_amp'] else None

    if config['early_stopping']:
        early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001),
            mode='min'
        )

    checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"STARTING TRAINING - Fold: {fold_name}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, config['epochs']):
        train_loss, train_metrics = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            grad_scaler=grad_scaler,
            use_amp=config['use_amp'],
            log_frequency=config['print_freq'],
            verbose=True
        )

        scheduler.step()

        val_loss, val_metrics = validate(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion,
            device=device,
            epoch=epoch,
            verbose=True,
            return_predictions=True
        )

        # 反归一化预测和目标，用于计算指标
        if val_metrics['predictions'] is not None:
            predictions = val_metrics['predictions']
            targets = val_metrics['targets']

            if hasattr(model, 'denormalize_score'):
                predictions_denorm = model.denormalize_score(predictions)
                targets_denorm = model.denormalize_score(targets)
                eval_metrics = compute_metrics(predictions_denorm, targets_denorm, verbose=False)
            else:
                eval_metrics = compute_metrics(predictions, targets, verbose=False)

            if tensorboard_writer is not None:
                global_step = epoch * len(dataloaders['train'])
                tensorboard_writer.add_scalar(f'{fold_name}/train/loss', train_metrics['train_loss'], global_step)
                tensorboard_writer.add_scalar(f'{fold_name}/val/loss', val_metrics['val_loss'], epoch)
                tensorboard_writer.add_scalar(f'{fold_name}/val/mae', eval_metrics['mae'], epoch)
                tensorboard_writer.add_scalar(f'{fold_name}/val/srcc', eval_metrics['srcc'], epoch)
                tensorboard_writer.add_scalar(f'{fold_name}/val/pcc', eval_metrics['pcc'], epoch)

        train_log = {
            'loss': train_metrics['train_loss'],
            'score_loss': train_metrics.get('train_score_loss', train_metrics['train_loss']),
            'epoch_time': train_metrics.get('epoch_time', 0.0)
        }

        val_log = {
            'loss': val_metrics['val_loss'],
            'score_loss': val_metrics.get('val_score_loss', val_metrics['val_loss'])
        }

        if val_metrics['predictions'] is not None:
            val_log.update({
                'mae': eval_metrics['mae'],
                'srcc': eval_metrics['srcc'],
                'pcc': eval_metrics['pcc'],
                'rmse': eval_metrics.get('rmse', 0.0)
            })

        logger.log_epoch(epoch, train_log, val_log)

        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            checkpoint_metrics = {'train': train_log, 'val': val_log}
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=checkpoint_metrics,
                save_dir=checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch}.pth',
                is_best=(val_metrics['val_loss'] < best_val_loss)
            )

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            # Save best model
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'fold': fold_name
            }, best_model_path)
            print(f"  *** New best validation loss: {best_val_loss:.4f} ***")

        if config['early_stopping']:
            if early_stopping(val_metrics['val_loss']):
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    logger.save_history()

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED - Fold: {fold_name}")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}")

    return {
        'best_val_loss': best_val_loss,
        'final_epoch': epoch
    }


def evaluate_on_test_set(model, dataloader, device, fold_name):
    """
    在测试集上评估模型

    Returns:
        Dict: 评估指标和预测结果
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION ON TEST SET - Fold: {fold_name}")
    print(f"{'='*70}")

    model.eval()
    all_predictions = []
    all_targets = []
    all_video_ids = []

    with torch.no_grad():
        for batch in dataloader:
            video = batch['frames'].to(device)
            score_gt = batch['score'].to(device)

            if 'masks' in batch and batch['masks'] is not None:
                masks = batch['masks'].to(device)
            else:
                masks = None

            score_pred, _ = model(video, masks)

            all_predictions.append(score_pred.detach().cpu())
            all_targets.append(score_gt.detach().cpu())

            if 'video_id' in batch:
                if isinstance(batch['video_id'], list):
                    all_video_ids.extend(batch['video_id'])
                else:
                    all_video_ids.append(batch['video_id'])

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 反归一化用于指标计算
    if hasattr(model, 'denormalize_score'):
        predictions_denorm = model.denormalize_score(all_predictions)
        targets_denorm = model.denormalize_score(all_targets)
    else:
        predictions_denorm = all_predictions
        targets_denorm = all_targets

    # 计算指标
    predictions_np = predictions_denorm.numpy()
    targets_np = targets_denorm.numpy()

    mae = np.mean(np.abs(predictions_np - targets_np))
    rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))
    nmae = mae / (np.max(targets_np) - np.min(targets_np))

    # Spearman相关系数
    srcc, p_value_s = spearmanr(targets_np, predictions_np)
    # Pearson相关系数
    pcc, p_value_p = pearsonr(targets_np, predictions_np)

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'nmae': float(nmae),
        'srcc': float(srcc),
        'pcc': float(pcc),
        'p_value_s': float(p_value_s),
        'p_value_p': float(p_value_p) if 'p_value_p' in locals() else 0.0
    }

    print(f"\nTest Set Metrics (Fold: {fold_name}):")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  NMAE: {metrics['nmae']:.4f}")
    print(f"  SRCC: {metrics['srcc']:.4f} (p={metrics['p_value_s']:.4f})")
    print(f"  PCC:  {metrics['pcc']:.4f} (p={metrics['p_value_p']:.4f})")

    # 返回详细预测结果
    if len(all_video_ids) > 0:
        predictions_per_video = []
        start_idx = 0
        for video_id in all_video_ids:
            num_clips = len([v for v in all_video_ids if v == video_id])
            if num_clips == 0:
                continue
            end_idx = start_idx + num_clips
            pred_mean = np.mean(predictions_np[start_idx:end_idx])
            gt_mean = np.mean(targets_np[start_idx:end_idx])
            predictions_per_video.append({
                'video_id': video_id,
                'prediction': float(pred_mean),
                'ground_truth': float(gt_mean),
                'error': abs(float(pred_mean - gt_mean))
            })
            start_idx = end_idx

        metrics['predictions_per_video'] = predictions_per_video

    return metrics


def create_loso_dataloaders(data_root, test_subject_group, batch_size=8, num_workers=4, dataset_kwargs=None):
    """
    创建LOSO-CV的数据加载器

    Args:
        data_root: 数据集根目录
        test_subject_group: 测试集的受试者组 (如 'B')
        batch_size: 批次大小
        num_workers: 工作进程数
        dataset_kwargs: 数据集参数

    Returns:
        Dict: {'train': train_loader, 'val': val_loader, 'test': test_loader,
               'train_subjects': [...], 'val_subjects': [...], 'test_subjects': [...]}
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    # 分析数据集
    subject_groups = analyze_dataset_by_subject(data_root)

    print(f"\n{'='*70}")
    print("数据集受试者组分析")
    print(f"{'='*70}")
    for group, videos in sorted(subject_groups.items()):
        print(f"  {group}: {len(videos)} videos")

    # 确定验证集受试者组 (使用循环下一个)
    group_list = sorted(subject_groups.keys())
    test_idx = group_list.index(test_subject_group)
    val_idx = (test_idx + 1) % len(group_list)
    val_subject_group = group_list[val_idx]

    # 确定训练集受试者组 (剩余所有)
    train_subject_groups = [g for g in group_list if g not in [test_subject_group, val_subject_group]]

    print(f"\nLOSO-CV划分 (测试集: {test_subject_group}):")
    print(f"  测试集: {test_subject_group}001-{test_subject_group}005")
    print(f"  验证集: {val_subject_group}001-{val_subject_group}005")
    print(f"  训练集: {', '.join([f'{g}001-{g}005' for g in train_subject_groups])}")

    # 获取各集合的视频ID
    test_videos = subject_groups[test_subject_group]
    val_videos = subject_groups[val_subject_group]
    train_videos = []
    for group in train_subject_groups:
        train_videos.extend(subject_groups[group])

    print(f"\n视频数量:")
    print(f"  训练集: {len(train_videos)} videos")
    print(f"  验证集: {len(val_videos)} videos")
    print(f"  测试集: {len(test_videos)} videos")

    # 导入必要的类
    from utils.data_loader_normalized import SurgicalVideoDatasetNormalized

    # 创建完整数据集 (用于索引)
    full_dataset = SurgicalVideoDatasetNormalized(data_root, **dataset_kwargs)

    # 创建数据集子集类 (在full_dataset创建后定义)
    class VideoSubsetDataset(Dataset):
        def __init__(self, full_dataset, video_ids):
            self.full_dataset = full_dataset
            self.video_ids = set(video_ids)
            self.indices = []

            # 遍历full_dataset的clips，筛选属于指定视频的clips
            if hasattr(full_dataset, 'cache_clips') and full_dataset.cache_clips:
                for idx, clip in enumerate(full_dataset.clips):
                    if clip['video_id'] in self.video_ids:
                        self.indices.append(idx)

            print(f"  选择的视频数: {len(video_ids)}")
            print(f"  选择的clips数: {len(self.indices)}")

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.full_dataset[self.indices[idx]]

    # 创建子集
    train_dataset = VideoSubsetDataset(full_dataset, train_videos)
    val_dataset = VideoSubsetDataset(full_dataset, val_videos)
    test_dataset = VideoSubsetDataset(full_dataset, test_videos)

    # 设置is_train标志
    train_dataset.full_dataset.is_train = True
    val_dataset.full_dataset.is_train = False
    test_dataset.full_dataset.is_train = False

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch=1,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_subjects': train_subject_groups,
        'val_subjects': [val_subject_group],
        'test_subjects': [test_subject_group],
        'train_video_count': len(train_videos),
        'val_video_count': len(val_videos),
        'test_video_count': len(test_videos)
    }


def run_single_fold(args, test_subject_group, config):
    """
    运行单个fold的训练和评估

    Args:
        args: 命令行参数
        test_subject_group: 测试集受试者组
        config: 配置字典

    Returns:
        Dict: 该fold的结果
    """
    fold_name = f"loso_{test_subject_group}"
    fold_output_dir = os.path.join(args.output_dir, fold_name)
    os.makedirs(fold_output_dir, exist_ok=True)

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    device = setup_device(args.gpus)

    # 构建数据集参数
    dataset_kwargs = {
        'clip_length': args.clip_length,
        'clip_stride': args.clip_stride,
        'spatial_size': args.spatial_size,
        'normalize': True,
        'horizontal_flip_prob': 0.5,
        'enable_rotation': True,
        'is_train': True,
        'score_min': args.score_min,
        'score_max': args.score_max,
        'target_min': 0.0,
        'target_max': 1.0,
        'use_cache': True
    }

    # 确定labels_root
    labels_root = args.labels_root
    if labels_root is None:
        # 默认路径：data_root的父目录 + 'labels'
        labels_root = os.path.join(os.path.dirname(os.path.dirname(args.data_root)), 'labels')

    # 创建LOSO-CV数据加载器
    dataloaders = JIGSAWSLOSODataLoader(
        data_root=args.data_root,
        labels_root=labels_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_subject_group=test_subject_group,
        dataset_kwargs=dataset_kwargs
    )

    # 评估模式
    if args.evaluate:
        if args.checkpoint is None:
            print("Error: --checkpoint is required for evaluation mode")
            return None

        # 加载模型
        model = build_model_bounded(config)
        model = model.to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")

        # 在测试集上评估
        test_metrics = evaluate_on_test_set(
            model=model,
            dataloader=dataloaders.get_loader('test'),
            device=device,
            fold_name=fold_name
        )

        return {
            'fold': fold_name,
            'test_subject_group': test_subject_group,
            'test_metrics': test_metrics,
            'splits': {
                'train_subjects': dataloaders.train_subject_groups if hasattr(dataloaders, 'train_subject_groups') else [],
                'val_subjects': dataloaders.val_subject_group if hasattr(dataloaders, 'val_subject_group') else [],
                'test_subjects': dataloaders.test_subject_group if hasattr(dataloaders, 'test_subject_group') else []
            }
        }

    # 训练模式
    config['output_dir'] = fold_output_dir

    # 构建模型
    print(f"\nBuilding model for fold {fold_name}...")
    model = build_model_bounded(config)
    model = model.to(device)
    model.count_parameters()

    # 构建优化器和调度器
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # 创建logger和TensorBoard
    logger = TrainingLogger(
        log_dir=os.path.join(fold_output_dir, 'logs'),
        log_file=f'training_{fold_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    tensorboard_writer = SummaryWriter(os.path.join(fold_output_dir, 'logs'))

    # 损失函数
    criterion = nn.MSELoss()

    # 训练
    train_result = train_single_fold(
        model=model,
        dataloaders={
            'train': dataloaders.get_loader('train'),
            'val': dataloaders.get_loader('val')
        },
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        logger=logger,
        tensorboard_writer=tensorboard_writer,
        fold_name=fold_name
    )

    # 在测试集上评估
    best_checkpoint_path = os.path.join(fold_output_dir, 'checkpoints', 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate_on_test_set(
            model=model,
            dataloader=dataloaders.get_loader('test'),
            device=device,
            fold_name=fold_name
        )
    else:
        print(f"Warning: Best checkpoint not found at {best_checkpoint_path}")
        test_metrics = None

    return {
        'fold': fold_name,
        'test_subject_group': test_subject_group,
        'train_result': train_result,
        'test_metrics': test_metrics,
        'splits': {
            'train_subjects': dataloaders['train_subjects'],
            'val_subjects': dataloaders['val_subjects'],
            'test_subjects': dataloaders['test_subjects']
        }
    }


def aggregate_results(all_results, output_dir):
    """
    聚合所有fold的结果

    Args:
        all_results: 所有fold的结果列表
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print("LOSO-CV 结果汇总")
    print(f"{'='*70}")

    # 汇总测试指标
    test_metrics_list = []
    for result in all_results:
        if result and result.get('test_metrics'):
            test_metrics_list.append(result['test_metrics'])

    if len(test_metrics_list) == 0:
        print("No test metrics found")
        return

    # 计算均值和标准差
    metric_names = ['mae', 'rmse', 'nmae', 'srcc', 'pcc']
    summary = {}

    for metric in metric_names:
        values = [m[metric] for m in test_metrics_list]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)

    # 打印结果
    print("\n各Fold测试集指标:")
    for i, result in enumerate(all_results):
        if result and result.get('test_metrics'):
            test_m = result['test_metrics']
            print(f"  {result['fold']:>12}: MAE={test_m['mae']:.4f}, SRCC={test_m['srcc']:.4f}, PCC={test_m['pcc']:.4f}")

    print(f"\n平均指标 ± 标准差:")
    print(f"  MAE:  {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"  RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"  NMAE: {summary['nmae_mean']:.4f} ± {summary['nmae_std']:.4f}")
    print(f"  SRCC: {summary['srcc_mean']:.4f} ± {summary['srcc_std']:.4f}")
    print(f"  PCC:  {summary['pcc_mean']:.4f} ± {summary['pcc_std']:.4f}")

    # 保存结果
    results_path = os.path.join(output_dir, 'loso_cv_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_fold': all_results,
            'num_folds': len(all_results)
        }, f, indent=2)

    print(f"\n结果已保存至: {results_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # 构建配置
    config = {
        'data_root': args.data_root,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'static_dim': args.static_dim,
        'dynamic_dim': args.dynamic_dim,
        'freeze_backbone': args.freeze_backbone,
        'use_mask_loss': args.use_mask_loss,
        'use_amp': args.use_amp,
        'clip_grad_norm': args.clip_grad_norm,
        'early_stopping': args.early_stopping,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'pretrained': args.pretrained,
        'epochs': args.epochs,
        'print_freq': args.print_freq,
        'save_freq': args.save_freq,
        'log_freq': args.log_freq,
        'backbone_lr': 1e-4,
        'static_lr': 1e-3
    }

    # 确定要运行的folds
    if args.fold is None or args.fold == 'all':
        folds_to_run = JIGSAWS_SUBJECT_GROUPS
    else:
        folds_to_run = [args.fold]

    print(f"\n{'='*70}")
    print("LOSO-CV (Leave-One-Subject-Group-Out Cross-Validation)")
    print(f"{'='*70}")
    print(f"数据集根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行folds: {', '.join(folds_to_run)}")
    print(f"{'='*70}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 运行每个fold
    all_results = []
    for fold in folds_to_run:
        print(f"\n{'#'*70}")
        print(f"# Running Fold: {fold} (Test Set: {fold}001-{fold}005)")
        print(f"{'#'*70}")

        result = run_single_fold(args, fold, config)
        all_results.append(result)

    # 聚合结果
    if len(folds_to_run) > 1:
        aggregate_results(all_results, args.output_dir)

    print(f"\n{'='*70}")
    print("LOSO-CV 完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
