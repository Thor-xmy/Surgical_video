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
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=None)
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
    # 🌟 新增：指定单独跑哪一折（0, 1, 2, 3）。如果不指定(None)，就跑完整的多折循环
    parser.add_argument('--target_fold', type=int, default=None, help='Specify a single fold to train/resume')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
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
'''
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
'''
def build_optimizer(model, config):
    # 成功从 yaml 读取配置
    backbone_lr = config.get('backbone_lr', 3e-4)
    static_lr = config.get('static_lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-5)
    opt_type = config.get('optimizer', 'adam').lower()
    momentum = config.get('momentum', 0.9)

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'dynamic_extractor' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    # 按照不同模块分配读取到的学习率
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': other_params, 'lr': static_lr}
    ]

    # 根据配置选择优化器
    if opt_type == 'adam':
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        
    return optimizer

    #==============================================================
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

def build_scheduler(optimizer, config):
    sched_type = config.get('lr_scheduler', 'cosine')
    if sched_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=config.get('epochs', 100))
    elif sched_type == 'step':
        return MultiStepLR(optimizer, 
                           milestones=config.get('lr_milestones', [30, 60, 90]), 
                           gamma=config.get('lr_gamma', 0.1))
    return None
#===============================================================================




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
    # 1. 从 config 获取真实分数的上下界
    score_min = config.get('score_min', 5.0)
    score_max = config.get('score_max', 25.0)
    score_range = score_max - score_min
    
    # 2. 将 [0, 1] 范围的分数反归一化回真实分数范围 (例如 5~25)
    pred_scores_real = pred_scores * score_range + score_min
    gt_scores_real = gt_scores * score_range + score_min
    
    # 3. 计算具有真实对比意义的 Metrics
    #metrics = compute_metrics(pred_scores, gt_scores)
    metrics = compute_metrics(pred_scores_real, gt_scores_real)
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
    #metrics = compute_metrics(pred_scores, gt_scores)
    score_min = config.get('score_min', 5.0)
    score_max = config.get('score_max', 25.0)
    score_range = score_max - score_min
    
    pred_scores_real = pred_scores * score_range + score_min
    gt_scores_real = gt_scores * score_range + score_min
    
    metrics = compute_metrics(pred_scores_real, gt_scores_real)
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

'''
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
    scheduler = build_scheduler(optimizer, config)  ############################################################
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
    # 解析 YAML 中的 split_ratio 数组
    split_ratio_config = config.get('split_ratio', [0.7, 0.15, 0.15])
    parsed_train_ratio = split_ratio_config[0]
    parsed_val_ratio = split_ratio_config[1]
    parsed_test_ratio = split_ratio_config[2]
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
        train_ratio=parsed_train_ratio,
        val_ratio=parsed_val_ratio,
        test_ratio=parsed_test_ratio,
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
        train_ratio=parsed_train_ratio,
        val_ratio=parsed_val_ratio,
        test_ratio=parsed_test_ratio,
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
        if scheduler is not None:  ####################################################################
            scheduler.step()

    print("\nTraining completed!")
    print(f"Best Spearman: {best_metric:.4f}")
    writer.close()
'''

def main():
    args = parse_args()
    config = load_config(args.config, args)

    setup_seed(args.seed)
    device = setup_device(args.gpu)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(log_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("\n" + "="*70)
    print("Multi-Clip Bounded Surgical QA Model - K-Fold Cross Validation")
    print("="*70)

    use_mask = config.get('use_mask', True)
    use_amp = config.get('use_amp', False)
    scaler = GradScaler() if use_amp else None
    
    # 🌟 设定 K折数量（建议写在 config 里，默认用 4 折）
    num_folds = config.get('num_folds', 4)
    fold_results_spearman = []
    # ================== 修改点：灵活的 K 折循环控制器 ==================
    if args.target_fold is not None:
        if args.target_fold < 0 or args.target_fold >= num_folds:
            raise ValueError(f"target_fold 必须在 0 到 {num_folds-1} 之间！")
        folds_to_run = [args.target_fold]
        print(f"\n🎯 模式：单独训练/续训指定折 FOLD {args.target_fold}")
    else:
        folds_to_run = range(num_folds)
        print(f"\n🎯 模式：完整运行 {num_folds}-Fold 交叉验证")
    # ===================================================================
    # 🌟🌟🌟 最外层的 K折循环开始
    for current_fold in folds_to_run:
        print("\n" + "★"*70)
        print(f"★ 开始训练 FOLD {current_fold + 1} / {num_folds}")
        print("★"*70)

        # 1. 每一折必须重新初始化一个全新的模型！绝对不能让模型携带上一折的记忆！
        model = SurgicalQAModelMultiClipBounded(config).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        
        # 为当前折创建专属的 Tensorboard 日志目录
        fold_writer = SummaryWriter(os.path.join(log_dir, f'fold_{current_fold}'))

        # ================== 🌟 新增：断点续训逻辑 ==================
        start_epoch = 0
        best_val_spearman = -float('inf')
        best_test_spearman_for_this_fold = 0.0

        if args.resume is not None:
            print(f"🔄 正在从 {args.resume} 加载断点权重...")
            # 加载权重
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复 epoch 数和历史最佳成绩
            start_epoch = checkpoint['epoch']  # 如果存的是 epoch 10，这里 start_epoch 就是 10（从第11轮开始）
            if 'metrics' in checkpoint and 'spearman' in checkpoint['metrics']:
                best_val_spearman = checkpoint['metrics']['spearman']
            
            print(f"✅ 成功恢复！将从 Epoch {start_epoch} 继续训练 Fold {current_fold}。历史最佳 Val Spearman: {best_val_spearman:.4f}")
        # ==========================================================

        # 2. 获取当前折的数据集
        train_loader = create_dataloader_with_split(
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            spatial_size=config.get('spatial_size', 112),
            clip_length=config['clip_length'],
            clip_stride=config['clip_stride'],
            score_min=config['score_min'],
            score_max=config['score_max'],
            subset='train',
            num_folds=num_folds,          # 👈 传入折数
            current_fold=current_fold,    # 👈 传入当前折的索引
            split_seed=config.get('split_seed', 42),
            is_train=True,
            use_mask=use_mask
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
            subset='val',                 # 👈 这一折的验证集
            num_folds=num_folds,
            current_fold=current_fold,
            split_seed=config.get('split_seed', 42),
            is_train=False,
            use_mask=use_mask
        )
        
        test_loader = create_dataloader_with_split(
            data_root=config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            spatial_size=config.get('spatial_size', 112),
            clip_length=config['clip_length'],
            clip_stride=config['clip_stride'],
            score_min=config['score_min'],
            score_max=config['score_max'],
            subset='test',                # 👈 这一折的测试集
            num_folds=num_folds,
            current_fold=current_fold,
            split_seed=config.get('split_seed', 42),
            is_train=False,
            use_mask=use_mask
        )

        #best_val_spearman = -float('inf')
        #best_test_spearman_for_this_fold = 0.0

        # 3. 正常的 Epoch 训练循环
        for epoch in range(start_epoch, config['epochs']):
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config, scaler)
            val_loss, val_metrics = validate(model, val_loader, device, config)

            # 将当前折的日志写入 tensorboard
            fold_writer.add_scalar('train/loss', train_loss, epoch)
            fold_writer.add_scalar('train/spearman', train_metrics['spearman'], epoch)
            fold_writer.add_scalar('val/loss', val_loss, epoch)
            fold_writer.add_scalar('val/spearman', val_metrics['spearman'], epoch)
            # ==========================================================
            # 🌟 【新增：修复 save_freq 不执行的问题】
            # 注意文件名加上了 fold_{current_fold}，防止不同折互相覆盖！
            if (epoch + 1) % config.get('save_freq', 20) == 0:
                save_checkpoint(
                    model, optimizer, epoch + 1, val_loss, val_metrics,
                    log_dir, f'checkpoint_fold_{current_fold}_epoch_{epoch + 1}.pth'
                )
            # ==========================================================
            # 保存当前折的最佳模型，并用它在测试集上跑一次最终成绩
            if val_metrics['spearman'] > best_val_spearman:
                best_val_spearman = val_metrics['spearman']
                save_checkpoint(
                    model, optimizer, epoch + 1, val_loss, val_metrics,
                    log_dir, f'best_model_fold_{current_fold}.pth'  # 👈 命名包含 fold 编号，防止覆盖
                )
                
                # 🌟 当验证集达到最好时，立刻在测试集上跑一次真实分数
                print(f"--> [Fold {current_fold}] 验证集提升，在测试集上评估...")
                test_loss, test_metrics = validate(model, test_loader, device, config)
                best_test_spearman_for_this_fold = test_metrics['spearman']
                print(f"--> [Fold {current_fold}] 当前测试集 Spearman: {best_test_spearman_for_this_fold:.4f}")

            if scheduler is not None:
                scheduler.step()
                
        fold_writer.close()
        
        # 记录这一折在测试集上的最终成绩
        fold_results_spearman.append(best_test_spearman_for_this_fold)
        print(f"★ FOLD {current_fold + 1} 训练结束！最终测试集 Spearman: {best_test_spearman_for_this_fold:.4f}")

    # 4. K折全部跑完，进行终极统计
    print("\n" + "🚀"*30)
    print("K-Fold 交叉验证全部完成！")
    print(f"各折测试集 Spearman 分数: {[f'{score:.4f}' for score in fold_results_spearman]}")
    mean_spearman = np.mean(fold_results_spearman)
    std_spearman = np.std(fold_results_spearman)
    print(f"👑 最终模型平均 Spearman: {mean_spearman:.4f} ± {std_spearman:.4f}")
    print("🚀"*30 + "\n")

if __name__ == '__main__':
    main()

