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
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded, build_model_multiclip_bounded
from models.surgical_qa_model_multiclip_bounded_transformer import SurgicalQAModelMultiClipBounded, build_model_multiclip_bounded
# Modified to use frame sequence loader with proper data splitting
from utils.data_loader_video_level_frames import create_dataloader_with_split
from utils.training import AverageMeter, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Clip Bounded Surgical QA Model')
    parser.add_argument('--config', type=str, default='configs/bounded.yaml')
    parser.add_argument('--data_root', type=str, default=None, help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='output_multiclip_bounded_transformer')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--static_dim', type=int, default=None)
    parser.add_argument('--dynamic_dim', type=int, default=1024)
    parser.add_argument('--clip_length', type=int, default=None)
    parser.add_argument('--clip_stride', type=int, default=None)
    parser.add_argument('--max_clips', type=int, default=None)
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--use_mixed_conv', action='store_true', default=False)
    
    # 🌟 修改点 1：将 score_min 和 score_max 的默认值设为 None
    # 这样它们就不会强行覆盖你 bounded.yaml 里设置的 5.0 和 25.0 了
    parser.add_argument('--score_min', type=float, default=None)
    parser.add_argument('--score_max', type=float, default=None)
    
    parser.add_argument('--resume', type=str, default=None)
    # 🌟 新增：指定单独跑哪一折（0, 1, 2, 3）。如果不指定(None)，就跑完整的多折循环
    parser.add_argument('--target_fold', type=int, default=None, help='Specify a single fold to train/resume')
    parser.add_argument('--pretrained', action='store_true', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=None)
    parser.add_argument('--save_freq', type=int, default=None)
    # 在 parse_args 里面添加
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累加步数')
    parser.add_argument('--skip_val', action='store_true', default=None, help='跳过独立验证集，直接使用测试集验证')
    parser.add_argument('--use_bottleneck', action='store_true', default=None, help='是否在送入Transformer前进行降维')
    parser.add_argument('--bottleneck_dim', type=int, default=None, help='瓶颈层的目标维度')
    return parser.parse_args()



def load_config(config_path, args):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config_updates = {
        #'data_root': args.data_root,
        #'batch_size': args.batch_size,
        #'num_workers': args.num_workers,
        #'learning_rate': args.learning_rate,
        #'weight_decay': args.weight_decay,
        #'static_dim': args.static_dim,
        #'dynamic_dim': args.dynamic_dim,
        #'clip_length': args.clip_length,
        #'clip_stride': args.clip_stride,
        #'max_clips': args.max_clips,
        #'freeze_backbone': args.freeze_backbone,
        #'use_mixed_conv': args.use_mixed_conv,
        #'score_min': args.score_min,
        #'score_max': args.score_max,
        #'use_pretrained': args.pretrained,
        #'epochs': args.epochs,
        #'print_freq': args.print_freq,
        #'save_freq': args.save_freq
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


def print_trainable_parameters(model):
    """
    打印模型中所有参与训练的层和参数量统计
    """
    print("\n" + "="*80)
    print("🚀 模 型 参 数 体 检 报 告 (Trainable Parameters Summary)")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    print(f"{'Layer Name':<55} | {'Shape':<18} | {'Params':<10}")
    print("-" * 85)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        # 只打印参与训练的层 (requires_grad == True)
        if param.requires_grad:
            trainable_params += num_params
            # 将 shape 转换为紧凑的字符串，如 [64, 128, 3, 3]
            shape_str = str(list(param.shape))
            print(f"{name:<55} | {shape_str:<18} | {num_params:<10,}")
            
    print("-" * 85)
    print(f"🧊 总参数量 (Total Parameters)       : {total_params:>15,}")
    print(f"🔥 参与训练参数 (Trainable Params)   : {trainable_params:>15,}")
    
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    print(f"📊 可训练参数比例 (Trainable Ratio)  : {percentage:>14.2f}%")
    print("="*80 + "\n")



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
'''
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
'''
def build_scheduler(optimizer, config):
    sched_type = config.get('lr_scheduler', 'cosine')
    total_epochs = config.get('epochs', 100)
    
    # 获取 warmup 参数，默认开启，预热 5 个 epoch
    use_warmup = config.get('use_warmup', True)
    warmup_epochs = config.get('warmup_epochs', 5) if use_warmup else 0

    if sched_type == 'cosine':
        def lr_lambda(epoch):
            # 1. Warmup 阶段：线性爬升到 1.0
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            # 2. Cosine 衰减阶段
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        return LambdaLR(optimizer, lr_lambda)
        
    elif sched_type == 'step':
        # Step LR 这里就不做 warmup 了，为了简单，或者你可以自己套用 LambdaLR
        return MultiStepLR(optimizer, 
                           milestones=config.get('lr_milestones', [30, 60, 90]), 
                           gamma=config.get('lr_gamma', 0.1))
    return None


def train_epoch(model, dataloader, optimizer, device, epoch, config, scaler=None):
    model.train()
    #=====================================================================
    def freeze_bn(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    #model.apply(freeze_bn)
    model.dynamic_extractor.apply(freeze_bn)
    #=====================================================================

    loss_meter = AverageMeter()
    grad_norm_meter = AverageMeter()  # 🌟 新增：梯度长度记录仪
    pred_scores = []
    gt_scores = []

    print_freq = config.get('print_freq', 10)
    # 获取累加步数，默认为1（即不累加）
    accumulation_steps = config.get('accumulation_steps', 1)
    optimizer.zero_grad()


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
        #optimizer.zero_grad()
        
        if scaler is not None:
            # 🌟 开启混合精度：让模型在 float16 (半精度) 下跑，显存砍半！
            with autocast():
                score_pred = model(video, masks)
                loss, loss_dict = model.compute_loss(score_pred, score_gt)
                loss = loss / accumulation_steps
            # 混合精度专用的反向传播
            scaler.scale(loss).backward()
            '''
            scaler.step(optimizer)
            scaler.update()
            '''
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader)):
                scaler.unscale_(optimizer)
                clip_norm = config.get('clip_grad_norm', 1.0)
                if clip_norm > 0:
                    # 🌟 核心魔法：它会返回【裁剪前】的真实原始梯度长度！
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    grad_norm_meter.update(total_norm.item(), 1) # 记录下来
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 正常的 float32 全精度跑法
            score_pred = model(video, masks)
            loss, loss_dict = model.compute_loss(score_pred, score_gt)
            loss = loss / accumulation_steps
            loss.backward()
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader)):
                clip_norm = config.get('clip_grad_norm', 0.0)
                if clip_norm > 0:
                    # 🌟 同理，全精度模式下也记录
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    grad_norm_meter.update(total_norm.item(), 1)
                    
                optimizer.step()
                optimizer.zero_grad()
            
            #optimizer.step()

        loss_meter.update(loss.item() * accumulation_steps, batch_size)
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
    metrics['grad_norm'] = grad_norm_meter.avg  # 🌟 将梯度长度打包
    # ====== 🌟 新增：提取最后一个 Batch 的动态权重传给外部 ======
    if config.get('use_dynamic_weights', False):
        # 此时的 loss_dict 是本 epoch 最后一个 batch 的输出，足够代表当前 epoch 的状态
        metrics['weight_score'] = loss_dict.get('weight_score', 0.0)
        metrics['weight_rank']  = loss_dict.get('weight_rank', 0.0)
        metrics['weight_mean']  = loss_dict.get('weight_mean', 0.0)
        metrics['weight_tie']   = loss_dict.get('weight_tie', 0.0)
    # ============================================================
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
    # ================== 🌟 智能目录继承逻辑 (终极版) ==================
    if args.resume is not None:
        # 情况 1：有 .pth 断点文件 (例如跑了一半断掉)
        log_dir = os.path.dirname(args.resume)
        print(f"\n📂 检测到断点续训！日志和权重将继续保存在原目录: {log_dir}")
        
    elif 'run_' in args.config:
        # 🌟 情况 2：没有断点文件，但传入了旧文件夹里的 config.yaml (例如新的一折刚开始)
        log_dir = os.path.dirname(args.config)
        print(f"\n📂 检测到继承训练！日志和权重将存入历史目录: {log_dir}")
        
    else:
        # 情况 3：完全从头开始的新训练 (传入的是 configs/bounded.yaml)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(args.output_dir, f'run_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
    # ==========================================================

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
        # 🌟 新增：只在第一折（或只跑单折）时打印一次参数报告，防止控制台刷屏
        if current_fold == folds_to_run[0]:
            print_trainable_parameters(model)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        
        # 为当前折创建专属的 Tensorboard 日志目录
        fold_writer = SummaryWriter(os.path.join(log_dir, f'fold_{current_fold}'))

        # ================== 🌟 新增：断点续训逻辑 ==================
        start_epoch = 0
        
        best_val_spearman = -float('inf')
        best_val_loss_for_tie = float('inf') # 🌟 新增：记录最高分对应的最低 Loss，初始化为正无穷大 (infinity)
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
        skip_val = config.get('skip_val', False)
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
            use_mask=use_mask,
            skip_val=skip_val  # 🌟 传给 DataLoader，让训练集吃掉验证集
        )
        '''
        val_loader = create_dataloader_with_split(
            data_root=config['data_root'],
            #batch_size=config['batch_size'],
            batch_size=config.get('val_batch_size', config['batch_size']),
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
        '''
        test_loader = create_dataloader_with_split(
            data_root=config['data_root'],
            #batch_size=config['batch_size'],
            batch_size=config.get('test_batch_size', config['batch_size']),
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
        # 3. 🌟 新增逻辑：可选择的验证集策略
        skip_val = config.get('skip_val', False)
        
        if skip_val:
            print(f"⚠️ [Fold {current_fold}] 开启【跳过独立验证集】模式：直接使用 Test Set 评估以保存 Best Model！")
            val_loader = test_loader  # 直接把测试集的指针赋给 val_loader
        else:
            print(f"✅ [Fold {current_fold}] 标准模式：创建独立的 Validation Set。")
            val_loader = create_dataloader_with_split(
                data_root=config['data_root'],
                batch_size=config.get('val_batch_size', config['batch_size']),
                num_workers=config['num_workers'],
                spatial_size=config.get('spatial_size', 112),
                clip_length=config['clip_length'],
                clip_stride=config['clip_stride'],
                score_min=config['score_min'],
                score_max=config['score_max'],# ... (保持你原本传的参数不变)
                subset='val',
                num_folds=num_folds,
                current_fold=current_fold,
                split_seed=config.get('split_seed', 42),
                is_train=False,
                use_mask=use_mask,
                skip_val=skip_val
            )

        #best_val_spearman = -float('inf')
        #best_test_spearman_for_this_fold = 0.0

        # 3. 正常的 Epoch 训练循环
        for epoch in range(start_epoch, config['epochs']):
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config, scaler)
            val_loss, val_metrics = validate(model, val_loader, device, config)

            # 将当前折的日志写入 tensorboard
            fold_writer.add_scalar('train/1_loss', train_loss, epoch)
            fold_writer.add_scalar('train/2_spearman', train_metrics['spearman'], epoch)
            fold_writer.add_scalar('train/3_grad_norm', train_metrics['grad_norm'], epoch) # 🌟 画出真实的梯度长度！
            # ====== 🌟 新增：将动态权重画到 TensorBoard ======
            # 为了在 Tensorboard 里排版好看，我给它们加了数字前缀 1_ 2_ 3_ 4_
            if config.get('use_dynamic_weights', False):
                fold_writer.add_scalar('weights/4_score_weight', train_metrics.get('weight_score', 0.0), epoch)
                fold_writer.add_scalar('weights/5_rank_weight', train_metrics.get('weight_rank', 0.0), epoch)
                if config.get('use_mean_penalty', False):
                    fold_writer.add_scalar('weights/6_mean_weight', train_metrics.get('weight_mean', 0.0), epoch)
                if config.get('use_tie_loss', False):
                    fold_writer.add_scalar('weights/7_tie_weight', train_metrics.get('weight_tie', 0.0), epoch)
            # ==================================================
            # 可选的美化修改（在 main 函数的循环里）
            prefix = 'test' if skip_val else 'val'
            fold_writer.add_scalar(f'{prefix}/1_loss', val_loss, epoch)
            fold_writer.add_scalar(f'{prefix}/2_spearman', val_metrics['spearman'], epoch)
            #fold_writer.add_scalar('val/loss', val_loss, epoch)
            #fold_writer.add_scalar('val/spearman', val_metrics['spearman'], epoch)
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
            # 改进的保存逻辑：Spearman 更高，或者 Spearman 相同但 Loss 更低
            if val_metrics['spearman'] > best_val_spearman or \
               (val_metrics['spearman'] == best_val_spearman and val_loss < best_val_loss_for_tie):
                
                best_val_spearman = val_metrics['spearman']
                best_val_loss_for_tie = val_loss  # 记录当前最小 Loss
                
                save_checkpoint(
                    model, optimizer, epoch + 1, val_loss, val_metrics,
                    log_dir, f'best_model_fold_{current_fold}.pth'
                )
                print(f"--> [Fold {current_fold}] 🌟 保存新 Best Model! (Spearman: {best_val_spearman:.4f}, Loss: {val_loss:.4f})")
                
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

