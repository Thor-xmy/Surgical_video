#!/usr/bin/env python3
"""
Evaluate a Single Checkpoint

针对单个 checkpoint 文件进行评估，严格复现训练时的数据集划分。

Usage:
    python evaluate_single_checkpoint.py \
        --config configs/bounded.yaml \
        --checkpoint /root/autodl-tmp/SV_test/output_multiclip_bounded/run_20260325_104408/checkpoint_fold_2_epoch_100.pth \
        --fold 2 \
        --output_dir test_results
"""

import os
import sys
import torch
import numpy as np
import argparse
import yaml
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded
from utils.data_loader_video_level_frames import create_dataloader_with_split
from utils.training import compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate a single checkpoint')
    parser.add_argument('--config',     type=str, required=True,  help='训练时使用的 YAML 配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,  help='待评估的 .pth 文件完整路径')
    parser.add_argument('--fold',       type=int, required=True,  help='该 checkpoint 对应的折编号（从 0 开始）')
    parser.add_argument('--subset',     type=str, default='test', choices=['test', 'val'],
                        help='评估哪个子集：test（默认）或 val')
    parser.add_argument('--output_dir', type=str, default='test_results', help='结果保存目录')
    parser.add_argument('--device',     type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 加载配置（必须与训练时完全一致）──────────────────────────────────
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_folds  = config.get('num_folds',  4)
    use_mask   = config.get('use_mask',   True)
    score_min  = config.get('score_min',  5.0)
    score_max  = config.get('score_max',  25.0)
    split_seed = config.get('split_seed', 42)
    score_range = score_max - score_min

    print("=" * 70)
    print(f"📂 Checkpoint : {args.checkpoint}")
    print(f"📋 Config     : {args.config}")
    print(f"🔢 Fold       : {args.fold}  |  num_folds={num_folds}  |  split_seed={split_seed}")
    print(f"📊 Subset     : {args.subset}")
    print(f"💻 Device     : {args.device}")
    print("=" * 70)

    # ── 2. 加载模型（读取 checkpoint，兼容两种保存格式）──────────────────────
    model = SurgicalQAModelMultiClipBounded(config).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # 兼容直接保存 state_dict 和保存完整字典两种情况
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ 已加载模型权重（epoch={saved_epoch}）")

        # 打印 checkpoint 里保存的训练指标（如果有）
        for key in ('val_spearman', 'val_l2', 'train_loss'):
            if key in checkpoint:
                print(f"   {key}: {checkpoint[key]:.4f}")
    else:
        # 直接是 state_dict
        model.load_state_dict(checkpoint)
        print("✅ 已加载模型权重（直接 state_dict 格式）")

    model.eval()

    # ── 3. 严格复现数据集划分（与训练时完全相同的参数）──────────────────────
    data_loader = create_dataloader_with_split(
        data_root    = config['data_root'],
        batch_size   = 1,                          # 逐视频推理
        num_workers  = config.get('num_workers', 4),
        spatial_size = config.get('spatial_size', 112),
        clip_length  = config['clip_length'],
        clip_stride  = config['clip_stride'],
        score_min    = config['score_min'],
        score_max    = config['score_max'],
        subset       = args.subset,                # 'test' 或 'val'
        num_folds    = num_folds,
        current_fold = args.fold,                  # ← 关键：必须与 checkpoint 一致
        split_seed   = split_seed,                 # ← 关键：必须与训练时一致
        is_train     = False,
        use_mask     = use_mask
    )

    print(f"\n📦 {args.subset} 集共 {len(data_loader.dataset)} 个视频，开始推理...\n")

    # ── 4. 逐视频推理 ────────────────────────────────────────────────────────
    results = []
    pred_list = []
    gt_list   = []

    with torch.no_grad():
        for batch in data_loader:
            video    = batch['video'].to(args.device)
            masks    = batch.get('masks', None)
            if masks is not None:
                masks = masks.to(args.device)

            gt_norm  = batch['score'].numpy()[0]
            video_id = batch['video_id'][0]

            pred_norm = model(video, masks).squeeze(-1).cpu().numpy()[0]

            pred_real = pred_norm * score_range + score_min
            gt_real   = gt_norm  * score_range + score_min
            abs_error = abs(pred_real - gt_real)

            results.append({
                'Video_ID':     video_id,
                'Ground_Truth': round(gt_real,   2),
                'Prediction':   round(pred_real,  2),
                'Abs_Error':    round(abs_error,  2)
            })
            pred_list.append(pred_real)
            gt_list.append(gt_real)

    # ── 5. 计算指标 ──────────────────────────────────────────────────────────
    metrics = compute_metrics(np.array(pred_list), np.array(gt_list))

    print(f"{'='*70}")
    print(f"📊 评估结果（{args.subset} 集，共 {len(results)} 个视频）")
    print(f"{'='*70}")
    print(f"   Spearman 相关系数 : {metrics['spearman']:.4f}")
    print(f"   L2 误差           : {metrics['l2']:.4f}")
    mae = np.mean([r['Abs_Error'] for r in results])
    print(f"   平均绝对误差(MAE) : {mae:.4f}")
    print(f"{'='*70}")

    # ── 6. 保存详细 CSV ──────────────────────────────────────────────────────
    ckpt_name  = os.path.splitext(os.path.basename(args.checkpoint))[0]
    csv_path   = os.path.join(args.output_dir, f'{ckpt_name}_{args.subset}_results.csv')

    results_sorted = sorted(results, key=lambda x: x['Abs_Error'], reverse=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Video_ID', 'Ground_Truth', 'Prediction', 'Abs_Error'])
        writer.writeheader()
        writer.writerows(results_sorted)

    print(f"\n💾 详细结果已保存至: {csv_path}")
    print("   （已按绝对误差从大到小排序，方便定位难样本）")


if __name__ == '__main__':
    main()