#!/usr/bin/env python3
"""
Evaluate K-Fold Test Sets Detailed Results (TRANSFORMER VERSION)

此脚本用于在跑完 K 折交叉验证后，回放每一折的“专属测试集”，
并输出详细的逐视频预测结果（Video ID, 真实分数, 预测分数, 绝对误差），
最终生成易于分析的 CSV 表格。

Usage:
    python evaluate_kfold_testsets_transformer.py --config configs/bounded.yaml \
        --weights_dir output_multiclip_bounded_transformer/run_2024xxxx_xxxx \
        --output_dir test_results_transformer
"""

import os
import sys
import torch
import numpy as np
import argparse
import yaml
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 🌟 核心修改点：这里的导入路径改为了带有 _transformer 后缀的文件
from models.surgical_qa_model_multiclip_bounded_transformer import SurgicalQAModelMultiClipBounded

from utils.data_loader_video_level_frames import create_dataloader_with_split
from utils.training import compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate specific test sets for K-Fold CV (Transformer Version)')
    parser.add_argument('--config', type=str, default='configs/bounded.yaml')
    parser.add_argument('--weights_dir', type=str, required=True, help='存放 best_model_fold_x.pth 的文件夹')
    # 🌟 默认输出文件夹改名，防止覆盖旧版结果
    parser.add_argument('--output_dir', type=str, default='test_results_transformer', help='详细结果表格保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_folds = config.get('num_folds', 4)
    use_mask = config.get('use_mask', True)
    score_min = config.get('score_min', 5.0)
    score_max = config.get('score_max', 25.0)
    score_range = score_max - score_min

    print("="*70)
    print("🎯 K-Fold 测试集详细回放与评估 (TRANSFORMER VERSION)")
    print("="*70)

    # 用于最后计算整体平均指标
    all_folds_spearman = []
    all_folds_l2 = []

    # 2. 遍历每一折
    for fold in range(num_folds):
        weight_path = os.path.join(args.weights_dir, f'best_model_fold_{fold}.pth')################################################
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 Fold {fold}: 找不到权重文件 {weight_path}")
            continue

        print(f"\n" + "★"*50)
        print(f"★ 正在评估 FOLD {fold}")
        print("★"*50)

        # 加载这一折的模型 (此时加载的是 Transformer 架构)
        model = SurgicalQAModelMultiClipBounded(config).to(args.device)
        checkpoint = torch.load(weight_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 🔥 核心：严格按照同一套随机种子和数学逻辑，把这一折的专属测试集切出来！
        test_loader = create_dataloader_with_split(
            data_root=config['data_root'],
            batch_size=1,                 # 逐个视频推理，batch_size 设为 1 最稳妥
            num_workers=config.get('num_workers', 4),
            spatial_size=config.get('spatial_size', 112),
            clip_length=config['clip_length'],
            clip_stride=config['clip_stride'],
            score_min=config['score_min'],
            score_max=config['score_max'],
            subset='test',                # 强制指定拉取测试集
            num_folds=num_folds,
            current_fold=fold,            # 告诉 DataLoader 当前是哪一折
            split_seed=config.get('split_seed', 42),
            is_train=False,
            use_mask=use_mask
        )

        fold_results = []
        pred_scores_for_metric = []
        gt_scores_for_metric = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                video = batch['video'].to(args.device)
                masks = batch.get('masks', None)
                if masks is not None:
                    masks = masks.to(args.device)
                
                gt_score_norm = batch['score'].numpy()[0]
                video_id = batch['video_id'][0]

                # 模型预测（归一化分数 [0,1]）
                pred_score_norm = model(video, masks).squeeze(-1).cpu().numpy()[0]

                # 反归一化为真实分数 [5, 25]
                pred_real = pred_score_norm * score_range + score_min
                gt_real = gt_score_norm * score_range + score_min
                abs_error = abs(pred_real - gt_real)

                fold_results.append({
                    'Video_ID': video_id,
                    'Ground_Truth': round(gt_real, 2),
                    'Prediction': round(pred_real, 2),
                    'Abs_Error': round(abs_error, 2)
                })

                pred_scores_for_metric.append(pred_real)
                gt_scores_for_metric.append(gt_real)

        # 3. 保存这一折的详细结果到 CSV
        csv_path = os.path.join(args.output_dir, f'fold_{fold}_detailed_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Video_ID', 'Ground_Truth', 'Prediction', 'Abs_Error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # 按绝对误差从大到小排序，方便你一眼看出最难判断的“毒药视频”
            fold_results_sorted = sorted(fold_results, key=lambda x: x['Abs_Error'], reverse=True)
            for row in fold_results_sorted:
                writer.writerow(row)

        # 4. 计算这一折的宏观指标
        metrics = compute_metrics(np.array(pred_scores_for_metric), np.array(gt_scores_for_metric))
        all_folds_spearman.append(metrics['spearman'])
        all_folds_l2.append(metrics['l2']) 
        
        print(f"✅ Fold {fold} 测试完毕！(共 {len(fold_results)} 个视频)")
        print(f"   Spearman: {metrics['spearman']:.4f}")
        print(f"   L2 Error: {metrics['l2']:.4f}")
        print(f"   详细对比表格已保存至: {csv_path}")

    # 最终汇总
    if all_folds_spearman:
        print("\n" + "="*70)
        print("📊 所有折测试集汇总成绩 (TRANSFORMER)")
        print("="*70)
        print(f"平均 Spearman: {np.mean(all_folds_spearman):.4f} ± {np.std(all_folds_spearman):.4f}")
        print(f"平均 L2 Error: {np.mean(all_folds_l2):.4f} ± {np.std(all_folds_l2):.4f}")
        print(f"数据全部保存在: {args.output_dir}/ 目录下。你可以用 Excel 打开它们进行深入分析。")
        print("="*70)

if __name__ == '__main__':
    main()