"""
评估指定 Fold 的所有 Checkpoints (包含详细的样本预测结果)
使用方法:
python eval_fold_checkpoints.py --config configs/bounded.yaml --run_dir output_multiclip_bounded_transformer/run_2024xxxx_xxxx --target_fold 0
"""

import os
import glob
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate

# 导入你的模型和DataLoader
from models.surgical_qa_model_multiclip_bounded_transformer import SurgicalQAModelMultiClipBounded
from utils.data_loader_video_level_frames import create_dataloader_with_split
from utils.training import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints for a specific fold with detailed results')
    parser.add_argument('--config', type=str, default='configs/bounded.yaml', help='Path to config file')
    parser.add_argument('--run_dir', type=str, required=True, help='Directory containing the checkpoints')
    parser.add_argument('--target_fold', type=int, required=True, help='Which fold to evaluate (e.g., 0, 1, 2, 3)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_checkpoint(model, dataloader, device, config):
    model.eval()
    pred_scores = []
    gt_scores = []
    video_ids = []

    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            masks = batch.get('masks', None)
            if masks is not None:
                masks = masks.to(device)
            
            score_gt = batch['score'].to(device)
            v_ids = batch['video_id']  # 提取视频ID

            score_pred = model(video, masks)
            
            pred_scores.extend(score_pred.detach().squeeze(-1).cpu().numpy())
            gt_scores.extend(score_gt.detach().cpu().numpy())
            video_ids.extend(v_ids)

    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)

    # 反归一化回真实分数范围
    score_min = config.get('score_min', 5.0)
    score_max = config.get('score_max', 25.0)
    score_range = score_max - score_min
    
    pred_scores_real = pred_scores * score_range + score_min
    gt_scores_real = gt_scores * score_range + score_min
    
    # 计算汇总 Metrics
    metrics = compute_metrics(pred_scores_real, gt_scores_real)
    
    # 打包每个视频的详细预测结果
    detailed_results = {
        'video_ids': video_ids,
        'ground_truth': gt_scores_real,
        'prediction': pred_scores_real
    }
    
    return metrics, detailed_results

def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    search_pattern = os.path.join(args.run_dir, f'*fold_{args.target_fold}*.pth')
    checkpoint_paths = glob.glob(search_pattern)
    
    if not checkpoint_paths:
        print(f"❌ 在 {args.run_dir} 中没有找到 Fold {args.target_fold} 的任何权重文件！")
        return

    checkpoint_paths.sort()
    print(f"🔍 找到 {len(checkpoint_paths)} 个 Fold {args.target_fold} 的权重文件")

    use_mask = config.get('use_mask', True)
    test_loader = create_dataloader_with_split(
        data_root=config['data_root'],
        batch_size=config.get('test_batch_size', config.get('batch_size', 4)),
        num_workers=config.get('num_workers', 4),
        spatial_size=config.get('spatial_size', 112),
        clip_length=config['clip_length'],
        clip_stride=config['clip_stride'],
        score_min=config['score_min'],
        score_max=config['score_max'],
        subset='test',
        num_folds=config.get('num_folds', 4),
        current_fold=args.target_fold,
        split_seed=config.get('split_seed', 42),
        is_train=False,
        use_mask=use_mask
    )

    model = SurgicalQAModelMultiClipBounded(config).to(device)

    summary_results = []
    all_detailed_results = []
    
    print("\n🚀 开始评估...")
    for ckpt_path in checkpoint_paths:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"⏳ 正在评估: {ckpt_name} ...")
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        metrics, detailed = evaluate_checkpoint(model, test_loader, device, config)
        
        # 1. 记录汇总指标
        summary_results.append({
            'Checkpoint': ckpt_name,
            'Spearman': metrics.get('spearman', 0.0),
            'Pearson': metrics.get('pearson', 0.0),
            'MSE': metrics.get('mse', 0.0),
            'L2': metrics.get('l2', 0.0),
            'RL2': metrics.get('rl2', 0.0)
        })

        # 2. 记录每一条测试视频的具体真值和预测值
        for vid, gt, pred in zip(detailed['video_ids'], detailed['ground_truth'], detailed['prediction']):
            all_detailed_results.append({
                'Checkpoint': ckpt_name,
                'Video_ID': vid,
                'Ground_Truth': round(float(gt), 4),
                'Prediction': round(float(pred), 4),
                'Abs_Error': round(float(abs(gt - pred)), 4)  # 顺便帮你算个绝对误差，方便排查极差样本
            })

    # --- 整理并保存结果 ---
    
    # 保存汇总表格
    df_summary = pd.DataFrame(summary_results)
    df_summary['is_best'] = df_summary['Checkpoint'].apply(lambda x: 'best' in x)
    df_summary = df_summary.sort_values(by=['is_best', 'Checkpoint']).drop(columns=['is_best']).reset_index(drop=True)

    print(f"\n✅ Fold {args.target_fold} 测试集汇总指标：")
    print(tabulate(df_summary, headers='keys', tablefmt='pretty', floatfmt=".4f"))

    summary_save_path = os.path.join(args.run_dir, f'eval_summary_fold_{args.target_fold}.tsv')
    df_summary.to_csv(summary_save_path, sep='\t', index=False)
    
    # 保存详细预测明细表格
    df_detailed = pd.DataFrame(all_detailed_results)
    detailed_save_path = os.path.join(args.run_dir, f'eval_detailed_fold_{args.target_fold}.tsv')
    # 使用 sep='\t' 保证制表符对齐
    df_detailed.to_csv(detailed_save_path, sep='\t', index=False)
    
    print(f"\n💾 汇总指标已保存至: {summary_save_path}")
    print(f"💾 详细预测(真值/预测值)已保存至: {detailed_save_path}")

if __name__ == '__main__':
    main()