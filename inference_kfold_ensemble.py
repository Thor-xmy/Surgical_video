#!/usr/bin/env python3
"""
K-Fold Ensemble Inference Script for Surgical QA Model (Multi-Clip Bounded)

核心特性：
1. 完美适配 SurgicalQAModelMultiClipBounded
2. 读取完整视频序列（支持 .mp4 和帧目录），由模型内部自动进行 Clip 切分和时序池化
3. 模型集成 (Ensemble)：同时加载 4 折的模型，输出平均分，极其稳定！
4. 自动读取 yaml 配置进行动态反归一化 (Denormalization)

Usage:
    python inference_kfold_ensemble.py --config configs/bounded.yaml \
        --weights_dir output_multiclip_bounded/run_2024xxxx_xxxx \
        --video /path/to/test_video_dir_or_file
"""

import os
import sys
import torch
import numpy as np
import cv2
import argparse
import yaml
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded


def load_video_as_tensor(video_path, spatial_size=(112, 112)):
    """加载视频（支持目录帧序列或MP4），返回 (1, C, T, H, W) 张量"""
    frames = []
    
    # 情况 1: 输入的是一个包含图片的目录 (与 DataLoader 一致)
    if os.path.isdir(video_path):
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg') or f.endswith('.png')])
        for ff in frame_files:
            frame = cv2.imread(os.path.join(video_path, ff))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, spatial_size)
                frames.append(frame)
                
    # 情况 2: 输入的是单个 .mp4 视频文件
    elif os.path.isfile(video_path) and video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, spatial_size)
            frames.append(frame)
        cap.release()
    else:
        raise ValueError("不支持的视频格式！请提供图像序列目录或 .mp4 文件")

    if len(frames) == 0:
        raise ValueError(f"未能从 {video_path} 加载到任何帧！")

    # 预处理：堆叠 -> 转维度 -> 归一化到 [-1, 1]
    frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
    frames_np = frames_np.transpose(3, 0, 1, 2)  # (C, T, H, W)
    video_tensor = torch.from_numpy(frames_np).float() / 255.0
    video_tensor = (video_tensor - 0.5) * 2.0  # 映射到 [-1, 1]
    
    return video_tensor.unsqueeze(0)  # 增加 Batch 维度 (1, C, T, H, W)


def main():
    parser = argparse.ArgumentParser(description='Surgical QA K-Fold Ensemble Inference')
    parser.add_argument('--config', type=str, default='configs/bounded.yaml', help='配置文件路径')
    parser.add_argument('--weights_dir', type=str, required=True, help='存放 best_model_fold_x.pth 的文件夹路径')
    parser.add_argument('--video', type=str, required=True, help='测试视频路径(MP4文件或图像目录)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    spatial_size = (config.get('spatial_size', 112), config.get('spatial_size', 112))
    score_min = config.get('score_min', 5.0)
    score_max = config.get('score_max', 25.0)
    score_range = score_max - score_min
    num_folds = config.get('num_folds', 4)

    print("="*70)
    print("🚀 启动 K-Fold 集成推理引擎 (Multi-Clip Bounded)")
    print("="*70)
    print(f"📁 视频路径: {args.video}")
    print(f"📂 权重目录: {args.weights_dir}")
    print(f"📊 分数反归一化范围: [{score_min}, {score_max}]")

    # 2. 组建专家委员会 (加载 4 折模型)
    models = []
    print("\n👨‍⚕️ 正在召集专家委员会...")
    for fold in range(num_folds):
        weight_path = os.path.join(args.weights_dir, f'best_model_fold_{fold}.pth')
        if not os.path.exists(weight_path):
            print(f"❌ 警告: 找不到 Fold {fold} 的权重文件 ({weight_path})，跳过该专家！")
            continue
            
        model = SurgicalQAModelMultiClipBounded(config).to(args.device)
        checkpoint = torch.load(weight_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  ✅ 专家 {fold+1} (Fold {fold}) 准备就绪！")

    if len(models) == 0:
        raise RuntimeError("未能加载任何模型权重！请检查 --weights_dir 路径。")

    # 3. 处理输入视频
    print("\n🎬 正在解析完整视频...")
    video_tensor = load_video_as_tensor(args.video, spatial_size).to(args.device)
    print(f"  ✅ 视频加载成功，形状: {list(video_tensor.shape)} (Batch, C, T, H, W)")

    # 4. 让每位专家独立打分 (Ensemble)
    print("\n🧠 专家正在评估全时序特征...")
    normalized_scores = []
    
    with torch.no_grad():
        # 注意：这里我们传入 masks=None，因为实际应用中通常没有现成的 Mask。
        # 模型内部会平稳处理，仅使用静态和动态的基础提取。
        for i, model in enumerate(models):
            # 模型输出是 [0, 1] 之间的一个值
            score_pred = model(video_tensor, masks=None)
            score_val = score_pred.item()
            normalized_scores.append(score_val)
            print(f"  -> 专家 {i+1} 原始归一化评分: {score_val:.4f}")

    # 5. 汇总并得出最终得分
    avg_norm_score = np.mean(normalized_scores)
    std_norm_score = np.std(normalized_scores)
    
    # 核心：反归一化
    final_real_score = avg_norm_score * score_range + score_min

    print("\n" + "🏆"*23)
    print(" 终 极 评 估 结 果 ")
    print("🏆"*23)
    print(f"📈 专家意见分歧 (Std): ±{std_norm_score:.4f} (越小说明专家意见越统一)")
    print(f"🌟 最终视频综合得分: {final_real_score:.2f} 分 / 满分 {score_max:.1f} 分")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()