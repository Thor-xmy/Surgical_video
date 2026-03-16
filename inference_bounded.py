#!/usr/bin/env python3
"""
Inference Script for Surgical QA Model (Bounded Version)

使用带边界的模型进行推理，包含分数反归一化

修改说明：
1. 使用 SurgicalQAModelBounded（带 Sigmoid 输出）
2. 模型输出范围 [0, 1]，反归一化到目标范围
3. JIGSAWS GRS 分数范围 [6, 30]，推理时输出 [6, 30]

Usage:
    python inference_bounded.py --config configs/bounded.yaml \
        --checkpoint checkpoints/best_model.pth \
        --video /path/to/video.mp4 --mask_dir /path/to/masks
"""

import os
import sys
import torch
import numpy as np
import cv2
import argparse
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.surgical_qa_model_bounded import SurgicalQAModelBounded, build_model_bounded


def process_video(model, video_path, mask_dir, device='cuda'):
    """
    处理单个视频并预测技能分数

    Args:
        model: 训练好的模型
        video_path: 视频文件路径
        mask_dir: 掩膜目录
        device: 推理设备

    Returns:
        score: 预测分数（JIGSAWS GRS范围 [6, 30]）
    """
    print(f"\n正在处理视频: {video_path}")

    # 1. 加载视频
    print("  1. 加载视频帧...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    print(f"      共 {total_frames} 帧")

    if total_frames < 16:
        print(f"      警告: 视频帧数 ({total) 少于 16，无法处理")
        return None

    # 2. 提取 clip（取中间16帧）
    print("  2. 提取 16 帧片段...")
    start_idx = max(0, total_frames // 2 - 8)
    end_idx = min(total_frames, start_idx + 16)

    clip = frames[start_idx:end_idx]
    if len(clip) < 16:
        clip.extend([clip[-1]] * (16 - len(clip)))

    # 3. 预处理
    print("  3. 预处理视频帧...")
    clip_resized = [cv2.resize(f, (112, 112)) for f in clip]
    clip_np = np.stack(clip_resized, axis=0)  # (T, H, W, C)
    clip_np = clip_np.transpose(3, 0, 1, 2)  # (C, T, H, W)
    clip_tensor = torch.from_numpy(clip_np).float() / 255.0
    clip_tensor = (clip_tensor - 0.5) * 2.0  # [-1, 1]
    clip_tensor = clip_tensor.unsqueeze(0)  # (1, C, T, H, W)

    # 4. 加载掩膜
    print("  4. 加载掩膜...")
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    masks = None

    if mask_dir and os.path.exists(mask_dir):
        # 尝试加载 NPY 格式
        mask_npy_path = os.path.join(mask_dir, f"{video_id}_masks.npy")
        if os.path.exists(mask_npy_path):
            masks = np.load(mask_npy_path)
            # 截取对应帧的掩膜
            masks = masks[start_idx:end_idx]
            if masks.shape[0] < 16:
                pad_width = 16 - masks.shape[0]
                masks = np.pad(masks, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
            masks = masks[:16]

            # Resize 到 112x112
            masks_resized = []
            for i in range(16):
                mask_resized.append(cv2.resize(masks[i], (112, 112)))
            masks = np.stack(masks_resized, axis=0)
            masks_tensor = torch.from_numpy(masks).float() / 255.0
            masks = masks_tensor.unsqueeze(0)  # (1, T, H, W)
            print(f"      成功加载 NPY 掩膜: {masks.shape}")
        else:
            # 尝试加载 PNG 格式
            mask_img_dir = os.path.join(mask_dir, video_id)
            if os.path.isdir(mask_img_dir):
                mask_files = sorted([f for f in os.listdir(mask_img_dir) if 'mask' in f])
                if len(mask_files) >= 16:
                    masks = []
                    for i in range(16):
                        mask_file = os.path.join(mask_img_dir, mask_files[i])
                        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (112, 112))
                        masks.append(mask)

                    masks_np = np.stack(masks, axis=0) / 255.0
                    masks = torch.from_numpy(masks_np).float().unsqueeze(0)
                    print(f"      成功加载 PNG 掩膜: {masks.shape}")
                else:
                    print(f"      警告: 掩膜文件不足或格式错误")

    if masks is None:
        print(f"      警告: 未找到掩膜，使用全1掩膜")

    # 5. 推理
    print("  5. 模型推理...")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        clip_tensor = clip_tensor.to(device)
        if masks is not None:
            masks = masks.to(device)

        score_normalized, _ = model(clip_tensor, masks)

    # 6. 反归一化分数（从 [0,1] 到 [6,30]）
    print("  6. 反归一化分数...")
    if hasattr(model, 'denormalize_score'):
        score = model.denormalize_score(
            score_normalized,
            target_min=6.0,   # JIGSAWS GRS 最小值
            target_max=30.0,  # JIGSAWS GRS 最大值
            norm_min=0.0,     # 归一化范围最小
            norm_max=1.0       # 归一化范围最大
        )
    else:
        # 手动反归一化（备用分支）
        score_normalized_np = score_normalized.cpu().numpy() if torch.is_tensor(score_normalized) else score_normalized

        if np.ndim(score_normalized_np) == 0:
            score_normalized_np = float(score_normalized_np)

        # 手动公式：(norm - 0) / 1 * (30 - 6) + 6 = (norm - 0) * 24 + 6
        score = score_normalized_np * 24.0 + 6.0

    return float(score)


def batch_inference(model, video_dir, mask_dir, config, device='cuda'):
    """
    批量推理

    Args:
        model: 训练好的模型
        video_dir: 视频目录
        mask_dir: 掩膜目录
        config: 配置字典
        device: 推理设备

    Returns:
        results: 字典 {video_id: score}
    """
    print("\n" + "="*60)
    print("批 量 推 理 模 式")
    print("="*60)

    # 获取所有视频文件
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    print(f"找到 {len(video_files)} 个视频文件")

    results = {}

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {os.path.basename(video_file)}")

        score = process_video(
            model=model,
            video_path=video_file,
            mask_dir=mask_dir,
            device=device
        )

        if score is not None:
            video_id = os.path.splitext(os.path.basename(video_file))[0]
            results[video_id] = {
                'score': score,
                'video_file': video_file
            }

    # 统计结果
    scores = [r['score'] for r in results.values()]
    if scores:
        print("\n" + "="*60)
        print("推 理 结 果 统 计")
        print("="*60)
        print(f"视频总数: {len(results)}")
        print(f"分数范围: [{min(scores):.2f}, {max(scores):.2f}]")
        print(f"平均分数: {np.mean(scores):.2f}")
        print(f"中位分数: {np.median(scores):.2f}")
        print(f"标准差: {np.std(scores):.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Surgical QA Inference (Bounded Version)')
    parser.add_argument('--config', type=str, default='configs/bounded.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--mask_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default='predictions.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = {}
        import yaml
        config.update(yaml.safe_load(f))

    print("="*60)
    print("Surgical QA 推理 (Bounded Version)")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"检查点: {args.checkpoint}")
    print(f"分数范围 (JIGSAWS GRS): [6.0, 30.0]")
    print("="*60)

    print("\n加载模型...")
    model = build_model_bounded(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型已加载: {args.checkpoint}")
    print(f"分数范围: [{model.score_min}, {model.score_max}]")

    # 单视频推理
    if args.video:
        score = process_video(
            model=model,
            video_path=args.video,
            mask_dir=args.mask_dir,
            device=args.device
        )

        print("\n" + "="*60)
        print("推 理 结 果")
        print("="*60)
        print(f"预测分数 (JIGSAWS GRS): {score:.2f}")
        print("="*60)
        return

    # 批量推理
    elif args.video_dir:
        results = batch_inference(
            model=model,
            video_dir=args.video_dir,
            mask_dir=args.mask_dir,
            config=config,
            device=args.device
        )

        # 保存结果
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n结果已保存到: {args.output}")
    else:
        print("错误: 必须指定 --video 或 --video_dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
