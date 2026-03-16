#!/usr/bin/env python3
"""
Inference Script for Surgical QA Model (Bounded Version)

使用带边界的模型进行推理，包含分数反归一化

修改说明：
1. 使用 SurgicalQAModelBounded（带 Sigmoid 输出）
2. 模型输出范围 [0, 1]，反归一化到目标范围
3. 支持多个目标范围：论文分数 [1, 10] 或原始 GRS [1, 30]

Usage:
    python inference_bounded.py \
        --config configs/bounded.yaml \
        --checkpoint checkpoints/best_model.pth \
        --video /path/to/video.mp4 \
        --mask_dir /path/to/masks
"""

import os
import sys
import torch
import numpy as np
import cv2
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.surgical_qa_model_bounded import SurgicalQAModelBounded, build_model_bounded


def process_video(model, video_path, mask_dir, device='cuda',
                  target_range=(1.0, 10.0)):
    """
    处理单个视频并预测技能分数

    Args:
        model: 训练好的模型
        video_path: 视频文件路径
        mask_dir: 掩膜目录
        device: 推理设备
        target_range: 目标分数范围 (min, max)

    Returns:
        score: 预测分数（在 target_range 范围内）
    """
    print(f"\n正在处理视频: {video_path}")
    print(f"目标分数范围: [{target_range[0]}, {target_range[1]}]")

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
        print(f"      警告: 视频帧数 ({total_frames}) 少于 16，无法处理")
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
        # 尝试加载 npy 格式
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

    # 6. 反归一化分数
    print("  6. 反归一化分数...")
    if hasattr(model, 'denormalize_score'):
        # 使用模型内置的反归一化函数
        score = model.denormalize_score(
            score_normalized,
            score_min=model.score_min,
            score_max=model.score_max,
            target_min=target_range[0],
            target_max=target_range[1]
        )
    else:
        # 手动反归一化
        target_min, target_max = target_range
        score_normalized = score_normalized.cpu().numpy() if torch.is_tensor(score_normalized) else score_normalized

        if score_normalized.ndim == 0:
            score_normalized = float(score_normalized)

        # 反归一化公式: (norm - target_min) / (target_max - target_min) * (score_max - score_min) + score_min
        # 归一化后的值：score_normalized = (original - 1) / 29 * 10
        # 反归一化回 1-10: original = normalized * 29 + 1
        # 所以公式简化为: original = normalized * (score_max - score_min) + score_min
        # 但由于归一化公式可能不同，使用通用公式
        score_range = model.score_max - model.score_min
        target_range_diff = target_max - target_min
        score = (score_normalized - target_min) / target_range_diff * score_range + model.score_min

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

    # 设置目标范围
    target_range = tuple(config.get('inference_target_range', [1.0, 10.0]))
    print(f"目标分数范围: {target_range}")

    # 获取所有视频文件
    import glob
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    print(f"找到 {len(video_files)} 个视频文件")

    results = {}

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {os.path.basename(video_file)}")

        score = process_video(
            model=model,
            video_path=video_file,
            mask_dir=mask_dir,
            device=device,
            target_range=target_range
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

        # 技能等级分布
        expert_count = sum(1 for s in scores if s >= 9.0)
        proficient_count = sum(1 for s in scores if 7.0 <= s < 9.0)
        intermediate_count = sum(1 for s in scores if 5.0 <= s < 7.0)
        novice_count = sum(1 for s in scores if s < 5.0)

        print(f"\n技能等级分布:")
        print(f"  Expert (9.0+): {expert_count} ({expert_count/len(scores)*100:.1f}%)")
        print(f"  Proficient (7.0-8.9): {proficient_count} ({proficient_count/len(results)*100:.1f}%)")
        print(f"  Intermediate (5.0-6.9): {intermediate_count} ({intermediate_count/len(results)*100:.1f}%)")
        print(f"  Novice (<5.0): {novice_count} ({novice_count/len(results)*100:.1f}%)")

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
    parser.add_argument('--target_range', type=str, default='1-10',
                       choices=['1-10', '1-30'], help='目标分数范围')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = {}
        import yaml
        config.update(yaml.safe_load(f))

    # 设置推理目标范围
    if args.target_range == '1-10':
        config['inference_target_range'] = [1.0, 10.0]
    else:  # '1-30'
        config['inference_target_range'] = [1.0, 30.0]

    print("="*60)
    print("Surgical QA 推理 (Bounded Version)")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"检查点: {args.checkpoint}")
    print(f"目标范围: {config['inference_target_range']}")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    model = build_model_bounded(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型已加载: {args.checkpoint}")
    print(f"分数范围 (配置): [{model.score_min}, {model.score_max}]")
    print(f"目标范围 (推理): {config['inference_target_range']}")

    # 单视频推理
    if args.video:
        score = process_video(
            model=model,
            video_path=args.video,
            mask_dir=args.mask_dir,
            device=args.device,
            target_range=tuple(config['inference_target_range'])
        )

        print("\n" + "="*60)
        print("推 理 结 果")
        print("="*60)
        print(f"预测分数: {score:.2f}")
        print(f"分数范围: {config['inference_target_range']}")

        # 技能等级
        if score >= 9.0:
            level = "Expert (专家)"
        elif score >= 7.0:
            level = "Proficient (熟练)"
        elif score >= 5.0:
            level = "Intermediate (中级)"
        else:
            level = "Novice (初级)"

        print(f"技能等级: {level}")
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
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n结果已保存到: {args.output}")

    else:
        print("错误: 必须指定 --video 或 --video_dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
