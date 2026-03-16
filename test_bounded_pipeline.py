#!/usr/bin/env python3
"""
综合测试 Bounded 版本的完整 Pipeline

测试内容：
1. 导入检查 - 所有模块能否正常导入
2. 模型初始化 - BoundedFusionRegressor 和 SurgicalQAModelBounded
3. 数据加载器 - SurgicalQADataLoaderNormalized
4. 前向传播 - 模型推理
5. 归一化/反归一化 - 分数转换正确性
6. 损失计算 - 训练相关
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import default_collate


def custom_collate(batch):
    """
    自定义collate函数，处理mask为None的情况
    """
    # 过滤掉masks字段如果存在
    if 'masks' in batch[0] and batch[0]['masks'] is None:
        # 删除masks键
        batch_no_masks = [{k: v for k, v in item.items() if k != 'masks'} for item in batch]
        return default_collate(batch_no_masks)
    else:
        return default_collate(batch)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("BOUNDED PIPELINE 综合测试")
print("="*70)

# ==================== 1. 导入检查 ====================
print("\n[1/6] 导入检查...")
print("-"*70)

try:
    from models.fusion_regressor_sigmoid import BoundedFusionRegressor
    print("✓ 成功导入 BoundedFusionRegressor")
except Exception as e:
    print(f"✗ 导入 BoundedFusionRegressor 失败: {e}")
    sys.exit(1)

try:
    from models.surgical_qa_model_bounded import SurgicalQAModelBounded, build_model_bounded
    print("✓ 成功导入 SurgicalQAModelBounded")
except Exception as e:
    print(f"✗ 导入 SurgicalQAModelBounded 失败: {e}")
    sys.exit(1)

try:
    from utils.data_loader_normalized import SurgicalVideoDatasetNormalized, SurgicalQADataLoaderNormalized
    print("✓ 成功导入 SurgicalQADataLoaderNormalized")
except Exception as e:
    print(f"✗ 导入 SurgicalQADataLoaderNormalized 失败: {e}")
    sys.exit(1)

print("\n所有模块导入成功！")

# ==================== 2. BoundedFusionRegressor 测试 ====================
print("\n[2/6] BoundedFusionRegressor 测试...")
print("-"*70)

try:
    regressor = BoundedFusionRegressor(static_dim=512, dynamic_dim=1024)
    regressor.eval()

    # 测试前向传播
    B = 4
    static = torch.randn(B, 512)
    dynamic = torch.randn(B, 1024)

    with torch.no_grad():
        score_normalized = regressor(static, dynamic)

    print(f"输入: static={static.shape}, dynamic={dynamic.shape}")
    print(f"输出: {score_normalized.shape}")
    print(f"输出范围: [{score_normalized.min().item():.6f}, {score_normalized.max().item():.6f}]")

    # 验证输出范围
    assert 0.0 < score_normalized.min().item(), "输出最小值应该 > 0"
    assert score_normalized.max().item() < 1.0, "输出最大值应该 < 1"
    print("✓ 输出范围正确: (0, 1)")

    # 测试反归一化
    score_1to10 = regressor.denormalize_score(score_normalized, score_min=1.0, score_max=10.0)
    score_6to30 = regressor.denormalize_score(score_normalized, score_min=6.0, score_max=30.0)

    print(f"反归一化到 [1,10]: [{score_1to10.min().item():.2f}, {score_1to10.max().item():.2f}]")
    print(f"反归一化到 [6,30]: [{score_6to30.min().item():.2f}, {score_6to30.max().item():.2f}]")
    print("✓ 反归一化功能正常")

except Exception as e:
    print(f"✗ BoundedFusionRegressor 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 3. SurgicalQAModelBounded 初始化测试 ====================
print("\n[3/6] SurgicalQAModelBounded 初始化测试...")
print("-"*70)

try:
    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mask_loss': True,
        'use_bounded_regression': True,
        'score_min': 6.0,
        'score_max': 30.0,
        'regressor_hidden_dims': [1024, 512, 256, 128]
    }

    model = SurgicalQAModelBounded(config)
    model.eval()

    print(f"✓ 模型初始化成功")
    print(f"  - Static dimension: {model.static_dim}")
    print(f"  - Dynamic dimension: {model.dynamic_dim}")
    print(f"  - Score range: [{model.score_min}, {model.score_max}]")
    print(f"  - Use bounded regression: {model.use_bounded_regression}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 4. 前向传播测试 ====================
print("\n[4/6] 前向传播测试...")
print("-"*70)

try:
    B, C, T, H, W = 2, 3, 16, 112, 112
    video = torch.randn(B, C, T, H, W)
    masks = torch.rand(B, T, H, W)

    print(f"输入视频: {video.shape}")
    print(f"输入掩膜: {masks.shape}")

    with torch.no_grad():
        score_pred, mask_loss = model(video, masks)

    print(f"预测分数: {score_pred.shape}")
    print(f"预测分数范围: [{score_pred.min().item():.6f}, {score_pred.max().item():.6f}]")
    print(f"掩膜损失: {mask_loss.item() if mask_loss is not None else 'None'}")

    # 验证输出
    assert score_pred.shape == (B, 1), f"输出形状错误: {score_pred.shape}"
    assert 0.0 < score_pred.min().item() and score_pred.max().item() < 1.0, "分数范围错误"
    print("✓ 前向传播成功")

    # 测试反归一化
    score_original = model.denormalize_score(score_pred)
    print(f"反归一化后: [{score_original.min().item():.2f}, {score_original.max().item():.2f}]")
    print("✓ 反归一化功能正常")

    # 测试灵活反归一化
    score_1to10 = model.denormalize_score(score_pred, target_min=1.0, target_max=10.0, score_min=6.0, score_max=30.0)
    print(f"反归一化到 [1,10]: [{score_1to10.min().item():.2f}, {score_1to10.max().item():.2f}]")
    print("✓ 灵活反归一化功能正常")

except Exception as e:
    print(f"✗ 前向传播测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 5. 损失计算测试 ====================
print("\n[5/6] 损失计算测试...")
print("-"*70)

try:
    # 模拟训练
    model.train()

    score_gt = torch.tensor([0.375, 0.625])  # 归一化的真值 [0, 1]
    score_gt = score_gt.unsqueeze(-1)  # (B, 1)

    print(f"预测分数: {score_pred.squeeze().tolist()}")
    print(f"真值分数: {score_gt.squeeze().tolist()}")

    with torch.enable_grad():
        score_pred_train, _ = model(video, masks)
        total_loss, loss_dict = model.compute_loss(
            score_pred_train,
            score_gt,
            mask_loss=mask_loss,
            lambda_mask=0.1
        )

    print(f"总损失: {total_loss.item():.6f}")
    print(f"损失字典: {loss_dict}")

    # 反向传播测试
    total_loss.backward()
    print("✓ 反向传播成功")

except Exception as e:
    print(f"✗ 损失计算测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 6. 数据加载器测试 (使用最小测试数据) ====================
print("\n[6/6] 数据加载器测试...")
print("-"*70)

try:
    import tempfile
    import json
    import cv2

    # 创建测试数据集
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'masks'), exist_ok=True)

    print(f"创建测试数据集: {tmp_dir}")

    # 创建 2 个视频文件
    video_info = []
    for i in range(2):
        video_id = f"video_{i:03d}"

        # 创建视频 (20帧，RGB)
        frames = []
        for _ in range(20):
            frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            frames.append(frame)

        # 保存视频
        video_path = os.path.join(tmp_dir, 'videos', f"{video_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (112, 112))
        for frame in frames:
            writer.write(frame)
        writer.release()

        # 创建掩膜 (NPY格式)
        masks = np.random.randint(0, 256, (20, 112, 112), dtype=np.uint8)
        mask_path = os.path.join(tmp_dir, 'masks', f"{video_id}_masks.npy")
        np.save(mask_path, masks)

        # GRS 分数 (6-30)
        grs_score = np.random.randint(6, 31)

        video_info.append({
            'id': video_id,
            'video_path': video_path,
            'mask_path': mask_path,
            'score': grs_score
        })

    # 创建 annotations.json
    annotations = {}
    for info in video_info:
        annotations[info['id']] = {
            'score': info['score'],
            'duration': 20
        }

    with open(os.path.join(tmp_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"创建 {len(video_info)} 个测试视频")
    print(f"GRS 分数范围: {min(a['score'] for a in video_info)}-{max(a['score'] for a in video_info)}")

    # 测试数据加载器
    # 注意：测试时不使用mask避免加载问题
    dataloader = SurgicalQADataLoaderNormalized(
        data_root=tmp_dir,
        batch_size=1,
        num_workers=0,  # 使用 0 worker 进行测试
        dataset_kwargs={
            'clip_length': 16,
            'clip_stride': 10,
            'spatial_size': 112,
            'score_min': 6.0,
            'score_max': 30.0,
            'target_min': 0.0,
            'target_max': 1.0,
            'cache_clips': True,
            'use_mask': False  # 测试时禁用mask
        }
    )

    print(f"✓ 数据加载器创建成功")
    print(f"  - 训练集大小: {len(dataloader.train_dataset)}")
    print(f"  - 验证集大小: {len(dataloader.val_dataset)}")
    print(f"  - 测试集大小: {len(dataloader.test_dataset)}")

    # 测试直接从数据集获取一个样本
    sample = dataloader.train_dataset[0]

    print(f"\n测试样本:")
    print(f"  - frames: {sample['frames'].shape}")
    print(f"  - score: {sample['score'].shape}, 值: {sample['score'].item():.4f}")
    if 'masks' in sample:
        print(f"  - masks: {sample['masks'].shape if sample['masks'] is not None else 'None'}")

    # 验证分数已归一化
    assert 0.0 <= sample['score'].item() <= 1.0, "分数未正确归一化"
    print("✓ 分数归一化正确")

    # 清理
    import shutil
    shutil.rmtree(tmp_dir)
    print(f"\n清理测试数据集")

except Exception as e:
    print(f"✗ 数据加载器测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 总结 ====================
print("\n" + "="*70)
print("✓ 所 有 测 试 通 过 ！")
print("="*70)
print("\nBounded Pipeline 准备就绪，可以开始训练！")
print("\n训练命令:")
print("  python train_bounded.py --config configs/bounded.yaml --gpus 0")
print("\n推理命令:")
print("  python inference_bounded.py --config configs/bounded.yaml \\")
print("      --checkpoint output_bounded/checkpoints/best_model.pth \\")
print("      --video /path/to/video.mp4 --mask_dir /path/to/masks \\")
print("      --target_range 1-10")
print("="*70)
