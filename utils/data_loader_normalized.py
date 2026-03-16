"""
Enhanced Data Loader with Score Normalization

基于原始 data_loader.py，添加分数归一化功能
用于处理 JIGSAWS 数据集的 GRS 分数范围（6-30）

关键修改：
1. 归一化分数：[6, 30] → [0, 1]
2. 使用相同的类名但添加 '_normalized' 后缀避免导入问题
3. 保持所有原始功能不变
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import glob
from typing import List, Dict, Tuple, Optional
import random

# 复制原始数据加载器的所有代码
from data_loader import SurgicalVideoDataset


class SurgicalVideoDatasetNormalized(SurgicalVideoDataset):
    """
    带分数归一化的数据集类

    继承自 SurgicalVideoDataset，添加分数归一化功能
    """
    def __init__(self,
                 data_root,
                 video_dir='videos',
                 mask_dir='masks',
                 annotation_file='annotations.json',
                 clip_length=16,
                 clip_stride=10,
                 spatial_size=224,
                 spatial_crop='center',
                 normalize=True,
                 transform=None,
                 use_mask=True,
                 mask_format='png',
                 cache_clips=True,
                 # NEW: Data augmentation parameters (paper requirements)
                 horizontal_flip_prob=0.5,
                 enable_rotation=True,
                 is_train=True,
                 # NEW: Score normalization parameters
                 score_min=6.0,     # Original score minimum (GRS: 6)
                 score_max=30.0,   # Original score maximum (GRS: 30)
                 target_min=0.0,    # Normalized target minimum
                 target_max=1.0):   # Normalized target maximum
                 ):
        """
        Args:
            data_root: Root directory of dataset
            video_dir: Subdirectory containing videos
            mask_dir: Subdirectory containing masks
            annotation_file: JSON file with video scores
            clip_length: Number of frames per clip
            clip_stride: Stride for clip extraction
            spatial_size: Resize video to (H, W)
            spatial_crop: Crop strategy ('center', 'random', 'none')
            normalize: Normalize pixel values to [-1, 1]
            transform: Additional transforms
            use_mask: Whether to load masks
            mask_format: Format of mask files
            cache_clips: Whether to pre-load all clips (paper1231 style)
            horizontal_flip_prob: Probability of horizontal flip (paper: random flipping)
            enable_rotation: Enable random rotation (paper: random rotation)
            is_train: Enable augmentations during training
            # NEW: Score normalization parameters
            score_min: Original score minimum (e.g., GRS: 6)
            score_max: Original score maximum (e.g., GRS: 30)
            target_min: Normalized target minimum (e.g., 0.0)
            target_max: Normalized target maximum (e.g., 1.0)
        """
        # 调用父类初始化（跳过重新加载annotations）
        SurgicalVideoDataset.__init__(
            self,
            data_root=data_root,
            video_dir=video_dir,
            mask_dir=mask_dir,
            annotation_file=annotation_file,
            clip_length=clip_length,
            clip_stride=clip_stride,
            spatial_size=spatial_size,
            spatial_crop=spatial_crop,
            normalize=normalize,
            transform=transform,
            use_mask=use_mask,
            mask_format=mask_format,
            cache_clips=cache_clips,
            horizontal_flip_prob=horizontal_flip_prob,
            enable_rotation=enable_rotation,
            is_train=is_train
        )

        # 存储归一化参数
        self.score_min = score_min
        self.score_max = score_max
        self.target_min = target_min
        self.target_max = target_max

        # 计算归一化因子
        self.score_range = score_max - score_min
        self.target_range = target_max - target_min

        # 预处理 clips 中的分数
        if self.cache_clips and self.clips is not None:
            print(f"\n归一化分数: [{score_min}, {score_max}] → [{target_min}, {target_max}]")
            print(f"  归一化公式: normalized = (grs - {score_min}) / {score_range} * {target_range} + {target_min}")
            self._normalize_clips()

    def _normalize_clips(self):
        """归一化所有 clips 的分数"""
        for clip in self.clips:
            if 'score' in clip:
                original_score = clip['score']
                # 归一化公式：(score - min) / (max - min) * (target_max - target_min) + target_min
                # 对于 JIGSAWS: normalized = (grs - 6) / 24 * 1.0 = grs / 30 * 1.0
                # 因为 score_range = 30 - 6 = 24, target_range = 1 - 0 = 1, target_min = 0
                # 简化后值：score_normalized = grs / 24 * 1.0 + 0 = (grs - 6) / 24 + 0
                normalized_score = (original_score - self.score_min) / self.score_range * self.target_range + self.target_min

                clip['score'] = normalized_score
                clip['score_original'] = original_score  # 保留原始分数

    def denormalize_score(self, normalized_score):
        """
        反归一化：将归一化后的分数还原到原始分数范围

        Args:
            normalized_score: (B,) or (B, 1) - 归一化后的预测分数，范围 [0, 1]

        Returns:
            score_original: 与输入同形状，范围 [score_min, score_max]
        """
        # 确保 normalized_score 是一维
        if normalized_score.dim() == 0:
            normalized_score = normalized_score.unsqueeze(0)
        elif normalized_score.dim() == 2:
            normalized_score = normalized_score.squeeze(-1)

        # 反归一化公式：(norm - target_min) / (target_max - target_min) * (score_max - score_min) + score_min
        score_range = self.score_range
        target_range = self.target_range

        score_original = (normalized_score - self.target_min) / target_range * score_range + self.score_min

        return score_original

    def get_score_statistics(self):
        """获取分数统计信息"""
        if not self.cache_clips or self.clips is None:
            return None

        scores = [clip['score'] for clip in self.clips]
        scores_original = [clip.get('score_original', clip['score']) for clip in self.clips]

        return {
            'normalized': {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            },
            'original': {
                'min': min(scores_original),
                'max': max(scores_original),
                'mean': np.mean(scores_original),
                'std': np.std(scores_original)
            }
        }


class SurgicalQADataLoaderNormalized:
    """
    带分数归一化的数据加载器
    """
    def __init__(self,
                 data_root,
                 batch_size=8,
                 num_workers=4,
                 train_split=0.8,
                 val_split=0.1,
                 dataset_kwargs=None):
        """
        Args:
            data_root: 数据集根目录
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            train_split: 训练集比例
            val_split: 验证集比例
            dataset_kwargs: SurgicalVideoDatasetNormalized 的参数
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset_kwargs is None:
            dataset_kwargs = {}

        # 创建归一化数据集
        full_dataset = SurgicalVideoDatasetNormalized(data_root, **dataset_kwargs)

        # 打印分数统计
        stats = full_dataset.get_score_statistics()
        if stats is not None:
            print("\n" + "="*60)
            print("分数统计信息")
            print("="*60)
            print("原始分数（GRS 6-30）:")
            print(f"  最小值: {stats['original']['min']:.2f}")
            print(f"  最大值: {stats['original']['max']:.2f}")
            print(f"  平均值: {stats['original']['mean']:.2f}")
            print(f"  标准差: {stats['original']['std']:.2f}")
            print("\n归一化后分数（0-1）:")
            print(f"  最小值: {stats['normalized']['min']:.2f}")
            print(f"  最大值: {stats['normalized']['max']:.2f}")
            print(f"  平均值: {stats['normalized']['mean']:.2f}")
            print(f"  标准差: {stats['normalized']['std']:.2f}")
            print("="*60)

        # 数据集划分
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        indices = list(range(total_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # 创建三个数据集实例（正确设置 is_train 标志）
        self.train_dataset_raw = SurgicalVideoDatasetNormalized(data_root, **dataset_kwargs)
        self.train_dataset_raw.is_train = True

        self.val_dataset_raw = SurgicalVideoDatasetNormalized(data_root, **dataset_kwargs)
        self.val_dataset_raw.is_train = False

        self.test_dataset_raw = SurgicalVideoDatasetNormalized(data_root, **dataset_kwargs)
        self.test_dataset_raw.is_train = False

        # 使用 Subset 包装
        from torch.utils.data import Subset
        self.train_dataset = Subset(self.train_dataset_raw, train_indices)
        self.val_dataset = Subset(self.val_dataset_raw, val_indices)
        self.test_dataset = Subset(self.test_dataset_raw, test_indices)

        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_indices)} 个 clips")
        print(f"  验证集: {len(val_indices)} 个 clips")
        print(f"  测试集: {len(test_indices)} 个 clips")

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )

    def get_loader(self, split='train'):
        """获取指定分割的数据加载器"""
        if split == 'train':
            return self.train_loader
        elif split == 'val':
            return self.val_loader
        elif split == 'test':
            return self.test_loader
        else:
            raise ValueError(f"Unknown split: {split}")


if __name__ == '__main__':
    # 测试归一化数据加载器
    import tempfile

    # 创建测试数据集
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'masks'), exist_ok=True)

    # 创建测试标注（GRS 6-30 范围）
    annotations = {}
    for i in range(10):
        video_id = f"video_{i:03d}"
        grs_score = np.random.randint(6, 31)  # 6-30范围
        annotations[video_id] = {
            'score': grs_score,  # 原始 GRS 分数
            'duration': np.random.randint(60, 300)
        }

    # 保存 annotations.json
    with open(os.path.join(tmp_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=2)

    print("测试数据集已创建")
    print(f"原始 GRS 分数范围: {min(a['score'] for a in annotations.values())}-{max(a['score'] for a in annotations.values())}")

    # 测试数据加载器
    dataloader = SurgicalQADataLoaderNormalized(
        data_root=tmp_dir,
        batch_size=4,
        num_workers=0,
        dataset_kwargs={
            'clip_length': 8,
            'spatial_size': 112,
            'score_min': 6.0,   # GRS 最小值
            'score_max': 30.0,  # GRS 最大值
            'target_min': 0.0,  # 归一化目标最小
            'target_max': 1.0   # 归一化目标最大
        }
    )

    # 测试获取一个 batch
    batch = next(iter(dataloader.get_loader('train')))
    print(f"\nBatch score 范围: {batch['score'].min():.2f} - {batch['score'].max():.2f}")
