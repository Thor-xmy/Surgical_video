"""
Data Loader for JIGSAWS Frame Dataset

适配 jigsawas_frames/ 数据集结构:
    data_root/
        Knot_tying/
            Knot_tying_B001_capture1_stride3_fps10/
                00000.jpg
                00001.jpg
                ...
        Needle_Passing/
            ...
        Suturing/
            ...

评分文件位置:
    jigsawas/labels/meta_file_Suturing.txt
    jigsawas/labels/meta_file_Needle_Passing.txt
    jigsawas/labels/meta_file_Knot_Tying.txt
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict, Optional
from PIL import Image


class JIGSAWSFrameDataset(Dataset):
    """
    JIGSAWS 帧级别数据集

    每个视频目录包含帧序列的图片
    评分从 meta_file_*.txt 读取
    """
    def __init__(self,
                 data_root,
                 labels_root,
                 clip_length=16,
                 clip_stride=10,
                 spatial_size=224,
                 normalize=True,
                 is_train=True,
                 score_min=6.0,
                 score_max=30.0,
                 target_min=0.0,
                 target_max=1.0,
                 horizontal_flip_prob=0.5,
                 enable_rotation=True,
                 use_cache=True):
        """
        Args:
            data_root: 帧数据根目录 (如: /path/to/jigsawas_frames)
            labels_root: 标签文件根目录 (如: /path/to/jigsawas/labels)
            clip_length: 每个clip的帧数
            clip_stride: 帧采样步长
            spatial_size: 空间尺寸 (H, W)
            normalize: 归一化像素值到 [-1, 1]
            is_train: 训练模式（启用数据增强）
            score_min: 原始分数最小值 (GRS: 6)
            score_max: 原始分数最大值 (GRS: 30)
            target_min: 归一化目标最小值 (如: 0.0)
            target_max: 归一化目标最大值 (如: 1.0)
            horizontal_flip_prob: 水平翻转概率
            enable_rotation: 启用随机旋转
            use_cache: 缓存clips到内存
        """
        self.data_root = data_root
        self.labels_root = labels_root
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.spatial_size = (spatial_size, spatial_size) if isinstance(spatial_size, int) else spatial_size
        self.normalize = normalize
        self.is_train = is_train
        self.use_cache = use_cache

        # 归一化参数
        self.score_min = score_min
        self.score_max = score_max
        self.target_min = target_min
        self.target_max = target_max
        self.score_range = score_max - score_min
        self.target_range = target_max - target_min

        # 数据增强参数
        self.horizontal_flip_prob = horizontal_flip_prob
        self.enable_rotation = enable_rotation

        # 加载标签和视频列表
        self.video_annotations = self._load_annotations()
        self.video_list = list(self.video_annotations.keys())
        self.video_list.sort()

        # 加载clips
        if use_cache:
            print("预加载clips...")
            self.clips = self._preload_clips()
            print(f"  总clips数: {len(self.clips)}")
        else:
            self.clips = None

        print(f"JIGSAWSFrameDataset 初始化完成:")
        print(f"  视频数: {len(self.video_list)}")
        if use_cache:
            print(f"  Clips数: {len(self.clips)}")
        print(f"  Clip长度: {clip_length}, 步长: {clip_stride}")

    def _load_annotations(self):
        """从meta_file_*.txt加载评分"""
        annotations = {}

        # 标准JIGSAWS受试者组 (只使用这8个组)
        JIGSAWS_SUBJECT_GROUPS = set(['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])

        # 任务名称映射
        task_files = {
            'Knot_tying': 'meta_file_Knot_Tying.txt',
            'Needle_Passing': 'meta_file_Needle_Passing.txt',
            'Suturing': 'meta_file_Suturing.txt'
        }

        for task, filename in task_files.items():
            filepath = os.path.join(self.labels_root, filename)
            if not os.path.exists(filepath):
                print(f"Warning: 标签文件不存在: {filepath}")
                continue

            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # 格式: Suturing_B001    N    13    3    2    1    2    2    3
                    parts = line.split()
                    if len(parts) < 3:
                        continue

                    video_name = parts[0]
                    grs_score = float(parts[2])  # GRS分数在第3列

                    # 检查是否为标准JIGSAWS受试者组
                    match = video_name.split('_')
                    if len(match) >= 2:
                        subject = match[1][0]  # B001 -> B
                        if subject not in JIGSAWS_SUBJECT_GROUPS:
                            continue  # 跳过非标准组

                    annotations[video_name] = {
                        'score': grs_score,
                        'task': task
                    }

        print(f"加载了 {len(annotations)} 个视频的评分")
        return annotations

    def _preload_clips(self):
        """预加载所有clips"""
        all_clips = []

        for video_name in self.video_list:
            clips = self._extract_clips_from_video(video_name)
            for clip in clips:
                clip['video_name'] = video_name
                clip['original_score'] = self.video_annotations[video_name]['score']
                # 归一化分数
                normalized_score = (clip['original_score'] - self.score_min) / self.score_range * self.target_range + self.target_min
                clip['score'] = normalized_score
                all_clips.append(clip)

        return all_clips

    def _extract_clips_from_video(self, video_name):
        """从视频提取clips"""
        # 确定视频目录
        video_dir = None
        for task_dir in ['Knot_tying', 'Needle_Passing', 'Suturing']:
            task_path = os.path.join(self.data_root, task_dir)
            if os.path.exists(task_path):
                # 查找匹配的视频目录
                for d in os.listdir(task_path):
                    if d.startswith(video_name + '_') or d == video_name:
                        video_dir = os.path.join(task_path, d)
                        break
                if video_dir:
                    break

        if not video_dir or not os.path.exists(video_dir):
            print(f"Warning: 视频目录不存在: {video_name}")
            return []

        # 获取所有帧文件
        frame_files = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(frame_files) < self.clip_length:
            print(f"Warning: 视频帧数不足 ({len(frame_files)} < {self.clip_length}): {video_name}")
            return []

        # 提取clips
        clips = []
        start_idx = 0

        while start_idx + self.clip_length <= len(frame_files):
            clip_frames = frame_files[start_idx:start_idx + self.clip_length]
            clips.append({
                'frame_files': clip_frames,
                'video_dir': video_dir,
                'start_idx': start_idx,
                'num_frames': len(clip_frames)
            })
            start_idx += self.clip_stride

        return clips

    def _load_frames(self, frame_files, video_dir, apply_flip=False, rotation_k=0):
        """加载并预处理帧"""
        frames = []

        for frame_file in frame_files:
            frame_path = os.path.join(video_dir, frame_file)

            # 加载图像
            img = cv2.imread(frame_path)
            if img is None:
                img = Image.open(frame_path)
                img = np.array(img)

            # BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize
            img = cv2.resize(img, self.spatial_size)

            # 数据增强
            if apply_flip:
                img = np.fliplr(img)

            if rotation_k > 0:
                img = np.rot90(img, k=rotation_k)

            frames.append(img)

        # 转换为张量
        frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
        frames_np = frames_np.transpose(0, 3, 1, 2)  # (T, C, H, W)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0  # (T, C, H, W)

        # 归一化到 [-1, 1]
        if self.normalize:
            frames_tensor = (frames_tensor - 0.5) * 2.0

        # 转置为 (C, T, H, W)
        frames_tensor = frames_tensor.transpose(0, 1)  # (C, T, H, W)

        return frames_tensor

    def __len__(self):
        if self.use_cache and self.clips is not None:
            return len(self.clips)
        else:
            return len(self.video_list)

    def __getitem__(self, idx):
        if self.use_cache and self.clips is not None:
            clip = self.clips[idx]
            video_name = clip['video_name']
            score = clip['score']
        else:
            video_name = self.video_list[idx]
            clips = self._extract_clips_from_video(video_name)
            if len(clips) == 0:
                return self.__getitem__((idx + 1) % len(self))
            clip = clips[0] if not self.is_train else clips[random.randint(0, len(clips)-1)]
            score = (self.video_annotations[video_name]['score'] - self.score_min) / self.score_range * self.target_range + self.target_min

        # 数据增强
        apply_flip = False
        rotation_k = 0

        if self.is_train:
            if self.horizontal_flip_prob > 0:
                apply_flip = (random.random() < self.horizontal_flip_prob)
            if self.enable_rotation and random.random() < 0.5:
                rotation_k = random.choice([1, 2, 3])

        # 加载帧
        frames = self._load_frames(clip['frame_files'], clip['video_dir'], apply_flip, rotation_k)

        return {
            'video_id': video_name,
            'frames': frames,
            'score': torch.tensor(score, dtype=torch.float32),
            'clip_info': clip
        }


class JIGSAWSLOSODataLoader:
    """
    JIGSAWS LOSO-CV 数据加载器

    按受试者组划分数据集
    """
    def __init__(self,
                 data_root,
                 labels_root,
                 batch_size=8,
                 num_workers=4,
                 test_subject_group='B',
                 dataset_kwargs=None):
        """
        Args:
            data_root: 帧数据根目录
            labels_root: 标签文件根目录
            batch_size: 批次大小
            num_workers: 工作进程数
            test_subject_group: 测试集受试者组 (B, C, D, E, F, G, H, I)
            dataset_kwargs: 数据集参数
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset_kwargs is None:
            dataset_kwargs = {}

        # 创建完整数据集
        full_dataset = JIGSAWSFrameDataset(data_root, labels_root, **dataset_kwargs)

        # 按受试者分组
        self.subject_groups = self._group_by_subject(full_dataset.video_list)

        print(f"\n{'='*60}")
        print("数据集受试者组分析")
        print(f"{'='*60}")
        for group, videos in sorted(self.subject_groups.items()):
            print(f"  {group}: {len(videos)} videos")

        # 确定划分
        group_list = sorted(self.subject_groups.keys())
        test_idx = group_list.index(test_subject_group)
        val_idx = (test_idx + 1) % len(group_list)
        val_subject_group = group_list[val_idx]
        train_subject_groups = [g for g in group_list if g not in [test_subject_group, val_subject_group]]

        print(f"\nLOSO-CV划分 (测试集: {test_subject_group}):")
        print(f"  测试集: {test_subject_group}组")
        print(f"  验证集: {val_subject_group}组")
        print(f"  训练集: {', '.join(train_subject_groups)}")

        # 获取视频列表
        test_videos = self.subject_groups[test_subject_group]
        val_videos = self.subject_groups[val_subject_group]
        train_videos = []
        for group in train_subject_groups:
            train_videos.extend(self.subject_groups[group])

        print(f"\n视频数量:")
        print(f"  训练集: {len(train_videos)} videos")
        print(f"  验证集: {len(val_videos)} videos")
        print(f"  测试集: {len(test_videos)} videos")

        # 创建子集
        train_dataset = VideoSubsetDataset(full_dataset, train_videos, use_cache=full_dataset.use_cache)
        val_dataset = VideoSubsetDataset(full_dataset, val_videos, use_cache=full_dataset.use_cache)
        test_dataset = VideoSubsetDataset(full_dataset, test_videos, use_cache=full_dataset.use_cache)

        # 设置is_train
        train_dataset.full_dataset.is_train = True
        val_dataset.full_dataset.is_train = False
        test_dataset.full_dataset.is_train = False

        # 保存划分信息
        self.train_subject_groups = train_subject_groups
        self.val_subject_group = val_subject_group
        self.test_subject_group = test_subject_group

        # 创建DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )

    def _group_by_subject(self, video_list):
        """按受试者分组"""
        groups = {}
        for video_name in video_list:
            # 提取受试者组 (如 Suturing_B001 -> B)
            match = video_name.split('_')
            if len(match) >= 2:
                subject = match[1][0]  # B001 -> B
                if subject not in groups:
                    groups[subject] = []
                groups[subject].append(video_name)
        return groups

    def get_loader(self, split='train'):
        if split == 'train':
            return self.train_loader
        elif split == 'val':
            return self.val_loader
        elif split == 'test':
            return self.test_loader
        else:
            raise ValueError(f"Unknown split: {split}")


class VideoSubsetDataset(Dataset):
    """视频子集数据集"""
    def __init__(self, full_dataset, video_ids, use_cache=True):
        self.full_dataset = full_dataset
        self.video_ids = set(video_ids)
        self.indices = []

        if use_cache and hasattr(full_dataset, 'clips') and full_dataset.clips is not None:
            for idx, clip in enumerate(full_dataset.clips):
                if clip['video_name'] in self.video_ids:
                    self.indices.append(idx)
            print(f"  选择的视频数: {len(video_ids)}")
            print(f"  选择的clips数: {len(self.indices)}")
        else:
            # 不使用缓存时，基于视频ID创建索引
            for idx, video_name in enumerate(full_dataset.video_list):
                if video_name in self.video_ids:
                    self.indices.append(idx)
            print(f"  选择的视频数: {len(video_ids)}")
            print(f"  视频索引数: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]
