"""
Video-Level Data Loader Loader for Multi-Clip Pipeline with Frame Sequence Support

This DataLoader:
1. Loads video frame sequences (not MP4 files)
2. Handles case-insensitive ID matching between JSON and directories
3. Supports proper train/val/test data splitting
4. Compatible with Multi-Clip Bounded Regression Pipeline

Key features:
- Loads frame sequences from directories (frame_0000.jpg, ...)
- Handles naming: JSON uses uppercase/underscore, dirs use lowercase/hyphen
- Automatic train/val/test splitting with configurable ratios
- Frame-level preprocessing and normalization
- Mask loading with automatic alignment
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import glob
from typing import List, Dict, Optional
import random
import torchvision.transforms.functional as TF



class VideoLevelDatasetFrames(Dataset):
    """
    Video-level dataset that loads frame sequences and supports proper data splitting.

    Each sample contains:
    - Entire video (all frames as sequence)
    - Entire masks (all frames)
    - Normalized score [0, 1]
    - Video ID

    Structure:
    data_root/
        heichole_frames/                    # Frame sequences
            Hei-Chole1_calot/
                frame_0000.jpg
                frame_0001.jpg
                ...
            Hei-Chole1_dissection/
                ...
        batch_masks_merged/                 # Mask files
            Hei-Chole1_calot/
                Hei-Chole1_calot_masks.npy
            Hei-Chole1_dissection/
                ...
        annotations_combined.json             # Scores (uppercase IDs)
            {
                "Hei-Chole1_Calot": {"score": 21, ...},
                "Hei-Chole1_Dissection": {"score": 21, ...},
                ...
            }
    """
    def __init__(self,
                 data_root,
                 frames_dir='heichole_frames',
                 mask_dir='batch_masks_merged',
                 annotation_file='annotations_combined.json',
                 # Data splitting parameters
                 subset='train',                      # 'train', 'val', 'test'
                 train_ratio=0.7,                     # 70% for training
                 val_ratio=0.15,                      # 15% for validation
                 test_ratio=0.15,                     # 15% for testing
                 split_seed=42,                        # Reproducible splits
                 # Other parameters
                 spatial_size=224,
                 normalize=True,
                 use_mask=False,
                 skip_val=False,
                 min_video_length=50,
                 clip_length=16,                           # For compatibility (not used in frames loader)
                 clip_stride=10,                           # For compatibility (not used in frames loader)
                 # Score normalization parameters
                 score_min=6.0,                         # Original score minimum
                 score_max=30.0,                        # Original score maximum
                 target_min=0.0,                        # Normalized target minimum
                 target_max=1.0,                        # Normalized target maximum
                 num_folds=None,
                 current_fold=0,
                 sub_score_min=1.0, 
                 sub_score_max=5.0,
                 normalize_scores=True,  # 🌟 新增参数
                 is_train=True):
        """
        Args:
            data_root: Root directory of dataset
            frames_dir: Directory containing frame sequences
            mask_dir: Directory containing mask files
            annotation_file: JSON file with video scores
            subset: Which subset to use ('train', 'val', 'test')
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            split_seed: Random seed for reproducible splits
            spatial_size: Resize frames to (H, W)
            normalize: Normalize pixel values to [-1, 1]
            use_mask: Whether to load masks
            min_video_length: Minimum video length (shorter videos are skipped)
            score_min: Original score minimum
            score_max: Original score maximum
            target_min: Normalized target minimum
            target_max: Normalized target maximum
            is_train: Enable augmentations during training
        """
        self.data_root = data_root
        self.frames_dir = os.path.join(data_root, frames_dir)
        self.mask_dir = os.path.join(data_root, mask_dir)
        self.annotation_path = os.path.join(data_root, annotation_file)

        # Data splitting parameters
        self.subset = subset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.num_folds = num_folds         # 👈 新增保存
        self.current_fold = current_fold   # 👈 新增保存
        self.spatial_size = (spatial_size, spatial_size) if isinstance(spatial_size, int) else spatial_size
        self.normalize = normalize
        self.use_mask = use_mask
        self.min_video_length = min_video_length

        # Clip parameters (for compatibility with original loader)
        self.clip_length = clip_length
        self.clip_stride = clip_stride

        # Score normalization parameters
        self.score_min = score_min
        self.score_max = score_max
        self.normalize_scores = normalize_scores # 🌟 保存归一化开关
        self.target_min = target_min
        self.target_max = target_max
        self.sub_score_min = sub_score_min  # 🌟 保存子项最小边界
        self.sub_score_max = sub_score_max  # 🌟 保存子项最大边界
        self.is_train = is_train
        self.skip_val = skip_val
        # Load all video IDs and annotations
        print("\n" + "="*70)
        print("Initializing VideoLevelDatasetFrames")
        print("="*70)
        self.all_video_ids = self._get_all_video_ids()
        self.annotations = self._load_annotations()

        # Perform data split
        self.video_ids = self._split_data()

        print(f"  Subset: {self.subset}")
        print(f"  Total videos: {len(self.all_video_ids)}")
        print(f"  Subset videos: {len(self.video_ids)}")
        print(f"  Spatial size: {self.spatial_size}")
        print(f"  Score range: [{self.score_min}, {self.score_max}] -> [{self.target_min}, {self.target_max}]")
        print(f"  Use masks: {self.use_mask}")
        print("="*70 + "\n")

    def _get_all_video_ids(self):
        """Get all video IDs from frames directory."""
        if not os.path.exists(self.frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")

        video_ids = sorted(os.listdir(self.frames_dir))
        # Filter directories only
        video_ids = [v for v in video_ids if os.path.isdir(os.path.join(self.frames_dir, v))]

        print(f"Found {len(video_ids)} video directories in {self.frames_dir}")
        return video_ids

    def _load_annotations(self):
        """Load annotations with case-insensitive ID matching."""
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        with open(self.annotation_path, 'r') as f:
            annotations = json.load(f)

        # Create case-insensitive mapping
        # JSON keys: Hei-Chole10_Calot (uppercase, underscore)
        # Dir names: Hei-Chole10_calot (lowercase, hyphen)
        # Convert: Hei-Chole10_Calot -> Hei-Chole10_calot
        annotations_normalized = {}
        for key, value in annotations.items():
            # Convert to full lowercase (replace underscore with hyphen)
            key_normalized = key.lower().replace('_', '-')
            annotations_normalized[key_normalized] = value

        print(f"Loaded {len(annotations_normalized)} annotations from {self.annotation_path}")
        print(f"Applied normalization (lowercase, _->-): Hei-Chole10_Calot -> hei-chole10-calot")

        return annotations_normalized
    '''
    def _split_data(self):
        """Split data into train/val/test subsets."""
        random.seed(self.split_seed)

        # Shuffle for split
        shuffled_ids = self.all_video_ids.copy()
        random.shuffle(shuffled_ids)

        n_total = len(shuffled_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train+n_val]
        test_ids = shuffled_ids[n_train+n_val:]

        print(f"Data split with seed={self.split_seed}:")
        print(f"  Train: {len(train_ids)} ({len(train_ids)/n_total*100:.1f}%)")
        print(f"  Val:   {len(val_ids)} ({len(val_ids)/n_total*100:.1f}%)")
        print(f"  Test:  {len(test_ids)} ({len(test_ids)/n_total*100:.1f}%)")

        if self.subset == 'train':
            print(f"Selected subset: train ({len(train_ids)} videos)")
            return train_ids
        elif self.subset == 'val':
            print(f"Selected subset: val ({len(val_ids)} videos)")
            return val_ids
        elif self.subset == 'test':
            print(f"Selected subset: test ({len(test_ids)} videos)")
            return test_ids
        else:
            print(f"Warning: Unknown subset '{self.subset}', using all videos")
            return shuffled_ids
    '''
    def _split_data(self):
        """Split data into train/val/test subsets or K-Folds."""
        random.seed(self.split_seed)

        # Shuffle for split
        shuffled_ids = self.all_video_ids.copy()
        random.shuffle(shuffled_ids)

        # 🌟 新增：K折交叉验证逻辑
        if hasattr(self, 'num_folds') and self.num_folds is not None:
            n_total = len(shuffled_ids)
            fold_size = int(np.ceil(n_total / self.num_folds))
            
            test_start = self.current_fold * fold_size
            test_end = min(test_start + fold_size, n_total)
            
            test_ids = shuffled_ids[test_start:test_end]
            remaining_ids = [vid for vid in shuffled_ids if vid not in test_ids]
            
            # 从剩余的样本中划出 15% 做验证集，其余做训练集
            #val_size = max(1, int(len(remaining_ids) * 0.15 + 1)) #####################################
            #val_ids = remaining_ids[:val_size]
            #train_ids = remaining_ids[val_size:]
            # 从剩余的样本中划出 15% 做验证集，其余做训练集
            if getattr(self, 'skip_val', False):
                val_ids = []  # 验证集置空
                train_ids = remaining_ids  # 🌟 训练集吃掉所有剩余视频！
            else:
                val_size = max(1, int(len(remaining_ids) * 0.15 + 1)) #####################################
                val_ids = remaining_ids[:val_size]
                train_ids = remaining_ids[val_size:]

            print(f"K-Fold {self.current_fold+1}/{self.num_folds} Data split:")
            print(f"  Train: {len(train_ids)} videos")
            print(f"  Val:   {len(val_ids)} videos")
            print(f"  Test:  {len(test_ids)} videos")

            if self.subset == 'train': return train_ids
            elif self.subset == 'val': return val_ids
            elif self.subset == 'test': return test_ids

        # 🌟 原有的简单划分逻辑保持不变
        n_total = len(shuffled_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train+n_val]
        test_ids = shuffled_ids[n_train+n_val:]

        if self.subset == 'train': return train_ids
        elif self.subset == 'val': return val_ids
        elif self.subset == 'test': return test_ids



    def _load_video_frames(self, video_id):
        """Load video frames from directory."""
        frames_path = os.path.join(self.frames_dir, video_id)

        if not os.path.exists(frames_path):
            raise FileNotFoundError(f"Frames directory not found: {frames_path}")

        # Get all frame files
        frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])

        if len(frame_files) == 0:
            raise ValueError(f"No frame files found in {frames_path}")

        # Load frames
        frames = []
        for ff in frame_files:
            frame_path = os.path.join(frames_path, ff)
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to load frame: {frame_path}")
            frames.append(frame)

        return np.array(frames)  # (T, H, W, 3)

    def _load_mask(self, video_id):
        """Load mask for a video."""
        if not self.use_mask:
            return None

        # Path: batch_masks_merged/video_id/video_id_masks.npy
        mask_path = os.path.join(self.mask_dir, video_id, f'{video_id}_masks.npy')

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found: {mask_path}")
            return None

        # Load mask
        mask = np.load(mask_path)  # (T, H, W) or (T, H, W, 1)

        # Handle different mask formats
        if len(mask.shape) == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # (T, H, W, 1) -> (T, H, W)

        return mask

    def _normalize_score(self, score):
        """Normalize score from original range to [0, 1]."""
        score_range = self.score_max - self.score_min
        target_range = self.target_max - self.target_min

        if score_range == 0:
            return self.target_min

        normalized = (score - self.score_min) / score_range * target_range + self.target_min
        return normalized

    def _preprocess_frames(self, frames):
        """Preprocess video frames: resize, normalize."""
        T, H, W, C = frames.shape
        processed_frames = []

        for i in range(T):
            frame = frames[i]  # (H, W, 3)

            # Resize
            # Resize (等比例缩放 + 中心裁剪)
            target_H, target_W = self.spatial_size
            if H != target_H or W != target_W:
                # 1. 计算缩放比例，使得短边对齐目标尺寸
                scale = max(target_H / H, target_W / W)
                new_H, new_W = int(H * scale), int(W * scale)
                frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

                # 2. 从中心裁剪出 112x112
                start_y = (new_H - target_H) // 2
                start_x = (new_W - target_W) // 2
                frame = frame[start_y : start_y + target_H, start_x : start_x + target_W]
            #if H != self.spatial_size[0] or W != self.spatial_size[1]:
                #frame = cv2.resize(frame, self.spatial_size, interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            processed_frames.append(frame)

        # Stack frames: (T, H, W, 3)
        frames_array = np.stack(processed_frames)

        # Convert to tensor: (C, T, H, W)
        frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2).float() / 255.0

        # Normalize to [-1, 1]
        if self.normalize:
            frames_tensor = (frames_tensor - 0.5) * 2.0

        return frames_tensor

    def _preprocess_mask(self, mask):
        """Preprocess mask: resize, normalize."""
        if mask is None:
            return None

        # Ensure shape is (T, H, W)
        if len(mask.shape) == 4:
            mask = mask.squeeze(-1)

        T, H, W = mask.shape
        processed_masks = []

        # Get video spatial size
        video_H = self.spatial_size[0]
        video_W = self.spatial_size[1]

        for i in range(T):
            m = mask[i]  # (H, W)

            # Resize
            # Resize Mask (必须与视频帧的等比缩放和裁剪保持完全一致)
            if H != video_H or W != video_W:
                scale = max(video_H / H, video_W / W)
                new_H, new_W = int(H * scale), int(W * scale)
                m = cv2.resize(m, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

                start_y = (new_H - video_H) // 2
                start_x = (new_W - video_W) // 2
                m = m[start_y : start_y + video_H, start_x : start_x + video_W]
            #if H != video_H or W != video_W:
                #m = cv2.resize(m, (video_W, video_H), interpolation=cv2.INTER_NEAREST)

            # Normalize to [0, 1]
            #m = m / 255.0
            m = (m > 0).astype(np.float32)
            processed_masks.append(m)

        # Stack masks: (T, H, W)
        mask_array = np.stack(processed_masks)
        mask_tensor = torch.from_numpy(mask_array).float()

        return mask_tensor

    def __len__(self):
        """Return number of videos in this subset."""
        return len(self.video_ids)

    def __getitem__(self, idx):
        """Get a video from dataset."""
        video_id = self.video_ids[idx]

        try:
            # Load video frames
            frames = self._load_video_frames(video_id)

            # Check minimum length
            if len(frames) < self.min_video_length:
                print(f"Warning: Video {video_id} has {len(frames)} frames (< {self.min_video_length}), skipping")
                # Return first valid video instead
                return self.__getitem__(0)

            # Load mask
            mask = self._load_mask(video_id)

            # Preprocess video
            video_tensor = self._preprocess_frames(frames)

            # Preprocess mask
            if mask is not None:
                mask_tensor = self._preprocess_mask(mask)

                # Ensure mask length matches video length
                T_video = video_tensor.shape[1]
                T_mask = mask_tensor.shape[0]

                if T_mask < T_video:
                    # Pad masks
                    pad_len = T_video - T_mask
                    mask_tensor = F.pad(mask_tensor, (0, 0, 0, 0, 0, pad_len), 'constant', 0)
                elif T_mask > T_video:
                    # Truncate masks
                    mask_tensor = mask_tensor[:T_video, :, :]
            else:
                mask_tensor = None

            # Get and normalize score
            # 🌟 修复：用同样的规则处理要查询的 key
            video_id_normalized = video_id.lower().replace('_', '-')
            
            if video_id_normalized in self.annotations:
                #original_score = self.annotations[video_id_normalized].get('score', 0)
                #normalized_score = self._normalize_score(original_score)
                anno = self.annotations[video_id_normalized]
                original_score = anno.get('score', 0)
                # 🌟 动态决定是否归一化总分
                normalized_score = self._normalize_score(original_score) if self.normalize_scores else float(original_score)
                #normalized_score = self._normalize_score(original_score)
                # 🌟 新增：读取子项分数
                individual_scores = anno.get('individual_scores', [])
            else:
                print(f"Warning: No score found for {video_id} (searched as: {video_id_normalized}), using 0")
                normalized_score = 0.0
                individual_scores = []
            # ==========================================
            # 🌟 新增：视频一致性数据增强 (Video-Consistent Augmentation)
            # ==========================================
            if self.is_train:  # 只有在训练集且开启训练模式时才进行增强
                
                # 策略 1：随机水平翻转 (概率 50%)
                if random.random() < 0.5:
                    # video_tensor 形状是 (C, T, H, W)，沿着宽度 W (dim=-1) 翻转
                    video_tensor = torch.flip(video_tensor, dims=[-1])
                    if mask_tensor is not None:
                        # mask_tensor 形状是 (T, H, W)，同样沿宽度翻转以保持对齐
                        mask_tensor = torch.flip(mask_tensor, dims=[-1])

                # 策略 2：随机色彩抖动 (概率 50%，仅对视频画面有效)
                if random.random() < 0.5:
                    # ⚠️ 关键：提取出一个全局固定的随机亮度/对比度系数，应用到所有帧，防止视频闪烁
                    brightness_factor = random.uniform(0.8, 1.2) # 亮度波动 ±20%
                    contrast_factor = random.uniform(0.8, 1.2)   # 对比度波动 ±20%

                    # torchvision 需要的输入格式是 (..., C, H, W)
                    # 我们将视频从 (C, T, H, W) 转置为 (T, C, H, W)
                    video_tensor_t = video_tensor.permute(1, 0, 2, 3)
                    
                    # 批量调整整段视频的亮度和对比度
                    video_tensor_t = TF.adjust_brightness(video_tensor_t, brightness_factor)
                    video_tensor_t = TF.adjust_contrast(video_tensor_t, contrast_factor)
                    
                    # 调整完毕后再转置回原始形状 (C, T, H, W)
                    video_tensor = video_tensor_t.permute(1, 0, 2, 3)
            # ==========================================

            # 🌟🌟🌟 这里是本次修复的核心：条件性地构建返回字典
            sample = {
                'video': video_tensor,                                             # (C, T, H, W)
                'score': torch.tensor(normalized_score, dtype=torch.float32),      # float in [0, 1]
                'video_id': video_id                                               # 这里依然返回原始 ID，方便后续查看
            }

            # 🌟 新增：如果 JSON 中有小分，将其独立归一化并存入 sample
            if len(individual_scores) > 0:
                if self.normalize_scores:
                    sub_range = self.sub_score_max - self.sub_score_min
                    sub_range = sub_range if sub_range > 0 else 1.0
                    norm_subs = [(s - self.sub_score_min) / sub_range for s in individual_scores]
                    sample['sub_scores'] = torch.tensor(norm_subs, dtype=torch.float32)
                else:
                    sample['sub_scores'] = torch.tensor([float(s) for s in individual_scores], dtype=torch.float32)
            # 如果 mask_tensor 存在，才往字典里加 'masks' 键
            if mask_tensor is not None:
                sample['masks'] = mask_tensor

            return sample
        #except Exception as e:
            #print(f"Error loading {video_id}: {e}")
            # Return first valid video instead
            #return self.__getitem__(0)
        except Exception as e:
            print(f"Error loading {video_id}: {e}")
            # 不要递归调用 self.__getitem__(0)
            # 建议返回一个全零的 Dummy 数据或抛出错误以便调试
            raise e


def create_dataloader_with_split(data_root,
                              frames_dir='heichole_frames',
                              mask_dir='batch_masks_merged',
                              annotation_file='annotations_combined.json',
                              batch_size=4,
                              num_workers=4,
                              spatial_size=224,
                              # Data splitting
                              subset='train',
                              train_ratio=0.7,
                              val_ratio=0.15,
                              test_ratio=0.15,
                              split_seed=42,
                              num_folds=None,    # 👈 1. 接收参数
                              current_fold=0,    # 👈 2. 接收参数
                              is_train=True,
                              skip_val=False,
                              sub_score_min=1.0,  # 🌟 接收外部传参
                              sub_score_max=5.0,  # 🌟 接收外部传参
                              normalize_scores=True,
                              **kwargs):
    """
    Factory function to create dataloader with proper data splitting.

    Args:
        data_root: Root directory of dataset
        frames_dir: Directory containing frame sequences
        mask_dir: Directory containing mask files
        annotation_file: JSON file with video scores
        batch_size: Batch size
        num_workers: Number of data loading workers
        spatial_size: Resize video to (H, W)
        subset: Which subset to use ('train', 'val', 'test')
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        split_seed: Random seed for reproducible splits
        is_train: Training mode
        **kwargs: Additional arguments for VideoLevelDatasetFrames

    Returns:
        dataloader: DataLoader instance
    """
    dataset = VideoLevelDatasetFrames(
        data_root=data_root,
        frames_dir=frames_dir,
        mask_dir=mask_dir,
        annotation_file=annotation_file,
        subset=subset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        num_folds=num_folds,          # 👈 3. 传给 Dataset
        current_fold=current_fold,    # 👈 4. 传给 Dataset
        skip_val=skip_val,            # 🌟 传给 Dataset
        sub_score_min=sub_score_min,  # 🌟 传给 Dataset
        sub_score_max=sub_score_max,  # 🌟 传给 Dataset
        normalize_scores=normalize_scores, # 🌟 传给 Dataset
        spatial_size=spatial_size,
        is_train=is_train,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )

    print(f"Created {subset} dataloader:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Shuffle: {is_train}")
    print(f"  Drop last: {is_train}")

    return dataloader


if __name__ == '__main__':
    # Test video-level dataloader with frame sequences
    print("Testing Video:LevelDatasetFrames...")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Test with your data
    test_data_root = '/root/autodl-tmp/SV_test/data'

    print(f"\nTest data root: {test_data_root}")

    # Test all subsets
    for subset in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Testing {subset} subset")
        print(f"{'='*70}")

        dataloader = create_dataloader_with_split(
            data_root=test_data_root,
            batch_size=2,
            num_workers=0,
            spatial_size=224,
            subset=subset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            split_seed=42,
            is_train=(subset == 'train')
        )

        print(f"\nLoading first batch...")
        for batch in dataloader:
            print(f"  Batch keys: {batch.keys()}")
            print(f"  Video shape: {batch['video'].shape}")
            print(f"  Score shape: {batch['score'].shape}")
            print(f"  Score range: [{batch['score'].min().item():.4f}, {batch['score'].max().item():.4f}]")
            print(f"  Video IDs: {batch['video_id']}")
            break

    print("\n" + "="*70)
    print("✓ Test passed!")
    print("="*70)