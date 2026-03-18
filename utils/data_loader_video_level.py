"""
Video-Level Data Loader for Multi-Clip Pipeline

This DataLoader returns entire videos (not clips) for multi-clip feature extraction.

Key differences from original data_loader.py:
1. Returns entire video: (C, T_total, H, W) instead of single clip (C, 16, H, W)
2. __len__() returns number of videos, not number of clips
3. Each sample corresponds to one video with one ground truth score

Score normalization:
- Original scores [6, 30] are normalized to [0, 1]
- During training: model outputs [0, 1], GT is [0, 1]
- During inference: model outputs [0, 1] -> denormalize to [6, 30] or [1, 10]
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


class VideoLevelDataset(Dataset):
    """
    Video-level dataset for multi-clip feature extraction.

    Each sample contains:
    - Entire video (all frames)
    - Entire masks (all frames)
    - Normalized score [0, 1]

    Structure:
    data_root/
        videos/
            video_001.mp4
            video_002.mp4
            ...
        masks/
            video_001_masks.npy or video_001/
                frame_0000_mask.png
                ...
        annotations.json
            {
                "video_001": {"score": 8.5, ...},
                ...
            }
    """
    def __init__(self,
                 data_root,
                 video_dir='videos',
                 mask_dir='masks',
                 annotation_file='annotations.json',
                 spatial_size=224,
                 spatial_crop='center',
                 normalize=True,
                 use_mask=True,
                 mask_format='png',
                 min_video_length=50,
                 # Score normalization parameters
                 score_min=6.0,     # Original score minimum (GRS: 6)
                 score_max=30.0,    # Original score maximum (GRS: 30)
                 target_min=0.0,    # Normalized target minimum
                 target_max=1.0,    # Normalized target maximum
                 is_train=True):
        """
        Args:
            data_root: Root directory of dataset
            video_dir: Subdirectory containing videos
            mask_dir: Subdirectory containing masks
            annotation_file: JSON file with video scores
            spatial_size: Resize video to (H, W)
            spatial_crop: Crop strategy ('center', 'random', 'none')
            normalize: Normalize pixel values to [-1, 1]
            use_mask: Whether to load masks
            mask_format: Format of mask files ('png', 'npy')
            min_video_length: Minimum video length (shorter videos are skipped)
            score_min: Original score minimum
            score_max: Original score maximum
            target_min: Normalized target minimum
            target_max: Normalized target maximum
            is_train: Enable augmentations during training
        """
        self.data_root = data_root
        self.video_dir = os.path.join(data_root, video_dir)
        self.mask_dir = os.path.join(data_root, mask_dir)
        self.annotation_path = os.path.join(data_root, annotation_file)

        self.spatial_size = (spatial_size, spatial_size) if isinstance(spatial_size, int) else spatial_size
        self.spatial_crop = spatial_crop
        self.normalize = normalize
        self.use_mask = use_mask
        self.mask_format = mask_format
        self.min_video_length = min_video_length

        # Score normalization parameters
        self.score_min = score_min
        self.score_max = score_max
        self.target_min = target_min
        self.target_max = target_max

        self.is_train = is_train

        # Load video list and annotations
        self.video_list = self._get_valid_videos()
        self.annotations = self._load_annotations()

        print(f"Video-Level Dataset initialized:")
        print(f"  Total videos: {len(self.video_list)}")
        print(f"  Spatial size: {self.spatial_size}")
        print(f"  Score range: [{self.score_min}, {self.score_max}] -> [{self.target_min}, {self.target_max}]")

    def _get_valid_videos(self):
        """Get list of video files that have annotations."""
        video_files = sorted(glob.glob(os.path.join(self.video_dir, '*.mp4')))

        # Filter videos that have annotations
        valid_videos = []
        for video_file in video_files:
            video_id = os.path.splitext(os.path.basename(video_file))[0]
            valid_videos.append(video_file)

        print(f"Found {len(valid_videos)} video files")
        return valid_videos

    def _load_annotations(self):
        """Load score annotations from JSON file."""
        with open(self.annotation_path, 'r') as f:
            annotations = json.load(f)
        return annotations

    def _normalize_score(self, score):
        """
        Normalize score from original range to target range.

        Formula: normalized = (score - score_min) / (score_max - score_min) * (target_max - target_min) + target_min

        Args:
            score: Original score in [score_min, score_max]

        Returns:
            normalized: Normalized score in [target_min, target_max]
        """
        score_range = self.score_max - self.score_min
        target_range = self.target_max - self.target_min

        if score_range == 0:
            return self.target_min

        normalized = (score - self.score_min) / score_range * target_range + self.target_min
        return normalized

    def _load_video(self, video_path):
        """Load entire video from file."""
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) < self.min_video_length:
            return None

        return np.array(frames)  # (T, H, W, 3)

    def _load_masks(self, video_id):
        """Load masks for a video."""
        if not self.use_mask:
            return None

        mask_path = os.path.join(self.mask_dir, video_id)

        if self.mask_format == 'npy':
            # Load from numpy file
            masks = np.load(mask_path)  # (T, H, W) or (T, H, W, 1)
            if len(masks.shape) == 4 and masks.shape[-1] == 1:
                masks = masks.squeeze(-1)
            return masks

        elif self.mask_format == 'png':
            # Load from PNG files
            mask_files = sorted(glob.glob(os.path.join(mask_path, '*.png')))

            if len(mask_files) == 0:
                return None

            masks = []
            for mask_file in mask_files:
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    masks.append(mask)

            if len(masks) > 0:
                return np.array(masks)  # (T, H, W)

        return None

    def _preprocess_frames(self, frames):
        """
        Preprocess video frames: resize, crop, normalize.

        Args:
            frames: (T, H, W, 3) numpy array

        Returns:
            frames_tensor: (C, T, H, W) tensor
        """
        T, H, W, C = frames.shape
        processed_frames = []

        for i in range(T):
            frame = frames[i]  # (H, W, 3)

            # Resize
            if H != self.spatial_size[0] or W != self.spatial_size[1]:
                frame = cv2.resize(frame, self.spatial_size, interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            processed_frames.append(frame)

        # Stack frames: (T, H, W, 3)
        frames_array = np.stack(processed_frames)

        # Convert to tensor: (T, H, W, 3) -> (C, T, H, W)
        frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2).float() / 255.0

        # Normalize to [-1, 1]
        if self.normalize:
            frames_tensor = (frames_tensor - 0.5) * 2.0

        return frames_tensor

    def _preprocess_masks(self, masks):
        """
        Preprocess masks: resize, normalize to [0, 1].

        Args:
            masks: (T, H, W) or (T, H, W, 1) numpy array

        Returns:
            masks_tensor: (T, H, W) tensor
        """
        if masks is None:
            return None

        # Ensure shape is (T, H, W)
        if len(masks.shape) == 4:
            masks = masks.squeeze(-1)

        T, H, W = masks.shape
        processed_masks = []

        # Get video spatial size from first frame
        video_H = self.spatial_size[0]
        video_W = self.spatial_size[1]

        for i in range(T):
            mask = masks[i]  # (H, W)

            # Resize
            if H != video_H or W != video_W:
                mask = cv2.resize(mask, (video_W, video_H), interpolation=cv2.INTER_NEAREST)

            # Normalize to [0, 1]
            mask = mask / 255.0

            processed_masks.append(mask)

        # Stack masks: (T, H, W)
        masks_array = np.stack(processed_masks)
        masks_tensor = torch.from_numpy(masks_array).float()

        return masks_tensor

    def __len__(self):
        """
        Return number of videos.

        Video-level: each video is one sample (not each clip).
        """
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Get a video from dataset.

        Returns:
            video: (C, T_total, H, W) - Entire video
            masks: (T_total, H, W) or None - Entire masks
            score: float - Normalized score [0, 1]
            video_id: str - Video identifier
        """
        video_path = self.video_list[idx]
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Load entire video
        frames = self._load_video(video_path)
        if frames is None:
            # Video too short, return first valid video
            return self.__getitem__(0)

        # Load masks
        masks = self._load_masks(video_id)

        # Preprocess
        video_tensor = self._preprocess_frames(frames)

        if masks is not None:
            # Preprocess masks
            mask_tensor = self._preprocess_masks(masks)

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
        if video_id in self.annotations:
            original_score = self.annotations[video_id].get('score', 0)
            normalized_score = self._normalize_score(original_score)
        else:
            print(f"Warning: No score found for {video_id}, using 0")
            normalized_score = 0.0

        return {
            'video': video_tensor,
            'masks': mask_tensor,
            'score': normalized_score,
            'video_id': video_id
        }


def create_video_level_dataloader(data_root,
                                video_dir='videos',
                                mask='masks',
                                annotation_file='annotations.json',
                                batch_size=4,
                                num_workers=4,
                                spatial_size=224,
                                clip_length=16,
                                clip_stride=10,
                                is_train=True,
                                **kwargs):
    """
    Factory function to create video-level dataloader.

    Args:
        data_root: Root directory of dataset
        video_dir: Subdirectory containing videos
        mask_dir: Subdirectory containing masks
        annotation_file: JSON file with video scores
        batch_size: Batch size
        num_workers: Number of data loading workers
        spatial_size: Resize video to (H, W)
        clip_length: For multi-clip extraction
        clip_stride: For multi-clip extraction
        is_train: Training mode
        **kwargs: Additional arguments for VideoLevelDataset

    Returns:
        dataloader: DataLoader instance
    """
    dataset = VideoLevelDataset(
        data_root=data_root,
        video_dir=video_dir,
        mask_dir=mask,
        annotation_file=annotation_file,
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

    print(f"Created {'training' if is_train else 'validation'} dataloader:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")

    return dataloader


if __name__ == '__main__':
    # Test video-level dataloader
    print("Testing VideoLevelDataset...")

    # Create dummy data for testing
    test_data_root = '/tmp/test_video_level_data'
    os.makedirs(test_data_root, exist_ok=True)
    os.makedirs(os.path.join(test_data_root, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(test_data_root, 'masks'), exist_ok=True)

    # Create a dummy video and annotation
    annotation = {
        'test_video': {'score': 18.0, 'duration': 100}
    }
    with open(os.path.join(test_data_root, 'annotations.json'), 'w') as f:
        json.dump(annotation, f)

    # Note: You need to provide actual video files for testing
    print(f"Test data root: {test_data_root}")
    print("Please provide actual video files for testing.")
