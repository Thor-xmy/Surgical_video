"""
Data Loader Module for Surgical QA Dataset

Handles loading video clips and corresponding:
- Ground truth skill scores
- Pre-computed instrument masks
- Video metadata
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


class SurgicalVideoDataset(Dataset):
    """
    Dataset for surgical video quality assessment.

    Structure:
    data_root/
        videos/
            video_001.mp4
            video_002.mp4
            ...
        masks/
            video_001_masks.npy  or  video_001/
                frame_0000_mask.png
                frame_0001_mask.png
                ...
        annotations.json
            {
                "video_001": {"score": 8.5, "duration": 120, ...},
                "video_002": {"score": 7.2, "duration": 115, ...},
                ...
            }
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
                 mask_format='png'):
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
        """
        self.data_root = data_root
        self.video_dir = os.path.join(data_root, video_dir)
        self.mask_dir = os.path.join(data_root, mask_dir)
        self.annotation_path = os.path.join(data_root, annotation_file)

        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.spatial_size = spatial_size
        self.spatial_crop = spatial_crop
        self.normalize = normalize
        self.transform = transform
        self.use_mask = use_mask
        self.mask_format = mask_format
        self.is_train = True

        # Load annotations
        self.annotations = self._load_annotations()

        # Get video list
        self.video_list = self._get_video_list()

        print(f"SurgicalVideoDataset initialized:")
        print(f"  Data root: {data_root}")
        print(f"  Number of videos: {len(self.video_list)}")
        print(f"  Clip length: {clip_length}, stride: {clip_stride}")
        print(f"  Spatial size: {spatial_size}")
        print(f"  Use masks: {use_mask}")

    def _load_annotations(self):
        """Load video annotations from JSON file."""
        if not os.path.exists(self.annotation_path):
            print(f"Warning: Annotation file not found: {self.annotation_path}")
            return {}

        with open(self.annotation_path, 'r') as f:
            annotations = json.load(f)

        print(f"Loaded {len(annotations)} annotations")
        return annotations

    def _get_video_list(self):
        """Get list of video files."""
        video_files = sorted(glob.glob(os.path.join(self.video_dir, '*.mp4')))

        # Filter videos that have annotations
        valid_videos = []
        for video_file in video_files:
            video_id = os.path.splitext(os.path.basename(video_file))[0]
            if video_id in self.annotations:
                valid_videos.append(video_file)

        print(f"Found {len(valid_videos)} valid videos with annotations")
        return valid_videos

    def _extract_clips_from_video(self, video_path):
        """
        Extract clips from a video file.

        Returns list of (frames, start_frame) tuples.
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames < self.clip_length:
            print(f"Warning: Video too short ({total_frames} < {self.clip_length})")
            cap.release()
            return []

        # Extract clips with stride
        clips = []
        start_frame = 0

        while start_frame + self.clip_length <= total_frames:
            # Extract frames
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for i in range(self.clip_length):
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            if len(frames) == self.clip_length:
                clips.append({
                    'frames': frames,
                    'start_frame': start_frame,
                    'end_frame': start_frame + self.clip_length,
                    'video_path': video_path
                })

            start_frame += self.clip_stride

        cap.release()
        return clips

    def _load_masks(self, video_id):
        """Load masks for a video."""
        if not self.use_mask:
            return None

        mask_path = os.path.join(self.mask_dir, video_id)

        if self.mask_format == 'npy':
            masks = np.load(mask_path + '_masks.npy')
            return torch.from_numpy(masks).float()
        else:
            # Load from image files
            mask_dir = os.path.join(self.mask_dir, video_id)
            if os.path.isdir(mask_dir):
                mask_files = sorted(glob.glob(os.path.join(mask_dir, '*_mask.png')))
                masks = []
                for mask_file in mask_files[:self.clip_length]:
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    mask = mask / 255.0
                    masks.append(mask)

                if len(masks) > 0:
                    masks = np.stack(masks, axis=0)
                    return torch.from_numpy(masks).float()

        return None

    def _preprocess_frames(self, frames):
        """
        Preprocess video frames.

        Args:
            frames: List of (H, W, C) numpy arrays

        Returns:
            frames_tensor: (C, T, H, W) torch tensor
        """
        T = len(frames)
        H, W, C = frames[0].shape

        # Convert to tensor
        frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
        frames_np = frames_np.transpose(0, 3, 1, 2)  # (T, C, H, W)

        # Resize
        if (H, W) != self.spatial_size:
            resized_frames = []
            for t in range(T):
                frame = frames_np[t]  # (C, H, W)
                frame = frame.transpose(1, 2, 0)  # (H, W, C)
                resized = cv2.resize(frame, self.spatial_size)
                resized = resized.transpose(2, 0, 1)  # (C, H, W)
                resized_frames.append(resized)
            frames_np = np.stack(resized_frames, axis=0)  # (T, C, H, W)

        # Crop
        if self.spatial_crop == 'center':
            # Already centered by resize
            pass
        elif self.spatial_crop == 'random':
            # Random crop would be done in training loop
            pass

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0

        # Normalize to [-1, 1]
        if self.normalize:
            frames_tensor = (frames_tensor - 0.5) * 2.0

        # Transpose to (C, T, H, W)
        frames_tensor = frames_tensor.transpose(0, 1)  # (C, T, H, W)

        return frames_tensor

    def __len__(self):
        """Return number of clips in dataset."""
        return len(self.video_list)

    def __getitem__(self, idx):
        """Get a clip from the dataset."""
        video_path = self.video_list[idx]
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Extract clips (or load from cache)
        clips = self._extract_clips_from_video(video_path)

        if len(clips) == 0:
            # Return next valid video
            return self.__getitem__((idx + 1) % len(self))

        # Select random clip (or deterministic during validation)
        clip_idx = random.randint(0, len(clips) - 1) if self.is_train else 0
        clip = clips[clip_idx]

        # Preprocess frames
        frames = self._preprocess_frames(clip['frames'])

        # Get score
        score = self.annotations[video_id]['score']

        # Load masks
        masks = self._load_masks(video_id)
        if masks is not None:
            # Resize masks to match spatial size
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1),
                size=self.spatial_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            # FIX: Slice masks according to current clip's start/end frames
            start_idx = clip['start_frame']
            end_idx = clip['end_frame']

            # Ensure indices are valid
            max_len = masks.size(0)
            valid_start = min(start_idx, max_len - 1)
            valid_end = min(end_idx, max_len)

            # Extract corresponding mask segment
            clip_masks = masks[valid_start:valid_end]

            # Pad if extracted mask is shorter than clip length
            if clip_masks.size(0) < frames.size(1):
                pad_size = frames.size(1) - clip_masks.size(0)
                clip_masks = torch.nn.functional.pad(
                    clip_masks.unsqueeze(0).unsqueeze(0),
                    (0, 0, 0, 0, 0, pad_size),
                    mode='constant',
                    value=0
                ).squeeze(0).squeeze(0)
            # Truncate if longer
            elif clip_masks.size(0) > frames.size(1):
                clip_masks = clip_masks[:frames.size(1)]

            masks = clip_masks

        # Return sample
        sample = {
            'video_id': video_id,
            'frames': frames,  # (C, T, H, W)
            'masks': masks,  # (T, H, W) or None
            'score': torch.tensor(score, dtype=torch.float32),
            'clip_info': clip
        }

        # Apply additional transforms
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class SurgicalQADataLoader:
    """
    Data loader wrapper for surgical QA dataset.

    Creates train/val/test splits and handles batching.
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
            data_root: Root directory of dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes
            train_split: Ratio for training set
            val_split: Ratio for validation set
            dataset_kwargs: Additional kwargs for SurgicalVideoDataset
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset_kwargs is None:
            dataset_kwargs = {}

        # Create full dataset
        full_dataset = SurgicalVideoDataset(data_root, **dataset_kwargs)

        # Split into train/val/test
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size

        # Create indices for splits
        indices = list(range(total_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create datasets with specific indices
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        print(f"\nDataset splits:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

        # Create dataloaders
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
        """Get dataloader for specific split."""
        if split == 'train':
            return self.train_loader
        elif split == 'val':
            return self.val_loader
        elif split == 'test':
            return self.test_loader
        else:
            raise ValueError(f"Unknown split: {split}")


def create_sample_annotations(output_dir, num_samples=100):
    """
    Create sample annotation file for testing.

    Args:
        output_dir: Directory to save annotations
        num_samples: Number of videos to generate
    """
    annotations = {}

    for i in range(num_samples):
        video_id = f"video_{i:03d}"
        score = np.random.uniform(5.0, 10.0)  # Random score [5, 10]
        duration = np.random.randint(60, 300)  # Random duration [60, 300] seconds

        annotations[video_id] = {
            'score': round(score, 2),
            'duration': duration,
            'num_frames': int(duration * 30)  # Assume 30 FPS
        }

    output_path = os.path.join(output_dir, 'annotations.json')
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Created sample annotations at {output_path}")
    return annotations


if __name__ == '__main__':
    # Test data loader
    import tempfile

    # Create dummy dataset
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, 'masks'), exist_ok=True)

    # Create sample annotations
    create_sample_annotations(tmp_dir, num_samples=5)

    # Create dummy video and mask files
    for i in range(5):
        video_id = f"video_{i:03d}"
        # Create dummy video
        video_path = os.path.join(tmp_dir, 'videos', f'{video_id}.mp4')
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                        30.0, (640, 480))
        for frame in range(10):
            writer.write(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8))
        writer.release()

        # Create dummy mask directory
        mask_dir = os.path.join(tmp_dir, 'masks', video_id)
        os.makedirs(mask_dir, exist_ok=True)
        for frame in range(10):
            mask = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
            cv2.imwrite(os.path.join(mask_dir, f'frame_{frame:04d}_mask.png'), mask)

    # Test dataset
    dataset = SurgicalVideoDataset(tmp_dir, clip_length=8, spatial_size=224)
    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Frames shape: {sample['frames'].shape}")
        if sample['masks'] is not None:
            print(f"Masks shape: {sample['masks'].shape}")
        print(f"Score: {sample['score']}")
