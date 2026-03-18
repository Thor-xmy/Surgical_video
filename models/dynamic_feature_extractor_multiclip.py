"""
Dynamic Feature Extractor with Multi-Clip Support

Modified to extract features from multiple video clips and preserve temporal order.

Key changes from original:
1. New extract_multiclip_features() method
2. Returns flattened features that preserve temporal sequence
3. Supports configurable clip length and stride

Usage:
    - Original: (B, 3, 16, H, W) -> (B, 1024)
    - Multi-clip: (B, 3, T_total, H, W) -> (B, 1024 * num_clips)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .dynamic_feature_extractor import (
    InceptionI3D, MixedConv3D, MaxPool3dSamePadding
)


class DynamicFeatureMultiClip(nn.Module):
    """
    Dynamic Feature Extractor with Multi-Clip Support.

    Extracts features from multiple overlapping clips and concatenates them
    to preserve temporal order (Flattening approach).

    Example:
        - Video: 100 frames
        - Clip length: 16 frames
        - Stride: 10 frames (overlap of 6 frames)
        - Number of clips: floor((100-16)/10) + 1 = 10 clips
        - Output: (B, 1024 * 10) = (B, 10240)
    """
    def __init__(self,
                 i3d_path=None,
                 use_pretrained_i3d=False,
                 output_dim=1024,
                 freeze_backbone=True,
                 use_mixed_conv=True,
                 clip_length=16,
                 clip_stride=10,
                 max_clips=None):
        """
        Args:
            i3d_path: Path to I3D checkpoint
            use_pretrained_i3d: Use pretrained I3D weights
            output_dim: Output feature dimension per clip (1024 for standard I3D)
            freeze_backbone: Freeze I3D backbone weights
            use_mixed_conv: Apply mixed convolution
            clip_length: Length of each clip (default: 16)
            clip_stride: Stride between clips (default: 10, overlap=6)
            max_clips: Maximum number of clips to process (None = use all)
        """
        super().__init__()

        self.output_dim = output_dim
        self.use_mixed_conv = use_mixed_conv
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.max_clips = max_clips

        # Standard I3D backbone
        self.i3d = InceptionI3D(in_channels=3, num_classes=400)

        # Feature extractor: all layers up to Mixed_5c
        self.feature_extractor = nn.Sequential(
            self.i3d.Conv3d_1a_7x7, self.i3d.MaxPool3d_2a_3x3,
            self.i3d.Conv3d_2b_1x1, self.i3d.Conv3d_2c_3x3, self.i3d.MaxPool3d_3a_3x3,
            self.i3d.Mixed_3b, self.i3d.Mixed_3c, self.i3d.MaxPool3d_4a_3x3,
            self.i3d.Mixed_4b, self.i3d.Mixed_4c, self.i3d.Mixed_4d,
            self.i3d.Mixed_4e, self.i3d.Mixed_4f, self.i3d.MaxPool3d_5a_2x2,
            self.i3d.Mixed_5b, self.i3d.Mixed_5c
        )

        # Mixed convolution layer
        if use_mixed_conv:
            self.mixed_conv = MixedConv3D(
                in_channels=output_dim,
                out_channels=output_dim,
                temporal_kernel=3
            )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Load checkpoint if provided
        if i3d_path is not None:
            self.load_checkpoint(i3d_path)

    def load_checkpoint(self, checkpoint_path):
        """Load I3D checkpoint."""
        print("Loading I3D checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = self.i3d.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print("  Missing keys (expected): {}".format(len(missing_keys)))
        if unexpected_keys:
            print("  Unexpected keys (ignored): {}".format(len(unexpected_keys)))

        print("  ✓ Checkpoint loaded successfully")

    def _split_into_clips(self, video):
        """
        Split video into overlapping clips.

        Args:
            video: (B, C, T, H, W)

        Returns:
            clips: list of (B, C, clip_length, H, W)
            start_indices: list of starting frame indices
        """
        B, C, T, H, W = video.shape

        # Calculate number of clips
        if T <= self.clip_length:
            # Video is shorter than clip length, pad or use as is
            clips = [video]
            start_indices = [0]
        else:
            # Calculate number of clips with stride
            num_clips = (T - self.clip_length) // self.clip_stride + 1

            if self.max_clips is not None and num_clips > self.max_clips:
                num_clips = self.max_clips

            start_indices = list(range(0, (num_clips - 1) * self.clip_stride + 1, self.clip_stride))

            # Extract clips
            clips = []
            for start_idx in start_indices:
                end_idx = start_idx + self.clip_length
                if end_idx > T:
                    # If clip extends beyond video length, pad with zeros
                    clip = video[:, :, start_idx:T, :, :]
                    pad_len = self.clip_length - (T - start_idx)
                    clip = F.pad(clip, (0, 0, 0, 0, 0, pad_len), mode='constant', value=0)
                else:
                    clip = video[:, :, start_idx:end_idx, :, :]
                clips.append(clip)

        return clips, start_indices

    def extract_multiclip_features(self, video):
        """
        Extract features from multiple clips.

        Args:
            video: (B, C, T, H, W) - Input video

        Returns:
            features_per_clip: (B, num_clips, output_dim) - Per-clip features
            num_clips: Number of clips extracted
        """
        B, C, T, H, W = video.shape

        # Split video into clips
        clips, start_indices = self._split_into_clips(video)
        num_clips = len(clips)

        if num_clips == 0:
            raise ValueError(f"Video has only {T} frames, which is too short for clip_length={self.clip_length}")

        # Stack all clips: (B*num_clips, C, clip_length, H, W)
        # Approach 1: Process clips one by one (simpler, less memory)
        features_list = []

        for clip in clips:
            # Extract features for this clip
            feat = self.feature_extractor(clip)  # (B, 1024, T', H', W')

            # Apply mixed convolution if enabled
            if self.use_mixed_conv:
                feat = self.mixed_conv(feat)  # (B, 1024, T', H', W')

            # Pool to get single feature vector per clip
            feat = F.adaptive_avg_pool3d(feat, (1, 1, 1))  # (B, 1024, 1, 1, 1)
            feat = feat.flatten(1)  # (B, 1024)

            features_list.append(feat)

        # Stack features: (num_clips, B, 1024)
        features_stack = torch.stack(features_list, dim=0)  # (num_clips, B, 1024)

        # Permute to get (B, num_clips, 1024)
        features_per_clip = features_stack.permute(1, 0, 2)  # (B, num_clips, 1024)

        # Return in order: (per_clip_features, num_clips)
        # This matches static extractor interface
        return features_per_clip, num_clips

    def forward(self, video, return_features_map=False):
        """
        Extract dynamic features from video (backward compatibility).

        This method keeps the original interface for single-clip processing.

        Args:
            video: (B, C, T, H, W)
            return_features_map: Return intermediate feature map

        Returns:
            If return_features_map:
                features_map: (B, 1024, T', H', W')
                pooled_features: (B, output_dim)
            Else:
                features: (B, output_dim)
        """
        features = self.feature_extractor(video)  # (B, 1024, 2, 4, 4)

        if self.use_mixed_conv:
            features = self.mixed_conv(features)  # (B,-1024, 2, 4, 4)

        features_map = features  # (B, 1024, T', H', W')

        features = F.max_pool3d(features, kernel_size=(1, features.size(3), features.size(4)),
                               stride=(1, 1, 1))  # (B, 1024, T', 1, 1)

        features = F.adaptive_avg_pool3d(features, (1, 1, 1))  # (B, 1024, 1, 1, 1)
        features = features.flatten(1)  # (B, 1024)

        if return_features_map:
            return features_map, features
        else:
            return features


if __name__ == '__main__':
    # Test multi-clip extraction
    print("Testing DynamicFeatureMultiClip...")

    config = {
        'clip_length': 16,
        'clip_stride': 10,
        'max_clips': None  # Use all clips
    }

    model = DynamicFeatureMultiClip(
        output_dim=1024,
        clip_length=config['clip_length'],
        clip_stride=config['clip_stride'],
        max_clips=config['max_clips']
    )

    # Test with different video lengths
    test_cases = [
        (2, 3, 100, 112, 112),  # 100 frames -> 10 clips
        (2, 3, 90, 112, 112),    # 90 frames -> 9 clips
        (2, 3, 50, 112, 112),    # 50 frames -> 5 clips
        (2, 3, 16, 112, 112),    # 16 frames -> 1 clip
    ]

    for video_shape in test_cases:
        video = torch.randn(*video_shape)
        flattened, per_clip, num_clips = model.extract_multiclip_features(video)

        expected_dim = 1024 * num_clips

        print(f"\nInput: {video_shape}")
        print(f"  Number of clips: {num_clips}")
        print(f"  Flattened features shape: {flattened.shape}")
        print(f"  Per-clip features shape: {per_clip.shape}")
        print(f"  Expected flattened dim: {expected_dim}")

        assert flattened.shape == (video_shape[0], expected_dim)
        assert per_clip.shape == (video_shape[0], num_clips, 1024)

    print("\n✓ All tests passed!")
