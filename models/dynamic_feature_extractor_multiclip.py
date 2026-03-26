"""
Dynamic Feature Extractor with Multi-Clip Support and Mask-Guided Attention

Modified to extract features from multiple video clips and preserve temporal order.

Key changes from original:
1. New extract_multiclip_features() method with mask support
2. Returns flattened features that preserve temporal sequence
3. Supports configurable clip length and stride
4. Added mask-guided attention mechanism for each clip

Usage:
    - Original: (B, 3, 16, H, W) -> (B, 1024)
    - Multi-clip: (B, 3, T_total, H, W) -> (B, 1024 * num_clips)

Reference: paper1231 Eq.11: F_dy(X_i) = F_clip(X_i) ⊙ (A + I)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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
                 max_clips=None,
                 use_early_fusion=False):
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
        self.use_early_fusion = use_early_fusion  # 🌟 保存为类的属性
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

    def _split_masks_into_clips(self, masks, num_clips):
        """
        Split masks into clips matching video clips.

        Args:
            masks: (B, T, H, W) or (B, 1, T, H, W)
            num_clips: Number of clips to extract

        Returns:
            mask_clips: list of (B, clip_length, H, W)
        """
        # Handle both (B, T, H, W) and (B, 1, T, H, W) formats
        if masks.dim() == 5:
            masks = masks.squeeze(1)  # (B, 1, T, H, W) -> (B, T, H, W)

        B, T, H, W = masks.shape

        mask_clips = []
        for i in range(num_clips):
            start_idx = i * self.clip_stride
            end_idx = start_idx + self.clip_length

            if end_idx > T:
                # Pad if clip extends beyond mask length
                mask_clip = masks[:, start_idx:T, :, :]
                pad_len = self.clip_length - (T - start_idx)
                mask_clip = F.pad(mask_clip, (0, 0, 0, 0, 0, pad_len), mode='constant', value=0)
            else:
                mask_clip = masks[:, start_idx:end_idx, :, :]

            mask_clips.append(mask_clip)

        return mask_clips

    def _temporal_smoothing(self, masks, target_T=None):
        """
        Apply temporal smoothing to masks as in paper1231.

        Paper formula: M_gt[w,h] = f_2D(sum(M^{2t-1,w,h} + M^{2t,w,h}))

        This reduces jitter and local instability in single-frame masks.

        Args:
            masks: (B, T, H, W)
            target_T: Target temporal dimension for I3D alignment

        Returns:
            smoothed_masks: (B, T_out, H', W')
        """
        B, T, H, W = masks.shape

        if T < 2:
            return masks

        # Sum adjacent frames: M^{2t-1} + M^{2t}
        num_pairs = T // 2
        even_frames = masks[:, 0:num_pairs*2:2, :, :]  # M^0, M^2, ...
        odd_frames = masks[:, 1:num_pairs*2:2, :, :]   # M^1, M^3, ...

        # Sum adjacent pairs
        summed = even_frames + odd_frames  # (B, num_pairs, H, W)

        # Apply 2D average pooling (f_2D)
        smoothed = F.avg_pool2d(summed, kernel_size=2, stride=2, padding=0)  # (B, num_pairs, H/2, W/2)

        # Normalize to [0, 1]
        smoothed = smoothed / 2.0

        # If target temporal dimension is specified, interpolate to match I3D features
        if target_T is not None and smoothed.size(1) != target_T:
            # Permute to (B, H, W, T) for interpolation, then permute back
            smoothed = smoothed.permute(0, 2, 3, 1)  # (B, H, W, T/2)
            smoothed = F.interpolate(
                smoothed.unsqueeze(1),  # Add channel dim: (B, 1, H, W, T/2)
                size=(smoothed.size(1), smoothed.size(2), target_T),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)  # (B, H, W, T)
            smoothed = smoothed.permute(0, 3, 1, 2)  # (B, T, H, W)

        return smoothed

    def extract_multiclip_features(self, video, masks=None):
        """
        Extract features from multiple clips with optional mask-guided attention.

        Args:
            video: (B, C, T, H, W) - Input video
            masks: (B, T, H, W) or (B, 1, T, H, W) - Optional instrument masks
                   If None, no mask-guided attention is applied

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

        # Handle masks
        if masks is not None:
            # Split masks into clips matching video clips
            mask_clips = self._split_masks_into_clips(masks, num_clips)
        else:
            mask_clips = None

        # Process clips one by one
        features_list = []

        for idx, clip in enumerate(clips):
            '''
            # Extract I3D features: (B, 1024, T', H', W')
            feat = self.feature_extractor(clip)

            # Apply mixed convolution if enabled
            if self.use_mixed_conv:
                feat = self.mixed_conv(feat)  # (B, 1024, T', H', W')

            # Apply mask-guided attention if masks provided
            if mask_clips is not None:
                mask_clip = mask_clips[idx]  # (B, clip_length, H, W)

                # Step 1: Temporal smoothing
                #mask_smoothed = self._temporal_smoothing(mask_clip, target_T=feat.size(2))
                mask_smoothed = mask_clip
                # Step 2: Spatial alignment with I3D feature map
                # I3D features: (B, C, T', H', W')
                # Masks: (B, T_mask, H_mask, W_mask)

                _, C_feat, T_feat, H_feat, W_feat = feat.shape

                # Upsample/Downsample masks to match I3D feature resolution
                masks_5d = mask_smoothed.unsqueeze(1)  # (B, 1, T, H', W')
                masks_aligned = F.interpolate(
                    masks_5d,
                    size=(T_feat, H_feat, W_feat),
                    mode='trilinear',
                    align_corners=False
                )  # (B, 1, T', H', W')

                # Step 3: Apply mask attention: F_masked = F_i3d * (A + 1.0)
                # This is paper1231 Eq.11: F_dy(X_i) = F_clip(X_i) ⊙ (A + I)
                attention_map = masks_aligned.squeeze(1)  # (B, T', H', W')
                feat = feat * (attention_map.unsqueeze(1) + 1.0)  # (B, C, T', H', W')
            '''
            # ==========================================
            # 🌟 策略 A：早期融合 (Early Fusion)
            # 在进入网络前，直接在像素层面屏蔽背景
            # ==========================================
            if mask_clips is not None and self.use_early_fusion:
                
                
                print(f"[Debug] 乘法前 Clip 像素总和: {clip.abs().sum().item():.2f}")
                mask_clip = mask_clips[idx].unsqueeze(1)  # (B, 1, 16, H, W)
                #soft_mask = mask_clip * 0.8 + 0.2    #############################################################
                #clip = clip * mask_clip  # 背景像素归零
                clip = clip * mask_clip + (-1.0) * (1.0 - mask_clip)
                print(f"[Debug] 乘法后 Clip 像素总和: {clip.abs().sum().item():.2f}")
            # 无论哪种融合，都把 clip 送进 I3D 提取特征
            feat = self.feature_extractor(clip)

            if self.use_mixed_conv:
                feat = self.mixed_conv(feat) 

            # ==========================================
            # 🌟 策略 B：晚期融合 (Late Fusion)
            # 网络提取特征后，在特征图上做 Attention
            # ==========================================
            if mask_clips is not None and not self.use_early_fusion:
                mask_clip = mask_clips[idx]  
                mask_smoothed = mask_clip # 取消平滑操作

                _, C_feat, T_feat, H_feat, W_feat = feat.shape
                masks_5d = mask_smoothed.unsqueeze(1) 
                
                masks_aligned = F.interpolate(
                    masks_5d,
                    size=(T_feat, H_feat, W_feat),
                    mode='trilinear',
                    align_corners=False
                ) 
                attention_map = masks_aligned.squeeze(1) 
                debug_flag = (random.random() < 0.05)
                if debug_flag:
                    #print(f"\n👉 [晚期融合探头]")
                    #print(f"  原 Mask    : min={mask_clip.min().item():.2f}, max={mask_clip.max().item():.2f}")
                    #print(f"  Attention : min={attention_map.min().item():.4f}, max={attention_map.max().item():.4f}")
                    feat_sum_before = feat.abs().sum().item()
                feat = feat * (attention_map.unsqueeze(1) + 1.0) 

                if debug_flag:
                    feat_sum_after = feat.abs().sum().item()
                    ratio = feat_sum_after / (feat_sum_before + 1e-8)
                    print(f"  特征总和   : {feat_sum_before:.2f} 放大至 {feat_sum_after:.2f} (约 {ratio:.3f} 倍)")
            # ==========================================

            # 后续正常的池化操作保持不变
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
