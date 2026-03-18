"""
Static Feature Extractor with Multi-Clip Support

Modified to extract static features from multiple video clips at clip level.

Key changes:
1. New extract_multiclip_features() method
2. For each clip, samples a keyframe and extracts ResNet features
3. Returns features per clip that can be concatenated with dynamic features

This follows the "方案A" (per-clip fusion) approach:
- For each clip: [Dynamic_clip, Static_clip] -> fused_clip
- Concatenate all fused clips -> temporal sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class StaticFeatureMultiClip(nn.Module):
    """
    Static Feature Extractor with Multi-Clip Support.

    Extracts static features from keyframes of video clips.

    For each clip:
    1. Sample a keyframe (e.g., middle frame)
    2. Extract ResNet-34 features
    3. Return per-clip features

    This allows per-clip fusion with dynamic features.
    """
    def __init__(self,
                 resnet_path=None,
                 use_pretrained=True,
                 freeze_early_layers=True,
                 output_dim=512,
                 keyframe_strategy='middle'):  # 'middle', 'first', 'last', 'random'
        """
        Args:
            resnet_path: Path to custom ResNet checkpoint
            use_pretrained: Use ImageNet pretrained weights
            freeze_early_layers: Freeze early ResNet layers
            output_dim: Output feature dimension (default 512)
            keyframe_strategy: How to select keyframe from each clip
        """
        super().__init__()

        self.output_dim = output_dim
        self.keyframe_strategy = keyframe_strategy

        # Load ResNet-34 backbone
        if use_pretrained:
            try:
                # PyTorch 2.0+ uses weights
                backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            except:
                # Fallback for older PyTorch
                backbone = models.resnet34(pretrained=True)
        else:
            backbone = models.resnet34(pretrained=False)

        # Get the feature extractor (excluding classification head)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Replace final AvgPool + FC with projection layer
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, output_dim)
        )

        # Freeze early layers if specified
        if freeze_early_layers:
            # Freeze conv1, layer1, layer2
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in ['conv1', 'layer1', 'layer2']):
                    param.requires_grad = False

        # Load custom checkpoint if provided

        if resnet_path is not None:
            self.load_checkpoint(resnet_path)

    def load_checkpoint(self, checkpoint_path):
        """Load ResNet checkpoint."""
        print(f"Loading ResNet checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load with strict=False to allow missing/extra keys
        self.backbone.load_state_dict(state_dict, strict=False)
        print("  ✓ ResNet checkpoint loaded")

    def _split_into_clips(self, video, clip_length=16, clip_stride=10):
        """
        Split video into overlapping clips.

        Args:
            video: (B, C, T, H, W)
            clip_length: Length of each clip
            clip_stride: Stride between clips

        Returns:
            clips: list of (B, C, clip_length, H, W)
            start_indices: list of starting frame indices
        """
        B, C, T, H, W = video.shape

        # Calculate number of clips
        if T <= clip_length:
            # Video is shorter than clip length, use as is
            clips = [video]
            start_indices = [0]
        else:
            # Calculate number of clips with stride
            num_clips = (T - clip_length) // clip_stride + 1
            start_indices = list(range(0, (num_clips - 1) * clip_stride + 1, clip_stride))

            # Extract clips
            clips = []
            for start_idx in start_indices:
                end_idx = start_idx + clip_length
                if end_idx > T:
                    # Pad if clip extends beyond video
                    clip = video[:, :, start_idx:T, :, :]
                    pad_len = clip_length - (T - start_idx)
                    clip = F.pad(clip, (0, 0, 0, 0, 0, pad_len), mode='constant', value=0)
                else:
                    clip = video[:, :, start_idx:end_idx, :, :]
                clips.append(clip)

        return clips, start_indices

    def _sample_keyframe(self, clip, clip_length=None, random_idx=None):
        """
        Sample a keyframe from a clip.

        Args:
            clip: (B, C, T, H, W)
            clip_length: Original clip length (for random sampling)
            random_idx: Pre-computed random index (for consistency)

        Returns:
            keyframe: (B, C, H, W)
        """
        B, C, T, H, W = clip.shape

        if self.keyframe_strategy == 'middle':
            # Sample middle frame
            keyframe_idx = T // 2
        elif self.keyframe_strategy == 'first':
            keyframe_idx = 0
        elif self.keyframe_strategy == 'last':
            keyframe_idx = T - 1
        elif self.keyframe_strategy == 'random':
            # Random frame (for augmentation)
            if clip_length is not None:
                keyframe_idx = random.randint(0, clip_length)
            elif random_idx is not None:
                keyframe_idx = random_idx
            else:
                keyframe_idx = random.randint(0, T)
        else:
            keyframe_idx = T // 2  # Default to middle

        return clip[:, :, keyframe_idx, :, :]  # (B, C, H, W)

    def extract_multiclip_features(self, video, clip_length=16, clip_stride=10, max_clips=None, return_num_clips=False):
        """
        Extract static features from multiple clips.

        For each clip:
        1. Sample a keyframe (e.g., middle frame)
        2. Extract ResNet-34 features
        3. Return per-clip static features

        Args:
            video: (B, C, T, H, W) - Input video
            clip_length: Length of each clip (default: 16)
            clip_stride: Stride between clips (default: 10)
            max_clips: Maximum number of clips (None = use all)

        Returns:
            per_clip_features: (B, num_clips, output_dim)
            num_clips: Number of clips extracted
        """
        B, C, T, H, W = video.shape

        # Split video into clips
        clips, start_indices = self._split_into_clips(video, clip_length, clip_stride)
        num_clips = len(clips)

        if max_clips is not None and num_clips > max_clips:
            clips = clips[:max_clips]
            start_indices = start_indices[:max_clips]
            num_clips = max_clips

        # Extract ResNet features for each clip
        features_list = []

        for clip in clips:
            # Sample keyframe from clip
            keyframe = self._sample_keyframe(clip)  # (B, C, H, W)

            # Extract ResNet features
            features = self.backbone(keyframe)  # (B, 512, H/32, W/32)
            features = self.projection(features)  # (B, 512)

            features_list.append(features)

        # Stack features: (num_clips, B, 512)
        features_stack = torch.stack(features_list, dim=0)

        # Permute to get (B, num_clips, 512)
        per_clip_features = features_stack.permute(1, 0, 2)

        return per_clip_features, num_clips

    def forward(self, video):
        """
        Extract static features from video (backward compatibility).

        This method samples a keyframe and extracts features.

        Args:
            video: (B, C, T, H, W)

        Returns:
            features: (B, output_dim)
        """
        # Sample middle frame
        B, C, T, H, W = video.shape
        keyframe = video[:, :, T // 2, :, :]  # (B, C, H, W)

        # Extract ResNet features
        features = self.backbone(keyframe)  # (B, 512, H/32, W/32)
        features = self.projection(features)  # (B, 512)

        return features


if __name__ == '__main__':
    # Test multi-clip static feature extraction
    print("Testing StaticFeatureMultiClip...")

    model = StaticFeatureMultiClip(
        output_dim=512,
        keyframe_strategy='middle'
    )

    # Test with different video lengths
    test_cases = [
        (2, 3, 100, 224, 224),  # 100 frames -> 10 clips
        (2, 3, 90, 224, 224),    # 90 frames -> 9 clips
        (2, 3, 50, 224, 224),    # 50 frames -> 5 clips
        (2, 3, 16, 224, 224),    # 16 frames -> 1 clip
    ]

    for video_shape in test_cases:
        video = torch.randn(*video_shape)
        per_clip_features, num_clips = model.extract_multiclip_features(video)

        print(f"\nInput: {video_shape}")
        print(f"  Number of clips: {num_clips}")
        print(f"  Per-clip features shape: {per_clip_features.shape}")

        assert per_clip_features.shape == (video_shape[0], num_clips, 512)

    print("\n✓ All tests passed!")
