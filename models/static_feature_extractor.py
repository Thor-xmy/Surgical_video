"""
Static Feature Extractor Module
Based on paper1231: Intra-Video Context Modeling Module (Section D)

This module extracts static global features using ResNet-34 backbone.
Captures key tissue regions and static global features of surgical field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiScaleDownsample(nn.Module):
    """
    Multi-branch downsampling module from paper1231.

    Captures features at multiple scales (s=2, 4, 8) to:
    - Large scale: capture macroscopic contextual variations
    - Small scale: capture subtle tissue variations
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Three parallel downsampling branches
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.scale8 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=8, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            concat_feat: (B, 3*out_channels, 1, 1)
        """
        f2 = self.scale2(x)
        f4 = self.scale4(x)
        f8 = self.scale8(x)

        # Pool all features to 1x1 before concatenation
        # Different strides produce different spatial sizes (e.g., 4x4, 2x2, 1x1)
        # which cannot be concatenated directly
        f2 = F.adaptive_avg_pool2d(f2, (1, 1))
        f4 = F.adaptive_avg_pool2d(f4, (1, 1))
        f8 = F.adaptive_avg_pool2d(f8, (1, 1))

        # Concatenate features from all scales
        return torch.cat([f2, f4, f8], dim=1)


class StaticFeatureExtractor(nn.Module):
    """
    Static Feature Extractor based on paper1231 Section D.

    Components:
    1. ResNet-34 backbone for high-level global features
    2. Multi-scale downsampling module for local details
    3. Frame sampling strategy (keyframe or temporal average)

    Output: 512-dimensional static features (matching paper1231)
    """
    def __init__(self,
                 resnet_path=None,
                 use_pretrained=True,
                 freeze_early_layers=True,
                 output_dim=512,
                 sampling_strategy='middle'):
        """
        Args:
            resnet_path: Path to custom ResNet checkpoint
            use_pretrained: Use ImageNet pretrained weights
            freeze_early_layers: Freeze early ResNet layers
            output_dim: Output feature dimension (default 512 as per paper1231)
            sampling_strategy: How to sample frames ('middle', 'average', 'first', 'last')
        """
        super().__init__()

        self.output_dim = output_dim
        self.sampling_strategy = sampling_strategy

        # Load ResNet-34 backbone
        # Use 'weights' parameter for PyTorch 2.0+ compatibility
        if use_pretrained:
            from torchvision.models import ResNet34_Weights
            resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet34(weights=None)

        # Remove the final classification layers (avgpool + fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Output: (B, 512, H', W')

        # Freeze early layers if specified
        if freeze_early_layers:
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name:  # Only train layer4
                    param.requires_grad = False

        # Multi-scale downsampling module
        # Input: 512 channels (ResNet output), Output: 128*3 = 384 channels
        self.multiscale_downsample = MultiScaleDownsample(in_channels=512, out_channels=128)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final projection to output dimension
        # 384 (multiscale) -> output_dim (default 512)
        if output_dim == 384:
            # No projection needed if output matches multiscale output
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(384, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, output_dim)
            )

        # Load custom checkpoint if provided
        if resnet_path is not None:
            self.load_checkpoint(resnet_path)

    def load_checkpoint(self, checkpoint_path):
        """Load ResNet checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded ResNet checkpoint from {checkpoint_path}")

    def forward(self, video):
        """
        Extract static features from video.

        Args:
            video: (B, C, T, H, W) - surgical video clip
        Returns:
            features: (B, output_dim) - static global features
        """
        B, C, T, H, W = video.shape

        # Frame sampling strategy
        if self.sampling_strategy == 'middle':
            # Use middle frame only
            frame_idx = T // 2
            selected_frames = video[:, :, frame_idx:frame_idx+1, :, :]
        elif self.sampling_strategy == 'average':
            # Sample frames and average features
            num_samples = min(4, T)
            indices = torch.linspace(0, T-1, num_samples, dtype=torch.long)
            selected_frames = video[:, :, indices, :, :]
        elif self.sampling_strategy == 'first':
            selected_frames = video[:, :, 0:1, :, :]
        elif self.sampling_strategy == 'last':
            selected_frames = video[:, :, -1:, :, :]
        else:
            selected_frames = video[:, :, 0:1, :, :]

        # Process frames
        features_list = []
        for t in range(selected_frames.size(2)):
            selected_frame = selected_frames[:, :, t, :, :]  # (B, C, H, W)
            feat = self.backbone(selected_frame)  # (B, 512, H', W')
            feat = self.multiscale_downsample(feat)  # (B, 384, 1, 1)
            feat = feat.flatten(1)  # (B, 384)
            features_list.append(feat)

        # Combine sampled frame features
        if self.sampling_strategy == 'average':
            features = torch.stack(features_list, dim=0).mean(dim=0)  # (B, 384)
        else:
            features = features_list[0]  # (B, 384)

        # Project to output dimension
        features = self.proj(features)  # (B, output_dim)

        return features


if __name__ == '__main__':
    # Test the module
    model = StaticFeatureExtractor(output_dim=512)
    video = torch.randn(2, 3, 16, 112, 112)  # B=2, C=3, T=16, H=W=112 (paper requirement)

    features = model(video)
    print(f"Input video shape: {video.shape}")
    print(f"Output features shape: {features.shape}")
