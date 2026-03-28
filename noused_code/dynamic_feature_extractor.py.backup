"""
Dynamic Feature Extractor Module
Based on paper1231: Dynamic Instrument Temporal Feature Extraction Module (Section C)

This module extracts spatiotemporal features using I3D backbone.
Captures instrument manipulation dynamics through 3D CNN.

Reference from paper1231:
- Video is divided into N overlapping snippets (stride=10, each has 16 frames)
- I3D extracts short-term spatiotemporal features
- Mixed convolution layers expand temporal receptive field
- Spatial max pooling applied (Eq.5): F_clip(X_i) = maxpool(Conv_mix(X_i))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MixedConv3D(nn.Module):
    """
    Mixed convolution layer from paper1231.

    Expands temporal receptive field by combining 3D and separable convolutions.
    """
    def __init__(self, in_channels, out_channels, temporal_kernel=3):
        super().__init__()

        # Standard 3D convolution
        self.conv3d = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, 3, 3),
            padding=(temporal_kernel//2, 1, 1)
        )

        # Temporal convolution (for temporal dynamics)
        self.conv_temporal = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel//2, 0, 0)
        )

        # Spatial convolution
        self.conv_spatial = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        # Fuse mixed features
        self.fuse = nn.Conv3d(
            out_channels * 2, out_channels,
            kernel_size=1
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            out: (B, out_channels, T, H, W)
        """
        # Standard 3D conv path
        out1 = self.conv3d(x)

        # Mixed conv path
        temporal_out = self.conv_temporal(x)
        spatial_out = self.conv_spatial(temporal_out)
        mixed_out = torch.cat([temporal_out, spatial_out], dim=1)
        out2 = self.fuse(mixed_out)

        # Combine both paths
        out = (out1 + out2) / 2
        out = self.bn(out)
        out = self.relu(out)

        return out


class InceptionModule3D(nn.Module):
    """
    3D Inception module for I3D backbone (standard implementation).

    Standard I3D branch outputs:
    - b0: 1x1x1 conv (out0_channels)
    - b1: 1x1x1 -> 3x3x3 conv (out1_channels)
    - b2: 1x1x1 -> 3x3x3 conv (out2_channels)
    - b3: maxpool3d -> 1x1x1 conv (out3_channels)

    Total output channels = out0 + out1 + out2 + out3
    """
    def __init__(self, in_channels, out_channels, branch_channels=None):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Total output channels
            branch_channels: Optional list of [out0, out1, out2, out3]
        """
        super().__init__()

        if branch_channels is None:
            # Default branch splits for standard I3D
            # Mixed_3b/c: 192 → 256 (64+64+96+32)
            # Mixed_4b-f: 256 → 480 (128+160+96+96)
            # Mixed_5b/c: 576 → 832 (192+224+224+192)
            if out_channels == 256:
                branch_channels = [64, 64, 96, 32]
            elif out_channels == 480:
                branch_channels = [128, 160, 96, 96]
            elif out_channels == 576:
                branch_channels = [160, 192, 128, 96]
            elif out_channels == 832:
                branch_channels = [192, 224, 224, 192]
            else:
                # Even split
                c = out_channels // 4
                branch_channels = [c, c, c, c]

        self.out0, self.out1, self.out2, self.out3 = branch_channels

        # Branch 0: 1x1x1
        self.b0 = nn.Conv3d(in_channels, self.out0, kernel_size=1)

        # Branch 1: 1x1x1 -> 3x3x3
        self.b1a = nn.Conv3d(in_channels, self.out1 // 2, kernel_size=1)
        self.b1b = nn.Conv3d(self.out1 // 2, self.out1, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Branch 2: 1x1x1 -> 3x3x3 (larger output)
        self.b2a = nn.Conv3d(in_channels, self.out2 // 2, kernel_size=1)
        self.b2b = nn.Conv3d(self.out2 // 2, self.out2, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Branch 3: maxpool3d -> 1x1x1
        self.b3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.b3b = nn.Conv3d(in_channels, self.out3, kernel_size=1)

        # Batch normalization and ReLU
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            out: (B, out_channels, T, H, W)
        """
        # Branch 0
        b0_out = self.b0(x)

        # Branch 1
        b1a_out = self.b1a(x)
        b1_out = self.b1b(b1a_out)

        # Branch 2
        b2a_out = self.b2a(x)
        b2_out = self.b2b(b2a_out)

        # Branch 3
        b3_out = self.b3(x)
        b3_out = self.b3b(b3_out)

        # Concatenate all branches
        out = torch.cat([b0_out, b1_out, b2_out, b3_out], dim=1)

        out = self.bn(out)
        out = self.relu(out)

        return out


class I3DBackbone(nn.Module):
    """
    Standard I3D backbone for dynamic feature extraction (832 output channels).

    Based on InceptionI3D architecture with standard channel dimensions:
    - Stem: 3 → 64 → 192
    - Mixed_3b, Mixed_3c: 192 → 256
    - Mixed_4b to Mixed_4f: 256 → 480
    - Mixed_5b, Mixed_5c: 480 → 832
    """
    def __init__(self, in_channels=3, num_classes=400):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7),
                              stride=(1, 2, 2), padding=(0, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv2 = nn.Conv3d(64, 64, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(192)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Mixed_3b, Mixed_3c: 192 → 256
        self.mixed_3b = InceptionModule3D(192, 256)
        self.mixed_3c = InceptionModule3D(256, 256)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # Mixed_4b to Mixed_4f: 256 → 480
        self.mixed_4b = InceptionModule3D(256, 480)
        self.mixed_4c = InceptionModule3D(480, 480)
        self.mixed_4d = InceptionModule3D(480, 480)
        self.mixed_4e = InceptionModule3D(480, 480)
        self.mixed_4f = InceptionModule3D(480, 480)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        # Mixed_5b, Mixed_5c: 480 → 832
        self.mixed_5b = InceptionModule3D(480, 832)
        self.mixed_5c = InceptionModule3D(832, 832)

        # Classification head
        # Use adaptive pooling to handle variable input sizes
        # After Mixed_5c, output is (B, 832, T_out, H_out, W_out)
        # AdaptiveAvgPool3d(1,1,1) will pool all dimensions to size 1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(832, num_classes)

    def forward(self, x):
        """Forward pass through I3D network."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxpool3(x)

        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxpool4(x)

        x = self.mixed_5b(x)
        x = self.mixed_5c(x)

        features = self.avgpool(x)
        features = self.dropout(features)
        logits = self.fc(features)

        return features, logits


class DynamicFeatureExtractor(nn.Module):
    """
    Dynamic Feature Extractor based on paper1231 Section C.

    Components:
    1. I3D backbone for spatiotemporal features (832 output channels)
    2. Mixed convolution layers for expanded temporal receptive field (Eq.5)
    3. Spatial max pooling (as described in paper1231 Eq.5)

    Paper formula Eq.5:
    F_clip(X_i) = maxpool (Conv_mix(X_i))
    """
    def __init__(self,
                 i3d_path=None,
                 use_pretrained_i3d=False,
                 output_dim=832,
                 freeze_backbone=True,
                 use_mixed_conv=True):
        """
        Args:
            i3d_path: Path to I3D checkpoint
            use_pretrained_i3d: Use pretrained I3D weights
            output_dim: Output feature dimension (832 = standard I3D output channels)
            freeze_backbone: Freeze I3D backbone weights
            use_mixed_conv: Apply mixed convolution as per paper1231 Eq.5
        """
        super().__init__()

        self.output_dim = output_dim
        self.use_mixed_conv = use_mixed_conv

        # I3D backbone
        self.i3d = I3DBackbone(in_channels=3, num_classes=400)

        # Remove classification head for feature extraction
        # Keep features up to Mixed_5c (output: 832 channels)
        self.feature_extractor = nn.Sequential(
            self.i3d.conv1, self.i3d.bn1, self.i3d.relu, self.i3d.maxpool1,
            self.i3d.conv2, self.i3d.bn2, self.i3d.relu,
            self.i3d.conv3, self.i3d.bn3, self.i3d.relu, self.i3d.maxpool2,
            self.i3d.mixed_3b, self.i3d.mixed_3c, self.i3d.maxpool3,
            self.i3d.mixed_4b, self.i3d.mixed_4c, self.i3d.mixed_4d,
            self.i3d.mixed_4e, self.i3d.mixed_4f, self.i3d.maxpool4,
            self.i3d.mixed_5b, self.i3d.mixed_5c
        )

        # Mixed convolution layer as per paper1231 Eq.5: F_clip(X_i) = maxpool(Conv_mix(X_i))
        # This expands temporal receptive field before applying spatial max pooling
        if use_mixed_conv:
            self.mixed_conv = MixedConv3D(in_channels=output_dim, out_channels=output_dim, temporal_kernel=3)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Load checkpoint if provided
        if i3d_path is not None:
            self.load_checkpoint(i3d_path)

    def load_checkpoint(self, checkpoint_path):
        """Load I3D checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.i3d.load_state_dict(state_dict, strict=False)
        print(f"Loaded I3D checkpoint from {checkpoint_path}")

    def forward(self, video, return_features_map=False):
        """
        Extract dynamic features from video.

        Args:
            video: (B, C, T, H, W) - surgical video clip
            return_features_map: Return intermediate feature map for attention

        Returns:
            If return_features_map:
                features_map: (B, C, T, H, W) - spatiotemporal feature map
                pooled_features: (B, output_dim) - pooled features
            Else:
                features: (B, output_dim) - spatiotemporal features
        """
        # I3D expects (B, C, T, H, W)
        features = self.feature_extractor(video)  # (B, 832, T', H', W')

        # Apply mixed convolution if enabled (paper1231 Eq.5: Conv_mix)
        if self.use_mixed_conv:
            features = self.mixed_conv(features)  # (B, 832, T', H', W')

        # Save feature map for attention module
        features_map = features  # (B, 832, T', H', W')

        # Spatial max pooling along spatial dimensions (H, W) as per paper1231 Eq.5
        # This preserves temporal dimension while reducing spatial dimension
        features = F.max_pool3d(features, kernel_size=(1, features.size(3), features.size(4)),
                               stride=(1, 1, 1))  # (B, 832, T', 1, 1)

        # Global average pooling over remaining dimensions (T, 1, 1)
        features = F.adaptive_avg_pool3d(features, (1, 1, 1))  # (B, 832, 1, 1, 1)
        features = features.flatten(1)  # (B, 832)

        if return_features_map:
            return features_map, features
        else:
            return features


if __name__ == '__main__':
    # Test module
    model = DynamicFeatureExtractor(output_dim=832)
    video = torch.randn(2, 3, 16, 224, 224)  # B=2, C=3, T=16, H=W=224

    features = model(video)
    print(f"Input video shape: {video.shape}")
    print(f"Output features shape: {features.shape}")

    # Test with feature map return
    features_map, pooled = model(video, return_features_map=True)
    print(f"Features map shape: {features_map.shape}")
    print(f"Pooled features shape: {pooled.shape}")
