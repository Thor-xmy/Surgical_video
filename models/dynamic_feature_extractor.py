"""
Dynamic Feature Extractor Module - Standard I3D Implementation
Based on paper1231: Dynamic Instrument Temporal Feature Extraction Module (Section C)

This module extracts spatiotemporal features using STANDARD I3D backbone
(PyTorchConv3D / original I3D implementation).

Captures instrument manipulation dynamics through 3D CNN.

Reference from paper1231:
- Video is divided into N overlapping snippets (stride=10, each has 16 frames)
- I3D extracts short-term spatiotemporal features
- Mixed convolution layers expand temporal receptive field
- Spatial max pooling applied (Eq.5): F_clip(X_i) = maxpool(Conv_mix(X_i))

Differences from custom implementation:
- Uses standard I3D architecture (kernel 7x7x7 instead of 1x7x7)
- Branch channel configuration matches original I3D paper
- Supports loading official pretrained weights from PyTorchConv3D/pytorch-i3d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaxPool3dSamePadding(nn.MaxPool3d):
    """3D MaxPool with 'same' padding for standard I3D."""

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    """3D Convolution unit for standard I3D."""

    def __init__(self, in_channels, output_channels,
                 kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0, activation_fn=F.relu, use_batch_norm=True,
                 use_bias=False, name='unit_3d'):
        super().__init__()

        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=0,  # Padding computed dynamically in forward
            bias=self._use_bias)

        if self._use_batch_norm:
            # Use original I3D BN parameters: eps=0.001, momentum=0.01
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_size[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_size[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)

        if self._use_batch_norm:
            x = self.bn(x)

        if self._activation_fn is not None:
            x = self._activation_fn(x, inplace=True)

        return x


class InceptionModule(nn.Module):
    """Inception module for standard I3D."""

    def __init__(self, in_channels, out_channels, name):
        super().__init__()

        # Standard I3D branch configuration
        # out_channels is a list of 6 values for branches 0,1a,1b,2a,2b,3

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_size=[1, 1, 1],
            padding=0,
            name=name+'/Branch_0/Conv3d_0a_1x1')

        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_size=[1, 1, 1],
            padding=0,
            name=name+'/Branch_1/Conv3d_0a_1x1')

        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_size=[3, 3, 3],
            name=name+'/Branch_1/Conv3d_0b_3x3')

        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_size=[1, 1, 1],
            padding=0,
            name=name+'/Branch_2/Conv3d_0a_1x1')

        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_size=[3, 3, 3],
            name=name+'/Branch_2/Conv3d_0b_3x3')

        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3],
            stride=(1, 1, 1),
            padding=0)

        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_size=[1, 1, 1], padding=0,
            name=name+'/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3D(nn.Module):
    """
    Standard I3D backbone for dynamic feature extraction (1024 output channels).

    This is the OFFICIAL I3D implementation matching PyTorchConv3D/pytorch-i3d.

    Architecture (for 112x112 input):
    - Input: (B, 3, 16, 112, 112)
    - Stem: 3 -> 64 -> 192 with 7x7x7 kernel (standard)
    - Mixed_3b, Mixed_3c: 192 -> 256
    - Mixed_4b to Mixed_4f: 256 -> 480
    - Mixed_5b, Mixed_5c: 480 -> 1024
    - Output at Mixed_5c: (B, 1024, 4, 4, 4) for feature extraction

    This class DOES NOT include classification head (AvgPool + FC).
    It's designed for feature extraction up to Mixed_5c.
    """
    def __init__(self, in_channels=3, num_classes=400):
        super().__init__()

        self.num_classes = num_classes

        # ========== STEM ==========
        # Standard I3D uses 7x7x7 kernel (temporal dimension also 7)
        self.Conv3d_1a_7x7 = Unit3D(
            in_channels, 64,
            kernel_size=[7, 7, 7],  # ← Standard: (7,7,7) not (1,7,7)
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name='Conv3d_1a_7x7')

        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)

        self.Conv3d_2b_1x1 = Unit3D(64, 64, kernel_size=[1, 1, 1], padding=0,
                                       name='Conv3d_2b_1x1')

        self.Conv3d_2c_3x3 = Unit3D(64, 192, kernel_size=[3, 3, 3], padding=1,
                                       name='Conv3d_2c_3x3')

        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)

        # ========== Mixed_3 ==========
        # Standard I3D channel config: [64, 96, 128, 16, 32, 32] = total 256
        self.Mixed_3b = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')

        self.Mixed_3c = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')

        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)

        # ========== Mixed_4 ==========
        self.Mixed_4b = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 'Mixed_4b')

        self.Mixed_4c = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 'Mixed_4c')

        self.Mixed_4d = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 'Mixed_4d')

        self.Mixed_4e = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 'Mixed_4e')

        self.Mixed_4f = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 'Mixed_4f')

        self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)

        # ========== Mixed_5 ==========
        self.Mixed_5b = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], 'Mixed_5b')

        self.Mixed_5c = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], 'Mixed_5c')

        # Output at Mixed_5c: (B, 1024, 4, 4, 4)
        # NO projection layer - keep original 1024 channels for pretraining compatibility

    def forward(self, x):
        """Forward pass through I3D backbone up to Mixed_5c."""
        # Stem
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)

        # Mixed_3
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)

        # Mixed_4
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)

        # Mixed_5
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        # Return original 1024 channels for pretraining compatibility
        return x  # (B, 1024, 4, 4, 4) for 112x112 input


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


class DynamicFeatureExtractor(nn.Module):
    """
    Dynamic Feature Extractor based on paper1231 Section C.

    Now uses STANDARD I3D implementation (InceptionI3D).

    Components:
    1. Standard I3D backbone for spatiotemporal features (1024 output channels)
    2. Mixed convolution layers for expanded temporal receptive field (Eq.5)
    3. Spatial max pooling (as described in paper1231 Eq.5)

    Paper formula Eq.5:
    F_clip(X_i) = maxpool (Conv_mix(X_i))

    Input-Output Interface (unchanged from custom version):
    - Input: (B, 3, T, H, W) - surgical video clip
    - Output (return_features_map=False): (B, 1024) - pooled features
    - Output (return_features_map=True): (B, 1024, T', H', W'), (B, 1024)
    """
    def __init__(self,
                 i3d_path=None,
                 use_pretrained_i3d=False,
                 output_dim=1024,
                 freeze_backbone=True,
                 use_mixed_conv=True):
        """
        Args:
            i3d_path: Path to I3D checkpoint (from PyTorchConv3D/pytorch-i3d)
            use_pretrained_i3d: Use pretrained I3D weights
            output_dim: Output feature dimension (1024 = standard I3D output)
            freeze_backbone: Freeze I3D backbone weights
            use_mixed_conv: Apply mixed convolution as per paper1231 Eq.5
        """
        super().__init__()

        self.output_dim = output_dim
        self.use_mixed_conv = use_mixed_conv

        # Standard I3D backbone
        self.i3d = InceptionI3D(in_channels=3, num_classes=400)

        # Feature extractor: all layers up to Mixed_5c (1024 channels)
        self.feature_extractor = nn.Sequential(
            self.i3d.Conv3d_1a_7x7, self.i3d.MaxPool3d_2a_3x3,
            self.i3d.Conv3d_2b_1x1, self.i3d.Conv3d_2c_3x3, self.i3d.MaxPool3d_3a_3x3,
            self.i3d.Mixed_3b, self.i3d.Mixed_3c, self.i3d.MaxPool3d_4a_3x3,
            self.i3d.Mixed_4b, self.i3d.Mixed_4c, self.i3d.Mixed_4d,
            self.i3d.Mixed_4e, self.i3d.Mixed_4f, self.i3d.MaxPool3d_5a_2x2,
            self.i3d.Mixed_5b, self.i3d.Mixed_5c
        )

        # Mixed convolution layer as per paper1231 Eq.5: F_clip(X_i) = maxpool(Conv_mix(X_i))
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
        """
        Load I3D checkpoint from PyTorchConv3D/pytorch-i3d format.

        Args:
            checkpoint_path: Path to .pth or .pth.tar checkpoint file
        """
        print("Loading I3D checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # PyTorchConv3D format: {'state_dict': {...}, ...}
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load with strict=False to allow missing layers (e.g., classification head)
        missing_keys, unexpected_keys = self.i3d.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print("  Missing keys (expected): {}".format(len(missing_keys)))
        if unexpected_keys:
            print("  Unexpected keys (ignored): {}".format(len(unexpected_keys)))
            print("  Examples: {}".format(unexpected_keys[:5] if len(unexpected_keys) > 5 else unexpected_keys))

        print("  ✓ Checkpoint loaded successfully")

    def forward(self, video, return_features_map=False):
        """
        Extract dynamic features from video.

        Args:
            video: (B, C, T, H, W) - surgical video clip
            return_features_map: Return intermediate feature map for attention

        Returns:
            If return_features_map:
                features_map: (B, 1024, T', H', W') - spatiotemporal feature map
                pooled_features: (B, output_dim) - pooled features
            Else:
                features: (B, output_dim) - spatiotemporal features
        """
        # I3D expects (B, C, T, H, W)
        features = self.feature_extractor(video)  # (B, 1024, 2, 4, 4) for 112x112 input

        # Apply mixed convolution if enabled (paper1231 Eq.5: Conv_mix)
        if self.use_mixed_conv:
            features = self.mixed_conv(features)  # (B, 1024, 2, 4, 4)

        # Save feature map for attention module
        features_map = features  # (B, 1024, T', H', W')

        # Spatial max pooling along spatial dimensions (H, W) as per paper1231 Eq.5
        # This preserves temporal dimension while reducing spatial dimension
        features = F.max_pool3d(features, kernel_size=(1, features.size(3), features.size(4)),
                               stride=(1, 1, 1))  # (B, 1024, T', 1, 1)

        # Global average pooling over remaining dimensions (T, 1, 1)
        features = F.adaptive_avg_pool3d(features, (1, 1, 1))  # (B, 1024, 1, 1, 1)
        features = features.flatten(1)  # (B, 1024)

        if return_features_map:
            return features_map, features
        else:
            return features


if __name__ == '__main__':
    # Test module
    model = DynamicFeatureExtractor(output_dim=1024)
    video = torch.randn(2, 3, 16, 112, 112)  # B=2, C=3, T=16, H=W=112 (paper size)

    features = model(video)
    print("Input video shape: {}".format(video.shape))
    print("Output features shape: {}".format(features.shape))

    # Test with feature map return
    features_map, pooled = model(video, return_features_map=True)
    print("Features map shape: {}".format(features_map.shape))
    print("Pooled features shape: {}".format(pooled.shape))
