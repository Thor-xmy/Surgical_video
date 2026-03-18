"""
Surgical QA Model with Multi-Clip Support and Bounded Regression

完整实现多clip特征提取和有界回归：
1. Static features: Extracted at clip level and concatenated
2. Dynamic features: Extracted from clips and concatenated
3. Per-clip fusion: [Static_clip, Dynamic_clip] -> Fused_clip
4. Temporal concatenation: [Fused_clip_1, ..., Fused_clip_N] -> Final features
5. Bounded regression: Sigmoid ensures output in [0, 1]

Key aspects:
- 使用方案A（按clip先融合动静特征）
- 使用BoundedFusionRegressor（Sigmoid激活）
- 支持视频级别的标签归一化
- 输出单个视频级别的总分

Pipeline:
    Video (B, 3, T, H, W)
      ├─> Static: split into N clips -> sample keyframes -> ResNet -> (B, N, 512)
      └─> Dynamic: split into N clips -> I3D -> (B, N, 1024)
            ↓ Per-clip fusion
      Concat: [(512+1024), ..., (512+1024)] -> (B, N*1536)
            ↓ Regressor
      Score: (B, 1) in [0, 1] (normalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .static_feature_extractor_multiclip import StaticFeatureMultiClip
from .dynamic_feature_extractor_multiclip import DynamicFeatureMultiClip
from .fusion_regressor_multiclip_bounded import BoundedFusionRegressorMultiClip


class SurgicalQAModelMultiClipBounded(nn.Module):
    """
    Surgical QA Model with Multi-Clip Support and Bounded Regression.

    完整实现：
    1. Video-level 输入
    2. Clip-level static and dynamic feature extraction
    3. Per-clip feature fusion
    4. Temporal sequence concatenation
    5. Bounded regression with Sigmoid
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dict with:
                - static_dim: Static feature dimension per clip (default: 512)
                - dynamic_dim: Dynamic feature dimension per clip (default: 1024)
                - clip_length: Length of each clip (default: 16)
                - clip_stride: Stride between clips (default: 10)
                - max_clips: Maximum number of clips (default: None)
                - resnet_path: Path to ResNet checkpoint
                - i3d_path: Path to I3D checkpoint
                - use_pretrained: Use pretrained weights
                - freeze_backbone: Freeze backbone weights
                - score_min: Original score minimum (default: 6.0)
                - score_max: Original score maximum (default: 30.0)
                - keyframe_strategy: Static feature keyframe sampling
                - regressor_hidden_dims: Hidden dimensions for regressor
        """
        super().__init__()

        self.config = config
        self.static_dim = config.get('static_dim', 512)
        self.dynamic_dim = config.get('dynamic_dim', 1024)

        # Multi-clip parameters
        self.clip_length = config.get('clip_length', 16)
        self.clip_stride = config.get('clip_stride', 10)
        self.max_clips = config.get('max_clips', None)

        # Score range parameters
        self.score_min = config.get('score_min', 6.0)
        self.score_max = config.get('score_max', 30.0)
        self.score_range = self.score_max - self.score_min

        # Static feature keyframe strategy
        self.keyframe_strategy = config.get('keyframe_strategy', 'middle')

        print("\n" + "="*70)
        print("Initializing Surgical QA Model (Multi-Clip + Bounded Version)")
        print("="*70)
        print(f"  Static dim per clip: {self.static_dim}")
        print(f"  Dynamic dim per clip: {self.dynamic_dim}")
        print(f"  Clip length: {self.clip_length}")
        print(f"  Clip stride: {self.clip_stride}")
        print(f"  Max clips: {self.max_clips if self.max_clips else 'auto'}")
        print(f"  Score range: [{self.score_min}, {self.score_max}]")
        print(f"  Keyframe strategy: {self.keyframe_strategy}")
        print("="*70 + "\n")

        # 1. Static Feature Extractor with Multi-Clip Support
        print("Initializing Static Feature Extractor (ResNet-34) with Multi-Clip...")
        self.static_extractor = StaticFeatureMultiClip(
            resnet_path=config.get('resnet_path', None),
            use_pretrained=config.get('use_pretrained', True),
            freeze_early_layers=config.get('freeze_backbone', True),
            output_dim=self.static_dim,
            keyframe_strategy=self.keyframe_strategy
        )

        # 2. Dynamic Feature Extractor with Multi-Clip Support
        print("Initializing Dynamic Feature Extractor (I3D) with Multi-Clip...")
        self.dynamic_extractor = DynamicFeatureMultiClip(
            i3d_path=config.get('i3d_path', None),
            use_pretrained_i3d=config.get('use_pretrained_i3d', False),
            output_dim=self.dynamic_dim,
            freeze_backbone=config.get('freeze_backbone', True),
            use_mixed_conv=config.get('use_mixed_conv', True),
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            max_clips=self.max_clips
        )

        # 3. Fusion and Regression Module
        # Will be initialized in first forward pass when we know num_clips
        self.fusion_regressor = None
        self.total_clip_dim = None  # static_dim + dynamic_dim per clip

        print("Fusion Regressor will be initialized in first forward pass...")
        print("Model initialized successfully!")

    def _per_clip_fusion(self, static_per_clip, dynamic_per_clip):
        """
        Fuse static and dynamic features per clip.

        方案A: 先按clip融合动静特征

        Args:
            static: (B, num_clips, static_dim)
            dynamic: (B, num_clips, dynamic_dim)

        Returns:
            fused_per_clip: (B, num_clips, static_dim + dynamic_dim)
        """
        # Concatenate along feature dimension: (B, num_clips, static_dim + dynamic_dim)
        fused = torch.cat([static_per_clip, dynamic_per_clip], dim=-1)
        return fused

    def forward(self, video, return_features=False):
        """
        Forward pass for surgical video quality assessment.

        Args:
            video: (B, C, T, H, W) - Entire video
            return_features: Return intermediate features for analysis

        Returns:
            score_normalized: (B, 1) - Predicted quality score in [0, 1]
            If return_features: also return features_dict
        """
        B, C, T, H, W = video.shape

        # 1. Extract multi-clip static features
        # 每个clip采样一个关键帧，提取ResNet特征
        static_per_clip, num_clips_static = self.static_extractor.extract_multiclip_features(
            video,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            max_clips=self.max_clips
        )
        # static_per_clip: (B, num_clips, 512)

        # 2. Extract multi-clip dynamic features
        # 每个clip提取I3D特征
        dynamic_per_clip, num_clips_dynamic = self.dynamic_extractor.extract_multiclip_features(video)
        # dynamic_per_clip: (B, num_clips, 1024)

        # 确保clips数量一致
        num_clips = min(num_clips_static, num_clips_dynamic)
        if num_clips_static != num_clips_dynamic:
            print(f"Warning: Static clips ({num_clips_static}) != Dynamic clips ({num_clips_dynamic}), using {num_clips}")

        if num_clips_static > num_clips:
            static_per_clip = static_per_clip[:, :num_clips, :]
        elif num_clips_dynamic > num_clips:
            dynamic_per_clip = dynamic_per_clip[:, :num_clips, :]

        # 3. Per-clip fusion: [Static_clip, Dynamic_clip] -> Fused_clip
        fused_per_clip = self._per_clip_fusion(static_per_clip, dynamic_per_clip)
        # fused_per_clip: (B, num_clips, 512 + 1024) = (B, num_clips, 1536)

        # 4. Temporal concatenation: Flatten to preserve temporal order
        # (B, num_clips, 1536) -> (B, num_clips * 1536)
        temporal_features = fused_per_clip.reshape(B, -1)

        # 5. Initialize fusion regressor if not done
        if self.fusion_regressor is None:
            self.total_clip_dim = self.static_dim + self.dynamic_dim
            total_input_dim = num_clips * self.total_clip_dim

            print(f"\nInitializing Fusion Regressor:")
            print(f"  Per-clip dimension: {self.total_clip_dim} (static {self.static_dim} + dynamic {self.dynamic_dim})")
            print(f"  Number of clips: {num_clips}")
            print(f"  Total input dimension: {total_input_dim}")

            self.fusion_regressor = BoundedFusionRegressorMultiClip(
                input_dim=total_input_dim,  # Total temporal features
                hidden_dims=self.config.get('regressor_hidden_dims', [1024, 512, 256, 128])
            )
            # Move to same device as video
            self.fusion_regressor = self.fusion_regressor.to(video.device)

        # 6. Regress to score (using fused temporal features directly)
        score_normalized = self.fusion_regressor(temporal_features)
        # score_normalized: (B, 1) in [0, 1]

        # Return based on flags
        if return_features:
            features_dict = {
                'static_per_clip': static_per_clip,
                'dynamic_per_clip': dynamic_per_clip,
                'fused_per_clip': fused_per_clip,
                'temporal_features': temporal_features,
                'num_clips': num_clips,
                'clip_length': self.clip_length,
                'clip_stride': self.clip_stride,
                'per_clip_dim': self.total_clip_dim,
                'total_dim': temporal_features.shape[1]
            }
            return score_normalized, features_dict
        else:
            return score_normalized

    def denormalize_score(self, score_normalized, target_min=None, target_max=None,
                        norm_min=None, norm_max=None):
        """
        反归一化：将归一化的分数从 [norm_min, norm_max] 转换回目标范围 [target_min, target_max]

        标准使用场景（JIGSAWS数据集）：
        1. 训练时原始分数 [6, 30] → 归一化到 [0, 1]
        2. 网络输出 [0, 1] → 反归一化回 [6, 30]（JIGSAWS原始范围，默认）
        3. 网络输出 [0, 1] → 反归一化到 [1, 10]（论文分数，用于对比）

        数学公式：
            target = (normalized - norm_min) / (norm_max - norm_min) * (target_max - target_min) + target_min

        Args:
            score_normalized: (B,) or (B, 1) - 归一化后的预测分数 [0, 1]
            target_min: 目标范围的最小值（如 6.0 或 1.0），默认使用 self.score_min (6.0)
            target_max: 目标范围的最大值（如 30.0 或 10.0），默认使用 self.score_max (30.0)
            norm_min: 归一化范围的下限（通常是 0.0），默认 0.0
            norm_max: 归一化范围的上限（通常是 1.0），默认 1.0

        Returns:
            score_target: 反归一化后的分数，范围 [target_min, target_max]

        Examples:
            # 将 [0,1] 映射回 JIGSAWS 原始范围 [6,30]（默认，不传参数）
            score_6to30 = model.denormalize_score(score_norm)

            # 或者显式指定
            score_6to30 = model.denormalize_score(score_norm, target_min=6.0, target_max=30.0)

            # 将 [0,1] 映射到论文范围 [1,10]（用于与其他论文对比）
            score_1to10 = model.denormalize_score(score_norm, target_min=1.0, target_max=10.0)
        """
        # 处理输入维度
        if score_normalized.dim() == 2:
            score_normalized = score_normalized.squeeze(-1)
        elif score_normalized.dim() == 0:
            score_normalized = score_normalized.unsqueeze(0)

        # 使用传入参数或默认值
        norm_min = norm_min if norm_min is not None else 0.0
        norm_max = norm_max if norm_max is not None else 1.0
        target_min = target_min if target_min is not None else self.score_min
        target_max = target_max if target_max is not None else self.score_max

        # 计算范围
        norm_range = norm_max - norm_min
        target_range = target_max - target_min

        # 反归一化公式
        if norm_range == 0:
            # 防止除零
            score_target = torch.full_like(score_normalized, target_min)
        else:
            score_target = (score_normalized - norm_min) / norm_range * target_range + target_min

        return score_target

    def compute_loss(self, score_pred, score_gt):
        """
        Compute total loss for training.

        注意：由于使用了 Sigmoid 和归一化标签，score_pred 和 score_gt 都在 [0, 1] 范围

        Args:
            score_pred: (B, 1) - Predicted scores [0, 1]
            score_gt: (B,) or (B, 1) - Ground truth scores [0, 1]

        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        # Score regression loss (MSE)
        if score_gt.dim() == 2:
            score_gt = score_gt.squeeze(-1)
        score_loss = F.mse_loss(score_pred.squeeze(-1), score_gt)

        loss_dict = {
            'score_loss': score_loss.item(),
            'total_loss': score_loss.item()
        }

        return score_loss, loss_dict

    def unfreeze_backbone(self, layers_to_unfreeze=['all']):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            layers_to_unfreeze: List of layers to unfreeze
                              ['all'] -> unfreeze all
                              ['layer4'] -> unfreeze only layer4
        """
        if 'all' in layers_to_unfreeze:
            for param in self.static_extractor.backbone.parameters():
                param.requires_grad = True
            for param in self.dynamic_extractor.feature_extractor.parameters():
                param.requires_grad = True
        else:
            for layer_name in layers_to_unfreeze:
                for name, param in self.static_extractor.backbone.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True
                for name, param in self.dynamic_extractor.feature_extractor.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True

        print(f"Unfroze layers: {layers_to_unfreeze}")

    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return total, trainable


def build_model_multiclip_bounded(config):
    """
    Factory function to build Multi-Clip Bounded Surgical QA Model.

    Args:
        config: Configuration dict or path to config file

    Returns:
        model: SurgicalQAModelMultiClipBounded instance
    """
    if isinstance(config, str):
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    model = SurgicalQAModelMultiClipBounded(config)

    # Load checkpoint if provided
    if 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {config['checkpoint_path']}")

    return model


if __name__ == '__main__':
    # Test model
    print("Testing SurgicalQAModelMultiClipBounded...")

    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'clip_length': 16,
        'clip_stride': 10,
        'max_clips': None,
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mixed_conv': True,
        'score_min': 6.0,
        'score_max': 30.0,
        'keyframe_strategy': 'middle',
        'regressor_hidden_dims': [1024, 512, 256, 128]
    }

    model = SurgicalQAModelMultiClipBounded(config)
    model.count_parameters()

    # Test with different video lengths
    test_cases = [
        (2, 3, 100, 224, 224),  # 100 frames -> 10 clips
        (2, 3, 90, 224, 224),    # 90 frames -> 9 clips
        (2, 3, 50, 224, 224),    # 50 frames -> 5 clips
        (2, 3, 16, 224, 224),    # 16 frames -> 1 clip
    ]

    for video_shape in test_cases:
        video = torch.randn(*video_shape)
        score = model(video)

        print(f"\nInput: {video_shape}")
        print(f"  Output score shape: {score.shape}")
        print(f"  Output score range: [{score.min().item():.4f}, {score.max().item():.4f}]")

        # Test with feature return
        score, features = model(video, return_features=True)
        print(f"  Features dict keys: {features.keys()}")
        print(f"  Static per-clip: {features['static_per_clip'].shape}")
        print(f"  Dynamic per-clip: {features['dynamic_per_clip'].shape}")
        print(f"  Fused per-clip: {features['fused_per_clip'].shape}")
        print(f"  Temporal features: {features['temporal_features'].shape}")
        print(f"  Number of clips: {features['num_clips']}")

        # Verify dimensions
        num_clips = features['num_clips']
        expected_total_dim = num_clips * (512 + 1024)

        assert features['static_per_clip'].shape == (video_shape[0], num_clips, 512)
        assert features['dynamic_per_clip'].shape == (video_shape[0], num_clips, 1024)
        assert features['fused_per_clip'].shape == (video_shape[0], num_clips, 1536)
        assert features['temporal_features'].shape == (video_shape[0], expected_total_dim)

    print("\n✓ All tests passed!")
