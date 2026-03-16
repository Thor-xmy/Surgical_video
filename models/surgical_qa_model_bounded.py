"""
Modified Surgical QA Model with Bounded Score Regression

将原有的 SurgicalQAModel 修改为使用带 Sigmoid 的有界回归器
配合归一化标签使用

修改说明：
1. FusionRegressor → BoundedFusionRegressor（添加 Sigmoid）
2. 支持 score_min 和 score_max 参数
3. 提供反归一化函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入原始模块
from .static_feature_extractor import StaticFeatureExtractor
from .dynamic_feature_extractor import DynamicFeatureExtractor
from .mask_guided_attention import MaskGuidedAttention
from .fusion_regressor_sigmoid import BoundedFusionRegressor


class SurgicalQAModelBounded(nn.Module):
    """
    Surgical Quality Assessment Model with Bounded Score Regression

    版本说明：
    - 使用 BoundedFusionRegressor 确保输出在 [0, 1] 范围
    - 配合归一化标签使用（标签范围 [0, 1]）
    - 提供反归一化函数将预测转换回原始分数范围
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dict with:
                - static_dim: Static feature dimension (default: 512)
                - dynamic_dim: Dynamic feature dimension (default: 1024)
                - resnet_path: Path to custom ResNet checkpoint
                - i3d_path: Path to custom I3D checkpoint
                - use_pretrained: Use ImageNet pretrained weights
                - freeze_backbone: Freeze backbone weights
                - use_mask_loss: Use mask supervision loss
                - sampling_strategy: Frame sampling for static features
                - use_bounded_regression: Use bounded regression (default: True)
                - score_min: Original score minimum (default: 1.0 for GRS)
                - score_max: Original score maximum (default: 30.0 for GRS)
        """
        super().__init__()

        self.config = config
        self.static_dim = config.get('static_dim', 512)
        self.dynamic_dim = config.get('dynamic_dim', 1024)  # Standard I3D outputs 1024 channels
        self.use_mask_loss = config.get('use_mask_loss', True)

        # 新增：分数范围参数
        self.use_bounded_regression = config.get('use_bounded_regression', True)
        self.score_min = config.get('score_min', 1.0)  # GRS 最小值
        self.score_max = config.get('score_max', 30.0)  # GRS 最大值
        self.score_range = self.score_max - self.score_min

        print("\n" + "="*60)
        print("Initializing Surgical QA Model (Bounded Version)")
        print("="*60)
        print(f"  Static dimension: {self.static_dim}")
        print(f"  Dynamic dimension: {self.dynamic_dim}")
        print(f"  Use bounded regression: {self.use_bounded_regression}")
        print(f"  Score range: [{self.score_min}, {self.score_max}]")
        print("="*60 + "\n")

        # 1. Static Feature Extractor (ResNet-34)
        print("Initializing Static Feature Extractor (ResNet-34)...")
        self.static_extractor = StaticFeatureExtractor(
            resnet_path=config.get('resnet_path', None),
            use_pretrained=config.get('use_pretrained', True),
            freeze_early_layers=config.get('freeze_backbone', True),
            output_dim=self.static_dim,
            sampling_strategy=config.get('sampling_strategy', 'middle')
        )

        # 2. Dynamic Feature Extractor (I3D)
        print("Initializing Dynamic Feature Extractor (I3D)...")
        self.dynamic_extractor = DynamicFeatureExtractor(
            i3d_path=config.get('i3d_path', None),
            use_pretrained_i3d=config.get('use_pretrained_i3d', False),
            output_dim=self.dynamic_dim,
            freeze_backbone=config.get('freeze_backbone', True)
        )

        # 3. Mask-Guided Attention Module
        print("Initializing Mask-Guided Attention Module (for offline SAM3 masks)...")
        self.masked_attention = MaskGuidedAttention(
            enable_temporal_smoothing=True
        )

        # 4. Fusion and Regression Module
        print("Initializing Fusion Regressor...")
        if self.use_bounded_regression:
            print("  → Using BoundedFusionRegressor (with Sigmoid)")
            self.fusion_regressor = BoundedFusionRegressor(
                static_dim=self.static_dim,
                dynamic_dim=self.dynamic_dim,
                hidden_dims=config.get('regressor_hidden_dims', [1024, 512, 256, 128])
            )
        else:
            print("  → Using original FusionRegressor (unbounded)")
            from .surgical_qa_model import FusionRegressor
            self.fusion_regressor = FusionRegressor(
                static_dim=self.static_dim,
                dynamic_dim=self.dynamic_dim,
                hidden_dims=config.get('regressor_hidden_dims', [1024, 512, 256, 128])
            )

        print("Model initialized successfully!")

    def forward(self, video, masks=None, return_features=False, return_attention=False):
        """
        Forward pass for surgical video quality assessment.

        Args:
            video: (B, C, T, H, W) - Surgical video
            masks: (B, T, H_mask, W_mask) - Instrument masks [0,1]
                   If None, masks are treated as all-ones (no masking)
            return_features: Return intermediate features for analysis
            return_attention: Return attention map for visualization

        Returns:
            If return_features and return_attention:
                score, features_dict
            Else:
                score_normalized: (B, 1) - Predicted quality score in [0, 1] range
        """
        B, C, T, H, W = video.shape
        device = video.device

        # Handle masks
        if masks is None:
            # Create all-ones mask (no masking)
            masks = torch.ones(B, T, H, W, device=device)

        # 1. Extract static features A
        # ResNet-34 on keyframes -> captures tissue state and surgical field clarity
        static_features = self.static_extractor(video)  # (B, static_dim)

        # 2. Extract dynamic features C
        # I3D on video -> captures spatiotemporal dynamics
        # Get both feature map (for attention) and pooled features (for fusion)
        dynamic_feat_map, raw_dynamic = self.dynamic_extractor(video, return_features_map=True)
        # dynamic_feat_map: (B, dynamic_dim, T', H', W') - for attention module
        # raw_dynamic: (B, dynamic_dim) - for final fusion

        # 3-4. Apply mask-guided attention: B作用于C得到D
        # Generate attention from masks and apply to dynamic features
        masked_dynamic_features, attention_map, mask_loss = self.masked_attention(
            dynamic_feat_map,  # Spatial-temporal feature map
            masks,  # Instrument masks
            return_attention_map=return_attention
        )  # (B, dynamic_dim), (B, T', H', W'), optional loss

        # 5. Static (tissue/field) + Dynamic (instrument) = comprehensive features
        # Note: Fusion is handled inside fusion_regressor, no need to concatenate here

        # 6. Regress to score
        if self.use_bounded_regression:
            # BoundedFusionRegressor already outputs [0, 1]
            score_normalized = self.fusion_regressor(static_features, masked_dynamic_features)  # (B, 1)
        else:
            # Original FusionRegressor outputs unbounded score
            score_normalized = self.fusion_regressor(static_features, masked_dynamic_features)  # (B, 1)

        # Return based on flags
        if return_features:
            fused_features = torch.cat([static_features, masked_dynamic_features], dim=1)
            features_dict = {
                'static': static_features,
                'dynamic_feat_map': dynamic_feat_map,
                'dynamic_raw': raw_dynamic,
                'dynamic_masked': masked_dynamic_features,
                'fused': fused_features,
                'attention_map': attention_map,
                'mask_loss': mask_loss,
                'score_normalized': score_normalized
            }
            return score_normalized, features_dict
        else:
            return score_normalized, mask_loss

    def denormalize_score(self, score_normalized, target_min=None, target_max=None,
                        score_min=None, score_max=None):
        """
        反归一化：将归一化的分数 [0, 1] 转换回目标分数范围

        Args:
            score_normalized: (B,) or (B, 1) - 归一化后的预测分数，范围 [0, 1]
            target_min: 归一化目标最小值（默认 0.0）
            target_max: 归一化目标最大值（默认 1.0）
            score_min: 原始分数最小值（默认 self.score_min）
            score_max: 原始分数最大值（默认 self.score_max）

        Returns:
            score: 反归一化后的分数，在 [score_min, score_max] 或自定义范围

        Note:
            - 如果指定 target_min/target_max，则从归一化范围转换到自定义范围
            - 支持灵活的分数映射：训练时归一化到 [0,1]，推理时可映射到 [1,10] 或 [6,30]
        """
        if score_normalized.dim() == 2:
            score_normalized = score_normalized.squeeze(-1)
        elif score_normalized.dim() == 0:
            score_normalized = score_normalized.unsqueeze(0)

        # 使用传入参数或默认值
        target_min = target_min if target_min is not None else 0.0
        target_max = target_max if target_max is not None else 1.0
        score_min = score_min if score_min is not None else self.score_min
        score_max = score_max if score_max is not None else self.score_max

        score_range = score_max - score_min
        target_range = target_max - target_min

        # 反归一化公式：(norm - target_min) / target_range * score_range + score_min
        # 如果 target_range=0（防止除零），直接返回 score_min
        if target_range == 0:
            score = torch.full_like(score_normalized, score_min)
        else:
            score = (score_normalized - target_min) / target_range * score_range + score_min

        return score

    def compute_loss(self, score_pred, score_gt, mask_loss=None, lambda_mask=0.1):
        """
        Compute total loss for training.

        注意：由于使用了 Sigmoid 和归一化标签，score_pred 和 score_gt 都在 [0, 1] 范围

        Args:
            score_pred: (B, 1) - Predicted scores [0, 1]
            score_gt: (B,) or (B, 1) - Ground truth scores [0, 1]
            mask_loss: Optional mask supervision loss
            lambda_mask: Weight for mask loss

        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        # Score regression loss (MSE)
        if score_gt.dim() == 2:
            score_gt = score_gt.squeeze(-1)
        score_loss = F.mse_loss(score_pred.squeeze(-1), score_gt)

        total_loss = score_loss

        loss_dict = {
            'score_loss': score_loss.item(),
            'mask_loss': mask_loss.item() if mask_loss is not None else 0.0,
            'total_loss': score_loss.item()
        }

        # Add mask supervision loss if provided
        if mask_loss is not None:
            total_loss = total_loss + lambda_mask * mask_loss
            loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

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


def build_model_bounded(config):
    """
    Factory function to build Bounded Surgical QA Model.

    Args:
        config: Configuration dict or path to config file

    Returns:
        model: SurgicalQAModelBounded instance
    """
    if isinstance(config, str):
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    model = SurgicalQAModelBounded(config)

    # Load checkpoint if provided
    if 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {config['checkpoint_path']}")

    return model


if __name__ == '__main__':
    # Test bounded model
    print("Testing SurgicalQAModelBounded...")

    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mask_loss': True,
        'use_bounded_regression': True,  # Use bounded regression
        'score_min': 1.0,  # GRS minimum
        'score_max': 30.0  # GRS maximum
    }

    model = SurgicalQAModelBounded(config)
    model.count_parameters()

    # Test forward pass
    video = torch.randn(2, 3, 16, 112, 112)  # (B, C, T, H, W)
    masks = torch.randint(0, 2, (2, 16, 112, 112)).float()

    score_normalized, _ = model(video, masks)
    print(f"\nInput video shape: {video.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Predicted score (shape): {score_normalized.shape}")
    print(f"Predicted score range: [{score_normalized.min().item():.4f}, {score_normalized.max().item():.4f}]")

    # Test denormalization
    score_original = model.denormalize_score(score_normalized)
    print(f"Denormalized score range: [{score_original.min().item():.4f}, {score_original.max().item():.4f}]")

    # Test with feature return
    score, features = model(video, masks, return_features=True, return_attention=True)
    print(f"\nFeatures dict keys: {features.keys()}")
    print(f"Static features shape: {features['static'].shape}")
    print(f"Dynamic features shape: {features['dynamic_masked'].shape}")
