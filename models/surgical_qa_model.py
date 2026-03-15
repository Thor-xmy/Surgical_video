"""
Main Surgical Quality Assessment Model
Based on paper1231 framework, adapted for single-video scoring

Framework Overview:
1. Input: Surgical video X (B, C, T, H, W)
2. Static Feature Extraction: ResNet-34 -> Feature A
3. Mask Extraction: Read offline masks -> Mask B
4. Dynamic Feature Extraction: I3D -> Feature C
5. Mask-Guided Attention: Apply B to C -> Feature D
6. Feature Fusion: Concat(A, D) -> Fused Features
7. Score Regression: FC layers -> Score y

Differences from paper1231:
- Paper1231: Contrastive learning with query/reference video pairs
- This model: Single video direct regression (no cross-attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .static_feature_extractor import StaticFeatureExtractor
from .dynamic_feature_extractor import DynamicFeatureExtractor
from .mask_guided_attention import MaskGuidedAttention


class FusionRegressor(nn.Module):
    """
    Feature Fusion and Score Regression Module.

    Concatenates static and dynamic features, then regresses to score.
    """
    def __init__(self, static_dim, dynamic_dim, hidden_dims=[1024, 512, 256, 128]):
        """
        Args:
            static_dim: Dimension of static features
            dynamic_dim: Dimension of dynamic features
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        input_dim = static_dim + dynamic_dim

        # Build MLP regressor
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Final output layer (single score)
        layers.extend([
            nn.Linear(prev_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        ])

        self.regressor = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, static_features, dynamic_features):
        """
        Fuse features and regress to score.

        Args:
            static_features: (B, static_dim)
            dynamic_features: (B, dynamic_dim)
        Returns:
            score: (B, 1)
        """
        # Concatenate features
        fused = torch.cat([static_features, dynamic_features], dim=1)  # (B, static_dim + dynamic_dim)

        # Regress to score
        score = self.regressor(fused)  # (B, 1)

        return score


class SurgicalQAModel(nn.Module):
    """
    Surgical Quality Assessment Model.

    Complete pipeline for surgical video skill assessment.
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dict with:
                - static_dim: Static feature dimension (default: 512)
                - dynamic_dim: Dynamic feature dimension (default: 1024)
                - resnet_path: Path to ResNet checkpoint (optional)
                - i3d_path: Path to I3D checkpoint (optional)
                - use_pretrained: Use ImageNet pretrained weights
                - freeze_backbone: Freeze backbone weights
                - use_mask_loss: Use mask supervision loss
                - sampling_strategy: Frame sampling for static features
        """
        super().__init__()

        self.config = config
        self.static_dim = config.get('static_dim', 512)
        self.dynamic_dim = config.get('dynamic_dim', 1024)  # Standard I3D outputs 1024 channels
        self.use_mask_loss = config.get('use_mask_loss', True)

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
        self.fusion_regressor = FusionRegressor(
            static_dim=self.static_dim,
            dynamic_dim=self.dynamic_dim,
            hidden_dims=config.get('regressor_hidden_dims', [1024, 512, 256])
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
                score: (B, 1) - Predicted quality score
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
        score = self.fusion_regressor(static_features, masked_dynamic_features)  # (B, 1)

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
                'mask_loss': mask_loss
            }
            return score, features_dict
        else:
            return score, mask_loss

    def compute_loss(self, score_pred, score_gt, mask_loss=None, lambda_mask=0.1):
        """
        Compute total loss for training.

        Args:
            score_pred: (B, 1) - Predicted scores
            score_gt: (B,) or (B, 1) - Ground truth scores
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


def build_model(config):
    """
    Factory function to build Surgical QA Model.

    Args:
        config: Configuration dict or path to config file

    Returns:
        model: SurgicalQAModel instance
    """
    if isinstance(config, str):
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    model = SurgicalQAModel(config)

    # Load checkpoint if provided
    if 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {config['checkpoint_path']}")

    return model


if __name__ == '__main__':
    # Test model
    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,  # Standard I3D outputs 1024 channels
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mask_loss': True
    }

    model = SurgicalQAModel(config)
    model.count_parameters()

    # Test forward pass
    video = torch.randn(2, 3, 16, 112, 112)  # (B, C, T, H, W) - paper requires 112x112
    masks = torch.randint(0, 2, (2, 16, 112, 112)).float()

    score = model(video, masks)
    print(f"Video shape: {video.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Predicted score shape: {score.shape}")
    print(f"Predicted score: {score.squeeze(-1)}")

    # Test with feature return
    score, features = model(video, masks, return_features=True, return_attention=True)
    print(f"\nFeatures dict keys: {features.keys()}")
    print(f"Static features shape: {features['static'].shape}")
    print(f"Dynamic features shape: {features['dynamic_masked'].shape}")
