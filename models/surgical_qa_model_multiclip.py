"""
Surgical Quality Assessment Model with Multi-Clip Support

Modified to extract features from multiple video clips and preserve temporal order
using the flattening approach.

Key changes:
1. Uses DynamicFeatureMultiClip instead of Dynamic
2. Features are extracted from multiple clips and concatenated
3. FusionRegressor accepts dynamic_dim = 1024 * num_clips
4. Only outputs a single video-level score (not per-clip scores)

Framework Overview:
1. Input: Surgical video X (B, C, T, H, W) - T can be > 16
2. Static Feature Extraction: ResNet-34 -> Feature A
3. Multi-Clip Dynamic Feature Extraction:
   - Split video into N clips (length=16, stride=10)
   - I3D extracts features from each clip -> [f1, f2, ..., fN]
   - Concatenate: F = [f1, f2, ..., fN] (preserves temporal order)
4. Feature Fusion: Concat(A, F) -> Fused Features
5. Score Regression: FC layers -> Single video-level score

Note: Mask-guided attention is NOT used in multi-clip mode because:
- Each clip is processed independently
- Masks would need to be split into clips as well
- This is a design choice for simplicity; can be added later
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .static_feature_extractor import StaticFeatureExtractor
from .dynamic_feature_extractor_multiclip import DynamicFeatureMultiClip


class FusionRegressorMultiClip(nn.Module):
    """
    Feature Fusion and Score Regression Module for Multi-Clip Features.

    Concatenates static and dynamic features, then regresses to score.

    The dynamic_features input has shape (B, dynamic_dim) where
    dynamic_dim = 1024 * num_clips (flattened temporal sequence).
    """
    def __init__(self, static_dim, dynamic_dim, hidden_dims=[1024, 512, 256, 128]):
        """
        Args:
            static_dim: Dimension of static features (e.g., 512)
            dynamic_dim: Dimension of dynamic features (e.g., 1024 * num_clips)
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
            dynamic_features: (B, dynamic_dim) where dynamic_dim = 1024 * num_clips

        Returns:
            score: (B, 1)
        """
        # Concatenate features
        fused = torch.cat([static_features, dynamic_features], dim=1)  # (B, static_dim + dynamic_dim)

        # Regress to score
        score = self.regressor(fused)  # (B, 1)

        return score


class SurgicalQAModelMultiClip(nn.Module):
    """
    Surgical Quality Assessment Model with Multi-Clip Feature Extraction.

    Complete pipeline for surgical video skill assessment using temporal-preserving
    multi-clip feature extraction.
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dict with:
                - static_dim: Static feature dimension (default: 512)
                - dynamic_dim: Dynamic feature dimension per clip (default: 1024)
                - clip_length: Length of each clip (default: 16)
                - clip_stride: Stride between clips (default: 10)
                - max_clips: Maximum number of clips (default: None)
                - resnet_path: Path to ResNet checkpoint (optional)
                - i3d_path: Path to I3D checkpoint (optional)
                - use_pretrained: Use ImageNet pretrained weights
                - freeze_backbone: Freeze backbone weights
                - use_mixed_conv: Use mixed convolution
        """
        super().__init__()

        self.config = config
        self.static_dim = config.get('static_dim', 512)
        self.dynamic_dim_per_clip = config.get('dynamic_dim', 1024)  # Per clip

        # Multi-clip parameters
        self.clip_length = config.get('clip_length', 16)
        self.clip_stride = config.get('clip_stride', 10)
        self.max_clips = config.get('max_clips', None)

        print("\n" + "="*60)
        print("Initializing Surgical QA Model (Multi-Clip Version)")
        print("="*60)
        print(f"  Static dimension: {self.static_dim}")
        print(f"  Dynamic dimension per clip: {self.dynamic_dim_per_clip}")
        print(f"  Clip length: {self.clip_length}")
        print(f"  Clip stride: {self.clip_stride}")
        print(f"  Max clips: {self.max_clips if self.max_clips else 'auto'}")
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

        # 2. Dynamic Feature Extractor with Multi-Clip Support
        print("Initializing Dynamic Feature Extractor (I3D) with Multi-Clip Support...")
        self.dynamic_extractor = DynamicFeatureMultiClip(
            i3d_path=config.get('i3d_path', None),
            use_pretrained_i3d=config.get('use_pretrained_i3d', False),
            output_dim=self.dynamic_dim_per_clip,
            freeze_backbone=config.get('freeze_backbone', True),
            use_mixed_conv=config.get('use_mixed_conv', True),
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            max_clips=self.max_clips
        )

        # Note: Mask-guided attention is NOT used in multi-clip mode
        # because each clip is processed independently

        # 3. Fusion and Regression Module
        # Calculate total dynamic dimension (will be set in first forward pass)
        self.total_dynamic_dim = None
        self.fusion_regressor = None
        print("Fusion Regressor will be initialized in first forward pass...")

        print("Model initialized successfully!")

    def forward(self, video, return_features=False, return_per_clip_features=False):
        """
        Forward pass for surgical video quality assessment with multi-clip features.

        Args:
            video: (B, C, T, H, W) - Surgical video
                    T should be > 16 for meaningful multi-clip extraction
            return_features: Return intermediate features for analysis
            return_per_clip_features: Return per-clip features

        Returns:
            score: (B, 1) - Predicted quality score (video-level)
            If return_features: also return features_dict
            If return_per_clip_features: also return per-clip features
        """
        B, C, T, H, W = video.shape

        # 1. Extract static features A
        static_features = self.static_extractor(video)  # (B, static_dim)

        # 2. Extract multi-clip dynamic features
        flattened_dynamic, per_clip_dynamic, num_clips = self.dynamic_extractor.extract_multiclip_features(video)
        # flattened_dynamic: (B, dynamic_dim_per_clip * num_clips)
        # per_clip_dynamic: (B, num_clips, dynamic_dim_per_clip)

        # Initialize fusion regressor if not done yet
        current_dynamic_dim = flattened_dynamic.shape[1]
        if self.fusion_regressor is None:
            self.total_dynamic_dim = current_dynamic_dim
            print(f"\nInitializing Fusion Regressor with dynamic_dim={self.total_dynamic_dim}...")
            self.fusion_regressor = FusionRegressorMultiClip(
                static_dim=self.static_dim,
                dynamic_dim=self.total_dynamic_dim,
                hidden_dims=self.config.get('regressor_hidden_dims', [1024, 512, 256])
            )
            # Move to same device as video
            self.fusion_regressor = self.fusion_regressor.to(video.device)
        elif self.total_dynamic_dim != current_dynamic_dim:
            # Note: In practice, you should fix video length or use max_clips
            # This warning is for flexibility in testing
            print(f"\nWarning: Input dynamic_dim ({current_dynamic_dim}) differs from "
                  f"initialized ({self.total_dynamic_dim}). This may cause shape errors.")
            print("  Consider setting max_clips in config or using fixed video length.")

        # 3. Regress to score
        score = self.fusion_regressor(static_features, flattened_dynamic)  # (B, 1)

        # Return based on flags
        if return_features or return_per_clip_features:
            features_dict = {
                'static': static_features,
                'dynamic_flattened': flattened_dynamic,
                'dynamic_per_clip': per_clip_dynamic,
                'num_clips': num_clips,
                'clip_length': self.clip_length,
                'clip_stride': self.clip_stride
            }
            if return_features:
                return score, features_dict
            else:
                return score, per_clip_dynamic
        else:
            return score

    def compute_loss(self, score_pred, score_gt):
        """
        Compute total loss for training.

        Args:
            score_pred: (B, 1) - Predicted scores
            score_gt: (B,) or (B, 1) - Ground truth scores

        Returns:
            total_loss: MSE loss
            loss_dict: Loss info
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


def build_model_multiclip(config):
    """
    Factory function to build Multi-Clip Surgical QA Model.

    Args:
        config: Configuration dict or path to config file

    Returns:
        model: SurgicalQAModelMultiClip instance
    """
    if isinstance(config, str):
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    model = SurgicalQAModelMultiClip(config)

    # Load checkpoint if provided
    if 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {config['checkpoint_path']}")

    return model


if __name__ == '__main__':
    # Test model
    print("Testing SurgicalQAModelMultiClip...")

    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'clip_length': 16,
        'clip_stride': 10,
        'max_clips': None,
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mixed_conv': True,
        'regressor_hidden_dims': [1024, 512, 256]
    }

    model = SurgicalQAModelMultiClip(config)
    model.count_parameters()

    # Test with different video lengths
    test_cases = [
        (2, 3, 100, 112, 112),  # 100 frames -> 10 clips
        (2, 3, 90, 112, 112),    # 90 frames -> 9 clips
        (2, 3, 50, 112, 112),    # 50 frames -> 5 clips
        (2, 3, 16, 112, 112),    # 16 frames -> 1 clip
    ]

    for video_shape in test_cases:
        video = torch.randn(*video_shape)

        # First forward pass (initializes fusion regressor)
        score = model(video)

        print(f"\nInput shape: {video_shape}")
        print(f"  Output score shape: {score.shape}")
        print(f"  Output score range: [{score.min().item():.4f}, {score.max().item():.4f}]")

        # Test with feature return
        score, features = model(video, return_features=True, return_per_clip_features=True)
        print(f"  Features dict keys: {features.keys()}")
        print(f"  Static features shape: {features['static'].shape}")
        print(f"  Flattened dynamic features shape: {features['dynamic_flattened'].shape}")
        print(f"  Per-clip dynamic features shape: {features['dynamic_per_clip'].shape}")
        print(f"  Number of clips: {features['num_clips']}")

    print("\n✓ All tests passed!")
