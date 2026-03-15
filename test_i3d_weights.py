#!/usr/bin/env python3
"""
Test I3D weights loading with official implementation
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/home/thor/pytorch-i3d')

import torch
from pytorch_i3d import InceptionI3d
from models.surgical_qa_model import build_model

print('='*60)
print("Testing I3D weights with official implementation")
print('='*60)

# 1. Test official I3D
print("\n1. Testing official InceptionI3d...")
i3d = InceptionI3d(
    num_classes=400,
    spatial_squeeze=True,
    in_channels=3,
    final_endpoint='Logits'
)

video = torch.randn(2, 3, 16, 112, 112)  # Paper requirement
print(f"   Input video shape: {video.shape}")

try:
    logits = i3d(video)
    print(f"   Output logits shape: {logits.shape}")
    print(f"   ✓ Official I3D works!")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 2. Test our model with official I3D weights
print("\n2. Testing SurgicalQAModel with official I3D...")

# Replace I3D with official implementation
from models.dynamic_feature_extractor import DynamicFeatureExtractor

# Create custom extractor using official weights
custom_extractor = DynamicFeatureExtractor(
    i3d_path='/home/thor/pytorch-i3d/models/rgb_imagenet.pt',
    use_pretrained_i3d=True,
    output_dim=1024,  # Standard I3D output
    freeze_backbone=True
)

print(f"   Custom extractor initialized")
print(f"   Feature extractor parameters: {sum(p.numel() for p in custom_extractor.feature_extractor.parameters()):,}")

video = torch.randn(2, 3, 16, 112, 112)  # Paper requirement
print(f"   Input video shape: {video.shape}")

try:
    features = custom_extractor(video)
    print(f"   Output features shape: {features.shape}")
    print(f"   ✓ Custom extractor works!")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("Test completed!")
print("="*60)
