#!/usr/bin/env python3
"""
Test script to verify Surgical QA Model data flow
"""
import torch
import sys
sys.path.insert(0, '.')

print('='*60)
print('Testing Complete Surgical QA Model')
print('='*60)

from models import SurgicalQAModel

# Create model with config matching paper1231
config = {
    'static_dim': 512,
    'dynamic_dim': 832,
    'use_pretrained': False,
    'freeze_backbone': True,
    'use_mask_loss': True
}

model = SurgicalQAModel(config)
print()

# Test forward pass
B, C, T, H, W = 2, 3, 16, 224, 224
video = torch.randn(B, C, T, H, W)
masks = torch.randint(0, 2, (B, T, H, W)).float()

print('Input shapes:')
print(f'  Video: {video.shape}')
print(f'  Masks: {masks.shape}')
print()

# Forward pass with features
score, features = model(video, masks, return_features=True, return_attention=True)

print('Output shapes:')
print(f'  Score: {score.shape}')
print(f'  Static features: {features["static"].shape}')
print(f'  Dynamic feature map: {features["dynamic_feat_map"].shape}')
print(f'  Dynamic raw: {features["dynamic_raw"].shape}')
print(f'  Dynamic masked: {features["dynamic_masked"].shape}')
print(f'  Fused features: {features["fused"].shape}')
print(f'  Attention map: {features["attention_map"].shape}')
print()
print('✓ Model forward pass successful!')
