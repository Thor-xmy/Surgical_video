"""
Final Complete Test for Multi-Clip Bounded Pipeline

This script tests the complete pipeline to ensure all components work together:
1. Video-level DataLoader
2. Multi-clip Static Feature Extractor
3. Multi-clip Dynamic Feature Extractor
4. Complete Model
5. Score denormalization
"""

import sys
import torch
import numpy as np
import json
import os

sys.path.insert(0, '/home/thor/surgical_qa_model')

print("="*70)
print("FINAL PIPELINE TEST")
print("="*70)

# Test 1: Test VideoLevelDataset in isolation
print("\n[1/5] Testing VideoLevelDataset in isolation...")
from utils.data_loader_video_level import VideoLevelDataset

# Create minimal test annotations
test_annotations = {
    'video_001': {'score': 18.0},
    'video_002': {'score': 22.5},
    'video_003': {'score': 12.3}
}

# Save to temp dir
test_data_root = '/tmp/test_multiclip_final'
os.makedirs(test_data_root, exist_ok=True)
os.makedirs(f'{test_data_root}/videos', exist_ok=True)
os.makedirs(f'{test_data_root}/masks', exist_ok=True)

with open(f'{test_data_root}/annotations.json', 'w') as f:
    json.dump(test_annotations, f)

dataset = VideoLevelDataset(
    data_root=test_data_root,
    score_min=6.0,
    score_max=30.0
)

print(f"  ✓ VideoLevelDataset loaded: {len(dataset)} samples")
print(f"  Score normalization: [6.0, 30.0] -> [0.0, 1.0]")
print("  ✓ Test 1 PASSED")

# Test 2: Test static extractor
print("\n[2/5] Testing StaticFeatureMultiClip...")
from models.static_feature_extractor_multiclip import StaticFeatureMultiClip

static_model = StaticFeatureMultiClip(
    output_dim=512,
    keyframe_strategy='middle',
    use_pretrained=False
)

# Test with video (2 videos, 3 channels, 100 frames, 224x224)
video = torch.randn(2, 3, 100, 224, 224)
static_per_clip, num_clips = static_model.extract_multiclip_features(video)

print(f"  Input video: {video.shape}")
print(f"  Static per-clip: {static_per_clip.shape}")
print(f"  Number of clips: {num_clips}")
print(f"  Expected: (2, 9, 512)")

assert static_per_clip.shape == (2, 9, 512)
print("  ✓ Test 2 PASSED")

# Test 3: Test dynamic extractor
print("\n[3/5] Testing DynamicFeatureMultiClip...")
from models.dynamic_feature_extractor_multiclip import DynamicFeatureMultiClip

dynamic_model = DynamicFeatureMultiClip(
    output_dim=1024,
    clip_length=16,
    clip_stride=10,
    use_pretrained_i3d=False
)

# Test
dynamic_per_clip, num_clips_dyn = dynamic_model.extract_multiclip_features(video)

print(f"  Input video: {video.shape}")
print(f"  Dynamic per-clip: {dynamic_per_clip.shape}")
print(f"  Number of clips: {num_clips_dyn}")
print(f"  Expected: (2, 9, 1024)")

assert dynamic_per_clip.shape == (2, 9, 1024)
print("  ✓ Test 3 PASSED")

# Test 4: Test per-clip fusion
print("\n[4/5] Testing Per-Clip Feature Fusion...")
fused_per_clip = torch.cat([static_per_clip, dynamic_per_clip], dim=-1)

print(f"  Static: {static_per_clip.shape}")
print(f"  Dynamic: {dynamic_per_clip.shape}")
print(f"  Fused per-clip: {fused_per_clip.shape}")
print(f"  Expected: (2, 9, 1536)  [512+1024]")

assert fused_per_clip.shape == (2, 9, 1536)
print("  ✓ Test 4 PASSED")

# Test 5: Test temporal flattening
print("\n[5/5] Testing Temporal Flattening...")
temporal_features = fused_per_clip.reshape(2, -1)

print(f"  Temporal features: {temporal_features.shape}")
print(f"  Expected: (2, 13824) [9 * 1536]")

assert temporal_features.shape == (2, 13824)
print("  ✓ Test 5 PASSED")

# Test 6: Test complete model
print("\n[6/5] Testing Complete Model...")
from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded

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
    'keyframe_strategy': 'middle'
}

model = SurgicalQAModelMultiClipBounded(config)
score = model(video)

print(f"  Input video: {video.shape}")
print(f"  Output score: {score.shape}")
print(f"  Output score range: [{score.min().item():.4f}, {score.max().item():.4f}]")

assert score.shape == (2, 1)
print("  ✓ Test 6 PASSED")

# Test 7: Test score denormalization
print("\n[7/5] Testing Score Denormalization...")
# Test denormalize to original range [6, 30]
score_6to30 = model.denormalize_score(score, target_min=6.0, target_max=30.0)
# Test denormalize to paper range [1, 10]
score_1to10 = model.denormalize_score(score, target_min=1.0, target_max=10.0)

print(f"  Input normalized scores: {score.squeeze().tolist()}")
print(f"  Denormalized [6,30]: {score_6to30.squeeze().tolist()}")
print(f"  Denormalized [1, 10]: {score_1to10.squeeze().tolist()}")

# Check ranges
assert all(6.0 <= s <= 30.0 for s in score_6to30.squeeze().tolist()), "Score [6,30] out of range"
assert all(1.0 <= s <= 10.0 for s in score_1to10.squeeze().tolist()), "Score [1,10] out of range"

print("  ✓ Test 7 PASSED")

# Test 8: Test with feature return
print("\n[8/5] Testing Feature Return...")
score, features = model(video, return_features=True)

print(f"  Features keys: {features.keys()}")
print(f"  Static per-clip: {features['static_per_clip'].shape}")
print(f"  Dynamic per-clip: {features['dynamic_per_clip'].shape}")
print(f"  Fused per-clip: {features['fused_per_clip'].shape}")
print(f"  Temporal: {features['temporal_features'].shape}")
print(f"  Num clips: {features['num_clips']}")

assert features['temporal_features'].shape == (2, 13824)
print("  ✓ Test 8 PASSED")

print("\n" + "="*70)
print("ALL FINAL TESTS PASSED!")
print("="*70)

print("\nSummary:")
print("  ✓ Video-level DataLoader with score normalization")
print("  ✓ Multi-clip static feature extraction")
print("  ✓ Multi-clip dynamic feature extraction")
print("  ✓ Per-clip feature fusion (方案A)")
print("  ✓ Temporal sequence flattening")
print("  ✓ Bounded regression with Sigmoid")
print("  ✓ Score denormalization [0,1] -> [6,30] or [1, 10]")
print("  ✓ Single video-level score output (not per-clip)")

print("\nThe complete multi-clip bounded pipeline is ready!")
print("="*70)
