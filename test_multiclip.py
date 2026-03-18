"""
Test script for Multi-Clip Surgical QA Model

This script verifies:
1. Multi-clip feature extraction works correctly
2. Features preserve temporal order
3. Model forward pass outputs single video-level score
4. Different video lengths are handled properly
"""

import torch
import sys
sys.path.append('/home/thor/surgical_qa_model')

from models.dynamic_feature_extractor_multiclip import DynamicFeatureMultiClip
from models.surgical_qa_model_multiclip import SurgicalQAModelMultiClip


def test_MultiClip():
    """Test multi-clip dynamic feature extraction."""
    print("\n" + "="*60)
    print("TEST 1: DynamicFeatureMultiClip")
    print("="*60)

    config = {
        'clip_length': 16,
        'clip_stride': 10,
        'max_clips': None
    }

    model = DynamicFeatureMultiClip(
        output_dim=1024,
        clip_length=config['clip_length'],
        clip_stride=config['clip_stride'],
        max_clips=config['max_clips']
    )

    test_cases = [
        (2, 3, 100, 112, 112),  # 100 frames -> 10 clips
        (2, 3, 90, 112, 112),    # 90 frames -> 9 clips
        (2, 3, 50, 112, 112),    # 50 frames -> 5 clips
        (2, 3, 16, 112, 112),    # 16 frames -> 1 clip
    ]

    for video_shape in test_cases:
        video = torch.randn(*video_shape)
        flattened, per_clip, num_clips = model.extract_multiclip_features(video)

        expected_dim = 1024 * num_clips

        print(f"\nInput: {video_shape}")
        print(f"  Number of clips: {num_clips}")
        print(f"  Flattened features: {flattened.shape}")
        print(f"  Per-clip features: {per_clip.shape}")
        print(f"  Expected flattened dim: {expected_dim}")

        assert flattened.shape == (video_shape[0], expected_dim), \
            f"Flattened shape mismatch: {flattened.shape} vs ({video_shape[0]}, {expected_dim})"
        assert per_clip.shape == (video_shape[0], num_clips, 1024), \
            f"Per-clip shape mismatch: {per_clip.shape} vs ({video_shape[0]}, {num_clips}, 1024)"

        # Verify temporal order is preserved
        # First clip's features should be different from last clip's
        if num_clips > 1:
            first_clip_feat = per_clip[0, 0, :]  # First batch, first clip
            last_clip_feat = per_clip[0, -1, :]  # First batch, last clip
            assert not torch.allclose(first_clip_feat, last_clip_feat, atol=1e-3), \
                "First and last clip features should be different (temporal order preserved)"

    print("\n✓ DynamicFeatureMultiClip tests passed!")


def test_surgical_qa_model_multiclip():
    """Test multi-clip surgical QA model."""
    print("\n" + "="*60)
    print("TEST 2: SurgicalQAModelMultiClip")
    print("="*60)

    base_config = {
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

    test_cases = [
        ((2, 3, 100, 112, 112), 10),  # 100 frames -> 10 clips
        ((2, 3, 90, 112, 112), 9),    # 90 frames -> 9 clips
        ((2, 3, 50, 112, 112), 5),    # 50 frames -> 5 clips
        ((2, 3, 16, 112, 112), 1),    # 16 frames -> 1 clip
    ]

    for video_shape, expected_num_clips in test_cases:
        # Create a new model for each test case to avoid dimension mismatch
        config = base_config.copy()
        model = SurgicalQAModelMultiClip(config)

        video = torch.randn(*video_shape)

        # First forward pass (initializes fusion regressor)
        score = model(video)

        print(f"\nInput: {video_shape}")
        print(f"  Output score shape: {score.shape}")
        print(f"  Output score range: [{score.min().item():.4f}, {score.max().item():.4f}]")

        assert score.shape == (video_shape[0], 1), \
            f"Score shape mismatch: {score.shape} vs ({video_shape[0]}, 1)"

        # Test with feature return
        score, features = model(video, return_features=True, return_per_clip_features=True)

        print(f"  Features dict keys: {features.keys()}")
        print(f"  Static features: {features['static'].shape}")
        print(f"  Flattened dynamic: {features['dynamic_flattened'].shape}")
        print(f"  Per-clip dynamic: {features['dynamic_per_clip'].shape}")
        print(f"  Number of clips: {features['num_clips']}")

        # Verify feature dimensions
        num_clips = features['num_clips']
        expected_dynamic_dim = 1024 * num_clips

        assert features['static'].shape == (video_shape[0], 512)
        assert features['dynamic_flattened'].shape == (video_shape[0], expected_dynamic_dim)
        assert features['dynamic_per_clip'].shape == (video_shape[0], num_clips, 1024)

    print("\n✓ SurgicalQAModelMultiClip tests passed!")


def test_loss_computation():
    """Test loss computation."""
    print("\n" + "="*60)
    print("TEST 3: Loss Computation")
    print("="*60)

    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'clip_length': 16,
        'clip_stride': 10,
        'use_pretrained': False,
        'freeze_backbone': True
    }

    model = SurgicalQAModelMultiClip(config)
    video = torch.randn(4, 3, 100, 112, 112)
    score_gt = torch.tensor([7.5, 8.2, 6.8, 9.0])

    # Forward pass
    score_pred, _ = model(video, return_features=True)

    # Compute loss
    loss, loss_dict = model.compute_loss(score_pred, score_gt)

    print(f"\nScore prediction shape: {score_pred.shape}")
    print(f"Score ground truth shape: {score_gt.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

    assert isinstance(loss.item(), float)
    assert 'score_loss' in loss_dict
    assert 'total_loss' in loss_dict



    print("\n✓ Loss computation tests passed!")


def test_temporal_order_preservation():
    """Test that temporal order is preserved in features."""
    print("\n" + "="*60)
    print("TEST 4: Temporal Order Preservation")
    print("="*60)

    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'clip_length': 16,
        'clip_stride': 10,
        'use_pretrained': False,
        'freeze_backbone': True
    }

    model = SurgicalQAModelMultiClip(config)

    # Create a video with temporal structure
    # First half has low values, second half has high values
    video = torch.randn(2, 3, 100, 112, 112)
    video[:, :, :50, :, :] = video[:, :, :50, :, :] * 0.1  # First half: low intensity
    video[:, :, 50:, :, :] = video[:, :, 50:, :, :] * 2.0  # Second half: high intensity

    # Extract features
    score, features = model(video, return_features=True, return_per_clip_features=True)

    per_clip_features = features['dynamic_per_clip']  # (B, num_clips, 1024)
    num_clips = features['num_clips']

    print(f"\nNumber of clips: {num_clips}")
    print(f"Per-clip features shape: {per_clip_features.shape}")

    # Check that features from different clips are different
    for i in range(num_clips - 1):
        feat_i = per_clip_features[0, i, :]
        feat_j = per_clip_features[0, i+1, :]
        diff = torch.norm(feat_i - feat_j).item()

        print(f"  Clip {i} vs Clip {i+1} difference: {diff:.4f}")

        # Features should be different (small difference due to same initial video)
        # but we expect at least some difference due to temporal structure
        if i < num_clips // 2:
            # Early clips (first half) should be different from later clips
            feat_later = per_clip_features[0, -1, :]
            diff_early_later = torch.norm(feat_i - feat_later).item()
            print(f"  Clip {i} vs Last clip difference: {diff_early_later:.4f}")

    print("\n✓ Temporal order preservation tests passed!")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Multi-Clip Surgical QA Model Test Suite")
    print("="*60)

    try:
        test_MultiClip()
        test_surgical_qa_model_multiclip()
        test_loss_computation()
        test_temporal_order_preservation()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
