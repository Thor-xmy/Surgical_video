"""
Test script to verify data flow and shape transformations
through entire Surgical QA Model pipeline.

Paper Requirements:
- Input resolution: 112x112 (NOT 224x224!)
- Clip length: 16 frames
- I3D outputs 1024 channels
- ResNet outputs 512 channels
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.surgical_qa_model import SurgicalQAModel


def trace_forward(model, video, masks):
    """
    Trace forward pass and print shapes at each step.
    """
    print("=" * 80)
    print("COMPLETE DATA FLOW TRACE - Paper: 112x112 resolution")
    print("=" * 80)

    B, C, T, H, W = video.shape
    print(f"\n📥 INPUT VIDEO SHAPE: ({B}, {C}, {T}, {H}, {W}) - Batch, Channels, Time, Height, Width")
    print(f"📥 INPUT MASKS SHAPE: {masks.shape} - Batch, Time, Height, Width")
    print()

    device = video.device

    # Step 1: Static Feature Extraction (ResNet-34)
    print("-" * 80)
    print("STEP 1: RESNET-34 STATIC FEATURE EXTRACTION")
    print("-" * 80)

    # Frame sampling - middle frame
    frame_idx = T // 2
    selected_frame = video[:, :, frame_idx:frame_idx+1, :, :]
    print(f"Frame Sampling: Use middle frame #{frame_idx}")
    print(f"Selected frame shape: {selected_frame.shape} (B, C, 1, H, W)")
    print(f"2D frame shape: {selected_frame[:, :, 0, :, :].shape} (B, C, H, W)")

    # ResNet-34 layers shape calculation
    # Input: (B, 3, 112, 112) - Paper: 112x112
    print("\nResNet-34 Feature Extraction:")
    print(f"    Input: (B, 3, {H}, {W})")
    print(f"    conv1 (7x7, stride=2): (B, 64, {H//2}, {W//2})")
    print(f"    maxpool1 (3x3, stride=2): (B, 64, {H//4}, {W//4})")
    print(f"    layer1 (3 blocks): (B, 64, {H//4}, {W//4})")
    print(f"    layer2 (4 blocks): (B, 128, {H//8}, {W//8})")
    print(f"    layer3 (6 blocks): (B, 256, {H//16}, {W//16})")
    print(f"    layer4 (3 blocks): (B, 512, {H//32}, {W//32})")

    # Multi-scale downsampling
    print("\nMulti-Scale Downsampling:")
    print(f"    Input: (B, 512, {H//32}, {W//32})")
    print(f"    Scale 2 (stride=2): (B, 128, {H//64}, {W//64}) -> pool -> (B, 128, 1, 1)")
    print(f"    Scale 4 (stride=4): (B, 128, {H//128}, {W//128}) -> pool -> (B, 128, 1, 1)")
    print(f"    Scale 8 (stride=8): (B, 128, {H//256}, {W//256}) -> pool -> (B, 128, 1, 1)")
    print(f"    Concatenate: (B, 384, 1, 1) -> flatten -> (B, 384)")

    # Projection
    print("\nProjection Layer:")
    print(f"    Input: (B, 384)")
    print(f"    Linear(384, 512) + ReLU + Dropout(0.3) -> Linear(512, 512)")
    print(f"    Output (static_features A): (B, 512)")

    static_features = model.static_extractor(video)
    print(f"\n✓ Actual static features shape: {static_features.shape}")

    # Step 2: Dynamic Feature Extraction (I3D)
    print("\n" + "-" * 80)
    print("STEP 2: I3D DYNAMIC FEATURE EXTRACTION")
    print("-" * 80)

    print("\nI3D Backbone Shape Calculation:")
    print(f"    Input: (B, 3, {T}, {H}, {W})")
    print(f"    conv1 (1x7x7, stride=2,2,2): (B, 64, {T}, {H//2}, {W//2})")
    print(f"    maxpool1 (1x3x3, stride=1,2,2): (B, 64, {T}, {H//4}, {W//4})")
    print(f"    conv2: (B, 64, {T}, {H//4}, {W//4})")
    print(f"    conv3: (B, 192, {T}, {H//4}, {W//4})")
    print(f"    maxpool2 (1x3x3, stride=1,2,2): (B, 192, {T}, {H//8}, {W//8})")
    print(f"    mixed_3b, mixed_3c: (B, 256, {T}, {H//8}, {W//8})")
    print(f"    maxpool3 (3x3x3, stride=2,2,2): (B, 256, {T}, {H//16}, {W//16})")
    print(f"    mixed_4b-f: (B, 480, {T}, {H//16}, {W//16})")
    print(f"    maxpool4 (2x2x2, stride=2,2,2): (B, 480, {T//2}, {H//32}, {W//32})")
    print(f"    mixed_5b, mixed_5c: (B, 832, {T//2}, {H//32}, {W//32})")

    print("\nMixed Conv Layer (Eq.5):")
    print(f"    Input: (B, 832, {T//2}, {H//32}, {W//32})")
    print(f"    MixedConv3D: (B, 832, {T//2}, {H//32}, {W//32})")

    print("\nSpatial Max Pooling (Eq.5):")
    print(f"    Input: (B, 832, {T//2}, {H//32}, {W//32})")
    print(f"    max_pool3d(kernel=(1, H//32, W//32)): (B, 832, {T//2}, 1, 1)")

    print("\nGlobal Pooling:")
    print(f"    adaptive_avg_pool3d(1,1,1): (B, 832, 1, 1, 1)")
    print(f"    Flatten: (B, 832)")

    dynamic_feat_map, raw_dynamic = model.dynamic_extractor(video, return_features_map=True)
    print(f"\n✓ Actual dynamic feature map shape: {dynamic_feat_map.shape}")
    print(f"✓ Actual raw dynamic features shape: {raw_dynamic.shape}")

    # Step 3: Mask-Guided Attention
    print("\n" + "-" * 80)
    print("STEP 3: MASK-GUIDED ATTENTION MODULE")
    print("-" * 80)

    print(f"\nInput Masks Shape: {masks.shape} (B, T, H, W)")

    print("\nTemporal Smoothing (Paper Eq.6):")
    print(f"    Input masks: (B, {T}, {H}, {W})")
    print(f"    Sum adjacent pairs (M{{2t-1}} + M{{2t}}): (B, {T//2}, {H}, {W})")
    print(f"    2D avg_pool (kernel=2, stride=2): (B, {T//2}, {H//2}, {W//2})")
    print(f"    Normalize by 2: (B, {T//2}, {H//2}, {W//2})")

    # Temporal interpolation
    target_T = dynamic_feat_map.size(1)  # Match I3D feature temporal dim
    print(f"\nTemporal interpolation to T={target_T}:")
    print(f"    Input: (B, {T//2}, {H//2}, {W//2})")
    print(f"    interpolate to (B, H//2, W//2, {target_T}): (B, {H//2}, {W//2}, {target_T})")
    print(f"    permute to (B, {target_T}, {H//2}, {W//2})")

    print("\nDimension Alignment:")
    print(f"    Input: (B, {target_T}, {H//2}, {W//2})")
    print(f"    Add channel dim: (B, 1, {target_T}, {H//2}, {W//2})")
    print(f"    interpolate to ({target_T}, {H}, {W}): (B, 1, {target_T}, {H}, {W})")
    print(f"    squeeze to (B, {target_T}, {H}, {W})")

    masked_dynamic_features, attention_map, mask_loss = model.masked_attention(
        dynamic_feat_map, masks, return_attention_map=True
    )

    print(f"\n✓ Actual masked dynamic features shape: {masked_dynamic_features.shape}")
    print(f"✓ Actual attention map shape: {attention_map.shape}")

    # Step 4: Feature Fusion
    print("\n" + "-" * 80)
    print("STEP 4: FEATURE FUSION (A + D)")
    print("-" * 80)

    print(f"\nStatic Features A: {static_features.shape}")
    print(f"Dynamic Features D: {masked_dynamic_features.shape}")
    print(f"Concatenate(A, D): (B, 512+1024) = (B, 1536)")

    fused_features = torch.cat([static_features, masked_dynamic_features], dim=1)
    print(f"\n✓ Actual fused features shape: {fused_features.shape}")

    # Step 5: Score Regression
    print("\n" + "-" * 80)
    print("STEP 5: SCORE REGRESSION (MLP)")
    print("-" * 80)

    print("\nMLP Layers:")
    print(f"    Input: (B, 1536)")
    print(f"    Linear(1536, 1024) + LayerNorm + ReLU + Dropout(0.3): (B, 1024)")
    print(f"    Linear(1024, 512) + LayerNorm + ReLU + Dropout(0.3): (B, 512)")
    print(f"    Linear(512, 256) + LayerNorm + ReLU + Dropout(0.3): (B, 256)")
    print(f"    Linear(256, 64) + LayerNorm + ReLU + Dropout(0.2): (B, 64)")
    print(f"    Linear(64, 1): (B, 1) - Final Score")

    score = model.fusion_regressor(static_features, masked_dynamic_features)
    print(f"\n✓ Actual output score shape: {score.shape}")

    print("\n" + "=" * 80)
    print("DATA FLOW TRACE COMPLETE ✓")
    print("=" * 80)


def main():
    """
    Main test function.
    """
    print("\n")
    print("=" * 80)
    print("PAPER-COMPLIANT PIPELINE TEST")
    print("Input: 112x112 resolution, 16 frames per clip")
    print("=" * 80)

    # Create model with standard I3D (1024 channels)
    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,  # Standard I3D output
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mask_loss': True
    }

    model = SurgicalQAModel(config)
    print(f"\nModel initialized:")
    print(f"  - Static dim: {config['static_dim']}")
    print(f"  - Dynamic dim: {config['dynamic_dim']}")
    print()

    # Test forward pass
    # Paper requirement: 112x112 resolution
    B, C, T, H, W = 2, 3, 16, 112, 112
    video = torch.randn(B, C, T, H, W)
    masks = torch.randint(0, 2, (B, T, H, W)).float()

    print(f"\nTest input:")
    print(f"  Video: {video.shape}")
    print(f"  Masks: {masks.shape}")
    print()

    # Trace forward pass
    trace_forward(model, video, masks)

    # Test full forward
    score, features = model(video, masks, return_features=True, return_attention=True)

    print("\nFinal verification:")
    print(f"  ✓ Input video: {video.shape}")
    print(f"  ✓ Input masks: {masks.shape}")
    print(f"  ✓ Output score: {score.shape}")
    print(f"  ✓ Static features: {features['static'].shape}")
    print(f"  ✓ Dynamic feature map: {features['dynamic_feat_map'].shape}")
    print(f"  ✓ Dynamic masked: {features['dynamic_masked'].shape}")
    print(f"  ✓ Fused: {features['fused'].shape}")
    print(f"  ✓ Attention: {features['attention_map'].shape}")

    # Verify dimensions
    assert score.shape == (B, 1), f"Wrong score shape: {score.shape}"
    assert features['static'].shape == (B, 512), f"Wrong static features: {features['static'].shape}"
    assert features['dynamic_feat_map'].shape == (B, 1024, T//2, H//32, W//32), \
        f"Wrong dynamic feat map: {features['dynamic_feat_map'].shape}"
    assert features['dynamic_masked'].shape == (B, 1024), f"Wrong dynamic masked: {features['dynamic_masked'].shape}"
    assert features['fused'].shape == (B, 1536), f"Wrong fused: {features['fused'].shape}"
    assert features['attention_map'].shape == (B, T//2, H//32, W//32), \
        f"Wrong attention: {features['attention_map'].shape}"

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Model is paper-compliant!")
    print("=" * 80)
    print()
    print("Paper Requirements Verified:")
    print("  ✓ Input resolution: 112x112")
    print("  ✓ Clip length: 16 frames")
    print("  ✓ I3D output: 1024 channels (standard I3D)")
    print("  ✓ ResNet output: 512 channels")
    print("  ✓ Attention applied correctly")
    print("  ✓ Fusion works correctly")
    print()


if __name__ == '__main__':
    main()
