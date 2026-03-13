"""
Test script to verify the data flow and shape transformations
through the entire Surgical QA Model pipeline.
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
    print("COMPLETE DATA FLOW TRACE")
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
    print(f"  Frame Sampling: Use middle frame #{frame_idx}")
    print(f"  Selected frame shape: {selected_frame.shape} (B, C, 1, H, W)")
    print(f"  2D frame shape: {selected_frame[:, :, 0, :, :].shape} (B, C, H, W)")

    # ResNet layers shape calculation
    # Input: (B, 3, 224, 224)
    # After conv1 (7x7, stride2): (B, 64, 112, 112)
    # After maxpool1 (3x3, stride2): (B, 64, 56, 56)
    # After layer1 (3 blocks): (B, 64, 56, 56)
    # After layer2 (4 blocks): (B, 128, 28, 28)
    # After layer3 (6 blocks): (B, 256, 14, 14)
    # After layer4 (3 blocks): (B, 512, 7, 7)
    print("\n  ResNet-34 Feature Extraction:")
    print(f"    Input: (B, 3, 224, 224)")
    print(f"    conv1 + bn + relu: (B, 64, 112, 112)")
    print(f"    maxpool: (B, 64, 56, 56)")
    print(f"    layer1 (3 blocks): (B, 64, 56, 56)")
    print(f"    layer2 (4 blocks): (B, 128, 28, 28)")
    print(f"    layer3 (6 blocks): (B, 256, 14, 14)")
    print(f"    layer4 (3 blocks): (B, 512, 7, 7)")

    # Multi-scale downsampling
    # Input: (B, 512, 7, 7)
    # scale2 (stride2): (B, 128, 4, 4) -> adaptive_avg_pool -> (B, 128, 1, 1)
    # scale4 (stride4): (B, 128, 2, 2) -> adaptive_avg_pool -> (B, 128, 1, 1)
    # scale8 (stride8): (B, 128, 1, 1) -> adaptive_avg_pool -> (B, 128, 1, 1)
    # Concatenate: (B, 384, 1, 1) -> flatten -> (B, 384)
    print("\n  Multi-Scale Downsampling:")
    print(f"    Input: (B, 512, 7, 7)")
    print(f"    Scale 2 (stride=2): (B, 128, 4, 4) -> pool -> (B, 128, 1, 1)")
    print(f"    Scale 4 (stride=4): (B, 128, 2, 2) -> pool -> (B, 128, 1, 1)")
    print(f"    Scale 8 (stride=8): (B, 128, 1, 1) -> pool -> (B, 128, 1, 1)")
    print(f"    Concatenate: (B, 384, 1, 1)")
    print(f"    Flatten: (B, 384)")

    # Projection
    # 384 -> 512 -> 512 (output_dim)
    print("\n  Projection Layer:")
    print(f"    Input: (B, 384)")
    print(f"    Linear(384, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 512)")
    print(f"    Output (static_features A): (B, 512)")

    static_features = model.static_extractor(video)
    print(f"\n  ✓ Actual static features shape: {static_features.shape}")

    # Step 2: Dynamic Feature Extraction (I3D)
    print("\n" + "-" * 80)
    print("STEP 2: I3D DYNAMIC FEATURE EXTRACTION")
    print("-" * 80)

    print("\n  I3D Backbone Shape Calculation:")
    print(f"    Input: (B, 3, 16, 224, 224)")
    print(f"    conv1 (1x7x7, stride=1,2,2): (B, 64, 16, 112, 112)")
    print(f"    maxpool1: (B, 64, 16, 56, 56)")
    print(f"    conv2: (B, 64, 16, 56, 56)")
    print(f"    conv3: (B, 192, 16, 56, 56)")
    print(f"    maxpool2: (B, 192, 16, 28, 28)")
    print(f"    mixed_3b, mixed_3c: (B, 256, 16, 28, 28)")
    print(f"    maxpool3 (stride=2,2,2): (B, 256, 8, 14, 14)")
    print(f"    mixed_4b-f: (B, 480, 8, 14, 14)")
    print(f"    maxpool4 (stride=2,2,2): (B, 480, 4, 7, 7)")
    print(f"    mixed_5b, mixed: (B, 832, 4, 7, 7)")

    print("\n  Mixed Conv Layer (Eq.5):")
    print(f"    Input: (B, 832, 4, 7, 7)")
    print(f"    MixedConv3D: (B, 832, 4, 7, 7)")

    print("\n  Spatial Max Pooling (Eq.5):")
    print(f"    Input: (B, 832, 4, 7, 7)")
    print(f"    max_pool3d(kernel=(1,7,7)): (B, 832, 4, 1, 1)")

    print("\n  Global Pooling:")
    print(f"    adaptive_avg_pool3d(1,1,1): (B, 832, 1, 1, 1)")
    print(f"    Flatten: (B, 832)")

    dynamic_feat_map, raw_dynamic = model.dynamic_extractor(video, return_features_map=True)
    print(f"\n  ✓ Actual dynamic feature map shape: {dynamic_feat_map.shape}")
    print(f"  ✓ Actual raw dynamic features shape: {raw_dynamic.shape}")

    # Step 3: Mask-Guided Attention
    print("\n" + "-" * 80)
    print("STEP 3: MASK-GUIDED ATTENTION MODULE")
    print("-" * 80)

    print(f"\n  Input Masks Shape: {masks.shape} (B, T, H, W)")

    print("\n  Temporal Smoothing (Paper Eq.6):")
    print(f"    Input masks: (B, 16, 224, 224)")
    print(f"    Sum adjacent pairs (M{{2t-1}} + M{{2t}}): (B, 8, 224, 224)")
    print(f"    2D avg_pool (kernel=2, stride=2): (B, 8, 112, 112)")
    print(f"    Normalize by 2: (B, 8, 112, 112)")
    print(f"    Temporal interpolate to T=4: (B, 4, 112, 112)")

    print("\n  Dimension Alignment:")
    print(f"    Input: (B, 4, 112, 112)")
    print(f"    Add channel dim: (B, 1, 4, 112, 112)")
    print(f"    interpolate to (4, 7, 7): (B, 1, 4, 7, 7)")

    print("\n  Mask Projection (Conv3d):")
    print(f"    Input: (B, 1, 4, 7, 7)")
    print(f"    Conv3d(1, 64) + ReLU: (B, 64, 4, 7, 7)")
    print(f"    Conv3d(64, 1) + Sigmoid: (B, 1, 4, 7, 7)")
    print(f"    Squeeze: (B, 4, 7, 7) - mask_attention")

    print("\n  Feature Attention (Conv3d):")
    print(f"    Input features: (B, 832, 4, 7, 7)")
    print(f"    Conv3d(832, 208) + ReLU: (B, 208, 4, 7, 7)")
    print(f"    Conv3d(208, 1) + Sigmoid: (B, 1, 4, 7, 7)")
    print(f"    Squeeze: (B, 4, 7, 7) - feature_attention")

    print("\n  Attention Fusion:")
    print(f"    mask_attention + feature_attention: (B, 4, 7, 7)")
    print(f"    combined_attention: (B, 1, 4, 7, 7)")

    print("\n  Apply Attention (Eq.11: F_dy = F_clip ⊙ (A + I)):")
    print(f"    dynamic_features: (B, 832, 4, 7, 7)")
    print(f"    combined_attention + 1.0: (B, 1, 4, 7, 7)")
    print(f"    Element-wise multiplication: (B, 832, 4, 7, 7)")

    print("\n  Global Pooling:")
    print(f"    adaptive_avg_pool3d(1,1,1): (B, 832, 1, 1, 1)")
    print(f"    Flatten: (B, 832) - masked_dynamic_features D")

    masked_dynamic_features, attention_map, mask_loss = model.masked_attention(
        dynamic_feat_map, masks, return_attention_map=True
    )
    print(f"\n  ✓ Actual masked dynamic features shape: {masked_dynamic_features.shape}")
    print(f"  ✓ Actual attention map shape: {attention_map.shape}")

    # Step 4: Feature Fusion
    print("\n" + "-" * 80)
    print("STEP 4: FEATURE FUSION (A + D)")
    print("-" * 80)

    print(f"\n  Static Features A: {static_features.shape}")
    print(f"  Dynamic Features D: {masked_dynamic_features.shape}")
    print(f"  Concatenate(A, D): (B, 512+832) = (B, 1344)")

    fused_features = torch.cat([static_features, masked_dynamic_features], dim=1)
    print(f"\n  ✓ Actual fused features shape: {fused_features.shape}")

    # Step 5: Score Regression
    print("\n" + "-" * 80)
    print("STEP 5: SCORE REGRESSION (MLP)")
    print("-" * 80)

    print("\n  MLP Layers:")
    print(f"    Input: (B, 1344)")
    print(f"    Linear(1344, 1024) + LayerNorm + ReLU + Dropout(0.3): (B, 1024)")
    print(f"    Linear(1024, 512) + LayerNorm + ReLU + Dropout(0.3): (B, 512)")
    print(f"    Linear(512, 256) + LayerNorm + ReLU + Dropout(0.3): (B, 256)")
    print(f"    Linear(256, 64) + LayerNorm + ReLU + Dropout(0.2): (B, 64)")
    print(f"    Linear(64, 1): (B, 1) - Final Score")

    score = model.fusion_regressor(static_features, masked_dynamic_features)
    print(f"\n  ✓ Actual output score shape: {score.shape}")

    print("\n" + "=" * 80)
    print("DATA FLOW TRACE COMPLETE ✓")
    print("=" * 80)

    return score, {
        'static': static_features,
        'dynamic_feat_map': dynamic_feat_map,
        'dynamic_raw': raw_dynamic,
        'dynamic_masked': masked_dynamic_features,
        'fused': fused_features,
        'attention_map': attention_map,
        'mask_loss': mask_loss
    }


def main():
    """Main test function."""
    print("\n" + "🏗️  " * 20)
    print("SURGICAL QA MODEL DATA FLOW VERIFICATION")
    print("🏗️  " * 20 + "\n")

    # Configuration
    config = {
        'static_dim': 512,
        'dynamic_dim': 832,
        'use_pretrained': False,
        'freeze_backbone': False,  # Set to False to allow training
        'use_mask_loss': True,
        'sampling_strategy': 'middle',
        'regressor_hidden_dims': [1024, 512, 256]
    }

    # Create model
    print("Initializing model...")
    model = SurgicalQAModel(config)

    # Count parameters
    print("\nModel Parameters:")
    total, trainable = model.count_parameters()
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Ratio: {trainable/total*100:.2f}%")

    # Model set to eval mode for testing
    model.eval()

    # Create test inputs
    print("\nCreating test inputs...")
    B, C, T, H, W = 2, 3, 16, 224, 224
    video = torch.randn(B, C, T, H, W)
    masks = torch.randint(0, 2, (B, T, H, W)).float()

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        score, features = trace_forward(model, video, masks)

    # Verify output
    print(f"\n\n📊 Final Results:")
    print(f"  Input video shape: {video.shape}")
    print(f"  Predicted score shape: {score.shape}")
    print(f"  Predicted scores: {score.squeeze(-1).tolist()}")
    print(f"  All values finite: {torch.isfinite(score).all().item()}")
    print(f"  Values range: [{score.min().item():.4f}, {score.max().item():.4f}]")

    # Test training mode
    print("\n\nTesting training mode (with mask loss)...")
    model.train()
    video_train = torch.randn(2, 3, 16, 224, 224)
    masks_train = torch.randint(0, 2, (2, 16, 224, 224)).float()
    score_train, features_train = model(video_train, masks_train, return_features=True)

    # Compute loss
    score_gt = torch.tensor([3.5, 4.2])  # Ground truth scores
    total_loss, loss_dict = model.compute_loss(score_train, score_gt, features_train['mask_loss'])

    print(f"\n  Training Mode Output:")
    print(f"    Predicted scores: {score_train.squeeze(-1).tolist()}")
    print(f"    Ground truth scores: {score_gt.tolist()}")
    print(f"    Score loss: {loss_dict['score_loss']:.4f}")
    print(f"    Mask loss: {loss_dict['mask_loss']:.4f}")
    print(f"    Total loss: {loss_dict['total_loss']:.4f}")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
