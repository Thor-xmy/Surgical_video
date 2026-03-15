"""
Mask-Guided Attention Module
Based on paper1231: Mask-guided Attention Module (Section C.2)

This module applies offline SAM3 masks as spatial attention to focus on
instrument manipulation regions.

IMPORTANT: This implementation is designed for OFFLINE SAM3 masks.
Unlike paper1231 where SAM2 provides features for learning attention A,
here we directly use the pre-computed high-precision SAM3 masks as attention.

Reference from paper1231:
- Formula: F_dy = F_clip * (A + I)
- Where A is the attention map and I is identity

Difference from paper1231:
- Paper: SAM2 provides features → learn attention A → use mask as supervision
- Here: SAM3 provides offline masks → use masks directly as attention A
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGuidedAttention(nn.Module):
    """
    Mask-Guided Attention Module for offline SAM3 masks.

    Implementation for high-precision offline SAM3 segmentation masks.
    The masks are used directly as spatial attention without additional learning.

    Formula: F_dy = F_clip * (A + I)

    Where:
    - F_clip: Short-term spatiotemporal features from I3D
    - A: Attention map (from SAM3 masks after temporal smoothing)
    - I: Identity (scalar 1.0)
    - F_dy: Masked dynamic features

    Key characteristics for offline masks:
    1. No learnable attention: SAM3 masks are already high-precision
    2. No mask projection: Convolution would destroy sharp edges
    3. Temporal smoothing: Only for stabilizing across frames
    4. Spatial interpolation: For aligning with I3D feature resolution
    """
    def __init__(self,
                 enable_temporal_smoothing=True):
        """
        Args:
            enable_temporal_smoothing: Apply temporal smoothing to reduce SAM3 jitter
        """
        super().__init__()

        self.enable_temporal_smoothing = enable_temporal_smoothing

        # NOTE: No learnable parameters for offline SAM3 masks!
        # The masks are used directly as attention after spatial/temporal alignment.

    def _temporal_smoothing(self, masks, target_T=None):
        """
        Apply temporal smoothing to SAM3 masks as in paper1231.

        Paper formula: M_gt[w,h] = f_2D(sum(M^{2t-1,w,h} + M^{2t,w,h}))

        This reduces jitter and local instability in single-frame masks from SAM3.

        Args:
            masks: (B, T, H, W)
            target_T: Target temporal dimension for I3D alignment (if None, keep T/2)

        Returns:
            smoothed_masks: (B, T_out, H', W') - Temporal length may change
        """
        B, T, H, W = masks.shape

        if T < 2:
            return masks

        # Sum adjacent frames: M^{2t-1} + M^{2t}
        # This reduces temporal dimension by half
        num_pairs = T // 2
        even_frames = masks[:, 0:num_pairs*2:2, :, :]  # M^0, M^2, ...
        odd_frames = masks[:, 1:num_pairs*2:2, :, :]   # M^1, M^3, ...

        # Sum adjacent pairs (paper1231 Eq.6)
        summed = even_frames + odd_frames  # (B, num_pairs, H, W)

        # Apply 2D average pooling (f_2D) as per paper
        # This is spatial downsampling step
        smoothed = F.avg_pool2d(summed, kernel_size=2, stride=2, padding=0)  # (B, num_pairs, H/2, W/2)

        # Normalize to [0, 1] range (since we summed two masks)
        smoothed = smoothed / 2.0

        # If target temporal dimension is specified, interpolate to match I3D features
        if target_T is not None and smoothed.size(1) != target_T:
            # Permute to (B, H, W, T) for interpolation, then permute back
            smoothed = smoothed.permute(0, 2, 3, 1)  # (B, H, W, T/2)
            smoothed = F.interpolate(
                smoothed.unsqueeze(1),  # Add channel dim: (B, 1, H, W, T/2)
                size=(smoothed.size(1), smoothed.size(2), target_T),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)  # (B, H, W, T)
            smoothed = smoothed.permute(0, 3, 1, 2)  # (B, T, H, W)

        return smoothed

    def forward(self, dynamic_features, masks, return_attention_map=False):
        """
        Apply SAM3 mask-guided attention to dynamic features.

        Args:
            dynamic_features: (B, C, T, H, W) - spatiotemporal features from I3D
            masks: (B, T_mask, H_mask, W_mask) - offline SAM3 masks [0,1]
            return_attention_map: Return final attention map for visualization

        Returns:
            masked_features: (B, C) - attention-ap-mplied features
            attention_map: (B, T, H, W) - optional, for visualization
            mask_loss: None (no supervision loss needed for offline masks)
        """
        B, C, T, H, W = dynamic_features.shape
        device = dynamic_features.device

        # ========================================================================
        # Step 1: Temporal smoothing (reduce SAM3 single-frame jitter)
        # ========================================================================
        if self.enable_temporal_smoothing:
            masks = self._temporal_smoothing(masks, target_T=T)
        # Shape after smoothing: (B, T, H', W') where H' and W' may be smaller

        # ========================================================================
        # Step 2: Spatial alignment with I3D feature map
        # ========================================================================
        # Upsample/Downsample masks to exactly match I3D feature resolution (T, H, W)
        # SAM3 masks: (B, T, H', W')
        # I3D features:  (B, C, T, H, W)

        masks_5d = masks.unsqueeze(1)  # (B, 1, T, H', W')
        masks_aligned = F.interpolate(
            masks_5d,
            size=(T, H, W),
            mode='trilinear',
            align_corners=False
        )  # (B, 1, T, H, W)

        # The aligned mask is our attention map A
        attention_map = masks_aligned.squeeze(1)  # (B, T, H, W)

        # ========================================================================
        # Step 3: Apply attention to features (paper1231 Eq.11)
        # F_dy = F_clip ⊙ (A + I)
        # ========================================================================
        # A + I: add identity to preserve background information
        # I is scalar 1.0, meaning all regions get at least 1x weight
        # A ∈ [0,1], so A+I ∈ [1,2]
        # Instrument regions (A≈1) get 2x weight
        # Background regions (A≈0) get 1x weight

        masked_features = dynamic_features * (attention_map.unsqueeze(1) + 1.0)
        # Shape: (B, C, T, H, W)

        # ========================================================================
        # Step 4: Global pooling to get compact representation
        # ========================================================================
        pooled_features = F.adaptive_avg_pool3d(masked_features, (1, 1, 1))  # (B, C, 1, 1, 1)
        pooled_features = pooled_features.flatten(1)  # (B, C)

        # ========================================================================
        # Return results
        # ========================================================================
        # Note: No mask_loss because masks are ground-truth, not predictions
        if return_attention_map:
            return pooled_features, attention_map, None
        else:
            return pooled_features, None, None


class AttentionVisualization(nn.Module):
    """
    Utility module for attention map visualization.
    """
    @staticmethod
    def visualize_attention(original_video, attention_map, save_path):
        """
        Visualize SAM3 mask attention overlayed on original frames.

        Args:
            original: (T, H, W, C) or (B, C, T, H, W)
            attention_map: (T, H, W) or (B, T, H, W)
            save_path: Path to save visualization
        """
        import cv2
        import numpy as np
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Handle different input formats
        if original_video.dim() == 4:  # (B, C, T, H, W)
            original = original_video[0]
            original = original.permute(1, 2, 3, 0)  # (T, H, W, C)
        elif original_video.dim() == 5:  # (B, C, T, H, W)
            original = original_video[0]
            original = original.permute(1, 2, 3, 0)  # (T, H, W, C)

        if attention_map.dim() == 3:  # (B, T, H, W)
            attention_map = attention_map[0]

        T, H, W, C = original.shape

        # Save each frame with attention overlay
        for t in range(T):
            frame = original[t].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)

            attention = attention_map[t].cpu().numpy()
            attention = (attention * 255).astype(np.uint8)

            # Convert to color
            if C == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            # Create heatmap
            heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_JR)

            # Blend original and heatmap
            blended = cv2.addWeighted(frame_bgr, 0.7, heatmap, 0.3, 0)

            cv2.imwrite(f"{save_path}/frame_{t:04d}.png", blended)

        print(f"Attention visualization saved to {save_path}")


if __name__ == '__main__':
    # Test mask-guided attention module
    print("Testing MaskGuidedAttention module...")

    # Create test inputs - Paper: 112x112 resolution
    B, C, T, H, W = 2, 832, 4, 7, 7
    dynamic_features = torch.randn(B, C, T, H, W)
    masks = torch.randint(0, 2, (B, 16, 112, 112)).float()

    # Initialize module
    mask_attention = MaskGuidedAttention(enable_temporal_smoothing=True)

    # Forward pass
    masked_features, attention_map, mask_loss = mask_attention(
        dynamic_features, masks, return_attention_map=True
    )

    print(f"\nInput shapes:")
    print(f"  Dynamic features: {dynamic_features.shape}")
    print(f"  Masks: {masks.shape}")

    print(f"\nOutput shapes:")
    print(f"  Masked features: {masked_features.shape}")
    print(f"  Attention map: {attention_map.shape}")
    print(f"  Mask loss: {mask_loss}")

    print("\n✓ MaskGuidedAttention test passed!")