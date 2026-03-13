# Architecture Design: Surgical Quality Assessment Model

## 1. Overall Framework Comparison

### paper1231 (Contrastive Learning Framework)
```
Query Video V_q                    Reference Video V_r
    |                                       |
    ├─> Static Features F_sta^q          ─> Static Features F_sta^r
    ├─> Dynamic Features F_dy^q           ─> Dynamic Features F_dy^r
    └─> Masks B_q                      ─> Masks B_r
                                             |
                    Cross-Attention Decoder ─────────────────┘
                            |
                    Diff Features (F_sta^diff, F_dy^diff)
                            |
                    Contrastive Regressor ──────────────────┘
                            |
                    Δs (relative score difference)
                            |
                    Final Score: y_q = y_r + Δs
```

### This Model (Single-Video Direct Regression)
```
Input Video X
    |
    ├─→ ResNet-34 ──────────────────┐
    │    (Static Feature Extractor)    │
    │                             Static Features A
    │                                    │
    ├─→ SAM3 Masks (Offline) ──────────┤
    │    (Read from disk)               │
    │                             Instrument Masks B ─→ Feature Fusion (Concatenation)
    │                                    │     │
    │                                    │     ↓
    ├─→ I3D ───────────────────────────┐    Fused Features
    │    (Dynamic Feature Extractor)      │     │
    │                             Dynamic C │     │
    │                                    │     │
    └─────────────────────────────────┐     │     │
              Mask-Guided Attention │     │     │
                      B acts on C → D │     │
                                       │     │
                                       │     │
                                       └─────┘
                                            ↓
                                      Final Score y
```

## 2. Module Details

### Module 1: Static Feature Extractor (ResNet-34)

**Purpose**: Extract static global features capturing tissue state and surgical field clarity.

**Implementation**:
```
Input: Video X (B, C, T, H, W)
    ↓
Frame Sampling Strategy:
    - Middle frame (default)
    - Average of multiple frames
    - First/Last frame
    ↓
ResNet-34 Backbone:
    - Conv1 (7x7, stride=2)
    - Layer1 (3 blocks)
    - Layer2 (4 blocks)
    - Layer3 (6 blocks)
    - Layer4 (3 blocks)
    ↓
Multi-scale Downsampling:
    - Scale 2: stride=2  → capture macroscopic variations
    - Scale 4: stride=4  → medium context
    - Scale 8: stride=8  → capture subtle tissue variations
    ↓
Concat & Global Pool: (512 → 384 channels)
    ↓
Projection: 384 → 512
    ↓
Output: Static Features A (B, 512)
```

**Key Formulas** (from paper1231):
```
F_global = R(X_i)  # ResNet backbone
F_mul = D_mul(F_global), s ∈ {2, 4, 8}
F_sta = Concat(D_mul^2(F_global), D_mul^4(F_global), D_mul^8(F_global))
```

### Module 2: Dynamic Feature Extractor (I3D)

**Purpose**: Extract spatiotemporal features for instrument manipulation dynamics.

**Implementation**:
```
Input: Video X (B, C, T, H, W)
    ↓
I3D Backbone:
    - Stem: 7x7 → 64 channels
    - Mixed_3b: 192 → 256 channels
    - Mixed_3c: 256 → 480 channels
    - Mixed_4b to Mixed_4f: 480 → 528 → 832 channels
    - Mixed_5b: 832 → 832 channels
    - Mixed_5c: 832 → 832 channels
    ↓
Mixed 3D Convolutions (optional for temporal receptive field):
    - Temporal convolutions
    - Spatial convolutions
    - Fusion
    ↓
Spatial Max Pooling (as in paper1231):
F_clip(X_i) = maxpool(Conv_mix(X_i))
    ↓
Global Temporal-Spatial Pooling:
    ↓
Output: Dynamic Features C (B, 832)
```

**Key Formula** (from paper1231):
```
F_clip(X_i) = maxpool(Conv_mix(X_i))
```

### Module 3: Mask-Guided Attention

**Purpose**: Focus network attention on surgical instrument regions using masks.

**Implementation**:
```
Input:
    - Dynamic Features C: (B, 832, T, H, W)
    - Masks B: (B, T, H_mask, W_mask)
    ↓
Temporal Smoothing (as in paper1231):
    M_gt[w,h] = f_2D(∑(M^(2t-1,w,h} + M^(2t,w,h)))
    ↓
Generate Attention from Masks:
    - Mean pooling: A_mean = (1/C) * ∑F_ms
    - Max pooling: A_max = max(F_ms)
    - Aggregation: A = σ(f_agg(A_mean, A_max))
    ↓
Projection to Feature Dimensions:
    - Conv3D: (B, 1, T, H, W) → (B, 1, T, H, W)
    ↓
Fusion with Learned Attention:
    - α * mask_attention + (1-α) * feature_attention
    ↓
Apply Attention (Element-wise multiplication):
    F_dy = F_clip * (A + I)  # I is identity matrix
    ↓
Global Pooling:
    ↓
Output: Masked Dynamic Features D (B, 832)
```

**Key Formulas** (from paper1231):
```
A_mean^(t,w,h) = (1/C_2) * ∑(F_ms(X_i))_c^(t,w,h)
A_max^(t,w,h) = max_c(F_ms(X_i))_c^(t,w,h)
A = σ(f_agg(A_mean, A_max))
Loss_mask = (1/(T*W*H)) * ∑∑∑||A - M_gt||^2
F_dy(X_i) = F_clip(X_i) ⊗ (A + I)
```

### Module 4: Fusion Regressor

**Purpose**: Concatenate static and dynamic features, regress to quality score.

**Implementation**:
```
Input:
    - Static Features A: (B, 512)
    - Masked Dynamic Features D: (B, 832)
    ↓
Concatenation:
    Fused = [A || D]  → (B, 1344)
    ↓
MLP Regressor:
    ├─ FC1: 1344 → 1024 (LayerNorm + ReLU + Dropout)
    ├─ FC2: 1024 → 512 (LayerNorm + ReLU + Dropout)
    ├─ FC3: 512 → 256 (LayerNorm + ReLU + Dropout)
    ├─ FC4: 256 → 128 (LayerNorm + ReLU + Dropout)
    ├─ FC5: 128 → 64 (LayerNorm + ReLU + Dropout)
    └─ Output: 64 → 1
    ↓
Output: Quality Score y (B, 1)
```

**Loss Function**:
```
Loss_score = ||ŷs - s||^2
```
Where:
- ŷs: Predicted score
- s: Ground truth score

**Optional Loss** (Mask Supervision):
```
Loss_total = Loss_score + λ_mask * Loss_mask
```

## 3. Data Flow Summary

| Step | Input | Module | Output | Shape |
|------|--------|--------|--------|
| 1 | Video X (B,3,T,H,W) | ResNet-34 | A (B,512) |
| 2 | Video X (B,3,T,H,W) | I3D | C (B,832,T',H',W') |
| 3 | Disk files | Mask Loader | B (B,T,H_mask,W_mask) |
| 4 | C, B | Mask-Guided Attention | D (B,832) |
| 5 | A, D | Concat | Fused (B,1344) |
| 6 | Fused | Fusion Regressor | y (B,1) |

## 4. Training Strategy

### Stage 1: Feature Extraction Only (Freeze Backbones)
```
freeze_backbone: true
trainable_params: [fusion_regressor]
```

### Stage 2: Partial Unfreeze (Fine-tune Upper Layers)
```
freeze_backbone: true
unfreeze_layers: [layer4]  # Only last ResNet layer
unfreeze_i3d_layers: [Mixed_5b, Mixed_5c]  # Last I3D layers
```

### Stage 3: Full Training (All Parameters)
```
freeze_backbone: false
trainable_params: [all]
```

## 5. Key Differences from paper1231

| Component | paper1231 | This Model | Rationale |
|-----------|-----------|------------|-----------|
| Video Input | Query + Reference pairs | Single video | Simplified for direct scoring |
| Feature Extraction | Same | Same | Uses same backbones |
| Mask Usage | Online (SAM2) | Offline (pre-computed) | Reduces complexity |
| Attention | Yes | Yes | Same mask-guided mechanism |
| Cross-Attention | Yes (query ↔ reference) | No | Not needed for single video |
| Score Prediction | Relative (Δs) | Absolute (y) | Direct regression |
| Loss Function | Relative differences | Absolute errors | MSE loss |
| Training Paradigm | Contrastive pairs | End-to-end regression | Simpler training |

## 6. Expected Performance

Based on paper1231 results and architectural similarities:

| Metric | Expected Range | paper1231 Baseline |
|--------|---------------|-------------------|
| MAE | 0.5 - 1.5 | ~1.0 - 2.0 (relative) |
| SRCC | 0.7 - 0.9 | 0.6 - 0.8 |
| PCC | 0.75 - 0.92 | 0.7 - 0.85 |
