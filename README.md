# Surgical Video Quality Assessment Model

A PyTorch implementation for surgical video skill assessment based on paper1231 framework.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
cd /root/autodl-tmp/surgical_qa_model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare your dataset (see Dataset Preparation section)
#    - Place videos in datasets/videos/
#    - Place masks in datasets/masks/
#    - Create datasets/annotations.json

# 4. Update config file: edit configs/default.yaml
#    Set data_root to your dataset path, e.g., data_root: /root/autodl-tmp/datasets

# 5. Start training
python train.py --config configs/default.yaml --gpus 0

# 6. Monitor training with TensorBoard
tensorboard --logdir=logs
```

---

## 📖 Overview

This project implements a surgical video quality assessment model that evaluates surgical skill levels from video clips. The model architecture is inspired by paper1231 but adapted for single-video scoring (without contrastive learning).

### Key Differences from paper1231

| Aspect | paper1231 | This Model |
|---------|-----------|-------------|
| Framework | Contrastive learning with query/reference video pairs | Single video direct regression |
| Cross-attention | Yes (between query and reference) | No |
| Score prediction | `y_q = y_r + CR(E(V_q), E(V_r))` | `y = R(Concat(F_static, F_dynamic))` |
| Training objective | Learn relative score differences | Learn absolute scores |

---

## 🏗️ Architecture

```
Input Video (B, C, T, H, W)
    ├─→ ResNet-34 (Static Feature Extractor) ──→ Static Features A
    │   ├─ ResNet Backbone
    │   └─ Multi-scale Downsampling (s=2,4,8)
    │
    ├─→ I3D (Dynamic Feature Extractor) ──→ Dynamic Features C
    │   └─ Mixed 3D Convolutions
    │
    └─→ SAM3 Masks (Offline) ──→ Instrument Masks: B
        └─→ Read from pre-computed files

Mask-Guided Attention Module
    ├─ Input: B (masks) and C (dynamic features)
    ├─ Generate attention from mask features
    ├─ Fuse with learned attention
    └─ Apply to features → Masked Dynamic Features D

Feature Fusion & Regression
    ├─ Concat(A, D) ──→ Fused Features
    └─ FC Layers ──→ Score y
```

### Model Components

**1. Static Feature Extractor (ResNet-34)**
- ResNet-34 backbone with 4 layers
- Multi-scale downsampling (strides: 2, 4, 8)
- Frame sampling strategy: middle, average, first, last
- Captures tissue state and surgical field clarity

**2. Dynamic Feature Extractor (I3D)**
- Inception-3D backbone
- Mixed 3D convolutions for expanded temporal receptive field
- Short-term clip processing with overlapping windows
- Extracts spatiotemporal features for instrument dynamics

**3. Mask-Guided Attention**
- Generates mean and max attention from mask features
- Learnable aggregation function
- Temporal smoothing to reduce mask jitter
- Element-wise multiplication: `F_dy = F_clip * (A + I)`

**4. Fusion Regressor**
- Input: `[static_dim, dynamic_dim] = [512, 832]`
- Hidden layers: `[1024, 512, 256, 128]`
- Output: `[1]` (quality score)

---

## 💻 Installation

### Prerequisites

- Python 3.7+
- CUDA 10.2+ (for GPU support)
- PyTorch 1.9+

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually
pip install torch torchvision
pip install numpy opencv-python
pip install pyyaml tqdm
pip install scipy tensorboard
```

### Step 2: (Optional) Install SAM3

If you need to generate masks online using SAM3:

```bash
# Follow official SAM3 installation
# Located at: /root/autodl-tmp/sam3
```

---

## 📁 Dataset Preparation

### Required Directory Structure

```
data_root/              # Path to set in configs/default.yaml
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── masks/
│   ├── video_001/
│   │   ├── frame_0000_mask.png
│   │   ├── frame_0001_mask.png
│   │   └── ...
│   ├── video_002/
│   │   └── ...
│   └── ...
└── annotations.json
```

### 1. Video Files

Place your surgical videos in `data_root/videos/`:
```bash
data_root/videos/
├── video_001.mp4
├── video_002.mp4
└── ...
```

### 2. Mask Files

#### Option A: Use Pre-computed Masks (Recommended)

Generate surgical instrument masks using SAM3 and save them as PNG files:

```python
# Example: Generate masks using SAM3
from sam3 import build_sam3_video_model
import cv2
import os

model = build_sam3_video_model(device='cuda')
inference_state = model.init_state('path/to/video.mp4')

# Add point prompts for instrument detection
model.add_points_prompts(
    inference_state,
    frame_idx=0,
    point_coords=[[x, y]],  # Positive point on instrument
    point_labels=[1]
)

# Propagate masks through video
masks = model.propagate_in_video(inference_state)

# Save masks
output_dir = f'masks/video_001'
os.makedirs(output_dir, exist_ok=True)
for frame_idx, mask in enumerate(masks):
    cv2.imwrite(f'{output_dir}/frame_{frame_idx:04d}_mask.png', mask)
```

#### Option B: Use NumPy Masks

Alternatively, save masks as a single NumPy array:
```python
import numpy as np
# masks shape: (num_frames, H, W)
np.save('data_root/masks/video_001_masks.npy', masks)
```

### 3. Annotation File

Create `data_root/annotations.json` with ground truth scores:

```json
{
    "video_001": {
        "score": 8.5,
        "duration": 120,
        "surgeon_id": "S001"
    },
    "video_002": {
        "score": 7.2,
        "duration": 115,
        "surgeon_id": "S002"
    }
}
```

**Note**: At minimum, each video needs a `score` field. Other fields are optional.

### 4. Create Sample Dataset (For Testing)

The code includes a helper function to create a dummy dataset:

```python
from utils.data_loader import create_sample_annotations
create_sample_annotations('/path/to/dataset', num_samples=100)
```

---

## 🏋️ Training

### Basic Training

```bash
# Use default config
python train.py --config configs/default.yaml --gpus 0

# Or override specific parameters
python train.py \
    --config configs/default.yaml \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.0001
```

### Resume Training

```bash
# Resume from checkpoint
python train.py \
    --config configs/default.yaml \
    --resume checkpoints/checkpoint_epoch_50.pth \
    --gpus 0
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir=logs

# Open in browser: http://localhost:6006
```

### Training Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|----------|
| `--config` | Path to config file | `configs/default.yaml` |
| `--data_root` | Override dataset path | from config |
| `--output_dir` | Output directory for logs | `output` |
| `--batch_size` | Batch size | 8 |
| `--epochs` | Number of epochs | 100 |
| `--learning_rate` | Learning rate | 0.0001 |
| `--optimizer` | Optimizer (sgd, adam, adamw) | adam |
| `--freeze_backbone` | Freeze backbone weights | True |
| `--use_amp` | Use mixed precision training | False |
| `--gpus` | GPU IDs (comma-separated) | 0 |
| `--resume` | Path to checkpoint | None |
| `--evaluate` | Evaluation mode only | False |

---

## 📊 Evaluation

```bash
# Evaluate on test set
python train.py \
    --config configs/default.yaml \
    --evaluate \
    --gpus 0
```

### Evaluation Metrics

Following paper1231:

1. **Mean Absolute Error (MAE)**
   ```
   MAE = (1/n) * sum(|y_pred - y_gt|)
   ```

2. **Spearman Rank Correlation Coefficient (SRCC)**
   ```
   SRCC = rank correlation between predictions and ground truth
   ```

3. **Pearson Correlation Coefficient (PCC)**
   ```
   PCC = linear correlation between predictions and ground truth
   ```

4. **Normalized MAE (NMAE)**
   ```
   NMAE = MAE / (score_range) * 100
   ```

---

## 📂 Project Structure

```
surgical_qa_model/
├── models/                          # Model architectures
│   ├── __init__.py
│   ├── static_feature_extractor.py   # ResNet-34 based static features
│   ├── dynamic_feature_extractor.py  # I3D based dynamic features
│   ├── mask_guided_attention.py     # Mask-guided attention module
│   └── surgical_qa_model.py        # Main model class
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── data_loader.py              # Dataset and dataloader
│   ├── mask_loader.py              # Mask loading utilities
│   ├── metrics.py                  # Evaluation metrics
│   └── training.py                 # Training loops and utilities
│
├── configs/                         # Configuration files
│   └── default.yaml                # Default configuration
│
├── datasets/                        # Dataset storage (create this)
│   ├── videos/                    # Surgical video files
│   ├── masks/                     # Pre-computed mask files
│   └── annotations.json           # Ground truth scores
│
├── checkpoints/                     # Saved model checkpoints
├── logs/                           # Training logs and TensorBoard
├── requirements.txt                 # Python dependencies
├── README.md                      # This file
├── ARCHITECTURE.md               # Detailed architecture documentation
└── train.py                        # Main training script
```

---

## ⚙️ Configuration

The `configs/default.yaml` file contains all hyperparameters:

### Data Settings
```yaml
data_root: /path/to/dataset    # Must be updated!
batch_size: 8
num_workers: 4
```

### Video Preprocessing
```yaml
clip_length: 16    # Frames per clip
clip_stride: 10    # Stride for clip extraction
spatial_size: 224    # Resize to (H, W)
```

### Model Settings
```yaml
static_dim: 512    # ResNet feature dimension
dynamic_dim: 832    # I3D feature dimension
freeze_backbone: true    # Freeze backbone weights
use_pretrained: true    # Use ImageNet pretrained
use_mask_loss: true    # Use mask supervision loss
```

### Training Settings
```yaml
epochs: 100
learning_rate: 0.0001
weight_decay: 0.0001
optimizer: adam
use_amp: false    # Mixed precision training
clip_grad_norm: 1.0    # Gradient clipping
```

### Learning Rate Schedule
```yaml
lr_scheduler: cosine    # Options: cosine, step, plateau
lr_milestones: [30, 60, 90]    # For step scheduler
lr_gamma: 0.1    # LR decay factor
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
```bash
# Reduce batch size
python train.py --config configs/default.yaml --batch_size 4

# Enable mixed precision
python train.py --config configs/default.yaml --use_amp

# Freeze backbone
# Edit configs/default.yaml: freeze_backbone: true
```

#### 2. Data Loading Issues

**Symptoms**: `FileNotFoundError` or mask loading errors

**Solutions**:
- Verify `data_root` is set correctly in `configs/default.yaml`
- Check that all videos in `videos/` have corresponding entries in `annotations.json`
- Ensure mask files exist in `masks/video_XXX/` for each video
- Verify mask dimensions match video frame dimensions

#### 3. SAM3 Mask Generation Issues

**Symptoms**: Mask generation fails or produces incorrect masks

**Solutions**:
- Ensure SAM3 is installed from the correct path: `/root/autodl-tmp/sam3`
- Check CUDA availability: `torch.cuda.is_available()`
- Verify point prompts are correctly positioned on instrument regions

#### 4. Poor Model Convergence

**Symptoms**: Training loss doesn't decrease or fluctuates wildly

**Solutions**:
```bash
# Lower learning rate
python train.py --config configs/default.yaml --learning_rate 1e-5

# Enable gradient clipping
python train.py --config configs/default.yaml --clip_grad_norm 1.0

# Use different optimizer
python train.py --config configs/default.yaml --optimizer adamw
```

#### 5. Validation Performance Issues

**Symptoms**: Validation loss is much higher than training loss

**Solutions**:
- Check dataset split for train/val/test
- Ensure `is_train` flag is correctly set (fixed in recent version)
- Verify no data leakage between splits

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_name_202X_surgical,
  title={Surgical Video Quality Assessment via Static and Dynamic Feature Fusion},
  author={Your Name},
  journal={ArXiv preprint},
  year={202X}
}
```

Also cite the original paper1231:

```bibtex
@article{paper1231_202X_fine_grained,
  title={Fine-Grained Contrastive Learning for Robotic Surgery Skill Assessment Through Joint Instrument-Tissue Modeling},
  author={Paper1231 Authors},
  journal={Original Paper1231},
  year={202X}
}
```

---

## 📄 License

This project is for research and educational purposes.

## 📧 Support

For questions or issues, please check the [Troubleshooting](#troubleshooting) section or open an issue on the project repository.
