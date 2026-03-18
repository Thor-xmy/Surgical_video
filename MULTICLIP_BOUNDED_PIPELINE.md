# Multi-Clip Bounded Surgical QA Model Pipeline 完整说明文档

## 1. 概述

这是一个完整的多clip手术视频质量评估模型pipeline，采用"方案A"（Per-clip Fusion）策略，并使用有界回归（Bounded Regression）确保输出在指定范围内。

### 核心特点

1. **Video-level处理**：输入整个视频，而不是分散的clip
2. **Multi-clip特征提取**：从多个重叠clip中提取静态和动态特征
3. **Per-clip融合**：先在每个clip级别融合静态和动态特征
4. **时序扁平化**：将所有clip的特征扁平化为时序序列
5. **有界回归**：使用Sigmoid激活确保输出在[0, 1]范围内
6. **分数归一化**：训练时归一化标签，推理时反归一化回原始范围

## 2. Pipeline架构

```
Video Input (B, 3, T, H, W)
    ↓
Split into N overlapping clips
    ├─→ Static Extractor: 每个clip采样关键帧 → ResNet → (B, N, 512)
    └─→ Dynamic Extractor: 每个clip通过I3D → (B, N, 1024)
    ↓
Per-clip Fusion: [(512+1024), ..., (512+1024)] → (B, N, 1536)
    ↓
Temporal Flattening: (B, N, 1536) → (B, N*1536)
    ↓
Bounded Regressor (Sigmoid) → (B, 1) ∈ [0, 1]
    ↓
Denormalize: [0, 1] → [6, 30] (JIGSAWS) 或 [1, 10] (论文)
```

## 3. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_length` | 16 | 每个clip的帧数 |
| `clip_stride` | 10 | clip之间的步长（产生6帧重叠） |
| `max_clips` | None | 最大clip数量（None=自动计算） |
| `static_dim` | 512 | ResNet输出维度 |
| `dynamic_dim` | 1024 | I3D输出维度 |
| `score_min` | 6.0 | 原始分数最小值（JIGSAWS） |
| `score_max` | 30.0 | 原始分数最大值（JIGSAWS） |
| `keyframe_strategy` | 'middle' | 静态特征关键帧采样策略 |

### Clip计算示例

对于100帧的视频：
- `clip_length=16`, `clip_stride=10`
- Clip数量: `(100 - 16) // 10 + 1 = 10` clips
- 重叠: 每个clip之间有6帧重叠
- 总特征维度: `10 * (512 + 1024) = 15360`

## 4. 文件结构

### 核心模型文件

```
models/
├── static_feature_extractor_multiclip.py      # 多clip静态特征提取器
├── dynamic_feature_extractor_multiclip.py     # 多clip动态特征提取器
├── fusion_regressor_multiclip_bounded.py     # 多clip有界融合回归器
└── surgical_qa_model_multiclip_bounded.py   # 完整模型
```

### 数据加载

```
utils/
└── data_loader_video_level.py                # 视频级别数据加载器
```

### 训练脚本

```
train_multiclip_bounded.py                   # 训练脚本
```

### 测试脚本

```
test_final_multiclip_bounded.py               # 完整pipeline测试
```

## 5. 详细组件说明

### 5.1 VideoLevelDataset (`utils/data_loader_video_level.py`)

**功能**：视频级别数据加载器，返回整个视频和归一化的标签

**关键方法**：
```python
def __getitem__(self, idx):
    video = load_video(...)              # (T, H, W, C)
    score_gt = annotation['score']        # 原始分数 [6, 30]
    score_normalized = self._normalize_score(score_gt)  # [0, 1]
    return {
        'video': video,                # 整个视频
        'score': score_normalized,      # 归一化分数
        'video_id': video_id
    }
```

**输出**：
- `video`: (T, H, W, C) - 整个视频
- `score`: float - 归一化到[0, 1]的分数

### 5.2 StaticFeatureMultiClip (`models/static_feature_extractor_multiclip.py`)

**功能**：从多个clip的每个clip提取静态特征（ResNet）

**关键方法**：
```python
def extract_multiclip_features(self, video, clip_length=16, clip_stride=10):
    """
    1. 将视频分成多个重叠的clip
    2. 对每个clip采样一个关键帧（默认中间帧）
    3. 通过ResNet-34提取特征
    4. 返回每个clip的特征

    Args:
        video: (B, C, T, H, W)

    Returns:
        per_clip_features: (B, num_clips, 512)
        num_clips: clip数量
    """
```

**关键帧采样策略**：
- `'middle'`: 使用clip的中间帧
- `'first'`: 使用clip的第一帧
- `'last'`: 使用clip的最后一帧
- `'random'`: 随机采样（用于数据增强）

### 5.3 DynamicFeatureMultiClip (`models/dynamic_feature_extractor_multiclip.py`)

**功能**：从多个clip的每个clip提取动态特征（I3D）

**关键方法**：
```python
def extract_multiclip_features(self, video):
    """
    1. 将视频分成多个重叠的clip
    2. 对每个clip通过I3D提取时序特征
    3. 返回每个clip的特征

    Args:
        video: (B, C, T, H, W)

    Returns:
        features_per_clip: (B, num_clips, 1024)
        num_clips: clip数量
    """
```

**I3D架构**：
- 使用Inception I3D作为backbone
- 提取到Mixed_5c层的特征
- 可选Mixed 3D卷积层
- 自适应平均池化到单一特征向量

### 5.4 BoundedFusionRegressorMultiClip (`models/fusion_regressor_multiclip_bounded.py`)

**功能**：有界回归器，使用Sigmoid确保输出在[0, 1]范围内

**架构**：
```python
Input (B, N*1536)
    ↓
Linear → LayerNorm → ReLU → Dropout
    ↓
[重复多层]
    ↓
Linear(64, 1) → Sigmoid
    ↓
Output (B, 1) ∈ [0, 1]
```

**关键点**：
- 最后一层使用Sigmoid激活
- 确保输出严格在(0, 1)之间
- 配合归一化标签使用MSE Loss训练

### 5.5 SurgicalQAModelMultiClipBounded (`models/surgical_qa_model_multiclip_bounded.py`)

**功能**：完整的手术视频质量评估模型

**Forward Pass流程**：
```python
def forward(self, video, return_features=False):
    # 1. 提取多clip静态特征
    static_per_clip: (B, N, 512) = self.static_extractor.extract_multiclip_features(video)

    # 2. 提取多clip动态特征
    dynamic_per_clip: (B, N, 1024) = self.dynamic_extractor.extract_multiclip_features(video)

    # 3. Per-clip融合
    fused_per_clip: (B, N, 1536) = cat([static_per_clip, dynamic_per_clip], dim=-1)

    # 4. 时序扁平化
    temporal_features: (B, N*1536) = fused_per_clip.reshape(B, -1)

    # 5. 有界回归
    score_normalized: (B, 1) ∈ [0, 1] = self.fusion_regressor(temporal_features)

    return score_normalized
```

**分数反归一化**：
```python
def denormalize_score(self, score_normalized, target_min=None, target_max=None):
    """
    将[0, 1]的分数映射回原始范围

    用例1: JIGSAWS原始范围 [6, 30]
        score_6to30 = model.denormalize_score(score_norm)

    用例2: 论文分数范围 [1, 10]
        score_1to10 = model.denormalize_score(score_norm, target_min=1.0, target_max=10.0)
    """
```

## 6. 训练流程

### 6.1 数据准备

```python
from utils.data_loader_video_level import create_video_level_dataloader

train_loader = create_video_level_dataloader(
    data_root='/path/to/data',
    batch_size=4,
    spatial_size=224,
    clip_length=16,
    clip_stride=10,
    score_min=6.0,      # JIGSAWS原始范围
    score_max=30.0,
    is_train=True
)
```

### 6.2 模型初始化

```python
from models.surgical_qa_model_multiclip_bounded import SurgicalQAModelMultiClipBounded

config = {
    'static_dim': 512,
    'dynamic_dim': 1024,
    'clip_length': 16,
    'clip_stride': 10,
    'max_clips': None,
    'use_pretrained': True,
    'freeze_backbone': True,
    'score_min': 6.0,
    'score_max': 30.0,
    'keyframe_strategy': 'middle'
}

model = SurgicalQAModelMultiClipBounded(config)
model = model.to(device)
```

### 6.3 训练循环

```python
for epoch in range(epochs):
    for batch in train_loader:
        video = batch['video'].to(device)      # (B, T, H, W, C)
        score_gt = batch['score'].to(device)    # (B,) ∈ [0, 1]

        # 前向传播
        score_pred = model(video)              # (B, 1) ∈ [0, 1]

        # 计算损失
        loss, loss_dict = model.compute_loss(score_pred, score_gt)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 6.4 推理和评估

```python
model.eval()
with torch.no_grad():
    score_pred = model(video)                  # (B, 1) ∈ [0, 1]

    # 反归一化到JIGSAWS原始范围
    score_6to30 = model.denormalize_score(score_pred)

    # 或反归一化到论文分数范围
    score_1to10 = model.denormalize_score(
        score_pred,
        target_min=1.0,
        target_max=10.0
    )
```

## 7. 评估指标

训练时使用归一化分数[0, 1]计算指标：
- MSE Loss
- Spearman相关系数
- L2距离
- RL2 (Relative L2)

评估时使用反归一化分数：
- JIGSAWS原始分数[6, 30]用于与原始标注对比
- 论文分数[1, 10]用于与其他论文对比

## 8. 与原版的主要区别

| 方面 | 原版 | 多clip有界版 |
|------|------|-------------|
| 数据加载 | Clip-level，返回分散的clip | Video-level，返回整个视频 |
| 特征提取 | 单个clip | 多个clip，保留时序信息 |
| 特征融合 | 先融合静态和动态特征 | 每个clip先融合，然后时序拼接 |
| 时序建模 | 无（单clip） | 通过时序扁平化 |
| 分数范围 | 原始范围 [6, 30] | 归一化到 [0, 1] 然后反归一化 |
| 回归器 | 无界回归 | 有界回归（Sigmoid） |
| 输出 | 原始分数 | 归一化分数，需反归一化 |

## 9. 使用示例

### 9.1 快速测试

```bash
python test_final_multiclip_bounded.py
```

### 9.2 训练

```bash
python train_multiclip_bounded.py \
    --data_root /path/to/data \
    --output_dir output_multiclip_bounded \
    --batch_size 4 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --score_min 6.0 \
    --score_max 30.0
```

### 9.3 推理

```python
import torch
from models.surgical_qa_model_multiclip_bounded import build_model_multiclip_bounded

# 加载模型
model = build_model_multiclip_bounded('configs/multiclip_bounded.yaml')
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载视频
video = load_video('/path/to/video.mp4')  # (T, H, W, C)
video = torch.from_numpy(video).permute(3, 0, 1, 2)  # (C, T, H, W)
video = video.unsqueeze(0)  # (1, C, T, H, W)

# 推理
with torch.no_grad():
    score_norm = model(video)
    score = model.denormalize_score(score_norm)

print(f"Predicted score: {score.item():.2f}")
```

## 10. 配置文件示例

`configs/multiclip_bounded.yaml`:

```yaml
# Data
data_root: '/path/to/data'
batch_size: 4
num_workers: 4

# Model Architecture
static_dim: 512
dynamic_dim: 1024
clip_length: 16
clip_stride: 10
max_clips: null

# Score Normalization
score_min: 6.0
score_max: 30.0

# Training
epochs: 100
learning_rate: 0.0001
weight_decay: 0.00001
freeze_backbone: true
use_pretrained: true

# Regressor
regressor_hidden_dims: [1024, 512, 256, 128]

# Keyframe sampling
keyframe_strategy: 'middle'
```

## 11. 注意事项

1. **内存使用**：多clip处理会增加内存使用，建议使用较小的batch_size
2. **GPU要求**：建议至少12GB显存用于batch_size=4的训练
3. **预处理**：确保视频帧统一缩放到224x224
4. **权重冻结**：初始训练时建议冻结backbone，后期可微调
5. **Clip配置**：clip_length和clip_stride会影响性能和计算量
6. **归一化**：训练和推理时都使用归一化，评估时反归一化

## 12. 性能优化建议

1. **减少clip数量**：设置`max_clips`参数限制处理的clip数
2. **增大步长**：增加`clip_stride`减少clip数量和重叠
3. **冻结backbone**：初始训练时冻结ResNet和I3D backbone
4. **混合精度训练**：使用FP16减少显存和加速
5. **梯度累积**：使用小的batch_size配合梯度累积
