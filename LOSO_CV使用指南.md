# LOSO-CV 训练脚本使用指南

## 简介

`train_loso_cv.py` 实现了 JIGSAWS 数据集标准的**留一受试者组交叉验证** (Leave-One-Subject-Group-Out Cross-Validation)。

## 数据集划分策略

JIGSAWS 数据集有 9 个受试者组 (A, B, C, D, E, F, G, H, I)，标准 LOSO-CV 使用其中 8 个 (A 组数据不完整，不使用):

### 每个折叠的划分

| 折叠 (Fold) | 测试集 | 验证集 | 训练集 |
|-------------|---------|---------|---------|
| B | B001-B005 | C001-C005 | D, E, F, G, H, I |
| C | C001-C005 | D001-D005 | B, E, F, G, H, I |
| D | D001-D005 | E001-E005 | B, C, F, G, H, I |
| E | E001-E005 | F001-F005 | B, C, D, G, H, I |
| F | F001-F005 | G001-G005 | B, C, D, E, H, I |
| G | G001-G005 | H001-H005 | B, C, D, E, F, I |
| H | H001-H005 | I001-I005 | B, C, D, E, F, G |
| I | I001-I005 | B001-B005 | C, D, E, F, G, H |

**关键特点**:
- 测试集和验证集来自不同受试者
- 同一受试者的所有视频都在同一集合中（无数据泄露）
- 测试集包含该受试者组的所有 3 个任务视频

## 使用方法

### 1. 运行完整的 8 折交叉验证

```bash
cd /home/thor/surgical_qa_model

python train_loso_cv.py \
    --data_root /home/thor/sam3/jigsawas_frames \
    --output_dir output_loso_cv \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --gpus 0
```

### 2. 仅运行单个折叠（用于快速测试）

```bash
# 只运行 B 折
python train_loso_cv.py \
    --data_root /home/thor/sam3/jigsawas_frames \
    --output_dir output_loso_cv \
    --fold B \
    --epochs 50 \
    --batch_size 8
```

### 3. 仅评估已有模型

```bash
python train_loso_cv.py \
    --data_root /home/thor/sam3/jigsawas_frames \
    --fold B \
    --evaluate \
    --checkpoint output_loso_cv/loso_B/checkpoints/best_model.pth
```

### 4. 调整超参数

```bash
python train_loso_cv.py \
    --data_root /home/thor/sam3/jigsawas_frames \
    --output_dir output_loso_cv \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --optimizer adamw \
    --early_stopping \
    --patience 20 \
    --gpus 0
```

## 输出结果

### 目录结构

```
output_loso_cv/
├── loso_B/                      # B 折
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_*.pth
│   └── logs/
│       ├── training_*.log
│       └── events.out.tfevents.*
├── loso_C/                      # C 折
├── ...
├── loso_I/                      # I 折
└── loso_cv_results.json          # 汇总结果
```

### 结果文件

`loso_cv_results.json` 包含:

```json
{
  "summary": {
    "mae_mean": 0.85,
    "mae_std": 0.12,
    "srcc_mean": 0.85,
    "srcc_std": 0.05,
    ...
  },
  "per_fold": [
    {
      "fold": "loso_B",
      "test_subject_group": "B",
      "test_metrics": {
        "mae": 0.82,
        "srcc": 0.87,
        ...
      },
      "splits": {
        "train_subjects": ["D", "E", "F", "G", "H", "I"],
        "val_subjects": ["C"],
        "test_subjects": ["B"]
      }
    },
    ...
  ]
}
```

## 与原始 train_bounded.py 的对比

| 特性 | train_bounded.py | train_loso_cv.py |
|------|-----------------|------------------|
| 划分方式 | 随机 clip 级别 (80/10/10) | LOSO-CV (受试者级别) |
| 数据泄露 | ❌ 是（同一视频 clips 分散） | ✅ 否 |
| 评估目标 | 训练/测试数据分布相似 | 新受试者泛化能力 |
| 折叠数 | 1 | 8 |
| 训练次数 | 1 | 8 |
| 计算成本 | 低 | 高 |
| 结果可比性 | 随机划分，不稳定 | JIGSAWS 标准做法 |
| 归一化 | ✅ Bounded | ✅ Bounded |

## 常见问题

### Q: 为什么 LOSO-CV 需要训练 8 次？

A: 每个折叠使用不同的受试者作为测试集，这是 JIGSAWS 论文的标准评估方式，可以全面评估模型对新人的泛化能力。

### Q: 能否减少折叠数？

A: 可以使用 `--fold` 参数运行单个折叠，但最终结果应该基于所有 8 折的平均值。

### Q: 训练时间太长怎么办？

A: 1) 使用 `--fold` 测试单个折叠；2) 减少 epochs；3) 增加 batch_size；4) 使用多 GPU。

### Q: 结果如何解读？

A: 主要关注 **SRCC (Spearman 相关系数)**，这是 JIGSAWS 评估技能评分的主要指标。

### Q: 如何与论文结果对比？

A: 使用相同的 LOSO-CV 划分和相同的评估指标 (MAE, SRCC, PCC)，可以直接对比。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | 必需 | JIGSAWS 帧数据根目录 (jigsawas_frames) |
| `--labels_root` | 自动推断 | JIGSAWS 标签文件根目录 (jigsawas/labels) |
| `--output_dir` | output_loso_cv | 输出目录 |
| `--fold` | all | 要运行的折叠 (B/C/D/E/F/G/H/I 或 all) |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--optimizer` | adam | 优化器 (adam/sgd/adamw) |
| `--weight_decay` | 0.0 | 权重衰减 |
| `--early_stopping` | True | 启用早停 |
| `--patience` | 15 | 早停耐心值 |
| `--gpus` | 0 | GPU ID |
| `--seed` | 42 | 随机种子 |

## 参考

- JIGSAWS 数据集: https://ieee-dataport.org/documents/suturingdataset
- JIGSAWS 论文: "JIGSAWS: A benchmark dataset for surgical activity assessment"
