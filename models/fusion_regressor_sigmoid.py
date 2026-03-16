"""
Bounded Fusion Regressor with Sigmoid Output

修改版的 FusionRegressor，添加 Sigmoid 激活函数确保输出在固定范围内

配合归一化标签使用：
- 输入标签范围：[0, 1]（归一化后的分数）
- 输出分数范围：[0, 1]（模型预测）
- 训练使用 MSE Loss
- 推理时反归一化回原始分数范围 [1, 10]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedFusionRegressor(nn.Module):
    """
    带边界的特征融合和分数回归模块

    使用 Sigmoid 确保输出在 [0, 1] 范围内

    配合归一化标签使用：
    - 归一化标签：[0, 1]
    - 模型输出：[0, 1]
    - 训练损失：MSE
    - 推理：反归一化到 [1, 10] 或 [1, 30]
    """
    def __init__(self, static_dim, dynamic_dim,
                 hidden_dims=[102440, 512, 256, 128]):
        """
        Args:
            static_dim: 静态特征维度
            dynamic_dim: 动态特征维度
            hidden_dims: 隐藏层维度
        """
        super().__init__()

        input_dim = static_dim + dynamic_dim

        # 构建 MLP 回归器（最后一层使用 Sigmoid）
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # 最终输出层：Linear → Sigmoid
        # Sigmoid 确保输出严格在 (0, 1) 之间
        layers.extend([
            nn.Linear(prev_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # ← 关键：添加 Sigmoid 激活函数
        ])

        self.regressor = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, static_features, dynamic_features):
        """
        融合特征并回归到分数

        Args:
            static_features: (B, static_dim)
            dynamic_features: (B, dynamic_dim)

        Returns:
            score_normalized: (B, 1) - 归一化后的分数，范围 [0, 1]
        """
        # 拼接特征
        fused = torch.cat([static_features, dynamic_features], dim=1)  # (B, static_dim + dynamic_dim)

        # 回归到分数（Sigmoid 输出 [0, 1]）
        score_normalized = self.regressor(fused)  # (B, 1) ∈ [0, 1]

        return score_normalized

    def denormalize_score(self, score_normalized,
                        score_min=1.0,
                        score_max=10.0,
                        target_min=0.0,
                        target_max=1.0):
        """
        反归一化：将 [0, 1] 的预测转换回原始分数范围

        Args:
            score_normalized: (B,) 或 (B, 1) - 归一化后的预测，范围 [0, 1]
            score_min: 原始分数最小值（如 JIGSAWS GRS: 1）
            score_max: 原始分数最大值（如 JIGSAWS GRS: 30）
            target_min: 归一化目标最小值（应为 0）
            target_max: 归一化目标最大值（应为 1）

        Returns:
            score_original: 与输入同形状，范围 [score_min, score_max]
        """
        # 确保 score_normalized 是一维
        if score_normalized.dim() == 2:
            score_normalized = score_normalized.squeeze(-1)

        # 反归一化公式：(norm - target_min) / (target_max - target_min) * (max - min) + min
        score_range = score_max - score_min
        target_range = target_max - target_min

        score_original = (score_normalized - target_min) / target_range * score_range + score_min

        return score_original


if __name__ == '__main__':
    # 测试有界回归器
    print("Testing BoundedFusionRegressor...")

    # 创建回归器
    regressor = BoundedFusionRegressor(512, 1024)
    regressor.eval()

    # 测试输入
    B = 2
    static_features = torch.randn(B, 512)
    dynamic_features = torch.randn(B, 1024)

    # 前向传播
    score_normalized = regressor(static_features, dynamic_features)

    print(f"\nInput shapes:")
    print(f"  static features: {static_features.shape}")
    print(f"  dynamic features: {dynamic_features.shape}")

    print(f"\nOutput (normalized):")
    print(f"  shape: {score_normalized.shape}")
    print(f"  min: {score_normalized.min().item():.4f}")
    print(f"  max: {score_normalized.max().item():.4f}")
    print(f"  mean: {score_normalized.mean().item():.4f}")

    # 验证输出范围
    assert 0.0 < score_normalized.min().item()
    assert score_normalized.max().item() < 1.0

    print("\n✓ Output is strictly in (0, 1)")

    # 测试反归一化
    print("\n" + "="*60)
    print("Testing denormalization...")
    print("="*60)

    # JIGSAWS GRS: 1-30 → 归一化到 0-1
    # 推理时反归一化回 1-10
    score_original = regressor.denormalize_score(
        score_normalized,
        score_min=1.0,   # GGRS 最小值
        score_max=30.0,  # GGRS 最大值
        target_min=0.0, # 归一化目标最小
        target_max=1.0  # 归一化目标最大
    )

    print(f"\nDenormalized score (1-30 range):")
    print(f"  shape: {score_original.shape}")
    print(f"  min: {score_original.min().item():.4f}")
    print(f"  max: {score_original.max().item():.4f}")
    print(f"  mean: {score_original.mean().item():.4f}")

    # 测试反归一化到 1-10（用于论文分数）
    score_paper = regressor.denormalize_score(
        score_normalized,
        score_min=1.0,   # GGRS 最小值
        score_max=30.0,  # GGRS 最大值
        target_min=0.0, # 归一化目标最小
        target_max=1.0  # 归一化目标最大
    )

    print(f"\nDenormalized score (paper 1-10 range):")
    # 手动计算：(score_1to30 - 1) / (30 - 1) * (10 - 1) + 1
    score_paper_value = (score_original - 1.0) / 29.0 * 9.0 + 1.0
    print(f"  min: {score_paper_value.min().item():.4f}")
    print(f"  max: {score_paper_value.max().item():.4f}")

    print("\n✓ All tests passed!")
