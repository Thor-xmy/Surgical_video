"""
Fusion Regressor for Multi-Clip Bounded Model

这个版本只接受单一输入（已经拼接好的时序特征），
输出带Sigmoid的分数[0, 1]。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedFusionRegressorMultiClip(nn.Module):
    """
    带边界的特征融合和分数回归模块（Multi-Clip版本）

    输入单一已经拼接好的时序特征，输出带Sigmoid的分数。

    配合归一化标签使用：
    - 归一化标签：[0, 1]
    - 模型输入：total_dim （static_dim + dynamic_dim）* num_clips
    - 模型输出：[0, 1]（Sigmoid）
    - 训练损失：MSE
    - 推理：反归一化到 [1, 10] 或 [6, 30]
    """
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128],dropout_rate=0.5,out_dim=1,use_sigmoid=True):
        """
        Args:
            input_dim: Total input dimension (already fused features)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        # Build MLP regressor（最后一层使用 Sigmoid）
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final output layer：Linear → Sigmoid
        layers.append(nn.Linear(prev_dim, out_dim))
        if use_sigmoid:
            layers.append(nn.Sigmoid())  # ← 关键：根据开关决定是否锁死 (0, 1)
        # Sigmoid 确保输出严格在 (0, 1) 之间
        #layers.extend([
            #nn.Linear(prev_dim, out_dim),
            #nn.LayerNorm(64),
            #nn.ReLU(inplace=True),
            #nn.Dropout(dropout_rate / 2.0),
            #nn.Linear(64, 1),
            #nn.LayerNorm(1), ###############################################
            #nn.Sigmoid()  # ← 关键：添加 Sigmoid 激活函数
        #])

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

    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, input_dim) - 已经拼接好的时序特征

        Returns:
            score: (B, 1) - 在 [0, 1] 范围内的预测分数
        """
        # 直接通过回归器（带Sigmoid）
        score = self.regressor(fused_features)  # (B, 1)

        return score


if __name__ == '__main__':
    # 测试
    print("Testing BoundedFusionRegressorMultiClip...")

    # 测试不同的输入维度（对应不同的clips数量）
    test_cases = [
        (2, 512, 1024, 9),    # 9 clips -> (512+1024)*9 = 18432
        (2, 512, 1024, 10),   # 10 clips -> (512+1024)*10 = 20480
        (2, 512, 1024, 5),    # 5 clips -> (512+1024)*5 = 10240
    (2, 512, 1024, 1),    # 1 clip -> (512+1024)*1 = 2048
    ]

    for input_dim in test_cases:
        model = BoundedFusionRegressorMultiClip(input_dim=input_dim)

        features = torch.randn(2, input_dim)
        score = model(features)

        print(f"\nInput dim: {input_dim} (≈{input_dim//(512+1024)} clips)")
        print(f"  Output shape: {score.shape}")
        print(f"  Output range: [{score.min().item():.4f}, {score.max().item():.4f}]")

        assert score.shape == (2, 1)
        assert score.min() >= 0.0 and score.max() <= 1.0

    print("  ✓ Test passed!")
