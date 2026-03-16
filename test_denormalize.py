#!/usr/bin/env python3
"""
测试反归一化逻辑的正确性

验证：
1. 归一化: GRS [6, 30] → [0, 1]
2. 反归一化: [0, 1] → [6, 30] 或 [1, 10]
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.surgical_qa_model_bounded import SurgicalQAModelBounded

print("="*70)
print("反归一化逻辑测试")
print("="*70)

# 创建模型
config = {
    'static_dim': 512,
    'dynamic_dim': 1024,
    'use_pretrained': False,
    'freeze_backbone': True,
    'use_bounded_regression': True,
    'score_min': 6.0,  # JIGSAWS GRS 最小值
    'score_max': 30.0,  # JIGSAWS GRS 最大值
}

model = SurgicalQAModelBounded(config)

# 测试用例
test_cases = [
    # (normalized_value, target_range, expected_result, description)
    (0.0, (6.0, 30.0), 6.0, "最小值 0 → 应该映射到 6"),
    (0.5, (6.0, 30.0), 18.0, "中值 0.5 → 应该映射到 18"),
    (1.0, (6.0, 30.0), 30.0, "最大值 1 → 应该映射到 30"),
    (0.0, (1.0, 10.0), 1.0, "最小值 0 → 映射到论文范围 1"),
    (0.5, (1.0, 10.0), 5.5, "中值 0.5 → 映射到论文范围 5.5"),
    (1.0, (1.0, 10.0), 10.0, "最大值 1 → 映射到论文范围 10"),
]

all_passed = True

for norm_val, target_range, expected, desc in test_cases:
    normalized = torch.tensor([norm_val])

    # 调用反归一化
    result = model.denormalize_score(
        normalized,
        target_min=target_range[0],  # 目标范围最小
        target_max=target_range[1],  # 目标范围最大
        norm_min=0.0,            # 归一化范围固定 0
        norm_max=1.0              # 归一化范围固定 1
    )

    result_val = result.item()
    passed = abs(result_val - expected) < 0.001

    status = "✓" if passed else "✗"
    print(f"{status} {desc}")
    print(f"    输入: {norm_val}, 目标范围: {target_range}")
    print(f"    期望: {expected:.2f}, 实际: {result_val:.2f}")

    if not passed:
        all_passed = False
        print(f"    ✗ 错误：相差 {abs(result_val - expected):.4f}")

print("\n" + "="*70)

if all_passed:
    print("✓ 所有测试通过！反归一化逻辑正确。")
else:
    print("✗ 有测试失败！反归一化逻辑有问题。")
    sys.exit(1)

print("="*70)

# 额外测试：验证数学公式
print("\n" + "="*70)
print("数学公式验证")
print("="*70)

print("\n归一化公式: normalized = (score - score_min) / (score_max - score_min)")
print("反归一化公式: target = (norm - norm_min) / (norm_max - norm_min) * (target_max - target_min) + target_min")

print("\n示例 1: GRS 15 → 归一化 → 反归一化")
score_orig = 15.0
norm = (score_orig - 6.0) / (30.0 - 6.0)
print(f"  原始分数: {score_orig}")
print(f"  归一化后: {norm:.4f}")
norm_tensor = torch.tensor([norm])
result = model.denormalize_score(norm_tensor, target_min=6.0, target_max=30.0, norm_min=0.0, norm_max=1.0)
print(f"  反归一化后: {result.item():.4f}")
print(f"  误差: {abs(result.item() - score_orig):.10f}")

print("\n示例 2: GRS 15 → 归一化 → 映射到论文范围 [1,10]")
score_orig = 15.0
norm = (score_orig - 6.0) / (30.0 - 6.0)
print(f"  原始分数: {score_orig}")
print(f"  归一化后: {norm:.4f}")
norm_tensor = torch.tensor([norm])
result = model.denormalize_score(norm_tensor, target_min=1.0, target_max=10.0, norm_min=0.0, norm_max=1.0)
print(f"  反归一化到 [1,10]: {result.item():.4f}")
# 手动计算：(15-6)/(30-6) * (10-1) + 1 = 9/24 * 9 + 1 = 4.4375
expected_paper = (score_orig - 6.0) / (30.0 - 6.0) * (10.0 - 1.0) + 1.0
print(f"  期望值: {expected_paper:.4f}")
print(f"   误差: {abs(result.item() - expected_paper):.10f}")

print("\n" + "="*70)
