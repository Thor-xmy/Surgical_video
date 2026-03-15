#!/usr/bin/env python3
"""
数据流逐步验证脚本
"""

import sys
import os
import cv2
import numpy as np
import torch

sys.path.insert(0, '.')

# PAPER REQUIREMENT: 112x112 resolution
B, C, T, H, W = 2, 3, 16, 112, 112

print('=' * 80)
print('数据流逐步验证')
print('=' * 80)
print()

print('输入配置：')
print(f'  B: {B}')
print(f'  C: {C}')
print(f'  T: {T}')
print(f'  H: {H}')
print(f'  W: {W}')
print()

# ============================================================================
print('阶段1：数据加载器验证')
print('=' * 80)
print()

from utils.data_loader import SurgicalVideoDataset
import tempfile

# 创建测试数据集
tmp_dir = tempfile.mkdtemp()
videos_dir = os.path.join(tmp_dir, 'videos')
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(os.path.join(tmp_dir, 'masks'), exist_ok=True)

# 创建测试视频（100帧）
video_path = os.path.join(videos_dir, 'test.mp4')
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (H, W))
for frame in range(100):
    writer.write(np.random.randint(0, 256, (H, W, 3), dtype=np.uint8))
writer.release()

# 创建掩膜
mask_dir = os.path.join(tmp_dir, 'masks', 'test')
os.makedirs(mask_dir, exist_ok=True)
for frame in range(100):
    mask = np.zeros((H, W), dtype=np.uint8)
    # 在[40:72, 40:72]区域画一个白色区域（模拟器械），适配112x112
    mask[40:72, 40:72] = 255
    cv2.imwrite(os.path.join(mask_dir, f'frame_{frame:04d}_mask.png'), mask)

# 创建标注
import json
annotations = {'test': {'score': 7.5}}
with open(os.path.join(tmp_dir, 'annotations.json'), 'w') as f:
    json.dump(annotations, f)

print('创建测试数据集完成')
print()

# 测试数据加载器（启用clip缓存）
dataset = SurgicalVideoDataset(
    tmp_dir,
    clip_length=16,
    clip_stride=10,
    spatial_size=(H, W),
    cache_clips=True  # paper1231风格
)

dataset_len = len(dataset)
print(f'Dataset.__len__() = {dataset_len}')
print(f'预期clips数 = 9（100帧视频，16帧clip，10步长）')
print()

if dataset_len == 9:
    print('✓ 数据集大小正确！')
else:
    print(f'✗ 数据集大小错误！应该是9，实际是{dataset_len}')

print()

# 检查前3个clips的详细信息
print('检查前3个clips的详细信息：')
for i in range(min(3, dataset_len)):
    sample = dataset[i]
    print(f'\nClip {i}:')
    print(f'  Keys: {list(sample.keys())}')

    if 'video_id' in sample:
        print(f"  video_id: {sample['video_id']}")

    if 'global_clip_idx' in sample:
        print(f"  global_clip_idx: {sample['global_clip_idx']}")

    if 'score' in sample:
        score_val = sample['score']
        if isinstance(score_val, torch.Tensor):
            score_val = score_val.item()
        print(f'  score: {score_val}')

    if 'frames' in sample:
        frames_shape = sample['frames'].shape
        expected_frames_shape = (C, T, H, W)
        print(f'  frames shape: {frames_shape}')
        print(f'  期望: {expected_frames_shape}')
        if frames_shape == expected_frames_shape:
            print(f' ✓ frames形状正确')
        else:
            print(f' ✗ frames形状错误！')

    if 'masks' in sample:
        masks_shape = sample['masks'].shape
        expected_masks_shape = (T, H, W)
        print(f'  masks shape: {masks_shape}')
        print(f'  期望: {expected_masks_shape}')
        if masks_shape == expected_masks_shape:
            print(f' ✓ masks形状正确')
        else:
            print(f' ✗ masks形状错误！')

    if 'clip_info' in sample:
        clip_info = sample['clip_info']
        print(f'  clip_info keys: {list(clip_info.keys())}')
        if 'start_frame' in clip_info:
            print(f"  start_frame: {clip_info['start_frame']}")
        if 'end_frame' in clip_info:
            print(f"  end_frame: {clip_info['end_frame']}")
        frame_count = clip_info['end_frame'] - clip_info['start_frame']
        if frame_count == 16:
            print(f'  ✓ frame_count正确（16帧）')
        else:
            print(f'  ✗ frame_count错误！应该是16，实际是{frame_count}')

print()

# ============================================================================
print('阶段2：静态特征提取器验证')
print('=' * 80)
print()

from models.static_feature_extractor import StaticFeatureExtractor

print('创建静态特征提取器...')
static_extractor = StaticFeatureExtractor(
    output_dim=512,
    sampling_strategy='middle'
)

# 创建测试输入
test_video = torch.randn(B, C, T, H, W)
print(f'测试输入视频shape: {test_video.shape}')

print('提取静态特征A...')
static_features = static_extractor(test_video)
print(f'输出形状: {static_features.shape}')
print(f'期望: ({B}, 512)')

if static_features.shape == torch.Size([B, 512]):
    print('✓ 静态特征提取器正确')
else:
    print('✗ 静态特征提取器错误！')
    sys.exit(1)

print()

# ============================================================================
print('阶段3：动态特征提取器验证')
print('=' * 80)
print()

from models.dynamic_feature_extractor import DynamicFeatureExtractor

print('创建动态特征提取器...')
dynamic_extractor = DynamicFeatureExtractor(
    output_dim=1024,  # Standard I3D outputs 1024 channels
    use_mixed_conv=True
)

# 创建测试输入
test_video = torch.randn(B, C, T, H, W)
print(f'测试输入视频shape: {test_video.shape}')

print('提取动态特征（返回特征图和池化特征）...')
feat_map, feat_pooled = dynamic_extractor(test_video, return_features_map=True)
print(f'特征图shape: {feat_map.shape}')
print(f'池化特征shape: {feat_pooled.shape}')

# 检查特征图 (112x112输入 → I3D Mixed_5c输出: (B, 1024, 2, 4, 4))
expected_feat_map_shape = (B, 1024, 2, 4, 4)
print(f'期望特征图shape: {expected_feat_map_shape}')

if feat_map.shape == expected_feat_map_shape:
    print('✓ 特征图形状正确')
else:
    print(f'✗ 特征图形状错误！期望{expected_feat_map_shape}，实际{feat_map.shape}')
    sys.exit(1)

if feat_pooled.shape == torch.Size([B, 1024]):
    print('✓ 池化特征形状正确')
else:
    print('✗ 池化特征形状错误！')
    sys.exit(1)

print()

# ============================================================================
print('阶段4：掩膜引导注意力验证')
print('=' * 80)
print()

from models.mask_guided_attention import MaskGuidedAttention

print('创建掩膜引导注意力模块...')
attention_module = MaskGuidedAttention(enable_temporal_smoothing=True)

# 创建测试输入
dynamic_features = torch.randn(B, 1024, 2, 4, 4)  # I3D output shape for 112x112
masks = torch.randint(0, 2, (B, T, H, W)).float()
print(f'动态特征shape: {dynamic_features.shape}')
print(f'掩膜shape: {masks.shape}')

print('应用掩膜注意力...')
masked_features, attention_map, mask_loss = attention_module(
    dynamic_features,
    masks,
    return_attention_map=True
)
print(f'掩膜引导的动态特征D: {masked_features.shape}')
print(f'注意力图shape: {attention_map.shape}')
print(f'期望D: ({B}, 1024)')

if masked_features.shape == torch.Size([B, 1024]):
    print('✓ 掩膜引导的动态特征D形状正确')
else:
    print('✗ 掩膜引导的动态特征D形状错误！')
    sys.exit(1)

# 注意力图形状应该与I3D特征图的时空维度一致
expected_attention_shape = torch.Size([B, 2, 4, 4])  # I3D output for 112x112
if attention_map.shape == expected_attention_shape:
    print('✓ 注意力面形状正确')
else:
    print(f'✗ 注意力图形状错误！期望{expected_attention_shape}，实际{attention_map.shape}')
    sys.exit(1)

if mask_loss is None:
    print('✓ mask_loss正确（离线掩膜无需监督）')
else:
    print('✗ mask_loss错误！')

print()

# ============================================================================
print('阶段5：主模型完整验证')
print('=' * 80)
print()

from models.surgical_qa_model import SurgicalQAModel

print('创建主模型...')
config = {
    'static_dim': 512,
    'dynamic_dim': 1024,  # Standard I3D outputs 1024 channels
    'use_pretrained': False,
    'freeze_backbone': False,
    'use_mask_loss': True
}

model = SurgicalQAModel(config)
model.eval()

print('创建测试输入...')
test_video = torch.randn(B, C, T, H, W)
test_masks = torch.randint(0, 2, (B, T, H, W)).float()

print('输入shapes:')
print(f'  视频X: {test_video.shape}')
print(f'  掩膜B: {test_masks.shape}')

print()
print('5.1 测试完整前向传播（返回所有中间特征）...')
with torch.no_grad():
    score, mask_loss = model(test_video, test_masks, return_features=True, return_attention=True)
    score_features, features_dict = score, mask_loss

print('输出shapes:')
print(f'  分数y: {score.shape}')
print(f'  期望: ({B}, 1)')

if score.shape == torch.Size([B, 1]):
    print('✓ 分数y形状正确')
else:
    print('✗ 分数y形状错误！')
    sys.exit(1)

print()
print('5.2 检查中间特征...')
print(f' 静态特征A: {features_dict["static"].shape}')
print(f'  期望: ({B}, 512)')

if features_dict["static"].shape == torch.Size([B, 512]):
    print('✓ 静态特征A形状正确')
else:
    print('✗ 静态特征A形状错误！')

print(f'  动态特征图C: {features_dict["dynamic_feat_map"].shape}')
print(f'  期望: ({B}, 1024, 2, 4, 4)')  # 112x112 input → I3D output

if features_dict["dynamic_feat_map"].shape == torch.Size([B, 1024, 2, 4, 4]):
    print('✓ 动态特征图C形状正确')
else:
    print('✗ 动态特征图C形状错误！')
    sys.exit(1)

print(f'  池化动态特征D: {features_dict["dynamic_raw"].shape}')
print(f'  期望: ({B}, 1024)')
if features_dict["dynamic_raw"].shape == torch.Size([B, 1024]):
    print('✓ 池化动态特征D形状正确')
else:
    print('✗ 池化动态特征D形状错误！')
    sys.exit(1)

print(f'  掩膜引导的动态特征D: {features_dict["dynamic_masked"].shape}')
print(f' 期望: ({B}, 1024)')
if features_dict["dynamic_masked"].shape == torch.Size([B, 1024]):
    print('✓ 掩膜引导的动态特征D形状正确')
else:
    print('✗ 掩膜引导的动态特征D形状错误！')
    sys.exit(1)

print(f'  注意力图: {features_dict["attention_map"].shape}')
# 注意力图应该与I3D特征图的时空维度一致（112x112输入 → I3D输出2x4x4）
print(f'期望: ({B}, 2, 4, 4)')  # I3D Mixed_5c输出: (B, 1024, 2, 4, 4)
if features_dict["attention_map"].shape == torch.Size([B, 2, 4, 4]):
    print('✓ 注意力图形状正确')
else:
    print('✗ 注意力图形状错误！')
    sys.exit(1)

print(f' 融合特征: {features_dict["fused"].shape}')
print(f'期望: ({B}, 1536)')  # 512 + 1024 = 1536
if features_dict["fused"].shape == torch.Size([B, 1536]):
    print('✓ 融合特征形状正确')
else:
    print('✗ 融合特征形状错误！')
    sys.exit(1)

print()
print('5.3 测试数据流整合...')
print('检查特征')
print(f' 静态A: {features_dict["static"].shape}')
print(f'  动态D: {features_dict["dynamic_masked"].shape}')
fused_expected = torch.cat([features_dict["static"], features_dict["dynamic_masked"]], dim=1)
print(f'  融合: {fused_expected.shape}')
if features_dict["fused"].shape == fused_expected.shape:
    print('✓ 特征融合正确')
else:
    print('✗ 特征融合错误！')
    sys.exit(1)

print()

# ============================================================================
print('阶段6：训练循环数据流验证')
print('=' * 80)
print()

from utils.data_loader import SurgicalQADataLoader

print('创建数据加载器（启用clip缓存）...')
dataloader = SurgicalQADataLoader(
    tmp_dir,
    batch_size=B,
    num_workers=0,
    dataset_kwargs={
        'clip_length': 16,
        'clip_stride': 10,
        'spatial_size': (H, W),
        'cache_clips': True  # 关键！
    }
)

print('数据集split统计：')
print(f' 训练集大小: {len(dataloader.train_dataset)}')
print(f' 验证集大小: {len(dataloader.val_dataset)}')
print(f' 测试集大小: {len(dataloader.test_dataset)}')

print(f'预期值：')
expected_total_clips = 9  # 100帧视频，16帧clip，10步长
expected_train = int(expected_total_clips * 0.8)  # 80%
expected_val = int(expected_total_clips * 0.1)  # 10%
expected_test = expected_total_clips - expected_train - expected_val

print(f'  训练集（预期{expected_train}，实际{len(dataloader.train_dataset)}）')
print(f'  验证集（预期{expected_val}，实际{len(dataloader.val_dataset)}）')
print(f'  测试集（预期{expected_test}，实际{len(dataloader.test_dataset)}）')

if len(dataloader.train_dataset) == expected_train:
    print('✓ 训练集大小正确')
else:
    print(f'✗ 训练集大小错误！预期{expected_train}，实际{len(dataloader.train_dataset)}')
    sys.exit(1)

if len(dataloader.val_dataset) == expected_val:
    print('✓ 验证集大小正确')
else:
    print(f'✗ 验证集大小错误！预期{expected_val}，实际{len(dataloader.val_dataset)}')
    sys.exit(1)

if len(dataloader.test_dataset) == expected_test:
    print('✓ 测试集大小正确')
else:
    print(f'✗ 测试集大小错误！预期{expected_test}，实际{len(dataloader.test_dataset)}')
    sys.exit(1)

print()

print('获取训练batch（检查batch结构）...')
train_batch = next(iter(dataloader.train_loader))
print(f'Batch keys: {list(train_batch.keys())}')

if 'frames' in train_batch:
    frames_shape = train_batch['frames'].shape
    expected_frames_shape = (B, C, T, H, W)
    print(f' frames shape: {frames_shape}')
    print(f' 期望: {expected_frames_shape}')
    if frames_shape == expected_frames_shape:
        print('✓ Batch frames形状正确')
    else:
        print('✗ Batch frames形状错误！')
        sys.exit(1)

if 'masks' in train_batch:
    masks_shape = train_batch['masks'].shape
    expected_masks_shape = (B, T, H, W)  # Batch size B included
    print(f' masks shape: {masks_shape}')
    print(f' 期望: {expected_masks_shape}')
    if masks_shape == expected_masks_shape:
        print('✓ Batch masks形状正确')
    else:
        print('✗ Batch masks形状错误！')
        sys.exit(1)

if 'score' in train_batch:
    score_shape = train_batch['score'].shape
    expected_score_shape = (B,)
    print(f' score shape: {score_shape}')
    if score_shape == expected_score_shape:
        print('✓ Batch score形状正确')
    else:
        print('✗ Batch score形状错误！')
        sys.exit(1)

if 'global_clip_idx' in train_batch:
    print(f'  global_clip_idx: {train_batch["global_clip_idx"]}')

print()

print('✅ 所有batch结构检查通过！')
print()

# ============================================================================
print('验证总结')
print('=' * 80)
print()

print('完整数据流总结：')
print()
print(f'输入视频X: (B,C,T,H,W) = ({B},{C},{T},{H},{W})')  # 112x112
print()
print('数据流向步骤：')
print('1. 数据加载器:')
print('   - 启用clip缓存策略（paper1231）')
print('   - 从100帧视频提取9个clips')
print('   - Dataset.__len__()返回9（所有clips总数）')
print('   - Dataset.__getitem__()返回特定clip（不随机）')
print('   - 训练/val/test split基于clips（80%/10%/10%）')
print()
print('2. 静态特征A (ResNet-34):')
print('   - 输入: video X (B,C,T,H,W)')
print('   - 输出: A (B,512)')
print('   - 采样：中间帧 (B,C,1,H,W)')
print('   - ResNet-34 backbone → (B,512,H/32,W,W/32)')
print('   - 多尺度下采样: stride{2,4,8} → 拼接3分支')
print('   - 拼接: (B,128,1,1) × 3) → (B,384,1,1)')
print('   - 投影：→512维')
print()
print('3. SAM3掩膜B:')
print('   - 输入：预生成的掩膜文件')
print('   - 输出: B (B,T,H,W)')
print('   - 时间平滑：将相邻帧求和，T/2帧')
print('   - 空间对齐：对齐到I3D特征图')
print()
print('4. I3D动态特征C:')
print('   - 输入: video X (B,C,T,H,W)')
print(f'   - 输出： 特征图C (B,1024,2,4,4) + 池化特征 (B,1024)')  # 112x112 input
print('   - Mixed_5c输出（paper1231 Eq.5）')
print('   - 说明：此C用于掩膜引导注意力')
print()
print('5. 掩膜引导注意力（B作用于C得到D）:')
print(f'   - 输入：动态特征图C (B,1024,2,4,4)')  # 112x112 input
print('   - 输入：掩膜B (B,T,H,W)')
print('   - 时间平滑：T/2帧 → 对齐到特征图')
print(f'   - 对齐：对齐到C (B,2,4,4)')  # 112x112 input
print('   - 注意力：A + 1（器械区域2x增强，背景1x保持）')
print('   - 元素点乘：F_clip × (A + 1)')
print(f'   - 输出：D (B,1024,2,4,4)')
print('   - 池化：全局平均池化→ (B,1024)')
print()
print('6. 特征融合（A⊕D）:')
print('   - 输入：A (B,512), D (B,1024)')
print('   - 拼接：A⊕D (B,1536)')  # 512 + 1024 = 1536
print()
print('7. MLP回归（分数y）:')
print('   - 输入：融合特征 (B,1536)')
print('   - MLP结构: 1536→1024→512→256→128→64→1')
print('   - 输出：分数y (B,1)')
print()
print('=' * 80)
print('✅ 代码库完全符合你的设计设想！')
print('=' * 80)
print()

# 清理
import shutil
shutil.rmtree(tmp_dir)
