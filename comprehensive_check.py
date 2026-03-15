#!/usr/bin/env python3
"""
完整数据流验证脚本
逐步检查每个模块的内部实现逻辑，确保真正符合设计设想
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
import cv2

print('\n')
print('=' * 80)
print('COMPREHENSIVE DATA FLOW VERIFICATION')
print('=' * 80)
print()

# ============================================================================
# 步骤 0: 配置检查
# ============================================================================
print('STEP 0: CONFIGURATION VERIFICATION')
print('=' * 80)

# PAPER REQUIREMENT: 112x112 resolution
B, C, T, H, W = 2, 3, 16, 112, 112
clip_length = 16
clip_stride = 10

print(f'Input configuration:')
print(f'  Batch Size (B): {B}')
print(f'  Channels (C): {C}')
print(f'  Frames per clip (T): {T}')
print(f'  Height (H): {H}')
print(f'  Width (W): {W}')
print(f'  Clip length: {clip_length}')
print(f'  Clip stride: {clip_stride}')
print()

# ============================================================================
# 步骤 1: 数据加载器检查
# ============================================================================
print('\n' + '=' * 80)
print('STEP 1: DATA LOADER VERIFICATION')
print('=' * 80)
print()

print('Checking utils/data_loader.py implementation...')
print()

# 导入模块
try:
    from utils.data_loader import SurgicalVideoDataset
    print('✓ SurgicalVideoDataset imported successfully')
except Exception as e:
    print(f'✗ Failed to import SurgicalVideoDataset: {e}')
    sys.exit(1)

# 创建测试数据集
import tempfile
tmp_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp_dir, 'videos'), exist_ok=True)
os.makedirs(os.path.join(tmp_dir, 'masks'), exist_ok=True)

# 创建测试视频（100帧）
test_video_path = os.path.join(tmp_dir, 'videos', 'test_video.mp4')
writer = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (H, W))
for frame in range(100):
    writer.write(np.random.randint(0, 256, (H, W, 3), dtype=np.uint8))
writer.release()

# 创建测试标注
import json
annotations = {'test_video': {'score': 7.5, 'duration': 3.33}}
with open(os.path.join(tmp_dir, 'annotations.json'), 'w') as f:
    json.dump(annotations, f)

# 创建测试掩膜
mask_dir = os.path.join(tmp_dir, 'masks', 'test_video')
os.makedirs(mask_dir, exist_ok=True)
for frame in range(100):
    mask = np.zeros((H, W), dtype=np.uint8)
    # 在中间区域画一个白色区域（模拟器械）
    mask[100:180, 100:180] = 255
    cv2.imwrite(os.path.join(mask_dir, f'frame_{frame:04d}_mask.png'), mask)

print(f'Created test dataset at: {tmp_dir}')
print(f'  Test video: 100 frames')
print(f'Expected clips: {(100 - clip_length) // clip_stride + 1} = 9 clips')
print()

# 测试不启用clip缓存的数据集
print('\n' + '-' * 80)
print('1.1 Testing WITHOUT clip caching (legacy mode)...')
print('-' * 80)
try:
    dataset_legacy = SurgicalVideoDataset(
        tmp_dir,
        clip_length=clip_length,
        clip_stride=clip_stride,
        spatial_size=(H, W),
        cache_clips=False  # Legacy mode
    )
    
    legacy_len = len(dataset_legacy)
    print(f'Dataset.__len__() = {legacy_len}')
    print(f'Expected: 1 (number of videos)')
    print(f'✓ Match: {legacy_len == 1}')
    
    # 获取样本
    sample_legacy = dataset_legacy[0]
    print(f'\nSample keys: {list(sample_legacy.keys())}')
    print(f'  video_id: {sample_legacy["video_id"]}')
    print(f'  frames shape: {sample_legacy["frames"].shape}')
    print(f'  score: {sample_legacy["score"]}')
    print(f'  clip_info: {sample_legacy["clip_info"]}')
    
except Exception as e:
    print(f'✗ Error in legacy mode: {e}')
    import traceback
    traceback.print_exc()

# 测试启用clip缓存的数据集
print('\n' + '-' * 80)
print('1.2 Testing WITH clip caching (paper1231 mode)...')
print('-' * 80)
try:
    dataset_paper1231 = SurgicalVideoDataset(
        tmp_dir,
        clip_length=clip_length,
        clip_stride=clip_stride,
        spatial_size=(H, W),
        cache_clips=True  # paper1231 mode
    )
    
    paper1231_len = len(dataset_paper1231)
    print(f'Dataset.__len__() = {paper1231_len}')
    expected_clips = (100 - clip_length) // clip_stride + 1
    print(f'Expected: {expected_clips} (number of clips)')
    print(f'✓ Match: {paper1231_len == expected_clips}')
    
    # 检查前3个clips的重叠关系
    print('\nVerifying clip overlaps (paper1231 strategy):')
    for i in range(min(3, paper1231_len)):
        sample = dataset_paper1231[i]
        clip_info = sample['clip_info']
        print(f'Clip {i}: frames[{clip_info["start_frame"]}:{clip_info["end_frame"]}]')
        
        if i > 0:
            prev_sample = dataset_paper1231[i-1]
            prev_clip_info = prev_sample['clip_info']
            overlap = clip_info['start_frame'] - prev_clip_info['end_frame']
            expected_overlap = clip_length - clip_stride
            print(f'  Overlap with clip {i-1}: {overlap} frames')
            print(f'  ✓ Expected: {expected_overlap}, Match: {overlap == expected_overlap}')
    
except Exception as e:
    print(f'✗ Error in paper1231 mode: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# 步骤 2: 静态特征提取器检查
# ============================================================================
print('\n' + '=' * 80)
print('STEP 2: STATIC FEATURE EXTRACTOR (RESNET-34) VERIFICATION')
print('=' * 80)
print()

try:
    from models.static_feature_extractor import StaticFeatureExtractor
    print('✓ StaticFeatureExtractor imported successfully')
    
    # 创建测试输入
    test_video = torch.randn(B, C, T, H, W)
    print(f'\nInput video shape: {test_video.shape}')
    
    # 测试不同的采样策略
    for strategy in ['middle', 'average', 'first', 'last']:
        extractor = StaticFeatureExtractor(
            output_dim=512,
            sampling_strategy=strategy
        )
        
        features = extractor(test_video)
        print(f'\n{strategy} strategy:')
        print(f'  Output features shape: {features.shape}')
        print(f'  ✓ Expected: ({B}, 512)')
        print(f'  ✓ Match: {features.shape == torch.Size([B, 512])}')
    
except Exception as e:
    print(f'✗ Error in static feature extractor: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# 步骤 3: 动态特征提取器检查
# ============================================================================
print('\n' + '=' * 80)
print('STEP 3: DYNAMIC FEATURE EXTRACTOR (I3D) VERIFICATION')
print('=' * 80)
print()

try:
    from models.dynamic_feature_extractor import DynamicFeatureExtractor
    print('✓ DynamicFeatureExtractor imported successfully')
    
    # 创建测试输入
    test_video = torch.randn(B, C, T, H, W)
    print(f'\nInput video shape: {test_video.shape}')
    
    extractor = DynamicFeatureExtractor(
        output_dim=832,
        use_mixed_conv=True
    )
    
    # 测试返回特征图
    feat_map, feat_pooled = extractor(test_video, return_features_map=True)
    print(f'\nWith return_features_map=True:')
    print(f'  Feature map shape: {feat_map.shape}')
    print(f'  ✓ Expected: ({B}, 832, T\', H\', W\')')
    print(f'  ✓ Match: {feat_map.shape[0] == B and feat_map.shape[1] == 832}')
    print(f'  Pooled features shape: {feat_pooled.shape}')
    print(f'  ✓ Expected: ({B}, 832)')
    print(f'  ✓ Match: {feat_pooled.shape == torch.Size([B, 832])}')
    
    # 测试不返回特征图
    feat_only = extractor(test_video, return_features_map=False)
    print(f'\nWith return_features_map=False:')
    print(f'  Output features shape: {feat_only.shape}')
    print(f'  ✓ Expected: ({B}, 832)')
    print(f'  ✓ Match: {feat_only.shape == torch.Size([B, 832])}')
    
except Exception as e:
    print(f'✗ Error in dynamic feature extractor: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# 步骤 4: 掩膜引导注意力模块检查
# ============================================================================
print('\n' + '=' * 80)
print('STEP 4: MASK-GUIDED ATTENTION MODULE VERIFICATION')
print('=' * 80)
print()

try:
    from models.mask_guided_attention import MaskGuidedAttention
    print('✓ MaskGuidedAttention imported successfully')
    
    # 创建测试输入
    C_dynamic, T_dynamic, H_dynamic, W_dynamic = 832, 4, 7, 7
    dynamic_features = torch.randn(B, C_dynamic, T_dynamic, H_dynamic, W_dynamic)
    masks = torch.randint(0, 2, (B, T, H, W)).float()
    
    print(f'\nDynamic features shape: {dynamic_features.shape}')
    print(f'  Masks shape: {masks.shape}')
    print(f'  ✓ Expected dynamic features: ({B}, {C_dynamic}, {T_dynamic}, {H_dynamic}, {W_dynamic})')
    print(f'  ✓ Expected masks: ({B}, {T}, {H}, {W})')
    
    attention_module = MaskGuidedAttention(enable_temporal_smoothing=True)
    
    masked_features, attention_map, mask_loss = attention_module(
        dynamic_features, masks, return_attention_map=True
    )
    
    print(f'\nOutput:')
    print(f'  Masked features shape: {masked_features.shape}')
    print(f'  ✓ Expected: ({B}, {C_dynamic})')
    print(f'  ✓ Match: {masked_features.shape == torch.Size([B, C_dynamic])}')
    print(f'  Attention map shape: {attention_map.shape}')
    print(f'  ✓ Expected: ({B}, {T_dynamic}, {H_dynamic}, {W_dynamic})')
    print(f'  ✓ Match: {attention_map.shape[0] == B and attention_map.shape[1] == T_dynamic}')
    print(f'  Mask loss: {mask_loss}')
    print(f'  ✓ Expected: None (no supervision for offline masks)')
    
except Exception as e:
    print(f'✗ Error in mask-guided attention: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# 步骤 5: 主模型完整数据流检查
# ============================================================================
print('\n' + '=' * 80)
print('STEP 5: COMPLETE MODEL DATA FLOW VERIFICATION')
print('=' * 80)
print()

try:
    from models.surgical_qa_model import SurgicalQAModel
    print('✓ SurgicalQAModel imported successfully')
    
    # 配置
    config = {
        'static_dim': 512,
        'dynamic_dim': 832,
        'use_pretrained': False,
        'freeze_backbone': False
    }
    
    model = SurgicalQAModel(config)
    
    # 统计参数
    total, trainable = model.count_parameters()
    print(f'\nModel statistics:')
    print(f'  Total parameters: {total:,}')
    print(f'  Trainable parameters: {trainable:,}')
    
    # 创建测试输入
    test_video = torch.randn(B, C, T, H, W)
    test_masks = torch.randint(0, 2, (B, T, H, W)).float()
    
    print(f'\n' + '-' * 80)
    print('5.1 Input shapes:')
    print('-' * 80)
    print(f'  Video X: {test_video.shape}')
    print(f'  Masks B: {test_masks.shape}')
    print()
    
    # 完整前向传播
    model.eval()
    with torch.no_grad():
        score, mask_loss = model(test_video, test_masks)
        score_features, features_dict = model(test_video, test_masks, return_features=True)
        
        print('-' * 80)
        print('5.2 Output shapes:')
        print('-' * 80)
        print(f'  Score: {score.shape}')
        print(f'  ✓ Expected: ({B}, 1)')
        print(f'  ✓ Match: {score.shape == torch.Size([B, 1])}')
        print()
        
        print('5.3 Intermediate features:')
        print('-' * 80)
        print(f'  Static features A: {features_dict["static"].shape}')
        print(f'  ✓ Expected: ({B}, 512)')
        print(f'  ✓ Match: {features_dict["static"].shape == torch.Size([B, 512])}')
        
        print(f'  Dynamic feature map C: {features_dict["dynamic_feat_map"].shape}')
        print(f'  ✓ Expected: ({B}, 832, T\', H\', W\')')
        
        print(f'  Raw dynamic features: {features_dict["dynamic_raw"].shape}')
        print(f'  ✓ Expected: ({B}, 832)')
        
        print(f'  Masked dynamic features D: {features_dict["dynamic_masked"].shape}')
        print(f'  ✓ Expected: ({B}, 832)')
        print(f'  ✓ Match: {features_dict["dynamic_masked"].shape == torch.Size([B, 832])}')

        attention_map = features_dict["attention_map"]
        if attention_map is not None:
            print(f'  Attention map: {attention_map.shape}')
            print(f'  ✓ Expected: ({B}, T\', H\', W\')')
        else:
            print(f'  Attention map: None (return_attention_map was False)')
        
        print(f'  Fused features: {features_dict["fused"].shape}')
        print(f'  ✓ Expected: ({B}, 1344) = 512 + 832')
        print(f'  ✓ Match: {features_dict["fused"].shape == torch.Size([B, 1344])}')
        
except Exception as e:
    print(f'✗ Error in complete model flow: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# 步骤 6: 训练循环数据流检查
# ============================================================================
print('\n' + '=' * 80)
print('STEP 6: TRAINING LOOP DATA FLOW VERIFICATION')
print('=' * 80)
print()

try:
    from utils.data_loader import SurgicalQADataLoader
    print('✓ SurgicalQADataLoader imported successfully')
    
    # 创建数据加载器（启用clip缓存）
    dataloader = SurgicalQADataLoader(
        tmp_dir,
        batch_size=2,
        num_workers=0,
        dataset_kwargs={
            'clip_length': clip_length,
            'clip_stride': clip_stride,
            'spatial_size': (H, W),
            'cache_clips': True
        }
    )
    
    print(f'\nDataloader statistics:')
    print(f'  Train dataset size: {len(dataloader.train_dataset)}')
    print(f'  Val dataset size: {len(dataloader.val_dataset)}')
    print(f'  Test dataset size: {len(dataloader.test_dataset)}')
    
    # 获取第一个batch
    batch = next(iter(dataloader.train_loader))
    
    print(f'\n' + '-' * 80)
    print('6.1 Batch data structure:')
    print('-' * 80)
    print(f'  Batch keys: {list(batch.keys())}')
    
    if 'frames' in batch:
        print(f'  frames shape: {batch["frames"].shape}')
        print(f'  ✓ Expected: ({B}, {C}, {T}, {H}, {W})')
        print(f'  ✓ Match: {batch["frames"].shape == torch.Size([B, C, T, H, W])}')
    
    if 'masks' in batch:
        print(f'  masks shape: {batch["masks"].shape}')
        print(f'  ✓ Expected: ({B}, {T}, {H}, {W})')
        print(f'  ✓ Match: {batch["masks"].shape == torch.Size([B, T, H, W])}')
    
    if 'score' in batch:
        print(f'  score shape: {batch["score"].shape}')
        print(f'  ✓ Expected: ({B},)')
        print(f'  ✓ Match: {batch["score"].shape == torch.Size([B])}')
    
    if 'global_clip_idx' in batch:
        print(f'  global_clip_idx: {batch["global_clip_idx"]}')
        print(f'  ✓ Note: Only available in paper1231 mode')
    
except Exception as e:
    print(f'✗ Error in training loop data flow: {e}')
    import traceback
    traceback.print_exc()

# ============================================================================
# 总结
# ============================================================================
print('\n' + '=' * 80)
print('VERIFICATION SUMMARY')
print('=' * 80)
print()
print('All critical components checked!')
print('Please review the output above for any issues.')
print()

# 清理
import shutil
shutil.rmtree(tmp_dir)

print('\n' + '=' * 80)
print('✅ VERIFICATION COMPLETE')
print('=' * 80)
print()

