"""
Surgical QA Model with Multi-Clip Support and Bounded Regression

完整实现多clip特征提取和有界回归：
1. Static features: Extracted at clip level and concatenated
2. Dynamic features: Extracted from clips and concatenated
3. Per-clip fusion: [Static_clip, Dynamic_clip] -> Fused_clip
4. Temporal concatenation: [Fused_clip_1, ..., Fused_clip_N] -> Final features
5. Bounded regression: Sigmoid ensures output in [0, 1]

Key aspects:
- 使用方案A（按clip先融合动静特征）
- 使用BoundedFusionRegressor（Sigmoid激活）
- 支持视频级别的标签归一化
- 输出单个视频级别的总分

Pipeline:
    Video (B, 3, T, H, W)
      ├─> Static: split into N clips -> sample keyframes -> ResNet -> (B, N, 512)
      └─> Dynamic: split into N clips -> I3D -> (B, N, 1024)
            ↓ Per-clip fusion
      Concat: [(512+1024), ..., (512+1024)] -> (B, N*1536)
            ↓ Regressor
      Score: (B, 1) in [0, 1] (normalized)
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .static_feature_extractor_multiclip import StaticFeatureMultiClip
from .dynamic_feature_extractor_multiclip import DynamicFeatureMultiClip
from .fusion_regressor_multiclip_bounded import BoundedFusionRegressorMultiClip

# === 🌟 新增代码块：定义位置编码层 ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150):
        super().__init__()
        # 初始化一个 max_len x d_model 的零矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 使用正弦和余弦函数生成位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加 batch 维度: (1, max_len, d_model)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # x 的形状是 (Batch, num_clips, feature_dim)
        seq_len = x.size(1)
        # 将位置编码叠加到原始特征上
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x
# ====================================

class SurgicalQAModelMultiClipBounded(nn.Module):
    """
    Surgical QA Model with Multi-Clip Support and Bounded Regression.

    完整实现：
    1. Video-level 输入
    2. Clip-level static and dynamic feature extraction
    3. Per-clip feature fusion
    4. Temporal sequence concatenation
    5. Bounded regression with Sigmoid
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dict with:
                - static_dim: Static feature dimension per clip (default: 512)
                - dynamic_dim: Dynamic feature dimension per clip (default: 1024)
                - clip_length: Length of each clip (default: 16)
                - clip_stride: Stride between clips (default: 10)
                - max_clips: Maximum number of clips (default: None)
                - resnet_path: Path to ResNet checkpoint
                - i3d_path: Path to I3D checkpoint
                - use_pretrained: Use pretrained weights
                - freeze_backbone: Freeze backbone weights
                - score_min: Original score minimum (default: 6.0)
                - score_max: Original score maximum (default: 30.0)
                - keyframe_strategy: Static feature keyframe sampling
                - regressor_hidden_dims: Hidden dimensions for regressor
        """
        super().__init__()

        self.config = config
        self.static_dim = config.get('static_dim', 512)
        self.dynamic_dim = config.get('dynamic_dim', 1024)

        # Multi-clip parameters
        self.clip_length = config.get('clip_length', 16)
        self.clip_stride = config.get('clip_stride', 10)
        self.max_clips = config.get('max_clips', None)

        # Expected number of clips for regressor initialization
        # If max_clips is set, use that. Otherwise use expected_clips or default 10
        self.expected_clips = self.max_clips if self.max_clips is not None else config.get('expected_clips', 10)

        # Score range parameters
        self.score_min = config.get('score_min', 6.0)
        self.score_max = config.get('score_max', 30.0)
        self.score_range = self.score_max - self.score_min

        # Static feature keyframe strategy
        self.keyframe_strategy = config.get('keyframe_strategy', 'middle')

        print("\n" + "="*70)
        print("Initializing Surgical QA Model (Multi-Clip + Bounded Version)")
        print("="*70)
        print(f"  Static dim per clip: {self.static_dim}")
        print(f"  Dynamic dim per clip: {self.dynamic_dim}")
        print(f"  Clip length: {self.clip_length}")
        print(f"  Clip stride: {self.clip_stride}")
        print(f"  Max clips: {self.max_clips if self.max_clips else 'auto'}")
        print(f"  Expected clips (for regressor): {self.expected_clips}")
        print(f"  Score range: [{self.score_min}, {self.score_max}]")
        print(f"  Keyframe strategy: {self.keyframe_strategy}")
        print("="*70 + "\n")

        # 1. Static Feature Extractor with Multi-Clip Support
        print("Initializing Static Feature Extractor (ResNet-34) with Multi-Clip...")
        self.static_extractor = StaticFeatureMultiClip(
            resnet_path=config.get('resnet_path', None),
            #use_pretrained=config.get('use_pretrained', True),
            use_pretrained=config.get('use_pretrained_resnet', True),
            #freeze_early_layers=config.get('freeze_backbone', True),
            freeze_early_layers=True,  # 🌟 强行设为 False，保证 ResNet34 始终全量参与训练！
            output_dim=self.static_dim,
            keyframe_strategy=self.keyframe_strategy
        )

        # 2. Dynamic Feature Extractor with Multi-Clip Support
        print("Initializing Dynamic Feature Extractor (I3D) with Multi-Clip...")
        self.dynamic_extractor = DynamicFeatureMultiClip(
            i3d_path=config.get('i3d_path', None),
            use_pretrained_i3d=config.get('use_pretrained_i3d', True),
            #use_pretrained_i3d=config.get('use_pretrained_i3d', False),
            output_dim=self.dynamic_dim,
            freeze_backbone=config.get('freeze_backbone', True),
            use_mixed_conv=config.get('use_mixed_conv', True),
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            max_clips=self.max_clips,
            # 🌟 新增这一行：将开关透传给动态特征提取器
            use_early_fusion=config.get('use_early_fusion', False)
        )

        # 3. Fusion and Regression Module
        # Initialize in __init__ with expected number of clips
        # This ensures optimizer can find all parameters from the start
        '''
        self.total_clip_dim = self.static_dim + self.dynamic_dim

        # === 🌟 新增：Transformer 时序建模模块 ===
        print("Initializing Transformer Encoder for Temporal Modeling...")
        # 1. 初始化位置编码，max_len 设为 150，足够容纳 max_clips: 71
        self.pos_encoder = PositionalEncoding(d_model=self.total_clip_dim, max_len=150)
        
        # 2. 从 config 中读取超参数 (如果没有则使用默认值)
        num_heads = self.config.get('transformer_heads', 8)       # 8个注意力头
        num_layers = self.config.get('transformer_layers', 2)     # 2层 Transformer
        dim_feedforward = self.config.get('transformer_ffn', 2048)# 前馈网络维度
        dropout = self.config.get('transformer_dropout', 0.1)     # Dropout 比例

        # 3. 定义 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_clip_dim,       # 输入特征维度 1536
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True                   # ⚠️ 关键：确保输入格式为 (Batch, Seq, Feature)
        )
        # 4. 定义完整的 Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # ========================================

        self.fusion_regressor = BoundedFusionRegressorMultiClip(
            #input_dim=self.expected_clips * self.total_clip_dim,
            input_dim=self.total_clip_dim,
            hidden_dims=self.config.get('regressor_hidden_dims', [1024, 512, 256, 128]),
            dropout_rate=self.config.get('regressor_dropout', 0.5)
        )
        '''
        self.total_clip_dim = self.static_dim + self.dynamic_dim

        # ==========================================================
        # 🌟 核心修改：动态瓶颈层 (Bottleneck Layer)
        # ==========================================================
        self.use_bottleneck = self.config.get('use_bottleneck', False)
        self.bottleneck_dim = self.config.get('bottleneck_dim', 128)

        if self.use_bottleneck:
            print(f"  [Bottleneck ENABLED] 压缩特征: {self.total_clip_dim}D -> {self.bottleneck_dim}D")
            self.feature_compressor = nn.Sequential(
                nn.Linear(self.total_clip_dim, self.bottleneck_dim),
                nn.LayerNorm(self.bottleneck_dim),
                nn.GELU(),
                nn.Dropout(0.5)
            )
            transformer_input_dim = self.bottleneck_dim
            # 如果开启了严重降维，回归头也应该随之精简，避免头重脚轻
            default_hidden_dims = [64, 32] 
        else:
            print(f"  [Bottleneck DISABLED] 保持原样: {self.total_clip_dim}D 直接进入 Transformer")
            self.feature_compressor = nn.Identity() # 占位符，不改变任何特征和维度
            transformer_input_dim = self.total_clip_dim
            default_hidden_dims = [1024, 512, 256, 128] # 恢复原版的大回归头

        # ==========================================================
        # 接下来，所有的组件都使用动态算出来的 `transformer_input_dim`
        # ==========================================================
        print("Initializing Transformer Encoder for Temporal Modeling...")
        
        # 1. 位置编码的维度根据开关动态变化
        self.pos_encoder = PositionalEncoding(d_model=transformer_input_dim, max_len=150)
        
        num_heads = self.config.get('transformer_heads', 4)       
        num_layers = self.config.get('transformer_layers', 1)     
        dim_feedforward = self.config.get('transformer_ffn', 512)
        dropout = self.config.get('transformer_dropout', 0.3)     

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,   # 👈 动态变化维度
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True                   
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fusion_regressor = BoundedFusionRegressorMultiClip(
            input_dim=transformer_input_dim, # 👈 回归头的输入也动态变化
            hidden_dims=self.config.get('regressor_hidden_dims', default_hidden_dims),
            dropout_rate=self.config.get('regressor_dropout', 0.5)
        )
        '''
        # ====== 🌟 新增：动态权重与分布锚点注册 ======
        self.use_mean_penalty = self.config.get('use_mean_penalty', False)
        self.use_dynamic_weights = self.config.get('use_dynamic_weights', False)
        
        if self.use_dynamic_weights:
            # 判断需要几个动态参数：如果开了均值锚点就是 3 个，否则是 2 个
            num_vars = 3 if self.use_mean_penalty else 2
            print(f"  [Auto-Loss] 开启自适应权重！模型将动态平衡 {num_vars} 个 Loss 的占比。")
            
            # 初始化可学习的标量参数，初始值为 0
            # log_vars[0]: 分数回归(MAE), log_vars[1]: 排序(Ranking), log_vars[2]: 分布锚点(Mean)
            self.log_vars = nn.Parameter(torch.zeros(num_vars))
        else:
            print("  [Auto-Loss] 动态权重未开启，使用固定超参数权重。")
        '''
        # ====== 🌟 新增：动态权重与多个辅助 Loss 注册 ======
        self.use_mean_penalty = self.config.get('use_mean_penalty', False)
        self.use_tie_loss = self.config.get('use_tie_loss', False)
        self.use_dynamic_weights = self.config.get('use_dynamic_weights', False)
        
        if self.use_dynamic_weights:
            # 采用字典来动态管理可学习参数的索引，极其安全！
            self.loss_indices = {'score': 0, 'rank': 1}
            num_vars = 2
            
            if self.use_mean_penalty:
                self.loss_indices['mean'] = num_vars
                num_vars += 1
                
            if self.use_tie_loss:
                self.loss_indices['tie'] = num_vars
                num_vars += 1
                
            self.log_vars = nn.Parameter(torch.zeros(num_vars))
            print(f"  [Auto-Loss] 开启自适应权重！模型将动态平衡 {num_vars} 个 Loss 的占比。")
        else:
            print("  [Auto-Loss] 动态权重未开启，使用固定超参数权重。")

        
        print(f"Fusion Regressor initialized with {self.expected_clips} clips")
        print(f"  Input dimension: {self.expected_clips * self.total_clip_dim}")
        print("Model initialized successfully!")

    def _per_clip_fusion(self, static_per_clip, dynamic_per_clip):
        """
        Fuse static and dynamic features per clip.

        方案A: 先按clip融合动静特征

        Args:
            static: (B, num_clips, static_dim)
            dynamic: (B, num_clips, dynamic_dim)

        Returns:
            fused_per_clip: (B, num_clips, static_dim + dynamic_dim)
        """
        # Concatenate along feature dimension: (B, num_clips, static_dim + dynamic_dim)
        fused = torch.cat([static_per_clip, dynamic_per_clip], dim=-1)
        return fused

    def forward(self, video, masks=None, return_features=False):
        """
        Forward pass for surgical video quality assessment.

        Args:
            video: (B, C, T, H, W) - Entire video
            masks: (B, T, H, W) or None - Optional instrument masks for mask-guided attention
            return_features: Return intermediate features for analysis

        Returns:
            score_normalized: (B, 1) - Predicted quality score in [0, 1]
            If return_features: also return features_dict
        """
        B, _, _, _, _ = video.shape

        # 1. Extract multi-clip static features
        # 每个clip采样一个关键帧，提取ResNet特征
        static_per_clip, num_clips_static = self.static_extractor.extract_multiclip_features(
            video,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            max_clips=self.max_clips
        )
        # static_per_clip: (B, num_clips, 512)

        # 2. Extract multi-clip dynamic features
        # 每个clip提取I3D特征，并应用mask-guided attention
        dynamic_per_clip, num_clips_dynamic = self.dynamic_extractor.extract_multiclip_features(video, masks)
        # dynamic_per_clip: (B, num_clips, 1024)

        # 确保clips数量一致
        num_clips = min(num_clips_static, num_clips_dynamic)
        # ==========================================
        # 🌟 插入你的监控探头：只打印第一个 Batch 的信息防止刷屏
        if random.random() < 0.05:  # 有 5% 的概率会在终端打印出来，足够你抽查了
            print(f"\n👉 [Debug] 当前视频的真实物理长度切出了: {num_clips} 个 Clips。即将被对齐到 {self.expected_clips} 个！")
        # ==========================================
        if num_clips_static != num_clips_dynamic:
            print(f"Warning: Static clips ({num_clips_static}) != Dynamic clips ({num_clips_dynamic}), using {num_clips}")

        if num_clips_static > num_clips:
            static_per_clip = static_per_clip[:, :num_clips, :]
        elif num_clips_dynamic > num_clips:
            dynamic_per_clip = dynamic_per_clip[:, :num_clips, :]

        # 3. Per-clip fusion: [Static_clip, Dynamic_clip] -> Fused_clip
        fused_per_clip = self._per_clip_fusion(static_per_clip, dynamic_per_clip)
        # fused_per_clip: (B, num_clips, 512 + 1024) = (B, num_clips, 1536)
        '''
        # 4. Temporal concatenation: Flatten to preserve temporal order
        # (B, num_clips, 1536) -> (B, num_clips * 1536)
        temporal_features = fused_per_clip.reshape(B, -1)

        # 5. Handle case where actual num_clips differs from expected_clips
        actual_input_dim = temporal_features.shape[1]
        expected_input_dim = self.expected_clips * self.total_clip_dim
        if actual_input_dim != expected_input_dim:
            # Need to adjust features to match expected dimension
            if actual_input_dim < expected_input_dim:
                # Pad with zeros
                padding = torch.zeros(B, expected_input_dim - actual_input_dim, device=video.device)
                temporal_features = torch.cat([temporal_features, padding], dim=1)
            else:
                # Truncate
                temporal_features = temporal_features[:, :expected_input_dim]
        '''
        # === 🌟 核心修改点：Transformer 序列交互 ===
        # 4.1 注入位置编码 (让模型知道每个 clip 的时间先后顺序)
        #fused_per_clip_pe = self.pos_encoder(fused_per_clip)
        # 🌟 送入位置编码和 Transformer 之前，先过一遍压缩器
        # 如果 use_bottleneck 为 True，这里会把 1152维 压缩成 128维
        # 如果 use_bottleneck 为 False，feature_compressor 是 nn.Identity，原样输出 1152维
        fused_compressed = self.feature_compressor(fused_per_clip)
        
        # 4.1 注入位置编码 
        fused_per_clip_pe = self.pos_encoder(fused_compressed)
        # 4.2 经过 Transformer，进行全局 Self-Attention
        # 每个 clip 会去“观察”其他所有的 clip，寻找关键的动作或失误
        # trans_out 形状仍然是 (B, num_clips, 1536)，但包含了全局上下文
        trans_out = self.transformer_encoder(fused_per_clip_pe)
        # ==========================================
        # 4 & 5. 【修改点】：时序平均池化 (Temporal Mean Pooling)
        # 对 num_clips 维度（维度索引为 1）求平均值
        # 无论视频被切成了 5 个 clip 还是 71 个 clip，这里都会被安全地聚合成 (B, 1536)
        temporal_features = trans_out.mean(dim=1) 
        # temporal_features: (B, 1536)

        # 6. Regress to score (using fused temporal features directly)
        score_normalized = self.fusion_regressor(temporal_features)
        # score_normalized: (B, 1) in [0, 1]

        # Return based on flags
        if return_features:
            features_dict = {
                'static_per_clip': static_per_clip,
                'dynamic_per_clip': dynamic_per_clip,
                'fused_per_clip': fused_per_clip,
                'temporal_features': temporal_features,
                'num_clips': num_clips,
                'clip_length': self.clip_length,
                'clip_stride': self.clip_stride,
                'per_clip_dim': self.total_clip_dim,
                'total_dim': temporal_features.shape[1]
            }
            return score_normalized, features_dict
        else:
            return score_normalized

    def denormalize_score(self, score_normalized, target_min=None, target_max=None,
                        norm_min=None, norm_max=None):
        """
        反归一化：将归一化的分数从 [norm_min, norm_max] 转换回目标范围 [target_min, target_max]

        标准使用场景（JIGSAWS数据集）：
        1. 训练时原始分数 [6, 30] → 归一化到 [0, 1]
        2. 网络输出 [0, 1] → 反归一化回 [6, 30]（JIGSAWS原始范围，默认）
        3. 网络输出 [0, 1] → 反归一化到 [1, 10]（论文分数，用于对比）

        数学公式：
            target = (normalized - norm_min) / (norm_max - norm_min) * (target_max - target_min) + target_min

        Args:
            score_normalized: (B,) or (B, 1) - 归一化后的预测分数 [0, 1]
            target_min: 目标范围的最小值（如 6.0 或 1.0），默认使用 self.score_min (6.0)
            target_max: 目标范围的最大值（如 30.0 或 10.0），默认使用 self.score_max (30.0)
            norm_min: 归一化范围的下限（通常是 0.0），默认 0.0
            norm_max: 归一化范围的上限（通常是 1.0），默认 1.0

        Returns:
            score_target: 反归一化后的分数，范围 [target_min, target_max]

        Examples:
            # 将 [0,1] 映射回 JIGSAWS 原始范围 [6,30]（默认，不传参数）
            score_6to30 = model.denormalize_score(score_norm)

            # 或者显式指定
            score_6to30 = model.denormalize_score(score_norm, target_min=6.0, target_max=30.0)

            # 将 [0,1] 映射到论文范围 [1,10]（用于与其他论文对比）
            score_1to10 = model.denormalize_score(score_norm, target_min=1.0, target_max=10.0)
        """
        # 处理输入维度
        if score_normalized.dim() == 2:
            score_normalized = score_normalized.squeeze(-1)
        elif score_normalized.dim() == 0:
            score_normalized = score_normalized.unsqueeze(0)

        # 使用传入参数或默认值
        norm_min = norm_min if norm_min is not None else 0.0
        norm_max = norm_max if norm_max is not None else 1.0
        target_min = target_min if target_min is not None else self.score_min
        target_max = target_max if target_max is not None else self.score_max

        # 计算范围
        norm_range = norm_max - norm_min
        target_range = target_max - target_min

        # 反归一化公式
        if norm_range == 0:
            # 防止除零
            score_target = torch.full_like(score_normalized, target_min)
        else:
            score_target = (score_normalized - norm_min) / norm_range * target_range + target_min

        return score_target
    
    
    def compute_loss(self, score_pred, score_gt):
        """
        高度可配置的计算 Loss 函数 (Tie-Loss 终极版)：
        支持：分数(Score) + 排序(Rank) + 锚点(Mean) + 同分聚合(Tie)
        """
        score_pred_flat = score_pred.view(-1)
        score_gt_flat = score_gt.view(-1)

        # ==========================================
        # 模块 A：计算基础的分数 Loss (Score Loss)
        # ==========================================
        score_loss_type = self.config.get('score_loss_type', 'mae').lower()
        if score_loss_type == 'mse':
            score_loss = F.mse_loss(score_pred_flat, score_gt_flat)
        elif score_loss_type == 'mae':
            score_loss = F.l1_loss(score_pred_flat, score_gt_flat)
        else:
            raise ValueError(f"未知的 score_loss_type: {score_loss_type}")

        loss_type = self.config.get('loss_type', 'score_plus_rank').lower()

        if loss_type == 'score_only':
            loss_dict = {
                'total_loss': score_loss.item(), 'score_loss': score_loss.item(),
                'rank_loss': 0.0, 'mean_loss': 0.0, 'tie_loss': 0.0
            }
            return score_loss, loss_dict

        # ==========================================
        # 模块 B：组合模式 (分数 + 排序 + 锚点 + 同分聚合)
        # ==========================================
        elif loss_type == 'score_plus_rank':
            batch_size = score_pred_flat.size(0)
            rank_loss = torch.tensor(0.0, device=score_pred.device)
            tie_loss = torch.tensor(0.0, device=score_pred.device)
            use_tie_loss = self.config.get('use_tie_loss', False)

            if batch_size > 1:
                pred_i = score_pred_flat.unsqueeze(1).expand(batch_size, batch_size)
                pred_j = score_pred_flat.unsqueeze(0).expand(batch_size, batch_size)
                gt_i = score_gt_flat.unsqueeze(1).expand(batch_size, batch_size)
                gt_j = score_gt_flat.unsqueeze(0).expand(batch_size, batch_size)
                
                # --- 1.1 排斥力 (Ranking Loss) ---
                
                # --- 1.1 排斥力 (Ranking Loss) ---
                mask_diff = gt_i > gt_j
                if mask_diff.sum() > 0:
                    # 🌟 读取 YAML 中的开关，默认设置为 True (推荐)
                    use_dynamic_margin = self.config.get('use_dynamic_margin', True)
                    
                    if use_dynamic_margin:
                        # 【分支 A：动态 Margin】提取这对视频的真实归一化分差，作为自适应安全距离
                        dynamic_margin = gt_i[mask_diff] - gt_j[mask_diff]
                        # 使用 F.relu 完美复刻 margin_ranking_loss: max(0, -(Pred_i - Pred_j) + Margin)
                        rank_loss = F.relu(-(pred_i[mask_diff] - pred_j[mask_diff]) + dynamic_margin).mean()
                    else:
                        # 【分支 B：固定 Margin】回退到传统的统一常数距离模式
                        target = torch.ones_like(pred_i[mask_diff])
                        margin = self.config.get('rank_margin', 0.05)
                        rank_loss = F.margin_ranking_loss(
                            pred_i[mask_diff], pred_j[mask_diff], target, margin=margin
                        )

                # --- 1.2 吸引力 (Tie Loss) 🌟 新增核心逻辑 ---
                if use_tie_loss:
                    # 创建对角线掩码（防止视频自己跟自己算损失）
                    eye_mask = torch.eye(batch_size, dtype=torch.bool, device=score_pred.device)
                    # 找出真值相等，且不是同一个视频的 Pairs
                    mask_tie = (gt_i == gt_j) & (~eye_mask)
                    
                    if mask_tie.sum() > 0:
                        # 强行让同分视频的预测值靠近（使用 MSE 效果更好，差距越大惩罚越重）
                        tie_loss = F.mse_loss(pred_i[mask_tie], pred_j[mask_tie])

            # --- 2. 计算分布锚点 Loss (Mean Loss) ---
            use_mean_penalty = self.config.get('use_mean_penalty', False)
            mean_loss = torch.tensor(0.0, device=score_pred.device)
            if use_mean_penalty:
                mean_loss = F.l1_loss(score_pred_flat.mean(), score_gt_flat.mean())

            # --- 3. 终极融合：动态 vs 手动 ---
            use_dynamic_weights = self.config.get('use_dynamic_weights', False)
            
            if use_dynamic_weights:
                # [处理 Score]
                idx_score = self.loss_indices['score']
                precision_score = torch.exp(-self.log_vars[idx_score])
                loss_score_weighted = precision_score * score_loss + self.log_vars[idx_score]
                
                # [处理 Rank]
                idx_rank = self.loss_indices['rank']
                if isinstance(rank_loss, torch.Tensor) and rank_loss.requires_grad:
                    precision_rank = torch.exp(-self.log_vars[idx_rank])
                    loss_rank_weighted = precision_rank * rank_loss + self.log_vars[idx_rank]
                else:
                    loss_rank_weighted = 0.0
                
                # [处理 Mean]
                if use_mean_penalty:
                    idx_mean = self.loss_indices['mean']
                    precision_mean = torch.exp(-self.log_vars[idx_mean])
                    loss_mean_weighted = precision_mean * mean_loss + self.log_vars[idx_mean]
                else:
                    loss_mean_weighted = 0.0

                # [处理 Tie]
                if use_tie_loss:
                    idx_tie = self.loss_indices['tie']
                    # ⚠️ 防爆零保护：如果这个 Batch 碰巧没有同分视频，绝对不能计算 log_vars 惩罚！
                    if isinstance(tie_loss, torch.Tensor) and tie_loss.requires_grad:
                        precision_tie = torch.exp(-self.log_vars[idx_tie])
                        loss_tie_weighted = precision_tie * tie_loss + self.log_vars[idx_tie]
                    else:
                        loss_tie_weighted = 0.0
                else:
                    loss_tie_weighted = 0.0
                    
                total_loss = loss_score_weighted + loss_rank_weighted + loss_mean_weighted + loss_tie_weighted
                
            else:
                # 【模式：手动固定权重】
                lambda_rank = self.config.get('lambda_rank', 1.0)
                lambda_mean = self.config.get('lambda_mean', 1.0) if use_mean_penalty else 0.0
                lambda_tie = self.config.get('lambda_tie', 1.0) if use_tie_loss else 0.0
                
                total_loss = score_loss + (lambda_rank * rank_loss) + (lambda_mean * mean_loss) + (lambda_tie * tie_loss)

            # --- 4. 日志输出 ---
            loss_dict = {
                'total_loss': total_loss.item(),
                'score_loss': score_loss.item(),  
                'rank_loss': rank_loss.item() if isinstance(rank_loss, torch.Tensor) else 0.0,
                'mean_loss': mean_loss.item() if isinstance(mean_loss, torch.Tensor) else 0.0,
                'tie_loss': tie_loss.item() if isinstance(tie_loss, torch.Tensor) else 0.0
            }
            
            # (可选) 记录自适应权重曲线
            if use_dynamic_weights:
                loss_dict['weight_score'] = torch.exp(-self.log_vars[self.loss_indices['score']]).item()
                loss_dict['weight_rank']  = torch.exp(-self.log_vars[self.loss_indices['rank']]).item()
                if use_mean_penalty:
                    loss_dict['weight_mean'] = torch.exp(-self.log_vars[self.loss_indices['mean']]).item()
                if use_tie_loss:
                    loss_dict['weight_tie'] = torch.exp(-self.log_vars[self.loss_indices['tie']]).item()

            return total_loss, loss_dict

        else:
            raise ValueError(f"未知的总 loss_type: {loss_type}")
    '''
    def compute_loss(self, score_pred, score_gt):
        """
        高度可配置的计算 Loss 函数 (终极版)：
        支持：1.纯分数 2.分数+排序+锚点(手动权重) 3.分数+排序+锚点(自动学习权重)
        """
        # 1. 展平为 1D Tensor: (B,)
        score_pred_flat = score_pred.view(-1)
        score_gt_flat = score_gt.view(-1)

        # ==========================================
        # 模块 A：计算基础的分数 Loss (Score Loss)
        # ==========================================
        score_loss_type = self.config.get('score_loss_type', 'mae').lower()
        if score_loss_type == 'mse':
            score_loss = F.mse_loss(score_pred_flat, score_gt_flat)
        elif score_loss_type == 'mae':
            score_loss = F.l1_loss(score_pred_flat, score_gt_flat)
        else:
            raise ValueError(f"未知的 score_loss_type: {score_loss_type}")

        loss_type = self.config.get('loss_type', 'score_plus_rank').lower()

        # ==========================================
        # 模式 1：纯净模式，只用 MSE / MAE
        # ==========================================
        if loss_type == 'score_only':
            loss_dict = {
                'total_loss': score_loss.item(),
                'score_loss': score_loss.item(),
                'rank_loss': 0.0,
                'mean_loss': 0.0
            }
            return score_loss, loss_dict

        # ==========================================
        # 模式 2 & 3：组合模式 (分数 + 排序 + 可选锚点)
        # ==========================================
        elif loss_type == 'score_plus_rank':
            # --- 1. 计算排序 Loss (Rank Loss) ---
            batch_size = score_pred_flat.size(0)
            rank_loss = torch.tensor(0.0, device=score_pred.device)
            if batch_size > 1:
                pred_i = score_pred_flat.unsqueeze(1).expand(batch_size, batch_size)
                pred_j = score_pred_flat.unsqueeze(0).expand(batch_size, batch_size)
                gt_i = score_gt_flat.unsqueeze(1).expand(batch_size, batch_size)
                gt_j = score_gt_flat.unsqueeze(0).expand(batch_size, batch_size)
                
                mask = gt_i > gt_j
                if mask.sum() > 0:
                    target = torch.ones_like(pred_i[mask])
                    margin = self.config.get('rank_margin', 0.15)
                    rank_loss = F.margin_ranking_loss(
                        pred_i[mask], pred_j[mask], target, margin=margin
                    )

            # --- 2. 计算分布锚点 Loss (Mean Loss) ---
            use_mean_penalty = self.config.get('use_mean_penalty', False)
            mean_loss = torch.tensor(0.0, device=score_pred.device)
            if use_mean_penalty:
                mean_loss = F.l1_loss(score_pred_flat.mean(), score_gt_flat.mean())

            # --- 3. 终极融合：判断是手动参数还是可学习参数 ---
            use_dynamic_weights = self.config.get('use_dynamic_weights', False)
            
            if use_dynamic_weights:
                # 【模式 3：动态可学习权重】
                # 处理 Score Loss
                precision_score = torch.exp(-self.log_vars[0])
                loss_score_weighted = precision_score * score_loss + self.log_vars[0]
                
                # 处理 Rank Loss
                precision_rank = torch.exp(-self.log_vars[1])
                loss_rank_weighted = precision_rank * rank_loss + self.log_vars[1]
                
                # 处理 Mean Loss
                if use_mean_penalty:
                    precision_mean = torch.exp(-self.log_vars[2])
                    loss_mean_weighted = precision_mean * mean_loss + self.log_vars[2]
                else:
                    loss_mean_weighted = 0.0
                    
                total_loss = loss_score_weighted + loss_rank_weighted + loss_mean_weighted
                
            else:
                # 【模式 2：手动固定权重】
                lambda_rank = self.config.get('lambda_rank', 2.0)
                lambda_mean = self.config.get('lambda_mean', 1.0) if use_mean_penalty else 0.0
                
                total_loss = score_loss + lambda_rank * rank_loss + lambda_mean * mean_loss

            # --- 4. 构造日志输出字典 ---
            loss_dict = {
                'total_loss': total_loss.item(),
                'score_loss': score_loss.item(),  
                'rank_loss': rank_loss.item() if isinstance(rank_loss, torch.Tensor) else 0.0,
                'mean_loss': mean_loss.item() if isinstance(mean_loss, torch.Tensor) else 0.0
            }
            
            # (可选) 将学到的动态权重数值顺便记录下来，方便调试时查看
            if use_dynamic_weights:
                loss_dict['weight_score'] = torch.exp(-self.log_vars[0]).item()
                loss_dict['weight_rank']  = torch.exp(-self.log_vars[1]).item()
                if use_mean_penalty:
                    loss_dict['weight_mean'] = torch.exp(-self.log_vars[2]).item()

            return total_loss, loss_dict

        else:
            raise ValueError(f"未知的总 loss_type: {loss_type}")
    '''

    def unfreeze_backbone(self, layers_to_unfreeze=['all']):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            layers_to_unfreeze: List of layers to unfreeze
                              ['all'] -> unfreeze all
                              ['layer4'] -> unfreeze only layer4
        """
        if 'all' in layers_to_unfreeze:
            for param in self.static_extractor.backbone.parameters():
                param.requires_grad = True
            for param in self.dynamic_extractor.feature_extractor.parameters():
                param.requires_grad = True
        else:
            for layer_name in layers_to_unfreeze:
                for name, param in self.static_extractor.backbone.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True
                for name, param in self.dynamic_extractor.feature_extractor.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True

        print(f"Unfroze layers: {layers_to_unfreeze}")

    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return total, trainable


def build_model_multiclip_bounded(config):
    """
    Factory function to build Multi-Clip Bounded Surgical QA Model.

    Args:
        config: Configuration dict or path to config file

    Returns:
        model: SurgicalQAModelMultiClipBounded instance
    """
    if isinstance(config, str):
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    model = SurgicalQAModelMultiClipBounded(config)

    # Load checkpoint if provided
    if 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {config['checkpoint_path']}")

    return model


if __name__ == '__main__':
    # Test model
    print("Testing SurgicalQAModelMultiClipBounded...")

    config = {
        'static_dim': 512,
        'dynamic_dim': 1024,
        'clip_length': 16,
        'clip_stride': 10,
        'max_clips': None,
        'expected_clips': 10,  # Expected number of clips for 100-frame video
        'use_pretrained': False,
        'freeze_backbone': True,
        'use_mixed_conv': True,
        'score_min': 6.0,
        'score_max': 30.0,
        'keyframe_strategy': 'middle',
        'regressor_hidden_dims': [1024, 512, 256, 128]
    }

    model = SurgicalQAModelMultiClipBounded(config)
    model.count_parameters()

    # Test with different video lengths
    test_cases = [
        (2, 3, 100, 224, 224),  # 100 frames -> 10 clips
        (2, 3, 90, 224, 224),    # 90 frames -> 9 clips
        (2, 3, 50, 224, 224),    # 50 frames -> 5 clips
        (2, 3, 16, 224, 224),    # 16 frames -> 1 clip
    ]

    for video_shape in test_cases:
        video = torch.randn(*video_shape)
        score = model(video)

        print(f"\nInput: {video_shape}")
        print(f"  Output score shape: {score.shape}")
        print(f"  Output score range: [{score.min().item():.4f}, {score.max().item():.4f}]")

        # Test with feature return
        score, features = model(video, return_features=True)
        print(f"  Features dict keys: {features.keys()}")
        print(f"  Static per-clip: {features['static_per_clip'].shape}")
        print(f"  Dynamic per-clip: {features['dynamic_per_clip'].shape}")
        print(f"  Fused per-clip: {features['fused_per_clip'].shape}")
        print(f"  Temporal features: {features['temporal_features'].shape}")
        print(f"  Number of clips: {features['num_clips']}")

        # Verify dimensions
        num_clips = features['num_clips']
        expected_total_dim =   512 + 1024

        assert features['static_per_clip'].shape == (video_shape[0], num_clips, 512)
        assert features['dynamic_per_clip'].shape == (video_shape[0], num_clips, 1024)
        assert features['fused_per_clip'].shape == (video_shape[0], num_clips, 1536)
        assert features['temporal_features'].shape == (video_shape[0], expected_total_dim)

    print("\n✓ All tests passed!")
