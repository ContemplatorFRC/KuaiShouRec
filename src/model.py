"""
DIEN-DCN 融合模型架构
用于短视频推荐CTR预测

特征说明:
- 用户特征: user_active_degree, is_live_streamer, is_video_author, follow_user_num, fans_user_num, friend_user_num, register_days
- 视频内容特征: video_type, video_duration, music_type, tag, category_id
- 视频统计特征: show_cnt, play_cnt, play_duration, like_cnt, comment_cnt, share_cnt, collect_cnt
- 时间特征: hour
- 文本特征: caption_embedding (768维 DeBERTa)
- 历史序列特征: history_video_ids (用于DIEN)

注意: post-click特征(is_like, is_comment, is_follow, is_forward, long_view, play_time_ms)不能作为输入特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# =============================================================================
# 特征配置
# =============================================================================
class FeatureConfig:
    """特征配置类"""

    # 数值特征 (需要归一化)
    NUMERIC_FEATURES = [
        'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days',
        'video_duration', 'music_type',
        'show_cnt', 'play_cnt', 'play_duration', 'like_cnt', 'comment_cnt', 'share_cnt', 'collect_cnt',
        'hour'
    ]

    # 类别特征 (需要embedding)
    CATEGORICAL_FEATURES = [
        'user_active_degree',  # 4类: 0-3
        'is_live_streamer',    # 2类: 0/1
        'is_video_author',     # 2类: 0/1
        'video_type',         # 2类: 0/1
        'tag',                # 需要统计
        'category_id',        # 需要统计
    ]

    # 类别特征的词汇表大小 (需要根据实际数据统计更新)
    CATEGORICAL_VOCAB_SIZE = {
        'user_active_degree': 4,
        'is_live_streamer': 2,
        'is_video_author': 2,
        'video_type': 2,
        'tag': 1000,           # 需要根据实际数据调整
        'category_id': 100,    # 需要根据实际数据调整
    }

    # 类别特征embedding维度
    EMBEDDING_DIM = 16

    # 文本embedding维度 (由DeBERTa生成，可通过output_dim参数调整)
    # 默认128维，减少存储空间
    TEXT_EMBEDDING_DIM = 128

    # 历史序列相关
    MAX_SEQ_LEN = 50
    VIDEO_EMBEDDING_DIM = 64  # 视频ID的embedding维度


# =============================================================================
# 辅助模块
# =============================================================================
class Dice(nn.Module):
    """Dice激活函数 (DIEN论文中使用)"""

    def __init__(self, emb_size: int, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.zeros(emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, emb_size]
        Returns:
            [batch_size, emb_size]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.epsilon)
        p = torch.sigmoid(norm_x)
        return p * x + (1 - p) * self.alpha * x


class AuxiliaryNet(nn.Module):
    """辅助网络 (用于DIEN的辅助损失)"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """预测下一个点击概率"""
        return self.net(x)


# =============================================================================
# GRU组件 (DIEN核心)
# =============================================================================
class GRUCell(nn.Module):
    """自定义GRU单元，支持Dice激活"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 重置门和更新门
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # 候选隐藏状态
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.dice = Dice(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
            h: [batch_size, hidden_dim]
        Returns:
            [batch_size, hidden_dim]
        """
        concat = torch.cat([x, h], dim=-1)

        # 重置门
        r = torch.sigmoid(self.W_r(concat))
        # 更新门
        z = torch.sigmoid(self.W_z(concat))

        # 候选隐藏状态
        h_candidate = self.dice(self.W_h(torch.cat([x, r * h], dim=-1)))

        # 新隐藏状态
        h_new = (1 - z) * h + z * h_candidate

        return h_new


class InterestExtractor(nn.Module):
    """兴趣提取层 (DIEN第一层)"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, seq_embed: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embed: [batch_size, seq_len, input_dim]
            seq_len: [batch_size] 实际序列长度
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        # 确保 seq_len 至少为 1，避免 pack_padded_sequence 错误
        seq_len = torch.clamp(seq_len, min=1)

        # Pack序列
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_embed, seq_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output


class InterestEvolvingLayer(nn.Module):
    """兴趣演化层 (DIEN第二层 - AUGRU)"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # AUGRU: Attentional Update Gate GRU
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # 注意力网络
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, interest_sequence: torch.Tensor, target_item: torch.Tensor,
                seq_len: torch.Tensor) -> torch.Tensor:
        """
        Args:
            interest_sequence: [batch_size, seq_len, hidden_dim] 来自兴趣提取层
            target_item: [batch_size, input_dim] 目标物品embedding
            seq_len: [batch_size] 实际序列长度
        Returns:
            [batch_size, hidden_dim] 最终兴趣表示
        """
        batch_size, max_seq_len, _ = interest_sequence.shape

        # 确保 seq_len 至少为 1
        seq_len = torch.clamp(seq_len, min=1)

        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.hidden_dim, device=interest_sequence.device)

        # 创建mask
        mask = torch.arange(max_seq_len, device=interest_sequence.device).unsqueeze(0) < seq_len.unsqueeze(1)
        mask = mask.float()

        for t in range(max_seq_len):
            # 当前时刻的兴趣
            h_t = interest_sequence[:, t, :]  # [batch_size, hidden_dim]

            # 注意力权重
            att_input = torch.cat([h_t, target_item], dim=-1)
            att_score = self.attention(att_input)  # [batch_size, 1]

            # 更新门与注意力融合
            concat = torch.cat([target_item, h], dim=-1)
            z = torch.sigmoid(self.W_z(concat))

            # 注意力加权的更新门
            att_z = att_score * z

            # 重置门
            r = torch.sigmoid(self.W_r(concat))

            # 候选隐藏状态
            h_candidate = torch.tanh(self.W_h(torch.cat([target_item, r * h], dim=-1)))

            # 新隐藏状态 (注意力加权)
            h_new = (1 - att_z) * h + att_z * h_candidate

            # 应用mask
            h = mask[:, t:t+1] * h_new + (1 - mask[:, t:t+1]) * h

        return h


# =============================================================================
# DIEN模块
# =============================================================================
class DIEN(nn.Module):
    """Deep Interest Evolution Network"""

    def __init__(self,
                 video_embedding_dim: int = 64,
                 hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__()

        self.video_embedding_dim = video_embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 兴趣提取层
        self.interest_extractor = InterestExtractor(
            input_dim=video_embedding_dim,
            hidden_dim=hidden_dim
        )

        # 兴趣演化层
        self.interest_evolving = InterestEvolvingLayer(
            input_dim=video_embedding_dim,
            hidden_dim=hidden_dim
        )

        # 辅助网络 (用于计算辅助损失)
        self.auxiliary_net = AuxiliaryNet(
            input_dim=video_embedding_dim + hidden_dim
        )

    def forward(self,
                history_embed: torch.Tensor,
                seq_len: torch.Tensor,
                target_video_embed: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            history_embed: [batch_size, max_seq_len, video_embedding_dim] 历史视频embedding
            seq_len: [batch_size] 实际序列长度
            target_video_embed: [batch_size, video_embedding_dim] 目标视频embedding
        Returns:
            interest: [batch_size, hidden_dim] 最终兴趣表示
            aux_loss: 辅助损失 (训练时使用)
        """
        # 兴趣提取
        interest_sequence = self.interest_extractor(history_embed, seq_len)
        # interest_sequence: [batch_size, seq_len, hidden_dim]

        # 兴趣演化
        final_interest = self.interest_evolving(interest_sequence, target_video_embed, seq_len)

        # 计算辅助损失 (训练时)
        aux_loss = None
        if self.training and history_embed.size(1) > 1:
            # 辅助任务: 预测下一个点击
            # 使用t时刻的兴趣预测t+1时刻的点击
            aux_input = torch.cat([
                interest_sequence[:, :-1, :],  # t时刻的兴趣
                history_embed[:, 1:, :]         # t+1时刻的物品
            ], dim=-1)
            aux_pred = self.auxiliary_net(aux_input)

            # 构建辅助标签
            # 简化版本: 假设用户点击了下一个物品
            # 实际应该使用真实的next-click标签
            aux_target = torch.ones_like(aux_pred)
            aux_loss = F.binary_cross_entropy_with_logits(aux_pred, aux_target)

        return final_interest, aux_loss


# =============================================================================
# DCN模块
# =============================================================================
class CrossNetwork(nn.Module):
    """DCN Cross Network"""

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers

        # 每层的权重和偏置
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim))
            for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            [batch_size, input_dim]
        """
        x0 = x
        for i in range(self.num_layers):
            x = x0 * (x @ self.W[i].unsqueeze(0).T + self.b[i].unsqueeze(0)) + x
        return x


class DeepNetwork(nn.Module):
    """DCN Deep Network (MLP)"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DCN(nn.Module):
    """Deep & Cross Network"""

    def __init__(self,
                 input_dim: int,
                 cross_layers: int = 3,
                 deep_dims: List[int] = [256, 128, 64]):
        super().__init__()

        self.cross = CrossNetwork(input_dim, cross_layers)
        self.deep = DeepNetwork(input_dim, deep_dims)

        # 融合层
        self.output_dim = input_dim + deep_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            [batch_size, output_dim] cross和deep的拼接
        """
        cross_out = self.cross(x)
        deep_out = self.deep(x)
        return torch.cat([cross_out, deep_out], dim=-1)


# =============================================================================
# 特征编码器
# =============================================================================
class FeatureEncoder(nn.Module):
    """特征编码器"""

    def __init__(self,
                 config: FeatureConfig,
                 video_vocab_size: int = 3000000,  # 视频ID词汇表大小
                 video_embedding_dim: int = 64):
        super().__init__()
        self.config = config

        # 类别特征embedding
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, config.EMBEDDING_DIM)
            for name, vocab_size in config.CATEGORICAL_VOCAB_SIZE.items()
        })

        # 视频ID embedding (用于历史序列和目标视频)
        self.video_embedding = nn.Embedding(video_vocab_size, video_embedding_dim, padding_idx=0)

        # 文本embedding投影层 (将TEXT_EMBEDDING_DIM降到64维，并LayerNorm归一化)
        self.text_projection = nn.Sequential(
            nn.Linear(config.TEXT_EMBEDDING_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64)  # 归一化，使输出范围与其他特征一致
        )

        # 数值特征归一化参数 (需要在训练前统计)
        self.register_buffer('numeric_mean', torch.zeros(len(config.NUMERIC_FEATURES)))
        self.register_buffer('numeric_std', torch.ones(len(config.NUMERIC_FEATURES)))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: 包含各种特征的字典
        Returns:
            [batch_size, total_feature_dim] 编码后的特征向量
        """
        features = []

        # 类别特征embedding
        for name in self.config.CATEGORICAL_FEATURES:
            if name in batch:
                embed = self.categorical_embeddings[name](batch[name].long())
                features.append(embed)

        # 数值特征 (VideoRecDataset返回预处理好的numeric_features tensor)
        if 'numeric_features' in batch:
            # 已经在Dataset中归一化过，直接使用
            features.append(batch['numeric_features'])
        else:
            # 兼容旧格式：单独的数值特征列
            numeric_features = []
            for name in self.config.NUMERIC_FEATURES:
                if name in batch:
                    numeric_features.append(batch[name].unsqueeze(-1))
            if numeric_features:
                numeric = torch.cat(numeric_features, dim=-1)
                numeric = (numeric - self.numeric_mean) / (self.numeric_std + 1e-8)
                features.append(numeric)

        # 文本embedding
        if 'caption_embedding' in batch:
            text_feat = self.text_projection(batch['caption_embedding'])
            features.append(text_feat)

        # 拼接所有特征
        return torch.cat(features, dim=-1)

    def encode_history(self, history_video_ids: torch.Tensor) -> torch.Tensor:
        """
        编码历史序列
        Args:
            history_video_ids: [batch_size, max_seq_len]
        Returns:
            [batch_size, max_seq_len, video_embedding_dim]
        """
        return self.video_embedding(history_video_ids.long())

    def encode_target_video(self, video_id: torch.Tensor) -> torch.Tensor:
        """
        编码目标视频
        Args:
            video_id: [batch_size]
        Returns:
            [batch_size, video_embedding_dim]
        """
        return self.video_embedding(video_id.long())


# =============================================================================
# DIEN-DCN融合模型
# =============================================================================
class DIENDCN(nn.Module):
    """DIEN-DCN融合模型用于CTR预测"""

    def __init__(self,
                 config: FeatureConfig,
                 video_vocab_size: int = 3000000,
                 video_embedding_dim: int = 64,
                 dien_hidden_dim: int = 64,
                 dcn_cross_layers: int = 3,
                 dcn_deep_dims: List[int] = [256, 128, 64],
                 mlp_dims: List[int] = [256, 128, 64]):
        super().__init__()
        self.config = config

        # 特征编码器
        self.feature_encoder = FeatureEncoder(config, video_vocab_size, video_embedding_dim)

        # DIEN模块
        self.dien = DIEN(
            video_embedding_dim=video_embedding_dim,
            hidden_dim=dien_hidden_dim,
            output_dim=dien_hidden_dim
        )

        # 计算特征总维度
        # 类别特征: len(CATEGORICAL_FEATURES) * EMBEDDING_DIM
        # 数值特征: len(NUMERIC_FEATURES)
        # 文本特征: 64 (投影后)
        # DIEN输出: dien_hidden_dim
        categorical_dim = len(config.CATEGORICAL_FEATURES) * config.EMBEDDING_DIM
        numeric_dim = len(config.NUMERIC_FEATURES)
        text_dim = 64
        dien_dim = dien_hidden_dim

        # DCN输入维度 (不包含DIEN输出)
        dcn_input_dim = categorical_dim + numeric_dim + text_dim

        # DCN模块
        self.dcn = DCN(
            input_dim=dcn_input_dim,
            cross_layers=dcn_cross_layers,
            deep_dims=dcn_deep_dims
        )

        # 最终MLP
        # 输入: DCN输出 + DIEN输出
        final_input_dim = self.dcn.output_dim + dien_dim
        mlp_layers = []
        prev_dim = final_input_dim
        for hidden_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # 输出层
        self.output = nn.Linear(mlp_dims[-1], 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: 包含以下字段的字典
                - 类别特征: user_active_degree, is_live_streamer, etc.
                - 数值特征: follow_user_num, fans_user_num, etc.
                - caption_embedding: [batch_size, 768]
                - history_video_ids: [batch_size, max_seq_len]
                - history_length: [batch_size] 实际序列长度
                - video_id: [batch_size] 目标视频ID
        Returns:
            dict:
                - logits: [batch_size, 1] CTR预测logits
                - aux_loss: 辅助损失 (训练时)
        """
        # 编码基础特征
        base_features = self.feature_encoder(batch)

        # 编码历史序列
        history_embed = self.feature_encoder.encode_history(batch['history_video_ids'])

        # 编码目标视频
        target_video_embed = self.feature_encoder.encode_target_video(batch['video_id'])

        # DIEN处理历史序列
        dien_output, aux_loss = self.dien(
            history_embed,
            batch['history_length'],
            target_video_embed
        )

        # DCN处理基础特征
        dcn_output = self.dcn(base_features)

        # 融合DIEN和DCN输出
        combined = torch.cat([dcn_output, dien_output], dim=-1)

        # MLP
        mlp_output = self.mlp(combined)

        # 输出
        logits = self.output(mlp_output)

        return {
            'logits': logits,
            'aux_loss': aux_loss
        }

    def compute_loss(self,
                     pred: Dict[str, torch.Tensor],
                     labels: torch.Tensor,
                     aux_weight: float = 0.1) -> torch.Tensor:
        """
        计算总损失
        Args:
            pred: 模型输出
            labels: [batch_size] 点击标签
            aux_weight: 辅助损失权重
        Returns:
            总损失
        """
        # 主损失: CTR预测损失
        main_loss = F.binary_cross_entropy_with_logits(
            pred['logits'].squeeze(-1),
            labels.float()
        )

        # 辅助损失
        if pred['aux_loss'] is not None:
            total_loss = main_loss + aux_weight * pred['aux_loss']
        else:
            total_loss = main_loss

        return total_loss


# =============================================================================
# 数据集类
# =============================================================================
class VideoRecDataset(torch.utils.data.Dataset):
    """视频推荐数据集"""

    def __init__(self,
                 df,  # pandas DataFrame
                 config: FeatureConfig,
                 video_id_to_idx: Dict[int, int],  # video_id到embedding索引的映射
                 caption_embeddings: Optional[np.ndarray] = None,  # video_id对应的caption embedding
                 numeric_mean: Optional[np.ndarray] = None,  # 数值特征均值
                 numeric_std: Optional[np.ndarray] = None,  # 数值特征标准差
                 tag_vocab: Optional[Dict[str, int]] = None,  # tag词汇表
                 is_train: bool = True):
        self.df = df
        self.config = config
        self.video_id_to_idx = video_id_to_idx
        self.caption_embeddings = caption_embeddings
        self.numeric_mean = numeric_mean if numeric_mean is not None else np.zeros(len(config.NUMERIC_FEATURES))
        self.numeric_std = numeric_std if numeric_std is not None else np.ones(len(config.NUMERIC_FEATURES))
        self.tag_vocab = tag_vocab if tag_vocab is not None else {}
        self.is_train = is_train

        # 预处理
        self._precompute_features()

        # 存储 user_id 用于 GAUC 计算
        if 'user_id' in self.df.columns:
            self.user_ids = self.df['user_id'].values
        else:
            self.user_ids = None

    def _precompute_features(self):
        """预计算特征索引"""
        # 将video_id映射到索引
        self.video_indices = self.df['video_id'].map(
            lambda x: self.video_id_to_idx.get(x, 0)
        ).values

        # 处理历史序列
        self.history_indices = []
        self.history_lengths = []
        for history in self.df['history_video_ids']:
            indices = [self.video_id_to_idx.get(int(vid), 0) for vid in history]
            self.history_indices.append(indices)
            self.history_lengths.append(len(indices))

        # 标签
        self.labels = self.df['is_click'].values

        # 预计算tag索引
        if 'tag' in self.df.columns:
            self.tag_indices = self.df['tag'].map(
                lambda x: self.tag_vocab.get(x, 0) if pd.notna(x) else 0
            ).values
        else:
            self.tag_indices = np.zeros(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx]

        # 类别特征
        for name in self.config.CATEGORICAL_FEATURES:
            if name == 'tag':
                sample[name] = torch.tensor(self.tag_indices[idx], dtype=torch.long)
            elif name in row.index and pd.notna(row[name]):
                sample[name] = torch.tensor(int(row[name]), dtype=torch.long)
            else:
                sample[name] = torch.tensor(0, dtype=torch.long)

        # 数值特征 (归一化)
        numeric_values = []
        for name in self.config.NUMERIC_FEATURES:
            if name in row.index and pd.notna(row[name]):
                numeric_values.append(float(row[name]))
            else:
                numeric_values.append(0.0)
        numeric_tensor = torch.tensor(numeric_values, dtype=torch.float32)
        # 归一化
        numeric_tensor = (numeric_tensor - torch.tensor(self.numeric_mean, dtype=torch.float32)) / \
                         (torch.tensor(self.numeric_std, dtype=torch.float32) + 1e-8)
        sample['numeric_features'] = numeric_tensor

        # 文本embedding
        if self.caption_embeddings is not None:
            video_idx = self.video_indices[idx]
            if video_idx < len(self.caption_embeddings):
                sample['caption_embedding'] = torch.tensor(
                    self.caption_embeddings[video_idx], dtype=torch.float32
                )
            else:
                sample['caption_embedding'] = torch.zeros(self.config.TEXT_EMBEDDING_DIM, dtype=torch.float32)
        else:
            sample['caption_embedding'] = torch.zeros(self.config.TEXT_EMBEDDING_DIM, dtype=torch.float32)

        # 历史序列
        history = self.history_indices[idx]
        history_len = len(history)

        # 保留最近的 MAX_SEQ_LEN 个视频（推荐系统中近期兴趣更重要）
        if history_len > self.config.MAX_SEQ_LEN:
            history = history[-self.config.MAX_SEQ_LEN:]  # 截取最近的
            history_len = self.config.MAX_SEQ_LEN

        padded_history = history + [0] * (self.config.MAX_SEQ_LEN - history_len)
        sample['history_video_ids'] = torch.tensor(padded_history, dtype=torch.long)
        # 确保 history_length 至少为 1，避免 pack_padded_sequence 错误
        sample['history_length'] = torch.tensor(max(1, history_len), dtype=torch.long)

        # 目标视频
        sample['video_id'] = torch.tensor(self.video_indices[idx], dtype=torch.long)

        # 标签
        sample['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)

        # user_id (用于 GAUC 计算)
        if self.user_ids is not None:
            sample['user_id'] = torch.tensor(self.user_ids[idx], dtype=torch.long)
        else:
            sample['user_id'] = torch.tensor(0, dtype=torch.long)

        return sample


# =============================================================================
# 模型工具函数
# =============================================================================
def build_video_vocab(df, min_freq: int = 5) -> Tuple[Dict[int, int], int]:
    """
    构建视频ID词汇表
    Args:
        df: 数据DataFrame
        min_freq: 最小出现次数
    Returns:
        video_id_to_idx: video_id到索引的映射
        vocab_size: 词汇表大小
    """
    from collections import Counter

    # 统计视频ID频率
    video_ids = df['video_id'].tolist()
    # 同时考虑历史序列中的视频
    for history in df['history_video_ids']:
        video_ids.extend([int(vid) for vid in history])

    video_freq = Counter(video_ids)

    # 构建映射
    video_id_to_idx = {0: 0}  # 0作为padding
    for video_id, freq in video_freq.items():
        if freq >= min_freq and video_id not in video_id_to_idx:
            video_id_to_idx[video_id] = len(video_id_to_idx)

    vocab_size = len(video_id_to_idx)
    print(f"视频词汇表大小: {vocab_size}")

    return video_id_to_idx, vocab_size


def compute_numeric_stats(df, numeric_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算数值特征的均值和标准差
    """
    means = df[numeric_features].mean().values
    stds = df[numeric_features].std().values
    return means, stds


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义batch整理函数
    """
    result = {}

    # 获取所有键
    keys = batch[0].keys()

    for key in keys:
        tensors = [sample[key] for sample in batch]
        if isinstance(tensors[0], torch.Tensor):
            if tensors[0].dim() == 0:
                result[key] = torch.stack(tensors)
            else:
                result[key] = torch.stack(tensors)
        else:
            result[key] = tensors

    return result


# =============================================================================
# 示例用法
# =============================================================================
if __name__ == "__main__":
    # 配置
    config = FeatureConfig()

    # 创建模型
    model = DIENDCN(
        config=config,
        video_vocab_size=3000000,  # 根据实际数据调整
        video_embedding_dim=64,
        dien_hidden_dim=64,
        dcn_cross_layers=3,
        dcn_deep_dims=[256, 128, 64],
        mlp_dims=[256, 128, 64]
    )

    # 打印模型结构
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 模拟输入测试
    batch_size = 32
    max_seq_len = 50

    mock_batch = {
        'user_active_degree': torch.randint(0, 4, (batch_size,)),
        'is_live_streamer': torch.randint(0, 2, (batch_size,)),
        'is_video_author': torch.randint(0, 2, (batch_size,)),
        'video_type': torch.randint(0, 2, (batch_size,)),
        'tag': torch.randint(0, 1000, (batch_size,)),
        'category_id': torch.randint(0, 100, (batch_size,)),
        'numeric_features': torch.randn(batch_size, len(config.NUMERIC_FEATURES)),
        'caption_embedding': torch.randn(batch_size, 768),
        'history_video_ids': torch.randint(0, 10000, (batch_size, max_seq_len)),
        'history_length': torch.randint(1, max_seq_len + 1, (batch_size,)),
        'video_id': torch.randint(0, 10000, (batch_size,))
    }

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(mock_batch)

    print(f"\n输出logits shape: {output['logits'].shape}")
    print(f"预测概率: {torch.sigmoid(output['logits'][:5])}")