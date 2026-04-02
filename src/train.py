"""
DIEN-DCN模型训练脚本
用于服务器端训练 (3x48G A6000)

使用方法:
    python train.py --config configs/train_config.yaml --gpu 2

注意:
    - 预先运行data_preprocess.py生成processed_data.parquet
    - 预先使用DeBERTa生成caption_embeddings.npy (见generate_caption_embedding.py)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml
import json
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# CUDA优化设置
# cuDNN与CUDA 12.4有兼容性问题，暂时禁用
use_cudnn = True
if use_cudnn:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
else:
    torch.backends.cudnn.enabled = False
    print("已禁用cuDNN（兼容性问题）")

# 修复 cuDNN 初始化问题：在模型加载前预热 CUDA
def warmup_cuda():
    """预热CUDA，解决cuDNN初始化问题"""
    if torch.cuda.is_available():
        # 创建一个小tensor触发CUDA初始化
        x = torch.zeros(1, device='cuda')
        del x
        torch.cuda.empty_cache()
        print("CUDA预热完成")

warmup_cuda()

# 混合精度训练的GradScaler
scaler = GradScaler()

# 导入模型
from model import (
    DIENDCN, FeatureConfig, VideoRecDataset,
    build_video_vocab, compute_numeric_stats, collate_fn
)


# =============================================================================
# 训练配置
# =============================================================================
class TrainConfig:
    """训练配置"""

    # 数据路径
    DATA_PATH = Path("../data/processed_data.parquet")
    CAPTION_EMBEDDING_PATH = Path("../data/caption_embeddings.npy")
    OUTPUT_DIR = Path("../outputs")

    # 模型参数
    VIDEO_EMBEDDING_DIM = 64
    DIEN_HIDDEN_DIM = 64
    DCN_CROSS_LAYERS = 3
    DCN_DEEP_DIMS = [256, 128, 64]
    MLP_DIMS = [256, 128, 64]

    # 训练参数
    BATCH_SIZE = 256
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    AUX_LOSS_WEIGHT = 0.1

    # 优化参数
    USE_AMP = False  # 混合精度训练
    NUM_WORKERS = 4  # 数据加载线程数

    # 其他
    MAX_SEQ_LEN = 50
    MIN_VIDEO_FREQ = 5  # 视频ID最小出现次数
    SAVE_EVERY = 1  # 每隔多少epoch保存模型


# =============================================================================
# 评估指标
# =============================================================================
def compute_metrics(preds: np.ndarray, labels: np.ndarray, user_ids: np.ndarray = None) -> Dict[str, float]:
    """
    计算评估指标
    Args:
        preds: 预测概率 [N]
        labels: 真实标签 [N]
        user_ids: 用户 ID [N] (用于 GAUC 计算)
    Returns:
        dict: AUC, LogLoss等指标
    """
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

    metrics = {}

    # AUC
    try:
        metrics['auc'] = roc_auc_score(labels, preds)
    except:
        metrics['auc'] = 0.0

    # LogLoss
    try:
        # 避免log(0)
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        metrics['logloss'] = log_loss(labels, preds_clipped)
    except:
        metrics['logloss'] = float('inf')

    # Accuracy (使用0.5阈值)
    pred_labels = (preds >= 0.5).astype(int)
    metrics['accuracy'] = accuracy_score(labels, pred_labels)

    # 分组 GAUC (按用户分组计算 AUC 后加权平均)
    metrics['gauc'] = compute_gauc(preds, labels, user_ids)

    return metrics


def compute_gauc(preds: np.ndarray, labels: np.ndarray, user_ids: np.ndarray = None) -> float:
    """
    计算Group AUC (按用户分组计算AUC后加权平均)
    Args:
        preds: 预测概率
        labels: 真实标签
        user_ids: 用户ID (如果为None则返回整体AUC)
    Returns:
        GAUC值
    """
    from sklearn.metrics import roc_auc_score

    # 如果没有用户ID，直接返回整体AUC，避免递归
    if user_ids is None:
        try:
            return roc_auc_score(labels, preds)
        except:
            return 0.0

    # 按用户分组计算AUC
    user_aucs = []
    user_weights = []

    unique_users = np.unique(user_ids)
    for user_id in unique_users:
        user_mask = user_ids == user_id
        user_preds = preds[user_mask]
        user_labels = labels[user_mask]

        # 需要同时有正负样本才能计算AUC
        if user_labels.sum() > 0 and user_labels.sum() < len(user_labels):
            try:
                auc = roc_auc_score(user_labels, user_preds)
                weight = len(user_labels)
                user_aucs.append(auc)
                user_weights.append(weight)
            except:
                pass

    if len(user_aucs) == 0:
        # 如果没有有效的用户组，返回整体AUC
        try:
            return roc_auc_score(labels, preds)
        except:
            return 0.0

    # 加权平均
    gauc = np.average(user_aucs, weights=user_weights)
    return gauc

# =============================================================================
# 数据加载
# =============================================================================
def load_data(config: TrainConfig) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """
    加载处理好的数据
    """
    print("加载处理后的数据...")
    df = pd.read_parquet(config.DATA_PATH)
    print(f"数据大小: {df.shape}")

    # 构建视频词汇表
    print("构建视频词汇表...")
    video_id_to_idx, vocab_size = build_video_vocab(df, min_freq=config.MIN_VIDEO_FREQ)

    return df, video_id_to_idx, vocab_size


def load_caption_embeddings(config: TrainConfig, vocab_size: int) -> Tuple[np.ndarray, int]:
    """
    加载caption embeddings
    如果不存在则返回零矩阵
    Returns:
        embeddings: numpy数组
        embedding_dim: embedding维度
    """
    # 尝试读取配置文件
    vocab_path = Path(config.CAPTION_EMBEDDING_PATH).parent / "video_id_to_idx.json"
    embedding_dim = 128  # 默认维度

    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab_config = json.load(f)
            embedding_dim = vocab_config.get('embedding_dim', 128)
            print(f"从配置文件读取embedding维度: {embedding_dim}")

    if config.CAPTION_EMBEDDING_PATH.exists():
        print("加载caption embeddings...")
        embeddings = np.load(config.CAPTION_EMBEDDING_PATH)
        print(f"Embedding shape: {embeddings.shape}")
        embedding_dim = embeddings.shape[1]
        return embeddings, embedding_dim
    else:
        print(f"Caption embedding文件不存在，使用零向量")
        return np.zeros((vocab_size, embedding_dim), dtype=np.float32), embedding_dim


def build_tag_vocab(df, min_freq: int = 5) -> Tuple[Dict[str, int], int]:
    """
    构建tag词汇表
    Args:
        df: 数据DataFrame
        min_freq: 最小出现次数
    Returns:
        tag_to_idx: tag到索引的映射
        vocab_size: 词汇表大小
    """
    from collections import Counter

    # 统计tag频率
    tags = df['tag'].tolist()
    tag_freq = Counter(tags)

    # 构建映射
    tag_to_idx = {'unknown': 0}  # 0作为padding/unknown
    for tag, freq in tag_freq.items():
        if freq >= min_freq and tag not in tag_to_idx and pd.notna(tag):
            tag_to_idx[tag] = len(tag_to_idx)

    vocab_size = len(tag_to_idx)
    print(f"Tag词汇表大小: {vocab_size}")

    return tag_to_idx, vocab_size


def get_category_vocab_size(df) -> int:
    """
    统计category_id的词汇表大小
    Args:
        df: 数据DataFrame
    Returns:
        vocab_size: 词汇表大小
    """
    # category_id已经是整数ID，直接统计唯一值数量
    unique_categories = df['category_id'].dropna().unique()
    vocab_size = len(unique_categories) + 1  # +1 for padding/unknown
    print(f"Category词汇表大小: {vocab_size}")
    return vocab_size


def prepare_datasets(df: pd.DataFrame,
                     config: TrainConfig,
                     video_id_to_idx: Dict[int, int],
                     caption_embeddings: np.ndarray,
                     embedding_dim: int) -> Tuple[VideoRecDataset, VideoRecDataset, VideoRecDataset, FeatureConfig]:
    """
    准备训练、验证、测试数据集
    """
    print("准备数据集...")

    # 创建FeatureConfig并更新embedding维度
    feature_config = FeatureConfig()
    feature_config.TEXT_EMBEDDING_DIM = embedding_dim
    print(f"文本embedding维度: {embedding_dim}")

    # 按split划分
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(val_df)}")
    print(f"测试集: {len(test_df)}")

    # 计算数值特征统计量 (使用训练集)
    numeric_mean, numeric_std = compute_numeric_stats(train_df, feature_config.NUMERIC_FEATURES)
    print(f"数值特征均值: {numeric_mean}")
    print(f"数值特征标准差: {numeric_std}")

    # 构建tag词汇表
    tag_to_idx, tag_vocab_size = build_tag_vocab(df, min_freq=config.MIN_VIDEO_FREQ)

    # 统计category_id词汇表大小
    category_vocab_size = get_category_vocab_size(df)

    # 更新FeatureConfig中的词汇表大小
    feature_config.CATEGORICAL_VOCAB_SIZE['tag'] = tag_vocab_size
    feature_config.CATEGORICAL_VOCAB_SIZE['category_id'] = category_vocab_size

    # 创建数据集
    train_dataset = VideoRecDataset(
        train_df, feature_config, video_id_to_idx, caption_embeddings,
        numeric_mean=numeric_mean, numeric_std=numeric_std,
        tag_vocab=tag_to_idx, is_train=True
    )
    val_dataset = VideoRecDataset(
        val_df, feature_config, video_id_to_idx, caption_embeddings,
        numeric_mean=numeric_mean, numeric_std=numeric_std,
        tag_vocab=tag_to_idx, is_train=False
    )
    test_dataset = VideoRecDataset(
        test_df, feature_config, video_id_to_idx, caption_embeddings,
        numeric_mean=numeric_mean, numeric_std=numeric_std,
        tag_vocab=tag_to_idx, is_train=False
    )

    return train_dataset, val_dataset, test_dataset, feature_config


# =============================================================================
# 训练器
# =============================================================================
class Trainer:
    """模型训练器"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config: TrainConfig,
                 device: torch.device,
                 use_amp: bool = False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.use_amp = use_amp

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )

        # 混合精度训练的GradScaler
        self.scaler = GradScaler() if use_amp else None

        # 记录最佳模型
        self.best_auc = 0.0
        self.best_epoch = 0

        # 输出目录
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # 移动到设备
            batch = self._move_to_device(batch)

            # 前向传播（混合精度）
            if self.use_amp:
                with autocast():
                    output = self.model(batch)
                    loss = self.model.compute_loss(
                        output,
                        batch['label'],
                        aux_weight=self.config.AUX_LOSS_WEIGHT
                    )

                # 反向传播（混合精度）
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(batch)
                loss = self.model.compute_loss(
                    output,
                    batch['label'],
                    aux_weight=self.config.AUX_LOSS_WEIGHT
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸导致NaN
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def evaluate(self, loader: DataLoader, desc: str = "Eval") -> Dict[str, float]:
        """评估模型"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_user_ids = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch = self._move_to_device(batch)
                if self.use_amp:
                    with autocast():
                        output = self.model(batch)
                else:
                    output = self.model(batch)
                preds = torch.sigmoid(output['logits']).cpu().numpy().flatten()
                labels = batch['label'].cpu().numpy()
                user_ids = batch['user_id'].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_user_ids.extend(user_ids)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_user_ids = np.array(all_user_ids)

        metrics = compute_metrics(all_preds, all_labels, all_user_ids)
        return metrics

    def train(self) -> Dict[str, float]:
        """完整训练流程"""
        print("\n开始训练...")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")

        history = []

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.NUM_EPOCHS}")
            print(f"{'='*50}")

            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"训练损失: {train_metrics['train_loss']:.4f}")

            # 验证
            val_metrics = self.evaluate(self.val_loader, desc="Validation")
            print(f"验证 AUC: {val_metrics['auc']:.4f}")
            print(f"验证 LogLoss: {val_metrics['logloss']:.4f}")
            print(f"验证 Accuracy: {val_metrics['accuracy']:.4f}")

            # 学习率调整
            self.scheduler.step(val_metrics['auc'])

            # 记录
            epoch_record = {
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                **val_metrics
            }
            history.append(epoch_record)

            # 保存最佳模型
            if val_metrics['auc'] > self.best_auc:
                self.best_auc = val_metrics['auc']
                self.best_epoch = epoch
                self._save_model(epoch, val_metrics, is_best=True)
                print(f"保存最佳模型 (AUC: {val_metrics['auc']:.4f})")

            # 定期保存
            if epoch % self.config.SAVE_EVERY == 0:
                self._save_model(epoch, val_metrics, is_best=False)

        # 最终测试
        print(f"\n{'='*50}")
        print("最终测试评估")
        print(f"{'='*50}")
        test_metrics = self.evaluate(self.test_loader, desc="Test")
        print(f"测试 AUC: {test_metrics['auc']:.4f}")
        print(f"测试 LogLoss: {test_metrics['logloss']:.4f}")
        print(f"测试 Accuracy: {test_metrics['accuracy']:.4f}")

        # 保存训练历史
        self._save_history(history, test_metrics)

        return test_metrics

    def _move_to_device(self, batch: Dict) -> Dict:
        """将batch移动到设备"""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result

    def _save_model(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存模型"""
        if is_best:
            path = self.output_dir / "best_model.pt"
        else:
            path = self.output_dir / f"model_epoch_{epoch}.pt"

        # 处理 DataParallel 包装的模型
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)

        # 保存后清理旧的检查点文件（只保留最近2个）
        if not is_best:
            self._cleanup_old_checkpoints(keep=2)

    def _cleanup_old_checkpoints(self, keep: int = 2):
        """清理旧的检查点文件，只保留最近的几个"""
        # 获取所有 model_epoch_*.pt 文件并按修改时间排序
        checkpoints = sorted(
            self.output_dir.glob("model_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        # 删除多余的旧文件
        if len(checkpoints) > keep:
            for old_ckpt in checkpoints[:-keep]:
                old_ckpt.unlink()
                print(f"删除旧检查点: {old_ckpt.name}")

    def _save_history(self, history: List, test_metrics: Dict):
        """保存训练历史"""
        path = self.output_dir / "training_history.json"
        with open(path, 'w') as f:
            json.dump({
                'history': history,
                'test_metrics': test_metrics,
                'best_epoch': self.best_epoch,
                'best_auc': self.best_auc
            }, f, indent=2)


# =============================================================================
# 主函数
# =============================================================================
def main():
    """主训练流程"""

    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--epochs', type=int, default=None, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--gpu', type=int, default=0, help='指定GPU编号 (默认0)')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')
    args = parser.parse_args()

    # 指定GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用 GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    # 配置
    config = TrainConfig()

    # 从yaml读取配置（修复：正确解析嵌套配置）
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)

        # 特殊字段映射（yaml字段名 -> TrainConfig属性名）
        field_mapping = {
            'epochs': 'NUM_EPOCHS',
            'learning_rate': 'LEARNING_RATE',
            'batch_size': 'BATCH_SIZE',
            'lr': 'LEARNING_RATE',
        }

        # 处理嵌套配置
        for key, value in yaml_config.items():
            if isinstance(value, dict):
                # 嵌套配置，展开处理
                for sub_key, sub_value in value.items():
                    # 先检查是否有映射
                    if sub_key in field_mapping:
                        attr_name = field_mapping[sub_key]
                    else:
                        attr_name = sub_key.upper()
                    if hasattr(config, attr_name):
                        setattr(config, attr_name, sub_value)
            else:
                # 先检查是否有映射
                if key in field_mapping:
                    attr_name = field_mapping[key]
                else:
                    attr_name = key.upper()
                if hasattr(config, attr_name):
                    setattr(config, attr_name, value)

    # 命令行参数覆盖
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.LEARNING_RATE = args.lr

    # 混合精度训练标志
    use_amp = args.amp or (hasattr(config, 'USE_AMP') and config.USE_AMP)

    print("\n" + "="*70)
    print("DIEN-DCN模型训练")
    print("="*70)
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"GPU: {args.gpu}")
    print(f"混合精度训练: {use_amp}")

    # 加载数据
    df, video_id_to_idx, vocab_size = load_data(config)

    # 加载caption embeddings
    caption_embeddings, embedding_dim = load_caption_embeddings(config, vocab_size)

    # 准备数据集
    train_dataset, val_dataset, test_dataset, feature_config = prepare_datasets(
        df, config, video_id_to_idx, caption_embeddings, embedding_dim
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 创建模型 (使用更新后的feature_config)
    model = DIENDCN(
        config=feature_config,
        video_vocab_size=vocab_size,
        video_embedding_dim=config.VIDEO_EMBEDDING_DIM,
        dien_hidden_dim=config.DIEN_HIDDEN_DIM,
        dcn_cross_layers=config.DCN_CROSS_LAYERS,
        dcn_deep_dims=config.DCN_DEEP_DIMS,
        mlp_dims=config.MLP_DIMS
    )

    # 移动到设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, config, device, use_amp
    )
    test_metrics = trainer.train()

    print("\n训练完成!")
    print(f"最佳验证 AUC: {trainer.best_auc:.4f} (Epoch {trainer.best_epoch})")
    print(f"测试 AUC: {test_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()