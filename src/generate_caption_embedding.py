"""
使用DeBERTa V3 base生成caption embeddings
在服务器端运行，利用GPU加速

支持降维输出，减少存储空间：
- 768维 (原始DeBERTa输出): 约15GB
- 256维: 约5GB
- 128维: 约2.5GB
- 64维: 约1.2GB

使用方法:
    python generate_caption_embedding.py --input ../data/processed_data.parquet --output ../data/caption_embeddings.npy --output_dim 128

注意:
    - 需要安装transformers库: pip install transformers
    - 建议在GPU服务器上运行
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Caption Embedding生成器
# =============================================================================
class CaptionEmbeddingGenerator:
    """使用DeBERTa生成caption语义embedding"""

    def __init__(self,
                 model_name: str = "/mnt/mechanical_drive/DATA/Models/deberta/deberta_v3_large",
                 device: str = "cuda",
                 batch_size: int = 256,
                 max_length: int = 128,
                 output_dim: int = 64):
        """
        Args:
            model_name: DeBERTa模型名称
            device: 设备 (cuda/cpu)
            batch_size: 批处理大小
            max_length: 文本最大长度
            output_dim: 输出embedding维度 (默认768，可设置为128/256等降维)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dim = output_dim

        # 加载模型和tokenizer
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # 获取DeBERTa原始输出维度
        self.hidden_size = self.model.config.hidden_size
        print(f"DeBERTa原始输出维度: {self.hidden_size}")

        # 降维投影层
        if output_dim != self.hidden_size:
            print(f"添加降维投影层: {self.hidden_size} -> {output_dim}")
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim)
            ).to(device)
            
            # 关键修改：让投影层使用与模型相同的数据类型
            # 获取模型的数据类型
            model_dtype = next(self.model.parameters()).dtype
            self.projection = self.projection.to(dtype=model_dtype)
            
            # 投影层也设置为eval模式
            self.projection.eval()
        else:
            self.projection = None

        # 移动到设备
        self.model = self.model.to(device)
        self.model.eval()

        print(f"输出embedding维度: {output_dim}")
        print(f"模型加载完成，设备: {device}")

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        编码一批文本
        Args:
            texts: 文本列表
        Returns:
            embeddings: [batch_size, output_dim] numpy数组
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 编码
        with torch.no_grad():
            outputs = self.model(**inputs)

            # 使用最后一层hidden state的平均作为句子embedding
            # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
            # 使用attention mask加权平均
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch_size, seq_len, 1]
            hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

            # 加权平均
            sum_hidden = (hidden_state * attention_mask).sum(dim=1)  # [batch_size, hidden_size]
            sum_mask = attention_mask.sum(dim=1)  # [batch_size, 1]
            embeddings = sum_hidden / sum_mask  # [batch_size, hidden_size]

            # 降维投影
            if self.projection is not None:
                embeddings = self.projection(embeddings)  # [batch_size, output_dim]

        # 转换为numpy
        embeddings = embeddings.cpu().numpy()

        return embeddings

    def encode_all(self, texts: List[str], desc: str = "Encoding") -> np.ndarray:
        """
        编码所有文本
        Args:
            texts: 文本列表
            desc: 进度条描述
        Returns:
            embeddings: [num_texts, output_dim] numpy数组
        """
        all_embeddings = []

        # 分批处理
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(embeddings)

        # 合并
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    def generate_embeddings_for_videos(self,
                                        df: pd.DataFrame,
                                        video_id_to_idx: Dict[int, int]) -> np.ndarray:
        """
        为所有视频生成caption embedding
        Args:
            df: 包含video_id和caption的数据DataFrame
            video_id_to_idx: video_id到索引的映射
        Returns:
            embeddings: [vocab_size, output_dim] numpy数组
        """
        vocab_size = len(video_id_to_idx)

        # 初始化embedding矩阵
        embeddings = np.zeros((vocab_size, self.output_dim), dtype=np.float32)

        # 获取唯一视频及其caption
        video_caption_df = df[['video_id', 'caption']].drop_duplicates(subset=['video_id'])
        video_caption_df = video_caption_df.reset_index(drop=True)

        print(f"唯一视频数: {len(video_caption_df)}")
        print(f"词汇表大小: {vocab_size}")

        # 映射video_id到caption
        video_to_caption = dict(zip(video_caption_df['video_id'], video_caption_df['caption']))

        # 按索引顺序生成embedding
        video_ids = []
        captions = []
        for video_id, idx in video_id_to_idx.items():
            if video_id != 0:  # 跳过padding
                video_ids.append(video_id)
                caption = video_to_caption.get(video_id, '')
                captions.append(caption)

        print(f"需要编码的视频数: {len(video_ids)}")

        # 生成embedding
        caption_embeddings = self.encode_all(captions, desc="Encoding captions")

        # 填充embedding矩阵
        for i, video_id in enumerate(video_ids):
            idx = video_id_to_idx[video_id]
            embeddings[idx] = caption_embeddings[i]

        return embeddings


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="使用DeBERTa生成caption embeddings")
    parser.add_argument('--input', type=str, default='../data/processed_data.parquet',
                        help='输入数据路径')
    parser.add_argument('--output', type=str, default='../data/caption_embeddings.npy',
                        help='输出embedding路径')
    parser.add_argument('--model', type=str, default='/mnt/mechanical_drive/DATA/Models/deberta/deberta_v3_large',
                        help='DeBERTa模型名称')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批处理大小')
    parser.add_argument('--max_length', type=int, default=128,
                        help='文本最大长度')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='输出embedding维度 (768为原始维度，可降维为128/256等)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("Caption Embedding生成")
    print("="*70)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"模型: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Length: {args.max_length}")
    print(f"Output Dim: {args.output_dim}")
    print(f"Device: {args.device}")

    # 加载数据
    print("\n加载数据...")
    df = pd.read_parquet(args.input)
    print(f"数据大小: {df.shape}")

    # 构建视频词汇表
    print("\n构建视频词汇表...")
    from collections import Counter

    # 统计视频ID频率
    video_ids = df['video_id'].tolist()
    for history in df['history_video_ids']:
        video_ids.extend([int(vid) for vid in history])

    video_freq = Counter(video_ids)

    # 构建映射 (最小频率5)
    min_freq = 5
    video_id_to_idx = {0: 0}  # 0作为padding
    for video_id, freq in video_freq.items():
        if freq >= min_freq and video_id not in video_id_to_idx:
            video_id_to_idx[video_id] = len(video_id_to_idx)

    vocab_size = len(video_id_to_idx)
    print(f"视频词汇表大小: {vocab_size}")

    # 创建embedding生成器
    generator = CaptionEmbeddingGenerator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dim=args.output_dim
    )

    # 生成embedding
    print("\n生成caption embeddings...")
    embeddings = generator.generate_embeddings_for_videos(df, video_id_to_idx)
    print(f"Embedding shape: {embeddings.shape}")

    # 保存
    print(f"\n保存到: {args.output}")
    np.save(args.output, embeddings)

    # 同时保存词汇表映射和embedding维度
    vocab_path = Path(args.output).parent / "video_id_to_idx.json"
    config_dict = {
        'vocab': {str(k): v for k, v in video_id_to_idx.items()},
        'vocab_size': vocab_size,
        'embedding_dim': args.output_dim,
        'model': args.model
    }
    with open(vocab_path, 'w') as f:
        json.dump(config_dict, f)
    print(f"词汇表保存到: {vocab_path}")

    # 验证
    loaded = np.load(args.output)
    print(f"\n验证加载: shape={loaded.shape}, dtype={loaded.dtype}")

    # 统计非零行数
    non_zero_count = (loaded.sum(axis=1) != 0).sum()
    print(f"非零embedding数: {non_zero_count}/{vocab_size}")

    print("\n完成!")


if __name__ == "__main__":
    import json
    main()