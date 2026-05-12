"""
model_config.py — 模型超参数配置

Centralized configuration for the CNN-BiLSTM-Attention sign language model.
"""
import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """CNN-BiLSTM-Attention 模型超参数."""

    # 输入
    input_dim: int = 126          # 手部关键点: 2手 × 21点 × 3坐标 (SLR500体关节: 75)
    seq_len: int = 45             # 序列帧数 (30~60)
    num_classes: int = 500        # 词汇量 (SLR500 = 500 词; ISW-1000 = 1000)

    # CNN 空间特征提取
    conv_channels: list[int] = field(default_factory=lambda: [128, 256, 256])
    conv_kernel: int = 3
    cnn_dropout: float = 0.3

    # Bi-LSTM 时序建模
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.3

    # Multi-Head Attention
    attn_heads: int = 8
    attn_dropout: float = 0.2

    # Bayesian Classifier
    fc_hidden: int = 256
    fc_dropout: float = 0.5
    mc_samples: int = 3            # MC Dropout 推理采样次数 (降低加速)

    # 滑动窗口
    window_size: int = 20          # 同 seq_len (降低加速响应)
    stride: int = 5                # 窗口步进帧数
    debounce_frames: int = 2       # 防抖连续帧数

    # 关键帧
    keyframe_threshold: float = 0.01  # 帧间差阈值 (降低提高数据率)

    # 训练
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stop_patience: int = 15
    warmup_epochs: int = 5

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


# 默认配置实例
DEFAULT_CONFIG = ModelConfig()
