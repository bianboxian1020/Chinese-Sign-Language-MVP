"""
inference.py — CNN + Bi-LSTM + Attention 手语推理系统

Core neural network for Chinese Sign Language recognition.
- SignLanguageModel: CNN(空间) → Bi-LSTM(时序) → Multi-Head Attention(聚焦) → Bayesian Classifier
- SlidingWindowPredictor: 从 DataQueue 读取特征帧, 滑动窗口批量推理

Architecture:
  Input (B, seq_len, 126) → Conv1d(空间特征) → Bi-LSTM(时序建模) →
  MultiHeadAttention(关键帧加权) → BayesianClassifier(输出+不确定性)
"""
from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import ModelConfig

try:
    from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
    _HAS_PYQT6 = True
except ImportError:
    _HAS_PYQT6 = False
    QObject = object  # type: ignore
    QThread = object  # type: ignore
    pyqtSignal = lambda *a, **kw: None  # type: ignore
    pyqtSlot = lambda *a, **kw: lambda f: f  # type: ignore


# ==============================================================================
# CNN-BiLSTM-Attention 模型
# ==============================================================================

class SpatialCNN(nn.Module):
    """1D 卷积空间特征提取器.

    对每一帧的 21 个手部关键点空间分布进行建模,
    提取手指间距、角度及相对位置特征.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers = []
        in_ch = config.input_dim

        for out_ch in config.conv_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=config.conv_kernel,
                          padding=config.conv_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(config.cnn_dropout),
            ])
            in_ch = out_ch

        self.conv_stack = nn.Sequential(*layers)
        self.output_dim = config.conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim) → (B, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        # → (B, output_dim, seq_len) → (B, seq_len, output_dim)
        return x.transpose(1, 2)


class TemporalBiLSTM(nn.Module):
    """双向 LSTM 时序建模层.

    手语词汇具有明显的动作轨迹 (起手→关键帧→收手),
    Bi-LSTM 同时处理过去帧和未来帧的信息, 精准捕捉动作连续性.
    """

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(config.lstm_hidden * 2)
        self.dropout = nn.Dropout(config.lstm_dropout)
        self.output_dim = config.lstm_hidden * 2  # 双向 → ×2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim)
        residual = x
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        # 残差连接 (维度不匹配时通过线性投影)
        if residual.shape[-1] != x.shape[-1]:
            residual = F.linear(residual,
                                torch.eye(x.shape[-1], residual.shape[-1],
                                          device=x.device))
        return x + residual


class MultiHeadAttentionLayer(nn.Module):
    """多头自注意力层.

    自动为动作序列中的关键转折帧分配更高权重,
    忽略动作间的冗余过渡帧.
    """

    def __init__(self, config: ModelConfig, embed_dim: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=config.attn_heads,
            dropout=config.attn_dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(config.attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_out))
        # 全局平均池化 → (B, embed_dim)
        return x.mean(dim=1)


class BayesianClassifier(nn.Module):
    """贝叶斯分类器 (MC Dropout).

    通过多次前向传播采样估计预测不确定性,
    利用均值作为最终 logits, 方差作为不确定性度量.
    目标延迟 < 0.7s (含全管线).
    """

    def __init__(self, config: ModelConfig, in_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, config.fc_hidden)
        self.dropout = nn.Dropout(config.fc_dropout)
        self.fc2 = nn.Linear(config.fc_hidden, config.num_classes)
        self.mc_samples = config.mc_samples

    def forward(self, x: torch.Tensor, num_samples: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: (B, in_features) 特征向量.
            num_samples: MC Dropout 采样次数 (0 = 单次确定性推理).

        Returns:
            (logits, uncertainty):
              - logits: (B, num_classes) 分类 logits.
              - uncertainty: (B,) 预测不确定性 (标准差均值).
        """
        if num_samples <= 1:
            h = F.relu(self.fc1(x))
            h = self.dropout(h)
            return self.fc2(h), torch.zeros(x.shape[0], device=x.device)

        # MC Dropout: 多次采样
        self.train()  # 保持 dropout 激活
        samples = []
        for _ in range(num_samples):
            h = F.relu(self.fc1(x))
            h = self.dropout(h)
            samples.append(F.log_softmax(self.fc2(h), dim=-1))
        stacked = torch.stack(samples, dim=0)  # (S, B, C)
        log_probs = stacked.mean(dim=0)         # (B, C)
        uncertainty = stacked.std(dim=0).mean(dim=-1)  # (B,)
        return log_probs, uncertainty


class SignLanguageModel(nn.Module):
    """CNN-BiLSTM-Attention 手语识别模型.

    输入:  (B, seq_len, 126)   归一化手部关键点序列
    输出:  (B, num_classes)    分类 logits
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # 空间特征提取
        self.spatial_cnn = SpatialCNN(config)

        # 时序建模
        self.temporal_bilstm = TemporalBiLSTM(config, self.spatial_cnn.output_dim)

        # 注意力聚焦
        self.attention = MultiHeadAttentionLayer(config, self.temporal_bilstm.output_dim)

        # 贝叶斯分类
        self.classifier = BayesianClassifier(config, self.temporal_bilstm.output_dim)

    def forward(self, x: torch.Tensor,
                num_mc_samples: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: (B, seq_len, 126) 归一化关键点序列.
            num_mc_samples: MC Dropout 采样次数 (训练时 0, 推理时 10).

        Returns:
            (logits, uncertainty): 分类预测与不确定性.
        """
        x = self.spatial_cnn(x)       # (B, seq_len, cnn_out)
        x = self.temporal_bilstm(x)   # (B, seq_len, lstm_hidden*2)
        x = self.attention(x)         # (B, lstm_hidden*2)
        return self.classifier(x, num_mc_samples)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, return_confidence: bool = True
                ) -> tuple[str, float]:
        """单样本推理接口.

        Args:
            x: (1, seq_len, 126) 或 (seq_len, 126).
            return_confidence: 是否计算置信度.

        Returns:
            (label_str, confidence).
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if return_confidence and self.config.mc_samples > 1:
            log_probs, uncertainty = self.forward(x, self.config.mc_samples)
            probs = torch.exp(log_probs)
        else:
            logits, uncertainty = self.forward(x, num_mc_samples=0)
            probs = F.softmax(logits, dim=-1)
            uncertainty = torch.zeros(1)

        pred_idx = int(probs.argmax(dim=-1)[0])
        confidence = float(probs[0, pred_idx])
        return pred_idx, confidence, float(uncertainty[0])


# ==============================================================================
# 滑动窗口推理调度器
# ==============================================================================

class SlidingWindowPredictor(QObject):
    """从 DataQueue 连续读取特征帧, 滑动窗口批量推理.

    工作机制:
    1. 定时从 DataQueue 获取最新 window_size 帧序列
    2. 构造输入张量 (1, window_size, 126)
    3. 调用 SignLanguageModel.predict()
    4. 发射 sign_recognized 信号
    5. 滑动窗口: 丢弃 stride 帧, 继续下一轮
    """

    sign_recognized = pyqtSignal(str, float, float)  # label, confidence, uncertainty
    predictor_error = pyqtSignal(str)                 # 错误信息
    predictor_status = pyqtSignal(str)                # 状态更新

    def __init__(self, config: ModelConfig | None = None,
                 label_map: dict[int, str] | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config or ModelConfig()
        self._label_map = label_map or {}  # {idx: "你好"}
        self._idx_map = {v: k for k, v in self._label_map.items()} if label_map else {}

        # 模型
        self._model: SignLanguageModel | None = None
        self._model_loaded = False

        # 滑动窗口缓冲
        self._buffer: deque = deque(maxlen=self.config.seq_len)
        self._window_size = self.config.window_size
        self._stride = self.config.stride

        # 防抖
        self._last_label: str | None = None
        self._stable_count: int = 0
        self._debounce_frames: int = self.config.debounce_frames

        # 设备
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_model()

    def _init_model(self) -> None:
        """初始化模型结构 (权重稍后通过 load_weights 加载)."""
        try:
            self._model = SignLanguageModel(self.config)
            self._model.to(self._device)
            self._model.eval()
            self.predictor_status.emit(
                f"模型结构初始化完成 (参数量: {self._count_params():,})"
            )
        except Exception as e:
            self.predictor_error.emit(f"模型初始化失败: {e}")

    def _count_params(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load_weights(self, path: str) -> bool:
        """加载预训练权重 (.pth 或 .onnx).

        Args:
            path: 权重文件路径.

        Returns:
            True 如果加载成功.
        """
        if self._model is None:
            self.predictor_error.emit("模型未初始化, 无法加载权重")
            return False

        if not os.path.exists(path):
            self.predictor_error.emit(f"权重文件不存在: {path}")
            return False

        try:
            if path.endswith(".pth") or path.endswith(".pt"):
                state = torch.load(path, map_location=self._device, weights_only=True)
                # 兼容不同保存格式
                if "model_state_dict" in state:
                    self._model.load_state_dict(state["model_state_dict"])
                    if "label_map" in state:
                        self._label_map = state["label_map"]
                        self._idx_map = {v: k for k, v in self._label_map.items()}
                else:
                    self._model.load_state_dict(state)
                self._model.eval()
                self._model_loaded = True
                self.predictor_status.emit(
                    f"权重加载成功: {path} (词汇量: {len(self._label_map)})"
                )
                return True

            elif path.endswith(".onnx"):
                self.predictor_status.emit("ONNX 格式暂不支持直接加载, 请先转换为 .pth")
                return False

            else:
                self.predictor_error.emit(f"不支持的权重格式: {path}")
                return False

        except Exception as e:
            self.predictor_error.emit(f"加载权重失败: {e}")
            return False

    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def label_map(self) -> dict[int, str]:
        return self._label_map

    def set_label_map(self, label_map: dict[int, str]) -> None:
        self._label_map = label_map
        self._idx_map = {v: k for k, v in label_map.items()}

    # ------------------------------------------------------------------
    # 特征帧输入
    # ------------------------------------------------------------------

    @pyqtSlot(np.ndarray)
    def on_landmarks(self, features: np.ndarray) -> None:
        """接收单帧归一化特征 (126,), 加入滑动窗口缓冲.

        当缓冲满 window_size 帧时, 触发推理.
        """
        self._buffer.append(features)
        if len(self._buffer) >= self._window_size:
            self._run_inference()

    # ------------------------------------------------------------------
    # 批量窗口推理
    # ------------------------------------------------------------------

    def _run_inference(self) -> None:
        """从 DataQueue 获取序列并执行推理."""
        if self._model is None:
            return

        try:
            sequence = np.array(list(self._buffer)[-self._window_size:], dtype=np.float32)
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(self._device)

            pred_idx, confidence, uncertainty = self._model.predict(
                tensor, return_confidence=self._model_loaded
            )

            # 查找标签
            label = self._label_map.get(pred_idx, f"ID_{pred_idx}")

            # 无模型权重时: 输出占位结果
            if not self._model_loaded:
                label = f"[未训练] {label}"

            # 防抖
            if label == self._last_label:
                self._stable_count += 1
            else:
                self._last_label = label
                self._stable_count = 1
                # 手势切换: 只保留最近5帧, 新手势即刻占主导
                keep = 5
                drop = max(0, len(self._buffer) - keep)
                for _ in range(drop):
                    if self._buffer:
                        self._buffer.popleft()

            if self._stable_count >= self._debounce_frames:
                self.sign_recognized.emit(label, confidence, uncertainty)
                self._stable_count = -4  # 冷却: 跳过接下来 4 次推理, 让旧帧清空

            # 滑动窗口
            for _ in range(self._stride):
                if self._buffer:
                    self._buffer.popleft()

        except Exception as e:
            self.predictor_error.emit(f"推理异常: {e}")

    @staticmethod
    def load_labels_from_file(path: str) -> dict[int, str]:
        """从文件加载标签映射.

        支持格式:
          - JSON: {"0": "你好", "1": "谢谢", ...}
          - TXT: 每行一个词, 行号即 ID
        """
        if not os.path.exists(path):
            return {}

        if path.endswith(".json"):
            import json
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}

        # 默认 TXT 格式
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return {i: word for i, word in enumerate(lines)}
