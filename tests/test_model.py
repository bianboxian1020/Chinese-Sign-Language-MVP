"""
tests/test_model.py — SignLanguageModel 单元测试

验证模型结构、前向传播、predict 接口。
"""
import os
import sys
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_DIR)

import torch
from model_config import ModelConfig
from inference import SignLanguageModel


@pytest.fixture
def config():
    """MVP 配置: 3 个手势词."""
    cfg = ModelConfig()
    cfg.num_classes = 3
    cfg.seq_len = 20  # 缩短序列加速测试
    cfg.mc_samples = 3
    return cfg


@pytest.fixture
def model(config):
    return SignLanguageModel(config)


class TestSignLanguageModel:

    def test_parameter_count(self, model, config):
        """验证参数量合理 (> 0 且 < 千万级)."""
        n = sum(p.numel() for p in model.parameters())
        assert n > 0
        assert n < 20_000_000, f"参数量 {n:,} 异常大"

    def test_forward_shape(self, model, config):
        """验证前向传播输出 shape."""
        B, T, D = 4, config.seq_len, config.input_dim
        x = torch.randn(B, T, D)
        logits, uncertainty = model(x, num_mc_samples=0)
        assert logits.shape == (B, config.num_classes), \
            f"期望 ({B}, {config.num_classes}), 实际 {logits.shape}"
        assert uncertainty.shape == (B,), \
            f"期望 ({B},), 实际 {uncertainty.shape}"

    def test_mc_dropout_shape(self, model, config):
        """验证 MC Dropout 推理输出 shape."""
        B, T, D = 2, config.seq_len, config.input_dim
        x = torch.randn(B, T, D)
        logits, uncertainty = model(x, num_mc_samples=config.mc_samples)
        assert logits.shape == (B, config.num_classes)
        assert uncertainty.shape == (B,)
        # MC Dropout 不应输出全零 uncertainty
        assert uncertainty.sum() > 0, "MC Dropout uncertainty 不应全为零"

    def test_predict_single_sample(self, model, config):
        """验证单样本 predict 接口."""
        T, D = config.seq_len, config.input_dim
        x = torch.randn(T, D)
        pred_idx, confidence, uncertainty = model.predict(
            x, return_confidence=True
        )
        assert isinstance(pred_idx, int)
        assert 0 <= pred_idx < config.num_classes
        assert 0.0 <= confidence <= 1.0

    def test_predict_batch(self, model, config):
        """验证批量 predict 接口."""
        B, T, D = 2, config.seq_len, config.input_dim
        x = torch.randn(B, T, D)
        pred_idx, confidence, uncertainty = model.predict(
            x, return_confidence=True
        )
        assert isinstance(pred_idx, int)
        assert 0.0 <= confidence <= 1.0

    def test_predict_no_confidence(self, model, config):
        """验证不计算置信度的 predict."""
        T, D = config.seq_len, config.input_dim
        x = torch.randn(T, D)
        pred_idx, confidence, uncertainty = model.predict(
            x, return_confidence=False
        )
        assert isinstance(pred_idx, int)
        # uncertainty 应接近零
        assert abs(uncertainty) < 1e-6

    def test_different_seq_lengths(self, model, config):
        """验证不同序列长度的前向传播."""
        for seq_len in [15, 30, 45]:
            x = torch.randn(1, seq_len, config.input_dim)
            logits, _ = model(x, num_mc_samples=0)
            assert logits.shape[0] == 1
            assert logits.shape[1] == config.num_classes

    def test_model_config_coupling(self, model, config):
        """验证 model.config 与传入的 config 一致."""
        assert model.config is config
        assert model.config.num_classes == 3
        assert model.config.input_dim == 126
