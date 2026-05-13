"""
tests/test_inference.py — SlidingWindowPredictor 单元测试

验证推理调度器初始化、权重加载、标签映射、防抖逻辑。
"""
import os
import sys
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_DIR)

import numpy as np
import torch
from model_config import ModelConfig
from inference import SlidingWindowPredictor

MODEL_PATH = os.path.join(_PROJECT_ROOT, "assets", "models", "xinyu_model_best.pth")
LABEL_FILE = os.path.join(_PROJECT_ROOT, "data", "labels.json")


@pytest.fixture
def config():
    cfg = ModelConfig()
    cfg.num_classes = 3
    cfg.window_size = 20
    cfg.stride = 5
    cfg.debounce_frames = 2
    cfg.seq_len = 45
    cfg.mc_samples = 3
    return cfg


@pytest.fixture
def label_map():
    return {0: "你好", 1: "好", 2: "谢谢"}


class TestSlidingWindowPredictorInit:

    def test_create_with_config(self, config, label_map):
        """验证用 config 和 label_map 创建 predictor."""
        p = SlidingWindowPredictor(config=config, label_map=label_map)
        assert p._model is not None
        assert p._window_size == config.window_size
        assert p._stride == config.stride
        assert p._debounce_frames == config.debounce_frames
        assert p._label_map == label_map
        assert not p.is_model_loaded  # 仅初始化结构, 未加载权重

    def test_create_without_config(self):
        """验证不传 config 时用默认值创建."""
        p = SlidingWindowPredictor()
        assert p._model is not None
        assert p._window_size > 0
        assert not p.is_model_loaded

    def test_create_without_label_map(self, config):
        """验证不传 label_map 时默认空字典."""
        p = SlidingWindowPredictor(config=config)
        assert p._label_map == {}
        assert p.label_map == {}


class TestWeightLoading:

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"权重文件不存在: {MODEL_PATH}"
    )
    def test_load_weights_success(self, config, label_map):
        """验证加载预训练权重."""
        p = SlidingWindowPredictor(config=config, label_map=label_map)
        assert p.load_weights(MODEL_PATH)
        assert p.is_model_loaded

    def test_load_nonexistent_weights(self, config):
        """验证加载不存在的权重文件返回 False."""
        p = SlidingWindowPredictor(config=config)
        assert not p.load_weights("/nonexistent/path/model.pth")
        assert not p.is_model_loaded


class TestLabelMap:

    def test_set_label_map(self, config, label_map):
        """验证动态设置标签映射."""
        p = SlidingWindowPredictor(config=config)
        assert p._label_map == {}
        p.set_label_map(label_map)
        assert p._label_map == label_map

    def test_load_labels_from_json(self):
        """验证从 labels.json 加载标签."""
        if not os.path.exists(LABEL_FILE):
            pytest.skip(f"标签文件不存在: {LABEL_FILE}")
        labels = SlidingWindowPredictor.load_labels_from_file(LABEL_FILE)
        assert labels == {0: "你好", 1: "好", 2: "谢谢"}


class TestLandmarkProcessing:

    def test_on_landmarks_buffering(self, config, label_map):
        """验证 on_landmarks 正确缓冲帧."""
        p = SlidingWindowPredictor(config=config, label_map=label_map)
        # 发送 15 帧 (不足 window_size=20)
        for _ in range(15):
            features = np.random.randn(126).astype(np.float32)
            p.on_landmarks(features)
        # 缓冲应积累但未触发推理
        assert len(p._buffer) == 15
        # 再发 10 帧 (总计 25, 超过 window_size)
        for _ in range(10):
            features = np.random.randn(126).astype(np.float32)
            p.on_landmarks(features)
        # 推理后滑动窗口移动，剩余 < window_size
        # 25 帧 → 推理一次 → 滑动 stride=5 → 剩余 20, 再推 → 剩余 15
        assert len(p._buffer) < config.window_size
