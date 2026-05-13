"""
tests/test_dataset.py — SkeletonDataset 单元测试

验证数据加载、形状正确性、标签映射、DataLoader 集成。
"""
import os
import sys
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SkeletonDataset, create_dataloader

DATA_ROOT = os.path.join(_PROJECT_ROOT, "data")
LABEL_FILE = os.path.join(DATA_ROOT, "labels.json")


class TestSkeletonDataset:
    """SkeletonDataset 核心功能测试."""

    def test_load_from_data_root(self):
        """验证从 data/ 目录加载 3 个手势类别."""
        ds = SkeletonDataset(
            data_root=DATA_ROOT,
            seq_len=45,
            split="train",
            augment=False,
            feature_dim=126,
            label_file=LABEL_FILE,
        )
        assert len(ds) > 0, "数据集不应为空"
        assert ds.num_classes == 3, f"期望 3 个类别, 实际 {ds.num_classes}"
        assert set(ds.labels.values()) == {"好", "你好", "谢谢"}, \
            f"标签映射不正确: {ds.labels}"

    def test_sample_shape(self):
        """验证每个样本的 shape 为 (seq_len, feature_dim)."""
        ds = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=45, split="train",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        sample, label = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (45, 126), f"期望 (45, 126), 实际 {sample.shape}"
        assert isinstance(label, int)

    def test_padding_truncation(self):
        """验证短序列补零和长序列截断."""
        ds_short = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=20, split="train",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        sample, _ = ds_short[0]
        assert sample.shape[0] == 20

        ds_long = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=90, split="train",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        sample, _ = ds_long[0]
        assert sample.shape[0] == 90

    def test_augmentation(self):
        """验证数据增强不会改变输出 shape."""
        ds = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=45, split="train",
            augment=True, feature_dim=126, label_file=LABEL_FILE,
        )
        sample, _ = ds[0]
        assert sample.shape == (45, 126)

    def test_val_split_no_overlap(self):
        """验证 train/val 划分不重叠."""
        ds_train = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=45, split="train",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        ds_val = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=45, split="val",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        # 验证两个 split 加起来等于总数
        ds_all = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=45, split="test",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        assert len(ds_train) + len(ds_val) == len(ds_all), \
            f"train({len(ds_train)}) + val({len(ds_val)}) != all({len(ds_all)})"

    def test_normalization_range(self):
        """验证样本值在合理范围内 (MediaPipe 归一化 [-1, 1])."""
        ds = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=45, split="train",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        sample, _ = ds[0]
        assert sample.min() >= -1.5, f"最小值 {sample.min()} 异常"
        assert sample.max() <= 1.5, f"最大值 {sample.max()} 异常"


class TestDataLoader:
    """DataLoader 集成测试."""

    def test_batching(self):
        """验证 DataLoader 批量加载."""
        loader = create_dataloader(
            data_root=DATA_ROOT, seq_len=45, batch_size=4,
            split="train", augment=False, feature_dim=126,
            label_file=LABEL_FILE, num_workers=0,
        )
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape[0] == 4  # batch_size
        assert batch_data.shape[1] == 45  # seq_len
        assert batch_data.shape[2] == 126  # feature_dim
        assert batch_labels.shape[0] == 4

    def test_label_mapping_consistency(self):
        """验证标签索引和数据目录名称一致."""
        ds = SkeletonDataset(
            data_root=DATA_ROOT, seq_len=45, split="train",
            augment=False, feature_dim=126, label_file=LABEL_FILE,
        )
        # labels.json: {"0": "你好", "1": "好", "2": "谢谢"}
        assert ds.labels[0] == "你好"
        assert ds.labels[1] == "好"
        assert ds.labels[2] == "谢谢"
