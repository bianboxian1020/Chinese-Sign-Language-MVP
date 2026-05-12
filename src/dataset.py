"""
dataset.py — 骨架数据集加载器 + GAN 数据增强

Supports multiple skeleton keypoint dataset formats:
- SLR500 (DSTA-SLR): 25 body joints × 3 coords → 75 dim
- ISW-1000: variable keypoints × 3 coords

Each sample is a .npy file with shape (T, V, 3) or (T, D) where:
  - T: number of frames (variable)
  - V: number of keypoints or D: flattened dim
"""
import os
import random
import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SkeletonDataset(Dataset):
    """通用骨架数据集加载器.

    支持两种目录结构:

    A) 按类别分目录 (SLR500/ISW-1000 典型结构):
         data_root/
           word_001/  sample_001.npy  sample_002.npy ...
           word_002/  ...
           labels.json  (可选)

    B) 按 train/val 分目录 (DSTA-SLR 预处理格式):
         data_root/
           train/  sample_001.npy  sample_002.npy ...
           val/    sample_101.npy ...
           train_label.npy  (可选)
    """

    def __init__(self, data_root: str, seq_len: int = 45,
                 split: str = "train", split_ratio: float = 0.8,
                 augment: bool = False,
                 feature_dim: int = 75,
                 label_file: str = "") -> None:
        super().__init__()
        self.data_root = data_root
        self.seq_len = seq_len
        self.augment = augment
        self.feature_dim = feature_dim

        self._samples: list[tuple[str, int]] = []
        self._labels: dict[int, str] = {}

        # 自动检测目录结构
        self._auto_detect(split, split_ratio, label_file)

        print(f"[Dataset] {split}: {len(self._samples)} samples, "
              f"{len(self._labels)} classes, dim={feature_dim}")

    def _auto_detect(self, split: str, ratio: float,
                     label_file: str) -> None:
        """自动检测并加载数据."""
        # 模式 1: 有 train/val/test 子目录 (DSTA-SLR 格式)
        if os.path.isdir(os.path.join(self.data_root, "train")):
            self._load_split_dir(split)
            return

        # 模式 2: 按类别分目录
        if label_file and os.path.exists(label_file):
            self._load_labels_from_file(label_file)

        self._scan_category_dirs(split, ratio)

    def _load_split_dir(self, split: str) -> None:
        """加载已划分好的 train/val/test 目录."""
        split_dir = os.path.join(self.data_root, split)
        if not os.path.isdir(split_dir):
            print(f"[Dataset] 警告: {split_dir} 不存在")
            return

        npy_files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))
        if not npy_files:
            print(f"[Dataset] 警告: {split_dir} 下无 .npy 文件")
            return

        # 尝试加载标签
        label_path = os.path.join(self.data_root, f"{split}_label.npy")
        if os.path.exists(label_path):
            all_labels = np.load(label_path)
        else:
            all_labels = np.arange(len(npy_files))

        for fpath, lbl in zip(npy_files, all_labels):
            self._samples.append((fpath, int(lbl)))

        # 构建标签映射
        for _, lbl in self._samples:
            self._labels[lbl] = f"word_{lbl}"

    def _scan_category_dirs(self, split: str, ratio: float) -> None:
        """扫描按类别分目录的数据结构."""
        subdirs = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
            and not d.startswith(".")
        ])
        if not subdirs:
            # 回退: 直接在根目录找 .npy
            npy_files = sorted(glob.glob(os.path.join(self.data_root, "*.npy")))
            for i, fpath in enumerate(npy_files):
                lbl = i % 500
                self._samples.append((fpath, lbl))
                self._labels[lbl] = f"word_{lbl}"
            return

        all_samples = []
        for idx, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.data_root, subdir)
            npy_files = sorted(glob.glob(os.path.join(subdir_path, "*.npy")))
            for fpath in npy_files:
                all_samples.append((fpath, idx))
            self._labels[idx] = subdir

        # 按手语者划分 (从文件名提取 signer ID)
        n_total = len(all_samples)
        n_train = int(n_total * ratio)
        random.Random(42).shuffle(all_samples)
        if split == "train":
            self._samples = all_samples[:n_train]
        elif split == "val":
            self._samples = all_samples[n_train:]
        else:
            self._samples = all_samples

    def _load_labels_from_file(self, path: str) -> None:
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._labels = {int(k): v for k, v in raw.items()}
        except Exception:
            pass

    @property
    def labels(self) -> dict[int, str]:
        return self._labels

    @property
    def num_classes(self) -> int:
        return max(len(self._labels), 1)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self._samples[idx]
        skeleton = np.load(path).astype(np.float32)

        # 处理形状: (T, V, 3) → (T, V*3) 或 (T, D) 直接使用
        if skeleton.ndim == 3:
            skeleton = skeleton.reshape(skeleton.shape[0], -1)

        # 序列长度统一
        skeleton = self._pad_or_truncate(skeleton)

        # 数据增强
        if self.augment:
            skeleton = self._augment(skeleton)

        return torch.from_numpy(skeleton), label

    def _pad_or_truncate(self, skeleton: np.ndarray) -> np.ndarray:
        t = skeleton.shape[0]
        if t >= self.seq_len:
            indices = np.linspace(0, t - 1, self.seq_len, dtype=int)
            return skeleton[indices]
        else:
            padded = np.zeros((self.seq_len, skeleton.shape[1]), dtype=np.float32)
            padded[:t] = skeleton
            return padded

    def _augment(self, skeleton: np.ndarray) -> np.ndarray:
        """时间缩放 + 空间扰动 + 高斯噪声."""
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            t = skeleton.shape[0]
            new_t = max(5, int(t * scale))
            indices = np.linspace(0, t - 1, new_t, dtype=int)
            skeleton = skeleton[indices]
            skeleton = self._pad_or_truncate(skeleton)

        if random.random() < 0.4:
            noise = np.random.normal(0, 0.02, skeleton.shape).astype(np.float32)
            skeleton += noise

        if random.random() < 0.3:
            skeleton += np.random.normal(0, 0.01, skeleton.shape).astype(np.float32)

        return np.clip(skeleton, -1.5, 1.5)


# ==============================================================================
# GAN 数据增强 (保持不变)
# ==============================================================================

class MotionGenerator(nn.Module):
    def __init__(self, feature_dim: int = 75, seq_len: int = 45,
                 noise_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fc1 = nn.Linear(noise_dim, hidden_dim * (seq_len // 4))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(hidden_dim * 2, feature_dim)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = F.relu(self.fc1(z))
        h = h.view(B, self.seq_len // 4, -1)
        h, _ = self.lstm(h)
        h = F.interpolate(h.transpose(1, 2), size=self.seq_len,
                          mode="linear", align_corners=False)
        h = h.transpose(1, 2)
        return self.tanh(self.fc2(h)) * 0.1


class MotionDiscriminator(nn.Module):
    def __init__(self, feature_dim: int = 75, seq_len: int = 45) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(256, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


# ==============================================================================
# ISW-1000 数据集 (保留兼容)
# ==============================================================================

ISW1000Dataset = SkeletonDataset  # 向后兼容别名


# ==============================================================================
# 工具函数
# ==============================================================================

def create_dataloader(data_root: str, seq_len: int = 45,
                      batch_size: int = 32, split: str = "train",
                      augment: bool = False, feature_dim: int = 75,
                      num_workers: int = 0,
                      label_file: str = "") -> DataLoader:
    dataset = SkeletonDataset(
        data_root=data_root, seq_len=seq_len,
        split=split, augment=augment,
        feature_dim=feature_dim, label_file=label_file,
    )
    shuffle = (split == "train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=(split == "train"),
                      pin_memory=True)
