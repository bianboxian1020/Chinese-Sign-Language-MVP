#!/usr/bin/env python
"""
train.py — CNN-BiLSTM-Attention 手语识别模型训练脚本

Usage:
    python train.py --data_root data/ISW-1000 --epochs 100 --batch_size 32

训练管线:
1. 加载 ISW-1000 骨架数据
2. 数据增强 (时间缩放 + 空间扰动 + GAN 生成)
3. 训练 CNN-BiLSTM-Attention 模型
4. 验证 + 早停
5. 保存最佳权重 (含标签映射)
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from model_config import ModelConfig
from inference import SignLanguageModel
from dataset import SkeletonDataset, MotionGenerator, MotionDiscriminator


# ==============================================================================
# 训练器
# ==============================================================================

class Trainer:
    """CNN-BiLSTM-Attention 模型训练器."""

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        # 模型
        self.model = SignLanguageModel(config).to(device)
        print(f"[Model] 参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 损失与优化器
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # GAN (可选)
        self.gan_generator: MotionGenerator | None = None
        self.gan_discriminator: MotionDiscriminator | None = None
        self._init_gan(device)

        # 状态
        self.best_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _init_gan(self, device: torch.device) -> None:
        """初始化 MotionGAN (用于数据增强)."""
        self.gan_generator = MotionGenerator(
            feature_dim=self.config.input_dim,
            seq_len=self.config.seq_len,
        ).to(device)
        self.gan_discriminator = MotionDiscriminator(
            feature_dim=self.config.input_dim,
            seq_len=self.config.seq_len,
        ).to(device)
        self.gan_optimizer_G = optim.Adam(self.gan_generator.parameters(), lr=1e-4)
        self.gan_optimizer_D = optim.Adam(self.gan_discriminator.parameters(), lr=1e-4)
        self.gan_criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """训练一个 epoch. 返回平均损失."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)          # (B, seq_len, feature_dim)
            targets = targets.to(self.device)    # (B,)

            # --- GAN 增强 (每 3 个 batch 一次) ---
            if batch_idx % 3 == 0 and self.gan_generator is not None:
                self._train_gan_step(data)

            # --- 主模型训练 ---
            self.optimizer.zero_grad()

            # 前向传播
            logits, _ = self.model(data, num_mc_samples=0)
            loss = self.criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % 20 == 0:
                acc = total_correct / max(total_samples, 1)
                print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d} | "
                      f"Loss: {loss.item():.4f} | Acc: {acc:.3f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    def _train_gan_step(self, real_data: torch.Tensor) -> None:
        """训练 GAN 一步: 更新 D 和 G."""
        B = real_data.size(0)
        noise = torch.randn(B, 64, device=self.device)

        # 训练判别器
        self.gan_optimizer_D.zero_grad()
        real_validity = self.gan_discriminator(real_data.transpose(1, 2))
        with torch.no_grad():
            fake_motion = self.gan_generator(noise)
        fake_data = real_data + fake_motion
        fake_validity = self.gan_discriminator(fake_data.transpose(1, 2))
        d_loss = (self.gan_criterion(real_validity, torch.ones_like(real_validity))
                  + self.gan_criterion(fake_validity, torch.zeros_like(fake_validity)))
        d_loss.backward()
        self.gan_optimizer_D.step()

        # 训练生成器
        self.gan_optimizer_G.zero_grad()
        fake_motion = self.gan_generator(noise)
        fake_data = real_data + fake_motion
        fake_validity = self.gan_discriminator(fake_data.transpose(1, 2))
        g_loss = self.gan_criterion(fake_validity, torch.ones_like(fake_validity))
        g_loss.backward()
        self.gan_optimizer_G.step()

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """验证. 返回 (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, targets in val_loader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            logits, _ = self.model(data, num_mc_samples=0)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / max(len(val_loader), 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    # ------------------------------------------------------------------
    # 完整训练流程
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            save_dir: str = "assets/models") -> None:
        """完整训练流程, 含早停和 checkpoint 保存."""
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"{'='*50}\n")

        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step()

            # 日志
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} Summary | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | LR: {lr:.6f}")

            # 早停 + 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(save_dir, epoch, val_acc)
                print(f"  >>> Best model saved (Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        print(f"\n{'='*50}")
        print(f"Training finished. Best Val Acc: {self.best_acc:.4f} "
              f"at epoch {self.best_epoch}")
        print(f"{'='*50}")

    def _save_checkpoint(self, save_dir: str, epoch: int, acc: float) -> None:
        """保存模型 checkpoint (含标签映射)."""
        # 获取标签映射
        labels = {}
        if hasattr(self, "_label_map"):
            labels = self._label_map

        checkpoint = {
            "epoch": epoch,
            "val_acc": acc,
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "label_map": labels,
        }

        path = os.path.join(save_dir, "xinyu_model_best.pth")
        torch.save(checkpoint, path)

        # 同时保存配置
        self.config.to_json(os.path.join(save_dir, "model_config.json"))

    def set_label_map(self, label_map: dict[int, str]) -> None:
        """设置标签映射 (供保存 checkpoint 时使用)."""
        self._label_map = label_map


# ==============================================================================
# 主入口
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="训练 CNN-BiLSTM-Attention 手语识别模型")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录 (SLR500/ISW-1000)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=45)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature_dim", type=int, default=126,
                        help="特征维度 (手部关键点=126, SLR500体关节=75)")
    parser.add_argument("--num_classes", type=int, default=0,
                        help="类别数 (0=自动从标签文件检测)")
    parser.add_argument("--label_file", type=str, default="data/labels.json",
                        help="标签文件路径")
    parser.add_argument("--save_dir", type=str, default="assets/models")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_gan", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    # 自动检测类别数
    num_classes = args.num_classes
    if num_classes <= 0 and os.path.exists(args.label_file):
        import json
        with open(args.label_file, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        num_classes = len(label_map)
        print(f"[Data] Auto-detected num_classes={num_classes} from {args.label_file}")
    if num_classes <= 0:
        num_classes = 8  # MVP 默认
        print(f"[Data] Using default num_classes={num_classes}")

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[Device] {device}")
    print(f"[Data] Num classes: {num_classes}, Feature dim: {args.feature_dim}")

    # 配置
    config = ModelConfig()
    config.seq_len = args.seq_len
    config.num_classes = num_classes
    config.input_dim = args.feature_dim
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr

    # 数据加载
    train_dataset = SkeletonDataset(
        data_root=args.data_root,
        seq_len=args.seq_len,
        split="train",
        augment=True,
        feature_dim=args.feature_dim,
        label_file=args.label_file,
    )
    val_dataset = SkeletonDataset(
        data_root=args.data_root,
        seq_len=args.seq_len,
        split="val",
        augment=False,
        feature_dim=args.feature_dim,
        label_file=args.label_file,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"[Data] Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # 训练
    trainer = Trainer(config, device)
    trainer.set_label_map(train_dataset.labels)

    if args.no_gan:
        trainer.gan_generator = None
        trainer.gan_discriminator = None

    trainer.fit(train_loader, val_loader, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
