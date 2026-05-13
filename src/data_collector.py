"""
data_collector.py — 训练数据采集模块

实时录制手部骨骼关键点 (126-dim) 并保存为 .npy 样本。
配合 main_gui.py 使用: 用户输入手势词 → 录制 → 自动保存到 data/<词>/sample_N.npy.
"""
import os
import glob
import json
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


DATA_ROOT = "data"
SEQ_LEN = 45  # 统一采样帧数，与模型输入一致


class DataCollector(QObject):
    """实时采集 MediaPipe 手部关键点，保存为训练样本."""

    sample_saved = pyqtSignal(str, str)          # (word, path)
    recording_changed = pyqtSignal(bool)          # is_recording
    collector_error = pyqtSignal(str)

    def __init__(self, data_root: str = DATA_ROOT,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # 解析为项目根目录下的绝对路径，避免工作目录变化导致路径偏移
        if not os.path.isabs(data_root):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(module_dir)
            data_root = os.path.join(project_root, data_root)
        self.data_root = data_root
        self._buffer: list[np.ndarray] = []
        self._current_word: str = ""
        self._recording = False

    # ---- properties ----

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    @property
    def current_word(self) -> str:
        return self._current_word

    # ---- recording control ----

    def start_recording(self, word: str) -> None:
        word = word.strip()
        if not word:
            self.collector_error.emit("手势词不能为空")
            return
        self._current_word = word
        self._buffer.clear()
        self._recording = True
        self.recording_changed.emit(True)

    def stop_recording(self) -> Optional[str]:
        if not self._recording:
            return None
        self._recording = False
        self.recording_changed.emit(False)

        if len(self._buffer) < 5:
            self.collector_error.emit(f"录制帧数过少 ({len(self._buffer)}), 请重新录制")
            return None

        path = self._save()
        if path:
            self._update_labels()
            self.sample_saved.emit(self._current_word, path)
        return path

    # ---- frame buffering ----

    @pyqtSlot(np.ndarray)
    def on_landmarks(self, features: np.ndarray) -> None:
        if not self._recording:
            return
        # 跳过全零帧 (无手部检测)
        if np.all(features == 0):
            return
        self._buffer.append(features.astype(np.float32))

    # ---- save ----

    def _save(self) -> Optional[str]:
        word_dir = os.path.join(self.data_root, self._current_word)
        os.makedirs(word_dir, exist_ok=True)

        # 自动编号
        existing = glob.glob(os.path.join(word_dir, "sample_*.npy"))
        max_idx = 0
        for f in existing:
            try:
                idx = int(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
        sample_name = f"sample_{max_idx + 1:03d}.npy"
        save_path = os.path.join(word_dir, sample_name)

        # 统一采样到 seq_len 帧
        frames = np.stack(self._buffer, axis=0)  # (T, 126)
        frames = self._resample(frames, SEQ_LEN)
        np.save(save_path, frames)
        return save_path

    @staticmethod
    def _resample(frames: np.ndarray, target_len: int) -> np.ndarray:
        """线性插值/降采样到固定帧数."""
        t = frames.shape[0]
        if t == target_len:
            return frames
        indices = np.linspace(0, t - 1, target_len)
        result = np.zeros((target_len, frames.shape[1]), dtype=np.float32)
        for i, idx in enumerate(indices):
            lo = int(np.floor(idx))
            hi = int(np.ceil(idx))
            if lo == hi:
                result[i] = frames[lo]
            else:
                frac = idx - lo
                result[i] = (1 - frac) * frames[lo] + frac * frames[hi]
        return result

    # ---- labels.json ----

    def _update_labels(self) -> None:
        """扫描 data/ 目录，重建 labels.json 索引."""
        if not os.path.isdir(self.data_root):
            return

        words = sorted(
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d)) and not d.startswith(".")
        )
        label_map = {str(i): w for i, w in enumerate(words)}
        labels_path = os.path.join(self.data_root, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)

    # ---- scan existing data ----

    @classmethod
    def scan_samples(cls, data_root: str = DATA_ROOT) -> dict[str, int]:
        """扫描已录数据，返回 {word: count}."""
        if not os.path.isdir(data_root):
            return {}
        result = {}
        for d in sorted(os.listdir(data_root)):
            dpath = os.path.join(data_root, d)
            if os.path.isdir(dpath) and not d.startswith("."):
                npy_files = glob.glob(os.path.join(dpath, "sample_*.npy"))
                if npy_files:
                    result[d] = len(npy_files)
        return result
