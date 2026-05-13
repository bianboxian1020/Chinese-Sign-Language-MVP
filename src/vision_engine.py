"""
vision_engine.py — 视频采集与预处理系统

Handles camera capture, preprocessing, and real-time hand landmark extraction.
- CameraWorker: 摄像头帧采集 + 高斯滤波降噪 (QThread)
- YOLOHandDetector: YOLOv8n 手部ROI检测
- VisionProcessor: MediaPipe Hands (21关键点×2手) + 关键帧提取 + 坐标归一化
- DataQueue: 线程安全环形队列, 供推理模块滑动窗口读取

Pipeline:
  摄像头 → 高斯滤波 → YOLO ROI → MediaPipe Hands → 关键帧筛选 →
  手腕原点归一化[-1,1] → DataQueue → 推理模块
"""
import time
import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot


# ==============================================================================
# 常量定义
# ==============================================================================

HAND_LANDMARK_COUNT = 21          # 单只手的关键点数量
FEATURE_DIM = 2 * HAND_LANDMARK_COUNT * 3  # 126 维 (左右手各 63)

# 手腕和中指MCP索引 (用于归一化)
WRIST = 0
MIDDLE_MCP = 9

# 手部骨架连接 (用于测试模式叠加显示)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),      # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),    # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20),    # 小指
    (5, 9), (9, 13), (13, 17),                # 掌心连接
]


# ==============================================================================
# 线程安全数据队列
# ==============================================================================

class DataQueue:
    """线程安全环形队列, 用于缓冲归一化后的关键点特征帧.

    支持滑动窗口读取: 推理模块调用 get_window(window_size)
    获取最近 N 帧的特征序列.
    """

    def __init__(self, maxlen: int = 90) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total_pushed = 0
        self._total_dropped = 0

    def push(self, features: np.ndarray) -> None:
        """推入一帧特征向量 (shape: 126,)."""
        with self._lock:
            self._buffer.append(features.copy())
            self._total_pushed += 1

    def get_window(self, window_size: int, stride: int = 0) -> Optional[np.ndarray]:
        """获取最近 window_size 帧作为推理序列.

        Args:
            window_size: 滑动窗口大小 (30~60).
            stride: 读取后丢弃的前置帧数 (用于滑动步进).

        Returns:
            np.ndarray shape (window_size, 126) 或 None (缓冲不足).
        """
        with self._lock:
            if len(self._buffer) < window_size:
                return None
            seq = list(self._buffer)[-window_size:]
            if stride > 0:
                for _ in range(min(stride, len(self._buffer))):
                    self._buffer.popleft()
                    self._total_dropped += 1
            return np.stack(seq, axis=0)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return self.size >= 30

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def stats(self) -> tuple[int, int]:
        return self._total_pushed, self._total_dropped


# ==============================================================================
# 摄像头采集线程
# ==============================================================================

class CameraWorker(QThread):
    """摄像头帧采集线程: 读取帧 → 高斯滤波 → 发射信号."""

    frame_ready = pyqtSignal(np.ndarray)        # 降噪后的 BGR 帧
    raw_frame_ready = pyqtSignal(np.ndarray)    # 原始 BGR 帧 (用于显示)
    fps_update = pyqtSignal(float)              # 实时 FPS
    camera_error = pyqtSignal(str)              # 错误信息
    camera_status = pyqtSignal(bool)            # 摄像头状态

    def __init__(self, camera_index: int = 0, fps_target: float = 30.0,
                 enable_blur: bool = True, blur_kernel: tuple = (5, 5),
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self.fps_target = fps_target
        self.enable_blur = enable_blur
        self.blur_kernel = blur_kernel
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self.camera_error.emit(f"无法打开摄像头 (index={self.camera_index})")
            self.camera_status.emit(False)
            return

        self.camera_status.emit(True)
        self._running = True
        frame_interval = 1.0 / max(self.fps_target, 1.0)
        last_time = time.perf_counter()
        fps_accum = 0.0
        fps_frames = 0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                self.camera_error.emit("摄像头帧读取失败 — 尝试重连...")
                self.camera_status.emit(False)
                self._cap.release()
                for _ in range(20):
                    if not self._running:
                        return
                    self.msleep(100)
                if self._running:
                    self._cap = cv2.VideoCapture(self.camera_index)
                    if self._cap.isOpened():
                        self.camera_status.emit(True)
                        continue
                break

            frame = cv2.flip(frame, 1)  # 镜像翻转

            self.raw_frame_ready.emit(frame.copy())

            if self.enable_blur:
                frame = cv2.GaussianBlur(frame, self.blur_kernel, 0)

            self.frame_ready.emit(frame)

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            fps_accum += elapsed
            fps_frames += 1
            if fps_accum >= 1.0:
                self.fps_update.emit(fps_frames / fps_accum)
                fps_accum = 0.0
                fps_frames = 0

            sleep_ms = max(0, int(frame_interval * 1000 - elapsed * 1000))
            if sleep_ms > 0:
                self.msleep(sleep_ms)

    def stop(self) -> None:
        self._running = False


# ==============================================================================
# YOLO 手部检测器
# ==============================================================================

class YOLOHandDetector:
    """YOLOv8n 手部检测器: 从帧中定位手部 ROI.

    使用 Bingsu/adetailer 的 hand_yolov8n.pt 预训练模型 (6.2MB).
    若 YOLO 模型不可用, 自动回退为全帧处理模式.
    """

    def __init__(self, model_path: str = "assets/models/hand_yolov8n.pt",
                 confidence: float = 0.5, device: str = "cpu") -> None:
        self._model = None
        self._fallback = True
        self.confidence = confidence
        self.device = device

        try:
            from ultralytics import YOLO
            import os
            if os.path.exists(model_path):
                self._model = YOLO(model_path)
                self._fallback = False
        except (ImportError, Exception):
            pass

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        if self._fallback or self._model is None:
            return []
        results = self._model(frame, conf=self.confidence, device=self.device,
                              verbose=False, max_det=2)
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), conf))
        return boxes

    def get_roi(self, frame: np.ndarray,
                box: tuple[int, int, int, int, float],
                margin: float = 1.3) -> np.ndarray:
        x, y, w, h, _ = box
        h_frame, w_frame = frame.shape[:2]
        cx, cy = x + w / 2, y + h / 2
        new_w, new_h = w * margin, h * margin
        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(w_frame, int(cx + new_w / 2))
        y2 = min(h_frame, int(cy + new_h / 2))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]

    @property
    def is_available(self) -> bool:
        return not self._fallback


# ==============================================================================
# 关键帧提取器
# ==============================================================================

class KeyFrameExtractor:
    """基于帧间差法的关键帧提取器.

    仅当相邻帧之间的手部特征差异超过阈值时才标记为关键帧,
    减少推入推理队列的数据量, 降低计算开销.
    """

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = threshold
        self._prev_features: Optional[np.ndarray] = None

    def is_key_frame(self, features: np.ndarray) -> bool:
        if self._prev_features is None:
            self._prev_features = features.copy()
            return True
        diff = float(np.mean(np.abs(features - self._prev_features)))
        self._prev_features = features.copy()
        return diff > self.threshold

    def reset(self) -> None:
        self._prev_features = None


# ==============================================================================
# 视觉处理引擎 (MediaPipe Hands)
# ==============================================================================

class VisionProcessor(QObject):
    """MediaPipe Hands 处理器: 提取 21×2 手部关键点, 归一化到 [-1, 1].

    工作流程:
    1. 接收降噪后的 BGR 帧
    2. 可选: YOLO 手部 ROI 裁剪
    3. MediaPipe Hands 推理 (每只手 21 个关键点)
    4. 坐标归一化: 手腕为原点, 腕-中指MCP距离为缩放因子 → [-1, 1]
    5. 关键帧筛选 (帧间差法)
    6. 推入 DataQueue 供推理模块消费
    7. 测试模式: 在帧上绘制手部骨架叠加层
    """

    landmarks_ready = pyqtSignal(np.ndarray)     # 归一化特征向量 (126,)
    frame_with_overlay = pyqtSignal(np.ndarray)  # 带骨架叠加的帧 (测试模式)
    keyframe_rate = pyqtSignal(float)             # 关键帧率
    vision_error = pyqtSignal(str)                # 错误信息
    vision_status = pyqtSignal(str)               # 状态信息

    def __init__(self, hand_model_path: str = "assets/models/hand_landmarker.task",
                 enable_keyframe: bool = True,
                 keyframe_threshold: float = 0.02,
                 enable_overlay: bool = False,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.enable_overlay = enable_overlay
        self._landmarker = None
        self._hand_detector: Optional[YOLOHandDetector] = None
        self._keyframe_extractor = KeyFrameExtractor(threshold=keyframe_threshold)
        self._enable_keyframe = enable_keyframe
        self._data_queue = DataQueue(maxlen=90)

        self._total_frames = 0
        self._key_frames = 0
        self._last_keyframe_rate = 0.0
        self._latest_raw_landmarks = None  # 最新原始关键点 (用于叠加绘制)
        self._frame_ts = 0  # MediaPipe VIDEO 模式帧时间戳 (毫秒)

        self._init_landmarker(hand_model_path)

    def _init_landmarker(self, model_path: str) -> None:
        """初始化 MediaPipe HandLandmarker (Tasks API, VIDEO 同步模式, GPU 加速)."""
        try:
            import importlib
            mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
            from mediapipe.tasks.python.core.base_options import BaseOptions

            # 尝试 GPU delegate, 回退 CPU
            delegate = BaseOptions.Delegate.CPU
            try:
                base = BaseOptions(model_asset_path=model_path,
                                   delegate=BaseOptions.Delegate.GPU)
                # 测试 GPU 是否可用: 创建临时实例
                test_opts = mp_vision.HandLandmarkerOptions(
                    base_options=base,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                )
                test_lm = mp_vision.HandLandmarker.create_from_options(test_opts)
                test_lm.close()
                delegate = BaseOptions.Delegate.GPU
                self.vision_status.emit("MediaPipe GPU 加速已启用")
            except Exception:
                self.vision_status.emit("MediaPipe GPU 不可用, 使用 CPU")

            base = BaseOptions(model_asset_path=model_path, delegate=delegate)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self.vision_status.emit("MediaPipe HandLandmarker 初始化成功")
        except FileNotFoundError:
            self.vision_error.emit(f"MediaPipe 模型文件未找到: {model_path}")
        except Exception as e:
            self.vision_error.emit(f"MediaPipe 初始化失败: {e}")

    def _process_landmark_result(self, result) -> None:
        """处理 MediaPipe 同步返回结果: 提取原始关键点用于叠加绘制."""
        if result.hand_landmarks:
            self._latest_raw_landmarks = result.hand_landmarks
            self._latest_handedness = result.handedness
        else:
            self._latest_raw_landmarks = None
            self._latest_handedness = None

    def set_yolo_detector(self, detector: YOLOHandDetector) -> None:
        self._hand_detector = detector

    @property
    def data_queue(self) -> DataQueue:
        return self._data_queue

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def keyframe_stats(self) -> tuple[int, int, float]:
        return self._key_frames, self._total_frames, self._last_keyframe_rate

    # ------------------------------------------------------------------
    # 主处理入口
    # ------------------------------------------------------------------

    @pyqtSlot(np.ndarray)
    def process_frame(self, frame: np.ndarray) -> None:
        """接收降噪帧并执行完整的预处理管线."""
        if self._landmarker is None:
            return

        self._total_frames += 1
        display_frame = frame.copy()

        # --- YOLO ROI (可选) ---
        process_frame = frame
        hand_boxes = []
        if self._hand_detector is not None and self._hand_detector.is_available:
            hand_boxes = self._hand_detector.detect(frame)
            if hand_boxes:
                # 取最大置信度的检测框
                best = max(hand_boxes, key=lambda b: b[4])
                process_frame = self._hand_detector.get_roi(frame, best)

        # --- MediaPipe 同步推理 ---
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, self._frame_ts)
            self._frame_ts += 33  # ~30fps
            self._process_landmark_result(result)
        except Exception as e:
            self.vision_error.emit(f"MediaPipe 推理异常: {e}")
            return

        # --- 特征提取与归一化 ---
        features = self._extract_features()
        if features is not None:
            is_key = True
            if self._enable_keyframe:
                is_key = self._keyframe_extractor.is_key_frame(features)
            if is_key:
                self._key_frames += 1
                self._data_queue.push(features)
                self.landmarks_ready.emit(features)

            if self._total_frames % 30 == 0:
                self._last_keyframe_rate = (
                    self._key_frames / max(self._total_frames, 1)
                )
                self.keyframe_rate.emit(self._last_keyframe_rate)
        else:
            self._keyframe_extractor.reset()

        # --- 骨架叠加 ---
        if self.enable_overlay:
            if self._latest_raw_landmarks is not None:
                display_frame = self._draw_hands_overlay(display_frame, hand_boxes)
            else:
                h, w = display_frame.shape[:2]
                cv2.putText(display_frame, "No hand", (w // 2 - 50, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_frame, f"FPS: {self._last_keyframe_rate:.0f} | Q: {self._data_queue.size}",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.frame_with_overlay.emit(display_frame)

    # ------------------------------------------------------------------
    # 特征提取与归一化
    # ------------------------------------------------------------------

    def _extract_features(self) -> Optional[np.ndarray]:
        """从最新 MediaPipe 结果中提取归一化手部关键点.

        Returns:
            (126,) 归一化到 [-1, 1] 的特征向量, 或 None (无手部).
            126 维 = 左手(63) + 右手(63), 每只手 = 21点 × (x, y, z).
            若某只手未检测到, 对应 63 维填 0.
        """
        if self._latest_raw_landmarks is None:
            return None

        # 按左右手分类
        left_landmarks = None
        right_landmarks = None
        if self._latest_handedness and len(self._latest_handedness) == len(
            self._latest_raw_landmarks
        ):
            for hand_idx, handedness in enumerate(self._latest_handedness):
                # handedness[0].category_name = "Left" or "Right"
                category = handedness[0].category_name
                lm = self._landmarks_to_xyz(self._latest_raw_landmarks[hand_idx])
                if category == "Left":
                    left_landmarks = lm
                else:
                    right_landmarks = lm
        else:
            # 回退: 无法区分左右手, 按检测顺序分配
            if len(self._latest_raw_landmarks) >= 1:
                right_landmarks = self._landmarks_to_xyz(
                    self._latest_raw_landmarks[0]
                )
            if len(self._latest_raw_landmarks) >= 2:
                left_landmarks = self._landmarks_to_xyz(
                    self._latest_raw_landmarks[1]
                )

        if left_landmarks is None and right_landmarks is None:
            return None

        features = []
        for hand_lm in (left_landmarks, right_landmarks):
            if hand_lm is not None:
                normalized = self._normalize_landmarks(hand_lm)
                features.append(normalized.flatten())
            else:
                features.append(np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float32))

        return np.concatenate(features)

    @staticmethod
    def _landmarks_to_xyz(hand_landmarks) -> np.ndarray:
        """将 MediaPipe NormalizedLandmark 列表转为 (21, 3) 数组."""
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
        )

    @staticmethod
    def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """归一化手部关键点: 手腕原点 → 缩放 → 裁剪到 [-1, 1].

        Algorithm:
        1. 以手腕 (landmark 0) 为原点平移
        2. 以手腕→中指MCP (landmark 9) 距离为缩放因子
        3. 裁剪到 [-1.0, 1.0]

        Args:
            landmarks: (21, 3) 原始归一化坐标 (MediaPipe 已归一化到 [0,1]).

        Returns:
            (21, 3) 归一化到 [-1, 1] 的坐标.
        """
        wrist = landmarks[WRIST]
        mcp = landmarks[MIDDLE_MCP]
        scale = float(np.linalg.norm(mcp - wrist))
        if scale < 1e-6:
            scale = 1.0
        normalized = (landmarks - wrist) / scale
        return np.clip(normalized, -1.0, 1.0)

    # ------------------------------------------------------------------
    # 骨架叠加绘制 (测试模式)
    # ------------------------------------------------------------------

    def _draw_hands_overlay(self, frame: np.ndarray,
                            hand_boxes: list) -> np.ndarray:
        """在帧上绘制手部骨架 + 检测框.

        Args:
            frame: 原始 BGR 帧 (会被原地修改).
            hand_boxes: YOLO 手部检测框列表.

        Returns:
            含叠加层的帧.
        """
        h, w = frame.shape[:2]

        # YOLO 检测框
        for (x, y, bw, bh, conf) in hand_boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.putText(frame, f"H:{conf:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 手部骨架
        if self._latest_raw_landmarks:
            colors = [(255, 0, 0), (0, 0, 255)]  # 左手蓝, 右手红
            for idx, hand_lm in enumerate(self._latest_raw_landmarks):
                color = colors[idx % 2]
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], color, 2)
                for px, py in points:
                    cv2.circle(frame, (px, py), 3, color, -1)

        # 状态信息
        cv2.putText(frame, f"KeyFrm: {self._last_keyframe_rate:.2f}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Queue: {self._data_queue.size}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._data_queue.clear()
