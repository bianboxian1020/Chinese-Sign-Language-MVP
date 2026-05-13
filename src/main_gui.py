"""
main_gui.py — 心语速译 PyQt6 主界面

CNN-BiLSTM-Attention 手语双向转译系统 GUI.
- 分屏界面: 左侧摄像头+骨架叠加 | 右侧识别结果+手语视频
- 四线程并发: 主线程(GUI) + CameraWorker + AudioRecorder
- VisionProcessor + SlidingWindowPredictor 在主线程 (信号驱动)
"""
import os
import sys
from functools import partial
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QListWidget, QListWidgetItem,
    QStatusBar, QGroupBox, QTextEdit, QInputDialog, QMessageBox,
    QSizePolicy, QCheckBox, QFileDialog, QComboBox, QLineEdit, QScrollArea,
)

from vision_engine import (
    CameraWorker, VisionProcessor, YOLOHandDetector, DataQueue, FEATURE_DIM,
)
from inference import SlidingWindowPredictor
from model_config import ModelConfig
from audio_engine import AudioRecorder, TTSPlayer
from data_collector import DataCollector

# ---- 常量 ----

MODEL_PATH = "assets/models/hand_landmarker.task"
WEIGHTS_PATH = "assets/models/xinyu_model_best.pth"
LABELS_PATH = "assets/models/labels.json"
DATA_LABELS_PATH = "data/labels.json"
YOLO_MODEL_PATH = "assets/models/hand_yolov8n.pt"
SIGN_VIDEOS_DIR = "assets/sign_videos"

KEYWORD_VIDEO_MAP = {
    "你好": "nihao.mp4", "谢谢": "xiexie.mp4", "对不起": "duibuqi.mp4",
    "我爱你": "woaini.mp4", "是": "shi.mp4", "不": "bu.mp4",
    "好": "hao.mp4", "再见": "zaijian.mp4",
}


class SignVideoPlayer(QWidget):
    """手语演示视频播放器 (MVP: 占位符)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel("手语演示视频")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "background-color: #1a1a2e; color: #e0e0e0; "
            "border: 2px solid #16213e; border-radius: 8px; "
            "font-size: 16px; min-height: 160px;"
        )
        layout.addWidget(self._label)

    def play_sign(self, label: str) -> None:
        self._label.setText(f"{label}\n(视频待录制)")


class MainWindow(QMainWindow):
    """主窗口."""

    def __init__(self) -> None:
        super().__init__()

        # 模块实例
        self._camera_worker: Optional[CameraWorker] = None
        self._vision_processor: Optional[VisionProcessor] = None
        self._predictor: Optional[SlidingWindowPredictor] = None
        self._audio_recorder: Optional[AudioRecorder] = None
        self._tts_player: Optional[TTSPlayer] = None
        self._yolo_detector: Optional[YOLOHandDetector] = None
        self._data_collector: Optional[DataCollector] = None

        # 状态
        self._camera_active = False
        self._last_sign: Optional[str] = None
        self._history: list[tuple[str, float, float, str]] = []  # (label, conf, unc, time)
        self._show_overlay = False

        self._init_ui()
        self._init_modules()
        self._wire_signals()

    # ==================================================================
    # UI
    # ==================================================================

    def _init_ui(self) -> None:
        self.setWindowTitle("心语速译 — CNN+BiLSTM 手语双向转译")
        self.setMinimumSize(1200, 750)
        self._apply_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # 主分屏
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        # 底部: 语音→手语
        root.addWidget(self._build_bottom_panel())

        # 状态栏
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._st_fps = QLabel("FPS: --")
        self._st_cam = QLabel("摄像头: 未开启")
        self._st_model = QLabel("模型: 未加载")
        self._st_queue = QLabel("队列: 0")
        self._status_bar.addWidget(self._st_fps)
        self._status_bar.addWidget(self._st_cam)
        self._status_bar.addWidget(self._st_model)
        self._status_bar.addWidget(self._st_queue)

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f23; color: #e0e0e0; }
            QGroupBox { color: #a0c4ff; font-weight: bold; border: 1px solid #2a2a4a;
                        border-radius: 6px; margin-top: 12px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QPushButton { background-color: #1a1a3e; color: #e0e0e0;
                          border: 1px solid #3a3a6a; border-radius: 4px;
                          padding: 6px 16px; font-size: 13px; }
            QPushButton:hover { background-color: #2a2a5e; border-color: #5a5a9a; }
            QPushButton:pressed { background-color: #3a3a7e; }
            QPushButton:checked { background-color: #2a4a2e; border-color: #4a8a4e; }
            QPushButton:disabled { background-color: #1a1a1a; color: #666; }
            QLabel { color: #e0e0e0; }
            QCheckBox { color: #e0e0e0; }
            QListWidget { background-color: #12122a; color: #ccc; border: 1px solid #2a2a4a; }
            QListView { background-color: #12122a; color: #ccc; }
            QTextEdit { background-color: #12122a; color: #ccc; border: 1px solid #2a2a4a; }
            QLineEdit { background-color: #12122a; color: #e0e0e0; border: 1px solid #2a2a4a;
                        border-radius: 4px; padding: 4px; }
            QComboBox { background-color: #12122a; color: #e0e0e0; border: 1px solid #2a2a4a;
                        border-radius: 4px; padding: 4px; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView { background-color: #12122a; color: #ccc;
                                          selection-background-color: #2a2a5e; }
            QProgressBar { background-color: #12122a; border: 1px solid #2a2a4a;
                           border-radius: 3px; text-align: center; color: #e0e0e0; }
            QProgressBar::chunk { background-color: #4a9a6a; border-radius: 2px; }
            QStatusBar { background-color: #0a0a1a; color: #888; }
            QSplitter::handle { background-color: #2a2a4a; width: 2px; }
            QScrollArea { background-color: #0f0f23; border: none; }
        """)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # 摄像头
        cam_group = QGroupBox("摄像头实时画面 (骨架叠加)")
        cam_layout = QVBoxLayout(cam_group)
        self._cam_label = QLabel("摄像头未开启")
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet(
            "background-color: #0a0a1a; border: 2px solid #1a1a3e; "
            "border-radius: 8px; min-height: 400px; font-size: 14px; color: #666;"
        )
        cam_layout.addWidget(self._cam_label)
        layout.addWidget(cam_group, stretch=1)

        # 控制按钮
        btn_row = QHBoxLayout()
        self._btn_camera = QPushButton("开启摄像头")
        self._btn_camera.setCheckable(True)
        self._btn_camera.clicked.connect(self._toggle_camera)
        btn_row.addWidget(self._btn_camera)

        self._chk_overlay = QCheckBox("骨架叠加")
        self._chk_overlay.stateChanged.connect(self._toggle_overlay)
        btn_row.addWidget(self._chk_overlay)

        self._btn_load_model = QPushButton("加载模型...")
        self._btn_load_model.clicked.connect(self._load_model_dialog)
        btn_row.addWidget(self._btn_load_model)

        layout.addLayout(btn_row)

        # 状态信息
        info_row = QHBoxLayout()
        self._lbl_keyframe = QLabel("关键帧率: --")
        self._lbl_keyframe.setStyleSheet("color: #888; font-size: 11px;")
        info_row.addWidget(self._lbl_keyframe)
        info_row.addStretch()
        layout.addLayout(info_row)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 识别结果
        result_group = QGroupBox("识别结果 (手语 → 语音)")
        result_layout = QVBoxLayout(result_group)

        self._model_status = QLabel("模型: 未加载")
        self._model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_status.setStyleSheet(
            "color: #cc4444; font-size: 11px; padding: 2px; "
            "background-color: #1a1a2e; border-radius: 3px;"
        )
        result_layout.addWidget(self._model_status)

        self._sign_label = QLabel("等待手势...")
        self._sign_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sign_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #a0ffa0; padding: 12px;"
        )
        result_layout.addWidget(self._sign_label)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("置信度:"))
        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        conf_row.addWidget(self._conf_bar)
        result_layout.addLayout(conf_row)

        self._unc_label = QLabel("不确定度: --")
        self._unc_label.setStyleSheet("color: #888; font-size: 11px;")
        result_layout.addWidget(self._unc_label)

        btn_row = QHBoxLayout()
        self._btn_speak = QPushButton("重新播报")
        self._btn_speak.clicked.connect(self._speak_again)
        self._btn_speak.setEnabled(False)
        btn_row.addWidget(self._btn_speak)

        self._btn_hist = QPushButton("历史记录 ▼")
        self._btn_hist.setCheckable(True)
        self._btn_hist.clicked.connect(self._toggle_history)
        btn_row.addWidget(self._btn_hist)
        result_layout.addLayout(btn_row)

        layout.addWidget(result_group)

        # 历史记录 (独立紧凑)
        self._hist_list = QListWidget()
        self._hist_list.setVisible(False)
        self._hist_list.setMaximumHeight(60)
        self._hist_list.setStyleSheet(
            "QListWidget { background-color: #0d0d20; color: #aaa; "
            "border: 1px solid #2a2a4a; border-radius: 4px; font-size: 11px; }"
        )
        layout.addWidget(self._hist_list)

        # 数据采集
        layout.addWidget(self._build_collection_panel())

        # 手语视频
        video_group = QGroupBox("手语演示 (语音 → 手语)")
        video_layout = QVBoxLayout(video_group)
        self._video_player = SignVideoPlayer()
        video_layout.addWidget(self._video_player)
        layout.addWidget(video_group, stretch=1)

        # 套 ScrollArea 防止内容溢出重叠
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        container = QWidget()
        container.setStyleSheet("background-color: #0f0f23;")
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(scroll)
        return container

    def _build_bottom_panel(self) -> QWidget:
        panel = QGroupBox("语音输入 (语音 → 手语)")
        layout = QVBoxLayout(panel)

        top_row = QHBoxLayout()
        self._btn_mic = QPushButton("按住说话")
        self._btn_mic.setMinimumHeight(44)
        self._btn_mic.setStyleSheet(
            "QPushButton { font-size: 15px; font-weight: bold; "
            "background-color: #1a3a1a; border: 2px solid #3a6a3a; }"
            "QPushButton:hover { background-color: #2a4a2a; }"
            "QPushButton:pressed { background-color: #3a6a3a; border-color: #5aaa5a; }"
        )
        self._btn_mic.pressed.connect(self._start_listening)
        self._btn_mic.released.connect(self._stop_listening)
        top_row.addWidget(self._btn_mic, stretch=2)

        self._asr_text = QTextEdit()
        self._asr_text.setReadOnly(True)
        self._asr_text.setPlaceholderText("识别文字将在此显示...")
        self._asr_text.setMaximumHeight(44)
        top_row.addWidget(self._asr_text, stretch=3)
        layout.addLayout(top_row)

        self._asr_status = QLabel("状态: 就绪")
        self._asr_status.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self._asr_status)
        return panel

    def _build_collection_panel(self) -> QWidget:
        panel = QGroupBox("数据采集 (录制训练样本)")
        layout = QVBoxLayout(panel)

        word_row = QHBoxLayout()
        word_row.addWidget(QLabel("手势词:"))
        self._word_input = QLineEdit()
        self._word_input.setPlaceholderText("输入手势词...")
        word_row.addWidget(self._word_input, stretch=2)
        self._word_combo = QComboBox()
        self._word_combo.setEditable(False)
        self._word_combo.currentTextChanged.connect(self._on_combo_word_changed)
        word_row.addWidget(self._word_combo, stretch=1)
        layout.addLayout(word_row)

        rec_row = QHBoxLayout()
        self._lbl_frame_count = QLabel("帧数: 0")
        self._lbl_frame_count.setStyleSheet("color: #888; font-size: 12px;")
        rec_row.addWidget(self._lbl_frame_count)
        rec_row.addStretch()
        self._btn_record = QPushButton("开始录制")
        self._btn_record.setCheckable(True)
        self._btn_record.setMinimumHeight(36)
        self._btn_record.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #1a3a1a; "
            "border: 2px solid #3a6a3a; }"
            "QPushButton:hover { background-color: #2a4a2a; }"
            "QPushButton:checked { background-color: #6a1a1a; "
            "border: 2px solid #aa4444; }"
            "QPushButton:checked:hover { background-color: #8a2a2a; }"
        )
        self._btn_record.clicked.connect(self._toggle_recording)
        rec_row.addWidget(self._btn_record)
        layout.addLayout(rec_row)

        self._lbl_samples = QLabel("已录样本: (无)")
        self._lbl_samples.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._lbl_samples)

        self._record_timer = QTimer(self)
        self._record_timer.timeout.connect(self._update_record_display)

        return panel

    # ==================================================================
    # 模块初始化
    # ==================================================================

    def _init_modules(self) -> None:
        # 视觉处理器
        model_path = self._resolve_path(MODEL_PATH)
        self._vision_processor = VisionProcessor(
            hand_model_path=model_path,
            enable_keyframe=False,  # 关掉关键帧过滤, 每帧都入缓冲, 加速响应
            enable_overlay=False,
            parent=self,
        )
        # YOLO 检测器
        yolo_path = self._resolve_path(YOLO_MODEL_PATH)
        self._yolo_detector = YOLOHandDetector(model_path=yolo_path)
        self._vision_processor.set_yolo_detector(self._yolo_detector)

        # 推理器 — 尝试从 checkpoint 读取配置以匹配 num_classes
        weights_path = self._resolve_path(WEIGHTS_PATH)
        num_classes = 500
        if os.path.exists(weights_path):
            try:
                import torch
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
                if "config" in ckpt and "num_classes" in ckpt["config"]:
                    num_classes = ckpt["config"]["num_classes"]
            except Exception:
                pass

        config = ModelConfig(seq_len=45, input_dim=126, num_classes=num_classes)
        self._predictor = SlidingWindowPredictor(config=config, parent=self)

        # 尝试加载标签 (优先资产目录, 回退数据目录)
        for lp in [LABELS_PATH, DATA_LABELS_PATH]:
            lp = self._resolve_path(lp)
            if os.path.exists(lp):
                labels = SlidingWindowPredictor.load_labels_from_file(lp)
                if labels:
                    self._predictor.set_label_map(labels)
                    break

        # 自动加载权重
        if os.path.exists(weights_path):
            self._predictor.load_weights(weights_path)

        self._update_model_status()

        # TTS
        self._tts_player = TTSPlayer(parent=self)

        # 音频录制
        self._audio_recorder = AudioRecorder(parent=self)

        # 数据采集
        self._data_collector = DataCollector(parent=self)
        # 特征 → 数据采集 (始终连接,不依赖摄像头开关)
        if self._vision_processor:
            self._vision_processor.landmarks_ready.connect(
                self._data_collector.on_landmarks, Qt.ConnectionType.QueuedConnection
            )

        self._refresh_sample_list()

    def _resolve_path(self, relative: str) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, relative)

    # ==================================================================
    # 信号连接
    # ==================================================================

    def _wire_signals(self) -> None:
        # 预测器
        if self._predictor:
            self._predictor.sign_recognized.connect(self._on_sign_recognized)
            self._predictor.predictor_error.connect(
                partial(self._on_error, "推理")
            )
            self._predictor.predictor_status.connect(
                partial(self._on_status, "模型")
            )

        # 视觉处理器
        if self._vision_processor:
            self._vision_processor.vision_error.connect(
                partial(self._on_error, "视觉")
            )
            self._vision_processor.vision_status.connect(
                partial(self._on_status, "视觉")
            )
            self._vision_processor.keyframe_rate.connect(self._on_keyframe_rate)

        # TTS
        if self._tts_player:
            self._tts_player.tts_error.connect(
                partial(self._on_error, "TTS")
            )

        # 音频录制
        if self._audio_recorder:
            self._audio_recorder.asr_result.connect(self._on_asr_result)
            self._audio_recorder.audio_error.connect(
                partial(self._on_error, "ASR")
            )

        # 数据采集
        if self._data_collector:
            self._data_collector.sample_saved.connect(self._on_sample_saved)
            self._data_collector.recording_changed.connect(self._on_recording_changed)
            self._data_collector.collector_error.connect(
                partial(self._on_error, "采集")
            )

    def _wire_camera(self) -> None:
        if self._camera_worker is None or self._vision_processor is None:
            return
        # 帧 → GUI 显示
        self._camera_worker.raw_frame_ready.connect(
            self._on_raw_frame, Qt.ConnectionType.QueuedConnection
        )
        # 帧 → 视觉处理
        self._camera_worker.frame_ready.connect(
            self._vision_processor.process_frame, Qt.ConnectionType.QueuedConnection
        )
        # 叠加帧 → GUI
        self._vision_processor.frame_with_overlay.connect(
            self._on_overlay_frame, Qt.ConnectionType.QueuedConnection
        )
        # 特征 → 推理
        self._vision_processor.landmarks_ready.connect(
            self._predictor.on_landmarks, Qt.ConnectionType.QueuedConnection
        )
        # FPS
        self._camera_worker.fps_update.connect(self._on_fps)
        # 摄像头状态
        self._camera_worker.camera_error.connect(
            partial(self._on_error, "摄像头")
        )
        self._camera_worker.camera_status.connect(self._on_camera_status)

    # ==================================================================
    # 槽
    # ==================================================================

    @pyqtSlot(np.ndarray)
    def _on_raw_frame(self, frame: np.ndarray) -> None:
        if self._show_overlay:
            return  # 等待叠加帧
        self._display_frame(frame)

    @pyqtSlot(np.ndarray)
    def _on_overlay_frame(self, frame: np.ndarray) -> None:
        if self._show_overlay:
            self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self._cam_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cam_label.setPixmap(pixmap)

    @pyqtSlot(float)
    def _on_fps(self, fps: float) -> None:
        self._st_fps.setText(f"FPS: {fps:.0f}")

    @pyqtSlot(bool)
    def _on_camera_status(self, ok: bool) -> None:
        self._st_cam.setText("摄像头: 正常" if ok else "摄像头: 异常")
        self._st_cam.setStyleSheet("color: #4a9a4a;" if ok else "color: #cc4444;")

    @pyqtSlot(float)
    def _on_keyframe_rate(self, rate: float) -> None:
        self._lbl_keyframe.setText(f"关键帧率: {rate:.2f}")
        qsize = self._vision_processor.data_queue.size if self._vision_processor else 0
        self._st_queue.setText(f"队列: {qsize}")

    @pyqtSlot(str, float, float)
    def _on_sign_recognized(self, label: str, confidence: float,
                            uncertainty: float) -> None:
        self._sign_label.setText(label)
        conf_pct = int(confidence * 100)
        self._conf_bar.setValue(conf_pct)
        self._unc_label.setText(f"不确定度: {uncertainty:.4f}")

        # 颜色反馈
        if conf_pct >= 85:
            color = "#a0ffa0"
        elif conf_pct >= 65:
            color = "#ffcc00"
        else:
            color = "#ff8888"
        self._sign_label.setStyleSheet(
            f"font-size: 36px; font-weight: bold; color: {color}; padding: 12px;"
        )

        self._last_sign = label
        self._btn_speak.setEnabled(True)

        # 历史
        ts = datetime.now().strftime("%H:%M:%S")
        self._history.append((label, confidence, uncertainty, ts))
        item = QListWidgetItem(f"[{ts}] {label} ({conf_pct}%, u={uncertainty:.3f})")
        self._hist_list.insertItem(0, item)

        # TTS
        if self._tts_player:
            self._tts_player.speak(label)

    @pyqtSlot(str)
    def _on_asr_result(self, text: str) -> None:
        self._asr_text.setText(text)
        self._asr_status.setText(f"已识别: {text}")
        self._asr_status.setStyleSheet("color: #4a9a4a; font-size: 12px;")
        # 关键词匹配
        for keyword in sorted(KEYWORD_VIDEO_MAP, key=len, reverse=True):
            if keyword in text:
                self._video_player.play_sign(keyword)
                return

    def _on_error(self, source: str, msg: str) -> None:
        self._status_bar.showMessage(f"[{source}] {msg}", 8000)

    def _on_status(self, source: str, msg: str) -> None:
        self._status_bar.showMessage(f"[{source}] {msg}", 5000)
        if source == "模型":
            self._st_model.setText(f"模型: {msg[:30]}")
        if source == "视觉":
            self._st_model.setText(f"视觉: {msg[:30]}")

    # ==================================================================
    # 摄像头控制
    # ==================================================================

    def _toggle_camera(self, checked: bool) -> None:
        if checked:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self) -> None:
        self._camera_worker = CameraWorker(camera_index=0, fps_target=30,
                                           enable_blur=True, parent=self)
        self._wire_camera()
        self._camera_worker.start()
        self._btn_camera.setText("关闭摄像头")
        self._st_cam.setText("摄像头: 启动中...")

    def _stop_camera(self) -> None:
        if self._camera_worker is None:
            return
        self._camera_worker.stop()
        self._camera_worker.wait(3000)
        self._camera_worker = None
        self._btn_camera.setText("开启摄像头")
        self._btn_camera.setChecked(False)
        self._cam_label.setText("摄像头未开启")
        self._st_cam.setText("摄像头: 未开启")
        self._st_cam.setStyleSheet("color: #888;")
        self._st_fps.setText("FPS: --")

    # ==================================================================
    # 骨架叠加
    # ==================================================================

    def _toggle_overlay(self, state: int) -> None:
        self._show_overlay = (state == Qt.CheckState.Checked.value)
        if self._vision_processor:
            self._vision_processor.enable_overlay = self._show_overlay

    # ==================================================================
    # 模型加载
    # ==================================================================

    def _load_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "加载模型权重", "assets/models/",
            "Model Files (*.pth *.pt);;All Files (*)"
        )
        if path and self._predictor:
            ok = self._predictor.load_weights(path)
            self._update_model_status()
            if ok:
                self._st_model.setText(
                    f"模型: {os.path.basename(path)}"
                )
                self._st_model.setStyleSheet("color: #4a9a4a;")
            else:
                self._st_model.setText("模型: 加载失败")
                self._st_model.setStyleSheet("color: #cc4444;")

    def _update_model_status(self) -> None:
        if self._predictor and self._predictor.is_model_loaded:
            n = len(self._predictor.label_map)
            words = list(self._predictor.label_map.values())[:8]
            word_str = ", ".join(words)
            if n > 8:
                word_str += f" ... (+{n - 8})"
            self._model_status.setText(f"模型已加载 | {n} 个词: {word_str}")
            self._model_status.setStyleSheet(
                "color: #4a9a4a; font-size: 11px; padding: 2px; "
                "background-color: #1a2a1a; border-radius: 3px;"
            )
        else:
            self._model_status.setText("模型: 未加载")
            self._model_status.setStyleSheet(
                "color: #cc4444; font-size: 11px; padding: 2px; "
                "background-color: #1a1a2e; border-radius: 3px;"
            )

    # ==================================================================
    # TTS
    # ==================================================================

    def _speak_again(self) -> None:
        if self._last_sign and self._tts_player:
            self._tts_player.speak(self._last_sign)

    # ==================================================================
    # 历史
    # ==================================================================

    def _toggle_history(self, checked: bool) -> None:
        self._hist_list.setVisible(checked)

    # ==================================================================
    # 语音录制
    # ==================================================================

    def _start_listening(self) -> None:
        if self._audio_recorder is None:
            return
        self._btn_mic.setText("正在聆听...")
        self._asr_status.setText("状态: 聆听中...")
        self._asr_status.setStyleSheet("color: #ffcc00; font-size: 12px;")
        if not self._audio_recorder.isRunning():
            self._audio_recorder.start()

    def _stop_listening(self) -> None:
        self._btn_mic.setText("按住说话")
        self._asr_status.setText("状态: 就绪")
        self._asr_status.setStyleSheet("color: #888; font-size: 12px;")

    # ==================================================================
    # 数据采集
    # ==================================================================

    def _toggle_recording(self, checked: bool) -> None:
        if self._data_collector is None:
            return
        if checked:
            word = self._word_input.text().strip()
            if not word:
                QMessageBox.warning(self, "提示", "请先输入手势词")
                self._btn_record.setChecked(False)
                return
            if self._camera_worker is None or not self._camera_worker.isRunning():
                QMessageBox.warning(self, "提示", "请先开启摄像头")
                self._btn_record.setChecked(False)
                return
            self._data_collector.start_recording(word)
            self._record_timer.start(100)
        else:
            self._data_collector.stop_recording()
            self._record_timer.stop()
            self._lbl_frame_count.setText("帧数: 0")

    def _on_recording_changed(self, is_recording: bool) -> None:
        if is_recording:
            self._btn_record.setText("停止录制")
            self._word_input.setEnabled(False)
            self._word_combo.setEnabled(False)
        else:
            self._btn_record.setText("开始录制")
            self._btn_record.setChecked(False)
            self._word_input.setEnabled(True)
            self._word_combo.setEnabled(True)

    def _update_record_display(self) -> None:
        if self._data_collector and self._data_collector.is_recording:
            self._lbl_frame_count.setText(f"帧数: {self._data_collector.frame_count}")

    @pyqtSlot(str, str)
    def _on_sample_saved(self, word: str, path: str) -> None:
        self._status_bar.showMessage(f"[采集] {word} 已保存: {path}", 5000)
        self._refresh_sample_list()

    def _on_combo_word_changed(self, text: str) -> None:
        if text:
            self._word_input.setText(text)

    def _refresh_sample_list(self) -> None:
        samples = DataCollector.scan_samples(self._data_collector.data_root) if self._data_collector else {}
        if not samples:
            self._lbl_samples.setText("已录样本: (无)")
        else:
            lines = [f"{w}: {n} 个" for w, n in samples.items()]
            self._lbl_samples.setText("已录样本: " + " | ".join(lines))

        # 更新下拉列表
        current = self._word_combo.currentText()
        self._word_combo.blockSignals(True)
        self._word_combo.clear()
        self._word_combo.addItem("— 选择已录词 —")
        for w in sorted(samples):
            self._word_combo.addItem(w)
        if current and current in samples:
            self._word_combo.setCurrentText(current)
        self._word_combo.blockSignals(False)

    def closeEvent(self, event) -> None:
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker.wait(3000)
            if self._camera_worker.isRunning():
                self._camera_worker.terminate()
        if self._vision_processor is not None:
            self._vision_processor.close()
        if self._audio_recorder is not None:
            self._audio_recorder.stop()
            self._audio_recorder.wait(3000)
            if self._audio_recorder.isRunning():
                self._audio_recorder.terminate()
        if self._tts_player is not None:
            self._tts_player.stop()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("心语速译")
    app.setApplicationVersion("2.0 CNN-BiLSTM")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
