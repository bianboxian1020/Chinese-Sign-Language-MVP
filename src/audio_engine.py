"""
audio_engine.py — 语音引擎模块

Handles microphone-based speech recognition (ASR) and text-to-speech (TTS) output.
- AudioRecorder: dedicated QThread for continuous mic listening + Google ASR (Chinese)
- TTSPlayer: pyttsx3 with non-blocking event loop (startLoop(False))
"""
import tempfile
import os
import threading
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot


class AudioRecorder(QThread):
    """Dedicated thread for microphone recording and speech recognition.

    Uses speech_recognition library with Google Web Speech API for Chinese.
    Runs a background listening loop that triggers on detected speech.
    """

    # Signals (cross-thread)
    asr_result = pyqtSignal(str)       # recognized Chinese text
    asr_partial = pyqtSignal(str)      # partial/interim result (if available)
    audio_error = pyqtSignal(str)      # error messages
    recording_state = pyqtSignal(bool) # True when actively listening

    def __init__(self, language: str = "zh-CN",
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.language = language
        self._running = False
        self._recognizer = None
        self._microphone = None

    def run(self) -> None:
        """Main loop: initialize recognizer and start background listening."""
        try:
            import speech_recognition as sr
        except ImportError:
            self.audio_error.emit("speech_recognition 库未安装")
            return

        self._recognizer = sr.Recognizer()
        # Adjust for ambient noise on start
        try:
            self._microphone = sr.Microphone()
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
        except OSError as e:
            self.audio_error.emit(f"麦克风初始化失败: {e}")
            return

        self._running = True
        self.recording_state.emit(True)

        # Use listen_in_background for non-blocking continuous recognition
        # This spawns its own thread internally; we run the callback on it.
        try:
            self._background = self._recognizer.listen_in_background(
                self._microphone,
                self._on_audio_callback,
                phrase_time_limit=5.0,  # max 5 sec per phrase
            )
        except Exception as e:
            self.audio_error.emit(f"启动语音监听失败: {e}")
            self._running = False
            return

        # Keep the thread alive while running; background listener runs its own thread
        while self._running:
            self.msleep(200)

        self.recording_state.emit(False)

    def _on_audio_callback(self, recognizer, audio) -> None:
        """Handle incoming audio: run ASR and emit results."""
        try:
            # Using Google Web Speech API (requires internet)
            text = recognizer.recognize_google(audio, language=self.language)
            if text:
                self.asr_result.emit(text)
        except self._get_sr_exception("UnknownValueError"):
            self.audio_error.emit("未能识别语音内容，请重试")
        except self._get_sr_exception("RequestError") as e:
            self.audio_error.emit(f"语音识别网络错误: {e}")
        except Exception as e:
            self.audio_error.emit(f"语音识别异常: {e}")

    @staticmethod
    def _get_sr_exception(name: str) -> type:
        """Resolve speech_recognition exception class by name."""
        try:
            import speech_recognition as sr
            return getattr(sr, name, Exception)
        except ImportError:
            return Exception

    def stop(self) -> None:
        """Signal the loop to exit and release resources."""
        self._running = False


class TTSPlayer(QObject):
    """Text-to-speech player — each utterance runs in a background thread."""

    tts_done = pyqtSignal()
    tts_error = pyqtSignal(str)

    def __init__(self, engine: str = "pyttsx3",
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine_type = engine
        self._tencent_appid: str | None = None
        self._tencent_secret_id: str | None = None
        self._tencent_secret_key: str | None = None
        self._busy = False

    @pyqtSlot(str)
    def speak(self, text: str) -> None:
        if not text:
            return
        if self._engine_type == "pyttsx3":
            self._busy = True
            t = threading.Thread(target=self._speak_pyttsx3, args=(text,), daemon=True)
            t.start()
        elif self._engine_type == "tencent":
            self._speak_tencent(text)
        else:
            self._speak_gtts(text)

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            for v in voices:
                name = v.name.lower()
                if "chinese" in name or "huihui" in name or "zh" in name:
                    engine.setProperty("voice", v.id)
                    break
            engine.setProperty("rate", 180)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            self.tts_error.emit(f"TTS 播报失败: {e}")
        finally:
            self._busy = False
            self.tts_done.emit()

    def set_tencent_credentials(self, appid: str, secret_id: str,
                                 secret_key: str) -> None:
        self._tencent_appid = appid
        self._tencent_secret_id = secret_id
        self._tencent_secret_key = secret_key
        self._engine_type = "tencent"

    def _speak_tencent(self, text: str) -> None:
        try:
            from tencentcloud.common import credential
            from tencentcloud.tts.v20190823 import tts_client, models
        except ImportError:
            self._speak_gtts(text)
        except Exception as e:
            self.tts_error.emit(f"腾讯云 TTS 失败: {e}")

    def _speak_gtts(self, text: str) -> None:
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="zh-CN", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)
            os.startfile(tmp_path)
        except Exception as e:
            self.tts_error.emit(f"gTTS 播报失败: {e}")

    def stop(self) -> None:
        self._busy = False
