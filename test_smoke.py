"""Smoke test for Xinyu Suyi — verifies GUI can initialize without a display."""
import sys
import os

# Add src/ directory to path (using absolute path from this script's location)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_ROOT)

from PyQt6.QtCore import QCoreApplication

app = QCoreApplication(sys.argv)

from main_gui import MainWindow

window = MainWindow()
print("MainWindow created: OK")
print("VisionEngine:", window._vision_engine is not None)
print("SignClassifier:", window._classifier is not None)
print("TTSPlayer:", window._tts_player is not None)
print("AudioRecorder:", window._audio_recorder is not None)
print("Signs in library:", window._classifier.sign_count)

window.close()
app.quit()
print("Smoke test: PASS")
