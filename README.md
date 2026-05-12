# 心语速译 (Xinyu Suyi) - Chinese Sign Language Bidirectional Translation

Real-time bidirectional translation system for Chinese Sign Language (CSL).

## Features

- **Module A (Sign → Speech):** Camera capture → MediaPipe hand landmark extraction → gesture classification → TTS speech output
- **Module B (Speech → Sign):** Microphone input → ASR (Chinese) → keyword mapping → sign video playback

## Installation

### Quick Setup (Windows)

Double-click `setup.bat` to automatically create a virtual environment and install all dependencies.

### Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv --system-site-packages

# 2. Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the MediaPipe HandLandmarker model
# Windows (PowerShell):
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "assets/models/hand_landmarker.task"
```

## Usage

### Quick Launch (Windows)
Double-click `run.bat`

### Manual Launch
```bash
venv\Scripts\python.exe src\main_gui.py
```

## Project Structure

```
xinyu_suyi/
├── src/
│   ├── vision_engine.py    # Camera capture & MediaPipe landmark extraction
│   ├── audio_engine.py     # ASR recording & TTS playback
│   ├── inference.py        # Gesture classification (Euclidean distance matching)
│   └── main_gui.py         # PyQt6 GUI with multi-threading
├── assets/
│   ├── models/             # Pre-trained model weights & reference landmarks
│   └── sign_videos/        # Sign language demo videos for Module B
└── requirements.txt
```

## MVP Sign Set

| Sign | Meaning |
|------|---------|
| 你好 | Hello |
| 谢谢 | Thank you |
| 对不起 | Sorry |
| 我爱你 | I love you |
| 是 | Yes |
| 不 | No |
| 好 | Good |
| 再见 | Goodbye |
