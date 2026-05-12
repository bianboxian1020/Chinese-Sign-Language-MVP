"""
test_hardware.py — 硬件可用性检测 (Hardware Smoke Test)

检测系统摄像头和麦克风是否可用，为后续开发提供环境确认。
"""
import sys


def test_camera() -> bool:
    """检测摄像头是否可用"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[FAIL] 摄像头未打开 — 请检查摄像头是否被其他程序占用")
            return False
        ret, frame = cap.read()
        if not ret:
            print("[FAIL] 摄像头无法读取帧")
            cap.release()
            return False
        h, w = frame.shape[:2]
        print(f"[ OK ] 摄像头正常 — 分辨率: {w}x{h}")
        cap.release()
        return True
    except Exception as e:
        print(f"[FAIL] 摄像头检测异常: {e}")
        return False


def test_microphone() -> bool:
    """检测麦克风是否可用"""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        default_input = p.get_default_input_device_info()
        print(f"[ OK ] 麦克风正常 — 设备: {default_input['name']}")
        p.terminate()
        return True
    except Exception as e:
        print(f"[FAIL] 麦克风检测异常: {e}")
        return False


def test_mediapipe() -> bool:
    """检测 MediaPipe 是否可用"""
    try:
        import mediapipe as mp
        print(f"[ OK ] MediaPipe 正常 — 版本: {mp.__version__}")
        return True
    except Exception as e:
        print(f"[FAIL] MediaPipe 异常: {e}")
        return False


def test_pyttsx3() -> bool:
    """检测 TTS 引擎是否可用"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f"[ OK ] pyttsx3 正常 — 可用语音数: {len(voices)}")
        for v in voices:
            print(f"       - {v.name} ({v.id})")
        return True
    except Exception as e:
        print(f"[FAIL] pyttsx3 异常: {e}")
        return False


def main():
    print("=" * 50)
    print("  心语速译 — 硬件可用性检测")
    print("=" * 50)
    print()

    results = {
        "摄像头": test_camera(),
        "麦克风": test_microphone(),
        "MediaPipe": test_mediapipe(),
        "TTS 引擎": test_pyttsx3(),
    }

    print()
    print("=" * 50)
    print("  检测结果汇总")
    print("=" * 50)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n所有硬件检测通过，可以开始开发！")
    else:
        print("\n部分硬件检测未通过，请根据上述提示排查问题。")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
