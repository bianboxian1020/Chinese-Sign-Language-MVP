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

# ---------------------------------------------------------------------------
# E2E: Data loading & model pipeline (no GUI required)
# ---------------------------------------------------------------------------
print("\n--- Data Loading E2E ---")

import torch
from model_config import ModelConfig
from dataset import SkeletonDataset
from inference import SignLanguageModel

config = ModelConfig()
config.num_classes = 3

# 1. 数据加载
ds = SkeletonDataset(
    data_root="data", seq_len=45, split="train",
    augment=False, feature_dim=126, label_file="data/labels.json",
)
print(f"Dataset: {len(ds)} samples, {ds.num_classes} classes")
assert len(ds) > 0, "Dataset is empty"
assert ds.num_classes == 3, f"Expected 3 classes, got {ds.num_classes}"

# 2. 样本 shape
sample, label = ds[0]
assert sample.shape == (45, 126), f"Shape mismatch: {sample.shape}"
print(f"Sample shape: {sample.shape}, label: {label} ({ds.labels[label]})")

# 3. 模型前向
model = SignLanguageModel(config)
x = sample.unsqueeze(0)  # (1, 45, 126)
logits, _ = model(x)
assert logits.shape == (1, 3), f"Logits shape mismatch: {logits.shape}"
print(f"Model forward: OK (logits shape: {logits.shape})")

# 4. 权重加载 (如果存在)
model_path = "assets/models/xinyu_model_best.pth"
if os.path.exists(model_path):
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded weights: {model_path}")
        print(f"  Epoch: {state.get('epoch', 'N/A')}")
        print(f"  Val Acc: {state.get('val_acc', 'N/A')}")
        if "label_map" in state:
            print(f"  Labels: {state['label_map']}")
    else:
        model.load_state_dict(state)
        print(f"Loaded weights (raw state_dict): {model_path}")
else:
    print(f"Weight file not found (skipped): {model_path}")

print("Data loading E2E: PASS")
