# 心语速译 (Xinyu Suyi) — 中国手语双向实时翻译

基于 CNN-BiLSTM-Attention 的中国手语（CSL）实时双向翻译系统。

## 功能

- **手语 → 语音**：摄像头采集 → MediaPipe 手部关键点提取 → CNN-BiLSTM-Attention 手势分类 → TTS 语音播报
- **语音 → 手语**：麦克风输入 → 中文语音识别 → 关键词匹配 → 手语视频演示

## 技术架构

```
摄像头 → 高斯降噪 → MediaPipe Hands (21关键点×2手) → 归一化[-1,1]
→ 滑动窗口 (20帧) → CNN(空间) → BiLSTM(时序) → Attention(聚焦) → 贝叶斯分类器
→ TTS 播报
```

## 快速开始 (Windows)

### 一键安装
双击 `setup.bat`，自动创建虚拟环境并安装所有依赖。

### 一键启动
双击 `run.bat` 启动 GUI。

## 使用流程

1. **采集数据**：在 GUI 右侧"数据采集"面板输入手势词，点击录制，对着摄像头做手语动作
2. **训练模型**：`python train.py --data_root data --epochs 80`
3. **实时识别**：GUI 启动后自动加载训练好的模型，开摄像头即可识别

## 项目结构

```
xinyu_suyi/
├── src/
│   ├── main_gui.py          # PyQt6 主界面 (深色主题)
│   ├── vision_engine.py      # 摄像头采集 + MediaPipe 手部关键点提取
│   ├── inference.py          # CNN-BiLSTM-Attention 推理模型
│   ├── audio_engine.py       # 语音识别 + TTS 播报
│   ├── data_collector.py     # 训练数据采集工具
│   ├── dataset.py            # 骨架数据集加载器 + GAN 数据增强
│   └── model_config.py       # 模型超参数配置
├── train.py                  # 训练脚本
├── assets/models/            # 预训练模型权重
├── data/                     # 采集的训练数据 (.npy)
└── requirements.txt
```

## MVP 手势集 (国家通用手语标准)

| 手势词 | 动作要点 |
|--------|----------|
| 你好 | 食指指对方 → 握拳竖拇指 |
| 谢谢 | 竖拇指弯曲两下 |
| 对不起 | 五指并拢贴额 → 小指在胸前点 |
| 我爱你 | 食指指自己 → 右手抚摸左手拇指 → 指对方 |
| 是 | 食中指相搭向下点 |
| 不 | 五指并拢掌心向外左右摆 |
| 好 | 握拳竖拇指 |
| 再见 | 手掌朝外左右摆 |

## 当前模型

- 已训练：3 词（好、谢谢、你好）
- 验证精度：100%
- 参数量：416 万
- 输入：126 维手部关键点 (21×2×3)
