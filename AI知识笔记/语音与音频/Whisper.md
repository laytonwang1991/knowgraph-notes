---
title: Whisper
category: 语音与音频
tags: [Whisper, 语音识别, OpenAI, 多语言]
date: 2026-03-31
created_by: Claude
---

# Whisper

## 一句话定义

Whisper 是 OpenAI 开发的大规模多语言语音识别模型，通过海量互联网音频数据进行训练，具备强大的泛化能力。

## 核心公式/技术要点

### Whisper 架构
```
输入: 30秒音频片段 → Mel Spectrogram (80 channels)
      ↓
Encoder (Transformer Encoder)
      ↓
Decoder (Transformer Decoder)
      ↓
输出: 文本序列 + 时间戳 + 语言预测
```

### Whisper 模型规模

| 模型 | 参数量 | 内存需求 | 相对速度 |
|------|--------|----------|----------|
| Tiny | 39M | ~1GB | 32x |
| Base | 74M | ~1GB | 16x |
| Small | 244M | ~2GB | 6x |
| Medium | 769M | ~5GB | 2x |
| Large | 1550M | ~10GB | 1x |

### 多任务输出格式
$$P(\text{text}, \text{timestamps} | \text{audio}) = \prod_{t} P(\text{token}_t | \text{audio}, \text{token}_{<t})$$

## 详细说明

### 1. 核心特性

- **多语言支持**: 100+ 语言识别
- **多任务学习**: 语音识别 + 语音翻译 + 语言识别 + 时间戳
- **强泛化能力**: 无需微调即可直接使用
- **开源**: Apache 2.0 许可证

### 2. 训练数据

- 基于互联网大规模音频数据
- 680,000 小时多语言语音
- 包含嘈杂和低质量音频
- 数据清洗和过滤

### 3. 可用模型版本

#### Whisper v2/v3
- 改进的编码器结构
- 更好的多语言性能
- 更准确的时间戳

#### Whisper Turbo
- 最新一代模型
- 速度和精度平衡
- FFmpeg 集成

#### Whisper AI API
- OpenAI API 服务
- 付费使用
- 集成便捷

### 4. 使用场景

#### 本地部署
```python
# 使用 Hugging Face Transformers
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# 音频输入
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

#### 命令行使用
```bash
# 安装
pip install openai-whisper

# 基础使用
whisper audio.mp3 --model medium --language Chinese

# 多语言自动检测
whisper audio.mp3 --model large --task translate
```

### 5. Whisper.cpp

- C/C++ 实现
- 量化模型 (INT8, INT4)
- CPU 推理优化
- 移动端部署

### 6. 优缺点

| 优点 | 缺点 |
|------|------|
| 开源可商用 | 幻觉问题 (hallucination) |
| 多语言支持 | 长音频需要分割 |
| 无需微调 | 英文以外精度下降 |
| 强泛化能力 | 实时性有限 |

### 7. 性能对比

| 数据集 | Whisper Large |人类水平|
|--------|--------------|--------|
| LibriSpeech (clean) | 2.5% WER | 1.4% |
| TED-LIUM | 3.5% WER | - |
| Multilingual LJ | 4.8% WER | - |

## 相关概念

- [[语音识别|ASR]] - Whisper 属于语音识别范畴
- [[语音合成|TTS]] - 语音合成技术
- [[深度学习]] - Transformer 架构基础
- [[自然语言处理]] - 文本处理相关

## 延伸阅读

- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- [Whisper GitHub Repository](https://github.com/openai/whisper)
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
