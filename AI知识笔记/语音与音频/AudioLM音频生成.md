---
title: AudioLM音频生成
alias: AudioLM
tags: [音频生成, 语言模型, Google, 音频标记化, 语音合成]
category: 语音与音频
created: 2026-03-31
updated: 2026-03-31
author: AI知识库
description: Google开发的音频生成语言模型，通过音频标记化和自监督学习实现高质量语音和音频生成。
mastery: 75
rating: 9
related_concepts: [音频标记化, AudioLM, SpeechTokenizer, 音频LM, 自监督学习, Neural Codec]
difficulty: 高
read_time: 25
prerequisites: [音频基础, 深度学习, Transformer架构]
---

# AudioLM音频生成

## 一句话定义

AudioLM是Google开发的一个音频生成语言模型，通过将音频信号标记化后使用类似GPT的自回归方式生成音频，无需依赖文本转录即可实现高质量的语音和音乐合成。

## 详细说明

### 1. 核心架构

AudioLM基于Transformer解码器架构，包含三个主要组件：

- **音频标记化器（Audio Tokenizer）**：将原始波形压缩为离散 token 序列
- **音频LM（Audio LM）**：在token序列上进行自回归建模
- **音频解码器**：将生成的token序列还原为波形

### 2. 音频标记化（Audio Tokenization）

AudioLM使用基于RVQ（残差向量量化）的音频标记化方案：

| 层级 | 作用 | 压缩率 |
|------|------|--------|
| 语义层 | 捕获高层语义（音素、语调） | ~50x |
| 声学层 | 捕获细粒度音质 | ~40x |

### 3. 自回归生成

AudioLM采用两阶段生成策略：

1. **语义阶段**：生成语义token，捕获语言内容
2. **声学阶段**：基于语义token生成声学token，还原音质

### 4. SpeechTokenizer

SpeechTokenizer是AudioLM的改进版本，引入：

- 统一tokenization框架
- 更好的 speaker embedding 保持
- 多语言支持增强

## 代码示例

```python
# AudioLM 音频生成示例（伪代码）
import torchaudio

# 加载预训练模型
model = AudioLM.load_pretrained("google/audiolm-base")

# 生成条件：可以是音频片段或文本
condition = "a person speaking in a calm voice"

# 生成音频
output_tokens = model.generate(
    condition=condition,
    max_new_tokens=1200,
    temperature=0.8,
    top_k=50
)

# 解码为波形
audio = model.decode(output_tokens)
torchaudio.save("generated.wav", audio, 16000)
```

```python
# SpeechTokenizer 示例
from speechtokenizer import SpeechTokenizer

tokenizer = SpeechTokenizer.load("swabha/SpeechTokenizer")

# 编码音频
audio_tensor = load_audio("input.wav")
semantic_tokens, acoustic_tokens = tokenizer.encode(audio_tensor)

# 解码
reconstructed = tokenizer.decode(semantic_tokens, acoustic_tokens)
```

## 应用场景

| 场景 | 说明 |
|------|------|
| 语音合成 | 生成自然语音，无需TTS管道 |
| 音乐生成 | 基于参考风格创作新音乐 |
| 语音增强 | 条件生成改善音质 |
| 语音续写 | 延续给定音频片段的风格 |
| 多语言配音 | 保持原始音色进行多语言转换 |

## 相关概念

- **音频标记化（Audio Tokenization）**：将连续音频信号离散化的过程
- **残差向量量化（RVQ）**：多层向量量化技术，用于压缩音频token
- **自监督学习（SSL）**：无需人工标签的表征学习
- **Neural Codec**：基于神经网络的音频编解码器
- **声学模型 vs 语义模型**：声学层保留音质，语义层保留内容

## 延伸阅读

- [AudioLM: A Language Model for Audio](https://arxiv.org/abs/2209.03143) - Google原始论文
- [SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models](https://arxiv.org/abs/2308.16692)
- [AudioGen: Text-to-Audio Generation](https://arxiv.org/abs/2209.15352)
- [MusicLM: Music Generation with AI](https://google-research.github.io/seanet/musiclm/examples/)
