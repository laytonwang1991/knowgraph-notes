---
title: HuggingFace
alias: HuggingFace
tags:
  - AI
  - NLP
  - 开源社区
  - 预训练模型
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: HuggingFace是AI领域的开源社区和平台，提供预训练模型库和工具，构建了全球最大的AI模型共享生态。
mastery: 0
rating: 0
related_concepts:
  - 大语言模型
  - PyTorch
  - Transformers
  - NLP
difficulty: 入门
read_time: 8分钟
prerequisites: []
---

# HuggingFace

## 一句话定义

> HuggingFace是一个AI领域的开源社区和平台，提供预训练模型库和工具，构建了全球最大的AI模型共享生态。

## 基本信息

| 字段 | 内容 |
|------|------|
| 创始组织 | HuggingFace Inc. |
| 首次发布 | 2016年 |
| GitHub星标 | 65,000+（Transformers） |
| 贡献者 | 1,800+ |
| 模型数量 | 800,000+ |
| 许可证 | Apache-2.0 |

## 详细说明

### 1. 核心特性

**模型中心化：**
- 汇聚全球开发者贡献的预训练模型
- 支持模型发现、下载、托管
- 支持推理Endpoints

** Transformers库：**
```python
from transformers import pipeline

# 零样本分类
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Python",
    candidate_labels=["education", "business", "tech"]
)
```

### 2. 核心产品

| 产品 | 功能 |
|------|------|
| Transformers | 预训练模型库 |
| Diffusers | 扩散模型库 |
| Datasets | 数据集库 |
| Tokenizers | 分词器 |
| PEFT | 参数高效微调 |
| Accelerate | 分布式训练加速 |
| Hub | 模型托管平台 |
| Spaces | Demo托管平台 |

### 3. 代码示例

```python
from transformers import AutoTokenizer, AutoModel

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 编码输入
inputs = tokenizer("Hello, world!", return_tensors="pt")

# 前向传播
outputs = model(**inputs)
```

### 4. 生态版图

```
HuggingFace生态
├── 模型库
│   ├── Transformers — 通用模型库
│   ├── Diffusers — 图像生成模型
│   └── Audio — 音频模型
├── 数据集
│   ├── Datasets — 10000+数据集
│   └── Evaluate — 评估指标
├── 工具库
│   ├── Tokenizers — 快速分词
│   ├── PEFT — LoRA等微调技术
│   └── Accelerate — 分布式训练
├── 平台服务
│   ├── Hub — 模型托管
│   ├── Spaces — Gradio/Demo托管
│   └── Inference Endpoints — 托管推理
└── 社区
    ├── 模型评论与下载统计
    └── 模型卡片文档
```

### 5. 知名模型系列

- **NLP：** BERT, GPT-2, T5, RoBERTa, Llama
- **视觉：** ViT, Stable Diffusion, CLIP
- **音频：** Whisper, SpeechRecognition
- **多模态：** GPT-4V, LLaVA, BLIP

## 相关概念

- [[大语言模型]] — HuggingFace托管大量LLM模型
- [[自然语言处理]] — NLP是HuggingFace的核心应用领域
- [[PyTorch]] — Transformers后端主要支持PyTorch
- [[Transformers]] — 注意力机制是模型核心架构
- [[扩散模型]] — Diffusers库支持图像生成

## 延伸阅读

- [HuggingFace官网](https://huggingface.co/)
- [Transformers文档](https://huggingface.co/docs/transformers/)
- [HuggingFace GitHub](https://github.com/huggingface)
- [HuggingFace Hub](https://huggingface.co/models)
