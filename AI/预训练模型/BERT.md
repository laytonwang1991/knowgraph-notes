---
title: BERT
alias: BERT,双向编码器,BERT模型
tags:
  - AI
  - 预训练模型
  - 自然语言处理
created: 2026-03-10
updated: 2026-03-28
author: KnowGraph
description: BERT是谷歌提出的双向预训练语言模型，通过遮蔽语言模型和下一句预测任务，在多项NLP基准上取得突破性成绩。
mastery: 0.6
rating: 4.7
related_concepts:
  - Transformer
  - GPT
  - 自注意力
  - 预训练
difficulty: 中等
read_time: 12分钟
---

# BERT（双向编码器表示）

## 概述

> **BERT**（Bidirectional Encoder Representations from Transformers）是由 Google 在 2018 年提出的预训练语言模型，革新了自然语言理解（NLU）任务的表现。

## 核心概念

### 定义

BERT 是一种**仅使用 Transformer 编码器**的预训练模型，通过**双向上下文建模**理解文本含义，在 11 项 NLP 基准上刷新纪录。

### 核心创新

| 创新点 | 说明 |
|--------|------|
| **双向编码** | 同时考虑左右上下文 |
| **遮蔽语言模型** | 预测被遮蔽的 token |
| **下一句预测** | 理解句子间关系 |
| **微调范式** | 预训练 + 下游任务微调 |

### 工作原理

1. **输入表示**：Token + Segment + Position Embedding
2. **双向上下文编码**：通过遮蔽自注意力双向建模
3. **预训练任务**：
   - **MLM（遮蔽语言模型）**：预测被遮蔽的 15% tokens
   - **NSP（下一句预测）**：判断是否为连续句子对
4. **下游微调**：添加任务输出层，训练全部参数

## 技术细节

### 遮蔽语言模型（MLM）

```
输入: 我 爱 机 器 [MASK] 学 习
预测: [MASK] -> "机器"
```

随机遮蔽 15% 的 token，让模型根据上下文预测被遮蔽的内容。

### 下一句预测（NSP）

```
句子A: 今天天气真好
句子B1: 我去公园散步 -> 正确（连续）
句子B2: 苹果是水果 -> 错误（非连续）
```

让模型判断两个句子是否为连续关系。

## 代码实现

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("你好，BERT！", return_tensors="pt")
outputs = model(**inputs)

# [batch_size, seq_len, hidden_dim]
embedding = outputs.last_hidden_state
```

## 模型规模

| 规模 | 层数 | 隐藏维度 | 注意力头数 | 参数 |
|------|------|----------|------------|------|
| BERT-base | 12 | 768 | 12 | 1.1亿 |
| BERT-large | 24 | 1024 | 16 | 3.4亿 |

## 应用场景

- **文本分类**：情感分析、垃圾邮件检测
- **命名实体识别**：人名、地名、机构名识别
- **问答系统**：阅读理解、机器阅读
- **句子对分类**：自然语言推理、语义匹配

## 相关概念

| 概念 | 关系 | 说明 |
|------|------|------|
| [[Transformer]] | 基础架构 | BERT 的核心组件 |
| [[GPT]] | 对比模型 | 仅用解码器的自回归模型 |
| [[自注意力]] | 核心技术 | 双向注意力的实现 |

## 延伸阅读

- [BERT 原始论文](https://arxiv.org/abs/1810.04805) - Google 2018
- [BERT 模型动物园](https://huggingface.co/bert) - Hugging Face 实现
- [中文 BERT](https://github.com/google-research/bert) - 预训练中文模型

## 学习记录

- [x] 理解 BERT 的双向编码原理
- [x] 掌握 MLM 和 NSP 预训练任务
- [ ] 能够进行下游任务微调
- [ ] 理解 BERT 与 GPT 的区别

---

*💡 提示：BERT 是 NLP 领域的里程碑，理解其预训练-微调范式对学习其他预训练模型很重要。*
