---
title: Transformer
alias: Transformer Architecture
tags:
  - AI
  - 深度学习
  - NLP
  - 大语言模型
category: 深度学习
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: Transformer是一种基于注意力机制的神经网络架构，是GPT、BERT等大语言模型的基础。
mastery: 0
rating: 0
related_concepts:
  - 注意力机制
  - BERT
  - GPT
  - 位置编码
difficulty: 中等
read_time: 12分钟
prerequisites:
  - 神经网络基础
  - 注意力机制
---

# Transformer

## 一句话定义

> Transformer是一种完全基于注意力机制的神经网络架构，摒弃了传统的RNN/LSTM结构，实现了并行训练并刷新了多项NLP任务记录。

## 核心公式

### Positional Encoding

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

由于Transformer不包含RNN，需要通过位置编码注入序列位置信息。

## 详细说明

### 1. 模型架构

```
输入嵌入 → 位置编码 →
  └→ Encoder (N层)
       └→ Multi-Head Self-Attention
       └→ Feed-Forward Network
  └→ Decoder (N层)
       └→ Masked Multi-Head Self-Attention
       └→ Cross-Attention (Encoder-Decoder)
       └→ Feed-Forward Network
```

### 2. 核心组件

**Encoder**
- 6层相同结构堆叠
- 每层：Multi-Head Self-Attention + Feed-Forward
- 残差连接 + Layer Normalization

**Decoder**
- 6层相同结构堆叠
- 每层：Masked MHA + Cross-MHA + FFN
- Masked 确保只能看到之前的token

### 3. 训练配置

| 参数 | 值 |
|------|-----|
| 层数 | 6 |
| 隐藏层维度 | 512 |
| FFN维度 | 2048 |
| 注意力头数 | 8 |
| 词表大小 | 37000+ |
| 批次大小 | 4096 tokens |
| 学习率 | warmup 4000步，峰值 0.0003 |
| 正则化 | Label Smoothing 0.1, Dropout 0.1 |

### 4. 代码实现

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x, mask=None):
        # 自注意力 + 残差
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
```

### 5. 影响力与后续工作

**开创性贡献：**
- 完全并行化训练
- 突破了序列建模的顺序限制
- 为大模型时代奠定基础

**衍生模型：**
- BERT：双向编码器，用于理解任务
- GPT系列：单向解码器，用于生成任务
- T5：Encoder-Decoder统一架构

## 优缺点

| 优点 | 缺点 |
|------|------|
| 并行训练效率高 | 计算复杂度 O(n²) |
| 长距离依赖建模强 | 位置编码是固定长度 |
| 通用性强 | 需要大量训练数据 |

## 相关概念

- [[注意力机制]] — Transformer的核心机制
- [[BERT]] — 基于Transformer编码器的预训练模型
- [[GPT]] — 基于Transformer解码器的大语言模型
- [[位置编码]] — 为序列注入位置信息

## 延伸阅读

- [《Attention Is All You Need》](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Harvard NLP注释版](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
