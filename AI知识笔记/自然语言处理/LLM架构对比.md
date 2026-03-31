---
title: LLM架构对比
alias: LLM Architecture Comparison
tags:
  - LLM
  - 架构
  - GPT
  - Claude
  - Gemini
  - Transformer
category: 自然语言处理
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 深入分析主流大语言模型的架构设计差异，涵盖GPT系、Claude系、Gemini系及开源模型的核心架构组件。
mastery: 90
rating: 95
related_concepts:
  - Transformer
  - Attention Mechanism
  - MoE
  - RLHF
  - Positional Encoding
difficulty: 高级
read_time: 45
prerequisites:
  - Transformer基础
  - 注意力机制
  - 深度学习基础
---

# LLM架构对比

## 一句话定义

大语言模型（LLM）架构对比是对主流模型的**编码器-解码器结构、注意力机制、归一化方式、位置编码**等核心组件进行系统性分析，以理解不同模型在性能、效率和能力上的差异来源。

---

## 核心公式

### Transformer 标准架构

$$
\text{Transformer}(x) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V + \text{FFN}(x)
$$

### GPT 系列的因果掩码

$$
M_{ij} =
\begin{cases}
0 & \text{if } i \leq j \text{ (可见)} \\
-\infty & \text{if } i > j \text{ (掩码)}
\end{cases}
$$

### Claude/Gemini 的并行注意力

$$
\text{Multi-Head Attention}_{\parallel} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

---

## 详细说明

### 1. GPT 系列（OpenAI）

**核心架构：仅解码器（Decoder-only）**

| 组件 | 配置 |
|------|------|
| 架构类型 | 单向因果注意力 |
| 层数 | 12 - 96 层 |
| 注意力头数 | 12 - 96 |
| 隐藏维度 | 768 - 12288 |
| 位置编码 | RoPE（旋转位置编码） |
| 归一化 | Layer Norm（后置归一化） |
| FFN | SwiGLU |

**关键特性：**
- **仅使用解码器块**：单向注意力确保自回归生成
- **RoPE 位置编码**：通过旋转矩阵实现相对位置编码，支持更长上下文
- **SwiGLU 激活**：提升非线性表达能力

**代表模型：**
- GPT-3 (175B)：96层，96注意力头，12288隐藏维度
- GPT-3.5：基于 GPT-3 的 RLHF 微调版本
- GPT-4：多模态支持，MoE 架构推测

---

### 2. Claude 系列（Anthropic）

**核心架构：仅解码器 + Pre-LN**

| 组件 | 配置 |
|------|------|
| 架构类型 | 单向因果注意力 |
| 层数 | 32 - 80 层 |
| 注意力头数 | 可变 |
| 隐藏维度 | 4096 - 8192 |
| 位置编码 | RoPE |
| 归一化 | Pre-LN（前置归一化） |
| FFN | MoE（混合专家） |

**关键特性：**
- **Pre-LN 前置归一化**：提升训练稳定性，理论基础来自《On Layer Normalization in the Pre-Trasnformer Architecture》
- **HAI（Human-Aligned Intelligence）**：RLHF + Constitutional AI 对齐
- **Long Context**：支持 200K token 上下文窗口

**代表模型：**
- Claude 3 Opus：200K 上下文，推理能力最强
- Claude 3 Sonnet：性价比最优
- Claude 3.5 Haiku：低延迟，适合实时应用

---

### 3. Gemini 系列（Google）

**核心架构：多模态原生 + MoE**

| 组件 | 配置 |
|------|------|
| 架构类型 | 解码器优先 + 编码器能力 |
| 层数 | 可变（MoE 分片） |
| 注意力机制 | 双向 + 单向自适应 |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| FFN | MoE（专家路由） |

**关键特性：**
- **原生多模态**：支持文本、图像、音频、视频统一处理
- **TPU 优化**：硬件协同设计，大量使用 TPU v5
- **多模态注意力**：图像 Token 与文本 Token 联合注意力

**代表模型：**
- Gemini Ultra：旗舰模型，多模态 SOTA
- Gemini Pro：平衡性能与成本
- Gemini Flash：快速响应，高并发

---

### 4. 开源模型架构

#### 4.1 LLaMA 系列（Meta）

| 组件 | 配置 |
|------|------|
| 架构类型 | 仅解码器 |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| FFN | SwiGLU |

**创新点：**
- 使用 RMSNorm 替代 Layer Norm
- SwiGLU 激活函数
- 开源推动生态繁荣

#### 4.2 Mistral / Mixtral

**关键创新：**
- **Mixtral 8x7B**：8 个专家的 MoE，路由选择 2 个专家
- **Sliding Window Attention**：稀疏注意力，降低计算复杂度至 $O(n \cdot w)$
- **Grouped Query Attention (GQA)**：减少 KV 缓存

#### 4.3 Qwen 系列（阿里）

- 支持超长上下文（128K - 1M）
- 多模态扩展（Qwen-VL）
- 对话优化

---

## 代码示例

### PyTorch 实现简化的 GPT 风格块

```python
import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    """旋转位置编码"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    """将 x 的后半部分旋转"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GPTBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SwiGLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN 风格
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

### MoE 实现示例

```python
class MoE(nn.Module):
    """混合专家层"""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.SwiGLU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        gate_logits = self.gate(x)
        weights = torch.softmax(gate_logits, dim=-1)

        # Top-K 选择
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 稀疏路由
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_weight = top_k_weights[mask][:, top_k_indices[mask] == i]
                output[mask] += expert(x[mask]) * expert_weight
        return output
```

---

## 应用场景

| 场景 | 推荐架构 | 理由 |
|------|----------|------|
| 代码生成 | GPT-4 / Claude 3.5 | 推理能力强，对齐良好 |
| 长文档分析 | Claude 3 / Gemini Ultra | 支持超长上下文 |
| 多模态任务 | Gemini 系列 | 原生多模态设计 |
| 边缘部署 | LLaMA / Qwen | 开源可定制，量化友好 |
| 低延迟应用 | Claude 3 Haiku / Mixtral | 响应速度快 |
| 开源研究 | LLaMA / Mistral | 透明可控 |

---

## 相关概念

- **Transformer**：LLM 的基础架构
- **RLHF**：人类反馈强化学习，用于模型对齐
- **Constitutional AI**：Anthropic 的对齐方法
- **MoE（混合专家）**：稀疏激活，降低计算成本
- **RoPE（旋转位置编码）**：相对位置编码，支持长上下文
- **GQA（分组查询注意力）**：减少 KV 缓存

---

## 延伸阅读

1. **GPT-3 论文**："Language Models are Few-Shot Learners" - 理解 GPT 架构的原始设计
2. **LLaMA 论文**："LLaMA: Open and Efficient Foundation Language Models" - 开源模型标杆
3. **Mixtral 论文**："Mixtral 8x7B: A Sparse Mixture of Experts" - MoE 实战解析
4. **RoPE 论文**："RoFormer: Enhanced Transformer with Rotary Position Embedding"
5. **Pre-LN 论文**："On Layer Normalization in the Pre-Transformer Architecture"
6. **Gemini 技术报告**：了解多模态原生架构设计
