---
title: Transformer改进
alias: Transformer Improvements
tags:
  - Transformer
  - Pre-LN
  - RMSNorm
  - GLU
  - RoPE
  - 架构演进
category: 自然语言处理
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 系统梳理Transformer架构从原始论文到现代LLM的核心改进，包括Pre-LN vs Post-LN、RMSNorm、GLU激活和旋转位置编码等关键技术。
mastery: 88
rating: 94
related_concepts:
  - Transformer
  - Layer Normalization
  - Feed-Forward Network
  - Positional Encoding
  - Attention Mechanism
difficulty: 高级
read_time: 38
prerequisites:
  - Transformer 基础
  - 归一化技术
  - 位置编码
---

# Transformer改进

## 一句话定义

Transformer 改进是围绕**归一化位置、激活函数、位置编码**等核心组件的系统性演进，通过 Pre-LN、RMSNorm、GLU、RoPE 等技术，解决原始架构的训练不稳定、推理效率低、长上下文建模能力弱等问题。

---

## 核心公式

### 原始 Post-LN Transformer

$$
x_{l+1} = x_l + \text{SubLayer}\left(\text{LayerNorm}(x_l)\right)
$$

**问题：子层输出在残差分支上，需要完美的信号路径**

### Pre-LN（前置归一化）

$$
x_{l+1} = x_l + \text{SubLayer}\left(\text{LayerNorm}(x_l)\right)
$$
$$
h_l = \text{LayerNorm}(x_l)
$$

**改进：主路径直接是 LayerNorm 输出，梯度更稳定**

### RMSNorm（均方根归一化）

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}
$$

**比 Layer Norm 少计算均值，只用 RMS**

### SwiGLU 激活

$$
\text{SwiGLU}(x) = \text{Swish}(x) \odot \sigma(x) = \frac{x}{1 + e^{-x}} \odot \sigma(x)
$$

### RoPE（旋转位置编码）

$$
f_q(x_m, m) = W_q \cdot R(m), \quad R(m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

---

## 详细说明

### 1. Pre-LN vs Post-LN

#### 问题分析

原始 Transformer（Post-LN）的结构：

```
x
  │
  ├──→ LayerNorm ──→ SubLayer ──→ + ──→ x_{l+1}
  │
x ──────────────────────────────────────→
```

**训练不稳定的原因：**
- 初始阶段，SubLayer 输出接近 0，主路径信号衰减
- 残差分支和主分支的信号强度不匹配
- 深层网络（>6层）梯度消失

#### Pre-LN 解决方案

```
x ──→ LayerNorm ──→ SubLayer ──→ + ──→ x_{l+1}
                          │
x ────────────────────────┘
```

**核心改进：**
- LayerNorm 在残差计算之前应用
- 主路径直接是 $\text{LN}(x_l)$，始终保持稳定方差
- 子层梯度直接流向输入，不再经过 LayerNorm 均值减法

#### 理论证明

Xiangting Li 等人（2018）证明：
- Pre-LN 的梯度范数在初始化时约为常数
- Post-LN 的梯度范数随层深指数衰减

#### 对比

| 特性 | Post-LN | Pre-LN |
|------|---------|--------|
| LayerNorm 位置 | 残差求和之后 | 残差求和之前 |
| 训练稳定性 | 不稳定 | 稳定 |
| 收敛速度 | 慢 | 快 |
| 理论支持 | 无 | 有 |
| 使用场景 | 早期 BERT | 现代 LLM（GPT、LLaMA） |

---

### 2. RMSNorm（均方根归一化）

#### Layer Norm 的冗余

Layer Norm 计算：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta, \quad
\mu = \frac{1}{d}\sum x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum x_i^2}
$$

**发现：** 均值 $\mu$ 和平移参数 $\beta$ 对模型性能影响极小

#### RMSNorm 定义

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}
$$

**节省计算：**
- 减少 $2d$ 次减法和 $d$ 次加法
- 无需计算 $\beta$ 参数

#### 效果

| 指标 | Layer Norm | RMS Norm |
|------|------------|----------|
| 参数量 | $2d$ | $d$ |
| 计算量 | $O(3d)$ | $O(2d)$ |
| 效果 | 基准 | 基本无差异 |
| 代表模型 | GPT-2 | LLaMA, Mistral |

---

### 3. GLU（Gated Linear Unit）

#### FFN 的问题

标准 FFN：

$$
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

**问题：单向激活，信息选择性差**

#### GLU 原理

引入门控机制：

$$
\text{GLU}(x) = \sigma(xW_1 + b_1) \odot (xW_2 + b_2)
$$

**关键思想：** 用 Sigmoid 门控决定哪些信息通过

#### SwiGLU 变体

Dauphin 等人（2017）提出用 Swish 替代 Sigmoid：

$$
\text{SwiGLU}(x) = \text{Swish}(xW_1 + b_1) \odot (xW_2 + b_2)
$$

其中：
$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

#### 代码实现

```python
class SwiGLU(nn.Module):
    """SwiGLU 激活函数"""
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_model, d_ffn)
        self.w3 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# 等价于：
# FFN(x) = SwiGLU(x) = [Silu(W1x) ⊙ (W2x)] W3
```

#### 参数量对比

| FFN 类型 | 参数量 | 激活函数 |
|----------|--------|----------|
| 标准 FFN | $2 \times d \times d_{ffn}$ | SiLU/GELU |
| SwiGLU | $3 \times d \times d_{ffn}$ | SiLU |

**注：SwiGLU 的隐藏维度通常为 $d_{ffn} \times \frac{2}{3}$ 以保持参数量相近**

---

### 4. 旋转位置编码（RoPE）

#### 绝对位置编码的问题

原始 Transformer 使用可学习的绝对位置编码：

$$
E_{pos} = \text{Embedding}(pos)
$$

**问题：**
- 无法泛化到训练时未见过的序列长度
- 相对位置信息需要额外建模

#### RoPE 核心思想

将位置信息**旋转**到 Query 和 Key 向量中：

$$
q_m = W_q \cdot R(m), \quad k_n = W_k \cdot R(n)
$$

其中 $R(m)$ 是旋转矩阵：

$$
R(m) = \begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) & 0 & 0 & \cdots \\
\sin(m\theta) & \cos(m\theta) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta) & -\sin(m\theta) & \cdots \\
0 & 0 & \sin(m\theta) & \cos(m\theta) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$

#### 内积性质（关键）

$$
\langle q_m, k_n \rangle = \langle R(m)q, R(n)k \rangle = \langle q, k \rangle_{(n-m)}
$$

**结论：旋转后的内积只依赖于相对位置 $(n-m)$！**

#### 远程衰减特性

$$
\text{AttentionScore}(m, n) \propto \cos\left((m-n)\theta\right)
$$

当 $|m-n|$ 增大时，余弦值衰减 → 天然建模相对距离

#### 与 RoPE 配合的注意力

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """RoPE 应用"""
    # q, k: (batch, heads, seq, head_dim)
    # cos, sin: (seq, head_dim // 2)
    q0, q1 = q[..., ::2], q[..., 1::2]
    k0, k1 = k[..., ::2], k[..., 1::2]
    q_rot = torch.cat([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1)
    k_rot = torch.cat([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1)
    return q_rot, k_rot
```

#### RoPE vs 其他位置编码

| 特性 | 绝对位置编码 | 相对位置编码 (T5) | RoPE |
|------|-------------|------------------|------|
| 位置信息载体 | Embedding | 偏置项 | Query/Key 旋转 |
| 相对位置 | 需额外建模 | 原生支持 | 原生支持 |
| 外推能力 | 差 | 中等 | 较好 |
| 计算开销 | 低 | 中等 | 中等 |
| 代表模型 | BERT | T5 | LLaMA, GPT-4 |

---

### 5. 其他重要改进

#### 5.1 字符串注意力（Flash Attention）

见《注意力机制变体》笔记

#### 5.2 混合专家（MoE）

$$
\text{MoE}(x) = \sum_{i=1}^{E} G(x)_i \cdot E_i(x), \quad G(x) = \text{TopK}(\text{Gate}(x))
$$

稀疏激活，降低计算量

#### 5.3 烟雾注意力（Grouped Query Attention）

见《注意力机制变体》笔记

---

## 现代 LLM 架构总结

```
Transformer Layer (Modern):
┌─────────────────────────────────────┐
│ Input: x                            │
│                                     │
│ 1. RMSNorm(x)                      │  ← Pre-LN + RMSNorm
│                                     │
│ 2. Q, K, V = Linear(x)              │
│    Apply RoPE to Q, K              │  ← 旋转位置编码
│                                     │
│ 3. Attention(Q,K,V)                │  ← Flash/GQA/Sparse
│    + Residual Connection           │
│                                     │
│ 4. RMSNorm(x')                     │
│                                     │
│ 5. SwiGLU(x')                       │  ← GLU 激活
│    + Residual Connection           │
│                                     │
│ Output: x''                         │
└─────────────────────────────────────┘
```

---

## 代码示例

### 完整的 Modern Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPosEmb(nn.Module):
    """旋转位置编码"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ModernTransformerBlock(nn.Module):
    """现代 Transformer 块：Pre-LN + RMSNorm + RoPE + SwiGLU"""
    def __init__(self, d_model, nhead, d_ffn=None, dropout=0.1):
        super().__init__()
        d_ffn = d_ffn or int(d_model * 8 / 3)
        self.head_dim = d_model // nhead

        # Pre-LN（RMSNorm）
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        # 多头注意力 + RoPE
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rope = RotaryPosEmb(self.head_dim)

        # SwiGLU FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.SiLU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, x, mask=None):
        # Pre-LN
        x_norm = self.norm1(x)

        # QKV 投影
        qkv = nn.functional.linear(x_norm, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # 应用 RoPE
        cos, sin = self.rope(q.size(1))
        q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(0), sin.unsqueeze(0))

        # 注意力
        attn_out, _ = self.attn(q, k, v, attn_mask=mask)
        x = x + attn_out  # 残差连接

        # Pre-LN + SwiGLU
        x = x + self.ffn(self.norm2(x))  # 残差连接

        return x
```

---

## 应用场景

| 改进 | 主要收益 | 适用场景 |
|------|----------|----------|
| Pre-LN | 训练稳定性 | 所有现代 LLM |
| RMSNorm | 效率提升 ~15% | 资源受限场景 |
| SwiGLU | 非线性表达能力 | 高质量生成 |
| RoPE | 长上下文 + 外推 | 长序列任务 |
| Flash Attention | 显存降低 ~60% | 长序列 / 低显存 |
| GQA | KV 缓存减少 | 推理优化 |

---

## 相关概念

- **残差连接（Residual Connection）**：原始 Transformer 的核心
- **Layer Norm vs Batch Norm**：LN 在 NLP 中更稳定
- **SiLU/Swish 激活**：Google 提出的自门控激活
- **ALiBi（Attention with Linear Biases）**：另一种位置编码外推方案

---

## 延伸阅读

1. **Pre-LN 论文**："On Layer Normalization in the Pre-Transformer Architecture"（2021）
2. **RMSNorm 论文**："Root Mean Square Layer Normalization"（Baidu, 2019）
3. **SwiGLU 论文**："GLU Variants Improve Transformer"（Noam Shazeer, 2020）
4. **RoPE 论文**："RoFormer: Enhanced Transformer with Rotary Position Embedding"
5. **GPT-3 架构博客**："GPT-3's architecture" by Jay Alammar
6. **LLaMA 论文**："LLaMA: Open and Efficient Foundation Language Models"
