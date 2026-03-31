---
title: RWKV模型
alias: Receptance Weighted Key Value, RWKV, 循环Transformer变体
tags:
  - RWKV
  - 循环Transformer
  - Transformer替代
  - 时间混合
  - 通道混合
  - 开源LLM
category: 模型架构创新
created: 2026-03-31
updated: 2026-03-31
author: AI知识笔记
description: RWKV是一种结合了Transformer强大表达能力和RNN高效推理特性的新型架构，通过时间混合和通道混合机制实现O(1)推理复杂度，同时保持对长上下文的建模能力
mastery: 0.6
rating: 9
related_concepts:
  - Transformer
  - RetNet
  - Mamba
  - 线性注意力
  - RNN
  - LSTM
difficulty: 高难度
read_time: 25分钟
prerequisites:
  - Transformer原理
  - 注意力机制
  - RNN/LSTM基础
  - 深度学习优化
---

# RWKV模型

## 一句话定义

RWKV（Receptance Weighted Key Value）是一种**循环Transformer变体**，通过时间混合（Time Mixing）和通道混合（Channel Mixing）机制实现Transformer级别的表达能力与RNN级别的O(1)推理效率，被广泛用于开源大语言模型。

## 核心公式

### RWKV名字由来

RWKV 分别代表：
- **R**：Receptance（接受度）- 类似于注意力分数
- **W**：Weight（权重）- 可学习的权重矩阵
- **K**：Key（键）
- **V**：Value（值）

### 时间混合（Time Mixing）核心公式

$$
r_t = \sigma(W_r \cdot (ReLU(w_k) \odot x_t + (1-w_k) \odot x_{t-1}))
$$

$$
k_t = w_u \odot x_t + (1-w_u) \odot x_{t-1}
$$

$$
v_t = W_v \cdot k_t
$$

$$
o_t = W_o \cdot \frac{\sum_{i=1}^{t} e^{-(t-i)w} r_i \odot k_i \odot v_i}{\sum_{i=1}^{t} e^{-(t-i)w}}
$$

简化表示：

$$
o_t = W_o \cdot \frac{\mathbf{r}_t^T \cdot (\mathbf{k}_t \odot \mathbf{v}_t)}{\sum_{i=1}^{t} e^{-(t-i)w} \mathbf{r}_i^T \mathbf{k}_i}
$$

### 通道混合（Channel Mixing）核心公式

$$
r'_t = \sigma(W_r' \cdot x_t)
$$

$$
k'_t = \tanh(W_k' \cdot x_t)
$$

$$
o'_t = \sigma(W_o' \cdot (r'_t \odot k'_t))
$$

### 循环形式（用于推理）

当处理第 $t$ 个token时，使用前一个状态的衰减：

$$
state_t = e^{-w} \cdot state_{t-1} + r_t \odot v_t
$$

$$
o_t = W_o \cdot \frac{state_t}{\sum_{i=1}^{t} e^{-(t-i)w}}
$$

## 详细说明

### 1. 架构设计哲学

RWKV的核心思想是**将Transformer的并行训练与RNN的高效推理结合**：

```
┌──────────────────────────────────────────────────────┐
│                    RWKV 架构                         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  输入层                                              │
│    ↓                                                 │
│  ┌─────────────────┐    ┌─────────────────┐         │
│  │   时间混合       │    │   通道混合       │         │
│  │ Time Mixing     │    │ Channel Mixing  │         │
│  │ (SSM-like)      │    │ (FFN-like)      │         │
│  └────────┬────────┘    └────────┬────────┘         │
│           ↓                      ↓                  │
│  ┌─────────────────────────────────────────┐         │
│  │        循环状态传递（推理时）             │         │
│  │   state_t = decay * state_{t-1} + r_t*v_t│         │
│  └─────────────────────────────────────────┘         │
│                    ↓                                  │
│                  输出层                               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 2. 时间混合详解

时间混合是RWKV的核心创新，类似于SSM的选择性扫描：

```python
def time_mixing(x_t, x_t_minus_1, state, w, u, r_key, k_key, v_key, o_key):
    """
    x_t: 当前输入
    x_t_minus_1: 前一个token（token-shift）
    state: 过去累积状态
    w, u: 衰减参数
    r_key, k_key, v_key, o_key: 投影矩阵
    """
    # Token-shift: 混合当前和前一个token
    k = w * x_t + (1 - w) * x_t_minus_1

    # 接受度计算（包含非线性）
    r = sigmoid(W_r @ (relu(w) * x_t + (1 - w) * x_t_minus_1))

    # Key和Value
    k = W_k @ k
    v = W_v @ x_t

    # 衰减的指数项
    decay = torch.exp(-torch.exp(w))

    # 更新状态
    new_state = decay * state + r * v

    # 输出
    o = sigmoid(W_o @ (r * k)) * new_state

    return o, new_state
```

### 3. 通道混合详解

通道混合类似于FFN，但使用Receptance机制：

```
通道混合数据流：

x_t → [W_r'] → r' = sigmoid(·)
      ↓
x_t → [W_k'] → k' = tanh(·)
      ↓
    r' ⊙ k' → [W_o'] → sigmoid(·) → output

特点：
- 非线性：sigmoid + tanh
- 门控：元素级乘法
- 类似GRU的门控机制
```

### 4. 三种计算模式

| 模式 | 描述 | 复杂度 | 用途 |
|------|------|--------|------|
| **并行模式** | 训练时一次性计算全部token | O(L²) | GPU训练 |
| **递归模式** | 逐token递推 | O(1) | 推理部署 |
| **分块模式** | 分块并行+块间递归 | O(L) | 长序列 |

### 5. 与其他架构对比

| 特性 | Transformer | RWKV | Mamba | RetNet |
|------|-------------|------|-------|--------|
| 训练 | O(L²) | O(L²) | O(L log L) | O(L²) |
| 推理 | O(L) | **O(1)** | O(1) | O(1) |
| 内存 | O(L²) | **O(L)** | O(L) | O(L) |
| 位置编码 | 需要 | 不需要 | 不需要 | 不需要 |
| 长上下文 | 受限 | 支持 | 支持 | 支持 |
| 开源生态 | 丰富 | **成熟** | 发展中 | 发展中 |

### 6. 数值稳定性和初始化

RWKV对初始化和数值稳定性有特殊要求：

```python
# 关键初始化策略

# 衰减参数w的初始化（确保0.99左右的衰减率）
with torch.no_grad():
    w.data.fill_(torch.log(torch.tensor(1.0 / 64)))  # ~0.985衰减率

# 通道混合的初始化（保持较小的初始值）
for name, param in model.named_parameters():
    if 'channel_mixing' in name:
        nn.init.uniform_(param, -0.01, 0.01)
```

### 7. 代码架构总览

```
RWKV Block
├── LayerNorm
├── Time Mixing
│   ├── token_shift: x ~mix~> x_shift
│   ├── receptance: r = sigmoid(W_r @ x_shift)
│   ├── key: k = W_k @ x_shift
│   ├── value: v = W_v @ x
│   ├── state update: state = decay*state + r*v
│   └── output: o = W_o @ (r*k) * state
├── LayerNorm
└── Channel Mixing
    ├── r' = sigmoid(W_r' @ x)
    ├── k' = tanh(W_k' @ x)
    └── output: W_o' @ (r' ⊙ k')
```

## 应用场景

### 1. 开源大语言模型
- RWKV-4系列（1B-14B参数）
- 中文语言模型
- 多语言支持

### 2. 长文本处理
- 文档摘要
- 代码生成
- 多轮对话

### 3. 边缘部署
- 个人电脑
- 嵌入式设备
- 移动端应用

### 4. 实时交互
- 聊天机器人
- 语音助手
- 游戏NPC

## 代码示例：RWKV Block实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RWKVBlock(nn.Module):
    """RWKV Block实现"""

    def __init__(self, d_model, num_heads=None, ff_dim=None):
        super().__init__()
        self.d_model = d_model
        self.head_size = d_model  # RWKV使用单一头

        # 时间混合参数
        self.time_mix = nn.Parameter(torch.ones(d_model))  # 用于token-shift
        self.time_decay = nn.Parameter(torch.zeros(d_model))  # 衰减系数
        self.time_first = nn.Parameter(torch.zeros(d_model))  # 初始接受度

        # 投影矩阵
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # 通道混合
        self.channel_mixing_r = nn.Linear(d_model, ff_dim or 4*d_model, bias=False)
        self.channel_mixing_k = nn.Linear(d_model, ff_dim or 4*d_model, bias=False)
        self.channel_mixing_o = nn.Linear(ff_dim or 4*d_model, d_model, bias=False)

        # LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # 初始化
        self.register_buffer('zero', torch.zeros(1, 1, 1))
        self._init_weights()

    def _init_weights(self):
        # 衰减参数初始化
        with torch.no_grad():
            self.time_decay.data.fill_(torch.log(torch.tensor(1.0 / 64)))

    def time_mixing_forward(self, x, last_state=None):
        """
        x: (batch, seq_len, d_model)
        last_state: 上一时刻的状态
        """
        B, L, D = x.shape

        # Token-shift: 混合当前和前一个token
        # x_shift: (batch, seq_len, d_model)
        x_shift = torch.cat([x[:, -1:, :], x[:, :-1, :]], dim=1) if L > 1 else x
        x_mix = self.time_mix * x + (1 - self.time_mix) * x_shift

        # 计算r, k, v
        r = self.receptance(x_mix)  # 接受度
        k = self.key(x_mix)         # Key
        v = self.value(x)           # Value

        # 归一化
        k = F.relu(k)  # 使用ReLU增加选择性
        r = torch.sigmoid(r)

        # 计算衰减
        decay = torch.exp(-torch.exp(self.time_decay.float()))

        # 如果是第一条序列，初始化state
        if last_state is None:
            last_state = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # 递归更新状态
        # state_new = decay * state + r * v
        state_new = decay.unsqueeze(0) * last_state + r * v

        # 计算输出
        # RWKV公式: o = r * k * state / sum
        wkv = r * k * state_new
        output = self.output(wkv)

        return output, state_new

    def channel_mixing_forward(self, x):
        """通道混合前向传播"""
        r = torch.sigmoid(self.channel_mixing_r(x))
        k = torch.tanh(self.channel_mixing_k(x))
        return self.channel_mixing_o(r * k)

    def forward(self, x, last_state=None):
        """
        x: (batch, seq_len, d_model)
        last_state: 用于递归的隐藏状态
        """
        # 时间混合 + 残差
        x_norm = self.ln1(x)
        tm_out, new_state = self.time_mixing_forward(x_norm, last_state)
        x = x + tm_out

        # 通道混合 + 残差
        x_norm = self.ln2(x)
        cm_out = self.channel_mixing_forward(x_norm)
        x = x + cm_out

        return x, new_state


# 使用示例
model = RWKVBlock(d_model=512, ff_dim=2048)
x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)

# 并行模式（训练）
y, _ = model(x)
print(f"并行模式 - 输入: {x.shape}, 输出: {y.shape}")

# 递归模式（推理）
state = None
for i in range(100):
    x_i = x[:, i:i+1, :]
    y_i, state = model(x_i, state)
print(f"递归模式 - 最终输出: {y_i.shape}, 状态形状: {state.shape}")
```

## 相关概念

### 1. Transformer
- 完整的注意力机制
- RWKV从中简化但保留核心思想

### 2. Mamba/SSM
- 类似的递归状态更新
- 都追求O(1)推理

### 3. RetNet
- 同样解决Transformer推理问题
- 不同的技术路线

### 4. Linear Attention
- 理论上的先驱工作
- RWKV是其工程化实现

### 5. GPT-2架构
- RWKV的设计受其启发
- 类似的层结构和归一化

## 延伸阅读

### 论文与资源

1. **RWKV: Reinventing RNNs for the Transformer Era** (2023)
   - 作者：Bo Peng
   - 官网：https://www.rwkv.com

2. **RWKV GitHub仓库**
   - https://github.com/BlinkDL/RWKV-LM
   - 包含完整训练代码和预训练模型

3. **RWKV-4模型卡**
   - 1B-14B参数规模
   - 支持32k上下文

### 中文资源

1. **RWKV中文社区**
   - HuggingFace模型：https://huggingface.co/BlinkDL
   - 中文训练语料

2. **相关解读**
   - CSDN/知乎技术博客
   - B站视频讲解

### 进一步研究方向

1. 更深层的理论分析
2. 混合架构探索
3. 硬件感知优化
4. 多模态扩展
5. 高效微调方法
