---
title: RetNet保留网络
alias: Retention Network, RetNet, Microsoft Retention
tags:
  - RetNet
  - 保留网络
  - Transformer替代
  - Microsoft
  - 多尺度保留
category: 模型架构创新
created: 2026-03-31
updated: 2026-03-31
author: AI知识笔记
description: RetNet是Microsoft提出的Transformer替代架构，通过多尺度保留机制同时支持并行、递归和分块计算三种范式，在保持高性能的同时实现低延迟推理
mastery: 0.6
rating: 8
related_concepts:
  - Transformer
  - RWKV
  - Mamba
  - 线性注意力
  - 状态空间模型
difficulty: 高难度
read_time: 20分钟
prerequisites:
  - Transformer原理
  - 注意力机制
  - 循环神经网络基础
---

# RetNet保留网络

## 一句话定义

RetNet（Retention Network）是由Microsoft提出的一种Transformer替代架构，通过**多尺度保留机制**同时实现并行训练的高效性与递归推理的低成本，在部署性能上显著优于Transformer。

## 核心公式

### 保留机制（Retention）

$$
Retain(x) = \ Attention(Q, K, V) \circ (DSR)
$$

更精确的数学表达：

$$
Q = W_Q x, \quad K = W_K x, \quad V = W_V x
$$

$$
R(i, j) = \begin{cases}
\alpha^{i-j} & \text{当 } i \geq j \text{（衰减语义）} \\
0 & \text{其他}
\end{cases}
$$

其中 $\alpha$ 为衰减因子，控制历史信息的保留强度。

### 多尺度保留机制

$$
\mathbf{Y}_i = \sum_{j=1}^{i} \frac{\alpha^{i-j}}{(1 - \alpha^{i-j})\sqrt{d_k}} Q_i^T K_j \cdot V_j + \beta \sum_{j=1}^{i} \alpha^{i-j} Q_i^T K_j \cdot X_j
$$

简化形式：

$$
Y = (QK^T \odot S) V + \beta (Q X^T \odot S) V
$$

其中：
- $S$：衰减矩阵 $S_{ij} = \alpha^{|i-j|}$
- $\beta$：并行保留的加权系数
- $\sqrt{d_k}$：缩放因子

### 递归形式（Ring Retention）

$$
S_n = \alpha S_{n-1} + K_n^T V_n
$$

$$
Y_n = Q_n S_n + \beta Q_n X_n^T S_n
$$

## 详细说明

### 1. 三种计算模式

RetNet的核心创新在于**同时支持三种计算范式**，这在训练和推理时可以灵活切换：

| 模式 | 描述 | 优势 | 适用场景 |
|------|------|------|----------|
| **并行模式** | 矩阵运算并行计算 | 训练高效 | GPU/TPU训练 |
| **递归模式** | 逐token递归更新 | 推理O(1) | 低延迟部署 |
| **分块模式** | 分段处理长序列 | 平衡效率 | 长上下文 |

### 2. 并行保留（Parallel Retention）

```
并行计算示意：
┌─────────────────────────────────────────────────────┐
│  Q1 Q2 Q3 Q4                                        │
│  K1 K2 K3 K4  →  注意力分数 →  softmax →  加权求和   │
│  V1 V2 V3 V4                                        │
│                                                     │
│  + 指数衰减矩阵D → 位置感知                          │
└─────────────────────────────────────────────────────┘

特点：
- 完全可并行化
- 利用矩阵乘法优化
- 适合GPU训练
```

### 3. 递归保留（Recurrent Retention）

```python
class RecurrentRetention:
    """递归保留机制"""

    def __init__(self, d_model, d_state, alpha=0.9):
        self.d_model = d_model
        self.d_state = d_state
        self.alpha = alpha

        # 投影矩阵
        self.W_Q = nn.Linear(d_model, d_state)
        self.W_K = nn.Linear(d_model, d_state)
        self.W_V = nn.Linear(d_model, d_state)

        # 衰减状态
        self.s = None  # 保留状态

    def forward(self, x_t):
        """
        x_t: 当前时刻输入 (batch, d_model)
        返回: 当前输出 (batch, d_model)
        """
        # 投影
        q_t = self.W_Q(x_t)  # (batch, d_state)
        k_t = self.W_K(x_t)
        v_t = self.W_V(x_t)

        # 更新衰减状态
        if self.s is None:
            self.s = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        else:
            self.s = self.alpha * self.s + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        # 计算输出
        y_t = torch.einsum('bnd,bnd->bn', q_t, k_t) / (torch.norm(k_t, dim=-1) + 1e-6)
        y_t = y_t.unsqueeze(-1) * v_t  # (batch, d_state, d_model)

        return y_t.sum(dim=-2)  # (batch, d_model)
```

### 4. 分块递归（Chunkwise Retention）

```
分块处理示意：

序列: [x1, x2, x3, x4, x5, x6, x7, x8]
分块:  [  chunk1  ], [  chunk2  ]
       [x1,x2,x3,x4], [x5,x6,x7,x8]

块内：并行计算
块间：递归连接

优势：
- 长序列分块处理，降低内存
- 块间信息通过递归传递
- 可与FlashAttention结合
```

### 5. 与Transformer对比

| 特性 | Transformer | RetNet |
|------|-------------|--------|
| 训练复杂度 | O(L²) | O(L²) |
| 推理延迟 | O(L) | **O(1)** |
| 内存占用 | O(L²) | **O(L)** |
| 长序列 | 受限于显存 | 可处理超长序列 |
| 表达能力 | 强 | 与Transformer相当 |

### 6. 位置编码的内在表示

RetNet通过**指数衰减**实现位置感知，无需额外的位置编码：

$$
P_{i,j} = \alpha^{i-j}
$$

- $\alpha \to 1$：长期依赖更强
- $\alpha \to 0$：更关注短期

这与RNN的隐藏状态衰减有本质联系，但通过矩阵形式实现了并行化。

## 应用场景

### 1. 大语言模型部署
- 云端推理服务
- 边缘设备部署
- 移动端应用

### 2. 长文本处理
- 文档摘要
- 书籍理解
- 代码补全

### 3. 语音识别与生成
- 流式ASR
- 低延迟TTS

### 4. 时间序列分析
- 金融预测
- 传感器数据

## 代码示例：完整RetNet Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleRetention(nn.Module):
    """多尺度保留机制"""

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gamma = nn.Parameter(torch.randn(num_heads, self.head_dim))

        # 输出投影
        self.o_proj = nn.Linear(d_model, d_model)

        # 初始化gamma（衰减因子）
        nn.init.uniform_(self.gamma, -0.1, 0.1)

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, d_model)
        state: None（并行）或 前一状态（递归）
        """
        B, L, D = x.shape

        # 残差路径
        residual = x

        # 投影到QKV
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # 跨维度转置
        q = q.transpose(1, 2)  # (B, heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 衰减矩阵
        gamma = self.gamma  # (num_heads, head_dim)
        decay = gamma.pow(torch.arange(len(x.shape[1]), device=x.device)).unsqueeze(0)
        decay = decay.unsqueeze(-1)  # (1, seq_len, 1) for broadcasting

        # 计算保留分数
        retention = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 应用指数衰减
        retention = retention * decay

        # Softmax归一化
        retention = F.softmax(retention, dim=-1)

        # 应用到V
        output = torch.matmul(retention, v)

        # 跨头维度重排并合并
        output = output.transpose(1, 2).contiguous().view(B, L, D)

        # 输出投影
        output = self.o_proj(output)

        # 门控残差连接
        output = output * F.silu(residual)

        return output


class RetNetBlock(nn.Module):
    """完整的RetNet块"""

    def __init__(self, d_model, num_heads, ff_dim=4*d_model):
        super().__init__()

        # 多尺度保留
        self.retention = MultiScaleRetention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 保留 + 残差
        x = x + self.retention(self.norm1(x))

        # FFN + 残差
        x = x + self.ffn(x)

        return x


# 使用示例
model = RetNetBlock(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
y = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {y.shape}")
```

## 相关概念

### 1. RWKV
- 同样是Transformer替代方案
- 使用token-shift机制
- 与RetNet有相似目标

### 2. Linear Attention
- 线性时间复杂度注意力
- RetNet是其理论推广

### 3. State Space Models (Mamba)
- 连续时间动态系统
- 与RetNet在设计上有关联

### 4. Flash Attention
- 注意力近似算法
- 可与RetNet结合

## 延伸阅读

### 论文

1. **RetNet: Retention Network is Efficient on Both Training and Inference** (2023)
   - 作者：Yutao Sun et al., Microsoft
   - 必读：完整技术细节

2. **What is Retention?** - 理论分析
   - 探讨RetNet与RNN/Transformer的数学联系

### 开源资源

1. **Microsoft RetNet官方实现**
   - https://github.com/microsoft/torchscale

2. **相关博客**
   - Lil'Log的RetNet解析
   - Jay Alammar可视化

### 进一步研究方向

1. 更大规模实验验证
2. 与其他架构的组合
3. 硬件协同优化
4. 多模态扩展
