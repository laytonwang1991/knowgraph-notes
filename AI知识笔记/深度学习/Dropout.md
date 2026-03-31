---
title: Dropout正则化
alias: Dropout Regularization
tags:
  - 深度学习
  - 正则化
  - 过拟合
category: 深度学习基础
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: Dropout是一种通过随机丢弃神经元来防止神经网络过拟合的正则化技术
mastery: 0
rating: 0
related_concepts:
  - L1正则化
  - L2正则化
  - 数据增强
  - 权重衰减
  - 变分Dropout
difficulty: 入门
read_time: 10
prerequisites:
  - 神经网络基础
  - 过拟合与欠拟合
  - 梯度下降
---

# Dropout 正则化

## 一句话定义

Dropout 通过在训练过程中随机「丢弃」（置零）部分神经元的输出，使每个神经元不能依赖于少数特定神经元工作，从而迫使网络学习到更鲁棒、更泛化的特征表示。

## 核心公式

**训练阶段（随机丢弃）：**

$$
y = f_W(x) \odot m, \quad m_j \sim \text{Bernoulli}(1-p)
$$

其中：
- $m$：伯努利随机掩码，$m_j = 0$ 表示丢弃第 $j$ 个神经元
- $p$：Dropout 比率（通常为 0.5）
- $\odot$：元素级乘法

**测试阶段（近似平均）：**

$$
\hat{y} = f_W(x) \cdot (1-p)
$$

或使用「 inverted dropout」：

$$
y = f_W(x) \odot \frac{m}{1-p}
$$

测试时直接使用完整网络，不做任何修改。

## 详细说明

### 1. Dropout 的核心思想

**类比生物学**：大脑中的神经元在不同的任务中会「死亡」或失去连接，神经网络也需要具备这种鲁棒性。

**集成学习视角**：
- 每次训练迭代相当于训练一个子网络
- 最终模型是所有子网络的集成
- 指数级数量的子网络共享参数

### 2. Bernoulli 采样

Dropout 使用伯努利分布生成掩码：

```python
def bernoulli_mask(p, shape):
    """生成伯努利掩码"""
    return np.random.rand(*shape) > p

# 示例：p=0.5 时
# mask = [True, False, True, True, False, ...]
# 输出 0/1 形式
mask = (np.random.rand(*shape) < (1-p)).astype(np.float32)
```

### 3. Dropout 比率的选择

| Dropout 比率 | 适用场景 |
|-------------|---------|
| 0.1 - 0.2 | 轻度正则化，用于大数据集 |
| 0.3 - 0.5 | 标准设置，平衡正则化强度 |
| 0.5 - 0.8 | 强正则化，用于小数据集或深层网络 |

**经验法则**：
- 全连接层：0.5
- 卷积层：0.1 - 0.3（卷积层参数较少，通常不需要强 Dropout）
- 循环层：0.2 - 0.3

### 4. 测试时近似

**原始方法（Weight Scaling）**：
- 训练时不做任何缩放
- 测试时将权重乘以 $(1-p)$
- 简单但不够精确

**Inverted Dropout（标准做法）**：
- 训练时将激活值除以 $(1-p)$
- 测试时保持权重不变
- 数值上更稳定

## 代码示例

### PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dropout(nn.Module):
    """Dropout 实现"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # 丢弃概率

    def forward(self, x):
        if self.training:
            # 训练时：生成掩码并应用
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return x * mask / (1 - self.p)  # Inverted Dropout
        else:
            # 测试时：直接返回
            return x
```

### nn.Dropout 使用示例

```python
class MLPWithDropout(nn.Module):
    """带 Dropout 的多层感知机"""
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Dropout 在激活函数之后
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 使用
model = MLPWithDropout(784, 512, 10, dropout_p=0.5)
model.train()  # 训练模式
model.eval()   # 测试模式
```

### 自定义 Dropout 函数

```python
def dropout_forward(x, dropout_p, mode='inverted'):
    """
    Dropout 前向传播

    Args:
        x: 输入张量
        dropout_p: 丢弃概率
        mode: 'inverted' 或 'scaling'
    """
    mask = np.random.random(x.shape) > dropout_p
    scale = 1.0 / (1 - dropout_p) if mode == 'inverted' else 1.0

    if mode == 'inverted':
        return x * mask * scale
    else:  # scaling
        return x * mask, mask

def dropout_backward(dout, mask, dropout_p, mode='inverted'):
    """Dropout 反向传播"""
    if mode == 'inverted':
        return dout * mask
    else:
        return dout * mask / (1 - dropout_p)
```

### Alpha Dropout（SELU 激活函数配套）

```python
class AlphaDropout(nn.Module):
    """Alpha Dropout：保持输出均值和方差为 SELU 要求的值"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            # 生成掩码
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p / 2))
            # 保持 SELU 的自归一化性质
            alpha = 1.673263242354817284
            return mask * x + (1 - mask) * (-alpha)
        return x
```

### Dropout 变体实现

```python
class Dropout2d(nn.Module):
    """Spatial Dropout：丢弃整个通道"""
    def forward(self, x):
        if self.training:
            # shape: (N, C, H, W)
            mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], 1, 1), 1 - self.p))
            return x * mask / (1 - self.p)
        return x

class Dropout3d(nn.Module):
    """3D Dropout"""
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1], 1, 1, 1), 1 - self.p))
            return x * mask / (1 - self.p)
        return x

class StochasticDepth(nn.Module):
    """随机深度（Skip Connection 层面的 Dropout）"""
    def __init__(self, survival_prob=0.5):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full((x.shape[0], 1, 1, 1), self.survival_prob))
            return x * mask / self.survival_prob
        return x
```

## 应用场景

### 1. 计算机视觉

```python
# CNN 中的 Dropout
class CNNWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.25)  # Spatial Dropout
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.25)
        self.fc = nn.Linear(128 * 8 * 8, 512)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        # ...
```

### 2. 自然语言处理

```python
# RNN + Dropout
class RNNWithDropout(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Dropout(p=0.3)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        return x
```

### 3. Transformer 中的 Dropout 位置

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Attention 后 + Dropout + 残差
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN 后 + Dropout + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
```

## Dropout 变体对比

| 变体 | 描述 | 适用场景 |
|------|------|---------|
| **Standard Dropout** | 随机丢弃神经元 | 全连接层 |
| **Spatial Dropout** | 丢弃整个通道 | CNN |
| **DropBlock** | 丢弃连续区域 | CNN（大特征图） |
| **Stochastic Depth** | 丢弃整个层 | ResNet 等深层网络 |
| **DropConnect** | 丢弃权重连接 | 全连接层 |
| **Variational Dropout** | 自适应丢弃率 | 需要稀疏激活的场景 |
| **Concrete Dropout** | 可学习的连续 Dropout | 端到端训练 |
| **Multi-Sample Dropout** | 多次采样平均 | 小数据集 |

### DropBlock 实现要点

```python
class DropBlock2D(nn.Module):
    """DropBlock：丢弃 CNN 中的连续区域"""
    def __init__(self, block_size=7, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        # 计算 gamma
        gamma = self.drop_prob / (self.block_size ** 2)
        # 生成掩码（使用 max_pool 实现）
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        mask = mask.unsqueeze(1)
        mask = F.max_pool2d(mask, kernel_size=self.block_size,
                           stride=self.block_size, padding=self.block_size//2)
        mask = 1 - mask

        return x * mask
```

## 相关概念

### Dropout vs 其他正则化方法

| 方法 | 原理 | 特点 |
|------|------|------|
| **L2 正则化** | 惩罚权重范数 | 使权重趋于小值 |
| **L1 正则化** | 惩罚权重绝对值 | 产生稀疏权重 |
| **Dropout** | 随机丢弃神经元 | 隐式集成学习 |
| **数据增强** | 增加训练样本多样性 | 扩大有效数据集 |
| **权重衰减** | 限制权重增长 | 防止权重过大 |

### Dropout 与 Batch Norm 的关系

- **早期实践**：Dropout 和 Batch Norm 经常一起使用
- **现代发现**：两者可能相互干扰（方差偏移问题）
- **现代架构**：Transformer 主要使用 Layer Norm + Dropout，很少用 Batch Norm

## 延伸阅读

1. **原始论文**：
   - Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", 2014

2. **变体论文**：
   - "DropBlock: A regularization technique for convolutional neural networks"
   - "Deep Networks with Stochastic Depth"
   - "Gaussian Dropout" and "Multi-Sample Dropout"

3. **实践资源**：
   - PyTorch Dropout 文档
   - TensorFlow Dropout 文档
   - fast.ai 课程中的现代 Dropout 使用技巧

4. **调试技巧**：
   - 如果模型在训练集上loss很低但验证集很高，考虑增加 Dropout
   - Dropout 比率过大可能导致欠拟合
   - 定期检查 `model.training` / `model.eval()` 模式切换
