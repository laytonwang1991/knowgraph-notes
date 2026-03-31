---
title: Flow模型
alias: Flow-based Generative Model, 可逆流模型
tags: [生成模型, 可逆变换, 似然估计, 归一化流]
category: 生成模型
created: 2026-03-31
updated: 2026-03-31
author: AI
description: Flow模型是一类基于可逆变换（bijective transformation）的生成模型，通过精心设计的双射将简单分布变换为复杂数据分布，同时能够精确计算似然。
mastery: 3
rating: 7
related_concepts: [GAN, 扩散模型, VAE, RealNVP, Glow, NICE]
difficulty: 4
read_time: 25
prerequisites: [概率论基础, 变量变换公式, 神经网络反向传播]
---

# Flow模型

## 一句话定义

Flow模型（Flow-based Generative Model）是一类利用可逆双射（bijection）网络将简单概率分布（如高斯分布）变换为复杂数据分布的生成模型，其核心优势是能够精确计算样本的似然而无需近似。

## 核心公式

### 变量变换公式（Change of Variables）

若随机变量 $z = f(x)$，其中 $f$ 是可逆双射，则：

$$
p_X(x) = p_Z(z) \left| \det \frac{\partial f^{-1}}{\partial x} \right| = p_Z(f(x)) \left| \det \frac{\partial f}{\partial x} \right|^{-1}
$$

对数似然为：

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f}{\partial x} \right|
$$

### 行列式的链式法则

对于复合变换 $x = f_K(\cdots f_2(f_1(z)))$：

$$
\log \left| \det \frac{\partial x}{\partial z} \right| = \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial u_{k-1}} \right|
$$

## 详细说明

### 1. 基本架构

- **双射约束**：模型必须满足可逆性，即存在 $f^{-1}$ 使得 $x = f(z)$ 且 $z = f^{-1}(x)$
- **行列式计算**：雅可比矩阵行列式 $\det \frac{\partial f}{\partial x}$ 必须可高效计算（$O(D)$ 而非 $O(D^3)$）
- **潜空间**：潜变量 $z$ 与数据 $x$ 维度相同，这是Flow模型与VAE的关键区别

### 2. 典型模型演进

| 模型 | 年份 | 核心贡献 |
|------|------|----------|
| NICE | 2014 | 首次提出加性耦合层，简化行列式计算 |
| RealNVP | 2016 | 仿射耦合层 + 多尺度架构 |
| Glow | 2018 | 可逆1x1卷积 +ActNorm，扩展到大规模人脸生成 |
| Flow++ | 2019 | 变分推理耦合层，自回归组件 |

### 3. 耦合层设计

**加性耦合层（NICE）**：
$$
\begin{cases}
x_{1:d} = z_{1:d} \\
x_{d+1:D} = z_{d+1:D} + m(z_{1:d})
\end{cases}
$$
行列式为1（无计算开销）。

**仿射耦合层（RealNVP）**：
$$
\begin{cases}
x_{1:d} = z_{1:d} \\
x_{d+1:D} = z_{d+1:D} \odot \exp(s(z_{1:d})) + t(z_{1:d})
\end{cases}
$$
其中 $s(\cdot)$ 和 $t(\cdot)$ 为尺度和平移函数。

### 4. 优缺点

**优点**：
- 精确的似然估计（ELBO下界等于真实似然）
- 隐变量可逆推理，$z$ 具有良好语义
- 生成过程可精确求逆，适合推理任务

**缺点**：
- 必须保持维度不变，内存和计算成本较高
- 网络架构受限（必须可逆），表达能力弱于GAN
- 难以堆叠非常深的网络

### 5. 与扩散模型的关系

扩散模型可以视为Flow模型的一种极限情况：当Flow的层数趋于无穷时，每一层的变换趋于无穷小，此时累积的雅可比行列式变为SDE中的扩散项。两者本质都是将简单分布变换为复杂分布，但扩散模型通过近似避免了显式计算行列式。

## 代码示例

```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    """仿射耦合层实现"""
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - dim // 2) * 2  # scale + shift
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        params = self.net(x1)
        scale, shift = params.chunk(2, dim=-1)
        scale = torch.exp(scale)  # 确保scale为正
        z1 = x1
        z2 = x2 * scale + shift
        log_det = scale.log().sum(dim=-1)
        return torch.cat([z1, z2], dim=-1), log_det

class GlowBlock(nn.Module):
    """Glow中的一个块：ActNorm + 置换 + 仿射耦合"""
    def __init__(self, dim):
        super().__init__()
        self.actnorm = nn.BatchNorm1d(dim)
        self.perm = nn.Linear(dim, dim, bias=False)  # 可逆1x1卷积
        self.coupling = AffineCoupling(dim)

    def forward(self, x):
        x = self.actnorm(x)
        x = self.perm(x)
        x, log_det = self.coupling(x)
        return x, log_det

# 前向过程（从噪声生成数据）
def flow_forward(z, blocks):
    log_det_total = 0
    for block in blocks:
        z, log_det = block(z)
        log_det_total += log_det
    return z, log_det_total

# 逆向过程（从数据推理潜码）
def flow_inverse(x, blocks):
    for block in reversed(blocks):
        x = block.inverse(x)
    return x
```

## 应用场景

1. **高保真图像生成**：Glow生成1024x1024高分辨率人脸，log-likelihood达到世界先进水平
2. **语音合成**：WaveGlow等模型用于高质量语音生成
3. **数据压缩**：精确似然使其适合熵编码压缩
4. **异常检测**：精确的密度估计可用于识别低概率异常样本
5. **潜变量插值**：$z$空间的线性插值具有良好语义，可用于图像编辑
6. **可逆推理任务**：如变分自编码器的后验近似

## 相关概念

- **GAN**：对抗训练，无需精确似然，但训练不稳定
- **VAE**：变分推断，允许维度变化（编码器下采样），但似然为下界
- **扩散模型**：Flow的极限形式，层数趋于无穷，使用随机微分方程
- **归一化流（Normalizing Flow）**：Flow模型的另一个名称，强调将分布归一化
- **RealNVP**：第一个实用的Flow模型，使用仿射耦合层
- **Glow**：基于可逆1x1卷积的大规模Flow模型

## 延伸阅读

- Dinh L, Krueger D, Bengio Y. "NICE: Non-linear Independent Components Estimation", ICLR 2015
- Dinh L, Sohl-Dickstein J, Bengio S. "Density Estimation using Real NVP", ICLR 2017
- Kingma D P, Dhariwal P. "Glow: Generative Flow with Invertible 1x1 Convolutions", NeurIPS 2018
- Ho J, Chen X, Srinivas A, et al. "Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design", ICML 2019
