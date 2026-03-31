---
title: 变分自编码器
alias: Variational Autoencoder, VAE
tags:
  - AI
  - 深度学习
  - 生成模型
category: 生成模型
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: VAE通过学习数据的潜在分布，使用编码器-解码器结构生成新样本。
mastery: 0
rating: 0
related_concepts:
  - 生成模型
  - 自编码器
  - 潜在变量模型
  - Diffusion
  - GAN
difficulty: 进阶
read_time: 10分钟
prerequisites:
  - 神经网络基础
  - 概率论基础
  - 自编码器基础
---

# 变分自编码器

## 一句话定义

> VAE通过将数据编码到潜在空间并学习潜在分布的参数，从潜在空间采样并解码生成新样本。

## 核心公式

### 重参数化技巧

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### VAE 损失函数

$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

其中：
- 第一项：重构损失（让解码器重建输入）
- 第二项：KL散度（让后验分布接近先验分布）

### KL散度解析解

$$
D_{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2} \sum_{j=1}^{J} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)
$$

## 详细说明

### 1. VAE 的核心思想

**两阶段架构：**
- **编码器 (Encoder)**：将输入 x 映射到潜在分布参数 (μ, σ)
- **解码器 (Decoder)**：从潜在空间采样并重建数据

```
输入 x → 编码器 → 潜在分布 N(μ, σ²)
                        ↓ 重参数化采样
              潜在向量 z → 解码器 → 重建 x'
```

### 2. 为什么要变分？

传统自编码器的潜在空间是不连续的，无法直接采样生成新数据。VAE通过：
- 假设潜在空间服从正态分布
- 学习分布参数而非离散编码
- 实现可连续的潜在空间

### 3. 重参数化技巧

为了让梯度能够反向传播，VAE使用重参数化：

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这样随机性被转移到可训练的确定变量外。

### 4. 经典 VAE 变体

| 模型 | 年份 | 贡献 |
|------|------|------|
| VAE | 2013 | 基础变分自编码器 |
| VAE++ | 2016 | 改进的生成模型 |
| CVAE | 2015 | 条件变分自编码器 |
| β-VAE | 2016 | 可控解耦表示学习 |
| VQ-VAE | 2017 | 矢量量化，离散潜在空间 |

### 5. 代码实现

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 重参数化
        z = self.reparameterize(mu, logvar)

        # 解码
        return self.decoder(z), mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss
```

## 应用场景

- 图像生成与重建
- 数据压缩
- 表示学习
- 异常检测
- 药物发现

## 优缺点

| 优点 | 缺点 |
|------|------|
| 训练稳定 | 生成质量通常不如GAN |
| 潜在空间连续 | 模糊的重建结果 |
| 概率解释 | 后验崩溃问题 |
| 可控生成 | 计算量较大 |

## 相关概念

- [[生成模型]] — VAE是生成模型的重要分支
- [[GAN]] — 另一种主流生成模型
- [[扩散模型]] — 现代扩散模型常使用VAE作为图像编码器
- [[自编码器]] — VAE的基线架构

## 延伸阅读

- [VAE论文](https://arxiv.org/abs/1312.6114)
- [变分推断详解](https://arxiv.org/abs/1601.00670)
- [The Illustrated VAE](https://jmetzen.github.io/2015-11-27/vae.html)
