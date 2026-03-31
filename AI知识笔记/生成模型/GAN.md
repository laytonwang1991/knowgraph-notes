---
title: 生成对抗网络
alias: Generative Adversarial Network, GAN
tags:
  - AI
  - 深度学习
  - 生成模型
category: 生成模型
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: GAN由生成器和判别器组成，通过对抗训练生成逼真的假数据。
mastery: 0
rating: 0
related_concepts:
  - 深度学习
  - 生成模型
  - Diffusion
  - VAE
difficulty: 进阶
read_time: 10分钟
prerequisites:
  - 神经网络基础
  - 概率论基础
---

# 生成对抗网络

## 一句话定义

> GAN通过让生成器和判别器相互对抗训练，使生成器学会生成逼真的假数据。

## 核心公式

### GAN 目标函数

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

- $D(x)$：判别器对真实数据 $x$ 的输出
- $G(z)$：生成器对噪声 $z$ 的输出
- $D(G(z))$：判别器对生成数据的判断

## 详细说明

### 1. GAN 的核心思想

**对抗训练：**
- **生成器 (G)**：学习生成假数据，目标是骗过判别器
- **判别器 (D)**：区分真实数据和生成数据，目标是不被骗

```
噪声 z → 生成器 G → 生成数据 G(z)
                ↓
          判别器 D (判断真假)
                ↓
        D(G(z)) → 损失 → 更新 G 和 D
```

### 2. 训练过程

1. 从真实数据采样 $x$
2. 从噪声分布采样 $z$
3. 生成假数据 $G(z)$
4. 判别器学习：$x$ 判真，$G(z)$ 判假
5. 生成器学习：让 $D(G(z))$ 判真

### 3. 经典 GAN 变体

| 模型 | 年份 | 贡献 |
|------|------|------|
| DCGAN | 2015 | 首个稳定训练的GAN，使用卷积 |
| WGAN | 2017 | Wasserstein距离，解决模式崩溃 |
| StyleGAN | 2018 | 可控生成，质量大幅提升 |
| BigGAN | 2019 | 大规模训练，高分辨率图像 |
| StyleGAN2 | 2019 | 改进架构，消除伪影 |

### 4. 代码实现

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
```

## 应用场景

- 图像生成（人脸、风景、艺术）
- 图像编辑（风格迁移）
- 数据增强
- 视频生成

## 优缺点

| 优点 | 缺点 |
|------|------|
| 生成质量高 | 训练不稳定 |
| 无需显式概率建模 | 模式崩溃 |
| 生成速度快 | 可解释性差 |

## 相关概念

- [[生成模型]] — GAN是生成模型的重要分支
- [[Diffusion]] — 另一种生成模型范式
- [[VAE]] — 变分自编码器，另一种生成模型

## 延伸阅读

- [GAN论文](https://arxiv.org/abs/1406.2661)
- [GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)
- [The Illustrated GAN](https://jonbruner.com/generative-adversarial-networks/)
