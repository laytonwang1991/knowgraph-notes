---
title: 扩散Transformer
alias: Diffusion Transformer, DiT, SDXL
tags:
  - AI
  - 深度学习
  - 生成模型
  - Transformer
category: 生成模型
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: Diffusion Transformer将Transformer架构引入扩散模型，大幅提升图像生成质量。
mastery: 0
rating: 0
related_concepts:
  - 扩散模型
  - Transformer
  - Stable Diffusion
  - U-Net
  - 注意力机制
difficulty: 进阶
read_time: 15分钟
prerequisites:
  - 扩散模型基础
  - Transformer架构
  - 深度学习基础
---

# 扩散Transformer

## 一句话定义

> DiT用Transformer替代U-Net作为扩散模型的去噪网络，通过自注意力机制处理图像块的序列，实现更高质量的图像生成。

## 核心公式

### DiT 块操作

$$
\text{DiT}(x) = x + \text{ScaleShift}(x) + \text{MLP}(\text{Attention}(x))
$$

### 自注意力计算

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 噪声预测目标

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]
$$

其中 $c$ 是条件信息（文本、时间步等）。

## 详细说明

### 1. DiT 的核心思想

**为什么用Transformer？**

| 传统U-Net | DiT优势 |
|-----------|---------|
| 卷积操作，局部感受野 | 自注意力，全局感受野 |
| 固定的归纳偏置 | 更灵活地学习空间关系 |
| 难以扩展 | 良好的扩展性 |
| 图像领域专用 | 统一处理图像和文本 |

### 2. DiT 架构

```
输入：噪声图像 x_t + 时间步 t + 条件 c
        ↓
图像分块（Patchify）：将图像分成 2x2 或 4x4 的patch
        ↓
线性嵌入 + 位置编码
        ↓
N x DiT Block（重复多次）
        ↓
Layer Norm + 线性解码
        ↓
输出：预测噪声 ε_θ
```

### 3. DiT Block 变体

| 变体 | 注意力 | 参数效率 |
|------|--------|----------|
| DiT-XL/2 | 局部分块注意力 | 高 |
| DiT-XL/4 | 更小patch | 更高质量 |
| DiT-L | 大模型 | 最高质量 |

### 4. SDXL 架构

SDXL是Stable Diffusion XL的核心模型：

```python
# SDXL 核心组件
class SDXL:
    def __init__(self):
        # 三个模型协同工作
        self.text_encoder = CLIPTextModel()      # 主文本编码器
        self.text_encoder_2 = OpenCLIPTextModel() # 辅助编码器
        self.diffusion_transformer = DiT_XL_2()  # 核心DiT

        # 图像编码器和解码器
        self.vae = VAE()                         # SDXL-VAE

    def generate(self, prompt, num_steps=50):
        # 多条件编码
        text_emb = self.text_encoder(prompt)
        text_emb_2 = self.text_encoder_2(prompt)
        combined_emb = concat([text_emb, text_emb_2])

        # 初始化噪声
        latents = torch.randn(1, 4, 128, 128)

        # 逐步去噪
        for t in reversed(range(num_steps)):
            noise_pred = self.diffusion_transformer(
                latents, t, combined_emb
            )
            latents = self.scheduler.step(noise_pred, t, latents)

        return self.vae.decode(latents)
```

### 5. 关键技术创新

**Patchified Images：**
- 将图像划分为固定大小的patch
- 每个patch线性投影为token
- 序列长度 = (H/patch_size) × (W/patch_size)

**自适应层归一化（AdaLN）：**
```python
# 替代标准LayerNorm，使用条件信息调制
def adaLN(x, c):
    scale, shift = c.split(c.shape[1] // 2, dim=1)
    return LayerNorm(x) * (1 + scale) + shift
```

## 应用场景

- 高分辨率图像生成（1024x1024+）
- 艺术创作与设计
- 图像编辑与修复
- 多模态生成（文本到图像）

## 相关概念

- [[扩散模型]] — DiT是扩散模型的架构升级
- [[Transformer]] — DiT使用Transformer作为去噪网络
- [[Stable Diffusion]] — SDXL基于DiT架构
- [[U-Net]] — 传统扩散模型使用的去噪网络
- [[ControlNet]] — 可与DiT结合的条件控制

## 延伸阅读

- [DiT论文](https://arxiv.org/abs/2212.09748)
- [SDXL论文](https://arxiv.org/abs/2307.01952)
- [Scalable Diffusion Models with Transformers](https://www.wpeebles.com/dit)
