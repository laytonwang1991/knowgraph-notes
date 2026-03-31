---
title: LoRA低秩适配器
alias: Low-Rank Adaptation, LoRA
tags:
  - AI
  - 深度学习
  - 模型微调
  - 参数高效微调
category: 生成模型
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: LoRA通过低秩分解减少微调参数量，在保持生成质量的同时实现高效模型定制。
mastery: 0
rating: 0
related_concepts:
  - 模型微调
  - 扩散模型
  - Stable Diffusion
  - 参数高效微调
  - Adapter
difficulty: 进阶
read_time: 10分钟
prerequisites:
  - 神经网络基础
  - 深度学习基础
  - 线性代数基础
---

# LoRA低秩适配器

## 一句话定义

> LoRA通过冻结预训练模型权重并添加低秩分解矩阵，大幅减少微调所需参数量。

## 核心公式

### 低秩分解

$$
W = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}
$$

其中 $r \ll \min(d, k)$ 是秩（rank）。

### 前向传播

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

### 训练目标

$$
\mathcal{L} = \mathcal{L}_{task} + \frac{\lambda}{r} \|W_0 + BA\|_F^2
$$

## 详细说明

### 1. LoRA 的核心思想

**问题：** 大模型全参数微调需要大量GPU显存和计算资源。

**解决方案：** 冻结原模型权重，只训练低秩增量。

```
原始权重 W_0 (冻结)
    │
    └──→ 输出: W_0 · x

新增低秩分支:
    x → A (降维) → 零初始化 → B (升维) → 输出: BA · x
                    ↓
              最终输出: W_0 · x + BA · x
```

### 2. 关键参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| rank $r$ | 低秩维度，越大越强 | 4, 8, 16, 32 |
| alpha $\alpha$ | 缩放因子，通常设为rank | 等于rank |
| target_modules | 应用LoRA的层 | q_proj, v_proj |

**参数量计算：**
$$
\text{params} = 2 \times r \times d_{model} \times n_{layers}
$$

例如：rank=4, dim=768, 12层 → 仅 ~150万参数 vs 全训练 ~1.8亿

### 3. 为什么有效？

**内在秩假说：**
- 预训练模型已经学习了丰富知识
- 任务适配只需要小的方向调整
- 这些方向存在于低秩子空间

**实验验证：**
- ImageNet上，rank=1 就能达到 rank=8 的90%效果
- 说明任务适配确实是低秩的

### 4. 代码实现

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=4):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 冻结原始权重
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # 低秩矩阵 A 和 B（可选bias）
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初始化 A 为随机，B 为零
        nn.init.normal_(self.lora_A, std=1.0 / rank)

    def forward(self, x):
        # 原始输出 + 低秩调整
        return F.linear(x, self.weight) + \
               self.scaling * F.linear(
                   F.linear(x, self.lora_A),
                   self.lora_B
               )
```

### 5. 扩散模型中的 LoRA

```python
# Stable Diffusion LoRA 典型配置
LoraConfig = {
    "rank": 16,
    "alpha": 16,
    "target_modules": [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0", "ff.net.2"
    ],
    "dropout": 0.1
}

# 使用示例
lora_model = load_lora_weights("artist_style.safetensors")
pipe = StableDiffusionPipeline.from_pretrained("sd-v1-5")
pipe.unet = apply_lora_to_unet(pipe.unet, lora_model)
```

### 6. LoRA 变体

| 变体 | 年份 | 改进 |
|------|------|------|
| LoRA | 2021 | 基础低秩适配 |
| QLoRA | 2023 | 量化+LoRA，4bit微调 |
| DoRA | 2024 | 权重分解方向 |
| LoRA+ | 2024 | 自适应学习率 |
| AdaLoRA | 2023 | 自适应秩调整 |

## 应用场景

- 个性化图像风格（定制艺术家风格）
- 角色定制（特定人物生成）
- 概念学习（物体、风格编码）
- 任务适配（特定领域微调）
- 边缘设备部署（减少计算需求）

## 优缺点

| 优点 | 缺点 |
|------|------|
| 参数量小 | 可能欠拟合复杂任务 |
| 训练快 | 需要选择合适的rank |
| 推理开销小 | 任务间可能冲突 |
| 可组合 | 超参数敏感 |

## 相关概念

- [[扩散模型]] — LoRA常用于Stable Diffusion等扩散模型
- [[Stable Diffusion]] — 最流行的LoRA应用底模
- [[ControlNet]] — 另一个控制生成的技术，可与LoRA结合
- [[参数高效微调]] — LoRA是PEFT的一种
- [[Adapter]] — 类似的高效微调方法

## 延伸阅读

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT库](https://github.com/huggingface/peft)
- [LoRA训练指南](https://moritz.pm/posts/loras/)
