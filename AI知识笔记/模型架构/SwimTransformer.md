---
title: SwimTransformer
alias: Swin Transformer
tags:
  - AI
  - 深度学习
  - 计算机视觉
  - Transformer
  - 视觉模型
category: 模型架构
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: Swin Transformer是一种高效的分层视觉Transformer，通过Window Attention和Shifted Windows机制在保持全局建模能力的同时大幅降低计算复杂度。
mastery: 0
rating: 0
related_concepts:
  - Transformer
  - ViT
  - Window Attention
  - 图像分类
  - 分层设计
difficulty: 较高
read_time: 18分钟
prerequisites:
  - Transformer
  - ViT
  - 卷积神经网络基础
---

# Swin Transformer

## 一句话定义

> Swin Transformer是一种分层设计的视觉Transformer，通过Shifted Window（移位窗口）机制，在局部窗口内计算自注意力，并跨层级逐步合并特征，实现类似卷积神经网络的层次化表示学习。

## 核心公式

### Window Attention

将特征图划分为非重叠窗口，每个窗口包含 $M \times M$ 个patches：

$$\text{Attention}(Q, K, V) = \text{SoftMax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中 $Q, K, V \in \mathbb{R}^{M^2 \times d}$

### Shifted Window分区

在相邻层之间，窗口偏移 $\lfloor M/2 \rfloor$ 个patches：

- Layer l：常规窗口分区
- Layer l+1：偏移后的窗口分区

偏移打破窗口边界限制，实现跨窗口信息传递。

### Patch Merging

相邻 $2 \times 2$ patches进行合并：

$$[\hat{X}_0, \hat{X}_1, \hat{X}_2, \hat{X}_3] = \text{Concat}(X_0, X_1, X_2, X_3) \cdot E$$

输出通道数变为2倍，分辨率降为1/2。

## 详细说明

### 1. 分层设计（Hierarchical Design）

**四级网络结构：**
- Stage 1: $H/4 \times W/4$ 分辨率，通道数C
- Stage 2: $H/8 \times W/8$ 分辨率，通道数2C
- Stage 3: $H/16 \times W/16$ 分辨率，通道数4C
- Stage 4: $H/32 \times W/32$ 分辨率，通道数8C

**与ViT的区别：**
- ViT：单一分辨率，全局注意力
- Swin：多分辨率，金字塔结构

### 2. Window Attention机制

**窗口划分：**
- 默认窗口大小 $M = 7$
- 每个窗口包含49个patches（7x7）
- 显著降低计算量：$O((HW)^2) \rightarrow O(HW \cdot M^2)$

**计算优势：**
- 固定窗口大小，图像尺寸增大时计算量线性增长
- 与图像分辨率成正比，而非平方关系

### 3. Shifted Windows机制

**信息跨窗口传递：**
- 常规窗口：每个patch只与同窗口内patch交互
- 移位窗口：窗口偏移后，patch可与不同窗口的patch交互

**实现细节：**
- 使用mask避免无效注意力计算
- 相邻层窗口位置不同，实现跨窗口连接

### 4. Swin Transformer变体

| 模型 | 窗口大小 | 层数 | 通道数 | 参数量 | ImageNet Top-1 |
|------|----------|------|--------|--------|----------------|
| Swin-T | 7 | 2,2,6,2 | 96,192,384,768 | 28M | 81.2% |
| Swin-S | 7 | 2,2,18,2 | 96,192,384,768 | 50M | 83.2% |
| Swin-B | 7 | 2,2,18,2 | 128,256,512,1024 | 88M | 83.5% |
| Swin-L | 7 | 2,2,18,2 | 192,384,768,1536 | 196M | 84.8% |

### 5. Relative Position Bias

$$B_{ij} = \text{RelativeBias}(i-j)$$

在注意力计算中加入相对位置偏置：

$$\text{Attention}(Q, K, V) = \text{SoftMax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$

位置偏置帮助模型感知空间关系。

## 代码示例

### Swin Transformer图像分类

```python
import torch
from transformers import SwinImageProcessor, SwinForImageClassification
from PIL import Image

processor = SwinImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')
model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224')

image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
print(f"Predicted: {model.config.id2label[predicted_class]}")
```

### Window Attention实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置
        self.relative_position_bias = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape  # B=batch, N=patches, C=dim
        h = w = int(N ** 0.5)
        x = x.view(B, h, w, C)

        # 填充到window size的倍数
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape

        # 划分为windows
        x = x.view(B, Hp // self.window_size[0], self.window_size[0],
                   Wp // self.window_size[1], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = x.view(B * (Hp // self.window_size[0]) * (Wp // self.window_size[1]),
                         self.window_size[0] * self.window_size[1], C)

        # QKV projection
        qkv = self.qkv(windows).reshape(-1, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.relative_position_bias
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size[0] * self.window_size[1], C)
        return x
```

### 使用timm库加载Swin

```python
import timm

# 加载预训练Swin Transformer
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
model.eval()

# 查看模型结构
print(model.default_cfg['url'])
print(f"Num parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# 推理
import torch
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = model(x)
print(f"Output shape: {y.shape}")
```

## 应用场景

| 任务 | 应用 | Swin优势 |
|------|------|----------|
| 图像分类 | ImageNet, COCO | 高精度，分层特征 |
| 目标检测 | COCO, LVIS | 多尺度检测 |
| 语义分割 | Cityscapes, ADE20K | 细粒度分割 |
| 实例分割 | COCO Mask API | 精确边界 |
| 视频理解 | Kinetics | 时空建模 |
| 医学图像 | 病理、CT分割 | 精细特征 |

## 相关概念

- **ViT**：Swin的基础架构
- **Window Attention**：局部注意力机制
- **Shifted Windows**：跨窗口信息传递
- **Patch Merging**：层级间分辨率降低
- **Hierarchical Design**：金字塔结构
- **Relative Position Bias**：空间感知偏置

## 延伸阅读

- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., Microsoft Research, 2021)
- Swin Transformer V2: Scaling Up Capacity and Resolution (Liu et al., 2022)
- MViT: Multiscale Vision Transformers (Li et al., Facebook, 2021)
- CSWin Transformer: A General Vision Transformer with CSWin Attention (Dou et al., 2022)
