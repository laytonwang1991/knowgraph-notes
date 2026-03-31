---
title: ViT视觉Transformer
alias: Vision Transformer (ViT)
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
description: ViT（Vision Transformer）是将Transformer架构应用于图像分类的开创性模型，通过将图像分割为patch序列实现视觉任务与Transformer的统一。
mastery: 0
rating: 0
related_concepts:
  - Transformer
  - 计算机视觉
  - 图像分类
  - Patch Embedding
  - DeiT
difficulty: 中等
read_time: 15分钟
prerequisites:
  - Transformer
  - 卷积神经网络基础
  - 图像分类基础
---

# ViT视觉Transformer

## 一句话定义

> ViT（Vision Transformer）将图像分割为固定大小的patches，通过线性投影将每个patch映射为token，引入Class Token进行图像分类，实现了Transformer在视觉领域的首次成功应用。

## 核心公式

### Patch Embedding

对于输入图像 $x \in \mathbb{R}^{H \times W \times C}$，分割为N个patches：

$$N = \frac{H \times W}{P^2}$$

其中P为patch大小（如16x16），C为通道数（3 for RGB）。

### 线性投影

每个patch $x_p^i \in \mathbb{R}^{P^2 \cdot C}$ 通过线性投影：

$$z_0^i = E \cdot x_p^i + E_{pos}$$

其中 $E \in \mathbb{R}^{D \times (P^2 \cdot C)}$ 为投影矩阵，$E_{pos}$ 为位置嵌入。

### Class Token

$$z_0^0 = x_{class}$$

添加可学习的[CLS] token，经过Transformer编码后，其输出用于图像分类。

### Transformer编码器输出

$$z_L^0$$ 作为图像表示，输入分类头进行类别预测。

## 详细说明

### 1. Patch Embedding

**图像分割：**
- 将 $224 \times 224$ 图像分割为 $16 \times 16$ 的patches
- 共产生 $14 \times 14 = 196$ 个patches
- 每个patch展平为 $16 \times 16 \times 3 = 768$ 维向量

**投影过程：**
- 通过线性层将768维映射到D维（如768, 1024, 1280）
- 添加位置编码保留空间信息
- 可使用2D位置编码（而非1D）保留空间关系

### 2. Class Token

**作用：**
- 聚合整个图像的信息
- 类似BERT的[CLS] token
- 最终用于图像分类

**原理：**
- Self-Attention允许[CLS] token与所有patch交互
- 最后一层[CLS] token输出包含全局信息

### 3. ViT变体规模

| 模型 | Patch大小 | 层数 | d_model | 参数量 | ImageNet Top-1 |
|------|-----------|------|---------|--------|----------------|
| ViT-B/16 | 16x16 | 12 | 768 | 86M | 79.9% |
| ViT-B/32 | 32x32 | 12 | 768 | 86M | 76.5% |
| ViT-L/16 | 16x16 | 24 | 1024 | 307M | 87.1% |
| ViT-L/32 | 32x32 | 24 | 1024 | 307M | 85.4% |
| ViT-H/14 | 14x14 | 32 | 1280 | 632M | 88.6% |

### 4. DeiT (Data-efficient Image Transformers)

**核心改进：**
- 使用蒸馏token从CNN教师网络学习
- 无需大规模预训练数据（JFT-300M）
- 在ImageNet上达到88.1% Top-1

**蒸馏方法：**
- 使用Hard distillation或Soft distillation
- 教师网络可以是ResNet或ViT

## 代码示例

### ViT图像分类推理

```python
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"Predicted class: {model.config.id2label[predicted_class]}")
```

### 手动实现Patch Embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 线性投影
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 可学习分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, n_patches_h, n_patches_w]
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]

        # 添加cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embed
        return x

# 示例使用
patch_embed = PatchEmbedding()
x = torch.randn(2, 3, 224, 224)
out = patch_embed(x)
print(f"Output shape: {out.shape}")  # [2, 197, 768]
```

### 使用DeiT进行推理

```python
import torch
from transformers import DeiTImageProcessor, DeiTForImageClassification
from PIL import Image

processor = DeiTImageProcessor.from_pretrained('facebook/deit-base-patch16-224')
model = DeiTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')

image = Image.open("dog.jpg")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
print(f"Predicted: {model.config.id2label[predicted_class]}")
```

## 应用场景

| 场景 | 应用 | 优势 |
|------|------|------|
| 图像分类 | ImageNet分类 | 大规模预训练效果好 |
| 细粒度分类 | 车辆型号、鸟类识别 | patch级别特征 |
| 图像分割 | SAM等分割模型骨干 | 统一架构 |
| 视频理解 | 时空patch建模 | 跨帧注意力 |
| 图像生成 | DALL-E等模型 | patch级生成 |

## 相关概念

- **Transformer**：底层架构
- **Patch Embedding**：图像到序列的转换
- **Class Token**：分类信息聚合
- **位置编码**：空间信息注入
- **DeiT**：高效版ViT
- **MAE**：掩码自编码器用于ViT预训练

## 延伸阅读

- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., Google, 2020)
- Training data-efficient image transformers & distillation through attention (Touvron et al., Facebook, 2021)
- Masked Autoencoders Are Scalable Vision Learners (MAE, He et al., 2021)
