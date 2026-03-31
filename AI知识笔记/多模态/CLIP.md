---
title: CLIP
category: AI
subcategory: 多模态
tags: [CLIP, 对比学习, 视觉语言, 预训练, 零样本]
created: 2026-03-31
updated: 2026-03-31
abstract: CLIP（Contrastive Language-Image Pre-training）是OpenAI提出的基于自然语言监督信号进行大规模视觉-语言对比学习的预训练框架，实现了强大的零样本图像分类和跨模态检索能力。
---

# CLIP

## 一句话定义

CLIP（Contrastive Language-Image Pre-training）是OpenAI于2021年提出的基于40亿图文对进行大规模对比学习的视觉-语言双塔模型，通过将图像和文本映射到统一语义空间实现零样本图像分类和跨模态检索。

## 核心公式与技术要点

### 1. 对比学习目标

**对称对比损失（Symmetric Contrastive Loss）：**
$$
\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_{\text{img→text}} + \mathcal{L}_{\text{text→img}}\right)
$$

**图像到文本损失：**
$$
\mathcal{L}_{\text{img→text}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp\left(\text{sim}(I_i, T_i)/\tau\right)}{\sum_{j=1}^{N}\exp\left(\text{sim}(I_i, T_j)/\tau\right)}
$$

**温度参数优化：**
$$
\tau^* = \underset{\tau}{\text{argmin}} \; \mathcal{L}_{\text{contrastive}}
$$

### 2. 视觉编码器（ViT）

**图像Patch嵌入：**
$$
z_I = \text{ViT}\left(\text{Conv2d}\left(I\right)\right) = \text{Transformer}\left(\left[\text{CLS}; \text{Patch}_1E; \ldots; \text{Patch}_NE\right]\right)
$$

**Logit缩放因子：**
$$
\text{logits} = \frac{I \cdot T^\top}{\sqrt{d}} \cdot e^{\alpha}
$$

### 3. 零样本分类

**Prompt模板化：**
$$
\hat{y} = \underset{y}{\text{argmax}} \; \sum_{c \in \text{classes}} \mathbb{1}_{c} \cdot p\left(\text{class}_c \,|\, I\right)
$$

**软提示合成：**
$$
t_c = \text{TextEncoder}\left(\text{PromptTemplate}(c)\right), \quad \text{zero-shot} = \frac{\exp\left(\text{sim}(I_{\text{feat}}, t_c)/\tau\right)}{\sum_{c'}\exp\left(\text{sim}(I_{\text{feat}}, t_{c'})/\tau\right)}
$$

### 4. 关键技术要点

- **双塔架构**：图像编码器（ViT/ResNet）+ 文本编码器（Transformer）
- **大规模预训练**：400M图文对，涵盖多种视觉概念
- **自然语言监督**：文本描述提供丰富语义监督信号
- **对比学习范式**：InfoNCE loss最大化正样本相似度
- **Prompt Engineering**：集成式提示模板提升零样本迁移

## 详细说明

CLIP是视觉-语言预训练领域的里程碑工作，首次证明了自然语言监督信号可以大规模训练出泛化能力极强的视觉模型。

### 核心创新

#### 1. 预训练范式革新

传统视觉模型：
- ImageNet预训练 + ImageNet分类头微调
- 需要人工标注的类别标签
- 泛化能力受限

CLIP预训练：
- 文本描述作为监督信号
- 开放词汇分类
- 强大的零样本迁移能力

#### 2. 双塔架构设计

```
[图像] → Image Encoder (ViT) → I_feat
                                      ↓
         Contrastive Learning → 相似度矩阵
                                      ↑
[文本] → Text Encoder (Transformer) → T_feat
```

#### 3. Prompt Ensemble

CLIP在推理时使用80个prompt模板集成：
```python
prompts = ["a photo of a {}", "an image of a {}", ...]
text_features = [encode(p.format(class)) for p in prompts]
text_features = mean(text_features)
```

### 训练数据

| 数据集规模 | 描述 |
|------------|------|
| WIT (WebImageText) | 4亿图文对 |
| 多样性覆盖 | 互联网各种图像-文本配对 |
| 领域分布 | 涵盖1000多个ImageNet类别 |

### 实验结果

| 任务 | CLIP表现 | 基线对比 |
|------|----------|----------|
| ImageNet零样本 | 76.2% | ResNet50: 14.1% |
| 零样本迁移 | 16/26数据集SOTA | 监督学习SOTA |
| 分布外检测 | 显著优于CLIP-free | 接近人类水平 |

### 应用场景

1. **零样本图像分类**：无需微调直接分类
2. **跨模态检索**：图文互检
3. **文本到图像检索**：以文搜图
4. **图像到文本检索**：以图搜文
5. **图像生成条件**：DALL-E、Stable Diffusion使用CLIP作为文本编码器
6. **开放词汇检测**：OWL-ViT等

### 局限性

- **分辨率限制**：ViT-L/14固定为224x224
- **文本理解深度**：文本编码器能力有限
- **长文本处理**：文本encoder对长文本建模不足
- **数字理解**：对数字类物体识别较弱
- **合成图像**：难以理解人类创造的合成图像

### 衍生模型

| 模型 | 改进方向 |
|------|----------|
| OpenCLIP | 开源复现+更大规模训练 |
| CLIP+ResNet | ConvNet架构版本 |
| ViT-L/14@336px | 高分辨率版本 |
| FLIP | 随机masking加速训练 |
| EVA-CLIP | 增强视觉表示 |

## 相关概念

- [[视觉语言模型]] — 处理图像和文本的多模态模型
- [[多模态学习]] — 跨模态信息处理的学习范式
- [[对比学习]] — 通过正负样本对比学习表示
- [[视觉Transformer]] — 图像表示学习的基础架构
- [[零样本学习]] — 未见类别上的泛化能力
- [[GPT-4V]] — OpenAI原生多模态模型
- [[LLaVA]] — 开源视觉语言模型

## 延伸阅读

- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML. https://arxiv.org/abs/2103.00020
- Ilharco, G., et al. (2021). OpenCLIP. https://github.com/mlfoundations/open_clip
- Li, J., et al. (2023). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML.
- Jia, C., et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. ICML.
- Schuhmann, C., et al. (2022). LAION-5B: An open large-scale dataset for training next generation image-text models. NeurIPS.
