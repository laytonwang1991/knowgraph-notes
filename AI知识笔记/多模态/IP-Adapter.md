---
title: IP-Adapter
alias: IP-Adapter: Decoupled Image-Text Attention
tags:
  - 图文解耦
  - 注意力适配器
  - 图像提示器
  - 风格迁移
  - 构图控制
  - 多模态生成
category: 多模态
created: 2026-03-31
updated: 2026-03-31
author: AI知识库
description: IP-Adapter是一种图文解耦的注意力适配器，通过分离图像和文本的特征注入路径，实现了对扩散模型生成结果的精确图像提示控制，同时保持了原有文本提示的兼容性。
mastery: 4
rating: 8
related_concepts:
  - Stable Diffusion
  - 注意力机制
  - 图像提示
  - 风格迁移
  - 构图控制
  - 特征解耦
difficulty: 中高级
read_time: 22
prerequisites:
  - 扩散模型基础
  - 注意力机制原理
  - CLIP图像编码器
---

# IP-Adapter

## 一句话定义

IP-Adapter是一种图文解耦的注意力适配器，通过将图像特征和文本特征的注入路径分离，使扩散模型能够在不干扰原有文本生成能力的情况下，精确地根据图像提示进行可控生成。

## 详细说明

### 1. 核心原理

IP-Adapter的核心创新在于**解耦的注意力机制**。传统的图像条件注入方法（如通过交叉注意力直接替换文本特征）存在两个主要问题：

- **文本-图像干扰**：图像条件与文本条件相互竞争，导致生成结果不稳定
- **信息丢失**：简单的特征拼接无法充分利用图像中的细粒度信息

IP-Adapter通过引入独立的图像特征注入分支来解决这些问题：

```
┌─────────────────────────────────────────────────────────────┐
│                     IP-Adapter 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   文本分支                    图像分支（新增）                │
│   ┌─────┐                   ┌─────┐                         │
│   │Text │                   │Image│                         │
│   │Emb  │                   │Encoder│                       │
│   └──┬──┘                   └──┬──┘                         │
│      │                         │                             │
│      ▼                         ▼                             │
│   ┌──────┐     独立注入      ┌──────┐                        │
│   │Text  │ ──────────────→ │Image │                         │
│   │Cross │    ✗ 不混合      │Cross │                        │
│   │Attn  │                 │Attn  │                         │
│   └──┬──┘                 └──┬──┘                          │
│      │                       │                              │
│      └─────────┬─────────────┘                              │
│                ▼                                              │
│          ┌─────────┐                                         │
│          │UNet     │                                         │
│          │Decoder  │                                         │
│          └─────────┘                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 关键组件

**图像编码器分支**
- 使用与CLIP相同的图像编码器架构
- 提取图像的语义特征和风格特征
- 通过Proj模块将图像特征映射到与文本特征相同的维度空间

**解耦交叉注意力**
- 为图像特征单独构建Cross Attention层
- 图像注意力和文本注意力独立计算
- 最终将两者的输出相加融合

**渐进式特征融合**
- 在UNet的多个层级注入图像特征
- 浅层主要注入纹理、颜色等低级特征
- 深层主要注入语义、构图等高级特征

### 3. 训练策略

IP-Adapter的训练分为两个阶段：

**阶段一：特征提取器预训练**
- 冻结SD模型全部权重
- 仅训练图像编码器和Proj投影层
- 使用图文对数据进行对比学习
- 目标：让图像编码器学会提取与文本等价的语义信息

**阶段二：适配器微调**
- 冻结SD模型的交叉注意力层
- 训练新增的图像交叉注意力层
- 使用图像条件生成数据进行训练
- 目标：学会图像特征的条件注入方式

### 4. 与ControlNet的对比

| 特性 | IP-Adapter | ControlNet |
|------|------------|------------|
| **注入方式** | 额外交叉注意力层 | 特征加法注入 |
| **条件类型** | 图像语义/风格 | 几何/结构控制 |
| **参数量** | 较小（约22M） | 较大（约361M） |
| **多条件联合** | 支持 | 支持 |
| **文本兼容性** | 保持 | 保持 |
| **训练难度** | 较低 | 中等 |

## 代码示例

### 1. 基本使用

```python
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor
import torch
from PIL import Image

# 加载基础模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# 加载IP-Adapter
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)
pipe.set_ip_adapter_scale(0.7)  # 设置图像提示强度

# 准备图像提示
image_prompt = Image.open("reference_style.jpg").convert("RGB")
image_prompt = image_prompt.resize((512, 512))

# 组合文本和图像提示生成
prompt = "a cat sitting on a couch, modern living room"
negative_prompt = "low quality, blurry, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    ip_adapter_image=image_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("ip_adapter_output.png")
```

### 2. 多图像提示

```python
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)

# 多图像提示（分别控制不同方面）
content_image = Image.open("content.jpg").resize((512, 512))
style_image = Image.open("style.jpg").resize((512, 512))

# 使用图像列表：第一个控制内容，第二个控制风格
multi_images = [content_image, style_image]

prompt = "a landscape painting"
image = pipe(
    prompt=prompt,
    ip_adapter_image=multi_images,
    num_inference_steps=30,
    guidance_scale=7.5,
    # 可分别为不同图像设置权重
).images[0]
```

### 3. 风格迁移应用

```python
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw
import numpy as np

class StyleTransferPipeline:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")

        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )

    def extract_style_features(self, style_image):
        """提取风格图像的特征"""
        style_img = style_image.resize((224, 224))
        with torch.no_grad():
            features = self.pipe.image_encoder(
                transforms.ToTensor()(style_img).unsqueeze(0).to("cuda")
            )
        return features

    def transfer_style(self, content_image, style_image, strength=0.7):
        """执行风格迁移"""
        self.pipe.set_ip_adapter_scale(strength)

        image = self.pipe(
            prompt="",  # 纯图像提示，无需文本
            ip_adapter_image=style_image,
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.0,  # 文本引导关闭
        ).images[0]

        return image

    def compose_style_and_content(self, content_image, style_image,
                                   content_weight=0.3, style_weight=0.7):
        """同时控制内容和风格"""
        # 内容强度控制
        self.pipe.set_ip_adapter_scale(content_weight)
        content_result = self.pipe(
            prompt="",  # 内容图像作为内容控制
            ip_adapter_image=content_image,
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.0,
        ).images[0]

        # 风格强度控制
        self.pipe.set_ip_adapter_scale(style_weight)
        final_result = self.pipe(
            prompt="",  # 风格图像作为风格控制
            ip_adapter_image=style_image,
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.0,
        ).images[0]

        return content_result, final_result

# 使用示例
pipeline = StyleTransferPipeline()
content = Image.open("photo.jpg")
style = Image.open("artwork.jpg")

result = pipeline.transfer_style(content, style, strength=0.8)
result.save("styled_output.png")
```

### 4. 自定义IP-Adapter实现

```python
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers import UNet2DConditionModel

class IPAdapterAttention(nn.Module):
    """IP-Adapter解耦注意力处理器"""

    def __init__(self, attention: Attention, hidden_dim: int = 768):
        super().__init__()
        self.attention = attention
        # 图像特征投影层
        self.image_proj = nn.Linear(hidden_dim, hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)

    def forward(self, attn, hidden_states, encoder_hidden_states, **kwargs):
        batch_size, sequence_length, dim = hidden_states.shape

        # 分离文本和图像encoder_hidden_states
        if isinstance(encoder_hidden_states, tuple):
            text_states = encoder_hidden_states[0]
            image_states = encoder_hidden_states[1]
        else:
            # 原始行为（仅文本）
            text_states = encoder_hidden_states
            image_states = None

        # 文本注意力（原始逻辑）
        if text_states is not None:
            text_hidden_states = self.attention(
                hidden_states, text_states, **kwargs
            ).sample
        else:
            text_hidden_states = hidden_states

        # 图像注意力（新增逻辑）
        if image_states is not None:
            # 投影图像特征
            image_states = self.image_proj(image_states)
            image_states = self.image_norm(image_states)

            # 图像交叉注意力
            image_hidden_states = self.attention(
                hidden_states, image_states, **kwargs
            ).sample

            # 融合（加权求和）
            alpha = 0.7  # 图像权重
            hidden_states = (1 - alpha) * text_hidden_states + \
                            alpha * image_hidden_states
        else:
            hidden_states = text_hidden_states

        return hidden_states

class IPAdapter:
    """IP-Adapter核心实现"""

    def __init__(self, unet: UNet2DConditionModel, image_encoder):
        self.unet = unet
        self.image_encoder = image_encoder
        self.image_proj_layers = nn.ModuleList([])

        # 替换注意力处理器
        self._replace_attention_processor()

    def _replace_attention_processor(self):
        """替换UNet中的注意力处理器"""
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention):
                # 为每个注意力层创建IP-Adapter版本
                ip_attn = IPAdapterAttention(
                    module,
                    hidden_dim=module.to_k.out_features
                )
                module.set_processor(ip_attn)

    def encode_image(self, image):
        """编码图像提示"""
        with torch.no_grad():
            image_features = self.image_encoder(image).image_embeds
        return image_features

    @torch.no_grad()
    def generate(self, prompt, image_prompt, **kwargs):
        """使用图像提示生成"""
        # 编码图像
        image_emb = self.encode_image(image_prompt)

        # 组合文本和图像条件
        text_emb = self.pipe.text_encoder(prompt)
        encoder_hidden_states = (text_emb, image_emb)

        # 生成
        return self.pipe(
            prompt_embeds=encoder_hidden_states[0],
            ip_adapter_image=image_emb,
            **kwargs
        )
```

## 应用场景

### 1. 风格迁移
- 将参考艺术作品的风格应用到照片上
- 保持照片内容的同时改变艺术风格
- 支持多种风格元素的组合与混合

### 2. 构图控制
- 参考图像的构图和布局
- 保持主体位置和画面平衡
- 调整视角和景深效果

### 3. 角色一致性
- 保持角色的外观特征
- 生成不同场景和动作下的同一角色
- 用于漫画和动画制作

### 4. 产品展示
- 将产品放置在参考场景中
- 保持产品的真实外观
- 生成多样化的展示图

### 5. 概念设计
- 快速探索设计概念
- 将草图转化为渲染图
- 保持设计意图的同时丰富细节

## 相关概念

| 概念 | 说明 |
|------|------|
| **CLIP** | 对比语言-图像预训练模型，IP-Adapter使用其作为图像编码器 |
| **注意力机制** | Transformer架构的核心组件，用于融合不同模态的信息 |
| **图文解耦** | 分离处理图像和文本信息的技术，避免不同模态特征相互干扰 |
| **特征投影** | 将不同维度的特征映射到统一空间的操作 |
| **IP-Adapter Plus** | IP-Adapter的增强版本，支持更强的图像控制能力 |
| **LCM-LoRA** | 与IP-Adapter结合使用可加速生成过程 |
| **StyleID** | 针对特定艺术家风格的专用适配器 |

## 延伸阅读

### 官方资源
- **论文**：[IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)
- **项目主页**：[h94/IP-Adapter](https://github.com/h94/IP-Adapter)
- **Hugging Face**：[h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- **在线Demo**：[IP-Adapter Demo on Replicate](https://replicate.com/

### 技术博客
- [IP-Adapter详解：图文解耦的图像提示适配器](https://stable-diffusion-art.com/ip-adapter/)
- [IP-Adapter与ControlNet对比分析](https://blog.xiangshan.org/)
- [多模态图像生成中的注意力机制](https://diffusionwiki.com/)

### 开源工具
- [ComfyUI IP-Adapter](https://github.com/cubiq/ComfyUI_IPAdapter_plus)：ComfyUI插件
- [sd-webui-ipadapter](https://github.com/tencentAILab/sd-webui-ipadapter)：WebUI插件
- [IP-Adapter Plus](https://github.com/cubiq/IPAdapter_plus)：增强版实现
