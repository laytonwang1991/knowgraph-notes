---
title: Stability AI
alias: stability-ai
tags:
  - AI厂商
  - 图像生成
  - 开源
  - Stability AI
category: AI厂商与产品
created: 2026-03-31
updated: 2026-03-31
author:
description: Stability AI是开源图像生成领域的领导者，开发了Stable Diffusion系列模型
mastery: ""
rating: ""
related_concepts:
  - Stable Diffusion
  - 图像生成
  - 开源模型
  - 文生图
  - DreamStudio
difficulty: 中级
read_time: ""
prerequisites:
  - 了解基础AI/ML概念
  - 了解图像生成基本原理
---

# Stability AI

## 一句话定义

Stability AI 是开源图像生成领域的领导者，通过开发 Stable Diffusion 系列模型降低了 AI 图像生成的技术门槛。

## 详细说明

### 1. 公司概况

- **成立时间**：2020年，总部位于英国伦敦
- **创始人**：Emad Mostaque（前对冲基金分析师）
- **核心使命**："开放源码的 AI 力量，让人类创造力民主化"
- **融资规模**：累计融资超过 1 亿美元，估值达 40 亿美元（2023年峰值）

### 2. 核心技术与产品

#### Stable Diffusion 系列

| 版本 | 发布年份 | 关键改进 |
|------|----------|----------|
| SD 1.5 | 2022年 | 社区最广泛使用的版本 |
| SD 2.0 | 2022年末 | 更高分辨率、更强语义理解 |
| SDXL | 2023年 | 1024x1024原生输出，质量大幅提升 |
| SD 3.x | 2024年 | 架构改进，文字渲染能力增强 |

#### DreamStudio

- 公司官方托管的 AI 图像生成平台
- 基于 Stable Diffusion 模型
- 提供 API 访问和商业化服务

### 3. 开源社区贡献

- **模型开放**：所有 Stable Diffusion 模型权重公开发布
- **生态系统**：催生了 ComfyUI、Automatic1111 WebUI 等大量社区工具
- **许可争议**：部分模型采用非完全开源许可（如 RC 许可证），引发社区讨论

### 4. 商业模式

1. **API 服务**：通过 DreamStudio API 收费
2. **企业授权**：为企业提供定制化模型和服务
3. **订阅服务**：个人用户订阅计划
4. **云计算合作**：与 AWS、Azure 等合作提供模型部署

### 5. 争议与挑战

- 训练数据版权争议（使用未经授权的艺术作品）
- 管理层动荡（CEO 辞职、内部治理问题）
- 商业化压力与开源理念的张力
- 财务状况不稳定（2024年传出裁员消息）

## 代码示例

### 使用 Stability AI API 生成图像

```python
import stability_sdk
from stability_sdk.api import GenerationRequest

# 初始化客户端
client = stability_sdk.Client(api_key="YOUR_API_KEY")

# 文本生成图像
response = client.generate(
    prompt="A beautiful sunset over mountain landscape, digital art",
    model="stable-diffusion-xl-1024-v1-0",
    width=1024,
    height=1024,
    steps=30,
    guidance_scale=7.5
)

# 保存结果
for artifact in response.artifacts:
    if artifact.type == GenerationRequest.ArtifactType.IMAGE:
        with open(f"generated_{artifact.id}.png", "wb") as f:
            f.write(artifact.binary)
```

### 使用开源模型（本地部署）

```python
# 使用 Hugging Face Diffusers 加载 Stable Diffusion
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut_riding_horse.png")
```

### ComfyUI 工作流配置

```json
{
  "nodes": [
    {
      "id": 1,
      "type": "CLIPTextEncode",
      "widgets": {
        "prompt": "masterpiece, best quality, highly detailed"
      }
    },
    {
      "id": 2,
      "type": "CLIPTextEncode",
      "widgets": {
        "prompt": "low quality, worst quality"
      }
    },
    {
      "id": 3,
      "type": "KSampler",
      "widgets": {
        "seed": 42,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler"
      }
    },
    {
      "id": 4,
      "type": "VAEDecode"
    }
  ]
}
```

## 应用场景

| 场景 | 描述 |
|------|------|
| 数字艺术创作 | 艺术家使用 SD 生成灵感草图、概念设计 |
| 游戏资产生成 | 快速生成游戏道具、场景、角色概念图 |
| 广告营销 | 创意团队快速产出广告素材 |
| 图像编辑增强 | 配合 ControlNet 实现精确控制的图像编辑 |
| 电商场景 | 产品展示图、模特图的 AI 辅助生成 |

## 相关概念

- **Diffusion Model（扩散模型）**：SD 使用的底层生成技术
- **Latent Space（潜在空间）**：SD 在潜在空间进行去噪操作
- **Prompt Engineering（提示工程）**：优化文本提示以获得更好的生成效果
- **ControlNet**：控制 SD 生成结果的神经网络结构
- **LoRA**：低秩适配技术，用于模型微调和风格定制
- **DreamBooth**：个性化主题微调技术

## 延伸阅读

1. [Stability AI 官方网站](https://stability.ai/)
2. [Stable Diffusion GitHub](https://github.com/CompVis/stable-diffusion)
3. [Hugging Face Stable Diffusion 页面](https://huggingface.co/runwayml/stable-diffusion-v1-5)
4. [Stable Diffusion 原理详解](https://arxiv.org/abs/2112.10752)
5. [A1111 WebUI 社区](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
