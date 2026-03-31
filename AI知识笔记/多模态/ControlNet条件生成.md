---
title: ControlNet条件生成
alias: ControlNet Conditional Generation
tags:
  - 可控图像生成
  - 扩散模型
  - 条件控制
  - SD插件
  - 姿态控制
  - 边缘检测
  - 深度估计
category: 多模态
created: 2026-03-31
updated: 2026-03-31
author: AI知识库
description: ControlNet通过引入条件分支网络实现对扩散模型生成过程的精确控制，无需重新训练底层模型即可实现姿态、边缘、深度等多种条件的可控图像生成。
mastery: 4
rating: 9
related_concepts:
  - Stable Diffusion
  - 条件扩散模型
  - 姿态估计
  - 边缘检测
  - 深度估计
  - 分割图
difficulty: 中高级
read_time: 25
prerequisites:
  - 扩散模型基础
  - Stable Diffusion原理
  - 图像分割基础
---

# ControlNet条件生成

## 一句话定义

ControlNet是一种神经网络结构，通过添加可训练的条件分支网络来控制预训练扩散模型的生成过程，实现无需重新训练基础模型即可完成多种条件的可控图像生成。

## 详细说明

### 1. 核心原理

ControlNet的核心思想是将预训练模型的权重复制一份作为"锁定副本"，同时添加一个"可训练副本"作为条件输入的编码器。这种设计有两个关键优势：

- **保留先验知识**：锁定副本保持预训练权重不变，确保模型仍然具备强大的生成能力
- **学习新条件**：可训练副本专门学习条件信息的特征表示，实现灵活的条件控制

具体来说，ControlNet将条件图像（如边缘图、姿态图、深度图等）通过一个独立的编码器网络处理，然后将处理后的条件特征通过零卷积层（zero convolution）注入到扩散模型的各层中。零卷积层在训练初期权重为零，随着训练逐渐学习到正确的特征融合方式。

### 2. 条件类型

ControlNet支持多种条件控制，主要包括：

| 条件类型 | 输入 | 控制方式 | 典型应用 |
|---------|------|---------|---------|
| Canny边缘 | 二值边缘图 | 边缘轮廓 | 精确轮廓生成 |
| 姿态(OpenPose) | 人体骨骼图 | 骨骼关键点 | 人物动作控制 |
| 深度(Depth) | 深度图 | 空间深度信息 | 景深控制 |
| 法线(Normal) | 法线图 | 表面朝向 | 3D结构控制 |
| 语义分割(Seg) | 分割掩码 | 区域语义 | 布局控制 |
| 线稿(Scribble) | 自由线稿 | 线条轮廓 | 草图生成 |
| 霍夫线(Hough) | 霍夫线检测 | 建筑线条 | 建筑渲染 |

### 3. 训练策略

ControlNet的训练采用两阶段策略：

**阶段一：锁定分支冻结**
- 保持原始SD模型所有权重冻结
- 仅训练条件编码器分支
- 使用条件图像-目标图像配对数据
- 训练损失主要来自条件分支的快速收敛

**阶段二：联合微调**
- 解锁SD模型最后几层进行微调
- 条件分支与主分支协同优化
- 使用低学习率进行稳定微调

### 4. 与其他方法的对比

```
┌─────────────────────────────────────────────────────────────┐
│                      条件控制方法对比                         │
├──────────────┬───────────────┬──────────────┬───────────────┤
│    方法       │   训练成本     │   控制精度   │   通用性       │
├──────────────┼───────────────┼──────────────┼───────────────┤
│ Text Inversion│   低         │    中       │    中         │
│ DreamBooth    │   高         │    高       │    高         │
│ LoRA          │   中         │    中       │    中         │
│ ControlNet    │   中         │    高       │    高         │
└──────────────┴───────────────┴──────────────┴───────────────┘
```

## 代码示例

### 1. 使用diffusers库调用ControlNet

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler
import torch
from PIL import Image
import numpy as np
import cv2

# 加载ControlNet模型（以Canny边缘为例）
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# 创建pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# GPU加速
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()  # 降低显存占用

# 准备条件图像（边缘检测）
def get_canny_image(image_path, low_threshold=100, high_threshold=200):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

# 生成图像
condition_image = get_canny_image("input_photo.jpg")
prompt = "a beautiful woman wearing a dress, professional photography"
negative_prompt = "low quality, blurry, distorted"

image = pipe(
    prompt=prompt,
    image=condition_image,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0
).images[0]

image.save("output.png")
```

### 2. 多ControlNet联合控制

```python
from diffusers import MultiControlNetModel

# 加载多个ControlNet
canny_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)
depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)

# 组合多个ControlNet
controlnet = MultiControlNetModel([canny_controlnet, depth_controlnet])

# 创建pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# 准备多条件图像
canny_image = get_canny_image("input.jpg")
depth_image = Image.open("depth_map.png").convert("RGB")
control_images = [canny_image, depth_image]

# 设置各ControlNet的权重
controlnet_params = {
    "canny_image": 0.8,
    "depth_image": 1.0
}

image = pipe(
    prompt="modern living room with large windows",
    image=control_images,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]
```

### 3. 自定义ControlNet训练

```python
import torch
from torch import nn
from diffusers import DDPMScheduler, UNet2DConditionModel
from controlnet_aux import HEDdetector, OpenPoseDetector

class ControlNetTrainer:
    def __init__(self, unet: UNet2DConditionModel, device="cuda"):
        self.unet = unet
        self.device = device
        # 复制UNet作为ControlNet分支
        self.controlnet = self._build_controlnet_branch()
        # 零卷积层用于特征注入
        self.zero_convs = self._build_zero_convs()

    def _build_controlnet_branch(self):
        """构建ControlNet的条件编码器分支"""
        controlnet = nn.ModuleList([])
        for layer in self.unet.down_blocks:
            controlnet.append(nn.Sequential(*layer.children()))
        return controlnet

    def _build_zero_convs(self):
        """构建零卷积层"""
        zero_convs = nn.ModuleList([
            nn.Conv2d(320, 320, kernel_size=1),
            nn.Conv2d(320, 320, kernel_size=1),
            nn.Conv2d(640, 640, kernel_size=1),
            nn.Conv2d(640, 640, kernel_size=1),
            nn.Conv2d(1280, 1280, kernel_size=1),
        ])
        # 初始化为零
        for conv in zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
        return zero_convs

    def forward(self, sample, timestep, control_cond, scale=1.0):
        """前向传播：注入条件信息"""
        for i, (down_block, zero_conv) in enumerate(
            zip(self.controlnet, self.zero_convs)
        ):
            sample = down_block(sample, timestep)
            # 零卷积注入条件特征
            sample = sample + scale * zero_conv(control_cond[i])
        return sample

    def training_step(self, batch, optimizer):
        """训练步骤"""
        clean_images = batch["images"].to(self.device)
        conditioning_images = batch["condition_images"].to(self.device)
        prompts = batch["prompts"]

        # 添加噪声
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],), device=self.device
        )
        noisy_images = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps
        )

        # 获取噪声预测
        noise_pred = self.unet(
            noisy_images, timesteps, encoder_hidden_states=prompts
        ).sample

        # 计算损失
        loss = nn.functional.mse_loss(noise_pred, noise)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss
```

## 应用场景

### 1. 人物姿态控制生成
- 根据参考姿态图生成具有相同姿势的人物图像
- 用于虚拟试衣、动作插画、动画制作
- 保持人物身份特征的姿态迁移

### 2. 建筑与室内设计
- 根据线稿图生成逼真的建筑渲染图
- 从深度图生成具有空间感的效果图
- 快速迭代设计方案的可视化

### 3. 动漫与游戏资产
- 将线稿自动上色和渲染
- 根据姿态图生成角色立绘
- 场景背景的批量生成

### 4. 图像编辑与修复
- 保持原图结构的同时改变风格
- 局部区域的精确修改
- 无损放大与细节增强

### 5. 跨域图像转换
- 边缘图到真实图像
- 分割图到渲染图
- 深度图到3D视角

## 相关概念

| 概念 | 说明 |
|------|------|
| **Stable Diffusion** | 基于潜在扩散的文生图模型，是ControlNet的主要应用基座 |
| **条件扩散模型** | 在扩散过程中引入额外条件信息的扩散模型变体 |
| **零卷积层** | 权重初始化为零的卷积层，用于渐进式特征融合 |
| **姿态估计** | 从图像中检测人体骨骼关键点的技术，如OpenPose |
| **边缘检测** | 提取图像边缘轮廓的图像处理技术，如Canny算法 |
| **深度估计** | 预测图像中每个像素深度值的技术，如MiDaS |
| **ControlNet模型库** | lllyasviel维护的多种ControlNet预训练模型集合 |
| **controlnet-aux** | ControlNet辅助工具库，提供多种预处理模型 |

## 延伸阅读

### 官方资源
- **论文**：[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- **项目主页**：[lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- **Hugging Face模型库**：[lllyasviel/sd-controlnet-collection](https://huggingface.co/lllyasviel/sd-controlnet-collection)
- **ControlNet App**：[Stability AI的ControlNet在线体验](https://stability.ai/controlnet)

### 技术博客
- [ControlNet详解：条件控制扩散模型生成](https://stable-diffusion-art.com/controlnet/)
- [ControlNet训练教程：从零开始微调](https://github.com/lllyasviel/ControlNet#train-your-own-controlnet)
- [多ControlNet联合使用技巧](https://stable-diffusion-art.com/multiple-controlnets/)

### 开源工具
- [controlnet-aux](https://github.com/lllyasviel/controlnet-aux)：ControlNet预处理工具集
- [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)：Automatic1111 WebUI插件
- [ComfyUI ControlNet](https://github.com/Fannovel16/comfyui_controlnet_aux)：ComfyUI节点
