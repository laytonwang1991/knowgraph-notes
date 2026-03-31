---
title: ControlNet控制网络
alias: ControlNet
tags:
  - AI
  - 深度学习
  - 生成模型
  - 条件生成
category: 生成模型
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: ControlNet通过附加条件控制信号，实现对扩散模型生成过程的精确控制。
mastery: 0
rating: 0
related_concepts:
  - 扩散模型
  - Stable Diffusion
  - 条件生成
  - 注意力机制
  - Canny边缘检测
difficulty: 进阶
read_time: 12分钟
prerequisites:
  - 扩散模型基础
  - 神经网络基础
  - 图像处理基础
---

# ControlNet控制网络

## 一句话定义

> ControlNet通过复制扩散模型的编码器并添加可学习的条件映射，实现了对图像生成的精确结构控制。

## 核心公式

### ControlNet 前向传播

$$
\mathcal{Y}_c = \mathcal{F}(x, \theta) + \mathcal{Z}(c, \theta_z)
$$

其中：
- $\mathcal{F}$：冻结的原始扩散模型
- $\mathcal{Z}$：ControlNet的零卷积层
- $c$：条件输入（如边缘图、姿态等）

### 零卷积初始化

$$
\mathcal{Z}(x, \theta_z) = 0 \cdot x + b, \quad b = 0
$$

在训练初期，零卷积确保ControlNet不影响原始模型。

### 联合损失

$$
\mathcal{L} = \mathbb{E}_{x_0, c, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]
$$

## 详细说明

### 1. ControlNet 的核心思想

**问题：** 扩散模型生成的图像具有随机性，难以精确控制结构。

**解决方案：** 将条件信息直接注入去噪过程。

```
                    条件 c（边缘/姿态/深度...）
                        ↓
原始扩散模型 ───────────────────────────┐
    │                                      │
    ↓                                      ↓
冻结编码器                                  ↓
    ↓                                      ↓
中间特征 ──→ ControlNet ──→ 与主分支融合 ──→ 去噪输出
```

### 2. 架构设计

**双分支结构：**
- **原始分支（Locked）：** 冻结预训练扩散模型权重
- **ControlNet分支（Trainable）：** 学习条件控制

**零卷积层：**
- 1x1卷积，权重初始化为0
- 训练初期输出为零，不影响主分支
- 逐渐学习到有意义的控制信号

### 3. 支持的条件类型

| 条件类型 | 输入 | 应用场景 |
|----------|------|----------|
| Canny边缘 | 边缘图 | 精确轮廓控制 |
| 深度图 | Midas/Adabins | 空间结构控制 |
| 姿态 | OpenPose | 人体姿态控制 |
| 涂鸦 | 手绘草图 | 自由形状控制 |
| 法线图 | 法线预测 | 3D结构控制 |
| 语义分割 | SEG | 区域内容控制 |
| 亮度图 | 灰度图 | 光照控制 |

### 4. 代码实现

```python
class ControlNet:
    def __init__(self, base_model):
        # 复制并冻结原始模型的编码器
        self.unet = copy.deepcopy(base_model.unet)
        for param in self.unet.parameters():
            param.requires_grad = False

        # ControlNet 额外的可训练副本
        self.controlnet = ControlNetBlock()

        # 零卷积层
        self.zero_conv = ZeroConv2d()

    def forward(self, x_t, t, cond_hint, conditioning):
        # 原始分支
        orig_feat = self.unet.encoder(x_t, t, conditioning)

        # ControlNet分支
        ctrl_feat = self.controlnet(cond_hint, t)

        # 融合特征
        controlled_feat = orig_feat + self.zero_conv(ctrl_feat)

        return self.unet.decoder(controlled_feat, t, conditioning)
```

### 5. 训练策略

**关键技巧：**
1. **锁定原始模型：** 保持生成能力
2. **零初始化：** 稳定训练开始
3. **条件 dropout：** 10%概率丢弃条件，避免依赖
4. **多条件组合：** 可同时使用多种条件

## 应用场景

- 精确边缘控制生成
- 人体姿态控制生成
- 建筑/室内设计控制
- 角色动作生成
- 图像编辑与重绘

## 相关概念

- [[扩散模型]] — ControlNet建立在扩散模型之上
- [[Stable Diffusion]] — 最常用ControlNet的底模
- [[条件生成]] — ControlNet是一种条件生成技术
- [[LoRA]] — 另一个模型微调技术，可与ControlNet结合
- [[注意力机制]] — ControlNet使用空间注意力融合条件

## 延伸阅读

- [ControlNet论文](https://arxiv.org/abs/2302.05543)
- [ControlNet官方仓库](https://github.com/lllyasviel/ControlNet)
- [ControlNet使用指南](https://stable-diffusion-art.com/controlnet/)
