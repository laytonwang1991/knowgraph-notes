---
title: NeRF神经辐射场
alias: Neural Radiance Field, 神经渲染, 3D场景重建
tags: [NeRF, 3D重建, 神经渲染, 体素渲染, 视图合成]
category: 生成模型
created: 2026-03-31
updated: 2026-03-31
author: AI
description: NeRF通过深度学习将多角度图像合成为稠密的3D场景表示，实现前所未有的视图合成质量，是计算机图形学与深度学习交叉的重要突破。
mastery: 3
rating: 8
related_concepts: [体素渲染, 位置编码, Mip-NeRF, BARF, Gaussian Splatting, 扩散模型3D]
difficulty: 4
read_time: 30
prerequisites: [相机成像原理, 神经网络基础, 体渲染积分]
---

# NeRF神经辐射场

## 一句话定义

NeRF（Neural Radiance Field）是一种利用多层感知器（MLP）将3D场景表示为连续体积密度和颜色函数的神经渲染方法，通过体素渲染（Volumetric Rendering）从任意角度合成逼真新视图。

## 核心公式

### 体素渲染方程

沿射线 $r(t) = o + t d$ 从 $t_n$ 到 $t_f$ 的渲染：

$$
C(r) = \int_{t_n}^{t_f} T(t) \cdot \sigma(r(t)) \cdot c(r(t), d) \, dt
$$

其中透射率（transmittance）：

$$
T(t) = \exp\left(-\int_{t_n}^{t} \sigma(r(s)) \, ds\right)
$$

### 离散近似（数值积分）

$$
\hat{C}(r) = \sum_{i=1}^{N} T_i \cdot (1 - \exp(-\sigma_i \delta_i)) \cdot c_i
$$

其中 $T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$，$\delta_i = t_{i+1} - t_i$。

### 位置编码（Positional Encoding）

NeRF使用高频位置编码增强MLP表达能力：

$$
\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))
$$

## 详细说明

### 1. 核心思想

NeRF的核心洞察是将3D场景表示为一个**连续的5D函数**：

$$
F_\Theta: (\mathbf{x}, d) \rightarrow (\mathbf{c}, \sigma)
$$

- $\mathbf{x} = (x, y, z)$：3D位置
- $d = (\theta, \phi)$：观察方向（相机射线方向）
- $\mathbf{c} = (R, G, B)$：颜色值
- $\sigma$：体积密度（不透明度）

### 2. 网络架构

典型MLP架构：

```
输入: (x, y, z, d) → 8层MLP(256通道) → 输出: (RGB, σ)
           ↓
    位置编码 γ(p) 在输入层应用
           ↓
    视角相关颜色：额外输入方向d到接近输出层
```

### 3. 体素渲染流程

1. 对每个像素，从相机中心发出射线
2. 在射线上采样N个点
3. 通过MLP获取每个点的颜色和密度
4. 用alpha合成（alpha compositing）累积颜色
5. 得到像素最终颜色

### 4. 关键训练技术

- **分层采样（Hierarchical Sampling）**： coarse + fine 两阶段采样，提高效率
- **位置编码**：让网络学习高频细节，避免"模糊"效应
- **射线与场景范围对齐**：裁剪到场景边界减少浪费

### 5. 主要改进方向

| 改进方向 | 代表工作 | 核心贡献 |
|----------|----------|----------|
| 训练速度 | BARF | 隐式对齐相机参数 |
| 抗锯齿 | Mip-NeRF | 消除混叠，视角连续性 |
| 更大场景 | Block-NeRF | 城市场景分块重建 |
| 动态场景 | Nerfies, HyperNeRF | 变形场处理非刚体 |
| 高效渲染 | 3D Gaussian Splatting | 光栅化替代体素积分 |
| 语义编辑 | Semantic NeRF | 语义标签解耦 |

### 6. 优缺点

**优点**：
- 视图合成质量极高，超越传统方法
- 连续3D表示，分辨率无网格限制
- 存储效率高于显式体素网格

**缺点**：
- 训练和渲染极慢（每帧需数百次MLP前向）
- 对相机姿态要求高（需COLMAP预处理）
- 难以处理遮挡和几何歧义

## 代码示例

```python
import torch
import torch.nn as nn
import math

class NeRFMLP(nn.Module):
    """简化版NeRF MLP"""
    def __init__(self, input_dim=3, freq_bands=10, hidden_dim=256):
        super().__init__()
        self.freq_bands = freq_bands
        # 位置编码
        self.input_ch = input_dim * 2 * freq_bands + input_dim
        # Coarse网络
        self.coarse = nn.Sequential(
            nn.Linear(self.input_ch, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # RGB + density
        )
        # Fine网络（更宽）
        self.fine = nn.Sequential(
            nn.Linear(self.input_ch, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def positional_encoding(self, p, L=10):
        """位置编码"""
        encoded = []
        for l in range(L):
            encoded.append(torch.sin(2**l * math.pi * p))
            encoded.append(torch.cos(2**l * math.pi * p))
        return torch.cat(encoded, dim=-1)

    def render_rays(self, rays_o, rays_d, network, N_samples=64):
        """渲染射线"""
        device = rays_o.device
        # 分层采样
        t_vals = torch.linspace(0, 1, N_samples, device=device)
        z_vals = 0.1 * (1 - t_vals) + 6.0 * t_vals  # 近远裁剪
        # 噪声扰动
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.cat([mids, z_vals[-1:]])
        lower = torch.cat([z_vals[:1], mids])
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        # 位置编码
        pts_encoded = self.positional_encoding(points)
        # MLP前向
        output = network(pts_encoded)
        sigma = torch.relu(output[..., 3])
        rgb = torch.sigmoid(output[..., :3])
        # Alpha合成
        dists = torch.cat([z_vals[1:] - z_vals[:-1], torch.tensor([1e10], device=device).expand(z_vals[..., :1].shape)])
        alpha = 1.0 - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        rgb_map = (weights[..., :-1] * rgb[..., 0, :]).sum(dim=-2)
        return rgb_map

def run_network(inputs, viewdirs, model):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = model.positional_encoding(inputs_flat)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = model.positional_encoding(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], dim=-1)
    outputs_flat = model.coarse(embedded)
    return outputs_flat.reshape(list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
```

## 应用场景

1. **电影视觉特效**：无需真实物体，通过照片即可创建可自由视角查看的3D场景
2. **文化遗产数字化**：对文物进行高质量3D存档，支持沉浸式展示
3. **机器人操作规划**：重建真实环境3D几何，辅助抓取和导航
4. **自动驾驶仿真**：合成逼真驾驶场景用于数据增强
5. **医学成像**：CT/MRI体积数据的3D可视化
6. **虚拟现实/增强现实**：从少量照片生成可交互3D内容
7. **建筑可视化**：室内场景的photorealistic新视图合成

## 相关概念

- **体素渲染（Volumetric Rendering）**：沿射线积分累积颜色和透明度，是NeRF的核心技术
- **位置编码（Positional Encoding）**：将低频坐标映射到高频空间，增强网络对细节的表达能力
- **3D Gaussian Splatting**：2023年新方法，用各向异性高斯代替MLP，实现实时渲染
- **BARF**：Bundle-Adjusting Neural Radiance Field，同时优化相机姿态和NeRF
- **Mip-NeRF**：改进抗锯齿性能的多尺度NeRF
- **神经隐式表示**：用神经网络表示3D场景 Continuous Representation
- **COLMAP**：传统多视角重建工具，用于估算相机参数

## 延伸阅读

- Mildenhall B, Srinivasan P P, Tancik M, et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020
- Barron J T, Mildenhall B, Tancik M, et al. "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radi radiance Fields", ICCV 2021
- Chen A, Xu Z, Wei X, et al. "BARF: Bundle-Adjusting Neural Radiance Fields", ICCV 2021
- Kerbl B, Kopanas G, Leimkuehler T, et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
- Park J J, Florence P, Straub J, et al. "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation", CVPR 2019
