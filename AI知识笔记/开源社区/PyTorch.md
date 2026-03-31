---
title: PyTorch
alias: PyTorch
tags:
  - AI
  - 深度学习
  - 开源框架
  - Facebook/Meta
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: PyTorch是Meta开源的深度学习框架，以其动态计算图和易用性成为研究领域最流行的框架。
mastery: 0
rating: 0
related_concepts:
  - 深度学习
  - TensorFlow
  - 机器学习框架
difficulty: 入门
read_time: 8分钟
prerequisites: []
---

# PyTorch

## 一句话定义

> PyTorch是一个基于Python的深度学习框架，以动态计算图和易用性著称，是当前AI研究领域最流行的框架。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发组织 | Meta AI (原Facebook AI) |
| 首次发布 | 2016年9月 |
| GitHub星标 | 82,000+ |
| 贡献者 | 3,800+ |
| 当前版本 | 2.x |
| 许可证 | BSD-3-Clause |

## 详细说明

### 1. 核心特性

**动态计算图：**
- 运行时定义计算图
- 调试友好（可直接print）
- 适合研究和小实验

**自动微分（autograd）：**
```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
z = y.sum()  # z = 1 + 4 = 5

z.backward()  # 自动计算梯度
print(x.grad)  # tensor([2., 4.])
```

### 2. 核心组件

| 组件 | 功能 |
|------|------|
| torch.Tensor | 多维数组，GPU加速 |
| torch.nn | 神经网络层与容器 |
| torch.optim | 优化器（SGD, Adam等） |
| torch.utils.data | 数据加载工具 |
| torchvision | CV工具库 |
| torchaudio | 音频工具库 |

### 3. 代码示例

```python
import torch
import torch.nn as nn

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 训练
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(100):
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 4. 生态版图

```
PyTorch生态
├── 训练框架
│   ├── PyTorch Lightning — 训练封装
│   ├── HuggingFace Transformers — NLP模型库
│   └── timm — CV预训练模型库
├── 推理部署
│   ├── TorchServe — 模型服务
│   ├── ONNX — 跨框架导出
│   └── torch.jit — TorchScript编译
└── 工具库
    ├── PyTorch Geometric — 图神经网络
    └── detectron2 — 目标检测
```

## 竞品对比

| 框架 | 优点 | 缺点 |
|------|------|------|
| PyTorch | 动态图、研究友好 | 生产部署稍弱 |
| TensorFlow | 生产成熟、TF Lite | 静态图、较复杂 |
| JAX | 函数式、自动微分 | 生态较新 |

## 相关概念

- [[深度学习]] — PyTorch是深度学习的主要框架
- [[TensorFlow]] — Google的深度学习框架

## 延伸阅读

- [PyTorch官网](https://pytorch.org/)
- [PyTorch文档](https://pytorch.org/docs/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
