---
title: ONNX
alias: ONNX
tags:
  - AI
  - 模型格式
  - 跨框架
  - 推理部署
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: ONNX是开放神经网络交换格式，支持不同深度学习框架之间的模型互转和跨平台部署。
mastery: 0
rating: 0
related_concepts:
  - 深度学习
  - PyTorch
  - TensorFlow
  - 模型部署
difficulty: 入门
read_time: 7分钟
prerequisites: []
---

# ONNX

## 一句话定义

> ONNX是开放神经网络交换格式（Open Neural Network Exchange），实现不同深度学习框架之间的模型互转和跨平台部署。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发组织 | Microsoft + Facebook |
| 首次发布 | 2017年9月 |
| GitHub星标 | 15,000+ |
| 贡献者 | 300+ |
| 当前版本 | 1.17+ |
| 许可证 | Apache-2.0 |

## 详细说明

### 1. 核心特性

**框架互转：**
- PyTorch -> ONNX
- TensorFlow -> ONNX
- 通用格式 -> 各推理引擎

**跨平台部署：**
- Windows、Linux、macOS
- CPU、GPU、移动端
- Web端（ONNX.js）

```python
import torch

# PyTorch模型导出为ONNX
model = MyModel()
model.eval()

# 构造示例输入
x = torch.randn(1, 3, 224, 224)

# 导出
torch.onnx.export(
    model,
    x,
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
```

### 2. 核心组件

| 组件 | 功能 |
|------|------|
| ONNX Model | 序列化模型文件 |
| ONNX Runtime | 高性能推理引擎 |
| ONNX.js | Web端推理 |
| ONNX Lite | 移动端部署 |
| ONNX TensorRT | NVIDIA加速 |

### 3. 代码示例

**Python推理：**
```python
import onnxruntime as ort
import numpy as np

# 创建推理会话
session = ort.InferenceSession("model.onnx")

# 准备输入
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 推理
outputs = session.run(None, {"input": x})
```

**JavaScript推理：**
```javascript
import * as onnx from "onnxjs";

const session = new onnx.InferenceSession();
await session.loadModel("model.onnx");

const x = new onnx.Tensor("float32", [1, 3, 224, 224], ...);
const outputs = await session.run([x]);
```

### 4. 生态版图

```
ONNX生态
├── 核心格式
│   ├── ONNX — 模型定义（Protobuf）
│   └── opset — 算子版本定义
├── 运行时
│   ├── ONNX Runtime — 核心推理引擎
│   ├── ONNX.js — Web推理
│   ├── ONNX Lite — 移动端
│   └── ONNX foredge — 边缘设备
├── 工具链
│   ├── onnxifier — 其他框架转ONNX
│   ├── onnx优化器 — 图优化
│   └── onnx查看器 — Netron可视化
└── 硬件加速
    ├── CUDA / cuDNN
    ├── TensorRT
    ├── OpenVINO — Intel
    └── Core ML — Apple
```

### 5. 导出支持

| 框架 | 支持程度 |
|------|----------|
| PyTorch | 原生支持（torch.onnx.export） |
| TensorFlow | via tf2onnx |
| Keras | via tf2onnx |
| JAX | via jax2tf |
| ONNX | 本身是格式 |

### 6. 推理引擎对比

| 引擎 | 特点 | 适用场景 |
|------|------|----------|
| ONNX Runtime | 通用、高性能 | 通用场景 |
| TensorRT | NVIDIA深度优化 | GPU部署 |
| OpenVINO | Intel CPU优化 | Intel硬件 |
| Core ML | Apple生态 | iOS/macOS |

## 相关概念

- [[深度学习]] — ONNX用于深度学习模型
- [[PyTorch]] — PyTorch支持原生导出ONNX
- [[TensorFlow]] — TensorFlow可通过tf2onnx转换
- [[模型部署]] — ONNX是模型部署的重要中间格式

## 延伸阅读

- [ONNX官网](https://onnx.ai/)
- [ONNX文档](https://onnx.ai/onnx/)
- [ONNX GitHub](https://github.com/onnx/onnx)
- [ONNX Runtime文档](https://onnxruntime.ai/docs/)
- [Netron模型可视化](https://netron.app/)
