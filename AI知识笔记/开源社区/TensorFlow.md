---
title: TensorFlow
alias: TensorFlow
tags:
  - AI
  - 深度学习
  - 开源框架
  - Google
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: TensorFlow是Google开源的端到端机器学习平台，以静态计算图和Production级部署能力著称。
mastery: 0
rating: 0
related_concepts:
  - 深度学习
  - PyTorch
  - 机器学习框架
difficulty: 入门
read_time: 10分钟
prerequisites: []
---

# TensorFlow

## 一句话定义

> TensorFlow是Google开源的端到端机器学习平台，以静态计算图和强大的生产环境部署能力著称。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发组织 | Google Brain |
| 首次发布 | 2015年11月 |
| GitHub星标 | 180,000+ |
| 贡献者 | 3,500+ |
| 当前版本 | 2.x |
| 许可证 | Apache-2.0 |

## 详细说明

### 1. 核心特性

**静态计算图（Graph模式）：**
- 定义时构建计算图，执行时运行
- 性能优化好，适合大规模部署
- 支持TPU/GPU分布式训练

**动态计算图（Eager模式）：**
```python
import tensorflow as tf

# Eager模式即写即执行
x = tf.constant([1.0, 2.0])
y = tf.constant([3.0, 4.0])
z = x * y + 1.0
print(z)  # tensor([4., 9.])
```

### 2. 核心组件

| 组件 | 功能 |
|------|------|
| tf.Tensor | 多维数组，支持GPU/TPU |
| tf.keras | 高层API，简化模型构建 |
| tf.data | 高效数据输入管道 |
| tf.function | 图形化编译加速 |
| TF Lite | 移动端/边缘部署 |
| TF Serving | 生产环境模型服务 |

### 3. 代码示例

```python
import tensorflow as tf

# 使用Keras API构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 训练
model.fit(X_train, y_train, epochs=100, batch_size=32)
```

### 4. 生态版图

```
TensorFlow生态
├── 核心框架
│   ├── TensorFlow Core — 核心功能
│   ├── TensorFlow.js — 浏览器端运行
│   └── TF Enterprise — 企业级支持
├── 训练工具
│   ├── TensorBoard — 可视化训练
│   ├── TensorFlow Hub — 预训练模型
│   └── TensorFlow Dataset — 数据集
├── 部署方案
│   ├── TF Lite — 移动端/IoT
│   ├── TF.js — Web部署
│   └── TF Serving — 生产服务
└── 扩展库
    ├── TensorFlow Probability — 概率编程
    └── TensorFlow Graphics — 图形学
```

### 5. 版本演进

- **TensorFlow 1.x：** 静态计算图，API复杂
- **TensorFlow 2.x：** 默认Eager模式，Keras集成，API简化
- **TensorFlow 2.16+：** Keras 3.0，多框架支持

## 竞品对比

| 框架 | 优点 | 缺点 |
|------|------|------|
| TensorFlow | 生产成熟、TPU支持、部署完善 | 静态图较复杂、版本兼容问题 |
| PyTorch | 动态图、研究友好 | 生产部署稍弱 |
| JAX | 函数式、自动微分快 | 生态较新、学习曲线陡 |

## 相关概念

- [[深度学习]] — TensorFlow是深度学习的主流框架之一
- [[PyTorch]] — Meta的深度学习框架，与TensorFlow竞争
- [[ONNX]] — 跨框架模型格式，可与TensorFlow互转
- [[机器学习框架]] — TensorFlow属于机器学习框架范畴

## 延伸阅读

- [TensorFlow官网](https://www.tensorflow.org/)
- [TensorFlow文档](https://www.tensorflow.org/api_docs/)
- [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)
- [TensorFlow Hub](https://tfhub.dev/)
