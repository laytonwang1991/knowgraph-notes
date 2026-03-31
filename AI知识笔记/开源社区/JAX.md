---
title: JAX
alias: JAX
tags:
  - AI
  - 深度学习
  - 高性能计算
  - Google
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: JAX是Google开发的高性能数值计算框架，以函数式编程和自动微分为核心，支持CPU/GPU/TPU。
mastery: 0
rating: 0
related_concepts:
  - 深度学习
  - 自动微分
  - NumPy
  - GPU加速
difficulty: 进阶
read_time: 10分钟
prerequisites:
  - NumPy基础
  - 函数式编程概念
---

# JAX

## 一句话定义

> JAX是Google开发的高性能数值计算框架，以纯函数式编程和精确自动微分为核心，可在CPU/GPU/TPU上运行。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发组织 | Google Brain |
| 首次发布 | 2018年12月 |
| GitHub星标 | 28,000+ |
| 贡献者 | 700+ |
| 当前版本 | 0.4.x |
| 许可证 | Apache-2.0 |

## 详细说明

### 1. 核心特性

**纯函数式设计：**
- 无副作用的函数转换
- 不可变数据结构
- 便于并行和向量化

**自动微分（grad）：**
```python
import jax
import jax.numpy as jnp

# 定义函数
def f(x):
    return x ** 2 + 3 * x + 1

# 自动求导
df = jax.grad(f)
print(df(1.0))  # 2*1 + 3 = 5

# 高阶导数
ddf = jax.grad(jax.grad(f))
print(ddf(1.0))  # 2
```

### 2. 核心转换

| 转换 | 功能 |
|------|------|
| grad | 自动求导 |
| jit | XLA编译加速 |
| vmap | 自动向量化/批处理 |
| pmap | 多设备并行 |

```python
import jax
import jax.numpy as jnp

# JIT编译加速
@jax.jit
def forward(W, x):
    return jnp.dot(W, x)

# 自动批处理
batched_forward = jax.vmap(forward, in_axes=0)

# 多设备并行
parallel_forward = jax.pmap(forward, axis_name='devices')
```

### 3. 代码示例

**神经网络训练：**
```python
import jax
import jax.numpy as jnp
from jax import random

# 初始化参数
key = random.key(0)
W = random.normal(key, (10, 5))
b = jnp.zeros(5)

# 前向传播
def predict(params, x):
    W, b = params
    return jnp.dot(x, W) + b

# 损失函数
def loss(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

# 训练步骤
grad_loss = jax.grad(loss)
params = [W, b]

for _ in range(100):
    grads = grad_loss(params, X_train, y_train)
    params = [p - 0.01 * g for p, g in zip(params, grads)]
```

### 4. 生态版图

```
JAX生态
├── 核心框架
│   ├── JAX — 核心库
│   ├── Flax — 神经网络库
│   └── Haiku — 神经网络库
├── 科学计算
│   ├── JAX-md — 分子动力学
│   ├── Jraph — 图神经网络
│   └── Oryx — 概率编程
├── 工具链
│   ├── Chex — 单元测试
│   ├── Optax — 优化器
│   └── RLax — 强化学习
└── 硬件支持
    ├── CPU — 本地运行
    ├── GPU — CUDA加速
    └── TPU — Google云TPU
```

### 5. 与NumPy对比

| 特性 | NumPy | JAX |
|------|-------|-----|
| 数据类型 | ndarray | DeviceArray |
| 设备 | CPU | CPU/GPU/TPU |
| 惰性计算 | 否 | 是 |
| 自动微分 | 否 | 是 |
| 可变性 | 支持 | 纯函数式 |

### 6. 竞品对比

| 框架 | 优点 | 缺点 |
|------|------|------|
| JAX | 高性能、TPU支持、函数式 | 学习曲线陡、生态较新 |
| PyTorch | 动态图、研究友好 | TPU支持弱 |
| TensorFlow | 生产成熟 | 静态图较复杂 |

## 相关概念

- [[深度学习]] — JAX是深度学习的框架之一
- [[自动微分]] — grad是JAX的核心功能
- [[PyTorch]] — 另一个主流深度学习框架
- [[TensorFlow]] — Google的另一个深度学习框架
- [[注意力机制]] — Transformer等模型架构

## 延伸阅读

- [JAX官网](https://jax.readthedocs.io/)
- [JAX文档](https://jax.readthedocs.io/en/latest/)
- [JAX GitHub](https://github.com/google/jax)
- [Flax文档](https://flax.readthedocs.io/)
