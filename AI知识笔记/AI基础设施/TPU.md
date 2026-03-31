---
title: TPU（张量处理单元）
alias: Tensor Processing Unit
tags: [AI芯片, 硬件加速, Google, 深度学习]
category: AI基础设施
created: 2026-03-31
updated: 2026-03-31
author: Claude
description: Google设计的专用AI加速器芯片，通过脉动阵列架构高效执行矩阵运算，是大规模深度学习训练和推理的重要硬件基础设施。
mastery: 0
rating: 0
related_concepts: [GPU, NPU, 分布式训练, 加速器芯片]
difficulty: advanced
read_time: 25
prerequisites: [计算机体系结构, 深度学习基础, 矩阵运算]
---

# TPU（张量处理单元）

## 一句话定义

TPU（Tensor Processing Unit）是 Google 设计的专用 AI 加速器芯片，通过脉动阵列（systolic array）架构高效执行大规模矩阵运算，专门优化深度学习训练和推理中的张量计算。

## 核心公式

TPU 的核心计算单元是脉动阵列，矩阵乘法通过数据在阵列中"脉动"流动完成：

$$
Y = W \cdot X + B
$$

其中 $W$ 是权重矩阵，$X$ 是输入激活，$B$ 是偏置。TPU 的 MXA（Matrix Multiply Array）包含 $256 \times 256$ 个计算单元，可在单周期内完成 65536 次乘加运算。

## 详细说明

### 1. TPU 架构

TPU 采用专为神经网络设计的异构架构，包含以下核心组件：

- **脉动阵列（MXA）**：TPU 的核心计算引擎，由 $256 \times 256$ 个处理单元组成，遵循"数据流过硬件"的设计理念，最大化计算与内存访问的重叠
- **统一缓冲区（UB）**：大小为 96MB 的片上 SRAM，作为高速中间结果缓存，可大幅减少对外部 HBM 内存的访问
- **激活单元（AU）**：独立的激活函数计算模块，负责 ReLU、Softmax、Sigmoid 等非线性函数的计算
- **HBM 内存**：高带宽内存，提供 300 GB/s 的访问带宽，存储权重和大的中间张量

### 2. v4 / v5 版本对比

| 特性 | TPU v4 | TPU v5 |
|------|--------|--------|
| 峰值算力 | 275 TFLOPS (bfloat16) | 459 TFLOPS (bfloat16) |
| 芯片数量/Pod | 4096 芯片 | 8960 芯片 |
| Pod 总算力 | 1.1 EFLOPS | 4.1 EFLOPS |
| 互连带宽 | 2.4 Tbps (ICDC) | 4.8 Tbps |
| HBM 容量/芯片 | 32 GB | 95 GB |
| 内存带宽/芯片 | 600 GB/s | 1200 GB/s |
| 制程 | 7nm | 5nm |
| 光互联 | 3D torus 拓扑 | 光电路交换 |

**关键差异分析**：
- v5 的芯片密度提升了约 3 倍，Pod 规模扩大至 2 万+芯片级别
- v5 采用光学电路交换（OCS），解决了 v4 3D torus 在大规模下的路径寻址复杂性
- bfloat16 成为 v5 的主要精度格式，相比 fp32 节省 50% 内存和带宽

### 3. 与 GPU 比较

| 维度 | TPU | NVIDIA GPU (H100) |
|------|-----|-------------------|
| 架构 | 脉动阵列 | 流式多处理器 (SM) |
| 峰值算力 | 459 TFLOPS (bf16) | 989 TFLOPS (fp8) |
| 内存带宽 | 1200 GB/s | 3.35 TB/s |
| HBM 容量 | 95 GB | 80 GB |
| 互连 | 光互联 | NVLink/CubeMesh |
| 适用场景 | 大规模训练 | 训练 + 推理 |
| 编程生态 | JAX/TensorFlow | CUDA/cuDNN |
| 灵活性 | 较低（固定数据流） | 较高（通用并行） |

**核心权衡**：
- TPU 在超大规模训练任务（万卡级别以上）中具备更好的可扩展性和Pod间通信效率
- GPU 在小规模实验、灵活模型架构、混合精度推理场景中更加通用
- TPU 的 JAX 生态在自动微分和向量并行上具有优势

### 4. 软件生态

- **JAX**：Google 主推的函数式数值计算框架，与 TPU 深度集成
- **TensorFlow TPU**：官方 TPU 支持，Estimator 和 Keras API 均支持 TPU 部署
- **PyTorch XLA**：通过 XLA 编译器将 PyTorch 图编译到 TPU 上执行
- **MaxText**：Google 开源的 LLMA 架构实现，针对 TPU 优化

## 应用场景

1. **大规模语言模型训练**：TPU Pod 是 Google 内部训练 PaLM、Gemini 等千亿参数模型的核心基础设施
2. **搜索排序模型**：Google 搜索排名模型在 TPU 上进行日级别重训练
3. **推荐系统**：双Tower模型、Deep & Wide 等架构的在线学习
4. **计算机视觉**：Vision Transformer (ViT) 的训练和推理
5. **科学研究**：AlphaFold 2 蛋白质结构预测的推理阶段

## 相关概念

- **脉动阵列**：数据在硬件单元间有节奏地"流动"，最小化内存访问的架构模式
- **bfloat16**：Google 设计的 16 位浮点格式，保留 fp32 的指数范围，仅压缩尾数，适合深度学习
- **XLA 编译器**：将高层计算图优化并编译到 TPU/GPU/CPU 后端执行
- **模型并行**：将大模型拆分到多个 TPU 设备，包括张量并行、流水线并行
- **JAX**：函数式编程 + 自动微分的数值计算框架，天然适合 TPU 执行模型

## 延伸阅读

- [Google TPU v5 Paper](https://arxiv.org/abs/2305.14432) — TPU v5 架构详解
- [AnandTech: Google's TPU Evolution](https://www.anandtech.com/) — 历年 TPU 架构对比分析
- [JAX Official Documentation](https://jax.readthedocs.io/) — TPU 编程指南
- [MaxText Repository](https://github.com/google/maxtext) — Google 开源 LLM 实现
