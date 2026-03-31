---
title: GPU架构
alias: GPU Architecture
tags:
  - GPU
  - CUDA
  - 硬件架构
  - NVIDIA
category: 硬件与训练
created: 2026-03-31
updated: 2026-03-31
author: AI知识笔记
description: 深入解析NVIDIA GPU架构，包括CUDA核心、Tensor Core、显存带宽及与CPU的对比分析
mastery: 4
rating: 5
related_concepts:
  - CUDA编程
  - 深度学习加速
  - GPU内存层次结构
  - Tensor Core
difficulty: 4
read_time: 25
prerequisites:
  - 计算机组成原理基础
  - 并行计算概念
  - 深度学习基础
---

# GPU架构

## 一句话定义

GPU（图形处理单元）是一种专为高并发浮点计算设计的处理器架构，最初用于图形渲染，现已成为深度学习训练的核心加速硬件。

## 核心公式

### 峰值算力公式

$$FLOPS_{peak} = N_{SM} \times N_{Cores/SM} \times F_{clock} \times 2 \times F_{ops}}$$

其中：
- $N_{SM}$：流多处理器数量
- $N_{Cores/SM}$：每个SM的CUDA核心数
- $F_{clock}$：GPU时钟频率
- $F_{ops}$：每周期浮点运算数（融合乘加计为2次）

### 显存带宽公式

$$BW_{global} = Clock_{memory} \times Bus_{width} \times 2 \div 8 \quad (GB/s)$$

### 计算密度

$$Compute\_Density = \frac{FLOPS}{BW} = \frac{N_{cores} \times IPC}{Memory\_Bandwidth}$$

---

## 详细说明

### 1. GPU与CPU架构对比

| 特性 | CPU | GPU |
|------|-----|-----|
| 设计目标 | 低延迟、单线程性能 | 高吞吐、并行计算 |
| 核心数 | 4-128核 | 数千-数万核心 |
| 控制单元 | 复杂分支预测 | 简单规则结构 |
| 缓存 | 多级大容量缓存 | 小容量共享缓存 |
| 内存 | 低延迟、高带宽 | 高带宽、大容量 |
| 适用场景 | 通用计算、逻辑运算 | SIMT并行计算 |

### 2. NVIDIA GPU架构演进

#### Turing架构（2018）
- 引入RT Core光线追踪
- 引入Tensor Core（INT8/INT4加速）
- 支持GDDR6显存

#### Ampere架构（2020）
- 第三代Tensor Core
- NVLink 3.0
- PCIe 4.0
- MIG（多实例GPU）

#### Hopper架构（2022）
- 第四代Tensor Core
- Transformer Engine
- NVLink 4.0
- DPX指令（动态规划加速）

#### Blackwell架构（2024）
- 第五代Tensor Core
- 第五代NVLink
- 支持FP8精度
- 增强的Transformer Engine

### 3. CUDA核心（Streaming Processor, SP）

CUDA核心是最基本的执行单元，负责执行浮点和整数运算。

```
SM (Streaming Multiprocessor) 结构：
┌─────────────────────────────────────┐
│  Instruction Cache                   │
│  ├─ CUDA Cores × 128 (A100)         │
│  ├─ Tensor Cores × 4 (A100)         │
│  ├─ Register File (64KB)             │
│  ├─ Shared Memory (128KB)            │
│  ├─ L1/Texture Cache (192KB)        │
│  └─ Warp Scheduler × 4              │
└─────────────────────────────────────┘
```

### 4. Tensor Core详解

Tensor Core是专门为矩阵运算设计的硬件单元，是深度学习加速的核心。

#### 矩阵乘法加速原理

$$C = A \times B + C$$

其中 A、B、C 均为矩阵，Tensor Core在一个时钟周期内完成 $4 \times 4$ 矩阵运算。

#### 各代架构Tensor Core能力对比

| 架构 | 精度支持 | 每SM算力(A100 vs H100) |
|------|----------|------------------------|
| Volta | FP16, BF16 | 512 GFLOPS |
| Turing | FP16, INT8, INT4 | 512 GFLOPS |
| Ampere | FP16, BF16, TF32, INT8 | 1024 GFLOPS |
| Hopper | FP8, FP16, BF16, TF32 | 4000 GFLOPS |

### 5. 显存类型与带宽

#### HBM（High Bandwidth Memory）

GPU使用HBM技术实现超高显存带宽：

```
HBM结构：
┌──────────────┐
│  GPU Die     │
├──────────────┤
│  Stack 0 ────┼──► 1024-bit per stack
│  Stack 1 ────┼──► 1024-bit per stack
│  Stack 2 ────┼──► 1024-bit per stack
│  Stack 3 ────┼──► 1024-bit per stack
└──────────────┘
Total: 4096-bit bus width
```

#### 主流GPU显存规格

| GPU | 显存类型 | 带宽 | 容量 | 带宽/容量比 |
|-----|----------|------|------|-------------|
| A100 | HBM2e | 2TB/s | 40/80GB | 25 GB/s per GB |
| H100 | HBM3 | 3.35TB/s | 80GB | 41.9 GB/s per GB |
| H200 | HBM3e | 4.8TB/s | 141GB | 34 GB/s per GB |
| B100 | HBM3e | 8TB/s | 192GB | 41.7 GB/s per GB |

### 6. 内存层次结构

```
GPU内存层次：
┌─────────────────────────────────────┐
│  Global Memory (HBM) - 80GB        │  ← 高延迟、大容量
├─────────────────────────────────────┤
│  L2 Cache (80MB)                   │  ← 中等延迟
├─────────────────────────────────────┤
│  L1/Shared Memory (192KB/SM)       │  ← 低延迟、SM间共享
├─────────────────────────────────────┤
│  Register File (64KB/SM)           │  ← 最低延迟
└─────────────────────────────────────┘
```

---

## 代码示例

### CUDA Hello World

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Host allocation
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Device allocation
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

### 使用Tensor Core进行矩阵乘法

```python
import torch

# 启用Tensor Core (Ampere+)
torch.set_float32_matmul_precision('high')

# 创建矩阵
A = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
B = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

# 使用Tensor Core计算 (自动启用混合精度)
with torch.cuda.amp.autocast():
    C = torch.matmul(A, B)

print(f"Result shape: {C.shape}")
print(f"Device: {C.device}")
print(f"Dtype: {C.dtype}")
```

### CUDA Memory Coalescing优化

```cuda
// 不好的访问模式（跨步访问）
__global__ void badAccess(float *data, int stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float value = data[i * stride];  // 非合并访问
    }
}

// 好的访问模式（合并访问）
__global__ void goodAccess(float *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float value = data[i];  // 合并访问，连续内存
    }
}
```

---

## 应用场景

### 1. 深度学习训练
- CNN、RNN、Transformer模型训练
- 大语言模型（LLM）预训练
- 扩散模型训练

### 2. 推理部署
- TensorRT优化推理
- INT8/FP8量化推理
- 多实例GPU（MIG）

### 3. 科学计算
- CFD（计算流体动力学）
- 分子动力学模拟
- 有限元分析

### 4. 图形渲染
- 实时光线追踪
- DLSS（深度学习超采样）
- 神经渲染

---

## 相关概念

| 概念 | 说明 |
|------|------|
| CUDA | NVIDIA的并行计算平台和编程模型 |
| NVLink | GPU间高速互联技术 |
| NVSwitch | 多GPU全互联交换机 |
| MIG | 多实例GPU，物理分区 |
| cuDNN | CUDA深度神经网络库 |
| TensorRT | NVIDIA推理优化框架 |
| Warp | 32个线程为一组的执行单元 |
| SM | 流多处理器，GPU基本计算单元 |

---

## 延伸阅读

### 官方文档
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Ampere Architecture Whitepaper](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Hopper Architecture Whitepaper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

### 深入学习
- 《CUDA编程：并行程序设计实用指南》
- 《大规模并行处理器编程实战》
- [NVIDIA Deep Learning SDK Documentation](https://docs.nvidia.com/deeplearning/)

### 实践资源
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [PyTorch CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
