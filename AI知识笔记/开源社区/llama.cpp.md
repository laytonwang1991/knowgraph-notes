---
title: llama.cpp
alias: llama.cpp
tags:
  - AI
  - 大语言模型
  - 量化
  - CPU推理
  - 开源工具
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: llama.cpp是纯C/C++实现的LLM推理引擎，支持多种量化方法，可在CPU和轻量级GPU上高效运行。
mastery: 0
rating: 0
related_concepts:
  - 量化技术
  - GGUF格式
  - Ollama
  - vLLM
difficulty: 进阶
read_time: 7分钟
prerequisites: []
---

# llama.cpp

## 一句话定义

> llama.cpp是纯C/C++实现的轻量级LLM推理引擎，支持模型量化技术，可在CPU和轻量级硬件上高效运行各种开源大语言模型。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发者 | Georgi Gerganov (ggerganov) |
| 首次发布 | 2023年2月 |
| GitHub星标 | 65,000+ |
| 核心语言 | C/C++ |
| 许可证 | MIT |
| 支持硬件 | CPU、GPU (Metal/CUDA/OpenCL) |

## 详细说明

### 1. 核心特性

**纯C/C++实现：**
- 无Python依赖
- 极快的初始化速度
- 跨平台支持（Windows/Linux/macOS）
- 嵌入式友好

**多硬件支持：**
```
llama.cpp后端
├── CPU — 纯CPU推理
├── Metal — Apple GPU加速
├── CUDA — NVIDIA GPU
├── OpenCL — 通用GPU
└── Vulkan — 跨平台GPU
```

**量化技术：**
| 量化级别 | 精度损失 | 内存占用 | 适用场景 |
|----------|----------|----------|----------|
| FP16 | 无 | 100% | 高质量需求 |
| Q8_0 | 极小 | 50% | 均衡之选 |
| Q6_K | 较小 | 37.5% | 资源受限 |
| Q4_K_S | 较小 | 25% | 内存紧张 |
| Q3_K_M | 中等 | 27% | 极致压缩 |
| Q2_K | 较大 | 18% | 最低资源 |

### 2. GGUF格式

GGUF（GPT-Generational Unified Format）是专为llama.cpp设计的模型格式：

```bash
# 转换HuggingFace模型为GGUF
llama-cli \
    --hf-repo myuser/mymodel \
    --hf-file model.gguf \
    -p "The meaning of life is"

# 量化模型
./quantize ./models/model-f16.gguf ./models/model-q4_k_m.gguf Q4_K_M
```

### 3. 基本使用

**编译：**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

**命令行推理：**
```bash
# 使用llama-cli
./llama-cli \
    -m ./models/llama-3-8b-q4_k_m.gguf \
    -n 256 \
    --temp 0.7 \
    -p "### User: 什么是量子计算？\n### Assistant:"
```

**服务器模式：**
```bash
# 启动REST API服务器
./llama-server \
    -m ./models/llama-3-8b-q4_k_m.gguf \
    --port 8080 \
    -c 4096

# API调用
curl http://localhost:8080/completion \
    -d '{"prompt":"为什么天空是蓝色的？","n_predict":128}'
```

### 4. 性能对比

| 方案 | 8B模型内存 | 70B模型内存 | 吞吐量 |
|------|-----------|-------------|--------|
| 原生FP16 | 16GB | 140GB | 基准 |
| llama.cpp Q4 | 5GB | 40GB | 较高 |
| vLLM | 18GB+ | 150GB+ | 最高 |
| Ollama | 依赖后端 | 依赖后端 | 中等 |

### 5. 量化原理

```
原始权重：FP16 (2字节/参数)
    ↓ Householder量化
Q4_K_M: 4.5 bit/参数 (平均)
    ├── 4bit量化核心
    ├── 2bit缩放因子
    └── k-quant技术
```

### 6. 跨平台支持

```bash
# macOS (Metal加速)
brew install llama.cpp
llama-cli -m model.gguf -p "Hello"

# Linux (CUDA加速)
./llama-cli -m model.gguf -ngl 99 -p "Hello"

# Windows (CUDA或CPU)
llama-cli.exe -m model.gguf -p "Hello"
```

## 应用场景

- **个人电脑运行LLM** — 无高端GPU也能运行大模型
- **边缘设备部署** — 嵌入式系统、IoT设备
- **快速原型验证** — 无需复杂环境配置
- **服务器端轻量推理** — 资源受限的生产环境

## 相关概念

- [[Ollama]] — 基于llama.cpp的易用本地LLM工具
- [[vLLM]] — 高性能GPU推理引擎
- [[量化技术]] — 模型压缩的核心方法
- [[GGUF格式]] — llama.cpp专用模型格式

## 延伸阅读

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [GGUF文档](https://github.com/ggerganov/llama.cpp/blob/master/gguf.md)
- [量化指南](https://github.com/ggerganov/llama.cpp/discussions/2948)
- [Model List GGUF](https://huggingface.co/models?library=gguf)
