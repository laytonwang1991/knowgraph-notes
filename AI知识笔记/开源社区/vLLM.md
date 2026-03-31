---
title: vLLM
alias: vLLM
tags:
  - AI
  - 大语言模型
  - 推理引擎
  - 高性能
  - vLLM
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: vLLM是由UC Berkeley开发的高效大模型推理服务引擎，采用PagedAttention技术实现高吞吐量和低显存占用。
mastery: 0
rating: 0
related_concepts:
  - PagedAttention
  - llama.cpp
  - TensorRT-LLM
  - OpenAI API
difficulty: 进阶
read_time: 8分钟
prerequisites: []
---

# vLLM

## 一句话定义

> vLLM是由UC Berkeley开发的开源LLM推理引擎，通过PagedAttention技术实现高吞吐量推理，显著降低vRAM占用，是生产环境部署大模型的首选方案之一。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发组织 | UC Berkeley (vLLM团队) |
| 首次发布 | 2023年6月 |
| GitHub星标 | 25,000+ |
| 支持模型 | Llama、Mistral、Vicuna等 |
| 许可证 | Apache-2.0 |
| 主要用户 | 各大AI实验室、云服务商 |

## 详细说明

### 1. 核心特性

**PagedAttention：**
- 受操作系统虚拟内存分页启发
- 动态分配KV Cache
- 减少内存碎片
- 支持更多并发请求

**高吞吐量：**
```bash
# vLLM吞吐量是HuggingFace Transformers的24倍
# 相比Loraserve提升3.5倍
```

**连续批处理（Continuous Batching）：**
- 动态批处理不同长度的请求
- 最大化GPU利用率
- 减少等待时间

**FlashAttention优化：**
- IO感知的精确注意力算法
- 减少HBM访问
- 加速训练和推理

### 2. 核心参数

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8b",
    tensor_parallel_size=2,        # 张量并行
    gpu_memory_utilization=0.9,    # GPU显存利用率
    max_model_len=4096,            # 最大上下文长度
    trust_remote_code=True,        # 信任远程代码
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)
```

### 3. API服务部署

```bash
# 命令行启动API服务器
vllm serve meta-llama/Llama-3-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2

# OpenAI兼容API调用
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3-8b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### 4. vRAM优化原理

```
传统方式（内存浪费）：
| Block 0 | Block 1 | Block 2 | ...  |
| KV Cache |    空   | KV Cache | 碎片 |

vLLM PagedAttention（紧凑布局）：
| Page 0 | Page 1 | Page 2 | Page 3 | 紧凑连续 |
```

### 5. 分布式推理

```python
# 多GPU张量并行
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-70b",
    tensor_parallel_size=4,  # 4张GPU
    pipeline_parallel_size=2,
    trust_remote_code=True,
)
```

## 性能对比

| 方案 | 吞吐量 | vRAM效率 | 延迟 |
|------|--------|----------|------|
| HuggingFace | 1x | 低 | 高 |
| vLLM | 24x | 高 | 低 |
| TensorRT-LLM | 20x | 高 | 最低 |

## 应用场景

- **生产级LLM服务** — 高并发、低延迟的API服务
- **大规模推理** — 需要处理大量请求的场景
- **多租户系统** — 共享GPU资源的服务平台
- **长上下文处理** — 100K+ token的超长上下文

## 相关概念

- [[PagedAttention]] — vLLM的核心技术
- [[llama.cpp]] — 轻量级CPU/GPU推理
- [[TensorRT-LLM]] — NVIDIA的高性能推理库

## 延伸阅读

- [vLLM官网](https://vllm.tech/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [PagedAttention论文](https://arxiv.org/abs/2309.06180)
- [vLLM文档](https://docs.vllm.tech/)
