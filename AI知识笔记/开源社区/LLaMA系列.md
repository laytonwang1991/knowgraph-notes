---
title: LLaMA系列
alias: LLaMA, LLaMA 1, LLaMA 2, LLaMA 3, Code Llama, Llama 3.1, Llama 3.2
tags:
  - 大语言模型
  - 开源模型
  - Meta
  - LLaMA
  - Code Llama
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: AI知识笔记
description: Meta开源的LLaMA模型族，涵盖LLaMA 1-3、Code Llama、Llama 3.1/3.2等多个版本，成为开源大模型发展的重要里程碑。
mastery: "3"
rating: "5"
related_concepts:
  - Transformer架构
  - Grouped Query Attention
  - RLHF
  - 开源大模型
difficulty: "4"
read_time: "25"
prerequisites:
  - 深度学习基础
  - Transformer架构
  - 自然语言处理基础
---

# LLaMA系列

## 一句话定义

LLaMA（Large Language Model Meta AI）是Meta开源的大语言模型族，提供从7B到70B参数规模的多种模型，成为开源大模型领域最重要的基础模型系列之一。

## 详细说明

### 1. 发展历程

#### LLaMA 1（2023年2月发布）
- 首批开源模型：7B、13B、33B、65B四种规模
- 仅开源权重，需申请后才能获取
- 在海量无标注文本上预训练
- 首次证明小规模模型经过充分训练可达到GPT-3水平

#### LLaMA 2（2023年7月发布）
- 7B、13B、34B、70B四种规模
- 首次开源可商用版本（需申请）
- 引入分组查询注意力机制（Grouped Query Attention）
- 提供对话优化版本Llama 2-Chat
- 训练数据增加40%，上下文长度扩展到4096 tokens

#### LLaMA 3（2024年发布）
- 8B和70B两个主力版本
- 采用更先进的训练数据和预处理流程
- 优化了多语言能力
- 引入更长的上下文窗口支持

#### Llama 3.1（2024年7月发布）
- 8B、70B、405B三种规模
- 405B版本采用稀疏MoE架构
- 支持128K超长上下文
- 增强的多语言理解和生成能力
- 整体性能逼近GPT-4

#### Llama 3.2（2024年9月发布）
- 11B和90B两个版本
- 引入视觉版本（支持图像理解）
- 轻量化版本：1B和3B（适合边缘设备）
- 多模态能力的重大升级

### 2. Code Llama

专门针对代码任务优化的LLaMA变体：
- 基于LLaMA 2训练的代码专用模型
- 支持7B、13B、34B、70B多种规模
- 支持高达100,000 tokens的上下文
- 支持Python、C++、Java、JavaScript等多种语言
- 提供Instruct版本，专门优化代码生成和解释任务

### 3. 技术架构

LLaMA系列采用的标准架构组件：

| 组件 | 说明 |
|------|------|
| Transformer | 基础模型架构 |
| RMSNorm | 归一化层替代LayerNorm |
| SwiGLU | 激活函数替代ReLU |
| RoPE | 旋转位置编码，支持更长上下文 |
| GQA | 分组查询注意力（部分版本） |

### 4. 模型规模对比

| 模型 | 参数量 | 上下文 | 备注 |
|------|--------|--------|------|
| LLaMA 1-7B | 7B | 2048 | 最小开源版本 |
| LLaMA 1-13B | 13B | 2048 | 中等规模 |
| LLaMA 1-65B | 65B | 2048 | 最大LLaMA 1 |
| LLaMA 2-70B | 70B | 4096 | 可商用 |
| Llama 3-70B | 70B | 8192 | 训练数据更优 |
| Llama 3.1-405B | 405B | 128K | 稀疏MoE架构 |
| Code Llama-34B | 34B | 100K | 代码专用 |

## 代码示例

### 使用transformers加载LLaMA模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 对话生成
messages = [
    {"role": "system", "content": "你是一个专业的AI助手。"},
    {"role": "user", "content": "解释一下什么是大语言模型。"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 使用Ollama本地运行LLaMA

```bash
# 拉取模型
ollama pull llama3.1:8b

# 运行对话
ollama run llama3.1:8b "用Python写一个快速排序算法"

# API服务
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "解释Transformer的工作原理"
}'
```

### 使用vLLM加速推理

```python
from vllm import LLM, SamplingParams

# 初始化vLLM引擎
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

# 批量推理
prompts = [
    "什么是机器学习？",
    "Python和JavaScript有什么区别？",
    "解释一下HTTP协议"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("---")
```

## 应用场景

### 1. 学术研究
- 作为研究baseline对比
- 微调实验平台
- 新算法验证

### 2. 企业应用
- 客户服务对话系统
- 内部知识库问答
- 文档处理与总结
- 代码开发辅助

### 3. 个人开发者
- 本地部署私人AI助手
- 嵌入式应用开发
- 原型快速验证

### 4. 多模态应用（Llama 3.2）
- 图像描述与理解
- 视觉问答
- 图文内容生成

## 相关概念

- **Transformer**: LLaMA的基础架构，所有变体都基于Transformerdecoder结构
- **RLHF**: 人类反馈强化学习，用于训练Llama 2-Chat等对话模型
- **Grouped Query Attention (GQA)**: Llama 2引入的注意力机制变体，提升推理效率
- **RoPE**: 旋转位置编码，LLaMA使用的位置编码方式，支持更长上下文
- **LoRA**: 低秩适配，常用于LLaMA的高效微调
- **量化技术**: GPTQ、AWQ、GGUF等用于压缩模型体积

## 延伸阅读

- [LLaMA 1论文: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [LLaMA 2论文: Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Llama 3论文: The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Code Llama论文: Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950)
- [Hugging Face LLaMA模型库](https://huggingface.co/models?search=llama)
- [Llama Official Website](https://llama.meta.com/)
