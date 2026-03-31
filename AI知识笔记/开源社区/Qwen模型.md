---
title: Qwen模型
alias: Qwen, 通义千问, Qwen-1.5, Qwen-2, Qwen-2.5, Qwen-Max, Qwen-Coder, Qwen-Wactor
tags:
  - 大语言模型
  - 开源模型
  - 阿里云
  - 通义千问
  - 多语言模型
  - 代码模型
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: AI知识笔记
description: 阿里云通义千问系列模型，涵盖Qwen1.5到Qwen2.5全系列，具备强大的多语言能力、代码生成能力，支持文本和代码等多种模态。
mastery: "3"
rating: "5"
related_concepts:
  - MoE架构
  - 多语言预训练
  - 代码大模型
  - 长上下文
  - 开源大模型
difficulty: "4"
read_time: "25"
prerequisites:
  - 深度学习基础
  - Transformer架构
  - 自然语言处理基础
---

# Qwen模型

## 一句话定义

Qwen（通义千问）是阿里云开发的大语言模型系列，从Qwen1.5到Qwen2.5涵盖多种规模，并衍生了Qwen-Coder（代码）、Qwen-Wactor（角色扮演）等专用模型，成为开源大模型的重要力量。

## 详细说明

### 1. 发展历程

#### Qwen-1（2023年发布）
- 首批版本：Qwen-7B、Qwen-14B
- 基于Transformer架构
- 支持中英双语
- 开源基础模型权重

#### Qwen-1.5（2024年初发布）
- 多种规模：0.5B、1.8B、4B、7B、14B、72B
- 显著提升多语言能力
- 改进的指令遵循能力
- 支持32K上下文
- 多个开源量化版本

#### Qwen-2（2024年发布）
- 多种规模：0.5B、1.5B、3B、7B、15B、57B、72B
- 新增57B稀疏MoE版本（Qwen-MoE）
- 支持128K超长上下文
- 显著提升代码和数学能力
- 多语言支持扩展到100+语言

#### Qwen-2.5（2024年发布）
- 多种规模：0.5B、1.5B、3B、7B、14B、32B、72B
- 知识更新至2024年
- 进一步优化的指令遵循
- 增强的数学和推理能力
- Qwen2.5-72B-Instruct表现卓越

#### Qwen-Max（2024年发布）
- 最强闭源版本
- 持续迭代优化
- API调用方式使用

### 2. Qwen-Coder系列

专注于代码生成的专用模型：

| 模型 | 规模 | 特点 |
|------|------|------|
| Qwen-Coder-1.5-7B | 7B | 轻量级代码模型 |
| Qwen-Coder-2-7B | 7B | 支持更长的代码上下文 |
| Qwen-Coder-72B | 72B | 最强代码能力 |

**支持功能**：
- 多语言代码生成
- 代码补全和修复
- 代码解释和问答
- 代码审查建议

### 3. Qwen-Wactor

角色扮演和对话优化版本：

- 针对角色一致性优化
- 更好的对话连贯性
- 适合游戏NPC、虚拟助手等场景

### 4. 技术架构

Qwen系列采用的标准架构：

| 组件 | 说明 |
|------|------|
| Transformer Decoder | 基础架构 |
| SwiGLU激活函数 | 替代ReLU，提升性能 |
| RoPE位置编码 | 支持长上下文 |
| 注意力机制 | 采用GQA（部分大模型） |
| Tokenizer | 基于BPE的qwen tokenizer |

### 5. 模型规模对比

| 模型版本 | 参数量 | 上下文 | 多语言 | 备注 |
|----------|--------|--------|--------|------|
| Qwen2.5-0.5B | 0.5B | 32K | 100+ | 轻量边缘部署 |
| Qwen2.5-1.5B | 1.5B | 32K | 100+ | 入门级 |
| Qwen2.5-7B | 7B | 128K | 100+ | 主力开源版本 |
| Qwen2.5-14B | 14B | 128K | 100+ | 中等规模 |
| Qwen2.5-72B | 72B | 128K | 100+ | 大规模开源 |
| Qwen-MoE-57B | 57B MoE | 128K | 100+ | 稀疏MoE架构 |
| Qwen-Max | 未知 | 128K | 100+ | 最强闭源版本 |

### 6. 多语言能力

Qwen2.5经过大规模多语言数据预训练：

- **支持语言**: 100+种语言
- **优势语言**: 中文、英文、法语、西班牙语、德语、日语、韩语等
- **中文能力**: 特别优化，在中文任务上表现优异
- **代码能力**: 覆盖主流编程语言

## 代码示例

### 使用transformers加载Qwen模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载Qwen2.5-7B-Instruct
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 对话生成
messages = [
    {"role": "system", "content": "你是一个专业的Python编程助手。"},
    {"role": "user", "content": "帮我写一个函数来计算斐波那契数列第n项"}
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

### 使用vLLM部署Qwen API服务

```python
from vllm import LLM, SamplingParams

# 初始化vLLM引擎
llm = LLM(model="Qwen/Qwen2.5-14B-Instruct")

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

# 批量请求
prompts = [
    "解释什么是Python中的装饰器",
    "用Python写一个二分查找算法",
    "Python中的生成器和迭代器有什么区别"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
    print("---")
```

### 使用Ollama本地运行Qwen

```bash
# 拉取Qwen模型
ollama pull qwen2.5:14b

# 运行对话
ollama run qwen2.5:14b "用中文解释什么是大语言模型"

# API调用
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:14b",
  "prompt": "解释HTTP和HTTPS的区别",
  "stream": false
}'
```

### Qwen-Coder代码生成示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载Qwen-Coder模型
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 代码生成任务
messages = [
    {"role": "system", "content": "你是一个专业的代码助手，擅长Python、JavaScript、Java等编程语言。"},
    {"role": "user", "content": """帮我写一个Python类，实现以下功能：
1. 一个栈数据结构
2. push方法压入元素
3. pop方法弹出元素
4. peek方法查看栈顶元素
5. is_empty方法判断栈是否为空
"""}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 长上下文处理示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载支持长上下文的模型
model_name = "Qwen/Qwen2.5-7B-Instruct-1M"  # 支持百万token上下文

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 长文档处理
long_document = """
[此处放入长文档内容...]
"""

messages = [
    {"role": "user", "content": f"请总结以下文章的主要观点：\n\n{long_document}"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 应用场景

### 1. 中文AI应用开发
- 中文对话系统
- 中文内容创作辅助
- 中文知识库问答
- 中文教育应用

### 2. 多语言应用
- 跨境电商多语言客服
- 翻译和本地化
- 多语言内容审核
- 国际业务支持

### 3. 代码开发和辅助
- 代码补全和生成
- 代码审查和优化建议
- 技术文档编写
- 代码学习和教学

### 4. 企业级应用
- 智能客服系统
- 内部知识管理
- 文档处理和分析
- 数据分析和报表生成

### 5. 研究和学术
- 学术论文写作辅助
- 研究数据分析
- 代码实现验证
- 跨学科研究支持

## 相关概念

- **MoE架构**: Qwen-MoE采用稀疏混合专家架构，提升效率
- **长上下文**: Qwen2.5支持128K上下文，部分版本支持1M tokens
- **多语言预训练**: 大规模多语言数据训练，支持100+语言
- **指令微调**: SFT和RLHF等技术提升指令遵循能力
- **BPE Tokenizer**: 基于BPE的分词器，专门优化中文分词
- **Qwen-Turbo/Qwen-Max**: 不同能力级别的商业API版本

## 延伸阅读

- [Qwen官方GitHub仓库](https://github.com/QwenLM/Qwen)
- [Qwen Hugging Face页面](https://huggingface.co/Qwen)
- [Qwen技术博客](https://qwenlm.github.io/)
- [Qwen2.5技术报告](https://arxiv.org/abs/2407.21783)
- [Qwen-Coder论文](https://arxiv.org/abs/2409.12186)
- [通义千问官网](https://tongyi.aliyun.com/)
