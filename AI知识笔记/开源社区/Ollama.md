---
title: Ollama
alias: Ollama
tags:
  - AI
  - 大语言模型
  - 本地部署
  - 开源工具
  - ollama
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: Ollama是一款让你在本地轻松运行大语言模型的工具，支持多种开源模型，提供简洁的API接口。
mastery: 0
rating: 0
related_concepts:
  - llama.cpp
  - vLLM
  - LocalAI
  - OpenAI API
difficulty: 入门
read_time: 6分钟
prerequisites: []
---

# Ollama

## 一句话定义

> Ollama是一款让你在本地轻松运行大语言模型的工具，通过简单的命令即可部署和交互，支持Llama、Mistral、Gemma等多种开源模型。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发组织 | Ollama团队 |
| 首次发布 | 2023年 |
| GitHub星标 | 90,000+ |
| 当前版本 | 0.x |
| 许可证 | MIT |
| 支持平台 | macOS, Linux, Windows, Docker |

## 详细说明

### 1. 核心特性

**一键部署：**
- 无需复杂配置，一条命令即可运行模型
- 自动下载模型权重
- 支持GPU加速（CUDA）

**模型管理：**
- 内置模型库（Model Library）
- 支持自定义模型导入
- 模型版本管理

**API接口：**
```bash
# 完整REST API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "为什么天空是蓝色的？",
  "stream": false
}'
```

### 2. 常用命令

```bash
# 运行模型
ollama run llama3

# 拉取模型
ollama pull mistral

# 列出已下载模型
ollama list

# 创建自定义模型
ollama create mymodel -f Modelfile

# 查看模型信息
ollama show llama3
```

### 3. Modelfile自定义

```dockerfile
# Modelfile示例
FROM llama3

# 设置系统提示词
SYSTEM """
你是一位专业的数据科学家，擅长用Python进行数据分析。
"""

# 配置参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

### 4. 代码示例

**Python SDK：**
```python
import ollama

response = ollama.chat(
    model='llama3',
    messages=[
        {'role': 'user', 'content': '解释一下什么是量子计算'}
    ]
)
print(response['message']['content'])
```

**流式输出：**
```python
import ollama

stream = ollama.chat(
    model='llama3',
    messages=[{'role': 'user', 'content': '写一个Python快速排序'}],
    stream=True
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## 与OpenAI API对比

| 特性 | Ollama | OpenAI API |
|------|--------|------------|
| 部署方式 | 本地运行 | 云端服务 |
| 成本 | 硬件成本 | 按Token计费 |
| 数据隐私 | 完全本地 | 数据上传云端 |
| 网络依赖 | 无需联网 | 需要网络 |
| 模型选择 | 开源模型 | GPT系列 |
| 延迟 | 取决于硬件 | 依赖网络 |

## 相关概念

- [[llama.cpp]] — 纯C/C++的LLM推理引擎
- [[vLLM]] — 高性能vRAM优化的推理服务
- [[HuggingFace]] — 模型托管与分享平台

## 延伸阅读

- [Ollama官网](https://ollama.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama模型库](https://ollama.com/library)
- [Ollama中文文档](https://github.com/ollama/ollama/tree/main/docs)
