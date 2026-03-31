---
title: Claude API
alias: Anthropic Claude API使用指南
tags: [Claude, Anthropic, API, LLM, 多模态, Haiku, Sonnet, Opus]
category: AI工具与平台
created: 2026-03-31
updated: 2026-03-31
author: AI开发者
description: Anthropic提供的Claude系列大语言模型API，支持Claude 3 Haiku/Sonnet/Opus，具备强大的推理、多模态和长上下文能力。
mastery: 8
rating: 9
related_concepts: [多模态, 长上下文, API密钥, Tokens, Haiku, Sonnet, Opus]
difficulty: 中等
read_time: 15
prerequisites: [Python/JavaScript基础, API调用概念]
---

# Claude API

## 一句话定义

Claude API是Anthropic公司提供的访问Claude系列大语言模型的编程接口，以其强大的推理能力、长上下文窗口和严格的安全对齐著称。

## 详细说明

### Claude 3系列对比

| 模型 | 速度 | 智能水平 | 上下文 | 视觉 | 成本 |
|------|------|---------|--------|------|------|
| Claude 3 Haiku | 最快 | 基础 | 200K | 支持 | 最低 |
| Claude 3 Sonnet | 快速 | 中高 | 200K | 支持 | 中等 |
| Claude 3 Opus | 较慢 | 最高 | 200K | 支持 | 较高 |

### 核心特性

- **200K上下文窗口**：可处理约15万字的长文档
- **多模态支持**：支持图片输入（PDF、PNG、JPG等）
- **超强推理**：在数学、编程、逻辑推理任务上表现优异
- **长文档分析**：可直接上传PDF进行内容理解与分析
- **低拒答率**：在保持安全性的同时减少误拒答

### API调用方式

- **REST API**：通过HTTP请求直接调用
- **官方SDK**：Python、TypeScript/JavaScript官方库
- **Bedrock集成**：通过AWS SageMaker或Bedrock访问
- **Google Cloud集成**：通过Vertex AI访问

## 代码示例

### Python - 基础对话

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "解释什么是RAG架构以及它的优势"
        }
    ]
)

print(message.content[0].text)
```

### Python - 带系统提示

```python
message = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=2048,
    system="你是一位资深架构师，擅长系统设计和云原生技术。回答要专业、简洁、有条理。",
    messages=[
        {
            "role": "user",
            "content": "请分析微服务架构的优缺点"
        }
    ]
)
```

### Python - 多模态（图片理解）

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "base64编码的图片数据..."
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这张图片的内容"
                }
            ]
        }
    ]
)
```

### Python - 流式输出

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "用Python写一个快速排序"}
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## 应用场景

1. **长文档处理**：合同审核、论文分析、书籍总结
2. **复杂推理**：数学证明、代码调试、逻辑分析
3. **多模态理解**：图表解读、UI设计评审、OCR后分析
4. **企业知识库**：基于私有文档的问答系统
5. **开发助手**：代码审查、重构建议、Bug分析

## 相关概念

- **Token**：与GPT类似，1个中文约2-3个Token
- **System Prompt**：Anthropic推荐使用`system`参数而非user消息
- **Temperature**：控制创造性，代码/分析任务通常0-0.3
- **Top P**：另一种采样策略，通常与Temperature配合
- **Haiku**：轻量级模型，适合简单任务和高频调用

## 延伸阅读

- [Anthropic官方文档](https://docs.anthropic.com/)
- [API Reference](https://docs.anthropic.com/api/reference)
- [Claude模型比较](https://docs.anthropic.com/en空能/choose-a-model)
- [Prompt设计指南](https://docs.anthropic.com/en空能/build-bet空r-prompts)
