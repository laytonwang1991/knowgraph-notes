---
title: Gemini API
alias: Google Gemini API使用指南
tags: [Gemini, Google, API, 多模态, Vertex AI, Gemini Pro, Gemini Ultra]
category: AI工具与平台
created: 2026-03-31
updated: 2026-03-31
author: AI开发者
description: Google DeepMind开发的Gemini系列多模态大模型API，支持文本、图像、视频、音频的统一处理，深度集成Google Cloud生态。
mastery: 7
rating: 8
related_concepts: [多模态, Vertex AI, Gemini Pro, Gemini Ultra, Google Cloud, PaLM]
difficulty: 中等
read_time: 12
prerequisites: [Python/JavaScript基础, Google Cloud基础概念]
---

# Gemini API

## 一句话定义

Gemini API是Google提供的访问Gemini系列多模态大模型的接口，能够同时理解和生成文本、图像、音频、视频内容，并深度集成Google Cloud服务。

## 详细说明

### 模型系列

| 模型 | 能力 | 上下文 | 适用场景 | 部署方式 |
|------|------|--------|---------|---------|
| Gemini 2.0 Flash | 最新旗舰 | 1M | 高效通用任务 | API/Bedrock |
| Gemini 1.5 Pro | 长上下文 | 2M | 复杂分析、文档理解 | API/Bedrock/Vertex |
| Gemini 1.5 Flash | 平衡性能 | 1M | 快速响应任务 | API/Bedrock/Vertex |
| Gemini 1.0 Pro | 基础能力 | 32K | 简单任务 | API |

### 核心优势

- **超长上下文**：Gemini 1.5 Pro支持200万Token上下文
- **原生多模态**：同一模型处理文本、图像、视频、音频
- **视频理解**：直接输入视频帧序列进行分析
- **代码生成**：Gemini在多项编程基准测试中领先
- **成本效益**：Gemini 1.5 Flash性价比极高

### 集成方式

- **Gemini API**：直接通过Google AI Studio调用
- **Vertex AI**：企业级，享有Google Cloud安全合规
- **Google Cloud Bedrock**：完全托管服务
- **Firebase**：移动应用集成

## 代码示例

### Python - 基础调用

```python
import google.genai as genai

client = genai.Client(api_key="your-api-key")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="解释什么是检索增强生成(RAG)"
)

print(response.text)
```

### Python - 多轮对话

```python
chat = client.chats.create(model="gemini-1.5-pro")

response = chat.send_message("什么是微服务架构？")
print(response.text)

response = chat.send_message("它与单体架构的主要区别是什么？")
print(response.text)
```

### Python - 多模态（图片理解）

```python
import PIL.Image

img = PIL.Image.open("chart.png")

response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[
        img,
        "分析这张图表，指出关键趋势和洞察"
    ]
)

print(response.text)
```

### Python - 流式响应

```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="详细解释Kubernetes的工作原理",
    config=genai.types.GenerateContentConfig(
        stream=True
    )
)

for chunk in response:
    print(chunk.text, end="")
```

### Python - Vertex AI（企业级）

```python
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.5-pro")

response = model.generate_content(
    prompt="解释什么是云原生架构",
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.7
    }
)

print(response.text)
```

## 应用场景

1. **长文档分析**：处理整本书籍、法律文档、科研论文
2. **视频理解**：视频摘要、内容审核、场景识别
3. **多模态内容生成**：图文配合的营销内容
4. **数据分析**：图表理解、报表自动解读
5. **移动应用**：通过Firebase集成到APP

## 相关概念

- **Token**：Gemini使用Google的Tokenizer
- **Temperature**：控制创造性，0确定性最强
- **Top P / Top K**：采样参数，控制生成多样性
- **Vertex AI**：Google Cloud的企业AI平台
- **AI Studio**：Google的AI开发和测试环境

## 延伸阅读

- [Gemini API文档](https://ai.google.dev/docs)
- [Google AI Studio](https://aistudio.google.com)
- [Vertex AI Gemini](https://cloud.google.com/vertex-ai/generative-ai)
- [Gemini模型比较](https://ai.google.dev/models/gemini)
