---
title: OpenAI API
alias: OpenAI API开发指南
tags: [OpenAI, GPT, API, ChatGPT, LLM, AI开发]
category: AI工具与平台
created: 2026-03-31
updated: 2026-03-31
author: AI开发者
description: OpenAI提供的RESTful API接口，用于访问GPT系列大语言模型，支持文本生成、对话补全、函数调用等功能。
mastery: 8
rating: 9
related_concepts: [ChatGPT API, Assistants API, Function Calling, Tokens, Prompt Engineering]
difficulty: 中等
read_time: 15
prerequisites: [Python/JavaScript基础, RESTful API概念, HTTP请求]
---

# OpenAI API

## 一句话定义

OpenAI API是一套通过HTTP请求访问OpenAI大语言模型（GPT系列）的RESTful接口服务。

## 详细说明

### 核心能力

- **Chat Completions API**：对话补全接口，支持多轮对话上下文
- **Assistants API**：构建AI助手的完整框架，支持代码解释器、检索、知识库
- **Function Calling**：让模型调用外部函数/工具的能力
- **Embeddings**：文本向量化，用于相似度计算、检索
- **Fine-tuning**：微调自定义模型

### 模型版本

| 模型 | 上下文长度 | 适用场景 | 定价等级 |
|------|----------|---------|---------|
| gpt-4o | 128K | 全能旗舰 | 高 |
| gpt-4o-mini | 128K | 高性价比 | 中 |
| gpt-4-turbo | 128K | 平衡之选 | 中高 |
| gpt-3.5-turbo | 16K | 简单任务 | 低 |

### 定价与限制

- **按Token计费**：输入输出分别计费
- **RPM**（每分钟请求数）：免费账户3 RPM，付费账户可达10000 RPM
- **TPM**（每分钟Token数）：根据账户等级从60K到百万级
- **上下文窗口**：根据模型从4K到128K不等

## 代码示例

### Python - 基础对话

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一个专业助手"},
        {"role": "user", "content": "解释什么是Transformer架构"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### Python - Function Calling

```python
import json

functions = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京今天天气如何？"}],
    tools=[
        {"type": "function", "function": functions[0]}
    ],
    tool_choice="auto"
)

# 解析函数调用
tool_call = response.choices[0].message.tool_calls[0]
if tool_call.function.name == "get_weather":
    args = json.loads(tool_call.function.arguments)
    print(f"需要调用get_weather，参数: {args['city']}")
```

### JavaScript - 流式响应

```javascript
import OpenAI from 'openai';

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const stream = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: '写一个快速排序算法' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

## 应用场景

1. **智能客服**：7x24小时自动回答用户问题
2. **内容生成**：文章写作、代码生成、营销文案
3. **数据分析**：从文本中提取结构化信息
4. **教育辅导**：智能问答、作业批改
5. **开发辅助**：代码补全、Bug诊断、文档生成

## 相关概念

- **Token**：文本处理的最小单位，1个中文词约2-3个Token
- **Temperature**：控制输出随机性，0最确定，2最随机
- **System Prompt**：系统级指令，定义AI角色和行为
- **Few-shot Learning**：通过示例教会模型特定任务模式
- **RAG**：检索增强生成，结合外部知识库

## 延伸阅读

- [OpenAI官方文档](https://platform.openai.com/docs)
- [API Reference](https://platform.openai.com/api-ref)
- [Token计算器](https://platform.openai.com/tokenizer)
- [OpenAI CookBook](https://cookbook.openai.com/)
