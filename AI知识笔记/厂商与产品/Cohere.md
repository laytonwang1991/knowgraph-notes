---
title: Cohere
alias: cohere
tags:
  - AI厂商
  - 企业NLP
  - 向量嵌入
  - RAG
  - Cohere
category: AI厂商与产品
created: 2026-03-31
updated: 2026-03-31
author:
description: Cohere是企业级NLP平台，提供Command、Rerank、Embed等模型，专注于企业AI应用
mastery: ""
rating: ""
related_concepts:
  - Command模型
  - Rerank模型
  - Embed模型
  - 企业NLP
  - 向量数据库
  - RAG
difficulty: 中级
read_time: ""
prerequisites:
  - 了解NLP基础概念
  - 了解向量嵌入原理
  - 了解RAG架构
---

# Cohere

## 一句话定义

Cohere 是一家专注于企业级 AI 的公司，提供 Command（对话）、Rerank（重排序）、Embed（向量嵌入）等 NLP 模型，帮助企业构建生产级 AI 应用。

## 详细说明

### 1. 公司概况

- **成立时间**：2019年，总部位于加拿大多伦多
- **创始人**：Aidan Gomez（Transformer 论文《Attention Is All You Need》作者之一）、Ivan Zhang、Nick Frosst
- **核心定位**：为企业提供可定制、可私有部署的 LLM 解决方案
- **融资规模**：累计融资超过 4.4 亿美元，估值达 22 亿美元

### 2. 核心产品线

#### Command 系列（对话模型）

| 模型 | 参数量 | 定位 | 特点 |
|------|--------|------|------|
| Command | ~52B | 通用对话 | 专注于命令执行，适合agent应用 |
| Command Light | ~14B | 轻量对话 | 更快推理，更低成本 |
| Command R | ~104B | 企业RAG | 优化检索增强生成 |
| Command R+ | ~104B | 旗舰RAG | 更强推理，多语言支持 |

#### Embed 模型（向量嵌入）

- **embed-english-v3.0**：英语向量嵌入
- **embed-multilingual-v3.0**：支持 100+ 语言的跨语言嵌入
- **维度**：1024 维向量输出
- **用途**：语义搜索、文档聚类、相似度匹配

#### Rerank 模型

- **cohere-rerank-3.5**：重排序模型
- **工作原理**：接收初步检索结果，重新排序以提升相关性
- **应用**：搜索系统、推荐系统、RAG 管道

### 3. 企业应用优势

1. **多云部署**：支持 AWS、Azure、GCP 私有部署
2. **数据隐私**：不训练客户数据，符合 SOC 2、GDPR
3. **定制化**：支持企业专属模型微调
4. **低延迟**：全球分布的 API 节点

### 4. Command R 与 GPT-4 对比

| 维度 | Command R+ | GPT-4 |
|------|------------|-------|
| 专注场景 | RAG/工具使用 | 通用对话 |
| 上下文窗口 | 128K | 128K |
| 价格 | 显著低于 GPT-4 | 较高 |
| 检索增强 | 原生优化 | 需额外配置 |
| 工具调用 | 原生支持 | 支持 |
| 开源 | 部分模型开源 | 闭源 |

### 5. 技术特点

- **检索优化**：Command R 系列原生支持 RAG 工作流
- **工具使用**：内置 function calling 能力
- **多语言**：支持英语、中文、日语、韩语等
- **可控输出**：支持 JSON 模式、格式约束

## 代码示例

### 使用 Command R+ 进行对话

```python
from cohere import Client
import json

co = Client(api_key="YOUR_API_KEY")

# 基础对话
response = co.chat(
    model="command-r-plus",
    message="Explain RAG architecture in simple terms",
    temperature=0.7,
    max_tokens=512
)
print(response.text)

# 带检索上下文的对话（RAG）
response = co.chat(
    model="command-r-plus",
    message="What was the revenue growth?",
    temperature=0.3,
    documents=[
        {"title": "Q4 Report", "snippet": "Revenue grew 25% year over year"},
        {"title": "Product Launch", "snippet": "New product launched in March"}
    ]
)
print(response.text)
```

### 使用 Embed 模型生成向量

```python
from cohere import Client

co = Client(api_key="YOUR_API_KEY")

# 单文本嵌入
response = co.embed(
    model="embed-english-v3.0",
    texts=["What is machine learning?"],
    input_type="search_query"
)
print(f"Vector dimension: {len(response.embeddings[0])}")

# 批量文档嵌入
doc_embeddings = co.embed(
    model="embed-english-v3.0",
    texts=[
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ],
    input_type="search_document"
)

# 计算相似度
import numpy as np
query_vec = co.embed(
    model="embed-english-v3.0",
    texts=["What is AI?"],
    input_type="search_query"
).embeddings[0]

doc_vecs = doc_embeddings.embeddings
similarities = [np.dot(query_vec, doc) for doc in doc_vecs]
print(f"Most similar: {np.argmax(similarities)}")
```

### 使用 Rerank 优化搜索

```python
from cohere import Client

co = Client(api_key="YOUR_API_KEY")

# 初始搜索结果（来自Elasticsearch/Solr等）
initial_results = [
    {"id": "1", "text": "Python tutorial for beginners"},
    {"id": "2", "text": "Advanced Python programming"},
    {"id": "3", "text": "JavaScript basics"},
    {"id": "4", "text": "Python data science handbook"}
]

# Rerank 重新排序
reranked = co.rerank(
    model="rerank-3.5",
    query="Python programming tutorial",
    documents=[r["text"] for r in initial_results],
    top_n=4
)

# 输出排序结果
for result in reranked.results:
    print(f"Rank {result.index + 1}: {initial_results[result.index]['text']} (relevance: {result.relevance_score:.3f})")
```

### RAG 完整流程实现

```python
from cohere import Client
from qdrant_client import QdrantClient
import numpy as np

co = Client(api_key="YOUR_API_KEY")

# 1. 文档分块并向量化
documents = [
    "Cohere provides enterprise AI solutions including Command, Embed, and Rerank models.",
    "The Command models are optimized for dialogue and agent applications.",
    "Embed models can generate vector representations for semantic search.",
    "Rerank improves search relevance by reordering initial results."
]

# 向量化文档
embeddings = co.embed(
    model="embed-english-v3.0",
    texts=documents,
    input_type="search_document"
).embeddings

# 2. 存储到向量数据库（如 Qdrant）
# qdrant.upsert(collection_name="docs", points=embeddings)

# 3. 查询时先检索后重排
query = "What AI models does Cohere offer?"

# 向量化查询
query_emb = co.embed(
    model="embed-english-v3.0",
    texts=[query],
    input_type="search_query"
).embeddings[0]

# 向量相似度检索（获取 top-20）
# retrieved = qdrant.search(collection_name="docs", query_vector=query_emb, limit=20)

# 重排（取 top-5）
# reranked = co.rerank(model="rerank-3.5", query=query, documents=retrieved_doc_texts, top_n=5)

# 4. 使用重排结果生成回答
context = "\n".join([f"- {doc}" for doc in ["Cohere offers Command models", "Cohere offers Embed models"]])  # 简化示例
response = co.chat(
    model="command-r-plus",
    message=query,
    documents=[{"title": "Cohere Models", "snippet": context}]
)
print(response.text)
```

## 应用场景

| 场景 | 使用产品 | 理由 |
|------|----------|------|
| 智能客服 | Command R+ | 原生 RAG 支持，多轮对话 |
| 语义搜索 | Embed + Rerank | 高质量向量检索 + 重排优化 |
| 文档问答 | Command + Embed | 检索 + 生成完整 RAG 管道 |
| 推荐系统 | Embed | 用户/物品向量表示，相似度计算 |
| 内容审核 | Command | 可控输出，分类任务 |
| 数据提取 | Command | 结构化输出，JSON 模式 |

## 相关概念

- **RAG (Retrieval-Augmented Generation)**：检索增强生成，结合检索与生成
- **向量嵌入 (Vector Embedding)**：将文本映射为高维向量
- **向量数据库**：存储和检索向量，如 Pinecone、Qdrant、Milvus
- **重排序 (Reranking)**：对初步检索结果进行相关性重排
- **Embedding Search**：基于向量相似度的语义搜索
- **Fine-tuning**：针对特定任务微调模型
- **Function Calling**：让模型调用外部工具/API

## 延伸阅读

1. [Cohere 官方网站](https://cohere.com/)
2. [Cohere Command R+ 介绍](https://txt.cohere.com/command-r-plus/)
3. [Cohere Embed 文档](https://docs.cohere.com/docs/embeddings)
4. [Cohere Rerank 文档](https://docs.cohere.com/docs/rerank)
5. [RAG 架构详解](https://arxiv.org/abs/2005.11401)
6. [Hugging Face Cohere 模型](https://huggingface.co/Cohere)
