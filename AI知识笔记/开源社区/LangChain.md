---
title: LangChain
alias: LangChain
tags:
  - AI
  - LLM
  - 开发框架
  - RAG
category: 开源社区
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: LangChain是用于构建LLM应用的开发框架，提供组件化和链式调用能力，简化AI应用的开发流程。
mastery: 0
rating: 0
related_concepts:
  - 大语言模型
  - RAG
  - 向量数据库
  - AI代理
difficulty: 进阶
read_time: 12分钟
prerequisites:
  - Python基础
  - LLM基本概念
---

# LangChain

## 一句话定义

> LangChain是用于构建大语言模型应用的开发框架，通过组件化和链式调用能力简化AI应用程序的开发流程。

## 基本信息

| 字段 | 内容 |
|------|------|
| 开发组织 | LangChain Inc. |
| 首次发布 | 2022年10月 |
| GitHub星标 | 95,000+ |
| 贡献者 | 2,000+ |
| 当前版本 | 0.x |
| 许可证 | MIT |

## 详细说明

### 1. 核心概念

**组件（Components）：**
- LLM接口：统一封装多种模型
- Prompt模板：可复用的提示词
- 工具抽象：搜索、数据库等
- 索引与检索：文档处理与向量存储

**链（Chains）：**
- 定义工作流程
- 组合多个组件
- 处理复杂任务

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model="gpt-4")
prompt = PromptTemplate.from_template("用{language}写一个{topic}的Hello World")

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"language": "Python", "topic": "基础"})
```

### 2. 核心模块

| 模块 | 功能 |
|------|------|
| langchain-core | 核心抽象与基类 |
| langchain-community | 第三方集成 |
| langchain-openai | OpenAI模型集成 |
| langchain-anthropic | Anthropic模型集成 |
| langgraph | 构建多代理工作流 |

### 3. 典型应用示例

**RAG（检索增强生成）：**
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 向量数据库
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

result = qa_chain.invoke({"query": "问题..."})
```

**AI代理：**
```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool

tools = [
    Tool(name="Search", func=search_fn, description="搜索信息"),
    Tool(name="Calculator", func=calc_fn, description="计算数学")
]

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

### 4. 生态版图

```
LangChain生态
├── 核心框架
│   ├── langchain-core — 核心抽象
│   ├── langchain — 主框架
│   └── langgraph — 多代理工作流
├── 模型集成
│   ├── langchain-openai — OpenAI系
│   ├── langchain-anthropic — Anthropic系
│   └── langchain-community — 社区集成
├── 数据集成
│   ├── 向量存储 — Pinecone, Weaviate, FAISS
│   ├── 文档加载器 — PDF, Web, CSV
│   └── 嵌入模型 — OpenAI, Cohere
└── 应用场景
    ├── RAG — 检索增强生成
    ├── 代理 — AI自主决策
    └── 对话 — 聊天机器人
```

### 5. 竞品对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| LangChain | 组件丰富、社区活跃 | 快速原型、研究 |
| LlamaIndex | 数据连接更强 | RAG优先场景 |
| LangGraph | 工作流编排 | 复杂多代理系统 |

## 相关概念

- [[大语言模型]] — LangChain的核心是调用LLM
- [[RAG]] — 检索增强生成是LangChain主要应用
- [[向量数据库]] — 用于存储嵌入向量
- [[AI代理]] — LangChain支持构建AI代理
- [[自然语言处理]] — NLP技术的应用场景

## 延伸阅读

- [LangChain官网](https://www.langchain.com/)
- [LangChain文档](https://python.langchain.com/docs/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
