---
title: LangChain Agent
category: AI Agent
tags: [LangChain, Agent框架, LLM, 工具调用]
date: 2026-03-31
created_by: Claude
---

# LangChain Agent

## 一句话定义

LangChain Agent 是 LangChain 框架中的核心组件，通过将 LLM 与外部工具结合，实现推理、决策和执行的一体化，是构建复杂 Agent 系统的主流开发框架。

## 核心公式/技术要点

### LangChain Agent 架构
```
用户输入
    ↓
LLM (推理引擎)
    ↓
Action (动作选择)
    ↓
Tool (工具执行)
    ↓
Observation (观察结果)
    ↓
┌─────────────────────────────────────┐
│         ReAct / Plan-Execute        │
│         Loop                        │
└─────────────────────────────────────┘
```

### Agent 类型体系
$$Agent_{type} \in \{ReAct, Plan\&Execute, Conversational, SelfAsk, React\_Textai\}$$

### 工具调用格式
```python
# 同步调用
result = tool.invoke(input_data)

# 异步调用
result = await tool.ainvoke(input_data)
```

## 详细说明

### 1. LangChain 概述

| 组件 | 功能 |
|------|------|
| **Models** | 模型接口和优化 |
| **Prompts** | Prompt模板管理 |
| **Chains** | 链式调用组合 |
| **Agents** | 自主决策执行 |
| **Memory** | 对话历史管理 |
| **Tools** | 外部工具集成 |
| **Indexes** | 文档检索增强 |

### 2. Agent 类型详解

#### ReAct Agent
结合推理(Reasoning)和行动(Acting)的经典范式。

```python
from langchain.agents import AgentType, initialize_agent
from langchain.agents import Tool
from langchain.llms import OpenAI

# 定义工具
tools = [
    Tool(name="Search", func=search_func, description="搜索信息"),
    Tool(name="Calculator", func=calc_func, description="数学计算")
]

# 初始化 Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

#### Plan-and-Execute Agent
先规划后执行的层次化Agent。

```python
from langchain.agents import PlanAndExecute
```

#### Conversational Agent
面向对话交互的Agent。

```python
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory
)
```

### 3. 工具系统

#### 内置工具
| 工具 | 功能 |
|------|------|
| SerpAPI | 搜索引擎 |
| Wolfram Alpha | 数学计算 |
| Python REPL | 代码执行 |
| Wikipedia | 百科查询 |
| ArXiv | 学术搜索 |

#### 自定义工具
```python
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper

search = SerpAPIWrapper()

def search_wrapper(query):
    return search.run(query)

tools = [
    Tool(
        name="Web Search",
        func=search_wrapper,
        description="用于搜索网络信息"
    )
]
```

### 4. 主流 Agent 实现

| Agent | 特点 | 使用场景 |
|-------|------|----------|
| **OpenAI Functions Agent** | 原生函数调用支持 | OpenAI模型 |
| **ReAct Agent** | 推理+行动循环 | 通用任务 |
| **Plan-and-Execute** | 先规划后执行 | 复杂多步骤任务 |
| **Self-Ask** | 自我追问 | 需要澄清的任务 |
| **Baby AGI** | 目标驱动的任务管理 | 自动化工作流 |

### 5. 记忆系统

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory
)
```

### 6. LCEL (LangChain Expression Language)

LangChain 的声明式链式调用语法：

```python
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

prompt = hub.pull("rlm/rag-prompt")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
```

### 7. 与其他框架对比

| 框架 | 开发方 | 生态 | 学习曲线 |
|------|--------|------|----------|
| LangChain | LangChain AI | 最丰富 | 较陡 |
| LlamaIndex | LlamaIndex | 检索为主 | 中等 |
| AutoGen | Microsoft | 多Agent | 中等 |
| CrewAI | CrewAI | 多Agent协作 | 平缓 |

### 8. 最新发展

- **LangGraph**: 图结构化的Agent工作流
- **LangSmith**: Agent调试和评估平台
- **LangServe**: 部署LangChain服务
- **LangChain Expression Language (LCEL)**: 新一代链式调用

## 相关概念

- [[AI Agent]] - LangChain Agent 属于AI Agent范畴
- [[AutoGPT]] - 另一个知名Agent实现
- [[工具调用|Tool Use]] - Agent的核心能力
- [[自然语言处理]] - LLM基础技术

## 延伸阅读

- [LangChain Official Documentation](https://docs.langchain.com/)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)
- [Building a ReAct Agent from scratch](https://python.langchain.com/docs/modules/agents/how_to/)
