---
title: ReAct范式
alias: ReAct Paradigm
tags:
  - AI Agent
  - 推理与行动
  - Thought-Action-Observation
  - 规划Agent
category: AI Agent
created: 2026-03-31
updated: 2026-03-31
author: Claude
description: 结合推理和行动的Agent框架，通过Thought-Action-Observation循环使模型能够动态决策、调用工具并从环境中学习。
mastery: 8
rating: 9
related_concepts:
  - AutoGPT
  - Toolformer
  - Chain of Thought
  - Reflexion
  - Plan-and-Execute
difficulty: 7
read_time: 12
prerequisites:
  - LLM基础概念
  - Prompt Engineering
  - Agent基础
---

# ReAct范式

## 一句话定义

ReAct（Reasoning + Acting）是一种让LLM交替进行推理和行动的Agent框架，通过Thought-Action-Observation循环动态结合语言推理和外部工具调用，解决复杂任务。

## 详细说明

### 1. 核心思想

ReAct的核心洞察是：**推理和行动不是孤立的，而是相互增强的**。

- **推理（Reasoning）**：帮助模型理解当前状态、规划下一步行动
- **行动（Acting）**：执行工具调用，从环境获取反馈
- **观察（Observation）**：将行动结果加入推理上下文

这种循环让模型能够：
- 动态决定是否需要调用工具
- 根据观察结果调整策略
- 追踪多步骤任务的执行进度
- 处理执行过程中的异常情况

### 2. Thought-Action-Observation循环

ReAct的标准循环结构：

```
Thought: 思考当前状态和下一步行动
    ↓
Action: 执行工具调用或生成回答
    ↓
Observation: 获取行动结果
    ↓
(循环直到任务完成)
```

### 3. 对话Agent中的应用

在对话场景中，ReAct处理用户query的方式：

1. **理解意图**：分析用户想要什么
2. **决定是否搜索**：是否需要调用检索工具
3. **执行搜索**：调用搜索引擎或知识库
4. **观察结果**：获取检索到的信息
5. **推理整合**：将信息整合成自然语言回答
6. **生成回复**：返回给用户

### 4. 规划Agent中的应用

对于复杂任务规划，ReAct的扩展模式：

```
目标分解 → 逐个执行 → 观察结果 → 动态调整 → 继续执行
```

每个子任务都是一个Thought-Action-Observation循环，直到所有子任务完成。

### 5. 与其他范式的区别

| 范式 | 特点 | 适用场景 |
|------|------|---------|
| **Chain of Thought** | 只推理不行动 | 简单推理任务 |
| **ReAct** | 推理+行动+观察 | 需要工具调用的任务 |
| **AutoGPT** | 完全自主决策 | 复杂长任务 |
| **Plan-and-Execute** | 先规划后执行 | 结构化复杂任务 |

### 6. ReAct的优势

- **可解释性**：每步都有明确的Thought记录推理过程
- **可控性**：人类可以干预和调整Thought方向
- **灵活性**：支持动态调用不同类型的工具
- **可调试**：容易定位哪一步出现问题

## 代码示例

### ReAct Agent核心实现

```python
from typing import List, Dict, Any, Callable
import json

class ReActAgent:
    def __init__(self, llm_client, tools: Dict[str, Callable]):
        self.llm = llm_client
        self.tools = tools

    def think(self, context: str) -> str:
        """思考步骤：分析当前状态，决定下一步行动"""
        prompt = f"""
        当前对话上下文：
        {context}

        可用工具：{list(self.tools.keys())}

        请分析：
        1. 当前任务进度如何？
        2. 下一步应该做什么？
        3. 需要调用工具吗？如果需要，调用哪个？

        以以下格式回复：
        Thought: 你的思考过程
        Action: 工具名或"FINAL_ANSWER"
        """
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def act(self, action: str, context: str) -> str:
        """行动步骤：执行工具调用或生成最终答案"""
        if action == "FINAL_ANSWER":
            # 生成最终回答
            return self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"基于以下上下文生成最终回答：\n{context}"}]
            ).choices[0].message.content

        elif action in self.tools:
            # 执行工具调用
            # 先获取工具参数
            params_prompt = f"""
            当前上下文：{context}
            需要调用的工具：{action}

            请生成调用该工具需要的JSON参数。
            """
            params = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": params_prompt}]
            ).choices[0].message.content

            try:
                params_dict = json.loads(params)
                result = self.tools[action](**params_dict)
                return json.dumps(result)
            except Exception as e:
                return f"工具执行错误：{e}"

        return "未知操作"

    def run(self, query: str, max_iterations: int = 10) -> str:
        """ReAct主循环"""
        context = f"用户问题：{query}\n\n"
        history = []

        for i in range(max_iterations):
            # Thought阶段
            thought_result = self.think(context)
            history.append(f"Thought {i+1}: {thought_result}")

            # 解析Thought结果
            lines = thought_result.strip().split('\n')
            action = None
            for line in lines:
                if line.startswith('Action:'):
                    action = line.replace('Action:', '').strip()
                    break

            if not action:
                action = "FINAL_ANSWER"

            # Action阶段
            action_result = self.act(action, context)
            history.append(f"Action: {action}\nResult: {action_result}")

            # Observation阶段
            observation = f"[Observation {i+1}]: {action_result}"
            history.append(observation)

            context += f"\n{observation}\n"

            if action == "FINAL_ANSWER":
                return action_result

        return f"达到最大迭代次数 {max_iterations}"

# 使用示例
tools = {
    "search": lambda query: f"搜索结果：关于'{query}'的信息...",
    "calculator": lambda expr: str(eval(expr)),
    "weather": lambda city: f"{city}的天气：晴朗，25度"
}

agent = ReActAgent(openai_client, tools)
result = agent.run("北京今天适合穿什么？")
```

### 带有外部搜索的ReAct实现

```python
import requests

def web_search(query: str) -> str:
    """使用搜索引擎的示例"""
    # 实际项目中替换为真实搜索API
    return f"搜索结果：{query}的相关信息..."

def wikipedia_lookup(topic: str) -> str:
    """维基百科查询示例"""
    return f"维基百科关于'{topic}'的条目内容..."

class WebReActAgent:
    """用于网络搜索和信息检索的ReAct Agent"""

    def __init__(self, llm):
        self.llm = llm

    def search_with_react(self, query: str) -> str:
        """ReAct驱动的网络搜索"""
        prompt = f"""
        用户查询：{query}

        你可以使用以下工具进行搜索：
        - web_search: 搜索互联网
        - wikipedia_lookup: 查询维基百科

        开始Thought-Action-Observation循环：
        """

        context = prompt
        for step in range(5):
            # 思考
            thought = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": context}]
            ).choices[0].message.content

            # 解析行动
            if "web_search" in thought.lower():
                # 提取搜索词
                search_query = self.extract_query(thought, "web_search")
                result = web_search(search_query)
            elif "wikipedia" in thought.lower():
                topic = self.extract_query(thought, "wikipedia")
                result = wikipedia_lookup(topic)
            else:
                return thought

            context += f"\n{thought}\n[Observation]: {result}\n"

        return "搜索完成"

    def extract_query(self, text: str, tool: str) -> str:
        """从思考中提取工具参数"""
        # 简化的提取逻辑
        if tool in text:
            start = text.find(f"{tool}[") + len(f"{tool}[")
            end = text.find("]", start)
            return text[start:end]
        return ""
```

## 应用场景

### 1. 智能问答系统

- **客服机器人**：理解问题，搜索知识库，返回准确答案
- **技术文档助手**：检索文档，定位相关信息，生成回答
- **法律咨询**：查询法条，分析案情，给出建议

### 2. 复杂任务执行

- **旅行规划**：搜索航班、酒店、景点，整合成行程
- **购物研究**：搜索产品、比较价格、分析评价
- **市场调研**：收集数据，分析趋势，生成报告

### 3. 编程辅助

- **代码调试**：分析错误信息，搜索解决方案，生成修复建议
- **API使用查询**：搜索文档，理解用法，生成示例代码
- **技术问题排查**：搜索类似问题，分析原因，给出步骤

### 4. 决策支持

- **投资分析**：收集市场数据，分析趋势，给出建议
- **健康咨询**：了解症状，搜索医学信息，给出建议
- **教育辅导**：评估学习情况，推荐资源，制定计划

## 相关概念

| 概念 | 关系 |
|------|------|
| **Chain of Thought** | CoT只包含推理，ReAct在此基础上增加行动和观察 |
| **Toolformer** | Toolformer关注工具学习，ReAct是工具使用的执行框架 |
| **AutoGPT** | AutoGPT使用ReAct类似的循环，但更强调自主决策 |
| **Reflexion** | Reflexion在ReAct基础上增加语言强化学习反思机制 |
| **Plan-and-Execute** | 先完整规划再执行，与ReAct的边想边做不同 |

## 延伸阅读

1. **ReAct原始论文**：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
2. **Synapse论文**：[Emergent Agentic Workflows from Successor Features](https://arxiv.org/abs/2401.00908)
3. **《Understanding ReAct》** - Lil'Log by Lilian Weng
4. **LangChain ReAct Agent**：[官方文档](https://python.langchain.com/docs/modules/agents/agent_types/react)
5. **《Building Agents with ReAct》** - Hugging Face Blog
6. **《Reasoning and Acting (ReAct) Tutorial》** - AI21 Blog

---

*本笔记由 Claude 生成，最后更新于 2026-03-31*
