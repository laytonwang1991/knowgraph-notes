---
title: Toolformer
alias: Toolformer
tags:
  - AI Agent
  - 工具调用
  - API调用
  - 检索增强
  - Tool Learning
category: AI Agent
created: 2026-03-31
updated: 2026-03-31
author: Claude
description: 让LLM学会使用工具，通过工具调用训练使模型能够调用计算器、搜索引擎、API等外部工具扩展能力边界。
mastery: 8
rating: 9
related_concepts:
  - AutoGPT
  - ReAct
  - Function Calling
  - RAG
  - Toolbench
difficulty: 7
read_time: 12
prerequisites:
  - LLM基础概念
  - API调用基础
  - NLP基础
---

# Toolformer

## 一句话定义

Toolformer是Meta提出的框架，通过自监督学习让LLM学会使用工具（计算器、搜索引擎、API等），显著扩展模型的能力边界和事实准确性。

## 详细说明

### 1. 核心思想

Toolformer的核心理念是：**让语言模型自己决定何时、如何使用工具**。

- 不依赖人工标注的工具使用数据
- 通过自监督方式学习工具调用的时机和格式
- 将工具调用自然融入语言模型的推理过程

### 2. 工具类型

Toolformer支持多种工具类型：

| 工具类型 | 功能 | 示例 |
|---------|------|------|
| **计算器** | 数学计算 | `(5 + 3) * 2 = 16` |
| **搜索引擎** | 实时信息查询 | `Search[2024年诺贝尔奖获奖者]` |
| **问答系统** | 知识检索 | `QA[什么是量子计算]` |
| **翻译API** | 文本翻译 | `Translate[你好, 英语]` |
| **日历API** | 时间查询 | `Calendar[今天日期]` |
| **自定义API** | 业务集成 | 各类REST API调用 |

### 3. 工具调用训练流程

Toolformer的训练分为三个阶段：

**阶段1：工具采样**
- 给模型少量工具使用示例
- 模型生成大量可能的工具调用序列
- 过滤保留有效且相关的调用

**阶段2：损失计算**
- 只在文本生成部分计算损失
- 工具调用结果不参与反向传播
- 避免工具输出干扰模型学习

**阶段3：微调**
- 使用筛选后的数据微调基础模型
- 模型学会何时调用工具
- 保持原有的语言生成能力

### 4. 工具调用格式

Toolformer定义了统一的工具调用语法：

```
<tool_call>
{
  "name": "计算器",
  "parameters": {
    "expression": "(15 * 3) + 20"
  }
}
</tool_call>
```

工具返回结果后，继续生成：

```
<tool_result>
{
  "result": 65
}
</tool_result>
现在我们知道 (15 * 3) + 20 = 65。
```

### 5. 关键设计决策

- **何时调用**：模型自己决定（不是强制调用）
- **调用什么**：从可用工具中选择最合适的
- **参数生成**：根据任务上下文生成参数
- **结果融合**：将工具结果自然融入回答

## 代码示例

### Toolformer风格的工具调用框架

```python
import json
import re
from typing import List, Dict, Any, Optional

# 定义工具注册表
TOOL_REGISTRY = {
    "calculator": {
        "description": "执行数学计算",
        "parameters": {
            "expression": "str: 要计算的数学表达式"
        }
    },
    "search": {
        "description": "搜索互联网获取信息",
        "parameters": {
            "query": "str: 搜索关键词",
            "max_results": "int: 返回结果数量"
        }
    },
    "retriever": {
        "description": "从知识库检索相关内容",
        "parameters": {
            "query": "str: 检索查询",
            "top_k": "int: 返回文档数量"
        }
    }
}

def execute_tool(tool_name: str, parameters: Dict) -> str:
    """执行工具调用"""
    if tool_name == "calculator":
        try:
            result = eval(parameters["expression"])
            return str(result)
        except Exception as e:
            return f"计算错误：{e}"

    elif tool_name == "search":
        # 模拟搜索引擎
        return f"搜索结果：关于'{parameters['query']}'的信息..."

    elif tool_name == "retriever":
        # 模拟检索系统
        return f"检索到关于'{parameters['query']}'的相关文档"

    return "未知工具"

class ToolformerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.tools = TOOL_REGISTRY

    def parse_tool_calls(self, text: str) -> List[Dict]:
        """从模型输出中解析工具调用"""
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        tool_calls = []
        for match in matches:
            try:
                tool_calls.append(json.loads(match))
            except json.JSONDecodeError:
                continue
        return tool_calls

    def generate_with_tools(self, prompt: str, max_iterations: int = 3) -> str:
        """带工具调用的生成循环"""
        current_prompt = prompt

        for iteration in range(max_iterations):
            # 调用LLM生成
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": current_prompt}],
                tools=[
                    {"type": "function", "function": fn}
                    for fn in self.tools.values()
                ]
            )

            message = response.choices[0].message

            # 检查是否有工具调用
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # 执行工具调用
                tool_results = []
                for tool_call in message.tool_calls:
                    fn = tool_call.function
                    params = json.loads(fn.arguments)
                    result = execute_tool(fn.name, params)
                    tool_results.append({
                        "tool": fn.name,
                        "parameters": params,
                        "result": result
                    })

                # 将工具结果加入上下文
                current_prompt += f"\n\n{message.content}\n\n"
                for tr in tool_results:
                    current_prompt += f"<tool_result>{json.dumps(tr)}</tool_result>\n"

            else:
                # 无工具调用，返回最终结果
                return message.content

        return "达到最大迭代次数"

# 使用示例
agent = ToolformerAgent(openai_client)
result = agent.generate_with_tools(
    "计算 (12 + 8) * 3 的结果，然后搜索这个结果的含义"
)
```

### 工具选择决策示例

```python
def should_use_tool(self, query: str, context: str) -> Optional[str]:
    """决定是否使用工具以及使用哪个工具"""
    decision_prompt = f"""
    用户问题：{query}
    当前上下文：{context}

    可用工具：
    - calculator: 数学计算
    - search: 搜索最新信息
    - retriever: 检索知识库
    - none: 不需要工具

    决策：
    1. 这个问题是否需要调用工具？
    2. 如果需要，应该调用哪个工具？
    3. 调用工具需要什么参数？

    请用以下JSON格式回答：
    {{
        "use_tool": true/false,
        "tool_name": "工具名或none",
        "parameters": {{}}
    }}
    """
    # LLM调用返回决策...
    return {"use_tool": True, "tool_name": "calculator", "parameters": {...}}
```

## 应用场景

### 1. 实时信息查询

- **新闻摘要**：获取最新新闻并总结
- **股价查询**：实时股票价格和市场数据
- **天气预报**：获取精确的天气信息
- **体育比分**：查询比赛结果和统计数据

### 2. 精确计算场景

- **财务分析**：复杂的投资收益计算
- **工程计算**：技术参数和公式计算
- **数据分析**：统计分析和数据处理
- **单位换算**：各种计量单位转换

### 3. 知识库增强

- **企业知识管理**：连接内部文档系统
- **产品手册查询**：技术规格和故障排除
- **客服系统**：快速检索解决方案
- **法律文档**：合同条款检索和分析

### 4. 外部系统集成

- **日历管理**：创建和查询日程
- **邮件处理**：发送和检索邮件
- **CRM集成**：客户信息查询和更新
- **电商系统**：库存查询和订单处理

## 相关概念

| 概念 | 关系 |
|------|------|
| **ReAct** | ReAct使用Toolformer类似的工具调用机制 |
| **Function Calling** | OpenAI等厂商提供的原生工具调用API |
| **RAG** | 检索增强生成，与Toolformer的信息获取目的相似 |
| **Toolbench** | 另一个工具学习框架，专注API学习 |
| **HuggingGPT** | 使用工具完成多模态任务的框架 |
| **AutoGPT** | 自主Agent，集成多种工具使用的典型案例 |

## 延伸阅读

1. **Toolformer原始论文**：[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
2. **Toolbench论文**：[Toolbench: An Open Platform for Software Development](https://arxiv.org/abs/2305.04081)
3. **OpenAI Function Calling**：[官方文档](https://platform.openai.com/docs/guides/function-calling)
4. **《ChatGPT Plugins》** - OpenAI Plugin系统
5. **《HuggingGPT: Solving AI Tasks with ChatGPT》** - Microsoft Research
6. **LangChain Tools**：[LangChain官方文档](https://python.langchain.com/docs/modules/agents/tools/)

---

*本笔记由 Claude 生成，最后更新于 2026-03-31*
