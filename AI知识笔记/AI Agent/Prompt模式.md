---
title: Prompt模式
alias: Prompt Design Patterns
tags: [Prompt, Agent, 设计模式, Few-shot, Chain-of-Thought]
category: AI Agent
created: 2026-03-31
updated: 2026-03-31
author: AI Developer
description: 探索AI Agent开发中核心的Prompt设计模式，包括Role Play、Few-shot Chain、Tree of Thoughts、React模式等，帮助构建更强大的AI推理与执行能力。
mastery: 3
rating: 8
related_concepts: [提示工程, Chain-of-Thought, ReAct, Tool-use Agent]
difficulty: intermediate
read_time: 15
prerequisites: [提示工程基础, Python基础]
---

# Prompt模式

## 一句话定义

Prompt模式是针对AI Agent推理与任务执行所设计的高阶提示结构化方法，通过特定的任务拆解、上下文组织和推理策略，使AI能够更准确地理解意图、规划路径并生成高质量响应。

## 详细说明

### 1. Role Play（角色扮演模式）

让AI扮演特定角色，利用角色设定来约束行为边界和输出风格。

**核心原理**：通过`system`提示词明确角色身份、技能、行为规范，使AI在既定角色框架内输出符合预期的内容。

**结构要素**：
- 角色身份定义（Who are you）
- 角色能力边界（What you can do）
- 行为约束（How you behave）
- 交互风格（Communication style）

### 2. Few-shot Chain（少样本链式模式）

通过在提示中嵌入少量示例，引导AI学习输入-输出的映射关系和推理模式。

**核心原理**：示例作为隐式规则载体，AI通过模式识别而非规则描述来理解任务。

**结构要素**：
- 任务描述（Task description）
- 示例输入-输出对（Input-output pairs）
- 过渡句（Transition phrase）
- 实际查询（Actual query）

### 3. Tree of Thoughts（思维树）

对复杂问题进行多路径探索，模拟人类的系统性思考过程。

**核心原理**：将问题分解为多个思考节点，每条路径代表一种可能的解决策略，通过评估和回溯找到最优解。

**结构要素**：
- 问题定义（Problem statement）
- 思考分支生成（Thought generation）
- 分支评估（Thought evaluation）
- 路径回溯（Backtracking）

### 4. React模式（推理-行动模式）

交替进行推理（Reasoning）和行动（Action），使AI能够动态调整策略。

**核心原理**：通过在推理过程中嵌入外部工具调用或环境交互，实现"边想边做"的动态推理循环。

**结构要素**：
- 推理步骤（Thought）
- 行动建议（Action）
- 观察结果（Observation）
- 下一轮推理（Next reasoning）

## 代码示例

### Role Play 示例

```python
from anthropic import Anthropic

client = Anthropic()

def role_play_prompt():
    return """你是一位资深的Python后端架构师，擅长使用FastAPI构建高性能微服务。

约束：
1. 始终考虑系统的可扩展性和可维护性
2. 代码必须包含完整的类型注解
3. 优先使用异步编程提升性能

当用户描述业务需求时，你需要：
- 分析需求的技术可行性和潜在风险
- 提供包含项目结构、核心代码、测试用例的完整方案
- 指出方案中的trade-offs

现在开始，请描述你的业务需求。"""

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=2048,
    system=role_play_prompt(),
    messages=[{"role": "user", "content": "我需要构建一个支持万人同时在线的实时聊天系统"}]
)
print(response.content)
```

### Few-shot Chain 示例

```python
def few_shot_classification():
    few_shot_prompt = """任务：判断文本的情感类别（positive/negative/neutral）

示例1：
输入：这家餐厅的食物非常美味，服务员态度也很好
输出：positive

示例2：
输入：等了整整两个小时菜才上桌，太失望了
输出：negative

示例3：
输入：明天预计有小雨，记得带伞
输出：neutral

现在请判断以下输入的情感：
输入：产品出乎意料的好用，性价比很高"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        system="你是一个情感分类专家，请严格按照示例格式输出",
        messages=[{"role": "user", "content": few_shot_prompt}]
    )
    return response.content

# Chain扩展：逐步引导
def few_shot_chain_extended():
    prompt = """任务：复杂问题分步解答

示例1：
问题：小明有15个苹果，给了小红7个，又买了5个，现在有多少？
解答：
第一步：15 - 7 = 8（给了小红后剩余）
第二步：8 + 5 = 13（买新后的总数）
最终答案：13个

示例2：
问题：一列火车长200米，以每秒30米的速度行驶，通过一条长400米的隧道需要多久？
解答：
第一步：火车完全通过隧道需要行驶的距离 = 火车长度 + 隧道长度 = 200 + 400 = 600米
第二步：时间 = 距离 ÷ 速度 = 600 ÷ 30 = 20秒
最终答案：20秒

现在请解答：
问题：一个水池有进水管和出水管，进水管每分钟注水10升，出水管每分钟出水6升，5分钟后水池有多少水？"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        system="你是数学解题专家，遵循分步解答格式",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content
```

### Tree of Thoughts 示例

```python
from anthropic import Anthropic
import json

client = Anthropic()

def tree_of_thoughts(problem: str):
    """思维树推理框架"""
    system_prompt = """你是一个问题解决专家。对于每个问题，你将：
1. 生成多个可能的思考方向（分支）
2. 评估每个分支的可行性和潜在价值
3. 选择最佳分支并深入分析
4. 如果遇到死胡同，回溯到上一个分支点

输出格式使用JSON：
{
    "thoughts": [
        {
            "id": 1,
            "content": "思考内容",
            "evaluation": "评估：优点/缺点",
            "selected": true/false
        }
    ],
    "path": ["选择路径的节点ID序列"],
    "conclusion": "最终结论"
}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": f"问题：{problem}"}]
    )
    return json.loads(response.content[0].text)

# 示例使用
result = tree_of_thoughts("如何减少公司服务器的云计算成本？")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

### React模式示例

```python
def react_agent_loop():
    """ReAct模式的简化实现"""
    tools = {
        "search": lambda query: f"搜索结果：关于'{query}'的最相关信息是...",
        "calculate": lambda expr: f"计算结果：{eval(expr)}",
        "web_scrape": lambda url: f"网页内容：来自{url}的摘要..."
    }

    system_prompt = """你是一个ReAct Agent。对于每个查询：

循环执行以下步骤：
1. Thought：分析当前状态，决定下一步行动
2. Action：根据决定调用工具（search/calculate/web_scrape）
3. Observation：分析工具返回的结果
4. 如果已得到答案，输出Final Answer

示例对话：
用户：北京的人口是多少？
Thought：我需要搜索北京人口数据
Action：search("北京人口 2024")
Observation：搜索结果显示北京常住人口约2189万
Final Answer：北京市常住人口约2189万（2024年数据）"""

    user_query = "特斯拉最新的季度财报营收是多少？"
    messages = [{"role": "user", "content": user_query}]

    # 模拟ReAct循环（实际生产中需要循环调用）
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )

    # 解析响应，提取Thought-Action-Observation-Final结构
    return response.content

# 更完整的ReAct Agent实现
class ReactAgent:
    def __init__(self):
        self.tools = tools
        self.max_iterations = 5

    def run(self, query: str):
        context = []
        for i in range(self.max_iterations):
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=512,
                system="你是一个ReAct Agent，严格按照Thought/Action/Observation/Final Answer格式输出",
                messages=[{"role": "user", "content": query}] + context
            )

            content = response.content[0].text

            if "Final Answer" in content:
                return content

            # 提取Action并执行
            if "Action:" in content:
                action_line = [l for l in content.split('\n') if 'Action:' in l][0]
                tool_name = action_line.split('(')[1].rstrip(')')
                tool_result = self.tools.get(tool_name, lambda x: "未知工具")("")
                context.append({"role": "user", "content": f"Observation: {tool_result}"})

        return "达到最大迭代次数"
```

## 应用场景

### Role Play
- **客服对话系统**：扮演售后客服、技术支持等角色，提供专业且符合场景的回复
- **面试模拟**：扮演面试官，生成针对性的面试问题和评估反馈
- **教学辅导**：扮演特定学科教师，提供个性化的学习指导

### Few-shot Chain
- **文本分类**：用少量标注样本引导模型理解分类边界
- **代码生成**：提供输入输出示例，让模型学习特定代码模式
- **格式转换**：通过示例说明复杂的数据格式转换规则

### Tree of Thoughts
- **战略规划**：商业决策中的多方案评估与选择
- **代码架构设计**：分析不同技术方案的优劣
- **复杂问题求解**：需要多步骤推理的数学或逻辑问题

### React模式
- **自主Agent系统**：需要调用外部工具完成复杂任务的Agent
- **实时问答系统**：结合搜索和推理的动态问答
- **数据分析Agent**：需要查询、处理、可视化的数据分析流程

## 相关概念

| 概念 | 关联说明 |
|------|---------|
| [提示工程](./提示工程.md) | Prompt模式的基础，是所有设计模式的底层支撑 |
| [Chain-of-Thought](https://arxiv.org/abs/2201.11903) | 思维链提示，ToT的前身，侧重单路径推理 |
| ReAct | 结合推理与行动的混合模式，React是其具体实现 |
| [Agent架构](./Agent架构.md) | Agent的整体架构设计，Prompt模式是其核心组件 |
| Tool-use Agent | 利用工具调用的Agent，React模式是常用实现方式 |

## 延伸阅读

1. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** - Yao et al., 2023
   - 论文链接：https://arxiv.org/abs/2305.10601
   - 核心贡献：提出用思维树替代思维链进行多路径探索

2. **ReAct: Synergizing Reasoning and Acting in Language Models** - Shunyu Yao et al., 2022
   - 论文链接：https://arxiv.org/abs/2210.03629
   - 核心贡献：提出推理-行动交替进行的Agent框架

3. **Prompt Engineering Guide** - Dair AI
   - https://www.promptingguide.ai/
   - 涵盖各类Prompt技术的系统指南

4. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** - Wei et al., 2022
   - https://arxiv.org/abs/2201.11903
   - 思维链提示的开创性工作

5. **LangChain Agents** - LangChain Documentation
   - https://python.langchain.com/docs/concepts/agents/
   - ReAct模式在LangChain中的实现参考
