---
title: Agent架构
alias: Agent Architecture
tags: [AI Agent, 架构设计, ReAct, 记忆系统]
category: AI Agent
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 深入解析AI Agent的核心架构组件，包括感知、规划、行动、记忆模块及ReAct推理模式。
mastery: 0.85
rating: 9
related_concepts: [ReAct, 思维链, 工具调用, 记忆系统, 规划算法]
difficulty: 高
read_time: 25分钟
prerequisites: [机器学习基础, Python编程, LLM API调用]
---

# Agent架构

## 一句话定义

AI Agent 是指能够感知环境、自主规划、调用工具并执行行动以完成复杂目标的智能系统，其核心在于将大语言模型作为"大脑"结合模块化的架构设计。

## 核心公式

### Agent执行循环

$$
\text{Agent} = \text{Perceive} \rightarrow \text{Plan} \rightarrow \text{Act} \rightarrow \text{Memory}
$$

$$
\text{Loop} = \text{While not done:} \quad (\text{观察} \rightarrow \text{推理} \rightarrow \text{执行})^n
$$

### ReAct模式

$$
\text{ReAct} = \text{Reason} + \text{Act} = \text{Thought} \rightarrow \text{Action} \rightarrow \text{Observation}
$$

## 详细说明

### 1. 核心组件

#### 1.1 感知模块 (Perception)

感知模块负责接收和解析外部输入：

- **多模态输入处理**：文本、图像、音频、视频的统一编码
- **上下文提取**：从原始输入中提取关键信息和状态
- **环境反馈解析**：解析工具返回结果、用户反馈等

```python
class PerceptionModule:
    def __init__(self, encoders: Dict[str, Encoder]):
        self.encoders = encoders
        self.context_window = ContextWindow(size=128000)

    def process(self, inputs: List[Input]) -> State:
        encoded = [self.encoders[i.type].encode(i) for i in inputs]
        return self.context_window.update(encoded)
```

#### 1.2 规划模块 (Planning)

规划模块是Agent的"大脑"，负责决策和计划生成：

- **任务分解**：将复杂目标拆解为可执行的子任务
- **路径规划**：确定最优的行动序列
- **自我反思**：评估当前计划的有效性并调整

```python
class PlanningModule:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.task_graph = TaskGraph()

    def plan(self, goal: str, state: State) -> List[Action]:
        # 零样本思维链规划
        prompt = f"""
        目标: {goal}
        当前状态: {state.summary()}
        思考你的推理过程，然后给出行动序列。
        """
        reasoning = self.llm.complete(prompt)
        return self.extract_actions(reasoning)

    def reflect(self, result: Result, goal: str) -> Reflection:
       反思提示 = f"检查结果: {result} 是否达成目标: {goal}"
        return self.llm.complete(反思提示)
```

#### 1.3 行动模块 (Action)

行动模块负责执行规划模块生成的行动：

- **工具选择**：根据上下文选择合适的工具
- **参数生成**：构造工具调用的参数
- **执行控制**：管理行动的顺序和条件执行

```python
class ActionModule:
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.executor = Executor()

    def execute(self, action: Action) -> Result:
        tool = self.tools[action.tool_name]
        params = self.build_params(tool, action.params)
        return self.executor.run(tool, params)

    def build_params(self, tool: Tool, params: Dict) -> Dict:
        # 使用LLM生成工具参数
        return tool.validate(params)
```

#### 1.4 记忆模块 (Memory)

记忆模块存储和管理Agent的经验和知识：

| 记忆类型 | 容量 | 持久性 | 用途 |
|---------|------|--------|------|
| 感官记忆 | 短 | 极短 | 即时感知数据 |
| 工作记忆 | 中 | 短 | 当前任务相关 |
| 情景记忆 | 长 | 中 | 经历和经验 |
| 语义记忆 | 无限 | 长 | 事实和知识 |

```python
class MemoryModule:
    def __init__(self):
        self.sensory = SensoryMemory()
        self.working = WorkingMemory(capacity=10)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    def store(self, experience: Experience):
        self.sensory.add(experience.raw)
        self.working.add(experience.relevant)
        self.episodic.add(experience)

    def retrieve(self, query: str) -> List[Memory]:
        # 向量检索
        results = self.episodic.search(query, top_k=5)
        return self.working.relevant(results)
```

### 2. ReAct模式详解

ReAct (Reasoning + Acting) 是一种结合推理和行动的模式：

```
Thought: 我需要找到北京的天气
Action: search(query="北京天气")
Observation: 北京今天晴，25度
Thought: 用户可能需要穿衣建议
Action: recommend(clothing, temperature=25)
```

```python
def react_loop(agent: Agent, task: str) -> str:
    history = []
    max_iterations = 10

    for i in range(max_iterations):
        # 推理阶段
        thought = agent.reason(task, history)
        history.append(f"Thought: {thought}")

        # 决定是否行动
        if agent.needs_action(thought):
            action = agent.extract_action(thought)
            result = agent.execute(action)
            history.append(f"Action: {action}")
            history.append(f"Observation: {result}")
        else:
            # 直接返回最终答案
            return agent.finalize(thought)

    return agent.finalize(history)
```

### 3. 架构变体

#### 3.1 单Agent架构

```
┌─────────────────────────────────┐
│         User Input              │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│        Perception               │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│    Planning (LLM + Tools)        │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│         Action                  │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│         Memory                  │
└─────────────┬───────────────────┘
              ▼
         Final Output
```

#### 3.2 多Agent协作架构

```
  ┌──────────────┐
  │  Coordinator  │
  │    Agent      │
  └──────┬───────┘
         │ 分解任务
    ┌────┴────┐
    ▼         ▼
┌──────┐  ┌──────┐
│Sub-  │  │Sub-  │
│Agent1│  │Agent2│
└──┬───┘  └──┬───┘
   │         │
   └────┬────┘
        ▼
   汇总结果
```

## 应用场景

1. **自动化助手**：日程管理、邮件处理、数据分析
2. **代码助手**：代码生成、调试、代码审查
3. **研究助理**：文献检索、摘要生成、假设验证
4. **智能客服**：多轮对话、问题解决、情感识别
5. **数据分析**：数据清洗、可视化、洞察发现

## 相关概念

- [ReAct模式](./ReAct模式.md)
- [工具调用](./工具调用.md)
- [思维链(Chain-of-Thought)](../提示工程.md)
- [自主Agent (AutoGPT../AutoGPT.md)
- [LangChain Agent](./LangChain%20Agent.md)

## 延伸阅读

1. Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
2. Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
3. OpenAI. "GPT-4 Technical Report" - Agent相关章节
4. Anthropic. "Claude的Agent能力解析"
5. LangChain Documentation - Agent Architecture

