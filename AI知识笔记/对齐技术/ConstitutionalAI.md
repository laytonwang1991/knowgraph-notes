---
title: ConstitutionalAI
alias: CAI, 宪法AI
tags:
  - AI
  - 对齐技术
  - 安全AI
  - 大语言模型
category: 对齐技术
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: ConstitutionalAI是一种通过少量原则（宪法）指导AI行为自我改进的对齐方法。
mastery: 0
rating: 0
related_concepts:
  - RLHF
  - AI对齐
  - AI安全
  - 大语言模型
difficulty: 困难
read_time: 15分钟
prerequisites:
  - 强化学习基础
  - 大语言模型基础
  - AI对齐概念
---

# ConstitutionalAI

## 一句话定义

> ConstitutionalAI是一种通过一组人类编写的原则（宪法）指导AI进行自我批判和改进的对齐方法，无需大量人类标注反馈。

## 核心公式

### 响应评估分数

$$
S = \sum_{i=1}^{N} w_i \cdot c_i(x, y)
$$

其中 $w_i$ 是第 $i$ 条原则的权重，$c_i$ 是原则 $i$ 对响应 $y$ 的评分函数。

### 自我批判损失

$$
L_{critique} = -\mathbb{E}_{x \sim D} \log p_{LM}(y_{critique} | x, y_{initial}, p_i)
$$

其中 $y_{initial}$ 是初始响应，$y_{critique}$ 是基于原则 $p_i$ 的批判，$p_{LM}$ 是语言模型。

## 详细说明

### 1. ConstitutionalAI核心思想

**传统RLHF的问题：**
- 需要大量人类标注反馈
- 人类标注成本高、速度慢
- 难以扩展到复杂行为规范

**ConstitutionalAI的解决思路：**
- 用少量明确的原则替代大量隐式的人类偏好
- 让AI模型自己评估和改进自己的输出
- 模仿人类价值观的内化过程

### 2. 双阶段训练流程

**第一阶段：批判阶段（Critique Phase）**

1. 从有害查询开始
2. 让模型生成初始响应
3. 随机选择一条宪法原则
4. 要求模型根据原则批判自己的响应
5. 生成改进后的响应

**第二阶段：微调阶段（Fine-tuning Phase）**

1. 使用(原始查询, 初始响应, 改进响应)三元组
2. 训练模型偏好改进后的响应
3. 使用SL（监督学习）而非RL

### 3. 宪法原则示例

```
1. 选择最不可能包含有害、非法、不道德或破坏性内容的回应。
2. 选择最能展示智能助手特征（如乐于助人、好奇心、诚实）的回应。
3. 选择更可能被认为有助益且安全的回应。
4. 选择更能展示健康价值观的回应，如公平、同理心。
```

### 4. 与RLHF对比

| 维度 | RLHF | ConstitutionalAI |
|------|------|------------------|
| 人类反馈量 | 大量 | 少量（仅宪法原则） |
| 训练方式 | 强化学习 | 监督学习 |
| 扩展性 | 难 | 易 |
| 透明度 | 低 | 高（原则明确） |

## 相关概念

- [[RLHF]] — 另一种主流对齐方法
- [[AI对齐]] — 更广泛的对齐问题
- [[AI安全]] — AI安全问题
- [[大语言模型]] — CAI主要应用于LLM

## 延伸阅读

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [AI Alignment Podcast: Constitutional AI](https://www.youtube.com/watch?v=EYqIycmiW3c)
