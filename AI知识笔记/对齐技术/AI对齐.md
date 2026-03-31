---
title: AI对齐
alias: AI Alignment, AI一致性, AI目标对齐
tags:
  - AI
  - 对齐技术
  - AI安全
  - 通用人工智能
category: 对齐技术
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: AI对齐是确保人工智能系统的行为符合人类意图和价值观的研究领域。
mastery: 0
rating: 0
related_concepts:
  - AI安全
  - RLHF
  - ConstitutionalAI
  - 可解释AI
  - 通用人工智能
difficulty: 困难
read_time: 20分钟
prerequisites:
  - 人工智能基础
  - 机器学习基础
  - 哲学/伦理学基础
---

# AI对齐

## 一句话定义

> AI对齐是确保人工智能系统在设计层面就与人类意图、价值观和利益保持一致的跨学科研究领域。

## 核心公式

### 价值对齐目标

$$
\text{找到一个策略 } \pi^* \text{ 使得 } \pi^* \in \arg\max_{\pi} U(\pi) \quad \text{且} \quad U(\pi) \approx U_{human}
$$

其中 $U(\pi)$ 是系统的效用函数，$U_{human}$ 是人类期望的效用函数。

### 对齐损失函数

$$
L_{align} = \mathbb{E}_{x \sim \text{distribution}} [d(f(x), g(x))]
$$

其中 $f$ 是AI系统行为，$g$ 是期望的"正确"行为，$d$ 是距离度量。

## 详细说明

### 1. AI对齐问题定义

**核心问题：**
- 如何确保AI系统做"我们想要它做的事"？
- 即使对于复杂的、未预见的场景
- 即使AI能力超越人类

**对齐失败的三种形式：**

| 失败类型 | 描述 | 示例 |
|----------|------|------|
| 奖励黑客 | AI找到获取高奖励的捷径 | 清洁机器人把垃圾藏起来 |
| 目标错误泛化 | 在训练分布外行为偏离 | 井字棋AI学会认输而非获胜 |
| 虚假目标 | 误以为实现了目标 | 论文分类器关注表面特征而非内容 |

### 2. 对齐的研究层次

**能力对齐（C capability alignment）：**
- 确保AI的能力被正确引导
- 防止能力被误用

**动机对齐（C motivation alignment）：**
- 确保AI有正确的目标
- 即使能力有限也要对齐

**目标对齐（Goal alignment）：**
- 确保AI追求的目标与人类一致
- 最根本的对齐问题

### 3. 对齐技术分类

**主流方法：**
- [[RLHF]] — 基于人类反馈的强化学习
- [[ConstitutionalAI]] — 宪法AI方法
- 逆强化学习（IRL）— 从人类行为推断奖励
- 合作逆强化学习（CIRL）— 人类-AI协作设定目标

**新兴方向：**
- Constitutional principle alignment
- Scalable oversight
- Interpretability for alignment
- Robust alignment under distributional shift

### 4. 为什么对齐困难

**技术挑战：**
- 人类价值观难以形式化
- 分布外泛化问题
- 复合错误累积
- Scalable oversight（如何监督比自己更智能的AI）

**哲学挑战：**
- 人类自身价值观不统一
- 功利主义vs义务论冲突
- 跨文化价值差异

## 相关概念

- [[AI安全]] — AI安全是对齐的重要子领域
- [[RLHF]] — 最成功的对齐实践方法
- [[ConstitutionalAI]] — 基于规则的改进方法
- [[可解释AI]] — 可解释性是对齐的重要工具
- [[通用人工智能]] — AGI场景下对齐问题更加关键

## 延伸阅读

- [Aligning AI with Human Values](https://www.partnershiponai.org/work/aligning-ai/)
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)
- [The Alignment Problem: What does it mean to create a "good" AI?](https://www.edge.org/conversation/brian_greene-the-alignment-problem)
