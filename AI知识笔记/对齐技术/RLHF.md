---
title: RLHF
alias: Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习
tags:
  - AI
  - 对齐技术
  - 大语言模型
  - 强化学习
category: 对齐技术
created: 2026-03-31
updated: 2026-03-31
author: KnowGraph
description: RLHF是一种通过人类反馈信号来训练奖励模型，进而优化语言模型行为的技术。
mastery: 0
rating: 0
related_concepts:
  - 强化学习
  - 大语言模型
  - Conenal AI
  - PPO算法
difficulty: 困难
read_time: 15分钟
prerequisites:
  - 强化学习基础
  - 大语言模型基础
  - 概率论基础
---

# RLHF

## 一句话定义

> RLHF（基于人类反馈的强化学习）是一种通过收集人类偏好数据训练奖励模型，再用强化学习算法（如PPO）优化语言模型使其行为更符合人类意图的技术。

## 核心公式

### 奖励模型损失函数

$$
L(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} [\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]
$$

其中 $y_w$ 是人类偏好的回复，$y_l$ 是不被偏好的回复，$r_\theta$ 是奖励模型。

### PPO更新目标

$$
\max_\phi \mathbb{E}_{x \sim D, y \sim \pi_\phi(y|x)} [r_\theta(x, y) - \beta \cdot KL(\pi_\phi(y|x) || \pi_{ref}(y|x))]
$$

其中 $\pi_\phi$ 是待优化的策略，$\pi_{ref}$ 是参考模型，$\beta$ 是KL惩罚系数。

## 详细说明

### 1. RLHF三阶段流程

**第一阶段：监督微调（SFT）**
- 使用标注数据对预训练语言模型进行微调
- 目标：让模型产生初步符合指令的回复
- 数据格式：人类写的"标准答案"

**第二阶段：训练奖励模型（RM）**
- 收集人类偏好对比数据
- 输入：(prompt, response_w, response_l)
- 训练一个奖励模型预测人类偏好
- 输出：标量奖励分数

**第三阶段：强化学习微调**
- 使用PPO算法优化策略
- 奖励信号来自奖励模型
- 加入KL散度约束防止策略偏离SFT模型太远

### 2. 关键技术细节

| 组件 | 说明 |
|------|------|
| 奖励模型 | 通常基于SFT模型最后一层加线性头 |
| KL约束 | 防止模型走向极端的"奖励黑客"行为 |
| PPO裁剪 | 限制策略更新幅度保证稳定性 |
| 值函数 | 估计累积奖励的期望 |

### 3. RLHF解决的问题

- 有害内容生成
- 幻觉问题
- 不遵循指令
- 对齐人类价值观

### 4. 代码示例

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 取最后一层hidden state的mean pooling
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        reward = self.reward_head(pooled)
        return reward.squeeze(-1)

# 奖励模型损失
def reward_model_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

## 相关概念

- [[强化学习]] — RLHF使用PPO作为强化学习算法
- [[大语言模型]] — RLHF主要用于优化LLM的行为
- [[ConstitutionalAI]] — 另一种对齐技术方法
- [[PPO算法]] — 近端策略优化算法

## 延伸阅读

- [RLHF: Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2203.02155)
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [ChatGPT原理详解](https://openai.com/blog/chatgpt/)
