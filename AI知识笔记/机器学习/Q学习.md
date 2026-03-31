---
title: Q学习
date: 2026-03-31
tags:
  - 机器学习
  - 强化学习
  - 无模型学习
category: 机器学习
mathjax: true
---

# Q学习 (Q-Learning)

## 一句话定义

Q学习是一种基于值函数的离策略（off-policy）强化学习算法，通过迭代更新动作-价值函数Q(s,a)来估计最优动作价值，最终使智能体能够选择最大化长期奖励的动作。

## 核心公式

**Q值更新规则（Bellman最优方程）：**
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

**最优动作选择：**
$$a^* = \arg\max_{a} Q(s, a)$$

**最优值函数：**
$$V^*(s) = \max_a Q^*(s, a)$$

## 详细说明

### 1. 基本原理

Q学习是一种无模型的强化学习算法，智能体通过与环境交互获得的即时奖励和后续状态来更新Q值估计。其核心是使用时序差分（TD）学习，逐步逼近最优动作价值函数 Q*(s, a)。由于采用离策略学习，智能体可以使用任意策略生成的数据来更新Q值。

### 2. 算法流程

1. 初始化 Q(s, a) 为任意值（通常为0）
2. 对于每个episode：
   - 初始化状态 s
   - 对于episode的每一步：
     - 根据 ε-greedy 策略选择动作 a
     - 执行动作，观察奖励 r 和下一状态 s'
     - 更新 Q(s, a)：$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$
     - s ← s'
   - 直到 s 为终止状态

### 3. 收敛条件

Q学习收敛的条件：
- 所有状态-动作对被无限次访问
- 学习率满足 $\sum \alpha_t = \infty$ 且 $\sum \alpha_t^2 < \infty$
- 满足以上条件时，Q(s,a) 依概率收敛到 Q*(s,a)

### 4. 优缺点

| 优点 | 缺点 |
|------|------|
| 离策略学习，样本利用率高 | 状态空间大时，Q表不现实 |
| 理论保证收敛 | 只能处理离散有限状态/动作 |
| 简单易实现 | 对连续动作空间处理困难 |

## 相关概念

[[强化学习]] | [[深度强化学习]] | [[时序差分学习]] | [[Bellman方程]] | [[SARSA]] | [[值函数]] | [[ε-greedy]] | [[探索与利用]]

## 延伸阅读

- [Watkins, C.J.C.H. (1989). Learning from Delayed Rewards](https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)
- [Q-Learning: An Introduction](https://en.wikipedia.org/wiki/Q-learning)
- [Reinforcement Learning: An Introduction - Chapter 6 (Sutton & Barto)](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)
- [Deep Q-Learning (DQN)](https://arxiv.org/abs/1312.5602)
