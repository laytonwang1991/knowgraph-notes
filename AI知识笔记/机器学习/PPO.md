---
title: PPO
date: 2026-03-31
tags:
  - 机器学习
  - 强化学习
  - 策略优化
category: 机器学习
mathjax: true
---

# 近端策略优化 (Proximal Policy Optimization, PPO)

## 一句话定义

PPO是一种策略优化算法，通过引入剪辑（clip）机制限制策略更新幅度，在保证策略单调提升的同时实现简单、稳定的高效训练，是目前应用最广泛的深度强化学习算法之一。

## 核心公式

**PPO剪辑目标函数：**
$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t)\right]$$

**概率比：**
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**GAE（广义优势估计）：**
$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**完整PPO损失函数：**
$$L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)$$

## 详细说明

### 1. 基本原理

PPO的核心思想是限制策略更新的步长，避免因策略变化过大导致的性能崩溃。TRPO使用KL散度约束来限制策略更新，但计算复杂。PPO通过引入简单的剪辑机制 $clip(r_t(\theta), 1-\epsilon, 1+\epsilon)$ 实现了类似效果，同时保持了算法的简洁性和可实施性。

### 2. 剪辑机制详解

- 当 $A_t > 0$（动作优于基准）：增加 $\pi_\theta(a_t|s_t)$ 的概率，但被clip限制
- 当 $A_t < 0$（动作差于基准）：减少 $\pi_\theta(a_t|s_t)$ 的概率，但被clip限制
- clip确保 $r_t(\theta)$ 不会超出 $[1-\epsilon, 1+\epsilon]$ 范围

### 3. 关键技术：GAE

广义优势估计（GAE）通过参数 $\lambda \in [0,1]$ 在偏差和方差之间取得平衡：
- $\lambda = 0$：$A_t^{GAE} = \delta_t$（高偏差，低方差）
- $\lambda = 1$：$A_t^{GAE} = \sum \gamma^l \delta_{t+l}$（低偏差，高方差）
- 实际常用 $\lambda \in [0.9, 0.99]$

### 4. PPO vs TRPO

| 特性 | PPO | TRPO |
|------|-----|------|
| 约束方式 | 剪辑目标 | KL散度约束 |
| 计算复杂度 | O(n) | O(n²) |
| 实现难度 | 简单 | 复杂（需要共轭梯度） |
| 超参数 | ε（剪辑范围） | δ（KL目标） |
| 实际性能 | 相当 | 相当 |

## 相关概念

[[策略梯度]] | [[ActorCritic]] | [[TRPO]] | [[强化学习]] | [[深度强化学习]] | [[优势函数]] | [[GAE]] | [[策略优化]] | [[DDPG]]

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.edu/en/latest/algorithms/ppo.html)
- [Towards Delving Deep into Policy Optimization: Trust Region Methods](https://arxiv.org/abs/1910.06808)
- [PPO Implementation: A Comprehensive Guide](https://nn.labml.ai/rl/ppo/index.html)
