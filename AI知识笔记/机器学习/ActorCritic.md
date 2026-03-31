---
title: ActorCritic
date: 2026-03-31
tags:
  - 机器学习
  - 强化学习
  - Actor-Critic
category: 机器学习
mathjax: true
---

# Actor-Critic（演员-评论家算法）

## 一句话定义

Actor-Critic是一种结合策略梯度（Actor）和值函数近似（Critic）的强化学习架构，其中Actor负责根据当前策略选择动作，Critic负责评估当前状态-动作对的价值并提供低方差的梯度估计信号。

## 核心公式

**优势函数（Advantage）：**
$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**Actor更新（策略梯度）：**
$$\nabla_\theta J(\theta) \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)]$$

**Critic更新（值函数近似）：**
$$L(\phi) = \mathbb{E}[(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2]$$

**TD误差（作为优势估计）：**
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## 详细说明

### 1. 基本原理

Actor-Critic架构的核心思想是将策略学习（Actor）和值函数学习（Critic）分离并协同优化。Actor根据当前的策略 $\pi_\theta(a|s)$ 选择动作，Critic根据TD学习估计状态值函数 $V_\phi(s)$ 或动作-状态值函数 $Q_\phi(s, a)$。Critic提供的优势估计用于降低Actor策略梯度估计的方差。

### 2. 算法流程

```
初始化 Actor 策略参数 θ，Critic 值函数参数 φ
for each episode:
    初始化状态 s
    for each step:
        Actor: 根据 πθ(a|s) 选择动作 a
        执行动作，观察奖励 r 和下一状态 s'
        Critic: 计算 TD 误差 δ = r + γV(s') - V(s)
        Critic: 更新值函数参数 φ（如 RMSprop）
        Actor: 更新策略参数 θ（使用 δ 作为优势估计）
        s ← s'
    until episode 结束
```

### 3. 优势与挑战

| 优势 | 挑战 |
|------|------|
| 低方差梯度估计 | 需要同时训练两个网络 |
| 适用于连续动作空间 | 超参数敏感（学习率、熵系数等） |
| 在线学习能力强 | Critic估计不准确会影响Actor |
| 可与其他技术结合（GAE、DDPG） | 训练稳定性问题 |

### 4. 重要变体

- **A3C (Asynchronous Advantage Actor-Critic)**：异步并行训练，多环境同时采集数据
- **A2C (Advantage Actor-Critic)**：A3C的同步版本
- **DDPG (Deep Deterministic Policy Gradient)**：确定性策略+Actor-Critic
- **SAC (Soft Actor-Critic)**：最大熵Actor-Critic，支持随机策略
- **TD3 (Twin Delayed DDPG)**：双 Critic 减少过估计

## 相关概念

[[策略梯度]] | [[PPO]] | [[DDPG]] | [[强化学习]] | [[优势函数]] | [[时序差分学习]] | [[值函数近似]] | [[A3C]] | [[SAC]]

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1999/file/6449f8a8e70a2c2c2fd4c6b2c3a6f9c3-Paper.pdf)
- [Deep Reinforcement Learning: A Hands-on Tutorial (Hessel et al.)](https://arxiv.org/abs/1710.02298)
- [Spinning Up: Actor-Critic](https://spinningup.openai.edu/en/latest/algorithms/actor_critic.html)
