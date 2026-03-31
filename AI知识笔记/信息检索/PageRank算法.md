---
title: PageRank算法
alias: PageRank
tags: [信息检索, 图算法, 排序算法, Google, 马尔可夫链]
category: 信息检索
created: 2026-03-31
updated: 2026-03-31
author: AI知识库
description: Google创始人Larry Page提出的基于网页链接关系对网页重要性进行排名的算法，核心思想是通过随机游走在网页图上的概率分布来确定页面权威度。
mastery: 0.6
rating: 8
related_concepts: [马尔可夫链, 随机游走, 阻尼因子, 链接分析, HITS算法, 特征向量中心性]
difficulty: 4
read_time: 25
prerequisites: [线性代数基础, 概率论基础, 图论基础]
---

# PageRank算法

## 一句话定义

PageRank是由Google创始人Larry Page于1996年提出的一种通过分析网页之间链接关系来评估网页重要性的迭代算法，核心思想是"被重要网页链接的网页更重要"。

## 核心公式

### 基本PageRank公式

$$PR(u) = \frac{1-d}{N} + d \sum_{v \in B(u)} \frac{PR(v)}{L(v)}$$

其中：
- $PR(u)$：页面 $u$ 的PageRank值
- $d$：阻尼因子（Damping Factor），通常取0.85
- $N$：总页面数
- $B(u)$：指向页面 $u$ 的所有页面集合
- $L(v)$：页面 $v$ 向外链接的数量

### 马尔可夫链矩阵形式

$$\vec{PR} = \left( (1-d)\frac{\mathbf{1}\mathbf{1}^T}{N} + d\mathbf{M} \right)^T \vec{PR}$$

其中 $\mathbf{M}$ 为转移矩阵，$\mathbf{1}$ 为全1向量。

## 详细说明

### 1. 基本思想

PageRank基于以下两个核心假设：
- **数量假设**：一个页面被越多的页面链接，其重要度越高
- **质量假设**：一个页面被越重要的页面链接，其重要度越高

这两个假设形成了一个递归的定义，需要通过迭代计算来求解。

### 2. 马尔可夫链视角

PageRank可以理解为一个随机游走过程：
- 假设一个用户在互联网上随机点击链接浏览页面
- 以概率 $1-d$ 随机跳转到任意页面（ teleport 机制）
- 以概率 $d$ 沿着当前页面的链接继续浏览
- 稳定状态下每个页面被访问的概率即为PageRank值

### 3. 阻尼因子的意义

阻尼因子 $d$ 控制随机跳转的比例：
- $d=1$：纯贪婪游走，无随机跳转，容易陷入悬挂节点
- $d=0$：完全随机游走，所有页面等概率
- $d=0.85$：Google原始采用的折中值，平衡探索与利用

### 4. 悬挂节点处理

对于没有出链的悬挂节点（dangling nodes），采用以下策略：
- 将其出链指向所有页面，避免概率流失
- 或在迭代过程中忽略其出链贡献

### 5. 收敛性保证

PageRank的收敛性由以下定理保证：
- 转移矩阵是素矩阵（primitive matrix）
- 根据Perron-Frobenius定理，迭代必然收敛到唯一的主特征向量
- 收敛条件：$\|\vec{PR}_{t+1} - \vec{PR}_t\|_1 < \epsilon$

## 代码实现

### Python实现（幂迭代法）

```python
import numpy as np

def pagerank(adj_matrix, d=0.85, epsilon=1e-6, max_iter=100):
    """
    PageRank算法实现

    参数:
        adj_matrix: 邻接矩阵，adj_matrix[i][j]=1表示页面i指向页面j
        d: 阻尼因子
        epsilon: 收敛阈值
        max_iter: 最大迭代次数

    返回:
        pageranks: 各页面的PageRank值
    """
    n = adj_matrix.shape[0]

    # 构建转移矩阵（按列归一化）
    out_links = adj_matrix.sum(axis=1, keepdims=True)
    out_links[out_links == 0] = 1  # 避免除零
    M = adj_matrix / out_links  # 按行归一化

    # 初始PageRank（均匀分布）
    pr = np.ones(n) / n

    # 幂迭代
    for _ in range(max_iter):
        pr_new = (1 - d) / n + d * M.T @ pr
        if np.linalg.norm(pr_new - pr, 1) < epsilon:
            break
        pr = pr_new

    return pr


# 示例：有向图邻接矩阵
# A -> B, C; B -> C; C -> A, B
adj = np.array([
    [0, 1, 1],  # A指向B, C
    [0, 0, 1],  # B指向C
    [1, 1, 0],  # C指向A, B
])

ranks = pagerank(adj)
print("PageRank值:", ranks)
```

### NetworkX库实现

```python
import networkx as nx

# 创建有向图
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A'), ('C', 'B')])

# 计算PageRank
pageranks = nx.pagerank(G, alpha=0.85)

print("PageRank值:")
for page, rank in sorted(pageranks.items(), key=lambda x: -x[1]):
    print(f"  {page}: {rank:.4f}")
```

## 应用场景

| 场景 | 说明 |
|------|------|
| 搜索引擎排名 | Google早期核心排序算法，决定网页搜索结果顺序 |
| 社交网络影响力 | 评估Twitter、微博等平台用户的影响力 |
| 学术论文重要性 | 评估论文的引用影响力（如PageRank引文分析） |
| 关键词提取 | 结合TextRank进行关键短语提取 |
| 推荐系统 | 基于图结构的物品推荐 |
| 网络安全 | 检测恶意网页或异常网络节点 |

## 相关概念

- **HITS算法**：另一个著名的链接分析算法，区分Hub（枢纽页）和Authority（权威页）
- **特征向量中心性**：PageRank是特征向量中心性的一个变体
- **TrustRank**：基于PageRank的链接信任传播算法，用于打击垃圾网页
- **TextRank**：PageRank在文本处理中的延伸，用于关键词抽取和摘要生成

## 延伸阅读

1. Page, L., et al. (1998). "The PageRank Citation Ranking: Bringing Order to the Web." Stanford InfoLab.
2. Langville, A.N., & Meyer, C.D. (2006). "Google's PageRank and Beyond: The Science of Search Engine Rankings." Princeton University Press.
3. 李航. (2019). 《统计学习方法》（第2版）. 清华大学出版社.
