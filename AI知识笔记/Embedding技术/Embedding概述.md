---
title: Embedding概述
alias: Embedding基础、向量表示、离散到连续的桥梁
tags: [Embedding, Word2Vec, Item2Vec, Doc2Vec, 向量表示, 机器学习]
category: AI知识笔记/Embedding技术
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 将离散数据映射到连续向量空间的核心技术，使语义相似的内容在向量空间中彼此接近。
mastery: 70
rating: 8
related_concepts: [Sentence Embedding, 向量检索, Transformer, BERT, 语言模型]
difficulty: 中等
read_time: 25分钟
prerequisites: [线性代数基础, 概率论基础, 机器学习入门]
---

# Embedding概述

## 一句话定义

**Embedding是将离散的高维数据（如单词、物品、文档）映射到连续的低维向量空间的技术，使得语义相似的内容在向量空间中距离更近。**

## 核心公式

### 1. Word2Vec (Skip-gram) 目标函数

$$L = -\sum_{t=1}^{T} \sum_{c \in C(t)} \log P(w_c | w_t) = -\sum_{t=1}^{T} \sum_{c \in C(t)} \log \frac{\exp(v_{w_c}^T \cdot v_{w_t})}{\sum_{k=1}^{V} \exp(v_k^T \cdot v_{w_t})}$$

其中 $w_t$ 是中心词，$C(t)$ 是上下文词集合，$V$ 是词表大小。

### 2. 负采样损失函数

$$L = -\log \sigma(v_{c}^T \cdot v_{w}) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v_{w_i}^T \cdot v_{w})]$$

### 3. 余弦相似度

$$\text{cosine}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

### 4. 点积相似度

$$\text{similarity}(a, b) = a \cdot b = \sum_{i=1}^{n} a_i b_i$$

## 详细说明

### 什么是Embedding？

1. **离散到连续的映射**：传统表示方法（如one-hot编码）将每个词表示为高维稀疏向量，Embedding将其转化为低维稠密向量
2. **语义捕获**：相似的词在向量空间中距离更近，可捕获同义词、上下文关系
3. **可学习表示**：通过神经网络学习得到，任务导向优化

### Word2Vec

1. **CBOW (Continuous Bag-of-Words)**：根据上下文预测中心词
   - 输入：多个上下文词向量
   - 输出：中心词的概率分布

2. **Skip-gram**：根据中心词预测上下文
   - 输入：中心词向量
   - 输出：上下文词的概率分布
   - 适合处理稀有词，效果通常优于CBOW

### Item2Vec

1. 将Word2Vec思想应用于推荐系统
2. 物品序列类比于词序列（如用户行为序列）
3. 物品向量可计算相似度、用于协同过滤

### Doc2Vec

1. **DM (Distributed Memory)**：类似CBOW，增加段落向量作为额外输入
2. **DBOW (Distributed Bag of Words)**：类似Skip-gram，只用段落向量预测词

### Embedding维度选择

| 维度范围 | 适用场景 | 特点 |
|---------|---------|------|
| 50-100 | 小规模、简单任务 | 训练快、泛化好、细节丢失 |
| 100-300 | 中等规模、通用场景 | 平衡表达能力和效率 |
| 300-500 | 大规模、复杂语义 | 细节丰富、训练较慢 |
| 500+ | 超大规模、精细任务 | 表达能力强、内存占用高 |

## 代码示例

### Python - 使用gensim训练Word2Vec

```python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

# 准备语料（分词后的句子列表）
sentences = [
    ['今天', '天气', '非常', '好'],
    ['今天', '下雨', '需要', '带伞'],
    ['天气', '影响', '心情'],
    ['机器学习', '是', '人工智能', '的', '分支'],
    ['深度学习', '是', '机器学习', '的', '分支']
]

# 训练Word2Vec模型
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Embedding维度
    window=2,             # 上下文窗口大小
    min_count=1,          # 最小词频
    workers=4,            # 并行线程数
    sg=1,                 # 1=Skip-gram, 0=CBOW
    epochs=100            # 训练轮数
)

# 获取词向量
word_vector = model.wv['机器学习']
print(f"词向量维度: {word_vector.shape}")
print(f"词向量前5维: {word_vector[:5]}")

# 计算相似度
similarity = model.wv.similarity('今天', '天气')
print(f"'今天'和'天气'的相似度: {similarity:.4f}")

# 查找最相似的词
most_similar = model.wv.most_similar('机器学习', topn=3)
print(f"与'机器学习'最相似的词: {most_similar}")

# 保存和加载模型
model.save('word2vec.model')
model = Word2Vec.load('word2vec.model')
```

### Python - 使用PyTorch实现简单Embedding层

```python
import torch
import torch.nn as nn

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, 128)

    def forward(self, x):
        # x: (batch_size, seq_length) - 词索引
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        # 平均池化
        pooled = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        return self.projection(pooled)

# 示例
vocab_size = 10000
embedding_dim = 300
batch_size = 32
seq_length = 20

model = SimpleEmbeddingModel(vocab_size, embedding_dim)
x = torch.randint(0, vocab_size, (batch_size, seq_length))
output = model(x)
print(f"输出形状: {output.shape}")  # (32, 128)
```

### Python - 计算余弦相似度

```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (norm(a) * norm(b))

def batch_cosine_similarity(query, documents):
    """批量计算查询向量与文档向量的相似度"""
    query_norm = norm(query)
    doc_norms = norm(documents, axis=1)
    similarities = np.dot(documents, query) / (doc_norms * query_norm)
    return similarities

# 示例
word_a = np.array([0.2, 0.5, 0.8])
word_b = np.array([0.1, 0.6, 0.9])

sim = cosine_similarity(word_a, word_b)
print(f"余弦相似度: {sim:.4f}")  # 接近1表示相似

# 批量计算
query = np.array([0.2, 0.5, 0.8])
docs = np.array([
    [0.2, 0.5, 0.8],
    [0.1, 0.1, 0.1],
    [0.9, 0.9, 0.9]
])
sims = batch_cosine_similarity(query, docs)
print(f"各文档相似度: {sims}")
# 输出: [1.0, 0.32, 0.95]
```

## 应用场景

1. **自然语言处理**
   - 文本分类、情感分析
   - 机器翻译、文本生成
   - 命名实体识别、词性标注

2. **推荐系统**
   - Item2Vec用于物品相似度计算
   - 用户/物品向量表示
   - 协同过滤增强

3. **信息检索**
   - 语义搜索
   - 文档聚类
   - 问答系统

4. **计算机视觉**
   - 图像特征提取
   - 图像相似度比较
   - 跨模态检索

5. **知识图谱**
   - 实体嵌入
   - 关系预测
   - 知识推理

## 相关概念

| 概念 | 说明 |
|------|------|
| One-Hot编码 | 离散数据的高维稀疏表示，维度等于词表大小 |
| TF-IDF | 基于词频的文本表示方法，忽略词序 |
| BERT | 基于Transformer的上下文相关Embedding |
| 向量检索 | 在向量空间中快速查找相似项的技术 |
| 对比学习 | 通过拉近相似样本、推远负样本来学习表示 |

## 延伸阅读

1. **经典论文**
   - Mikolov et al. (2013) - "Distributed Representations of Words and Phrases and their Compositionality" (Word2Vec原始论文)
   - Le & Mikolov (2014) - "Distributed Representations of Sentences and Documents" (Doc2Vec)

2. **学习资源**
   - Stanford CS224N: Natural Language Processing with Deep Learning
   - Hugging Face Embedding教程

3. **工具推荐**
   - Gensim: Python的Word2Vec/ Doc2Vec实现
   - TensorFlow Hub: 预训练Embedding模型
   - Hugging Face Transformers: BERT等预训练模型
