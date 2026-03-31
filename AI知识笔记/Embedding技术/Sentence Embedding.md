---
title: Sentence Embedding
alias: 句子嵌入、句子向量、句嵌入
tags: [Sentence Embedding, SBERT, Instructor, SimCSE, 对比学习, 句向量]
category: AI知识笔记/Embedding技术
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 将整个句子映射到固定维度向量空间的技术，捕获句子级别的语义信息，用于语义搜索、文本相似度等任务。
mastery: 65
rating: 8
related_concepts: [Embedding概述, 向量检索, Transformer, BERT, 对比学习, 自然语言处理]
difficulty: 较高
read_time: 30分钟
prerequisites: [深度学习基础, Transformer架构, Embedding概述, PyTorch/TensorFlow基础]
---

# Sentence Embedding

## 一句话定义

**Sentence Embedding是将整句文本映射到固定维度稠密向量的技术，捕获句子级别的语义，使语义相似的句子在向量空间中距离更近。**

## 核心公式

### 1. SBERT (Sentence-BERT) 孪生网络结构

给定句子 $u$ 和 $v$，SBERT使用以下策略生成句子向量：

**Mean Pooling**:
$$\mathbf{u} = \frac{1}{|u|} \sum_{i=1}^{|u|} \mathbf{u}_i$$

其中 $\mathbf{u}_i$ 是BERT输出中第 $i$ 个token的向量表示。

**余弦相似度**:
$$\text{similarity}(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

### 2. SimCSE 对比学习损失

$$L = -\log \frac{\exp(\text{sim}(h_i, h_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(h_i, h_j^+) / \tau) + \sum_{j=1}^{N} \exp(\text{sim}(h_i, h_j^-) / \tau)}$$

其中：
- $h_i$ 是当前句子向量
- $h_i^+$ 是正样本（通过Dropout产生的同义句）
- $h_j^-$ 是负样本（其他句子）
- $\tau$ 是温度系数（temperature）

### 3. Instructor 指令调优Embedding

$$\mathbf{s} = \mathbf{E}_\theta([INST]) \cdot \mathbf{x} + \mathbf{E}_\theta([INST])_{special}$$

模型根据任务指令 $[INST]$ 自适应地生成针对特定任务的句子向量。

### 4. 向量归一化与相似度

**L2归一化**:
$$\mathbf{u}_{norm} = \frac{\mathbf{u}}{\|\mathbf{u}\|}$$

**内积相似度（归一化后等价于余弦）**:
$$\text{IP}(u, v) = \mathbf{u}_{norm} \cdot \mathbf{v}_{norm}$$

## 详细说明

### 为什么需要Sentence Embedding？

1. **句子级语义**：Word Embedding只捕获词级语义，无法表达句子整体含义
2. **变长输入**：神经网络需要固定维度输入，句子长度可变
3. **语义相似度**：直接计算句子对的相似度（如STS任务）需要句子向量
4. **下游任务**：信息检索、问答、聚类等需要句子级表示

### SBERT (Sentence-BERT)

1. **提出背景**：原生BERT计算句子相似度需要 $O(N^2)$ 次前向传播，SBERT将其降至 $O(N)$

2. **网络结构**：
   - 孪生网络（Siames Network）：两个共享参数的BERT编码器
   - 三种池化策略：CLS token、Mean pooling、Max pooling

3. **训练目标**：
   - Classification Objective：$o = \text{softmax}(W \cdot (u, v, |u-v|))$
   - Regression Objective：使用MSE损失预测相似度分数
   - Triplet Objective：$\|u - u^+\| - \|u - u^-\|$

4. **预训练任务**：
   - Next Sentence Prediction (NSP)
   - Masked Language Modeling (MLM)

### SimCSE (Simple Contrastive Sentence Embedding)

1. **核心思想**：利用Dropout作为数据增强的对比学习方法

2. **正负样本**：
   - 正样本：同一句子通过两次编码（不同Dropout mask）
   - 负样本：batch内其他句子

3. **无监督SimCSE**：
   - 仅使用单语数据
   - 通过Dropout构造正样本对
   - 在大规模无标注语料上训练

4. **有监督SimCSE**：
   - 使用标注的句子对数据
   - 将标注的相似句子作为正样本
   - 挖掘硬负例提升效果

### Instructor (Instruction-based Embedding)

1. **核心思想**：根据任务指令生成定制化的句子向量

2. **指令格式**：
   ```
   [INST] Represent the Wikipedia document for retrieving supporting evidence [/INST]
   [INST] Represent the question for checking if it answers a retrieved passage [/INST]
   ```

3. **优势**：
   - 单一模型支持多种任务
   - 通过指令引导模型关注任务相关信息
   - 在MTEB基准上表现优异

### 对比学习训练技巧

1. **Temperature（温度系数）**：
   - 较小值（0.01-0.1）：放大差异，更hard的对比
   - 较大值（0.5-1.0）：平滑分布，更soft的对比

2. **Hard Negative Mining**：
   - 选取消极样本中较难的（相似但不同）
   - 提升模型区分能力

3. **Batch Construction**：
   - 批次大小越大，负样本越多，效果通常越好
   - 需要在效果和显存间权衡

## 代码示例

### Python - 使用sentence-transformers (SBERT)

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 加载预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 编码句子
sentences = [
    "今天天气真好",
    "今天阳光明媚",
    "今天要下雨了",
    "机器学习是人工智能的分支"
]

embeddings = model.encode(sentences)
print(f"Embeddings形状: {embeddings.shape}")  # (4, 384)

# 计算相似度矩阵
similarity_matrix = cos_sim(embeddings, embeddings)
print("相似度矩阵:")
print(similarity_matrix)
# 期望："今天天气真好"与"今天阳光明媚"相似度高
#      "今天天气真好"与"今天要下雨了"相似度低

# 语义搜索示例
query = "推荐一本好看的科幻小说"
corpus = [
    "《三体》是一部优秀的科幻小说",
    "今天的股市行情不错",
    "学习Python编程的好资源",
    "《流浪地球》电影很好看"
]

query_emb = model.encode([query])
corpus_emb = model.encode(corpus)
similarities = cos_sim(query_emb, corpus_emb)[0]

# 按相似度排序
top_k = 3
top_indices = similarities.argsort(descending=True)[:top_k]
print(f"\nQuery: {query}")
print(f"Top {top_k} 结果:")
for idx in top_indices:
    print(f"  [{similarities[idx]:.4f}] {corpus[idx]}")
```

### Python - SimCSE实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class SimCSE(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooler = nn.Linear(768, 768)

    def pooler_output(self, last_hidden_state):
        # Mean pooling
        attention_mask = self.bert.get_extended_attention_mask(
            last_hidden_state.size()[:2],
            last_hidden_state.device
        )
        masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1, keepdim=True)
        return summed / counts

    def forward(self, sent1, sent2):
        # sent1, sent2: 句子对，用于有监督SimCSE
        out1 = self.bert(sent1)
        out2 = self.bert(sent2)

        z1 = self.pooler(out1.last_hidden_state)
        z2 = self.pooler(out2.last_hidden_state)

        return z1, z2

def contrastive_loss(z1, z2, temperature=0.05):
    """
    计算SimCSE对比损失
    z1, z2: (batch_size, hidden_dim) 句子向量
    """
    # 归一化
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # 计算相似度矩阵
    sim_matrix = torch.matmul(z1, z2.T) / temperature  # (batch, batch)

    # 对角线为正样本
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    # Cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss

# 使用示例
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
simcse = SimCSE()

sentences = ["I love natural language processing", "NLP is fascinating"]
encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

z1, z2 = simcse(encoded['input_ids'], encoded['attention_mask'])
loss = contrastive_loss(z1, z2)
print(f"Contrastive Loss: {loss.item():.4f}")
```

### Python - Instructor使用示例

```python
from sentence_transformers import SentenceTransformer

# 加载Instructor模型
model = SentenceTransformer('hkunlp/instructor-large')

# 不同任务使用不同指令
tasks = [
    {
        'instruction': "Represent a Wikipedia article for retrieving supporting evidence:",
        'text': "Machine learning is a subset of artificial intelligence..."
    },
    {
        'instruction': "Represent a scientific question for checking if a passage answers it:",
        'text': "What is the relationship between temperature and pressure?"
    },
    {
        'instruction': "Represent a movie review for sentiment analysis:",
        'text': "This film is absolutely brilliant and thought-provoking."
    }
]

# 生成向量
embeddings = model.encode([
    {"instruction": t['instruction'], "text": t['text']}
    for t in tasks
])

print(f"Embeddings形状: {embeddings.shape}")  # (3, 1024)
```

## 应用场景

1. **语义搜索**
   - 基于向量相似度的文档检索
   - 超越关键词匹配，支持语义理解
   - 支持多语言、跨语言搜索

2. **文本相似度**
   - 论文查重、剽窃检测
   - 客服对话理解
   -  paraphrase 检测

3. **聚类分析**
   - 新闻主题聚类
   - 用户评论分类
   - 商品类目归类

4. **问答系统**
   - 语义匹配的问题-答案对
   - 开放域问答
   - 检索增强生成(RAG)

5. **推荐系统**
   - 用户兴趣向量表示
   - 内容向量表示
   - 向量召回

6. **文本分类（Few-shot）**
   - 通过指令引导分类
   - 无需微调即可分类

## 相关概念

| 概念 | 说明 |
|------|------|
| SBERT | 基于BERT的句子向量模型，使用孪生网络结构 |
| SimCSE | 利用Dropout对比学习的句子向量方法 |
| Instructor | 指令调优的多任务句子向量模型 |
| 对比学习 | 通过拉近正例、推远负例学习表示的范式 |
| Pooling策略 | CLS、Mean、Max等将变长序列转为定长向量的方法 |
| Hard Negative | 与正样本相似但实际不同的困难负样本 |

## 延伸阅读

1. **经典论文**
   - Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
   - Gao et al. (2021) - "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
   - Su et al. (2023) - "INSTRUCTOR: A General-purpose Instructor Model for Demonstrating Knowledge"

2. **评测基准**
   - MTEB (Massive Text Embedding Benchmark): 综合评测16种Embedding任务
   - STS (Semantic Textual Similarity): 语义相似度评测
   - BEIR: 信息检索评测基准

3. **工具与资源**
   - sentence-transformers库: 封装了SBERT、SimCSE等多种模型
   - MTEB排行榜: 评估各模型性能
   - Hugging Face Models: 大量预训练Sentence Embedding模型
