---
title: 检索增强生成 (RAG)
category: 信息检索
tags: [RAG, LLM, 知识库, 问答系统]
date: 2026-03-31
---

# 检索增强生成 (RAG)

## 一句话定义

检索增强生成（Retrieval-Augmented Generation）是通过检索外部知识来增强大语言模型回答质量的技术架构。

## 核心公式

### RAG流程
$$P(output) = P(response | retrieved\_docs, query)$$

### 生成条件概率
$$P(y|x) = \sum_{z \in Z} P(y|z, x) P(z|x)$$

其中 $z$ 为检索文档，$x$ 为查询，$y$ 为生成回答

### 混合检索分数
$$Score = \alpha \cdot BM25(q, d) + (1-\alpha) \cdot cosine(emb(q), emb(d))$$

## 技术要点

- **检索器**：稀疏（BM25）与稠密（向量）混合
- **生成器**：LLM根据检索结果生成回答
- **索引构建**：文档分块、嵌入、索引
- **重排序**：检索后对候选文档重排序

## 详细说明

RAG解决了LLM的两大局限：知识陈旧和幻觉问题。通过从外部知识库检索相关文档，RAG让模型能够访问最新、最准确的信息，同时提供生成答案的依据，提升可解释性和可信度。

### 核心组件

1. **数据处理**：文档加载、分割、清洗
2. **索引构建**：Embedding、向量索引
3. **检索模块**：向量检索、关键词检索
4. **生成模块**：Context组装、Prompt工程
5. **评估优化**：RAGAS等评估框架

## 相关概念

- [[语义搜索]] - 检索技术基础
- [[向量数据库]] - RAG的向量存储层
- [[知识图谱]] - 结构化知识来源
- [[自然语言处理]] - NLP技术支撑

## 延伸阅读

- [RAG论文](https://arxiv.org/abs/2005.11401)
- [LangChain RAG教程](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAGAS评估框架](https://github.com/explodinggradients/ragas)
