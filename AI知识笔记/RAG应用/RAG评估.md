---
title: RAG评估
alias: RAG-Evaluation
tags: [RAG, 评估基准, RAGAs, LLM评估, 系统评估]
category: RAG应用
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: RAG系统评估方法与基准详解，包括RAGAs、Trulens等主流评估框架。
mastery: 75
rating: 8
related_concepts: [RAGAs, Trulens, 上下文相关性, 答案准确性, 忠诚度]
difficulty: 中级
read_time: 12分钟
prerequisites: [RAG工作流, LLM基础, 评估指标基础]
---

# RAG评估

## 一句话定义

RAG评估是衡量检索增强生成系统性能的多维度体系，通过分解为检索质量与生成质量两大类指标，系统性地评估RAG系统在答案准确性、上下文相关性、忠诚度等方面的表现。

---

## 详细说明

### 1. RAGAs基准概述

RAGAs（Retrieval-Augmented Generation Assessment）是评估RAG系统的权威基准框架。

**核心评估维度：**

| 维度 | 描述 | 评估对象 |
|------|------|----------|
| Context Relevance | 检索上下文与问题的相关性 | 检索阶段 |
| Answer Faithfulness | 生成答案对检索上下文的忠诚度 | 生成阶段 |
| Answer Relevance | 生成答案与原始问题的相关性 | 生成阶段 |

**评估流程：**

```
问题 → RAG系统 → 答案
         ↓
    检索上下文
         ↓
    LLM评判（使用CoT提示）
         ↓
    各维度评分（0-1）
```

### 2. 上下文相关性（Context Relevance）

衡量检索到的文档块与用户问题的相关程度。

**评估指标：**

| 指标 | 计算方式 | 理想值 |
|------|----------|--------|
| Precision@k | 前k个结果中相关文档比例 | 越高越好 |
| Recall@k | 全部相关文档中被召回比例 | 越高越好 |
| NDCG | 综合考虑相关性和排序位置 | 接近1 |
| MRR | 首个相关结果的位置倒数均值 | 越高越好 |

**核心代码示例：**

```python
from ragas import evaluate
from ragas.metrics import (
    context_relevancy,
    answer_correctness,
    faithfulness
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "user_input": [
        "什么是向量数据库？",
        "RAG的工作原理是什么？",
        "如何优化RAG性能？"
    ],
    "retrieved_contexts": [
        ["向量数据库是存储高维向量的系统...", "向量表示是AI的核心技术..."],
        ["RAG结合检索和生成两阶段...", "检索阶段使用向量相似度..."],
        ["RAG优化包括索引优化、查询优化..."]
    ],
    "response": [
        "向量数据库是专门用于存储和检索向量...",
        "RAG通过检索相关文档增强生成...",
        "可以通过混合检索、索引优化等方法..."
    ],
    "reference": [
        "向量数据库是一种专门存储高维向量...",
        "RAG是检索增强生成技术...",
        "RAG性能优化包括多个方面..."
    ]
}

# 创建数据集
dataset = Dataset.from_dict(eval_data)

# 执行评估
result = evaluate(
    dataset,
    metrics=[context_relevancy, faithfulness, answer_correctness]
)

print(result)
```

### 3. 答案准确性（Answer Correctness）

评估生成答案相对于标准答案的准确性。

**评估方法：**

```python
from ragas.metrics import answer_correctness
import anthropic

def evaluate_answer_correctness(question, answer, reference):
    """
    使用LLM评估答案准确性
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""评估以下答案的准确性。

问题: {question}
标准答案: {reference}
待评估答案: {answer}

从以下维度评分（0-1）：
1. 语义相似度：答案与标准答案的语义接近程度
2. 事实正确性：答案中的事实是否正确
3. 完整性：答案是否覆盖问题的所有方面

输出格式：
{{
    "semantic_similarity": 0.0-1.0,
    "factual_accuracy": 0.0-1.0,
    "completeness": 0.0-1.0,
    "overall_score": 0.0-1.0
}}
"""
            }
        ]
    )
    return response.content[0].text

# 评估示例
score = evaluate_answer_correctness(
    question="RAG的核心组件有哪些？",
    answer="RAG主要由检索器和生成器两大组件构成...",
    reference="RAG系统包含文档加载器、文本分割器、Embedding模型、向量数据库、检索器和生成器六大核心组件..."
)
```

### 4. 忠诚度（Faithfulness）

衡量生成答案对检索上下文的忠实程度，避免幻觉。

**评估框架：**

```python
from ragas.metrics import faithfulness

# RAGAs faithfulness评估
faithfulness_score = faithfulness(dataset)

# 自定义忠诚度评估器
class FaithfulnessEvaluator:
    def __init__(self, llm_client):
        self.llm = llm_client

    def evaluate(self, question, context, response):
        """
        评估答案对上下文的忠诚度
        """
        prompt = f"""判断以下回答是否忠实于给定的上下文信息。

上下文：
{context}

回答：
{response}

分析：
1. 回答中的每个陈述是否都能在上下文中找到依据？
2. 是否有任何上下文未提及的推测或虚构信息？
3. 如果有虚构信息，具体是什么？

最终判断：忠诚 / 不忠诚
如果忠诚，置信度评分（0-1）：
"""

        response = self.llm.generate(prompt)
        return self._parse_response(response)

    def _parse_response(self, llm_response):
        """解析LLM响应"""
        if "不忠诚" in llm_response:
            return 0.0
        # 提取置信度分数
        import re
        match = re.search(r'(\d+\.?\d*)\s*/\s*1', llm_response)
        return float(match.group(1)) if match else 0.5
```

### 5. Trulens评估框架

Trulens是另一个流行的RAG评估框架，提供更丰富的反馈。

**核心代码示例：**

```python
from trulens_eval import TruChain, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider import Langchain

# 初始化反馈函数
provider = Langchain()

# 定义反馈函数
f_context_relevance = Feedback(
    provider.context_relevance,
    prompt_template="将上下文对问题的相关程度评为0-10"
).on_input().on_output()

# 基础性评估（groundedness）
groundedness = Groundedness(provider=provider)
f_groundedness = Feedback(
    groundedness.groundedness_measure,
    prompt_template="评估回答对上下文的事实依赖程度0-10"
).on(Select.Record.calls[1].main.output).on_input()

# 答案相关性
f_answer_relevance = Feedback(
    provider.relevance,
    prompt_template="评估回答对问题的相关性0-10"
).on_input().on_output()

# 创建TruChain记录器
tru = TruChain(
    rag_chain,
    app_id="RAG_App_v1",
    feedbacks=[f_context_relevance, f_groundedness, f_answer_relevance]
)

# 执行评估
with tru as recording:
    response = rag_chain.invoke("RAG的工作原理是什么？")

# 获取评估结果
print(tru.get_records_and_feedback()[0])
```

---

## 应用场景

### 1. RAG系统选型与对比

- 不同向量数据库性能对比
- 不同Embedding模型效果评估
- 不同LLM的生成质量对比

### 2. RAG系统优化迭代

- 评估不同chunk_size对效果的影响
- 对比不同检索策略的效果
- 优化提示词模板

### 3. 生产环境监控

- 持续监控生产环境RAG效果
- 捕捉效果下降及时告警
- A/B测试不同配置

---

## 评估指标对比

| 指标 | 评估阶段 | 取值范围 | 优化方向 |
|------|----------|----------|----------|
| Context Precision | 检索 | 0-1 | 越高越好 |
| Context Recall | 检索 | 0-1 | 越高越好 |
| Faithfulness | 生成 | 0-1 | 越高越好 |
| Answer Relevance | 生成 | 0-1 | 越高越好 |
| Answer Correctness | 生成 | 0-1 | 越高越好 |
| Context Entity Recall | 检索 | 0-1 | 越高越好 |

---

## 核心代码：综合RAG评估系统

```python
"""
RAG系统综合评估器
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class RAGEvaluationResult:
    """RAG评估结果"""
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    answer_correctness: float
    overall_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "answer_correctness": self.answer_correctness,
            "overall_score": self.overall_score
        }

class RAGEvaluator:
    """RAG系统综合评估器"""

    def __init__(self, rag_system, evaluator_llm):
        self.rag = rag_system
        self.evaluator = evaluator_llm

    def evaluate_single(
        self,
        question: str,
        reference_answer: str,
        reference_contexts: List[str]
    ) -> RAGEvaluationResult:
        """评估单个问题"""

        # 获取RAG系统回答
        result = self.rag.query(question)
        response = result["result"]
        retrieved_contexts = [
            doc.page_content for doc in result.get("source_documents", [])
        ]

        # 并行评估各维度
        metrics = {}

        # 1. 上下文精确度（评估检索质量）
        metrics["context_precision"] = self._evaluate_context_precision(
            question, retrieved_contexts, reference_contexts
        )

        # 2. 上下文召回率
        metrics["context_recall"] = self._evaluate_context_recall(
            retrieved_contexts, reference_contexts
        )

        # 3. 忠诚度
        metrics["faithfulness"] = self._evaluate_faithfulness(
            question, retrieved_contexts, response
        )

        # 4. 答案相关性
        metrics["answer_relevance"] = self._evaluate_answer_relevance(
            question, response
        )

        # 5. 答案准确性
        metrics["answer_correctness"] = self._evaluate_answer_correctness(
            response, reference_answer
        )

        # 计算综合得分
        overall = (
            metrics["context_precision"] * 0.2 +
            metrics["context_recall"] * 0.2 +
            metrics["faithfulness"] * 0.25 +
            metrics["answer_relevance"] * 0.15 +
            metrics["answer_correctness"] * 0.2
        )

        return RAGEvaluationResult(
            context_precision=metrics["context_precision"],
            context_recall=metrics["context_recall"],
            faithfulness=metrics["faithfulness"],
            answer_relevance=metrics["answer_relevance"],
            answer_correctness=metrics["answer_correctness"],
            overall_score=overall
        )

    def _evaluate_context_precision(self, question, retrieved, reference):
        """评估上下文精确度"""
        # 使用LLM评判每个检索结果的相关性
        relevant = 0
        for ctx in retrieved:
            score = self.evaluator.judge(
                f"问题: {question}\n上下文: {ctx}\n是否相关？(是/否)"
            )
            if "是" in score:
                relevant += 1
        return relevant / len(retrieved) if retrieved else 0

    def _evaluate_context_recall(self, retrieved, reference):
        """评估上下文召回率"""
        # 检查参考上下文中有多少被检索到
        covered = 0
        for ref_ctx in reference:
            for ret_ctx in retrieved:
                if self._semantic_similarity(ref_ctx, ret_ctx) > 0.8:
                    covered += 1
                    break
        return covered / len(reference) if reference else 0

    def _evaluate_faithfulness(self, question, contexts, response):
        """评估忠诚度"""
        prompt = f"""评估回答是否忠实于上下文。

上下文: {' '.join(contexts)}
回答: {response}

检查回答中的每个陈述是否有上下文支持。输出0-1之间的分数。
"""
        score = self.evaluator.generate(prompt)
        return float(score.strip())

    def _evaluate_answer_relevance(self, question, response):
        """评估答案相关性"""
        prompt = f"""评估回答对问题的相关程度。

问题: {question}
回答: {response}

回答是否直接针对问题？输出0-1之间的分数。
"""
        score = self.evaluator.generate(prompt)
        return float(score.strip())

    def _evaluate_answer_correctness(self, response, reference):
        """评估答案准确性"""
        prompt = f"""评估回答相对于标准答案的准确性。

标准答案: {reference}
回答: {response}

输出0-1之间的分数。
"""
        score = self.evaluator.generate(prompt)
        return float(score.strip())

    def _semantic_similarity(self, text1, text2):
        """计算语义相似度（简化版）"""
        # 实际实现应使用Embedding模型
        common_words = set(text1) & set(text2)
        return len(common_words) / max(len(set(text1)), len(set(text2)))

    def evaluate_batch(
        self,
        test_cases: List[Dict]
    ) -> Dict[str, Any]:
        """批量评估"""
        results = []
        for case in test_cases:
            result = self.evaluate_single(
                question=case["question"],
                reference_answer=case["reference_answer"],
                reference_contexts=case["reference_contexts"]
            )
            results.append(result.to_dict())

        # 汇总统计
        avg_scores = {
            metric: sum(r[metric] for r in results) / len(results)
            for metric in results[0].keys()
        }

        return {
            "individual_results": results,
            "average_scores": avg_scores
        }
```

---

## 相关概念

- **RAGAs**: 专门评估RAG系统的基准框架
- **Trulens**: Truera开源的LLM应用评估框架
- **幻觉(Hallucination)**: LLM生成的不基于上下文的内容
- **上下文窗口**: LLM单次能处理的token数量
- **Embedding**: 将文本转换为向量的技术

---

## 延伸阅读

1. [RAGAs: Retrieval Augmented Generation Assessment](https://arxiv.org/abs/2310.15216) - RAGAs官方论文
2. [Trulens Eval Documentation](https://www.trulens.org/trulens_eval/) - Trulens官方文档
3. [RAG系统评估完全指南](https://www.ionos.com/digitalguide/online-marketing/search-marketing/rag-evaluation/) - 评估实践指南
4. [LLM评估基准综述](https://arxiv.org/abs/2309.15296) - LLM评估方法综述
