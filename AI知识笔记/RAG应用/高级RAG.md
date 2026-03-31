---
title: 高级RAG技术
alias: Advanced-RAG
tags: [高级RAG, Query重写, HyDE, 混合检索, 重排序, Self-RAG]
category: RAG应用
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 深入解析Query重写、HyDE、混合检索、重排序、Self-RAG等提升RAG效果的高级技术。
mastery: 70
rating: 9
related_concepts: [Query重写, HyDE, 混合检索, 重排序, Self-RAG, RAG工作流]
difficulty: 高级
read_time: 18分钟
prerequisites: [RAG工作流, 向量检索基础, LLM API使用]
---

# 高级RAG技术

## 一句话定义

高级RAG技术是通过Query重写、HyDE、混合检索、重排序、Self-RAG等方法解决基础RAG的检索质量低、上下文不完整、幻觉等问题的进阶技术集合。

---

## 详细说明

### 1. Query重写（Query Rewriting）

Query重写通过LLM改写用户查询，提升检索的准确性和召回率。

**为什么需要Query重写：**
- 用户问题表达不清晰或存在歧义
- 原始查询与知识库文档的表述方式不同
- 口语化表达与书面语不匹配

**核心代码示例：**

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Query重写提示模板
QUERY_REWRITE_PROMPT = PromptTemplate(
    template="""你是一个查询优化专家。将用户的问题改写成更适合检索的形式。

原问题: {query}

要求：
1. 去除口语化表达，转换为标准表述
2. 补充必要的上下文信息
3. 分解复合问题为多个简单问题
4. 保持原意不变

重写后的问题:""",
    input_variables=["query"]
)

class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm
        self.chain = LLMChain(llm=llm, prompt=QUERY_REWRITE_PROMPT)

    def rewrite(self, query: str) -> str:
        """重写查询"""
        result = self.chain.invoke({"query": query})
        return result["text"].strip()

    def multi_rewrite(self, query: str, n_versions: int = 3) -> list[str]:
        """生成多个重写版本用于并行检索"""
        extended_prompt = f"""{QUERY_REWRITE_PROMPT.template}

请生成{n_versions}个不同角度的重写版本："""
        response = self.llm.invoke([
            {"role": "user", "content": extended_prompt.format(query=query)}
        ])
        versions = response.content.strip().split("\n")
        return [v.strip() for v in versions if v.strip()]

# 使用示例
rewriter = QueryRewriter(ChatOpenAI(model="gpt-4"))

# 单一重写
refined_query = rewriter.rewrite("RAG是啥意思啊")
print(refined_query)  # "RAG（检索增强生成）的含义是什么？"

# 多版本重写
multiple_queries = rewriter.multi_rewrite("RAG和微调的区别", n_versions=3)
# ["RAG与微调（Fine-tuning）的区别", "检索增强生成vs大模型微调", "RAG和Fine-tuning各有什么优缺点"]
```

### 2. HyDE（Hypothetical Document Embeddings）

HyDE通过让LLM先根据问题生成假设性文档，再用这个假设文档去检索，从而捕获更抽象的语义意图。

**原理：**
1. 给定问题Q，让LLM生成一个假设性的回答文档D_hypo
2. 使用D_hypo的Embedding去向量数据库检索
3. 返回与假设文档最相似的真实文档

**核心代码示例：**

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# HyDE文档生成提示
HYDE_PROMPT = PromptTemplate(
    template="""基于以下问题，生成一个假设性的回答文档。这个文档应该：
1. 直接回答问题
2. 包含可能的相关细节和解释
3. 使用准确、专业的表述

问题: {query}

假设性回答文档:""",
    input_variables=["query"]
)

class HyDERetriever:
    def __init__(self, vectorstore, llm, embeddings):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.hyde_chain = LLMChain(llm=llm, prompt=HYDE_PROMPT)

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        HyDE检索流程
        """
        # Step 1: 生成假设性文档
        hypothetical_doc = self.hyde_chain.invoke({"query": query})
        hypo_text = hypothetical_doc["text"]

        # Step 2: 将假设文档向量化
        hypo_vector = self.embeddings.embed_query(hypo_text)

        # Step 3: 用假设文档的向量检索
        results = self.vectorstore.similarity_search_by_vector(
            embedding=hypo_vector,
            k=k
        )

        return results

    def retrieve_with_fallback(self, query: str, k: int = 5) -> list:
        """
        HyDE + 原始Query混合检索
        """
        # HyDE检索结果
        hyde_results = self.retrieve(query, k)

        # 原始查询检索结果
        original_results = self.vectorstore.similarity_search(query, k=k)

        # 合并去重
        seen_ids = set()
        merged = []
        for doc in hyde_results + original_results:
            if doc.metadata.get("id") not in seen_ids:
                seen_ids.add(doc.metadata.get("id"))
                merged.append(doc)

        return merged[:k]

# 使用示例
hyde_retriever = HyDERetriever(
    vectorstore=vectorstore,
    llm=ChatOpenAI(model="gpt-4"),
    embeddings=embeddings
)

results = hyde_retriever.retrieve("为什么RAG比微调更适合知识更新频繁的场景？")
```

### 3. 混合检索（Hybrid Search）

混合检索结合关键词检索（如BM25）与向量检索的优势，提升检索的精准性和语义理解能力。

**为什么需要混合检索：**
- 向量检索对语义相似但词汇不同的问题效果好
- 关键词检索对精确匹配、专业术语效果好
- 两者结合可以互补短板

**核心代码示例：**

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vectorstore, documents, embeddings):
        self.vectorstore = vectorstore
        self.documents = documents
        self.embeddings = embeddings

    def create_bm25_retriever(self, k: int = 5):
        """创建BM25检索器"""
        # 准备文本语料库
        texts = [doc.page_content for doc in self.documents]

        # 初始化BM25
        bm25 = BM25Okapi(texts)

        # 创建LangChain BM25检索器
        retriever = BM25Retriever(
            index=bm25,
            documents=self.documents,
            k=k
        )
        return retriever

    def create_ensemble_retriever(
        self,
        k: int = 5,
        weights: list = [0.5, 0.5]
    ):
        """
        创建融合检索器
        weights: [向量检索权重, BM25权重]
        """
        # 向量检索器
        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # BM25检索器
        bm25_retriever = self.create_bm25_retriever(k=k)

        # 融合检索器
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights
        )

        return ensemble

    def reciprocal_rank_fusion(
        self,
        results_list: list,
        k: int = 60
    ) -> list:
        """
        倒数排序融合（RRF）算法
        """
        doc_scores = {}

        for results in results_list:
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get("id", doc.page_content[:100])
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0}
                # RRF公式: 1 / (k + rank)
                doc_scores[doc_id]["score"] += 1 / (k + rank)

        # 按分数排序
        ranked = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [item["doc"] for item in ranked]

# 使用示例
hybrid_retriever = HybridRetriever(vectorstore, chunks, embeddings)

# 方法1: 使用EnsembleRetriever
ensemble_retriever = hybrid_retriever.create_ensemble_retriever(
    k=5,
    weights=[0.6, 0.4]  # 向量检索60%，BM25 40%
)
results = ensemble_retriever.invoke("RAG的工作原理")

# 方法2: 手动RRF融合
vector_results = vectorstore.similarity_search("RAG的工作原理", k=20)
bm25_results = hybrid_retriever.create_bm25_retriever(k=20).invoke("RAG的工作原理")
fused_results = hybrid_retriever.reciprocal_rank_fusion(
    [vector_results, bm25_results],
    k=60
)
```

### 4. 重排序（Re-ranking）

重排序是在初步检索后使用更精准的模型对结果进行二次排序，提升相关性。

**常见重排序模型：**

| 模型 | 类型 | 特点 |
|------|------|------|
| Cohere Rerank | 云服务API | 效果好，需付费 |
| bge-reranker-large | 开源模型 | 中文支持好 |
| mpnet-rerank | 开源模型 | 通用性强 |
| LLM Rerank | LLM | 可解释性强 |

**核心代码示例：**

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int = 5
    ) -> list:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 初步检索到的文档列表
            top_k: 返回前k个最相关文档

        Returns:
            重排序后的文档列表（包含相关性分数）
        """
        # 准备(query, document)对
        pairs = [
            (query, doc.page_content if hasattr(doc, 'page_content') else str(doc))
            for doc in documents
        ]

        # 获取相关性分数
        scores = self.model.predict(pairs)

        # 按分数排序并返回
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {"document": doc, "score": float(score)}
            for doc, score in doc_scores[:top_k]
        ]

class CohereReranker:
    """使用Cohere API的重排序器"""

    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int = 5,
        model: str = "rerank-english-v2.0"
    ) -> list:
        """
        使用Cohere Rerank API
        """
        doc_texts = [
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in documents
        ]

        response = self.client.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_k,
            model=model
        )

        results = []
        for result in response.results:
            results.append({
                "document": documents[result.index],
                "score": result.relevance_score
            })

        return results

# 使用示例
reranker = Reranker("BAAI/bge-reranker-large")

# 初步检索（召回阶段）
initial_results = vectorstore.similarity_search(query, k=20)

# 重排序（精排阶段）
reranked = reranker.rerank(query, initial_results, top_k=5)

print("重排序结果：")
for item in reranked:
    print(f"[Score: {item['score']:.4f}] {item['document'].page_content[:100]}...")
```

### 5. Self-RAG

Self-RAG是一种自适应检索与生成框架，通过LLM自己判断是否需要检索来减少不必要的检索开销。

**核心思想：**
- 在生成过程中插入特殊的反思标记
- LLM自行判断：是否需要检索？检索的内容是否相关？答案是否完整？

**反思标记：**

| 标记 | 含义 | 后续动作 |
|------|------|----------|
| [Retrieval] | 需要检索 | 执行检索 |
| [No Retrieval] | 不需要检索 | 直接生成 |
| [Relevant] | 检索内容相关 | 使用内容生成 |
| [Irrelevant] | 检索内容不相关 | 忽略或补充生成 |
| [Supported] | 内容支持答案 | 保持答案 |
| [Contradict] | 内容与答案矛盾 | 修正答案 |
| [Partially Supported] | 部分支持 | 补充说明 |

**核心代码示例：**

```python
from enum import Enum
from dataclasses import dataclass

class SelfRAGToken(Enum):
    RETRIEVAL = "[Retrieval]"
    NO_RETRIEVAL = "[No Retrieval]"
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    SUPPORTED = "[Supported]"
    CONTRADICT = "[Contradict]"
    PARTIALLY_SUPPORTED = "[Partially Supported]"

@dataclass
class SelfRAGResponse:
    """Self-RAG响应"""
    output_text: str
    tokens: list
    use_retrieval: bool
    is_supported: bool
    reflection_tokens: list

class SelfRAGChain:
    """
    Self-RAG实现
    基于自我反思的检索增强生成
    """

    def __init__(self, llm, retriever, generator_prompt):
        self.llm = llm
        self.retriever = retriever
        self.generator_prompt = generator_prompt

    def _create_reflection_prompt(self, query: str, generation: str = None) -> str:
        """创建反思提示"""
        if generation is None:
            return f"""判断以下问题是否需要检索外部知识来回答：

问题: {query}

需要检索吗？回答 [Retrieval] 或 [No Retrieval]："""
        else:
            return f"""评估以下回答是否得到了检索上下文的支持：

问题: {query}
回答: {generation}

判断：[Supported] / [Contradict] / [Partially Supported]："""

    def _is_relevant(self, query: str, context: str) -> bool:
        """判断检索内容是否相关"""
        relevance_prompt = f"""判断上下文是否与问题相关：

问题: {query}
上下文: {context}

回答 [Relevant] 或 [Irrelevant]："""
        response = self.llm.invoke([{"role": "user", "content": relevance_prompt}])
        return "[Relevant]" in response.content

    def invoke(self, query: str, max_iterations: int = 3) -> SelfRAGResponse:
        """执行Self-RAG流程"""

        tokens = []
        use_retrieval = False
        context_used = False
        final_text = ""

        for iteration in range(max_iterations):
            # Step 1: 判断是否需要检索
            reflection_prompt = self._create_reflection_prompt(query)
            reflection_response = self.llm.invoke([
                {"role": "user", "content": reflection_prompt}
            ])
            tokens.append(reflection_response.content)

            if "[Retrieval]" in reflection_response.content:
                use_retrieval = True

                # Step 2: 执行检索
                retrieved_docs = self.retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in retrieved_docs])

                # Step 3: 检查检索内容相关性
                if not self._is_relevant(query, context):
                    tokens.append("[Irrelevant]")
                    continue

                tokens.append("[Relevant]")

                # Step 4: 基于上下文生成
                gen_prompt = f"""{self.generator_prompt}

上下文：
{context}

问题：{query}

回答时使用以下格式标记：
- [Supported] - 你的回答基于上下文
- [Contradict] - 上下文与回答矛盾
- [Partially Supported] - 回答部分基于上下文

回答："""
                gen_response = self.llm.invoke([
                    {"role": "user", "content": gen_prompt}
                ])
                final_text = gen_response.content
                tokens.append(final_text)

                # Step 5: 检查支持度
                support_check = self._create_reflection_prompt(query, final_text)
                support_response = self.llm.invoke([
                    {"role": "user", "content": support_check}
                ])
                tokens.append(support_response.content)

                if "[Supported]" in support_response.content:
                    context_used = True
                    break
                elif "[Contradict]" in support_response.content:
                    # 重新检索或补充信息
                    continue

            else:
                # Step 6: 无需检索，直接生成
                gen_prompt = f"""{self.generator_prompt}

问题：{query}

回答："""
                gen_response = self.llm.invoke([
                    {"role": "user", "content": gen_prompt}
                ])
                final_text = gen_response.content
                tokens.append(final_text)
                break

        return SelfRAGResponse(
            output_text=final_text,
            tokens=tokens,
            use_retrieval=use_retrieval,
            is_supported=context_used,
            reflection_tokens=tokens
        )

# 使用示例
self_rag = SelfRAGChain(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(k=5),
    generator_prompt="你是一个有帮助的AI助手。"
)

result = self_rag.invoke("RAG的完整工作流程是什么？")
print(result.output_text)
print(f"使用了检索: {result.use_retrieval}")
print(f"反思标记: {result.tokens}")
```

---

## 应用场景

### 1. 企业级知识问答

- Query重写处理口语化问题
- HyDE处理抽象概念查询
- 混合检索处理专业术语

### 2. 医疗/法律领域

- Self-RAG减少幻觉风险
- 重排序确保专业准确性
- 混合检索处理专业术语

### 3. 实时问答系统

- Self-RAG减少不必要的检索延迟
- Query重写提升用户体验
- 重排序提升答案质量

---

## 高级RAG技术对比

| 技术 | 解决问题 | 额外开销 | 效果提升 |
|------|----------|----------|----------|
| Query重写 | 查询表达不清 | 1次LLM调用 | 中等 |
| HyDE | 语义鸿沟 | 1次LLM调用 | 高 |
| 混合检索 | 精确/语义互补 | 索引+检索 | 高 |
| 重排序 | 检索精度 | 模型推理 | 高 |
| Self-RAG | 减少不必要检索 | 可变LLM调用 | 中高 |

---

## 相关概念

- **Query扩展**: 将单一查询扩展为多个相关查询
- **后退提示 (Step-back Prompting)**: 先抽象再具体的检索策略
- **IR-CoT (推理链)**: 多步推理的检索增强
- **FLARE**: 前瞻性主动检索生成
- **SELF-RAG**: 自我反思检索增强生成

---

## 延伸阅读

1. [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496) - HyDE论文
2. [SELF-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) - Self-RAG论文
3. [Query Rewriting in Retrieval-Augmented Open-Domain Question Answering](https://arxiv.org/abs/2212.10496) - Query重写论文
4. [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding) - 开源重排序模型
5. [Advanced RAG Techniques: An Overview](https://arxiv.org/abs/2401.05856) - 高级RAG综述
