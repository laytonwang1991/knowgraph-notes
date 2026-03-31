---
title: RAG工作流
alias: RAG-Workflow
tags: [RAG, LLM, 向量检索, NLP, 检索增强生成]
category: RAG应用
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 深入解析检索增强生成(RAG)的完整工作流程，从文档加载到答案生成的六个核心阶段。
mastery: 80
rating: 9
related_concepts: [Embedding, 向量数据库, LLM, 文本分割, 混合检索]
difficulty: 中级
read_time: 15分钟
prerequisites: [Python基础, NLP基础概念, 向量数学基础]
---

# RAG工作流

## 一句话定义

RAG（Retrieval Augmented Generation，检索增强生成）是一种结合向量检索与传统LLM生成能力的技术架构，通过先检索相关文档再生成答案的方式，提升大语言模型在知识密集型任务中的准确性和可信度。

---

## 详细说明

### 1. 文档加载（Document Loading）

文档加载是RAG管道的起点，负责将各种格式的原始文档转换为可处理的统一格式。

**支持的文档格式：**
- PDF文档（结构化/非结构化）
- Word文档（.docx）
- Markdown文件
- CSV/JSON表格数据
- 网页HTML
- 纯文本文件

**核心代码示例：**

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# 加载PDF文档
pdf_loader = PyPDFLoader("document.pdf")
pdf_documents = pdf_loader.load()

# 加载目录下的所有文档
loader = DirectoryLoader(
    path="./docs",
    glob="**/*.md",
    loader_cls=TextLoader
)
documents = loader.load()
```

**注意事项：**
- 大文档需要进行分页或分段处理
- 保留文档元数据（来源、页码、创建时间等）
- 特殊格式（如表格）需要专用解析器

### 2. 文本分割（Text Chunking）

文本分割是将长文档切分为较小语义单元的关键步骤直接影响检索质量。

**分割策略：**

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| 固定长度分割 | 按字符或词数硬切分 | 快速原型开发 |
| 递归字符分割 | 按层级分隔符递归切分 | 通用场景 |
| 语义分割 | 基于Embedding相似度 | 高质量需求 |
| 模型段落分割 | 按自然段落边界 | 保留语义完整 |

**核心代码示例：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 递归字符分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 每段字符数
    chunk_overlap=200,    # 段落重叠区域
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

# 打印分割结果
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk.page_content)} chars")
```

**最佳实践：**
- chunk_size建议在500-1500字符之间
- 保留15-20%的重叠以维持上下文连贯性
- 根据下游LLM的上下文窗口调整大小

### 3. Embedding向量化

将文本转换为稠密向量表示是实现语义检索的核心。

**主流Embedding模型：**

| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-ada-002 | 1536 | 效果好，付费 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 开源快速 |
| BGE-large-zh | 1024 | 中文优化 |
| m3e-large | 1024 | 中文开源 |

**核心代码示例：**

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# 使用OpenAI Embedding
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# 使用开源Embedding（中文优化）
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 向量化单个文本
query_vector = embeddings.embed_query("什么是RAG？")

# 批量向量化
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_vectors = embeddings.embed_documents(chunk_texts)
```

### 4. 向量存储（Vector Storage）

将向量与原始文本块存储到向量数据库中支持高效检索。

**主流向量数据库：**

| 数据库 | 向量维度 | 特点 |
|--------|----------|------|
| Chroma | 任意 | 轻量级，本地优先 |
| FAISS | 任意 | Facebook开源，索引丰富 |
| Milvus | 任意 | 分布式，云原生 |
| Pinecone | 任意 | 全托管，云服务 |
| Qdrant | 任意 | Rust实现，高性能 |
| Weaviate | 任意 | 混合检索强 |

**核心代码示例：**

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 初始化Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 持久化存储
vectorstore.persist()

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # 返回最相似的5个文档块
)
```

### 5. 向量检索（Vector Retrieval）

根据查询向量在向量数据库中找到最相关的文档块。

**检索策略：**

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| Similarity Search | 基于余弦相似度 | 通用语义匹配 |
| MMR (最大边际相关) | 平衡相关性与多样性 | 避免结果重复 |
| 混合检索 | 关键词+向量混合 | 精准匹配需求 |
| 过滤检索 | 带元数据条件 | 结构化查询 |

**核心代码示例：**

```python
# 基础相似度检索
results = vectorstore.similarity_search(
    query="RAG的工作原理是什么？",
    k=5
)

# 带相似度分数的检索
results_with_scores = vectorstore.similarity_search_with_score(
    query="RAG的工作原理是什么？",
    k=5
)

# MMR检索（减少重复增加多样性）
results_mmr = vectorstore.max_marginal_relevance_search(
    query="RAG的工作原理是什么？",
    k=5,
    fetch_k=20,  # 初始获取20个
    lambda_mult=0.5  # 多样性参数，0最多样，1最相关
)

# 打印检索结果
for doc, score in results_with_scores:
    print(f"[Score: {score:.4f}] {doc.page_content[:200]}...")
```

### 6. 生成阶段（Generation）

将检索到的文档与原始问题组合，提示LLM生成最终答案。

**核心代码示例：**

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 初始化LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3,
    api_key="your-api-key"
)

# 自定义提示模板
prompt_template = """基于以下上下文回答问题。如果上下文中没有相关信息，请如实说明。

上下文：
{context}

问题：{question}

回答："""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 创建RAG链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 执行查询
result = qa_chain({"query": "RAG的完整工作流程是什么？"})

print("答案：", result["result"])
print("\n来源文档：")
for doc in result["source_documents"]:
    print(f"- {doc.metadata.get('source', 'Unknown')}")
```

---

## 应用场景

### 1. 企业知识库问答

- 内部文档检索与问答
- 客服机器人知识增强
- 产品文档智能问答

### 2. 学术研究与文献分析

- 论文摘要与引用分析
- 研究趋势分析
- 实验数据问答

### 3. 代码开发辅助

- 代码文档问答
- API使用指南生成
- Bug排查知识库

### 4. 法律文档分析

- 合同条款检索
- 法规遵从性检查
- 判例分析

---

## 完整RAG管道示例

```python
"""
完整RAG工作流实现
"""
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

class RAGPipeline:
    def __init__(self, docs_path: str, model_name: str = "gpt-4"):
        self.docs_path = docs_path
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self):
        """加载文档"""
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.md",
            show_progress=True
        )
        return loader.load()

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """分割文档"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self, chunks):
        """创建向量数据库"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        return self.vectorstore

    def setup_qa_chain(self):
        """设置问答链"""
        llm = ChatOpenAI(model=self.model_name, temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

    def query(self, question: str):
        """执行查询"""
        return self.qa_chain({"query": question})

# 使用示例
if __name__ == "__main__":
    rag = RAGPipeline(docs_path="./docs")
    documents = rag.load_documents()
    chunks = rag.split_documents(documents)
    rag.create_vectorstore(chunks)
    rag.setup_qa_chain()

    result = rag.query("RAG的核心组件有哪些？")
    print(result["result"])
```

---

## 相关概念

- **Embedding**: 将文本转换为稠密向量的技术
- **向量数据库**: 专门存储和检索高维向量的数据库系统
- **LLM**: Large Language Model，大语言模型
- **Prompt Engineering**: 提示词工程，优化输入提示的技术
- **上下文窗口**: LLM单次能处理的最大token数

---

## 延伸阅读

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - 原始RAG论文
2. [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) - LangChain官方教程
3. [Vector Store Comparison](https://www.pinecone.io/docs/vector-stores/) - 向量数据库对比指南
4. [Advanced RAG Techniques](https://arxiv.org/abs/2312.10938) - 高级RAG技术综述
