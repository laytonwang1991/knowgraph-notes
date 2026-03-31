---
title: LangChain开发
alias: LangChain Development
tags:
  - LangChain
  - LLM应用
  - AI Agent
  - RAG
  - 大语言模型
category: 编程实践
created: 2026-03-31
updated: 2026-03-31
author: AI Practice Team
description: 使用LangChain构建大语言模型应用的开发指南，涵盖Chains、Agents、Memory、Retrieval、Callback等核心组件的实践技巧。
mastery: 75
rating: 8
related_concepts:
  - 大语言模型
  - RAG
  - 向量数据库
  - Prompt Engineering
  - AI Agent
  - 提示词工程
difficulty: 中级
read_time: 55分钟
prerequisites:
  - Python基础
  - 大语言模型基础
  - API调用基础
  - 向量数据库概念
---

# LangChain开发

## 一句话定义

LangChain是一个用于构建大语言模型（LLM）应用的开发框架，通过组件化的方式组合Chains（链）、Agents（代理）、Memory（记忆）和Retrieval（检索）模块，使开发者能够快速构建复杂的AI应用，如问答系统、聊天机器人和自动化工作流。

## 详细说明

### 1. Chains（链）

Chain是LangChain的核心抽象，将多个组件串联起来完成复杂任务：

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# 初始化LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key="your-api-key"
)

# 基础链：提示词 + LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的{subject}顾问"),
    ("human", "请用简洁的语言解释{concept}，不超过50字")
])

chain = LLMChain(llm=llm, prompt=prompt)

# 执行链
result = chain.run({"subject": "物理学", "concept": "量子纠缠"})
print(result)

# 顺序链：将多个链串联执行
chain1 = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一位新闻记者"),
        ("human", "请用一句话总结以下事件：{event}")
    ]),
    output_key="summary"
)

chain2 = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一位社评家"),
        ("human", "基于以下新闻摘要，写一篇简短的评论：\n{summary}")
    ]),
    output_key="commentary"
)

sequential_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

result = sequential_chain.run("今天苹果公司发布了新一代iPhone")

# 路由链：根据输入选择不同的处理路径
physics_template = """
你是一位物理学教授。请解释以下物理概念：
{input}
"""

math_template = """
你是一位数学教授。请解释以下数学概念：
{input}
"""

prompt_templates = {
    "physics": physics_template,
    "math": math_template
}

# 创建多提示链
from langchain.chains import LLMMathChain
from langchain.agents import initialize_agent, Tool

# 自定义路由逻辑
router_chain = LLMRouterChain.from_llm(
    llm=llm,
    routing_prompts=[
        {"name": "physics", "description": "物理相关问题", "prompt_template": physics_template},
        {"name": "math", "description": "数学相关问题", "prompt_template": math_template}
    ]
)
```

### 2. Agents（代理）

Agent能够自主决策并调用工具完成复杂任务：

```python
from langchain.agents import Agent, Tool, initialize_agent
from langchain.agents.types import AgentType
from langchain.tools import BaseTool
from langchain.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory

# 定义自定义工具
def calculate(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

def get_weather(city: str) -> str:
    """获取城市天气信息"""
    # 实际应用中调用天气API
    return f"{city}今天晴天，温度25度"

# 注册工具
tools = [
    Tool(
        name="Calculator",
        func=calculate,
        description="用于数学计算，输入应该是数学表达式，如：2+2、sin(45)等"
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="获取城市天气，输入应该是城市名称"
    ),
    Tool(
        name="Search",
        func=SerpAPIWrapper(serpapi_api_key="your-key").run,
        description="搜索网络信息，输入应该是搜索关键词"
    ),
    Tool(
        name="Wikipedia",
        func=WikipediaAPIWrapper().run,
        description="查询维基百科，输入应该是要查询的词条"
    )
]

# 初始化Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

# 执行任务
result = agent.run("北京的天气如何？顺便帮我计算一下(12 + 8) * 3")
print(result)

# 自定义Agent类
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate

PREFIX = """你是一个乐于助人的AI助手。你可以使用以下工具来回答问题："""
FORMAT_INSTRUCTIONS = """使用以下格式：

问题：你需要回答的输入问题
思考：你应该总是思考下一步
行动：要采取的行动，应该是 {tool_names} 之一
行动输入：行动的输入
观察：行动的结果
...（这个思考/行动/行动输入/观察循环可以重复多次）
最终答案：你作为代理生成的最终答案"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=PREFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad"]
)

# ReAct Agent的实现
from langchain.agents import AgentExecutor, Tool
from langchain.tools import StructuredTool

def search_entity(entity_name: str) -> str:
    """搜索实体信息"""
    return f"关于{entity_name}的信息：这是一个重要概念..."

search_tool = StructuredTool.from_function(
    func=search_entity,
    name="EntitySearch",
    description="搜索实体相关信息"
)

tools = [search_tool]
```

### 3. Memory（记忆）

Memory组件让Agent能够记住对话历史：

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
    CombinedMemory
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(temperature=0)

# 缓冲记忆：保存完整对话历史
buffer_memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    output_key="response"
)

# 令牌缓冲记忆：限制记忆长度（按token数）
token_memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=1000,
    memory_key="history",
    return_messages=True
)

# 摘要记忆：自动总结历史对话
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="history",
    return_messages=True
)

# 组合记忆：同时使用多种记忆
combined_memory = CombinedMemory(
    memories=[
        ConversationBufferMemory(memory_key="last_20_messages", return_messages=True),
        ConversationSummaryMemory(memory_key="summary", llm=llm)
    ]
)

# 对话链使用记忆
conversation = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

# 多轮对话示例
conversation.run("我叫张三，是一名软件工程师")
conversation.run("我的职业是什么？")
conversation.run("我刚才说我叫什么名字？")

# 保存和加载记忆
import pickle

# 保存记忆
with open("memory.pkl", "wb") as f:
    pickle.dump(buffer_memory, f)

# 加载记忆
with open("memory.pkl", "rb") as f:
    loaded_memory = pickle.load(f)

# 在新的对话链中使用
new_conversation = ConversationChain(
    llm=llm,
    memory=loaded_memory
)
result = new_conversation.run("还记得我叫什么名字吗？")
print(result)
```

### 4. Retrieval（检索）

Retrieval组件实现RAG（检索增强生成）模式：

```python
from langchain.document_loaders import TextLoader, PDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 1. 文档加载
loader = DirectoryLoader(
    "./docs",
    glob="**/*.md",
    loader_cls=TextLoader
)
documents = loader.load()

# 2. 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "，", " "]
)
texts = text_splitter.split_documents(documents)

# 3. 生成embeddings并存储到向量数据库
embeddings = OpenAIEmbeddings(openai_api_key="your-api-key")

# Chroma向量数据库
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./vectorstore"
)
vectorstore.persist()

# FAISS向量数据库（适合大规模数据）
faiss_vectorstore = FAISS.from_documents(
    documents=texts,
    embedding=embeddings
)

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. 检索增强问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 可选: stuff, map_reduce, refine
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# 执行问答
result = qa_chain({"query": "这份文档的主要内容是什么？"})
print(result["result"])
print(f"来源文档: {result['source_documents']}")

# 6. 带压缩的检索器（提升检索质量）
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 7. 多向量存储检索
from langchain.storage import InMemoryStore
from langchain.retrievers import MultiVectorRetriever

# 父文档存储（完整文档）
parent_store = InMemoryStore()

multi_vector_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=parent_store,
    id_key="doc_id"
)

# 为每个块生成摘要并存储
from langchain.output_parsers import StrOutputParser
summary_chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_template("总结：{text}"))
```

### 5. Callback（回调）

Callback机制用于监控和记录LangChain执行过程：

```parameter name="content">python
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler, FileCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import logging

# 自定义回调处理器
class CustomCallbackHandler(BaseCallbackHandler):
    """自定义回调：记录LLM调用和Agent执行"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM开始调用，提示词数量: {len(prompts)}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM调用完成，耗时: {kwargs.get('elapsed', 'unknown')}")

    def on_llm_error(self, error, **kwargs):
        print(f"LLM调用错误: {error}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"链开始执行: {serialized.get('name', 'unnamed')}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"链执行完成，输出: {outputs}")

    def on_chain_error(self, error, **kwargs):
        print(f"链执行错误: {error}")

    def on_agent_action(self, action, **kwargs):
        print(f"Agent执行动作: {action.tool}")
        print(f"动作输入: {action.tool_input}")

    def on_agent_finish(self, finish, **kwargs):
        print(f"Agent完成任务: {finish.return_values}")

# 使用回调处理器
handler = CustomCallbackHandler()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位AI助手"),
    ("human", "{input}")
])

chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])

result = chain.run("你好，请介绍一下自己", callbacks=[handler])

# 多回调处理器
std_out_handler = StdOutCallbackHandler()
file_handler = FileCallbackHandler("execution.log")

from langchain.callbacks import CombinedCallbackHandler
combined_handler = CombinedCallbackHandler([std_out_handler, file_handler])

# 在Agent中使用回调
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    callbacks=[combined_handler]
)

# Streaming回调（实时输出）
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

streaming_chain = LLMChain(
    llm=ChatOpenAI(
        model="gpt-4",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    ),
    prompt=prompt
)

result = streaming_chain.run("给我讲一个关于AI的科幻故事")
```

## 代码示例

### 完整的RAG问答系统

```python
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. 加载文档
print("加载文档...")
loader = DirectoryLoader(
    "./knowledge_base",
    glob="**/*.txt",
    show_progress=True
)
documents = loader.load()

# 2. 分块
print(f"文档数量: {len(documents)}")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
texts = splitter.split_documents(documents)
print(f"分块后数量: {len(texts)}")

# 3. 向量化存储
print("生成embeddings...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("向量数据库已保存")

# 4. 创建检索QA链
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 自定义提示词
CUSTOM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库问答助手。请根据以下参考材料回答用户问题。
    如果参考材料中没有相关信息，请如实告知，不要编造答案。

    参考材料：
    {context}"""),
    ("human", "{question}")
])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    return_source_documents=True
)

# 5. 问答
def ask_question(question: str):
    print(f"\n问题: {question}")
    result = qa_chain({"query": question})
    print(f"回答: {result['result']}")
    print(f"参考来源: {len(result['source_documents'])}个文档")
    return result

# 交互式问答
if __name__ == "__main__":
    while True:
        question = input("\n请输入问题（输入q退出）: ")
        if question.lower() == 'q':
            break
        ask_question(question)
```

## 应用场景

1. **智能问答系统**：基于私有知识库的RAG问答、企业内部知识检索
2. **聊天机器人**：多轮对话、任务型对话、情感陪伴
3. **文档处理**：自动摘要、内容分析、信息提取
4. **数据分析**：自然语言查询数据库、数据可视化生成
5. **自动化工作流**：多步骤任务编排、定时任务执行
6. **代码助手**：代码生成、代码审查、Bug修复建议
7. **Agent应用**：自动研究助理、个人效率工具、自动化代理

## 相关概念

| 概念 | 说明 |
|------|------|
| Chain | 将多个组件串联执行的抽象 |
| Agent | 能够自主决策和调用工具的智能体 |
| Memory | 管理对话历史和上下文的组件 |
| Retrieval | 检索增强生成（RAG）的核心 |
| Callback | 监控和记录执行过程的钩子 |
| Vector Store | 存储和检索向量嵌入的数据库 |
| Prompt Template | 结构化提示词模板 |
| LCEL | LangChain Expression Language，声明式组合链的语法 |

## 延伸阅读

1. [LangChain官方文档](https://docs.langchain.com/) - 完整的LangChain文档
2. [LangChain GitHub](https://github.com/hwchase17/langchain) - 源代码和示例
3. [LangChain Hub](https://smith.langchain.com/hub) - 预构建提示词和链
4. [LangSmith](https://docs.smith.langchain.com/) - LangChain调试和监控平台
5. [RAG论文](https://arxiv.org/abs/2005.11401) - 检索增强生成原始论文
6. [ LlamaIndex](https://www.llamaindex.ai/) - 另一主流RAG框架
7. [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - 自主Agent示例
