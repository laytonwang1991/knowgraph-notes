---
title: Few-shot学习
alias: Few-shot Learning
tags: [LLM, Prompt Engineering, In-context Learning, 小样本学习]
category: AI Agent
created: 2026-03-31
updated: 2026-03-31
author: Claude
description: 通过在 prompt 中提供少量示例，让大语言模型从中学习模式并生成符合预期输出的技术，涵盖示例选择、示例格式、动态示例以及 K-shot 与 Zero-shot 的对比。
mastery: 9
rating: 9
related_concepts: [Prompt工程, Zero-shot, In-context Learning, CoT, K-shot]
difficulty: 中等
read_time: 18分钟
prerequisites: [LLM基础, Prompt工程基础]
---

# Few-shot Learning

## 一句话定义

Few-shot 学习（少样本学习）是在 prompt 中提供少量输入-输出示例，让大语言模型从中推断出任务模式和规则的技术，属于 In-context Learning（上下文学习）的范畴。

## 详细说明

### 1. 核心概念

Few-shot Learning 的本质是**不更新模型权重**，而是通过示例来"激活"模型已学到的知识。与传统的监督学习不同，Few-shot 依赖的是模型在预训练阶段积累的世界知识和泛化能力。

**关键特点：**
- 无需梯度更新或微调
- 示例作为情境信息输入
- 模型从示例中推断任务模式
- 效果高度依赖示例质量

### 2. 示例选择

示例的选择对 Few-shot 的效果有决定性影响。

**选择策略：**

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| 随机选择 | 随机选取示例 | 基线测试、初步探索 |
| 相似选择 | 选择与输入相似的示例 | 任务明确、分布清晰 |
| 多样性选择 | 确保示例覆盖不同情况 | 复杂任务、边界 case |
| 困难选择 | 选择模型容易出错的典型 | 针对性优化 |

**相似性度量方法：**
- 语义相似度（使用 embedding 模型）
- 词重叠度（BM25、TF-IDF）
- 任务特定特征相似度

### 3. 示例格式

示例的格式直接影响模型对任务的理解。

**常用格式：**

```
输入: [具体输入]
输出: [具体输出]

---
输入: [具体输入]
输出: [具体输出]
```

**格式设计原则：**
- 保持示例之间格式一致
- 使用清晰的输入输出分隔符
- 示例数量适中（通常 1-10 个）
- 示例应覆盖主要模式

### 4. 动态示例

动态 Few-shot 是在推理时根据当前输入动态选择最相关的示例。

**实现方式：**

1. **语义检索**：使用 embedding 模型计算输入与示例的相似度
2. **最近邻选择**：选择 top-k 最相似的示例
3. **混合策略**：结合多样性选择和相似性选择

### 5. K-shot vs Zero-shot

| 维度 | Zero-shot | Few-shot |
|------|-----------|----------|
| 示例数量 | 0 | 1-N |
| 提示复杂度 | 高（需要详细描述） | 低（示例即说明） |
| 任务适配性 | 需要显式说明任务 | 从示例中推断 |
| 适用场景 | 新奇任务、模型强 | 模式明确的任务 |
| 计算成本 | 较低 | 较高（示例占用 token） |

**K-shot 的 K 值选择：**
- K=1（One-shot）：最小示例，适合简单任务
- K=3-5（Few-shot）：常用配置，平衡效果和成本
- K>10：通常效果收益递减，可能引入噪声

## 代码示例

### 示例 1：基础 Few-shot

```python
from openai import OpenAI

client = OpenAI()

def sentiment_classifier(text: str) -> str:
    """情感分类的 Few-shot 示例"""

    prompt = """对以下文本进行情感分类，只能输出 Positive、Negative 或 Neutral。

示例 1：
输入：我今天心情特别好，阳光明媚！
输出：Positive

示例 2：
输入：这产品太差了，完全是浪费钱。
输出：Negative

示例 3：
输入：今天吃了米饭。
输出：Neutral

现在请分类：
输入：{text}
输出：""".format(text=text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0
    )

    return response.choices[0].message.content.strip()

# 测试
print(sentiment_classifier("这个电影太精彩了！"))  # Positive
print(sentiment_classifier("服务态度极其恶劣"))    # Negative
```

### 示例 2：动态 Few-shot（基于语义相似度）

```python
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

# 示例库
examples = [
    {"input": "这个手机拍照很清晰，电池也很耐用", "output": "Positive"},
    {"input": "耳机音质一般，降噪效果不好", "output": "Negative"},
    {"input": "今天天气不错", "output": "Neutral"},
    {"input": "物流速度很快，第二天就到了", "output": "Positive"},
    {"input": "包装破损，产品有划痕", "output": "Negative"},
    {"input": "收到了，确认收货", "output": "Neutral"},
]

def get_embedding(text: str) -> list[float]:
    """获取文本的 embedding"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def select_top_k_examples(query: str, examples: list[dict], k: int = 3) -> list[dict]:
    """基于语义相似度选择 top-k 示例"""

    # 计算查询与所有示例的相似度
    query_emb = get_embedding(query)
    example_embs = [get_embedding(ex["input"]) for ex in examples]

    # 计算余弦相似度
    similarities = cosine_similarity([query_emb], example_embs)[0]

    # 选择 top-k
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    return [examples[i] for i in top_k_idx]

def build_few_shot_prompt(query: str, examples: list[dict]) -> str:
    """构建 Few-shot prompt"""
    prompt = "判断以下文本的情感：Positive（正面）、Negative（负面）或 Neutral（中性）。\n\n"

    for ex in examples:
        prompt += f"输入：{ex['input']}\n输出：{ex['output']}\n\n"

    prompt += f"输入：{query}\n输出："
    return prompt

def dynamic_few_shot_classify(text: str) -> str:
    """动态 Few-shot 情感分类"""

    # 选择最相关的示例
    selected = select_top_k_examples(text, examples, k=3)

    # 构建 prompt
    prompt = build_few_shot_prompt(text, selected)

    # 调用模型
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )

    return response.choices[0].message.content.strip()

# 测试
print(dynamic_few_shot_classify("手机屏幕碎了，很失望"))  # Negative
print(dynamic_few_shot_classify("新到的裙子很漂亮"))     # Positive
```

### 示例 3：Few-shot + CoT（Chain of Thought）

```python
def math_reasoner(problem: str) -> str:
    """Few-shot + CoT 数学推理"""

    prompt = """请逐步推理并给出答案。

示例 1：
问题：小明有 12 个苹果，给了小红 5 个，又买了 8 个，现在有多少个？
推理：起始有 12 个苹果，给出 5 个后剩 12-5=7 个，又买了 8 个，所以 7+8=15 个。
答案：15

示例 2：
问题：一辆汽车以 60 公里/小时的速度行驶 2.5 小时，走了多少公里？
推理：距离 = 速度 × 时间 = 60 × 2.5 = 150 公里。
答案：150

示例 3：
问题：一个班级有 45 人，其中 60% 是男生，男生有多少人？
推理：男生人数 = 45 × 60% = 45 × 0.6 = 27 人。
答案：27

现在请推理：
问题：{problem}
推理：""".format(problem=problem)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )

    return response.choices[0].message.content.strip()

# 测试
print(math_reasoner("小张有 200 元，买了 3 本书，每本 35 元，还剩多少元？"))
```

### 示例 4：K-shot 对比实验

```python
def compare_kshot_performance(test_cases: list[tuple[str, str]], k_values: list[int]) -> dict:
    """对比不同 K 值的 Few-shot 效果"""

    base_examples = [
        {"input": "这件衣服质量很好", "output": "Positive"},
        {"input": "太差劲了，完全是假货", "output": "Negative"},
        {"input": "收到了", "output": "Neutral"},
        {"input": "性价比超高，推荐购买", "output": "Positive"},
        {"input": "等了半个月才到，太慢了", "output": "Negative"},
    ]

    results = {}

    for k in k_values:
        correct = 0
        total = len(test_cases)

        for text, expected in test_cases:
            # 选择前 k 个示例
            selected = base_examples[:k]

            # 构建 prompt
            prompt = "判断情感（Positive/Negative/Neutral）：\n"
            for ex in selected:
                prompt += f"{ex['input']} -> {ex['output']}\n"
            prompt += f"{text} -> "

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )

            predicted = response.choices[0].message.content.strip()
            if expected in predicted or predicted in expected:
                correct += 1

        results[k] = correct / total

    return results

# 运行对比
test_data = [
    ("非常满意，五星好评", "Positive"),
    ("失望透顶，不会再买了", "Negative"),
    ("还行吧，一般般", "Neutral"),
]

print(compare_kshot_performance(test_data, [1, 3, 5]))
```

## 应用场景

| 场景 | K 值建议 | 说明 |
|------|----------|------|
| 简单分类 | 1-3 | 模式简单，少量示例即可 |
| 复杂推理 | 5-10 | 需要多种模式覆盖 |
| 代码生成 | 3-5 | 需要符合特定代码风格 |
| 翻译任务 | 3-5 | 需要符合目标语言习惯 |
| 格式转换 | 1-3 | 格式明确，示例直观 |

## 相关概念

- **Zero-shot**：不使用示例，完全依赖模型泛化能力
- **In-context Learning**：上下文学习的总称，Few-shot 是其子集
- **Chain of Thought (CoT)**：思维链，可以与 Few-shot 结合使用
- **Prompt 工程**：Few-shot 是 Prompt 工程的重要技术
- **K-shot**：K 个示例的 Few-shot

## 延伸阅读

1. **"Language Models are Few-Shot Learners"（GPT-3 论文）** - 首次系统性地研究 Few-shot 学习的论文
2. **Prompt Engineering Guide** - 详细的各种 Prompt 技术指南
3. **Anthropic's Prompt Engineering Tutorial** - 官方 Prompt 工程教程
4. **"Fantastic Prompt Variables"** - 深入分析影响 Prompt 效果的各种因素
5. **"Rethinking the Role of Demonstrations"** - 重新审视 Few-shot 中示例的作用
