---
title: Prompt优化
alias: Prompt Optimization
tags:
  - Prompt工程
  - LLM优化
  - Prompt压缩
  - Prompt缓存
category: AI基础设施
created: 2026-03-31
updated: 2026-03-31
author: AI基础设施团队
description: 系统性提升Prompt效果的方法论，涵盖Prompt压缩、Prompt缓存、Prompt版本管理和Prompt调优等核心技术。
mastery: 9
rating: 9
related_concepts:
  - 推理服务架构
  - LLM工厂
  - 模型推理优化
difficulty: 中高
read_time: 20分钟
prerequisites:
  - LLM基本概念
  - API调用基础
  - JSON数据处理
---

# Prompt优化

## 一句话定义

Prompt优化是通过系统性方法提升大语言模型输出质量、降低推理成本和延迟的技术集合，涵盖压缩、缓存、版本管理和自动化调优等策略。

## 详细说明

### 1. Prompt压缩（Prompt Compression）

Prompt压缩在保持核心信息的前提下减少token数量，直接降低推理成本和延迟。

#### 压缩技术分类

**a) 文本压缩**
- 移除冗余修饰词和重复表达
- 使用缩写和代号替代长描述
- 提取关键信息，删除无效上下文

**b) 结构化压缩**
- 将自然语言转为JSON/XML结构
- 使用模板替代描述性语言
- 提取少样本示例的核心模式

**c) 学习式压缩**
- 使用小模型学习压缩策略
- 训练专用压缩器（Contrastive Learning）
- 基于信息论的压缩算法

#### 代码示例：文本压缩

```python
import re

def compress_prompt(prompt: str, preserve_format=True) -> str:
    """基础Prompt压缩"""
    # 移除多余空格
    prompt = re.sub(r'\s+', ' ', prompt).strip()

    if not preserve_format:
        # 移除换行符
        prompt = prompt.replace('\n', ' ')

    # 常用缩写映射
    abbreviations = {
        'please': 'pls',
        'thank you': 'thx',
        'you are': "ur",
        'for example': 'e.g.',
        'that is': 'i.e.',
    }

    for full, abbr in abbreviations.items():
        prompt = prompt.lower().replace(full, abbr)

    return prompt

def smart_compress(prompt: str, context: dict) -> str:
    """基于上下文的智能压缩"""
    # 提取关键实体
    key_entities = context.get('entities', [])

    # 用ID替代已知的实体描述
    compressed = prompt
    for entity in key_entities:
        full_desc = entity['description']
        entity_id = entity['id']
        compressed = compressed.replace(full_desc, f"[{entity_id}]")

    return compressed
```

#### 代码示例：结构化压缩

```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class StructuredPrompt:
    """结构化Prompt模板"""
    role: str
    task: str
    constraints: List[str]
    examples: List[dict]
    output_format: str

    def to_prompt(self) -> str:
        parts = [f"Role: {self.role}", f"Task: {self.task}"]

        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")

        if self.examples:
            parts.append("Examples:")
            for ex in self.examples:
                parts.append(f"  Input: {ex['input']}")
                parts.append(f"  Output: {ex['output']}")

        parts.append(f"Output Format: {self.output_format}")

        return "\n".join(parts)

    def to_json(self) -> str:
        """转为JSON格式，减少token数"""
        return json.dumps({
            "r": self.role,
            "t": self.task,
            "c": self.constraints,
            "e": self.examples,
            "o": self.output_format
        }, ensure_ascii=False)
```

### 2. Prompt缓存（Prompt Caching）

Prompt缓存通过复用相同前缀的计算结果，大幅提升推理效率。

#### 缓存策略

**a) 前缀缓存**
- 识别多个请求的共同前缀
- 缓存前缀的KV Cache
- 新请求复用已计算的prefix

**b) 语义缓存**
- 基于语义相似度缓存
- 支持轻微变体的prompt
- 使用向量数据库存储

**c) 层级缓存**
```
L1: 内存缓存 (最快，容量小)
L2: 本地SSD缓存 (中等)
L3: 分布式缓存 (最慢，容量大)
```

#### 代码示例：语义缓存

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import redis
import json

class SemanticPromptCache:
    def __init__(self, similarity_threshold=0.95):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
        self.threshold = similarity_threshold
        self._init_index()

    def _init_index(self):
        """初始化向量索引"""
        # 使用Faiss构建向量索引
        import faiss
        self.index = faiss.IndexFlatIP(384)  # MiniLM维度
        self.prompt_ids = []
        self.prompt_texts = {}

    def get(self, prompt: str) -> Optional[str]:
        """查询缓存"""
        # 编码查询
        query_vec = self.encoder.encode([prompt]).astype('float32')

        # 搜索最近邻
        scores, indices = self.index.search(query_vec, k=1)

        if indices[0][0] != -1 and scores[0][0] >= self.threshold:
            cached_id = self.prompt_ids[indices[0][0]]
            return self.cache.get(cached_id)

        return None

    def set(self, prompt: str, response: str):
        """写入缓存"""
        prompt_id = f"prompt:{hash(prompt)}"

        # 存储响应
        self.cache.set(prompt_id, response)

        # 更新向量索引
        vec = self.encoder.encode([prompt]).astype('float32')
        self.index.add(vec)
        self.prompt_ids.append(prompt_id)
        self.prompt_texts[prompt_id] = prompt

    def invalidate(self, pattern: str = "*"):
        """清除缓存"""
        for key in self.cache.scan_iter(pattern):
            self.cache.delete(key)
```

#### 代码示例：vLLM前缀缓存

```python
from vllm import LLM, SamplingParams

# vLLM自动支持前缀缓存
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    enable_prefix_caching=True,  # 启用前缀缓存
)

# 相同系统prompt的不同用户query
system_prompt = "You are a helpful assistant. Context: ..."

requests = [
    (system_prompt + "What is Python?", ...),
    (system_prompt + "What is Java?", ...),
    (system_prompt + "What is Go?", ...),
]

# vLLM自动复用system_prompt的KV Cache
outputs = llm.generate(requests)
```

### 3. Prompt版本管理（Prompt Versioning）

企业级应用需要系统化管理Prompt的多个版本。

#### 版本管理策略

**a) 结构化版本**
```
prompt_versions/
├── v1.0/
│   ├── system.txt
│   ├── user_template.txt
│   └── config.yaml
├── v1.1/
│   ├── system.txt
│   ├── user_template.txt
│   └── config.yaml
└── v2.0/
    └── ...
```

**b) 数据库存储**
```sql
CREATE TABLE prompt_versions (
    id SERIAL PRIMARY KEY,
    prompt_key VARCHAR(100),
    version VARCHAR(20),
    system_prompt TEXT,
    user_template TEXT,
    variables JSONB,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);
```

#### 代码示例：版本管理

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

@dataclass
class PromptVersion:
    version: str
    system_prompt: str
    user_template: str
    variables: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: datetime
    created_by: str

class PromptVersionManager:
    def __init__(self, storage_backend):
        self.storage = storage_backend

    def create_version(
        self,
        prompt_key: str,
        system_prompt: str,
        user_template: str,
        variables: Dict[str, Any],
        created_by: str
    ) -> PromptVersion:
        """创建新版本"""
        # 生成版本号
        latest = self.get_latest(prompt_key)
        version = self._increment_version(latest.version if latest else "0.0")

        prompt_hash = hashlib.sha256(
            (system_prompt + user_template).encode()
        ).hexdigest()[:8]

        version_str = f"{version}-{prompt_hash}"

        prompt_version = PromptVersion(
            version=version_str,
            system_prompt=system_prompt,
            user_template=user_template,
            variables=variables,
            metrics={},
            created_at=datetime.now(),
            created_by=created_by
        )

        self.storage.save(prompt_key, version_str, prompt_version)
        return prompt_version

    def get_active(self, prompt_key: str) -> Optional[PromptVersion]:
        """获取当前激活版本"""
        return self.storage.get_active(prompt_key)

    def rollback(self, prompt_key: str, version: str) -> bool:
        """回滚到指定版本"""
        target = self.storage.get(prompt_key, version)
        if target:
            self.storage.set_active(prompt_key, version)
            return True
        return False

    def compare_versions(self, prompt_key: str, v1: str, v2: str) -> Dict:
        """对比两个版本"""
        pv1 = self.storage.get(prompt_key, v1)
        pv2 = self.storage.get(prompt_key, v2)

        return {
            "system_diff": self._text_diff(pv1.system_prompt, pv2.system_prompt),
            "template_diff": self._text_diff(pv1.user_template, pv2.user_template),
            "metrics_v1": pv1.metrics,
            "metrics_v2": pv2.metrics,
        }
```

### 4. Prompt调优（Prompt Tuning）

通过系统化实验找到最优Prompt配置。

#### 调优方法

**a) A/B测试**
```python
class PromptABTester:
    def __init__(self, llm_client, metrics_collector):
        self.llm = llm_client
        self.metrics = metrics_collector

    def run_experiment(
        self,
        prompt_key: str,
        variants: List[Dict],
        traffic_split: List[float],
        duration_hours: int
    ):
        """运行A/B测试"""
        assert abs(sum(traffic_split) - 1.0) < 0.001

        experiment = Experiment(
            prompt_key=prompt_key,
            variants=variants,
            split=traffic_split,
            start_time=datetime.now(),
            duration=duration_hours
        )

        # 分配流量
        for req_id, request in enumerate(self._collect_requests()):
            variant_idx = req_id % len(variants)
            variant = variants[variant_idx]

            response = self.llm.generate(
                self._render_prompt(variant, request)
            )

            self.metrics.record(
                experiment_id=experiment.id,
                variant_id=variant['id'],
                latency=response.latency,
                quality_score=self._evaluate(response),
                cost=response.tokens * COST_PER_TOKEN
            )

        return self._analyze_results(experiment.id)
```

**b) 自动Prompt优化**
```python
from openai import OpenAI
import anthropic

class AutoPromptOptimizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.evaluation_prompt = """Evaluate this prompt on:
1. Clarity (1-10)
2. Task completion (1-10)
3. Conciseness (1-10)
4. Consistency (1-10)

Prompt: {prompt}
Output: {output}
Expected: {expected}

Provide scores and improvement suggestions."""

    def optimize(
        self,
        initial_prompt: str,
        task_description: str,
        examples: List[Tuple[str, str]],
        iterations: int = 5
    ) -> str:
        """迭代优化Prompt"""
        current_prompt = initial_prompt

        for i in range(iterations):
            # 收集当前prompt的表现
            scores = []
            for input_text, expected in examples:
                prompt = current_prompt.format(input=input_text)
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )

                score = self._evaluate(
                    response.choices[0].message.content,
                    expected
                )
                scores.append(score)

            avg_score = sum(scores) / len(scores)

            if avg_score >= 9.0:
                break

            # 用LLM生成改进版本
            improvement_prompt = f"""Current prompt: {current_prompt}
Average score: {avg_score}/10
Task: {task_description}

Generate an improved version of this prompt that would score higher.
Consider:
- Clearer instructions
- Better examples
- More specific constraints"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": improvement_prompt}]
            )

            current_prompt = response.choices[0].message.content

        return current_prompt
```

## 应用场景

### 1. 高频调用场景

- **客服机器人**：大量相似query，前缀缓存效果显著
- **代码补全**：重复的上下文代码
- **文档摘要**：相同的文档结构

### 2. 成本敏感场景

- **大规模内容生成**：每token成本累积显著
- **A/B测试**：需要运行大量实验
- **模型微调数据生成**：合成数据批量生成

### 3. 质量敏感场景

- **Prompt版本管理**：生产环境需要可回滚
- **A/B测试**：精确对比不同策略
- **自动化调优**：追求最优效果

### 4. 复杂推理场景

- **Chain-of-Thought**：长推理链的Prompt优化
- **工具调用**：多步骤任务的Prompt设计
- **多模态**：图文混合输入的Prompt

## 相关概念

| 概念 | 说明 |
|------|------|
| Few-shot Learning | 通过示例教会模型任务模式 |
| Chain-of-Thought | 引导模型逐步推理的技术 |
| System Prompt | 系统级指令，定义模型行为 |
| Temperature | 控制输出随机性的参数 |
| Top-p Sampling | 核采样，控制输出多样性 |

## 延伸阅读

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [vLLM Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html)
- [Linguistic Acceptance in Prompt Compression](https://arxiv.org/abs/2310.00524)
- [Automated Prompt Optimization](https://arxiv.org/abs/2403.17780)
- [DSPy: Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714)
