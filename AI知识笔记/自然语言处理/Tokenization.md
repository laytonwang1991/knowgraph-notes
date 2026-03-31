---
title: Tokenization详解
alias: Tokenization
tags:
  - NLP
  - 分词
  - Tokenization
  - 词表
  - 子词
  - 编码
  - 解码
category: 自然语言处理
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 深入理解Tokenization的完整流程、词表设计考量、UNK问题及编码解码细节
mastery: 0
rating: 0
related_concepts:
  - 分词器
  - BPE
  - WordPiece
  - SentencePiece
  - 词表大小
  - UNK问题
  - 子词分词
  - 编码解码
difficulty: 中等
read_time: 20
prerequisites:
  - 自然语言处理基础
  - 概率统计基础
  - Python编程
---

# Tokenization详解

## 一句话定义

Tokenization是将文本字符串转换为模型可处理的Token整数序列的过程，是NLP管道的起点，涉及词表构建、分词算法选择和编码解码流程设计。

## 核心公式

### 1. 词表映射

$$ID = \text{Vocab}[Token]$$

$$\hat{t} = \text{IVocab}[ID]$$

其中 $\text{Vocab}$ 为Token到ID的映射字典，$\text{IVocab}$ 为反向映射。

### 2. BPE合并操作

$$P_{merge} = \frac{\text{count}(xy)}{\text{count}(x) \cdot \text{count}(y)}$$

合并得分是相邻Token对共现频率与独立频率之比。

### 3. Unigram语言模型似然

$$\mathcal{L} = \sum_{i} \log \left( \sum_{j} p(T_j | S_i) \cdot p(S_i) \right)$$

优化词表以最大化训练语料的似然。

## 详细说明

### 1. Token化流程详解

```
Step 1: 原始文本输入
         ↓
Step 2: 归一化（Unicode规范化、大小写转换）
         ↓
Step 3: 预分词（按空格、标点初步切分）
         ↓
Step 4: 子词分词（BPE/WordPiece/Unigram）
         ↓
Step 5: 添加特殊Token（[CLS], [SEP], [PAD]等）
         ↓
Step 6: ID映射（查表转整数序列）
```

### 2. 词表大小的影响

词表大小是Tokenization最核心的超参数之一，直接影响模型性能。

| 词表大小 | 特点 | 适用场景 |
|---------|------|---------|
| 小（<10K） | 序列短，粒度粗，OOV率高 | 词级模型 |
| 中等（10K-50K） | 平衡点 | 通用模型 |
| 大（50K-100K） | 序列长，子词丰富 | 大多数预训练模型 |
| 超大（>100K） | 字节级，序列很长 | 代码模型、多语言 |

**词表大小的权衡：**

- **词表太小**：序列短，但单个Token信息量低，OOV严重
- **词表太大**：序列长，Embedding矩阵大，训练困难

现代LLM通常采用中等偏大的词表（约32K-200K），如GPT-4 Turbo使用100K词表。

### 3. UNK问题（Unknown Token Problem）

UNK是Tokenization中最核心的问题之一。

**问题来源：**
- 测试集中出现训练集词表中不存在的词
- 词表无法覆盖所有语言现象（拼写错误、新词、网络用语）

**解决思路：**

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| 字符级分词 | 最极端的子词分词，完全避免OOV | 序列极长，丢失语义 |
| 子词分词 | 用子词组合表示未知词 | 最佳平衡 |
| Byte-level | UTF-8字节作为基础单元 | 序列长度增加，任意Unicode |
| 字符n-gram | 固定字符组合 | 效果一般 |

### 4. 子词分词的优势

**为什么子词分词是目前的主流？**

1. **OOV处理能力**：未知词可以分解为已知子词
   - 例如："unhappiness" → "un" + "happi" + "ness"

2. **合理的序列长度**：比字符级短，比词级灵活

3. **语言学意义**：子词往往是有意义的语素
   - "readable" → "read" + "able"

4. **跨语言泛化**：不同语言可以共享子词
   - 德语复合词：Frau(夫人) + Arzt(医生) → Frauenarzt(妇科医生)

5. **数字处理**：可以将数字分解为统一子词
   - "123" → "1" + "23" 或 "123" 作为整体

### 5. 编码流程（Encode）

```python
def encode(tokenizer, text, max_length=None, truncation=False, padding=False):
    # Step 1: 归一化
    text = normalize(text)  # NFKC规范化、小写化

    # Step 2: 预分词
    words = pre_tokenize(text)  # 空格+标点切分

    # Step 3: 子词分词
    tokens = []
    for word in words:
        subwords = tokenize_word(tokenizer, word)
        tokens.extend(subwords)

    # Step 4: 添加特殊Token
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    # Step 5: ID映射
    ids = tokenizer.convert_tokens_to_ids(tokens)

    # Step 6: 填充/截断
    if max_length:
        if len(ids) > max_length:
            if truncation:
                ids = ids[:max_length]
        if padding and len(ids) < max_length:
            ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))

    return ids
```

### 6. 解码流程（Decode）

```python
def decode(tokenizer, ids, skip_special_tokens=True):
    # Step 1: ID转Token
    tokens = tokenizer.convert_ids_to_tokens(ids)

    # Step 2: 移除特殊Token
    if skip_special_tokens:
        tokens = [t for t in tokens if not tokenizer.is_special_token(t)]

    # Step 3: 拼接子词
    text = ""
    for token in tokens:
        if token.startswith("##"):
            # 处理BPE风格的子词拼接
            text += token[2:]
        elif token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        else:
            text += token + " "

    # Step 4: 后处理（去除多余空格、还原空格等）
    text = post_process(text)

    return text.strip()
```

### 7. 常见Tokenization策略对比

| 策略 | 词表大小 | 序列长度 | OOV处理 | 计算效率 |
|------|---------|---------|--------|---------|
| 词级（Word-level） | ~50K | 短 | 差 | 高 |
| 字符级（Char-level） | ~256 | 很长 | 无OOV | 低 |
| BPE | 可控 | 中等 | 子词组合 | 高 |
| WordPiece | 可控 | 中等 | 子词组合 | 中 |
| Unigram | 可控 | 中等 | 子词组合 | 中 |
| Byte-level BPE | ~256 | 中等 | 无OOV | 高 |

### 8. 训练细节

**训练语料预处理：**
- 语料质量直接影响词表质量
- 需要足够的语料量（通常数十GB）
- 多语言场景需要平衡各语言比例

**词表大小确定：**
- 经验公式：$\text{vocab\_size} \approx 0.75 \times \sqrt{\text{corpus\_size}}$
- 或直接根据下游任务调整

**训练算法选择：**
- BPE：简单稳定，是大多数场景的首选
- Unigram + SentencePiece：支持删除子词，词表更优化
- WordPiece：Google系模型专用

## 代码示例

### Python - 完整Tokenization流程

```python
import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}

    def normalize(self, text):
        """Unicode NFKC规范化"""
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        return text

    def pre_tokenize(self, text):
        """空格和标点分词"""
        # 简单实现：按空格和常见标点切分
        text = re.sub(r'([.,!?;:"\'\(\)])', r' \1 ', text)
        words = text.split()
        return words

    def train(self, corpus):
        """BPE训练"""
        # 初始化词表为所有字符
        words = []
        vocab = Counter()

        for text in corpus:
            normalized = self.normalize(text)
            tokens = self.pre_tokenize(normalized)
            for word in tokens:
                word_tokens = list(word) + ['</w>']
                words.append(word_tokens)
                vocab.update(word_tokens)

        # 构建初始词表
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab['</w>'] = 256

        # BPE合并
        for i in range(self.vocab_size - 257):
            # 统计所有相邻对
            pairs = Counter()
            for word in words:
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    pairs[pair] += 1

            if not pairs:
                break

            # 找到最高频的对
            best_pair = pairs.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]

            # 合并
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words
            self.vocab[new_token] = len(self.vocab)

        # 构建反向词表
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """编码"""
        normalized = self.normalize(text)
        tokens = self.pre_tokenize(normalized)
        result = []

        for word in tokens:
            word_tokens = list(word) + ['</w>']
            # BPE分词
            i = 0
            while i < len(word_tokens):
                found = False
                for j in range(len(word_tokens), i, -1):
                    subword = ''.join(word_tokens[i:j])
                    if subword in self.vocab:
                        result.append(self.vocab[subword])
                        i = j
                        found = True
                        break
                if not found:
                    result.append(self.vocab.get(word_tokens[i], self.vocab['[UNK]']))
                    i += 1
        return result

    def decode(self, ids):
        """解码"""
        tokens = [self.reverse_vocab.get(id, '[UNK]') for id in ids]
        result = []
        for token in tokens:
            if token == '</w>':
                result.append(' ')
            elif token.startswith('##'):
                result.append(token[2:])
            else:
                result.append(token)
        return ''.join(result).strip()

# 使用示例
corpus = [
    "hello world",
    "natural language processing",
    "machine learning is amazing",
    "deep learning models are powerful"
]

tokenizer = SimpleTokenizer(vocab_size=500)
tokenizer.train(corpus)

text = "hello"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
print(f"原文: {text}")
print(f"IDs: {ids}")
print(f"解码: {decoded}")
```

### HuggingFace Transformers 完整使用

```python
from transformers import AutoTokenizer

# 加载多种分词器对比
tokenizers = {
    'bert-base-chinese': AutoTokenizer.from_pretrained("bert-base-chinese"),
    'gpt2': AutoTokenizer.from_pretrained("gpt2"),
    't5-base': AutoTokenizer.from_pretrained("t5-base"),
}

text = "深度学习是机器学习的子领域"

for name, tokenizer in tokenizers.items():
    print(f"\n=== {name} ===")
    encoded = tokenizer.encode(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(encoded[0])
    print(f"原文: {text}")
    print(f"Token数: {len(tokens)}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {encoded.tolist()}")
```

## 应用场景

| 场景 | 推荐词表大小 | 策略 | 特殊处理 |
|------|------------|------|---------|
| 中文NLP | 20K-50K | BERT分词/Chars | 数字/拼音处理 |
| 英文NLP | 30K-50K | BPE/WordPiece | 连字符/缩写 |
| 多语言 | 50K-250K | SentencePiece | 语种平衡 |
| 代码生成 | 50K-100K | Byte-level BPE | 特殊符号处理 |
| 语音识别 | 10K-50K | BPE | 音素+字符混合 |

## 相关概念

- **词表（Vocabulary）**：所有Token到ID的映射
- **OOV（Out-of-Vocabulary）**：未知词问题
- **子词（Subword）**：介于词和字符之间的单位
- **特殊Token（Special Tokens）**：[CLS], [SEP], [PAD], [UNK], [MASK]
- **覆盖率（Coverage）**：词表对语料的覆盖程度
- **分词率（Tokenization Rate）**：原词与Token数的比值

## 延伸阅读

- [Subword Regularization: Multiple Candidate Tokens](https://arxiv.org/abs/1804.10959) — 多次子词分词
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE在NMT中的应用
- [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — SentencePiece论文
- [Google's Tokenizer Comparison](https://github.com/google/sentencepiece/blob/master/python/tokenization_demo.ipynb) — 分词器对比实验
