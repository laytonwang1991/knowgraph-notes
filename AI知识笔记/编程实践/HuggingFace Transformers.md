---
title: HuggingFace Transformers
alias: HuggingFace Transformers
tags:
  - HuggingFace
  - Transformers
  - 预训练模型
  - NLP
  - 大语言模型
category: 编程实践
created: 2026-03-31
updated: 2026-03-31
author: AI Practice Team
description: 使用HuggingFace Transformers库进行模型推理、微调和部署的实践指南，涵盖Pipeline、模型微调、Tokenizer使用、AutoClass等核心功能。
mastery: 80
rating: 9
related_concepts:
  - Transformer架构
  - 预训练与微调
  - BERT
  - GPT
  - Tokenizer
  - 注意力机制
difficulty: 中级
read_time: 50分钟
prerequisites:
  - Python基础
  - 深度学习基础
  - NLP基础概念
  - PyTorch或TensorFlow基础
---

# HuggingFace Transformers

## 一句话定义

HuggingFace Transformers是一个开源的自然语言处理库，提供了数千预训练模型（如BERT、GPT、T5等）的统一API，使得研究者和开发者能够轻松使用、微调和部署Transformer架构的模型。

## 详细说明

### 1. Pipeline（管道）

Pipeline是HuggingFace最易用的接口，将预处理、模型推理、后处理封装成简单易用的函数调用：

```python
from transformers import pipeline

# 情感分析（开箱即用）
classifier = pipeline('sentiment-analysis')
result = classifier("I love using HuggingFace Transformers, it's amazing!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 命名实体识别
ner = pipeline('ner', aggregation_strategy='simple')
text = "HuggingFace is a company based in New York."
entities = ner(text)
# [{'entity_group': 'ORG', 'word': 'HuggingFace', 'score': 0.99, ...},
#  {'entity_group': 'LOC', 'word': 'New York', 'score': 0.99, ...}]

# 问答系统
qa = pipeline('question-answering')
context = "HuggingFace provides state-of-the-art NLP models."
question = "What does HuggingFace provide?"
answer = qa(question=question, context=context)
# {'answer': 'state-of-the-art NLP models', 'score': 0.99, ...}

# 文本生成
generator = pipeline('text-generation', model='gpt2')
prompt = "In the future, artificial intelligence will"
generated = generator(prompt, max_length=50, num_return_sequences=2)
# [{'generated_text': 'In the future, artificial intelligence will...' }, ...]

# 机器翻译
translator = pipeline('translation_en_to_fr')
french_text = translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]

# 文本摘要
summarizer = pipeline('summarization')
article = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast
to the natural intelligence displayed by humans and animals. Leading AI textbooks
define the field as the study of "intelligent agents": any device that perceives
its environment and takes actions that maximize its chance of success.
"""
summary = summarizer(article, max_length=50, min_length=20)
# [{'summary_text': 'Artificial intelligence is intelligence demonstrated by machines...'}]

# 使用本地模型
local_classifier = pipeline('sentiment-analysis', model='./my_local_model')
```

### 2. 模型微调 (Fine-tuning)

微调是将预训练模型适应特定任务的关健技术：

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np

# 加载数据集
raw_datasets = load_dataset("glue", "sst2")
print(raw_datasets)
# DatasetDict({
#     train: Dataset({ features: ['sentence', 'label', 'idx'], num_rows: 67349 })
#     validation: Dataset({ features: ['sentence', 'label', 'idx'], num_rows: 872 })
#     test: Dataset({ features: ['sentence', 'label', 'idx'], num_rows: 1821 })
# })

# 加载预训练模型和tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard"
)

# 自定义评估指标
from datasets import load_metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 评估模型
results = trainer.evaluate()
print(f"Evaluation results: {results}")

# 保存微调后的模型
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
```

### 3. Tokenizer使用

Tokenizer是将文本转换为模型可处理格式的核心组件：

```python
from transformers import AutoTokenizer, BertTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 基本编码
text = "Hello, how are you today?"
encoded = tokenizer(text)
print(encoded)
# {'input_ids': [101, 7592, 1010, 2129, 2024, 2021, 102], ...}

# 解码
decoded = tokenizer.decode(encoded['input_ids'])
print(decoded)
# [CLS] hello, how are you today? [SEP]

# 批量编码
sentences = ["I love AI", "Transformers are powerful"]
encoded_batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_batch)
# {'input_ids': tensor([[...]]), 'token_type_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}

# 特殊token
print(f"UNK token: {tokenizer.unk_token}, id: {tokenizer.unk_token_id}")
print(f"PAD token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")
print(f"CLS token: {tokenizer.cls_token}, id: {tokenizer.cls_token_id}")
print(f"SEP token: {tokenizer.sep_token}, id: {tokenizer.sep_token_id}")

# 获取词表
vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")

# 添加自定义token
num_added_tokens = tokenizer.add_tokens(['<AI>', '<HUMAN>'])
print(f"Added {num_added_tokens} tokens")

# 调整模型embedding大小以适应新token
model.resize_token_embeddings(len(tokenizer))

# 处理不同长度的句子
long_text = "This is a very long text " * 100
# 截断到最大长度
encoded = tokenizer(
    long_text,
    max_length=512,
    truncation=True,
    return_tensors="pt"
)
print(f"Truncated length: {encoded['input_ids'].shape[1]}")

# Fast tokenizer vs Slow tokenizer
from transformers import BertTokenizerFast, BertTokenizer

slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
fast_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Fast tokenizer支持offset mapping，用于词级别任务
text = "I love AI"
fast_result = fast_tokenizer(text, return_offsets_mapping=True)
print(f"Offset mapping: {fast_result['offset_mapping']}")
# [(0, 1), (2, 6), (7, 9), (10, 12)]
```

### 4. AutoClass

AutoClass是HuggingFace的自动模型加载机制，根据预训练模型名称自动推断模型类型：

```python
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoTokenizer
)

# 自动加载模型配置
config = AutoConfig.from_pretrained("bert-base-uncased")
print(f"Model type: {config.model_type}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num attention heads: {config.num_attention_heads}")

# 加载主干网络（无任务头）
model = AutoModel.from_pretrained("bert-base-uncased")
print(f"Model type: {type(model)}")
# <class 'transformers.models.bert.modeling_bert.BertModel'>

# 加载序列分类模型
clf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
print(f"Classifier type: {type(clf_model)}")
# <class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>

# 加载问答模型
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
print(f"QA model type: {type(qa_model)}")

# 加载因果语言模型（GPT类）
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
print(f"GPT model type: {type(gpt_model)}")

# 加载掩码语言模型（BERT类）
mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
print(f"MLM model type: {type(mlm_model)}")

# 加载序列到序列模型（T5、BART类）
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
print(f"T5 model type: {type(t5_model)}")

# 批量加载多个模型
models_to_load = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased"
]

for model_name in models_to_load:
    model = AutoModel.from_pretrained(model_name)
    print(f"{model_name}: {type(model).__name__}, {model.config.hidden_size} hidden units")

# 自定义模型配置
custom_config = AutoConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
custom_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=custom_config
)

# 从本地目录加载
local_model = AutoModel.from_pretrained("./my_local_model")
local_tokenizer = AutoTokenizer.from_pretrained("./my_local_tokenizer")

# 延迟加载（节省内存）
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True,
    output_attentions=True
)
```

## 代码示例

### 完整的情感分析微调流程

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np

# 1. 准备数据
dataset = load_dataset("yelp_polarity", split={"train": "train[:5000]", "test": "test[:1000]"})
print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

# 2. 加载tokenizer和模型
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3. 预处理数据
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. 训练配置
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    seed=42
)

# 5. 评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

# 6. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# 7. 训练
trainer.train()

# 8. 预测
sample_text = "This restaurant has amazing food!"
inputs = tokenizer(sample_text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"
print(f"Sentiment: {label}")
```

## 应用场景

1. **文本分类**：情感分析、垃圾邮件检测、主题分类、意图识别
2. **命名实体识别**：信息抽取、关系抽取、知识图谱构建
3. **问答系统**：开放域问答、阅读理解、客服机器人
4. **文本生成**：故事创作、代码生成、文章摘要
5. **机器翻译**：多语言翻译、术语一致性处理
6. **对话系统**：聊天机器人、语音助手、多轮对话
7. **文本嵌入**：语义搜索、相似度计算、推荐系统
8. **多模态任务**：图像描述、视觉问答、跨模态检索

## 相关概念

| 概念 | 说明 |
|------|------|
| Pre-training | 在大规模无标注数据上学习通用语言表示 |
| Fine-tuning | 在特定任务数据上微调预训练模型 |
| Transfer Learning | 将一个领域学习的知识迁移到另一个领域 |
| Pipeline | 端到端的推理封装 |
| AutoClass | 根据模型名称自动推断并加载对应模型类 |
| Tokenizer | 文本到ID的转换，支持各种编码方式 |
| Dataset | HuggingFace的数据集抽象，支持大规模数据处理 |
| Trainer | 封装好的训练循环，支持分布式和混合精度训练 |

## 延伸阅读

1. [HuggingFace官方文档](https://huggingface.co/docs/transformers/) - 完整的Transformers库文档
2. [HuggingFace Hub](https://huggingface.co/models) - 模型库，包含数万预训练模型
3. [datasets库文档](https://huggingface.co/docs/datasets/) - 高效数据加载和处理
4. [tokenizers库文档](https://huggingface.co/docs/tokenizers/) - 高速tokenizer实现
5. [PEFT库](https://huggingface.co/docs/peft/) - 参数高效微调（LoRA、Adapter等）
6. [trl库](https://github.com/huggingface/trl) - 强化学习训练（PPO等）
7. [Transformer论文](https://arxiv.org/abs/1706.03762) - 原始Transformer架构论文
