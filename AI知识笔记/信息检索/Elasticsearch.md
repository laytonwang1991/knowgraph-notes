---
title: Elasticsearch
alias: ES, Elastic
tags: [信息检索, 搜索引擎, 分布式系统, NoSQL, 倒排索引]
category: 信息检索
created: 2026-03-31
updated: 2026-03-31
author: AI知识库
description: Elasticsearch是一个基于Lucene构建的分布式、RESTful风格的搜索和分析引擎，支持实时全文搜索、结构化搜索、分析等功能，广泛应用于日志分析、全文检索、业务搜索等场景。
mastery: 0.65
rating: 9
related_concepts: [倒排索引, Lucene, 分片机制, 分布式一致性, RESTful API, 聚合分析]
difficulty: 4
read_time: 30
prerequisites: [HTTP协议基础, RESTful API概念, 分布式系统基础]
---

# Elasticsearch

## 一句话定义

Elasticsearch是一个基于Lucene的分布式、RESTful搜索和分析引擎，通过倒排索引实现高效的全文搜索，并提供强大的聚合分析能力，适用于大规模数据的实时搜索和数据分析场景。

## 核心概念

### 文档与索引

| 概念 | 说明 |
|------|------|
| Document（文档） | Elasticsearch中的基本数据单位，JSON格式 |
| Index（索引） | 文档的逻辑集合，类似数据库 |
| Mapping（映射） | 文档的 schema 定义，控制字段类型和索引方式 |
| Shard（分片） | 索引的水平分片单元 |

### 倒排索引结构

```
正排索引: 文档ID -> 文档内容
  Doc1 -> "Elasticsearch 分布式 搜索"
  Doc2 -> "分布式 系统 高可用"

倒排索引: 词项 -> 包含该词项的文档列表
  "Elasticsearch" -> [Doc1]
  "分布式" -> [Doc1, Doc2]
  "搜索" -> [Doc1]
  "系统" -> [Doc2]
  "高可用" -> [Doc2]
```

倒排索引是Elasticsearch实现毫秒级全文搜索的核心数据结构。

## 核心公式

### TF-IDF权重计算

Elasticsearch默认使用TF-IDF作为文档 relevance scoring 的基础：

$$score(q, d) = \sum_{t \in q} tf(t, d) \cdot idf(t)^2 \cdot fieldNorm(d) \cdot \text{boost}$$

其中：
- $tf(t, d)$：词项 $t$ 在文档 $d$ 中的词频
- $idf(t) = 1 + \log\frac{N+1}{df(t)+1}$：逆文档频率
- $fieldNorm(d)$：字段长度归一化因子
- $\text{boost}$：人工提升权重

### BM25（Elasticsearch 8.x默认）

$$score = \sum_{i} \log\left(1 + \frac{N - n_i + 0.5}{n_i + 0.5}\right) \cdot \frac{tf_i \cdot (k_1 + 1)}{tf_i + k_1 \cdot (1 - b + b \cdot \frac{L}{L_{avg}})}$$

其中 $k_1=1.2$，$b=0.75$ 为默认参数。

## 详细说明

### 1. 分片机制

Elasticsearch通过分片实现水平扩展：

- **主分片（Primary Shard）**：索引的物理存储单元，每个分片是一个独立的Lucene索引
- **副本分片（Replica Shard）**：主分片的副本，提供高可用和读扩展
- **分片策略**：
  - 索引创建时指定主分片数量（不可更改）
  - 副本分片数量可动态调整
- **分片分配**：通过Cluster-level的shard allocation机制实现

### 2. 数据写入流程

```
客户端 -> Coordinating Node -> Primary Shard -> Replica Shard
                              (同步复制)
```

- 写入请求首先到达任意一个节点（Coordinating Node）
- 路由到对应的主分片执行写入
- 主分片写入成功后，并行复制到副本分片
- 半数以上副本确认后，返回客户端成功

### 3. 查询流程

Elasticsearch的查询分为两个阶段：

**第一阶段：Query Phase**
- Coordinating Node广播查询到所有相关分片
- 各分片独立执行查询，返回Top-K文档ID和score
- Coordinating Node合并结果，获取全局Top-K文档ID

**第二阶段：Fetch Phase**
- Coordinating Node向包含这些文档ID的分片发起fetch请求
- 获取完整的文档内容
- 返回客户端

### 4. Query DSL

Elasticsearch提供强大的查询DSL：

```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "Elasticsearch 教程" } }
      ],
      "filter": [
        { "range": { "publish_date": { "gte": "2024-01-01" } } },
        { "term": { "status": "published" } }
      ],
      "should": [
        { "match": { "content": "分布式" } }
      ],
      "minimum_should_match": 1
    }
  },
  "aggs": {
    "category_stats": {
      "terms": { "field": "category.keyword" },
      "aggs": {
        "avg_rating": { "avg": { "field": "rating" } }
      }
    }
  },
  "highlight": {
    "fields": {
      "content": { "pre_tags": ["<em>"], "post_tags": ["</em>"] }
    }
  }
}
```

### 5. 聚合分析

Elasticsearch的聚合分为三类：

- **Metric聚合**：计算单一指标（avg, sum, max, min, stats, cardinality）
- **Bucket聚合**：按条件分组（terms, range, date_histogram, filters）
- **Pipeline聚合**：对聚合结果再聚合（avg_bucket, cumulative_sum, moving_avg）

## 代码示例

### Python客户端使用

```python
from elasticsearch import Elasticsearch

# 连接集群
es = Elasticsearch(
    ["http://localhost:9200"],
    basic_auth=("elastic", "password"),
    verify_certs=True
)

# 创建索引
es.indices.create(
    index="blog_posts",
    body={
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "ik": {
                        "type": "custom",
                        "tokenizer": "ik_max_word",
                        "filter": ["synonym", "stop"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "title": { "type": "text", "analyzer": "ik_max_word" },
                "content": { "type": "text", "analyzer": "ik_max_word" },
                "tags": { "type": "keyword" },
                "publish_date": { "type": "date" },
                "views": { "type": "long" }
            }
        }
    }
)

# 索引文档
es.index(
    index="blog_posts",
    id="1",
    body={
        "title": "Elasticsearch入门指南",
        "content": "本文介绍Elasticsearch的基本概念和使用方法...",
        "tags": ["Elasticsearch", "搜索", "教程"],
        "publish_date": "2024-03-01",
        "views": 1520
    }
)

# 全文搜索
result = es.search(
    index="blog_posts",
    body={
        "query": {
            "multi_match": {
                "query": "Elasticsearch 搜索",
                "fields": ["title^2", "content"]
            }
        },
        "aggs": {
            "popular_tags": {
                "terms": { "field": "tags.keyword", "size": 10 }
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "content": { "fragment_size": 150 }
            }
        }
    }
)

# 输出结果
for hit in result["hits"]["hits"]:
    print(f"文档ID: {hit['_id']}, 得分: {hit['_score']}")
    print(f"标题: {hit['_source']['title']}")
    print(f"高亮: {hit.get('highlight', {})}")
```

### 地理位置查询

```python
# 搜索附近地点
result = es.search(
    index="pois",
    body={
        "query": {
            "geo_distance": {
                "distance": "5km",
                "location": {
                    "lat": 39.908,
                    "lon": 116.397
                }
            }
        },
        "sort": [
            {
                "_geo_distance": {
                    "location": {
                        "lat": 39.908,
                        "lon": 116.397
                    },
                    "order": "asc",
                    "unit": "km"
                }
            }
        ]
    }
)
```

## 应用场景

| 场景 | 典型用例 |
|------|----------|
| 全文搜索 | 电商商品搜索、企业文档搜索 |
| 日志分析 | ELK Stack（Elasticsearch + Logstash + Kibana） |
| 应用性能监控 | 业务指标监控、告警 |
| 安全分析 | SIEM、威胁检测 |
| 地理位置搜索 | 附近地点查找、物流追踪 |
| 推荐系统 | 物品召回、特征存储 |

## 相关概念

- **Lucene**：Apache顶级项目，Elasticsearch的核心搜索引擎库
- **倒排索引**：Elasticsearch实现快速全文检索的核心数据结构
- **分片与副本**：实现数据水平扩展和高可用的机制
- **ELK Stack**：Elasticsearch、Logstash、Kibana组合的日志分析套件
- **向量搜索**：Elasticsearch 8.x支持的dense_vector类型，用于语义搜索

## 延伸阅读

1. Elastic官方文档: https://www.elastic.co/guide/index.html
2. 《Elasticsearch实战》- Radu Gheorghe, Matthew Lee Hinman, Roy Russo
3. 《Elasticsearch: The Definitive Guide》- Clinton Gormley, Zachary Tong
4. 向量搜索实战: https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html
