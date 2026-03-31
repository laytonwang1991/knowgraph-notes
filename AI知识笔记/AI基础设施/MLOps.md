---
title: MLOps机器学习运维
alias: Machine Learning Operations
tags: [MLOps, 模型版本管理, 实验跟踪, 部署流程, CI/CD, MLflow]
category: AI基础设施
created: 2026-03-31
updated: 2026-03-31
author: Claude
description: MLOps是机器学习工程与运维的融合实践，涵盖模型版本管理、实验跟踪、特征工程、部署上线、监控告警的全生命周期，是实现ML系统可靠生产和持续迭代的关键方法论。
mastery: 0
rating: 0
related_concepts: [DevOps, CI/CD, MLflow, Kubeflow, Docker, Kubernetes, 模型注册, 实验跟踪]
difficulty: intermediate
read_time: 30
prerequisites: [机器学习基础, 软件工程, Docker基础, 云计算基础]
---

# MLOps机器学习运维

## 一句话定义

MLOps（Machine Learning Operations）是将 DevOps 原则应用于机器学习系统的工程实践，通过自动化流水线、版本化管理、持续监控等手段，实现 ML 模型从实验到生产的快速、可靠、可持续迭代交付。

## 核心概念

### ML 生命周期

ML 系统的生命周期包含六个核心阶段：

$$
\text{Data} \rightarrow \text{Features} \rightarrow \text{Model} \rightarrow \text{Evaluate} \rightarrow \text{Deploy} \rightarrow \text{Monitor} \rightarrow \text{Data (循环)}
$$

每个阶段的输出作为下一阶段的输入，形成持续迭代闭环。

### MLOps 成熟度模型（Google）

| 级别 | 特征 | 自动化程度 |
|------|------|------------|
| **Level 0** | 手工实验，脚本驱动，模型手动部署 | 无自动化 |
| **Level 1** | 自动训练流水线，模型版本管理，特征存储 | 训练自动化 |
| **Level 2** | 完整的 CI/CD，自动化测试，灰度发布 | 全流程自动化 |
| **Level 3** | A/B 测试，持续监控，漂移检测，自动回滚 | 生产级运维 |

## 详细说明

### 1. 实验跟踪

实验跟踪记录每次模型训练的输入、配置、指标和输出，解决"哪个参数组合效果最好"的问题。

核心记录内容：
- **超参数**：学习率、batch size、网络结构、正则化系数
- **数据集版本**：训练集/验证集指纹（SHA256 hash）
- **训练指标**：loss 曲线、评估指标（Accuracy、AUC 等）
- **输出产物**：模型权重、日志、配置、采样预测结果
- **环境信息**：Python 版本、依赖包版本、GPU 型号

```python
# MLflow Tracking 示例
import mlflow

mlflow.set_experiment("bert-sentiment-classification")

with mlflow.start_run(run_name="lr=1e-4_batch=32"):
    # 记录参数
    mlflow.log_params({
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 10,
        "model_name": "bert-base-chinese"
    })

    # 训练循环
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader)
        val_metrics = evaluate(model, val_loader)
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"]
        }, step=epoch)

    # 保存模型
    mlflow.pytorch.log_model(model, "model")
```

### 2. 模型版本管理

#### 2.1 模型注册表（Model Registry）

模型注册表集中管理模型版本的生命周期状态：

| 阶段 | 含义 | 操作 |
|------|------|------|
| **Staged** | 待评审 | 人工评审通过后进入下一阶段 |
| **Production** | 生产就绪 | 可部署到线上服务 |
| **Archived** | 已归档 | 历史版本，不再使用 |
| **Rejected** | 已拒绝 | 未通过评审的版本 |

#### 2.2 模型打包格式

- **pickle**：Python 对象序列化，存在安全风险，不推荐
- **ONNX**：跨框架模型格式，开放神经网络交换格式
- **TorchScript**：PyTorch 模型的序列化格式
- **TensorFlow SavedModel**：TensorFlow 官方模型格式
- **PMML/PFA**：传统预测模型标记语言

### 3. 特征工程与特征存储

#### 3.1 特征存储（Feature Store）

特征存储解决训练-推理特征一致性问题：

```
┌─────────────────────────────────────────────────────┐
│                  Feature Store                        │
├──────────────────────┬──────────────────────────────┤
│  Online Store (低延迟) │  Offline Store (高吞吐)     │
│  毫秒级特征查询        │  批量特征计算                │
│  用于实时推理          │  用于模型训练                │
└──────────────────────┴──────────────────────────────┘
```

核心功能：
- **特征注册表**：统一管理特征定义、血缘、口径
- **特征版本化**：特征工程变更可追溯
- **训练-推理一致性**：线上使用同一特征计算逻辑
- **特征回填**：新特征上线后可回填历史数据

#### 3.2 常用特征存储工具

| 工具 | 提供商 | 特点 |
|------|--------|------|
| **Feast** | Feasts.dev (Linux Foundation) | 开源，Kubernetes 原生 |
| **Tecton** | Tecton.ai | 企业级，实时特征支持 |
| **SageMaker Feature Store** | AWS | 云原生集成 |
| **Vertex AI Feature Store** | Google Cloud | GCP 集成 |
| **Databricks Feature Store** | Databricks | Delta Lake 集成 |

### 4. 模型部署与推理优化

#### 4.1 部署模式

| 部署方式 | 延迟 | 吞吐量 | 适用场景 |
|----------|------|--------|----------|
| **在线推理（Real-time）** | 毫秒级 | 低-中 | 用户请求级响应，搜索/推荐 |
| **批处理推理（Batch）** | 分钟-小时 | 极高 | 离线评分、报表生成 |
| **边缘推理（Edge）** | 毫秒级 | 低 | 移动端、IoT 设备 |
| **流式推理（Streaming）** | 毫秒-秒级 | 高 | 实时数据流处理 |

#### 4.2 推理优化技术

模型轻量化：
- **量化（Quantization）**：fp32 → fp16/int8/int4，精度损失通常 < 1%
- **剪枝（Pruning）**：移除不重要的权重/神经元
- **知识蒸馏（Distillation）**：大模型训练小模型（Teacher-Student）

推理引擎：
- **TensorRT**（NVIDIA）：INT8/FP16 优化，CUDA kernel fusion
- **ONNX Runtime**：跨平台推理，支持 CPU/GPU/NPU
- **TorchServe**：PyTorch 模型Serving
- **Triton Inference Server**：多模型推理，支持动态 batching

```python
# TorchServe 模型打包示例
# 1. 保存模型
torch.save(model.state_dict(), "model.pt")

# 2. 创建 handler
class MyHandler(BaseHandler):
    def preprocess(self, data):
        inputs = torch.tensor(data[0]["body"])
        return inputs

    def inference(self, inputs):
        return self.model(inputs)

    def postprocess(self, outputs):
        return outputs.tolist()

# 3. 启动服务
# torchserve --model-store /models --config-file config.properties
```

### 5. 持续训练（Continuous Training）

当生产数据分布发生变化时，自动触发模型重训练：

```
生产数据 → 数据漂移检测 → 触发训练流水线 → 自动评估 → 合格则部署
```

**数据漂移检测**：
- **分布检测**：使用 Kolmogorov-Smirnov 检验或 Population Stability Index (PSI)
  $$
  \text{PSI} = \sum_{i=1}^{n} (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)
  $$
  其中 $A_i$ 为实际分布，$E_i$ 为期望分布。PSI > 0.2 通常表示显著漂移。

- **性能监控**：监控实时预测准确率，突然下降通常表示数据异常或漂移

### 6. 监控与可观测性

ML 系统需要监控的特殊指标：

| 指标类型 | 具体内容 | 告警阈值 |
|----------|----------|----------|
| **模型性能** | 预测准确率、AUC、loss | 下降 > 5% |
| **数据质量** | 缺失率、异常值率、分布漂移 | PSI > 0.2 |
| **系统性能** | 延迟、吞吐量、GPU 利用率 | P99 > 100ms |
| **业务指标** | CTR、CVR、转化率 | 下降 > 10% |

**可观测性三支柱**：
- **指标（Metrics）**：Prometheus + Grafana
- **日志（Logs）**：结构化日志，ELK Stack
- **链路追踪（Traces）**：Jaeger/Zipkin，端到端请求追踪

## 应用场景

1. **推荐系统**：用户行为数据实时采集，小时级模型更新，AB 测试迭代
2. **金融风控**：欺诈检测模型日级别重训练，模型可解释性要求高
3. **广告投放**：CTR/CVR 模型小时级更新，支持竞价实时决策
4. **自然语言处理**：LLM 微调流水线，Prompt 版本管理，RLHF 实验跟踪
5. **计算机视觉**：质检模型边缘部署，实时推理延迟监控

## 相关概念

- **MLflow**：开源 ML 生命周期管理平台，提供 Tracking、Projects、Models、Registry 组件
- **Kubeflow**：Kubernetes 上的 ML 流水线框架，支撑大规模分布式训练
- **ML Pipeline**：由多个组件（数据处理 → 特征工程 → 训练 → 评估 → 部署）串联组成的自动化流程
- **A/B 测试**：对比两个模型版本在真实流量上的表现差异
- **Canary Release**：将新模型逐步放量至小比例流量，降低全量发布风险
- **Model Card**：模型说明书，记录模型的训练数据、评估结果、使用限制、伦理考量

## 延伸阅读

- [Google MLOps Guidelines](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines) — Google MLOps 成熟度模型
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html) — 官方文档
- [Kubeflow Documentation](https://www.kubeflow.org/docs/) — 流水线部署指南
- [Feast Feature Store](https:// feast.dev/) — 开源特征存储
- [Why MLOps Is Replacing DevOps](https://thenewstack.io/why-mlops-is-replacing-devops/) — MLOps vs DevOps 对比分析
