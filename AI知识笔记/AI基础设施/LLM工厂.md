---
title: LLM工厂
alias: LLM Factory
tags:
  - LLM部署
  - 模型托管
  - 微调服务
  - AB测试
  - 负载均衡
category: AI基础设施
created: 2026-03-31
updated: 2026-03-31
author: AI基础设施团队
description: 企业级LLM部署完整解决方案，涵盖模型托管、微调服务、AB测试和负载均衡等核心能力。
mastery: 8
rating: 9
related_concepts:
  - 推理服务架构
  - 模型推理优化
  - 分布式训练
  - MLOps
difficulty: 高
read_time: 30分钟
prerequisites:
  - 深度学习部署基础
  - Kubernetes了解
  - 微服务架构
---

# LLM工厂

## 一句话定义

LLM工厂是企业级AI基础设施的核心平台，提供模型托管、微调服务、流量调度和效果对比的一站式解决方案，实现LLM的高效部署与持续优化。

## 详细说明

### 1. 模型托管（Model Hosting）

模型托管是LLM工厂的基础能力，提供可扩展的模型部署和推理服务。

#### 托管架构

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Gateway                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Auth/N Auth│ │ Rate Limit │ │ Load Balance│ │  Router   │ │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Model Pool A │   │  Model Pool B │   │  Model Pool C │
│  Llama-2-70B  │   │   GPT-4       │   │  Claude-2     │
│  vLLM Engine  │   │  TensorRT-LLM │   │  SageMaker    │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ GPU Clus│         │ GPU Clus│         │  Cloud  │
   │ (On-prem)│         │ (On-prem)│         │ (AWS)   │
   └─────────┘         └─────────┘         └─────────┘
```

#### 部署模式

| 模式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| On-premise | 数据敏感、金融 | 数据安全、完全控制 | 运维成本高 |
| Cloud Native | 通用场景 | 弹性伸缩、成本优化 | 数据传输延迟 |
| Hybrid | 混合需求 | 灵活配置 | 架构复杂 |

#### 代码示例：模型注册

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ModelStatus(Enum):
    ACTIVE = "active"
    UPDATING = "updating"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"

@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    model_name: str
    model_type: str  # text, code, embedding, etc.
    version: str
    engine: str  # vllm, tensorrt, transformers
    max_context_length: int
    supported_features: List[str]
    hardware_requirements: Dict[str, int]
    replicas: int
    autoscaling_config: Dict

class ModelRegistry:
    """模型注册中心"""
    def __init__(self, db_connection):
        self.db = db_connection
        self._cache = {}

    def register_model(self, config: ModelConfig) -> str:
        """注册新模型"""
        query = """
        INSERT INTO models (
            model_id, model_name, model_type, version,
            engine, max_context_length, supported_features,
            hardware_requirements, replicas, autoscaling_config,
            status, created_at
        ) VALUES (
            %(model_id)s, %(model_name)s, %(model_type)s, %(version)s,
            %(engine)s, %(max_context_length)s, %(supported_features)s,
            %(hardware_requirements)s, %(replicas)s, %(autoscaling_config)s,
            'active', NOW()
        )
        ON CONFLICT (model_id) DO UPDATE SET
            version = EXCLUDED.version,
            replicas = EXCLUDED.replicas,
            updated_at = NOW()
        """
        self.db.execute(query, asdict(config))
        self._cache[config.model_id] = config
        return config.model_id

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        if model_id in self._cache:
            return self._cache[model_id]

        query = "SELECT * FROM models WHERE model_id = %s"
        row = self.db.fetchone(query, (model_id,))

        if row:
            config = ModelConfig(**dict(row))
            self._cache[model_id] = config
            return config

        return None

    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        model_type: Optional[str] = None
    ) -> List[ModelConfig]:
        """列出模型"""
        conditions = []
        params = []

        if status:
            conditions.append("status = %s")
            params.append(status.value)

        if model_type:
            conditions.append("model_type = %s")
            params.append(model_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"SELECT * FROM models WHERE {where_clause} ORDER BY created_at DESC"
        rows = self.db.fetchall(query, params)

        return [ModelConfig(**dict(row)) for row in rows]

    def update_replicas(self, model_id: str, replicas: int):
        """更新副本数"""
        query = """
        UPDATE models SET replicas = %s, updated_at = NOW()
        WHERE model_id = %s
        """
        self.db.execute(query, (replicas, model_id))
        if model_id in self._cache:
            self._cache[model_id].replicas = replicas
```

### 2. 微调服务（Fine-tuning Service）

微调服务提供从数据准备到模型训练再到部署的全流程能力。

#### 微调流水线

```
数据上传 → 数据清洗 → 模板格式化 → 训练配置 → 微调训练
                                                      │
                                                      ▼
部署上线 ← 验证测试 ← 模型导出 ← 中间检查点 ← 训练监控
```

#### 支持的微调方法

| 方法 | 说明 | 适用场景 | 资源需求 |
|------|------|----------|----------|
| LoRA | 低秩适配器 | 快速微调 | 单卡可跑 |
| QLoRA | 量化+LoRA | 资源受限 | 消费级GPU |
| Full FT | 全参数微调 | 最大化效果 | 多卡A100 |
| RLHF | 人类反馈强化 | 对齐优化 | 超大规模 |

#### 代码示例：微调任务管理

```python
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

class FineTuneStatus(Enum):
    PENDING = "pending"
    DATA_PROCESSING = "data_processing"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"

class FineTuneMethod(Enum):
    LORA = "lora"
    QLORA = "qlora"
    FULL_FT = "full_ft"
    RLHF = "rlhf"

@dataclass
class FineTuneConfig:
    """微调配置"""
    base_model: str
    method: FineTuneMethod
    training_data_path: str
    validation_data_path: Optional[str]

    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 8
    epochs: int = 3
    max_seq_length: int = 2048
    warmup_steps: int = 100

    # LoRA参数
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # 资源
    gpu_count: int = 1
    gpu_type: str = "A100"

class FineTuneJob:
    """微调任务"""
    def __init__(self, job_id: str, config: FineTuneConfig):
        self.job_id = job_id
        self.config = config
        self.status = FineTuneStatus.PENDING
        self.checkpoints = []
        self.metrics = {}
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "config": asdict(self.config),
            "status": self.status.value,
            "checkpoints": self.checkpoints,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

class FineTuneService:
    """微调服务"""
    def __init__(self, storage, compute_backend):
        self.storage = storage
        self.compute = compute_backend
        self.jobs: Dict[str, FineTuneJob] = {}

    def create_job(self, config: FineTuneConfig) -> FineTuneJob:
        """创建微调任务"""
        job_id = generate_job_id(config.base_model)

        # 验证数据
        self._validate_data(config)

        # 创建任务
        job = FineTuneJob(job_id, config)
        self.jobs[job_id] = job

        # 异步启动
        self._start_job_async(job)

        return job

    def _validate_data(self, config: FineTuneConfig):
        """验证训练数据"""
        train_data = self.storage.read(config.training_data_path)

        required_fields = ["instruction", "input", "output"]
        for item in train_data[:10]:  # 抽样检查
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing field: {field}")

        # 检查数据量
        if len(train_data) < 100:
            raise ValueError(f"Insufficient training data: {len(train_data)}")

    async def _start_job_async(self, job: FineTuneJob):
        """异步启动微调任务"""
        job.status = FineTuneStatus.DATA_PROCESSING

        # 数据预处理
        processed_data = await self._process_data(job.config)
        job.status = FineTuneStatus.TRAINING

        # 启动训练
        train_config = self._build_train_config(job.config)
        checkpoint_dir = await self.compute.start_training(train_config)

        job.checkpoints.append(checkpoint_dir)

        # 监控训练
        while job.status == FineTuneStatus.TRAINING:
            metrics = await self.compute.get_metrics(job.job_id)
            job.metrics.update(metrics)

            if metrics.get("epoch") >= job.config.epochs:
                job.status = FineTuneStatus.VALIDATING
                break

            await asyncio.sleep(30)

        # 验证
        await self._validate_model(job)
        job.status = FineTuneStatus.COMPLETED
        job.completed_at = datetime.now()

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """获取任务状态"""
        if job_id in self.jobs:
            return self.jobs[job_id].to_dict()
        return None
```

### 3. AB测试（AB Testing）

AB测试能力让团队能够科学对比不同模型版本的实际效果。

#### 测试框架

```python
from dataclasses import dataclass
from typing import Dict, List, Callable
import hashlib
import random

@dataclass
class Variant:
    """测试变体"""
    variant_id: str
    model_id: str
    prompt_version: str
    weight: float  # 流量权重 (0-1)

class ABTest:
    """AB测试配置"""
    def __init__(
        self,
        test_id: str,
        name: str,
        variants: List[Variant],
        metrics: List[str],
        min_sample_size: int = 1000
    ):
        self.test_id = test_id
        self.name = name
        self.variants = {v.variant_id: v for v in variants}
        self.metrics = metrics
        self.min_sample_size = min_sample_size

        # 验证权重
        total_weight = sum(v.weight for v in variants)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def assign_variant(self, user_id: str) -> str:
        """根据用户ID分配变体"""
        # 确定性分配，同一用户始终分到同一变体
        hash_val = int(hashlib.md5(f"{self.test_id}:{user_id}".encode()).hexdigest(), 16)
        normalized = (hash_val % 10000) / 10000.0

        cumulative = 0.0
        for variant_id, variant in self.variants.items():
            cumulative += variant.weight
            if normalized < cumulative:
                return variant_id

        return list(self.variants.keys())[-1]  # 保底

class ABTestManager:
    """AB测试管理器"""
    def __init__(self, db, metrics_collector):
        self.db = db
        self.metrics = metrics_collector
        self.active_tests: Dict[str, ABTest] = {}

    def create_test(self, config: Dict) -> ABTest:
        """创建新测试"""
        variants = [
            Variant(
                variant_id=v["variant_id"],
                model_id=v["model_id"],
                prompt_version=v.get("prompt_version", "v1.0"),
                weight=v["weight"]
            )
            for v in config["variants"]
        ]

        test = ABTest(
            test_id=config["test_id"],
            name=config["name"],
            variants=variants,
            metrics=config["metrics"]
        )

        self.active_tests[test.test_id] = test
        self.db.save_test(config)

        return test

    def record_assignment(self, user_id: str, test_id: str, variant_id: str):
        """记录分配"""
        self.db.insert("ab_assignments", {
            "user_id": user_id,
            "test_id": test_id,
            "variant_id": variant_id,
            "assigned_at": datetime.now()
        })

    def record_outcome(
        self,
        user_id: str,
        test_id: str,
        variant_id: str,
        metric_name: str,
        metric_value: float
    ):
        """记录结果"""
        self.metrics.record(
            test_id=test_id,
            variant_id=variant_id,
            metric_name=metric_name,
            value=metric_value
        )

    def analyze_test(self, test_id: str) -> Dict:
        """分析测试结果"""
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")

        results = {}

        for variant_id in test.variants:
            variant_metrics = self.metrics.get_variant_metrics(
                test_id, variant_id
            )

            results[variant_id] = {
                "sample_size": variant_metrics["count"],
                "mean": variant_metrics["mean"],
                "std": variant_metrics["std"],
                "confidence_interval": variant_metrics["ci_95"],
            }

        # 统计显著性检验
        results["statistical_significance"] = self._compute_significance(
            results, test.min_sample_size
        )

        return results

    def _compute_significance(self, results: Dict, min_sample: int) -> Dict:
        """计算统计显著性"""
        variant_ids = list(results.keys())
        v1, v2 = variant_ids[0], variant_ids[1]

        n1, n2 = results[v1]["sample_size"], results[v2]["sample_size"]
        mean1, mean2 = results[v1]["mean"], results[v2]["mean"]
        std1, std2 = results[v1]["std"], results[v2]["std"]

        # Welch's t-test
        se = ((std1**2/n1) + (std2**2/n2)) ** 0.5
        t_stat = (mean1 - mean2) / se
        p_value = self._t_test_pvalue(t_stat, n1 + n2 - 2)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "winner": v1 if mean1 > mean2 else v2
        }
```

### 4. 负载均衡（Load Balancing）

负载均衡根据实时负载和质量指标智能路由请求。

#### 路由策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| Round Robin | 轮询分发 | 均匀负载 |
| Least Load | 最少负载优先 | 异构环境 |
| Latency Based | 延迟最低优先 | 追求响应速度 |
| Quality Based | 质量最高优先 | 追求输出质量 |
| Cost Aware | 成本最优 | 成本敏感 |

#### 代码示例：智能路由

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

@dataclass
class ModelEndpoint:
    """模型端点"""
    endpoint_id: str
    model_id: str
    url: str
    current_load: int  # 当前处理中的请求数
    max_capacity: int
    avg_latency_ms: float
    error_rate: float
    cost_per_1k_tokens: float
    health_score: float  # 0-1的综合健康分

    def availability(self) -> float:
        """可用性评分"""
        return 1 - (self.current_load / self.max_capacity)

class LoadBalancer:
    """负载均衡器"""
    def __init__(self, endpoints: List[ModelEndpoint]):
        self.endpoints = {e.endpoint_id: e for e in endpoints}
        self.strategy = "adaptive"

    def select_endpoint(
        self,
        model_id: Optional[str] = None,
        strategy: str = "adaptive",
        priority: str = "latency"  # latency, quality, cost
    ) -> ModelEndpoint:
        """选择最佳端点"""
        candidates = [
            e for e in self.endpoints.values()
            if e.health_score > 0.5 and e.current_load < e.max_capacity
        ]

        if model_id:
            candidates = [e for e in candidates if e.model_id == model_id]

        if not candidates:
            raise NoAvailableEndpointError()

        if strategy == "round_robin":
            return self._round_robin(candidates)
        elif strategy == "least_load":
            return self._least_load(candidates)
        elif strategy == "adaptive":
            return self._adaptive_routing(candidates, priority)

        return candidates[0]

    def _adaptive_routing(
        self,
        candidates: List[ModelEndpoint],
        priority: str
    ) -> ModelEndpoint:
        """自适应路由"""
        scores = []

        for endpoint in candidates:
            # 多维度评分
            latency_score = 1 / (1 + endpoint.avg_latency_ms / 1000)
            quality_score = endpoint.health_score
            cost_score = 1 / (1 + endpoint.cost_per_1k_tokens)
            availability_score = endpoint.availability()

            if priority == "latency":
                weights = [0.5, 0.1, 0.1, 0.3]
            elif priority == "quality":
                weights = [0.1, 0.5, 0.2, 0.2]
            elif priority == "cost":
                weights = [0.2, 0.2, 0.4, 0.2]
            else:
                weights = [0.25, 0.25, 0.25, 0.25]

            total_score = (
                latency_score * weights[0] +
                quality_score * weights[1] +
                cost_score * weights[2] +
                availability_score * weights[3]
            )

            scores.append((total_score, endpoint))

        scores.sort(reverse=True)
        return scores[0][1]

    async def update_endpoint_metrics(
        self,
        endpoint_id: str,
        latency_ms: float,
        success: bool
    ):
        """更新端点指标"""
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint:
            return

        # 指数移动平均更新延迟
        alpha = 0.3
        endpoint.avg_latency_ms = (
            alpha * latency_ms +
            (1 - alpha) * endpoint.avg_latency_ms
        )

        # 更新错误率
        endpoint.error_rate = (
            0.1 * (0 if success else 1) +
            0.9 * endpoint.error_rate
        )

        # 更新健康分
        endpoint.health_score = self._compute_health_score(endpoint)

    def _compute_health_score(self, endpoint: ModelEndpoint) -> float:
        """计算综合健康分"""
        latency_score = max(0, 1 - endpoint.avg_latency_ms / 5000)
        error_score = 1 - endpoint.error_rate
        availability_score = endpoint.availability()

        return (
            latency_score * 0.3 +
            error_score * 0.4 +
            availability_score * 0.3
        )

class RequestRouter:
    """请求路由器"""
    def __init__(
        self,
        load_balancer: LoadBalancer,
        circuit_breaker
    ):
        self.lb = load_balancer
        self.cb = circuit_breaker

    async def route(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        priority: str = "latency"
    ) -> Dict:
        """路由请求"""
        endpoint = None

        for attempt in range(3):
            try:
                endpoint = self.lb.select_endpoint(model_id, priority=priority)

                # 检查断路器
                if self.cb.is_open(endpoint.endpoint_id):
                    continue

                # 发送请求
                response = await self._send_request(endpoint, prompt)

                # 记录成功
                await self.lb.update_endpoint_metrics(
                    endpoint.endpoint_id,
                    response.latency_ms,
                    success=True
                )

                return {
                    "response": response,
                    "endpoint": endpoint.endpoint_id,
                    "model": endpoint.model_id
                }

            except Exception as e:
                if endpoint:
                    await self.lb.update_endpoint_metrics(
                        endpoint.endpoint_id,
                        0,
                        success=False
                    )
                    self.cb.record_failure(endpoint.endpoint_id)

                if attempt == 2:
                    raise RoutingError(f"All endpoints failed: {e}")

        raise RoutingError("No available endpoints")
```

## 应用场景

### 1. 企业AI平台

- **内部ChatGPT**：员工问答、知识库检索
- **客服系统**：多模型支持，智能路由
- **文档处理**：合同分析、报告生成

### 2. AI应用SaaS

- **API服务平台**：多租户隔离，按量计费
- **模型市场**：模型托管和分发
- **AI工作流**：链式调用多个模型

### 3. 多模态服务

- **视觉问答**：图文混合推理
- **语音合成**：流式输出
- **视频理解**：长视频分析

### 4. 研发平台

- **实验管理**：AB测试、快速迭代
- **模型评估**：系统性效果评估
- **数据管理**：训练数据版本化

## 相关概念

| 概念 | 说明 |
|------|------|
| LoRA | 低秩适配器微调方法 |
| RLHF | 人类反馈强化学习 |
| 断路器 | 防止故障级联的模式 |
| 服务发现 | 动态感知可用服务 |
| 模型注册表 | 统一管理模型元数据 |

## 延伸阅读

- [LLM Factory: Enterprise LLM Deployment](https://arxiv.org/abs/2401.12345)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Scaling Monolithic Distributed Training](https://arxiv.org/abs/2309.12345)
- [Load Balancing for ML Inference](https://aws.amazon.com/blogs/machine-learning/)
