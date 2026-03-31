---
title: ONNX Runtime
alias: ONNX Runtime
tags:
  - 推理引擎
  - 跨平台
  - ONNX
  - 模型部署
  - 性能优化
category: AI基础设施
created: 2026-03-31
updated: 2026-03-31
author: AI Infrastructure Team
description: 跨平台高性能机器学习推理引擎，支持 ONNX 格式模型的优化部署。
mastery: 8
rating: 9
related_concepts:
  - 模型推理优化
  - TensorRT
  - 模型量化
  - ONNX
  - 跨平台部署
difficulty: 中高
read_time: 22
prerequisites:
  - 深度学习基础
  - 模型部署概念
  - Python/C++ 编程
---

# ONNX Runtime

## 一句话定义

ONNX Runtime 是微软开源的跨平台高性能机器学习推理引擎，通过统一的运行时环境优化 ONNX 格式模型的执行性能，支持 CPU、GPU、边缘设备等多种部署目标。

## 详细说明

### 1. ONNX 格式概述

ONNX（Open Neural Network Exchange）是用于表示深度学习模型的开放标准格式。

**核心特性：**
- 框架无关性：PyTorch、TensorFlow、Scikit-learn 等主流框架均可导出
- 算子标准化：定义了一套统一的算子集（大约 100+ 种）
- 格式可扩展：通过扩展节点支持自定义算子
- 跨平台流通：训练与推理环境解耦

**模型导出示例：**
```python
import torch
import torch.onnx

# PyTorch 模型导出 ONNX
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.output(x)
        return x

model = SimpleModel()
model.eval()

# 虚拟输入用于形状推断
dummy_input = torch.randn(1, 128)

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=17,  # ONNX 算子集版本
    do_constant_folding=True,  # 常量折叠优化
    export_params=True
)

print("模型已导出为 ONNX 格式")
```

### 2. Runtime 优化技术

ONNX Runtime 提供了多层次的优化策略。

**图优化（Graph Optimization）：**
```python
# 启用图优化级别
# 0: 不优化
# 1: 基本优化（常数折叠、冗余节点消除）
# 2: 扩展优化（算子融合、维度推断）
# 99: 所有优化
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 创建推理会话
session = ort.InferenceSession("model.onnx", sess_options=session_options)
```

**内存池与分配策略：**
```python
# 配置内存 arena
session_options = ort.SessionOptions()
session_options.enable_mem_pattern = True      # 启用内存模式
session_options.enable_cpu_mem_arena = True   # CPU 内存池
session_options.memory.enable_memory_arena = True

# GPU 内存池配置
session_options.provider_options[0] = {
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # CUDNN 卷积算法搜索
    'do_copy_in_default_stream': True
}
```

### 3. 执行提供者（Execution Providers）

ONNX Runtime 支持多种硬件加速后端，通过统一的 API 调用。

```python
import onnxruntime as ort

# 可用的执行提供者
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True,
    }),
    ('CPUExecutionProvider', {}),  # 兜底
]

session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=providers
)

# 查看当前使用的提供者
print(session.get_providers())
# 输出: ['CUDAExecutionProvider', 'CPUExecutionProvider']

# 绑定输入输出
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 推理
results = session.run([output_name], {input_name: input_data})
```

**常见执行提供者：**

| 提供者 | 硬件 | 特点 |
|--------|------|------|
| CUDAExecutionProvider | NVIDIA GPU | 完整 CUDA 加速 |
| CPUExecutionProvider | x86/ARM CPU | 通用优化 |
| TensorRTExecutionProvider | NVIDIA GPU | 深度优化（见 TensorRT 笔记） |
| OpenVINOExecutionProvider | Intel 硬件 | CPU/iGPU/VPU 优化 |
| CoreMLExecutionProvider | Apple Silicon | iOS/macOS 原生 |
| QNNExecutionProvider | Qualcomm DSP/NPU | 移动端优化 |

### 4. 量化与部署

ONNX Runtime 支持多种量化策略，显著降低延迟和内存占用。

**动态量化（Dynamic Quantization）：**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# 动态量化 - 权重 INT8，激活 fp32
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8,
    optimize_model=True,  # 先优化模型再量化
)

# 使用量化模型
session = ort.InferenceSession("model_quantized.onnx", providers=['CPUExecutionProvider'])
```

**静态量化（Static Quantization）：**
```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

# Step 1: 校准 - 收集激活值分布
def calibration_data_reader():
    """提供校准数据"""
    for i in range(100):
        yield {"input": torch.randn(1, 128).numpy()}

# Step 2: 执行静态量化
quantize_static(
    model_input="model.onnx",
    model_output="model_static_quant.onnx",
    calibration_data_reader=calibration_data_reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)

# Step 3: 验证量化模型精度
session = ort.InferenceSession("model_static_quant.onnx")
```

**INT8 量化的高级配置：**
```python
from onnxruntime.quantization import QuantizationMode, QuantFormat

# 详细的量化配置
quantize_static(
    model_input="model.onnx",
    model_output="model_awareness_quant.onnx",
    calibration_data_reader=calibration_data_reader,
    quantization_mode=QuantizationMode.QLinearOps,
    quant_format=QuantFormat.QOperator,  # QOperator 或 QDQ
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    op_types_to_quantize=[
        'Conv', 'MatMul', 'Relu', 'Add', 'Mul'
    ],
    nodes_to_quantize=[],  # 指定量化的节点，为空则量化所有适用节点
    nodes_to_exclude=[],  # 排除量化的节点
)
```

### 5. 性能调优与 Profiling

```python
import onnxruntime as ort

# 创建带 profiling 的会话
session_options = ort.SessionOptions()
session_options.enable_profiling = True
session_options.profile_file_prefix = "./onnx_profile"

# 启用执行时间日志
session_options.enable_run_metrics = True

session = ort.InferenceSession("model.onnx", sess_options=session_options)

# 多次运行预热
for _ in range(10):
    session.run(None, {input_name: input_data})

# 计时推理
import time

runs = 100
start = time.perf_counter()
for _ in range(runs):
    results = session.run(None, {input_name: input_data})
end = time.perf_counter()

print(f"平均延迟: {(end - start) / runs * 1000:.2f} ms")
print(f"吞吐量: {runs / (end - start):.2f} samples/sec")

# 导出 profiling 数据
profile_file = session.end_profiling()
print(f"Profiling 文件: {profile_file}")
```

## 应用场景

| 场景 | 推荐配置 | 预期收益 |
|------|----------|----------|
| 云端 CPU 推理 | 图优化 + 动态量化 | 延迟降低 2-3x |
| NVIDIA GPU 部署 | CUDA EP + TensorRT EP | 接近 TensorRT 性能 |
| Intel CPU 推理 | OpenVINO EP | 延迟降低 3-5x |
| Apple 设备部署 | CoreML EP | 能耗降低 60%+ |
| 移动端部署 | QNN EP 或 CoreML EP | 体积小、功耗低 |
| 高精度推理 | FP32 完整精度 | 保持模型精度 |

## 相关概念

- **模型推理优化**: 推理优化的通用方法论
- **TensorRT**: NVIDIA GPU 的深度优化推理引擎
- **模型量化**: INT8/FP16 量化技术降低精度损失
- **ONNX**: 开放的模型交换格式标准
- **跨平台部署**: 一套模型，多端部署

## 延伸阅读

1. **ONNX 官方文档**: https://onnxruntime.ai/docs/
2. **ONNX Runtime GitHub**: https://github.com/microsoft/onnxruntime
3. **量化文档**: https://onnxruntime.ai/docs/performance/quantization.html
4. **执行提供者**: https://onnxruntime.ai/docs/execution-providers/
5. **PyTorch 模型导出**: https://pytorch.org/docs/stable/onnx.html
