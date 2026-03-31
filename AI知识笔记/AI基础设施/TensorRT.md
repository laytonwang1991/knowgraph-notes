---
title: TensorRT
alias: TensorRT
tags:
  - 推理优化
  - NVIDIA
  - GPU
  - INT8量化
  - 深度学习
category: AI基础设施
created: 2026-03-31
updated: 2026-03-31
author: AI Infrastructure Team
description: NVIDIA 的高性能深度学习推理优化库，提供 INT8 量化、层融合、内核自动调优等能力。
mastery: 9
rating: 10
related_concepts:
  - 模型推理优化
  - ONNX Runtime
  - CUDA
  - 模型量化
  - 层融合
difficulty: 高
read_time: 28
prerequisites:
  - CUDA 编程基础
  - 深度学习模型部署
  - GPU 架构知识
---

# TensorRT

## 一句话定义

TensorRT 是 NVIDIA 开发的高性能深度学习推理引擎，通过层融合、内核自动调优、INT8 量化、动态形状等技术将模型部署到 NVIDIA GPU 并获得极致性能。

## 详细说明

### 1. INT8 量化

INT8 量化通过将 FP32 权重和激活值转换为 INT8 表示，显著提升吞吐量和降低内存占用。

**核心原理：**
- INT8 精度足以满足大多数深度学习推理任务
- INT8 指令吞吐量是 FP16 的 2x，是 FP32 的 4x
- TensorRT 提供校准（Calibration）机制确保精度

**量化流程：**
```python
import tensorrt as trt

# Step 1: 创建 builder 和 network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# Step 2: 导入 ONNX 模型
import onnx
onnx_model = onnx.load("model.onnx")
onnx_helper = trt.OnnxParser(network, logger)
onnx_helper.parse(onnx_model.SerializeToString())

# Step 3: 配置 INT8 量化
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)  # FP16 作为备选

# Step 4: 设置 INT8 校准器
class CalibrationData:
    """提供校准数据"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.max_batch_size = 32

    def get_batch(self, names):
        """返回一批校准数据"""
        try:
            batch = next(self.data_loader)
            return batch.astype(np.float32)
        except StopIteration:
            return None

    def get_batch_size(self):
        return self.max_batch_size

# 使用 Python API 不支持原生 INT8 校准，需要通过 trt 模块
# 实际使用建议通过 Polygraphy 或 TensorRT OSS
```

**使用 Polygraphy 进行 INT8 量化（推荐方式）：**
```bash
# 安装 polygraphy
pip install polygraphy

# 1. 生成校准数据缓存
polygraphy run model.onnx \
    --trt \
    --int8 \
    --calibration-cache ./calibration_cache.bin \
    --batch-size 32 \
    --data-loader-file calibration_data.py \
    --save-engine model_int8.engine

# 2. calibration_data.py 示例
"""
import numpy as np

def load_calibration_data():
    # 返回校准数据迭代器
    for _ in range(100):
        yield np.random.randn(32, 3, 224, 224).astype(np.float32)
"""
```

### 2. 层融合（Layer Fusion）

层融合是 TensorRT 最重要的优化手段，将多个计算层合并为单个内核。

**常见融合模式：**

| 融合前 | 融合后 | 收益 |
|--------|--------|------|
| Conv + BN + ReLU | Single Conv | 减少内存访问 3x |
| Conv + Add + ReLU | Single Conv | 减少内存访问 2x |
| MatMul + Softmax | Single Kernel | 融合注意力计算 |
| Multiple Residual Add | Single Add | 减少同步开销 |

```python
# TensorRT 自动进行层融合，可通过 API 检查
# 构建时启用详细日志查看融合过程
logger = trt.Logger(trt.Logger.VERBOSE)
# ... 构建过程 ...
# VERBOSE 日志会显示每个层的融合信息
```

### 3. 内核自动调优（Kernel Auto-Tuning）

TensorRT 会为给定 GPU 架构自动选择最优 CUDA kernel 实现。

**调优过程：**
```python
# 构建时启用 Tactic Sources 控制调优范围
config = builder.create_builder_config()
config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) |
                         1 << int(trt.TacticSource.CUDNN) |
                         1 << int(trt.TacticSource.TENSORRT))

# 控制优化级别（影响调优时间）
# 0: 快速构建，不调优
# 1: 默认调优
# 2: 深度调优
# 3: 最多调优（构建时间很长）
config.set_builder_optimization_level(3)

# 构建 engine（首次构建会进行内核调优）
engine = builder.build_serialized_network(network, config)
```

**持久化缓存 Tuning Cache：**
```python
# 保存 tuning cache 避免重复调优
cache = builder.create_tuning_cache()
config.set_tuning_cache(cache)

# 保存到文件
with open("tuning_cache.bin", "wb") as f:
    f.write(cache.serialize())

# 下次构建时加载
with open("tuning_cache.bin", "rb") as f:
    cache = builder.create_tuning_cache(f.read())
    config.set_tuning_cache(cache)
```

### 4. 动态形状（Dynamic Shapes）

动态形状支持使一个 engine 可以处理不同尺寸的输入。

**配置动态维度：**
```python
import tensorrt as trt
import numpy as np

# 创建具有动态输入的 network
# 假设输入维度为 [batch, channels, height, width]
# 其中 batch 和 height/width 是动态的

# 使用 -1 或 trt.Dimension.STATIC_KEY 标记动态维度
batch_dim = -1  # 动态 batch size
height_dim = -1  # 动态高度
width_dim = -1   # 动态宽度

# 标记输入维度
input_tensor = network.get_input(0)
input_tensor.shape = [batch_dim, 3, height_dim, width_dim]

# 配置优化配置
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# 设置动态范围（用于 INT8）
# 需要为每个可能的 shape 提供校准数据
profile = builder.create_optimization_profile()
profile.set_shape_input(input_tensor.name,
                        min=(1, 3, 224, 224),      # 最小
                        opt=(4, 3, 224, 224),      # 优化
                        max=(16, 3, 512, 512))     # 最大
config.add_optimization_profile(profile)

# 构建 engine
engine = builder.build_serialized_network(network, config)
```

**运行时指定实际 shape：**
```python
# 创建 execution context
context = engine.create_execution_context()

# 绑定优化 profile
profile_idx = 0
context.set_optimization_profile_async(profile_idx, stream)

# 设置实际输入 shape
context.set_input_shape("input", (8, 3, 224, 224))

# 创建输出 buffer
output = np.zeros((8, 1000), dtype=np.float32)

# 推理
context.execute_v2(bindings=[input_device_ptr, output_device_ptr])
```

### 5. 完整 Pipeline 示例

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda auto init
import numpy as np
import onnx

class TensorRTEngine:
    """TensorRT 推理引擎封装"""

    def __init__(self, engine_path=None, onnx_path=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        if engine_path:
            # 加载已构建的 engine
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        elif onnx_path:
            # 从 ONNX 构建 engine
            self.engine = self._build_engine(onnx_path)

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def _build_engine(self, onnx_path, max_batch_size=32):
        """从 ONNX 文件构建 TensorRT engine"""

        # 创建 builder 和 network
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()

        # 启用 FP16
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # 设置 memory pool
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # 解析 ONNX
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        # 设置动态 batch
        profile = builder.create_optimization_profile()
        profile.set_shape("input",
                         min=(1, 3, 224, 224),
                         opt=(max_batch_size, 3, 224, 224),
                         max=(max_batch_size, 3, 512, 512))
        config.add_optimization_profile(profile)

        # 构建 engine
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            raise RuntimeError("TensorRT engine build failed")

        return engine

    def allocate_buffers(self):
        """预分配 GPU buffer"""
        self.h_inputs = []
        self.d_inputs = []
        self.h_outputs = []
        self.d_outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)

            # 计算内存大小
            size = trt.volume(shape) * self.engine.get_tensor_dtype(name).itemsize
            host_mem = cuda.pagelocked_empty(shape, dtype=np.float32)
            device_mem = cuda.mem_alloc(size)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.h_inputs.append(host_mem)
                self.d_inputs.append(device_mem)
            else:
                self.h_outputs.append(host_mem)
                self.d_outputs.append(device_mem)

    def inference(self, input_data):
        """执行推理"""
        # 拷贝输入数据
        np.copyto(self.h_inputs[0], input_data.ravel())
        cuda.memcpy_htod_async(self.d_inputs[0], self.h_inputs[0], self.stream)

        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # 拷贝输出数据
        cuda.memcpy_dtoh_async(self.h_outputs[0], self.d_outputs[0], self.stream)
        self.stream.synchronize()

        return self.h_outputs[0].reshape(self.engine.get_tensor_shape("output"))
```

## 应用场景

| 场景 | 推荐配置 | 预期收益 |
|------|----------|----------|
| 生产环境推理 | FP16 + 层融合 | 延迟降低 5-10x |
| 高精度需求场景 | FP32 完整精度 | 保持模型精度 |
| 边缘部署 | INT8 量化 | 延迟降低 2-3x，显存减少 4x |
| 实时推理（视频） | FP16 + 动态 batch | 吞吐提升 10x+ |
| 批量推理 | INT8 + 静态 shape | 吞吐提升 15x+ |

## 相关概念

- **模型推理优化**: 通用推理优化方法论
- **ONNX Runtime**: 跨平台通用推理引擎
- **CUDA**: NVIDIA GPU 并行计算平台
- **层融合**: 将多个计算层合并减少内存访问
- **模型量化**: INT8/FP16 等低精度推理

## 延伸阅读

1. **TensorRT 官方文档**: https://docs.nvidia.com/deeplearning/tensorrt/
2. **TensorRT GitHub**: https://github.com/NVIDIA/TensorRT
3. **TensorRT Samples**: https://github.com/NVIDIA/TensorRT/tree/main/samples
4. **Polygraphy**: https://github.com/NVIDIA/pygraphy - TensorRT 调优工具
5. **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM - LLM 专用推理优化
