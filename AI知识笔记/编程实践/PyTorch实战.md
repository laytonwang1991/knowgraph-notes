---
title: PyTorch实战
alias: PyTorch Practice
tags:
  - PyTorch
  - 深度学习
  - 神经网络
  - 机器学习
category: 编程实践
created: 2026-03-31
updated: 2026-03-31
author: AI Practice Team
description: PyTorch深度学习框架的实际编程技巧与最佳实践，涵盖数据加载、自定义层、训练循环、模型保存与加载等核心内容。
mastery: 85
rating: 9
related_concepts:
  - 神经网络
  - 反向传播
  - GPU加速
  - 模型优化
  - 分布式训练
difficulty: 中级
read_time: 45分钟
prerequisites:
  - Python基础
  - 深度学习基础
  - 线性代数基础
---

# PyTorch实战

## 一句话定义

PyTorch实战是使用Meta公司开发的开源深度学习框架PyTorch进行神经网络构建、训练和部署的实际编程过程，核心在于利用其动态计算图和GPU加速能力实现高效模型开发。

## 详细说明

### 1. 数据加载 (Data Loading)

PyTorch提供了一套完整的数据处理和加载机制，主要包括：

- **Dataset**：自定义数据抽象类，需要实现`__len__`和`__getitem__`方法
- **DataLoader**：批量加载数据，支持多进程、随机 shuffle、自动批处理
- **transforms**：图像数据的预处理和增强工具

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_data()

    def _load_data(self):
        # 实际应用中从文件或数据库加载数据
        return [(torch.randn(3, 224, 224), torch.randint(0, 10, (1,))) for _ in range(1000)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 定义数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据加载器
train_dataset = CustomDataset('path/to/data', transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
```

### 2. 自定义层 (Custom Layers)

当预置层无法满足需求时，PyTorch允许灵活地创建自定义层：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义全连接层
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout_rate=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

# 自定义多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)

# 组合自定义层构建模型
class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=8):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.fc1 = CustomLinear(input_dim, hidden_dim)
        self.fc2 = CustomLinear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        x = self.fc2(x)
        return self.classifier(x)
```

### 3. 训练循环 (Training Loop)

完整的训练循环包含前向传播、损失计算、反向传播和参数更新：

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss / len(val_loader), 100. * correct / total

# 主训练流程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomModel(512, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

num_epochs = 100
best_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
```

### 4. 模型保存与加载 (Model Save & Load)

PyTorch提供了灵活的模型保存和加载机制：

```python
# 保存整个模型和参数
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc,
}, 'checkpoint.pth')

# 加载模型
def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    return epoch, best_acc

# 仅保存模型参数（推荐方式，便于迁移学习）
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型参数
model = CustomModel(512, 256, 10)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()  # 设置为评估模式

# 模型导出为ONNX格式
dummy_input = torch.randn(1, 512).to(device)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## 代码示例

### 完整的图像分类训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设置随机种子保证可复现性
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# GPU配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
val_dataset = datasets.CIFAR10(root='./data', train=False, transform=val_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 定义简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练和验证
for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    print(f'Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%')
```

## 应用场景

1. **计算机视觉**：图像分类、目标检测、语义分割、人脸识别等视觉任务
2. **自然语言处理**：文本分类、命名实体识别、机器翻译、文本生成
3. **语音处理**：语音识别、语音合成、音频分类
4. **推荐系统**：用户行为序列建模、 embedding学习
5. **强化学习**：游戏AI、机器人控制、自动驾驶决策
6. **科研实验**：快速验证新算法、新模型架构

## 相关概念

| 概念 | 说明 |
|------|------|
| 动态计算图 | PyTorch的核心特性，支持即时执行便于调试 |
| GPU加速 | CUDA张量操作，显著提升训练速度 |
| Autograd | 自动微分系统，自动计算梯度 |
| nn.Module | 所有神经网络模块的基类 |
| optim | 优化器模块，包含SGD、Adam、RMSprop等 |
| 分布式训练 | DataParallel和DistributedDataParallel多GPU训练 |

## 延伸阅读

1. [PyTorch官方文档](https://pytorch.org/docs/) - 完整的API参考和教程
2. [PyTorch torchvision](https://pytorch.org/vision/) - 计算机视觉专用库
3. [PyTorch Lightning](https://www.pytorchlightning.ai/) - 轻量级训练封装框架
4. [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - 预训练模型库
5. [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) - JIT编译加速
6. [ONNX](https://onnx.ai/) - 模型跨平台部署标准
