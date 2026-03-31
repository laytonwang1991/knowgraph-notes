---
title: K近邻
alias: K-Nearest Neighbors (KNN)
tags:
  - 机器学习
  - 基于实例的学习
  - 惰性学习
  - 监督学习
category: 监督学习
created: 2026-03-31
updated: 2026-03-31
author: AI Assistant
description: 基于实例的学习方法，通过找到测试样本的K个最近邻居来预测其类别或值，是一种简单直观的非参数方法。
mastery: 0.7
rating: 8
related_concepts:
  - 距离度量
  - K值选择
  - KD树
  - 球树
  - 惰性学习
  - 维度灾难
difficulty: 简单
read_time: 15
prerequisites:
  - 距离度量（欧氏距离、曼哈顿距离）
  - 排序算法
  - 基本统计学概念
---

# K近邻（K-Nearest Neighbors, KNN）

## 一句话定义

K近邻是一种基于实例的分类/回归算法，通过计算测试样本与训练集中所有样本的距离，找到最近的K个邻居，根据邻居的标签投票或平均值来预测该样本的类别或数值。

## 核心公式

**欧氏距离（L2距离）：**

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} = \|x - y\|_2$$

**曼哈顿距离（L1距离）：**

$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i| = \|x - y\|_1$$

**闵可夫斯基距离（Lp距离）：**

$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

**余弦相似度：**

$$\cos(x, y) = \frac{x \cdot y}{\|x\| \|y\|}$$

**KNN分类投票：**

$$\hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} I(y_i = c)$$

**KNN回归（加权平均）：**

$$\hat{y} = \frac{\sum_{i \in N_K(x)} w_i y_i}{\sum_{i \in N_K(x)} w_i}$$

其中 $w_i = \frac{1}{d(x, x_i)}$（距离倒数作为权重）

## 详细说明

### 1. 算法原理

- **非参数方法**：KNN不学习显式的模型参数，训练阶段只是存储训练数据
- **惰性学习**：所有计算都推迟到预测时进行，因此训练时间复杂度为O(1)
- **距离度量**：通过某种距离度量衡量样本之间的相似性
- **K值选择**：K值的选择对模型性能有重要影响
- **投票机制**：分类时使用多数投票，回归时使用加权平均

### 2. 距离度量方法

| 度量方法 | 公式 | 适用场景 |
|----------|------|----------|
| **欧氏距离** | $\sqrt{\sum(x_i-y_i)^2}$ | 连续特征，特征尺度相近 |
| **曼哈顿距离** | $\sum|x_i-y_i|$ | 网格状路径，高维数据 |
| **闵可夫斯基** | 上述Lp距离的泛化 | 需要调参p的场景 |
| **余弦相似度** | $\frac{x \cdot y}{\|x\|\|y\|}$ | 文本相似度，高维稀疏数据 |
| **马氏距离** | $\sqrt{(x-y)^T S^{-1}(x-y)}$ | 特征相关性重要的场景 |

### 3. K值选择

- **K值太小**：容易过拟合，对噪声敏感
- **K值太大**：模型过于简单，可能欠拟合
- **通常选择**：使用交叉验证选择最优K值，常用范围3-20
- **奇数原则**：对于二分类，建议选择奇数避免平票

### 4. KD树与球树

**KD树（K-Dimensional Tree）：**
- 一种高效检索K维空间最近邻的数据结构
- 构建复杂度：$O(n \log n)$
- 搜索复杂度（平均）：$O(\log n)$
- 适用维度：一般小于20维

**球树（Ball Tree）：**
- 使用超球体划分空间的数据结构
- 比KD树更适合高维数据
- 构建复杂度：$O(n \log n)$
- 搜索复杂度：$O(\log n)$

### 5. 优缺点

**优点：**
- 简单直观，易于理解和实现
- 不需要训练过程（惰性学习）
- 对数据分布没有假设
- 自然支持多分类
- 适合作为基线模型

**缺点：**
- 预测时间复杂度高 $O(n \cdot d)$
- 对特征尺度敏感
- 维度灾难问题
- 对噪声和异常值敏感
- 存储成本高（需要存储全部训练数据）

## 代码示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 生成示例数据（KNN分类）
np.random.seed(42)
class_0 = np.random.randn(100, 2) + np.array([-2, -2])
class_1 = np.random.randn(100, 2) + np.array([2, 2])
class_2 = np.random.randn(100, 2) + np.array([-2, 2])
X = np.vstack([class_0, class_1, class_2])
y = np.hstack([np.zeros(100), np.ones(100), np.ones(100) * 2])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 特征标准化（重要！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用网格搜索选择最优K值
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"最优K值: {grid_search.best_params_['n_neighbors']}")
print(f"最优交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最优模型预测
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 可视化决策边界
def plot_knn_decision_boundary(X, y, model, scaler):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
    plt.show()

plot_knn_decision_boundary(X_test, y_test, best_knn, scaler)
```

```python
# 从零实现KNN分类器
class KNNClassifier:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'cosine':
            return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            # 计算与所有训练样本的距离
            distances = [self._compute_distance(x, x_train)
                        for x_train in self.X_train]
            # 找到K个最近邻
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            # 多数投票
            counts = np.bincount(k_nearest_labels.astype(int))
            predictions.append(np.argmax(counts))
        return np.array(predictions)


# KNN回归实现（带距离加权）
class KNNRegressor:
    def __init__(self, k=5, weighted=True):
        self.k = k
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_distances = distances[k_nearest_indices]
            k_nearest_values = self.y_train[k_nearest_indices]

            if self.weighted:
                # 距离倒数作为权重，避免除零
                weights = 1 / (k_nearest_distances + 1e-9)
                predictions.append(np.sum(weights * k_nearest_values) / np.sum(weights))
            else:
                predictions.append(np.mean(k_nearest_values))
        return np.array(predictions)
```

```python
# KD树实现
from collections import PriorityQueue

class KDNode:
    def __init__(self, point, left=None, right=None, axis=0):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

class KDTree:
    def __init__(self, leaf_size=1):
        self.root = None
        self.leaf_size = leaf_size

    def _build(self, points, depth=0):
        if len(points) <= self.leaf_size:
            return KDNode(points)

        axis = depth % points.shape[1]
        points = points[points[:, axis].argsort()]
        mid = len(points) // 2

        return KDNode(
            point=points[mid],
            left=self._build(points[:mid], depth + 1),
            right=self._build(points[mid+1:], depth + 1),
            axis=axis
        )

    def build(self, X):
        self.root = self._build(np.array(X))

    def _query(self, node, target, k):
        if node is None:
            return []

        # 计算当前节点距离
        dist = np.linalg.norm(node.point - target)
        results = [(dist, node.point)]

        # 递归搜索
        axis = node.axis
        if target[axis] < node.point[axis]:
            next_node = node.left
            other_node = node.right
        else:
            next_node = node.right
            other_node = node.left

        if next_node:
            results.extend(self._query(next_node, target, k))
        if other_node:
            # 剪枝：检查是否需要搜索另一侧
            if len(results) < k or abs(target[axis] - node.point[axis]) < results[k-1][0]:
                results.extend(self._query(other_node, target, k))

        return results

    def query(self, target, k=1):
        results = self._query(self.root, target, k)
        results.sort(key=lambda x: x[0])
        return results[:k]
```

## 应用场景

1. **分类问题**：
   - 手写数字识别（MNIST）
   - 图像识别（人脸识别中的特征匹配）
   - 推荐系统（协同过滤）
   - 异常检测

2. **回归问题**：
   - 时间序列预测
   - 房价预测
   - 销量预测

3. **搜索相关**：
   - 相似商品推荐
   - 文本相似度检索
   - 最近邻图像搜索

4. **数据预处理**：
   - 缺失值填充（用最近邻的值替代）
   - 特征匹配

## 相关概念

- **距离度量**：衡量样本间相似性的度量方法
- **K值选择**：决定考虑多少个最近邻
- **KD树**：K维空间的高效最近邻索引结构
- **球树**：另一种空间划分结构，比KD树更适合高维
- **惰性学习**：训练时不学习，预测时才计算
- **维度灾难**：高维空间中距离度量失效的问题
- **多数投票**：KNN分类的决策机制

## 延伸阅读

1. **基础理论**：
   - 李航《统计学习方法》第3章
   - "The Elements of Statistical Learning" 第13章
   - Cover & Hart, "Near Neighbor Classification"

2. **距离度量学习**：
   - LMNN (Large Margin Nearest Neighbor)
   - NCA (Neighborhood Components Analysis)
   - 学习适合数据的距离度量

3. **优化技术**：
   - KD树和球树的构建算法
   - 近似最近邻搜索（LSH、ANNOY、HNSW）
   - 如何加速KNN计算

4. **实践资源**：
   - Scikit-learn neighbors 模块文档
   - FAISS（Facebook AI相似度搜索库）
   - Annoy（Spotify近似最近邻库）
