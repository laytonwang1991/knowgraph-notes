---
title: Agent评估
alias: AI Agent Evaluation
tags: [Agent, 评估, 基准测试, HumanEval, 成功率]
category: AI Agent
created: 2026-03-31
updated: 2026-03-31
author: AI Developer
description: 系统性评估AI Agent性能的方法论，涵盖任务完成率、效率指标、成功率等核心维度，以及HumanEval for Agents等权威基准测试。
mastery: 3
rating: 8
related_concepts: [AI Agent, 提示工程, Tool-use, Agent架构]
difficulty: advanced
read_time: 18
prerequisites: [Agent架构基础, 提示工程]
---

# Agent评估

## 一句话定义

Agent评估是衡量AI Agent在真实任务中表现的系统性方法，通过任务完成率、效率指标、成功率等多维度指标，以及HumanEval for Agents等标准化基准测试，全面量化Agent的能力边界和实用性。

## 详细说明

### 1. 核心评估维度

#### 1.1 任务完成率（Task Completion Rate）

衡量Agent成功完成给定任务的比例，是最直接的效能指标。

**计算公式**：
```
任务完成率 = 成功完成任务数 / 总任务数 × 100%
```

**评估标准**：
- 完全成功：输出正确、完整，无需后续修正
- 部分成功：核心目标达成，但存在瑕疵或需人工补充
- 失败：无法完成任务或输出完全错误

#### 1.2 效率指标（Efficiency Metrics）

衡量Agent完成任务所需的资源消耗和时间成本。

**关键指标**：
- **Token消耗率**：完成任务平均消耗的token数量
- **响应延迟**：从接收任务到首次输出的时间
- **迭代次数**：ReAct等循环模式中的平均循环次数
- **工具调用次数**：完成任务所需的工具调用总次数

**计算公式**：
```
效率得分 = 质量权重 × 成功率 + 成本权重 × (1 - 相对token消耗) + 时间权重 × (1 - 相对响应时间)
```

#### 1.3 成功率（Success Rate）

特定类型任务的成功率，用于识别Agent的能力长短板。

**分类维度**：
- 按任务类型（问答、代码生成、数据分析等）
- 按难度等级（简单、中等、复杂）
- 按领域（金融、医疗、技术等）

#### 1.4 轨迹质量（Trajectory Quality）

评估Agent完成任务过程中推理路径的质量。

**评估要素**：
- 推理连贯性：思考步骤之间的逻辑关系
- 工具选择合理性：是否选择了合适的工具
- 错误恢复能力：遇到错误时的处理策略
- 路径最优性：是否选择了最高效的解决路径

### 2. HumanEval for Agents

由OpenAI等机构提出的，专门用于评估Agent代码生成和问题解决能力的基准测试集。

#### 2.1 核心测试类别

| 测试类别 | 描述 | 示例任务 |
|---------|------|---------|
| 代码生成 | 生成可执行的代码 | "写一个快速排序函数" |
| 代码修复 | 修复有bug的代码 | "修复以下Python代码的越界错误" |
| 代码优化 | 改进代码性能 | "优化以下SQL查询" |
| 测试生成 | 生成测试用例 | "为以下函数生成单元测试" |
| 代码解释 | 解释代码逻辑 | "解释这段正则表达式的作用" |
| 多文件协作 | 跨文件代码理解 | "在项目中添加新功能" |

#### 2.2 评估流程

```python
import anthropic
from typing import List, Dict
import json

client = anthropic.Anthropic()

class HumanEvalAgent:
    """HumanEval for Agents 评估框架"""

    def __init__(self, benchmark_file: str):
        self.benchmark_file = benchmark_file
        self.results = []

    def evaluate(self, agent_fn, num_samples: int = 100) -> Dict:
        """运行评估"""
        with open(self.benchmark_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)[:num_samples]

        passed = 0
        detailed_results = []

        for task in tasks:
            result = self._evaluate_single(agent_fn, task)
            if result['passed']:
                passed += 1
            detailed_results.append(result)

        success_rate = passed / len(tasks) * 100

        return {
            'success_rate': success_rate,
            'passed_count': passed,
            'total_count': len(tasks),
            'details': detailed_results,
            'metrics': self._compute_metrics(detailed_results)
        }

    def _evaluate_single(self, agent_fn, task: Dict) -> Dict:
        """评估单个任务"""
        prompt = self._build_prompt(task)

        try:
            response = agent_fn(prompt)
            execution_result = self._execute_and_validate(response, task)

            return {
                'task_id': task['task_id'],
                'passed': execution_result['passed'],
                'execution_output': execution_result.get('output'),
                'expected': task.get('expected_output'),
                'token_used': response.get('usage', {}).get('total_tokens', 0)
            }
        except Exception as e:
            return {
                'task_id': task['task_id'],
                'passed': False,
                'error': str(e)
            }

    def _build_prompt(self, task: Dict) -> str:
        """构建任务提示"""
        return f"""{task['description']}

要求：
{task['requirements']}

请完成代码实现。"""

    def _execute_and_validate(self, response, task: Dict) -> Dict:
        """执行代码并验证结果"""
        # 提取代码
        code = self._extract_code(response)

        # 执行代码
        try:
            exec_globals = {}
            exec(code, exec_globals)

            # 验证结果
            test_cases = task.get('test_cases', [])
            for test in test_cases:
                expected = test['expected']
                actual = eval(test['call'], exec_globals)
                if actual != expected:
                    return {'passed': False, 'output': f'测试失败: {actual} != {expected}'}

            return {'passed': True, 'output': '所有测试通过'}
        except Exception as e:
            return {'passed': False, 'output': f'执行错误: {str(e)}'}

    def _extract_code(self, response) -> str:
        """从响应中提取代码"""
        content = response.content[0].text
        # 提取markdown代码块
        if '```' in content:
            parts = content.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # 代码块内容
                    lines = part.split('\n')
                    if lines[0].startswith('python'):
                        return '\n'.join(lines[1:])
                    return part
        return content

    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """计算详细指标"""
        total_tokens = sum(r.get('token_used', 0) for r in results)
        passed_results = [r for r in results if r['passed']]

        return {
            'total_tokens': total_tokens,
            'avg_tokens_per_task': total_tokens / len(results),
            'avg_tokens_per_passed': total_tokens / len(passed_results) if passed_results else 0,
            'pass_by_difficulty': self._analyze_by_difficulty(results)
        }

    def _analyze_by_difficulty(self, results: List[Dict]) -> Dict:
        """按难度分析结果"""
        # 简化实现，实际需要关联任务难度信息
        return {'easy': 0, 'medium': 0, 'hard': 0}
```

### 3. 评估框架与工具

#### 3.1 AgentBench

由清华大学提出的多领域Agent评估基准，覆盖8个真实场景。

```python
# AgentBench 评估调用示例
def evaluate_with_agentbench(agent, benchmark_name: str):
    """使用AgentBench进行评估"""
    agentbench_config = {
        'database': '评估数据库路径',
        'domains': ['shopping', 'code', 'knowledge', 'digital_platforms'],
        'max_tokens': 4096,
        'temperature': 0.0
    }

    evaluator = AgentBenchEvaluator(agentbench_config)
    results = evaluator.run(agent, benchmark_name)

    return {
        'overall_score': results['score'],
        'domain_scores': results['domain_breakdown'],
        'trajectory_analysis': results['trajectory_metrics']
    }
```

#### 3.2 GAIA基准测试

通用AI助手评估基准，涵盖真实世界问答、数据处理、多步骤推理。

```python
# GAIA 评估框架
class GAIABenchmark:
    def __init__(self):
        self.tasks = self._load_gaia_tasks()

    def evaluate(self, agent) -> Dict:
        results = []
        for task in self.tasks:
            # 模拟GAIA任务评估
            result = {
                'task_id': task['id'],
                'category': task['category'],
                'requires_tools': task.get('requires_tools', False),
                'passed': self._validate_response(agent, task)
            }
            results.append(result)

        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        category_stats = {}
        for r in results:
            cat = r['category']
            if cat not in category_stats:
                category_stats[cat] = {'passed': 0, 'total': 0}
            category_stats[cat]['total'] += 1
            if r['passed']:
                category_stats[cat]['passed'] += 1

        return {
            'overall_accuracy': sum(r['passed'] for r in results) / len(results),
            'by_category': {k: v['passed']/v['total'] for k, v in category_stats.items()},
            'tool_dependency_analysis': self._analyze_tool_dependency(results)
        }

    def _analyze_tool_dependency(self, results: List[Dict]) -> Dict:
        with_tools = [r for r in results if r['requires_tools']]
        without_tools = [r for r in results if not r['requires_tools']]

        return {
            'accuracy_with_tools': sum(r['passed'] for r in with_tools) / len(with_tools) if with_tools else 0,
            'accuracy_without_tools': sum(r['passed'] for r in without_tools) / len(without_tools) if without_tools else 0,
        }
```

### 4. 评估最佳实践

#### 4.1 A/B测试框架

```python
def ab_test_agents(agent_a, agent_b, test_tasks: List[Dict], num_runs: int = 3):
    """A/B测试比较两个Agent版本"""

    def run_experiment(agent, task):
        scores = []
        for _ in range(num_runs):
            result = agent.execute(task)
            scores.append(result['score'])
        return {
            'mean_score': sum(scores) / len(scores),
            'scores': scores,
            'variance': sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
        }

    results_a = [run_experiment(agent_a, task) for task in test_tasks]
    results_b = [run_experiment(agent_b, task) for task in test_tasks]

    # 统计显著性检验（简化版）
    avg_a = sum(r['mean_score'] for r in results_a) / len(results_a)
    avg_b = sum(r['mean_score'] for r in results_b) / len(results_b)

    return {
        'agent_a_avg': avg_a,
        'agent_b_avg': avg_b,
        'improvement': (avg_b - avg_a) / avg_a * 100 if avg_a > 0 else 0,
        'winner': 'B' if avg_b > avg_a else 'A'
    }
```

#### 4.2 持续评估流水线

```python
class ContinuousAgentEvaluation:
    """生产环境的持续评估系统"""

    def __init__(self, production_agent, shadow_agent=None):
        self.production_agent = production_agent
        self.shadow_agent = shadow_agent
        self.metrics_store = []

    def log_and_evaluate(self, task: Dict, response: Dict, user_feedback: int = None):
        """记录并评估每次交互"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task.get('type'),
            'response_length': len(response.get('content', '')),
            'token_usage': response.get('usage', {}).get('total_tokens', 0),
            'latency': response.get('latency', 0),
            'user_feedback': user_feedback  # 1-5分评分
        }

        self.metrics_store.append(metrics)

        # 实时告警
        if metrics['user_feedback'] and metrics['user_feedback'] < 3:
            self._alert_low_quality(metrics)

        return metrics

    def generate_report(self) -> str:
        """生成评估报告"""
        df = pd.DataFrame(self.metrics_store)

        return f"""
        # Agent评估报告

        ## 概览
        - 总交互数：{len(df)}
        - 平均用户评分：{df['user_feedback'].mean():.2f}
        - 平均Token消耗：{df['token_usage'].mean():.0f}
        - 平均延迟：{df['latency'].mean():.2f}s

        ## 按任务类型分析
        {df.groupby('task_type').agg({
            'user_feedback': 'mean',
            'token_usage': 'mean',
            'latency': 'mean'
        }).to_markdown()}
        """
```

## 应用场景

### 离线评估
- **版本迭代验证**：新版本Agent上线前的离线基准测试
- **回归测试**：确保更新未引入功能退化
- **对比研究**：不同架构或提示策略的效果对比

### 在线评估
- **用户反馈收集**：生产环境中实时收集用户评分
- **影子模式测试**：新Agent在影子模式下与生产Agent对比
- **金丝雀发布**：小流量验证后逐步放量

### 专项评估
- **安全红队**：评估Agent对对抗性输入的鲁棒性
- **合规检查**：评估输出是否符合特定领域规范
- **能力分级**：基于评估结果划分Agent能力等级

## 相关概念

| 概念 | 关联说明 |
|------|---------|
| [Agent架构](./Agent架构.md) | Agent的架构设计决定其评估维度 |
| [Prompt模式](./Prompt模式.md) | Prompt优化是提升评估分数的重要手段 |
| Tool-use Agent | 工具调用能力是Agent评估的核心指标之一 |
| HumanEval | 专门针对代码Agent的评估基准 |

## 延伸阅读

1. **AgentBench: Learning to Evaluate LLMs as Agents** - Liu et al., 2023
   - 论文链接：https://arxiv.org/abs/2308.03688
   - 多领域Agent评估基准的系统性研究

2. **GAIA: A General Assessment Tool for AI Assistants** - Meta AI
   - https://huggingface.co/gaia-benchmark
   - 通用AI助手评估数据集

3. **HumanEval for Agents** - OpenAI
   - https://openai.com/index/humaneval
   - 代码Agent评估的权威基准

4. **WebArena: A Realistic Web Environment for Building Autonomous Agents**
   - 论文链接：https://arxiv.org/abs/2307.13854
   - Web环境下Agent评估基准

5. **AgentOps** - Agent监控与评估平台
   - https://www.agentops.ai/
   - 生产环境Agent评估与监控工具
