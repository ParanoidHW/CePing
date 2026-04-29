# CePing 项目系统提示词

---

## 项目定位

CePing 是 **LLM Performance Evaluator**，用于评估大语言模型的性能表现。

### 核心能力

- **性能评估**：评估模型在不同硬件配置下的性能（MFU、QPS、TTFT、TPOT等）
- **策略搜索**：搜索最优并行策略（TP/PP/DP/EP/SP）
- **混布分析**：分析多模型混布场景的资源分配

### 架构层次

```
Application Layer (Evaluator, Optimizer, ColocateAnalyzer)
  ↓
Workload Layer (YAML配置，training/inference/diffusion/rl)
  ↓
Unified Modeling Layer (ShardedModule, bind机制)
  ↓
Analyzer Layer (UnifiedAnalyzer, MFU/QPS计算)
  ↓
Kernel Layer (可插拔Backend, functional API)
  ↓
Hardware Layer (Device, Cluster, NetworkTopology)
  ↓
Preset Layer (YAML配置，model/workload presets)
```

---

## 开发原则

### 必须遵循

- **需求支持和问题处理通过 subagent 执行**
- **分层解耦**：kernel 层修改不影响 analyzer/modeling/web
- **小步快跑**：每个特性单独 commit
- **测试先行**：新增特性必须有测试覆盖，存量测试必须全部通过
- **禁止参考 legacy 目录**
- **进度刷新到 session.log**：开发进度、关键决策、问题记录刷新到本地 session.log（不提交）

### 代码规范

- 使用 ruff 格式化，行长度 100 字符
- 使用类型注解
- 公开函数必须有 docstring
- commit 格式：`<type>(<scope>): <subject>`

---

## Skills

| Skill        | 位置                                   | 主要内容                           |
| ------------ | -------------------------------------- | ---------------------------------- |
| project      | `.agents/skills/project/SKILL.md`      | 设计-开发-验证流程编排             |
| architecture | `.agents/skills/architecture/SKILL.md` | 架构设计原则、解耦规范             |
| coder        | `.agents/skills/coder/SKILL.md`        | 代码风格、开发流程、测试规范       |
| kernel       | `.agents/skills/kernel/SKILL.md`       | Kernel API、cache-aware、3 backend |
| model        | `.agents/skills/model/SKILL.md`        | 模型建模、workload解耦、bind建模   |
| reviewer     | `.agents/skills/reviewer/SKILL.md`     | 架构符合性、泛化测试、边界测试     |

---

## Workflow

所有设计和开发必须遵循 project skill 的流程编排：

### 阶段1：设计

**由 subagent 发起，调用architecture skill**

- 分析需求合理性
- 参考外部实践
- 输出设计方案
- **特殊场景提示用户讨论和决策**

### 阶段2：开发

**由 subagent 发起，调用coder skill**

- 遵循 coder skill 规范
- 小步快跑
- 测试先行

### 阶段3：验证测试

**由 subagent 发起，调用reviewer skill**

- 架构符合性检视
- 新增代码测试
- 边界用例测试
- 输出到 review.log

### 迭代循环

```
开发 → 验证 → 发现问题 → 修复 → 重新验证 → ...
```

**必须由 subagent 发起调用，禁止直接启动全流程开发。**

---

## 特殊场景提示

以下场景需提示用户进行讨论和决策：

1. 需求不明确或存在歧义
2. 多个方案选择，需要权衡
3. 影响范围大，需要评估风险
4. 新架构设计，需要确认方案
5. 发现设计缺陷，需要讨论修复方案
6. 兼容性无法覆盖所有现有组件

---

## 架构设计要点

### 解耦原则

- **后端数据结构**：使用字典结构，新增类别自动出现
- **前端渲染**：使用 `Object.keys()` 自动发现字段
- **模型定义**：`_submodule_name` 属性 + 自动注册

### 核心机制

- **bind 机制**：`model.bind(ctx)` → `ModuleInstance`
- **kernel API**：所有层特征必须通过 kernel API 获取
- **workload 解耦**：模型结构和计算模式分离
- **Handler 机制**：模型类型显式分发（LLM/Diffusion/Vision）

---

## 工具命令速查

### 代码检查

```bash
ruff check llm_perf/modeling/*.py --select=F401,F841,E741
ruff check llm_perf/kernels/*.py --select=F401,F841,E741
```

### 测试运行

```bash
# 模型测试
python -m pytest tests/test_models.py -v -n 4

# 全量测试
python -m pytest tests/ -v -n 4

# 禁用分布式（调试）
python -m pytest tests/ -n 0
```

### Git 操作

```bash
git status
git diff --stat
git log --oneline -5
git add <files>
git commit -m "<message>"
git push origin main
```

### GitCode平台内容获取

如果碰到需要从[GitCode平台](gitcode.com)获取文档、代码，通过私钥从GitCode拉取对应文件/代码到临时目录下进行分析

---

*文档版本: 2.0*
*更新日期: 2026-04-28*
*说明: 通用系统提示词，长期适用*