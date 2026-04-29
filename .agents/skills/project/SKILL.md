---
name: project
description: 项目开发流程整合，按照"设计-开发-验证测试"编排，支持渐进式披露和交替迭代
---

# 项目开发流程

本文档整合项目所有 skill，按照"设计-开发-验证测试"流程编排开发任务。

---

## §1 项目概述

### 1.1 项目定位

CePing 是 LLM Performance Evaluator，用于评估大语言模型的性能表现。

### 1.2 核心能力

- **性能评估**：评估模型在不同硬件配置下的性能（MFU、QPS、TTFT、TPOT等）
- **策略搜索**：搜索最优并行策略（TP/PP/DP/EP/SP）
- **混布分析**：分析多模型混布场景的资源分配

### 1.3 架构层次

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

## §2 开发流程（常规）

按照"设计-开发-验证测试"三阶段编排，每个阶段都由 subagent 发起调用对应 skill 执行。

### 强制要求

| 约束 | 说明 |
|------|------|
| 每阶段由 subagent 发起 | **禁止直接运行 pytest/ruff**，由 subagent 发起调用 reviewer skill 执行验证 |
| 渐进披露 | 先披露需求背景 → 设计概览 → 详细实现 → 测试方案 |
| 交替迭代 | 开发和验证交替进行，不是一步到位 |
| 遵循用户检视要求 | 用户要求"完成全部N步后再检视"时，不得中途检视 |

### 禁止事项

| 禁止 | 说明 |
|------|------|
| 直接运行测试 | ❌ 直接运行 pytest/ruff，违反 Workflow 规定 |
| 跨步检视 | ❌ 用户要求"完成全部5步后再检视"时，中途检视 |
| 非 subagent 发起 | ❌ 直接启动全流程开发 |

### 阶段1：设计

**由 subagent 发起，调用architecture skill**

#### 1.1 需求分析合理性

- 新特性需先分析合理性
- 从外部公开实践中获取数据、信息（paper、官方代码、白皮书）
- 参考相关方案

#### 1.2 审视现有组件兼容性

- 设计需审视所有注册过的现有模型、kernel、硬件如何支持新特性
- 避免只针对目标模型定制化设计

#### 1.3 输出设计方案

设计方案格式：
1. 问题分析：详细描述问题和根因
2. 设计方案：给出修复/实现方案，包括代码示例
3. 实现步骤：分步骤说明如何实现
4. 测试方案：说明如何验证
5. 影响评估：改动的影响范围

#### 1.4 特殊场景提示

以下场景需提示用户进行讨论和决策：
- 需求不明确或存在歧义
- 多个方案选择，需要权衡
- 影响范围大，需要评估风险
- 新架构设计，需要确认方案
- 发现设计缺陷，需要讨论修复方案
- 需要审视现有组件兼容性但无法覆盖所有场景

### 阶段2：开发

**由 subagent 发起，调用coder skill**

#### 2.1 遵循 coder skill 规范

- 代码风格：ruff 格式化，行长度 100 字符
- 类型安全：使用类型注解
- 公开函数必须有 docstring
- 测试先行：新增特性必须有测试用例覆盖
- 存量测试必须全部通过

#### 2.2 小步快跑

- 每个特性完成后立即提交一个 commit
- 避免一次性提交大量改动
- commit message 格式：`<type>(<scope>): <subject>`

#### 2.3 特殊场景调用

- 新增 kernel 评估：调用 kernel skill
- 新增模型评估：调用 model skill

### 阶段3：验证测试

**由 subagent 发起，调用reviewer skill**

#### 3.1 架构符合性检视

根据 architecture skill 设计原则检视：
- 模块职责是否清晰
- 层次是否分明，避免跨层调用
- 是否避免循环依赖
- 是否遵循解耦原则

#### 3.2 新增代码测试

- 优先随机构造不同规格数据进行泛化测试
- 重点设计边界用例（TP=1、batch_size=1、seq_len=1等）
- 确保功能真正 ok

#### 3.3 边界用例测试

边界测试示例：
- TP/PP 切分边界：TP=1（无切分）、TP=world_size（最大切分）
- 形状边界：batch_size=1、seq_len=1、hidden_size=1024
- 数值边界：极小值、极大值、负值

#### 3.4 输出到 review.log

将检视意见和打分输出到 `review.log`，格式：
- 代码质量得分
- 架构符合性检查
- 测试结果（基础用例、泛化测试、边界用例）
- 结论：检视完成或问题清单

### 迭代循环

开发和验证测试是交替迭代的过程：

```
开发 → 验证 → 发现问题 → 修复 → 重新验证 → ...
```

不是一步到位，而是：
1. 完成特性开发
2. reviewer 检视并测试
3. 发现问题反馈到 review.log
4. coder 修复问题
5. reviewer 重新验证
6. 循环直到检视完成

---

## §3 特殊场景提示

### 需要提示用户讨论和决策的场景

1. **需求不明确**
   - 用户描述模糊，无法确定具体实现方式
   - 提示：请明确 XXX 的具体要求

2. **多个方案选择**
   - 存在多个可行方案，各有优劣
   - 提示：方案A优点是...，方案B优点是...，请选择

3. **影响范围大**
   - 改动可能影响多个模块或现有功能
   - 提示：此改动影响范围包括...，请评估风险

4. **新架构设计**
   - 需要引入新的架构模式或模块
   - 提示：拟引入新架构...，请确认方案

5. **设计缺陷发现**
   - reviewer 发现设计层面的问题
   - 提示：发现设计缺陷...，建议修复方案是...

6. **兼容性无法覆盖**
   - 新特性无法保证所有现有组件兼容
   - 提示：新特性对...组件可能有影响，请决策


### 设计问题记录

识别到以往设计方案存在比较大的漏洞时，需要固化为经验增加到architecture skill中作为指导和约束。

---

## §4 Skill 调用规范

### 必须由 subagent 发起

所有 skill 都需要由 subagent 发起调用，禁止直接启动全流程开发。

#### 正确调用方式

```
# 设计阶段
Task(subagent_type="architecture", prompt="分析XXX需求，设计实现方案")

# 开发阶段
Task(subagent_type="coder", prompt="根据设计方案实现XXX")

# 验证阶段
Task(subagent_type="reviewer", prompt="检视XXX代码，进行泛化测试")

# 特殊场景
Task(subagent_type="kernel", prompt="新增XXX kernel评估支持")
Task(subagent_type="model", prompt="新增XXX模型评估支持")
```

#### 调用时机

| Skill | 调用时机 | 职责 |
|-------|----------|------|
| architecture | 需求分析、方案设计 | 设计原则审视、输出设计方案 |
| coder | 具体开发实施 | 代码实现、测试覆盖、提交 |
| kernel | 新增 kernel 评估 | Kernel API + cache-aware + 3 backend |
| model | 新增模型评估 | 模型建模 + workload解耦 + bind建模 |
| reviewer | 代码检视、测试验证 | 架构符合性、泛化测试、边界测试 |

---

## §5 渐进式披露

### 设计阶段渐进披露

1. **先披露需求背景**
   - 简要说明需求来源、目标
   
2. **再披露设计方案概览**
   - 核心设计思路、主要模块
   
3. **最后披露详细实现方案**
   - 具体代码结构、接口定义、测试方案

### 开发阶段渐进披露

1. **先披露核心功能**
   - 主要功能实现
   
2. **再披露辅助功能**
   - 边界处理、错误处理
   
3. **最后披露测试验证**
   - 测试用例、验证结果

---

## §6 检查清单

### 设计阶段

- [ ] 分析需求合理性
- [ ] 参考外部实践（paper、官方代码）
- [ ] 审视现有组件兼容性
- [ ] 输出设计方案
- [ ] 特殊场景提示用户

### 开发阶段

- [ ] subagent 发起调用
- [ ] 遵循 coder skill 规范
- [ ] 小步快跑，每个特性一个 commit
- [ ] 新增特性测试覆盖
- [ ] 存量测试全部通过

### 验证阶段

- [ ] subagent 发起调用
- [ ] 架构符合性检视
- [ ] 新增代码测试
- [ ] 边界用例测试
- [ ] 输出到 review.log
- [ ] 修复问题后重新验证

---

## §7 架构设计要点

### 7.1 解耦原则

参考 `docs/architecture_decoupling.md`：

- **后端数据结构原则**：使用字典结构，新增类别自动出现
- **前端渲染原则**：使用 `Object.keys()` 自动发现字段
- **模型定义原则**：`_submodule_name` 属性 + 自动注册

### 7.2 分层结构

参考 `docs/architecture.md`：

- Application Layer → Workload Layer → Unified Modeling Layer
- Analyzer Layer → Kernel Layer → Hardware Layer → Preset Layer

### 7.3 核心机制

- **bind 机制**：`model.bind(ctx)` → `ModuleInstance`
- **kernel API**：所有层特征必须通过 kernel API 获取
- **workload 解耦**：模型结构和计算模式分离

---

## §8 相关 Skill 参考

| Skill | 位置 | 主要内容 |
|-------|------|----------|
| architecture | `.agents/skills/architecture/SKILL.md` | 设计原则、架构规范、数据结构设计 |
| coder | `.agents/skills/coder/SKILL.md` | 代码风格、开发流程、测试规范 |
| kernel | `.agents/skills/kernel/SKILL.md` | Kernel API、cache-aware、3 backend |
| model | `.agents/skills/model/SKILL.md` | 模型建模、workload解耦、bind建模 |
| reviewer | `.agents/skills/reviewer/SKILL.md` | 代码检视、泛化测试、边界测试 |

---

## §9 工具命令速查

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

---

*文档版本: 1.0*
*创建日期: 2026-04-28*
*整合自: architecture + coder + kernel + model + reviewer skills*