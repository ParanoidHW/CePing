# CePing 项目上下文

---

## Goal

修复 HunyuanImage 3.0 diffusion 激活内存计算问题并开发 cache-aware kernel 评估模块。

---

## Constraints & Preferences

- 所有设计和开发通过 subagent 执行
- 分层解耦：kernel 层修改不影响 analyzer/modeling/web
- 小步快跑，每个特性单独 commit
- 新增特性必须有测试覆盖，存量测试必须全部通过
- 遵循 architecture skill 和 coder skill

---

## Progress

### Done

- 修复激活内存计算（45GB → 0.57GB）：timestep shape 错误 + forward 静默失败
- 引入 Handler 机制：`llm_perf/analyzer/handlers/` (LLM/Diffusion/Vision 三类处理器)
- 新增 WorkloadType.DIFFUSION：显式类型而非 inference 子类
- 精简测试用例 47%：删除低价值测试，保留核心覆盖
- 新增 xlsx 导出功能：通信分解、子模块详情、配置信息 sheet
- 创建 cache-aware kernel 设计文档：`docs/cache_aware_kernel_design.md`（策略1唯一正确模型）
- 实现 cache-aware kernel 模块：`llm_perf/kernels/backend/cache_aware/` (linear/attention)
- 创建 kernel skill：`.agents/skills/kernel/SKILL.md`（API + cache-aware + 3 backend）
- 创建 model skill：`.agents/skills/model/SKILL.md`（解耦、依赖 kernel、bind 建模）
- 创建 reviewer skill：`.agents/skills/reviewer/SKILL.md`（架构符合性 + 边界测试）
- 创建 project skill：`.agents/skills/project/SKILL.md`（设计-开发-验证流程编排）

### In Progress

- (none)

### Blocked

- (none)

---

## Key Decisions

- **矩阵乘法 Tiling 策略**：采用策略1（激活加载一次，权重重复加载 num_tiles_M 次）
- **Memory/Compute Bound 判断**：时间对比 `max(compute_time, memory_time)`，不用 ridge_point
- **3 Backend 要求**：趋势一致性而非结果一致性（大规格 > 小规格，近似线性）
- **Backend 命名**：Microarch（而非 Cache-aware）
- **开发流程编排**：设计 → 开发 → 验证测试，必须由 subagent 发起

---

## Next Steps

- P1: 添加更多 kernel 的 cache-aware 计算（conv2d, norm）
- P2: 完善 analyzer 层使用 cache-aware 结果

---

## Critical Context

- 激活内存 45GB 根因：timestep shape `(batch, 256)` 与模型期望 `(batch,)` 不匹配，forward 静默失败
- 198 倍 TTFT 差异原因：bs=1 memory bound (21.9μs) vs bs=200 compute bound (3474μs)
- 策略1 核心公式：`weight_bytes_actual = K × N × num_tiles_M`
- 3 backend：Theory Roofline、Microarch、Profiling（趋势一致，结果不可能一致）

---

## Relevant Files

- `llm_perf/analyzer/handlers/` - Handler 机制（显式 model type dispatch）
- `llm_perf/analyzer/unified.py` - 替换静默失败为明确报错
- `llm_perf/kernels/backend/cache_aware/` - Cache-aware 计算模块
- `docs/cache_aware_kernel_design.md` - Kernel 设计文档（策略1）
- `.agents/skills/project/SKILL.md` - 项目开发流程（设计-开发-验证）
- `.agents/skills/kernel/SKILL.md` - Kernel 开发规范（611 行）
- `.agents/skills/model/SKILL.md` - Model 开发规范（594 行）
- `.agents/skills/reviewer/SKILL.md` - Reviewer 规范（245 行）
- `tests/test_diffusion_activation.py` - Diffusion 激活内存测试
- `tests/test_cache_aware_linear.py` - Cache-aware linear 测试

---

## Skills

| Skill | 位置 | 主要内容 |
|-------|------|----------|
| project | `.agents/skills/project/SKILL.md` | 设计-开发-验证流程编排 |
| architecture | `.agents/skills/architecture/SKILL.md` | 架构设计原则、解耦规范 |
| coder | `.agents/skills/coder/SKILL.md` | 代码风格、开发流程、测试规范 |
| kernel | `.agents/skills/kernel/SKILL.md` | Kernel API、cache-aware、3 backend |
| model | `.agents/skills/model/SKILL.md` | 模型建模、workload解耦、bind建模 |
| reviewer | `.agents/skills/reviewer/SKILL.md` | 架构符合性、泛化测试、边界测试 |

---

## Workflow

所有设计和开发必须遵循 project skill 的流程编排：

```
阶段1：设计（architecture subagent）
  → 需求分析合理性
  → 参考外部实践
  → 输出设计方案
  → 特殊场景提示用户

阶段2：开发（coder subagent）
  → 遵循 coder skill 规范
  → 小步快跑
  → 测试先行

阶段3：验证测试（reviewer subagent）
  → 架构符合性检视
  → 新增代码测试
  → 边界用例测试
  → 输出到 review.log

迭代循环：开发 → 验证 → 修复 → 重新验证 → ...
```

**必须由 subagent 发起调用，禁止直接启动全流程开发。**

---

*文档版本: 1.0*
*创建日期: 2026-04-28*