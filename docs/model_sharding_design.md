# 模型切分架构设计方案

## 问题背景

当前架构问题：
1. Model 层 `build_layers()` 使用原始（未切分）的 `num_attention_heads`、`intermediate_size`
2. TP 切分后，实际每个 GPU 处理 `heads / tp_degree`
3. Analyzer 层有自己的计算逻辑，但与 Model 层的 `layer.flops` 不一致
4. 不同 Backend（Profiling/Microarch）需要知道切分后的具体 shape

## 核心矛盾

| 需求 | 当前方案的问题 |
|------|----------------|
| TheoryBackend | 只需 FLOPs，当前 Model 层 FLOPs 未切分 |
| ProfilingBackend | 需要切分后 shape 查找实测数据 |
| MicroarchBackend | 需要切分后 shape 和切分方式建模 |
| 通信建模 | 切分后前后算子 shape 变化，通信开销需要统一建模 |

## 设计原则

1. **Model 层职责**：定义模型结构，能生成"切分视图"
2. **Analyzer 层职责**：统一计算切分后的计算时间、通信时间
3. **Kernel 层职责**：提供"切分后 kernel"的评估接口
4. **分层清晰**：Model 不直接依赖 Strategy，但提供切分生成机制

## 方案：Model + Strategy → ShardedModel

### 1. Model 层扩展

```python
# llm_perf/models/base.py

class BaseModel(ABC):
    """模型基类，支持生成切分视图."""
    
    def build_layers(self) -> List[LayerConfig]:
        """构建原始层（未切分）."""
        pass
    
    def build_sharded_layers(
        self,
        strategy: StrategyConfig,
    ) -> List[LayerConfig]:
        """构建切分后的层视图.
        
        Args:
            strategy: 并行切分策略
            
        Returns:
            切分后的 LayerConfig 列表，每个 LayerConfig 代表
            单个 GPU 上的实际计算负载
        """
        pass
    
    def get_layer_sharding_info(self) -> Dict[str, ShardingInfo]:
        """获取各层的切分元信息.
        
        Returns:
            各层的切分信息，用于 KernelBackend 评估切分算子
        """
        pass
```

### 2. ShardingInfo 数据结构

```python
# llm_perf/models/sharding.py

@dataclass
class ShardingInfo:
    """层的切分信息."""
    
    # 切分维度信息
    shardable_dims: Dict[str, ShardableDim]  # 可切分维度
    
    # 通信信息
    comm_pattern: Optional[CommPattern] = None  # 该层需要的通信
    
    # 约束
    min_shard_size: int = 1  # 最小切分单元（如 head 最小 1）
    
@dataclass
class ShardableDim:
    """可切分维度."""
    name: str  # 维度名（如 "heads", "intermediate_size", "seq_len"）
    original_size: int  # 原始大小
    parallel_types: List[ParallelType]  # 支持的切分方式（TP/SP/EP等）
    
@dataclass  
class CommPattern:
    """通信模式."""
    comm_type: str  # allreduce, allgather, alltoall, reducescatter
    position: str  # "before" / "after" / "between"
    data_shape: Tuple[int, ...]  # 通信数据 shape
```

### 3. Kernel 层扩展

```python
# llm_perf/kernels/backend/base.py

class KernelBackend(ABC):
    """Kernel Backend 基类，支持切分评估."""
    
    def estimate_sharded_compute_time(
        self,
        original_result: KernelResult,
        sharding_info: ShardingInfo,
        strategy: StrategyConfig,
    ) -> ShardedKernelResult:
        """评估切分后的算子性能.
        
        Args:
            original_result: 原始 kernel 结果
            sharding_info: 切分信息
            strategy: 切分策略
            
        Returns:
            切分后的 kernel 结果，包含：
            - sharded_shape: 切分后的 shape
            - sharded_flops: 切分后的 FLOPs
            - sharded_time: 切分后的执行时间
        """
        pass
```

### 4. Analyzer 层统一计算

```python
# llm_perf/analyzer/training.py

class TrainingAnalyzer:
    def analyze(self, batch_size, seq_len):
        # 获取切分后的模型视图
        sharded_layers = self.model.build_sharded_layers(self.strategy)
        
        # 计算切分后的计算时间
        compute_time = self._estimate_sharded_compute_time(
            sharded_layers, batch_size, seq_len
        )
        
        # 计算通信时间（使用 CommPattern）
        comm_time = self._estimate_communication_time(
            sharded_layers, batch_size, seq_len
        )
        
        return result
    
    def _estimate_sharded_compute_time(self, sharded_layers, batch_size, seq_len):
        total_time = 0.0
        for layer in sharded_layers:
            # 使用 KernelBackend 评估切分后算子
            kernel_result = self.backend.estimate_sharded_compute_time(
                layer.original_kernel_result,
                layer.sharding_info,
                self.strategy,
            )
            total_time += kernel_result.sharded_time
        return total_time
```

### 5. 具体实现示例

```python
# llm_perf/models/llama.py

class LlamaModel(BaseModel):
    def build_layers(self) -> List[LayerConfig]:
        """构建原始层."""
        # 使用 cfg.num_attention_heads（原始值）
        return self._build_layers_with_heads(
            q_heads=cfg.num_attention_heads,
            kv_heads=cfg.num_key_value_heads,
            intermediate_size=cfg.intermediate_size,
        )
    
    def build_sharded_layers(self, strategy) -> List[LayerConfig]:
        """构建切分后的层."""
        tp = strategy.tp_degree
        sp = strategy.sp_degree
        
        # 计算切分后的值
        q_heads_per_gpu = cfg.num_attention_heads // tp
        kv_heads_per_gpu = cfg.num_key_value_heads // tp
        intermediate_per_gpu = cfg.intermediate_size // tp
        seq_per_gpu = cfg.max_seq_len // sp
        
        # 使用切分后的值构建层
        return self._build_layers_with_heads(
            q_heads=q_heads_per_gpu,
            kv_heads=kv_heads_per_gpu,
            intermediate_size=intermediate_per_gpu,
            seq_len=seq_per_gpu,
        )
    
    def get_layer_sharding_info(self) -> Dict[str, ShardingInfo]:
        """获取切分信息."""
        return {
            "attention": ShardingInfo(
                shardable_dims={
                    "heads": ShardableDim(
                        name="heads",
                        original_size=cfg.num_attention_heads,
                        parallel_types=[ParallelType.TENSOR],
                    ),
                    "kv_heads": ShardableDim(
                        name="kv_heads",
                        original_size=cfg.num_key_value_heads,
                        parallel_types=[ParallelType.TENSOR],
                    ),
                },
                comm_pattern=CommPattern(
                    comm_type="allreduce",
                    position="after",  # O projection 后
                    data_shape=(batch, seq, hidden),
                ),
            ),
            "ffn": ShardingInfo(
                shardable_dims={
                    "intermediate_size": ShardableDim(
                        name="intermediate_size",
                        original_size=cfg.intermediate_size,
                        parallel_types=[ParallelType.TENSOR],
                    ),
                },
                comm_pattern=CommPattern(
                    comm_type="allreduce",
                    position="after",  # Down projection 后
                    data_shape=(batch, seq, hidden),
                ),
            ),
        }
```

### 6. Backend 实现

```python
# llm_perf/kernels/backend/theory.py

class TheoryBackend(KernelBackend):
    def estimate_sharded_compute_time(self, original_result, sharding_info, strategy):
        # FLOPs 按切分比例减少
        tp = strategy.tp_degree
        sharded_flops = original_result.flops // tp
        
        # Shape 按切分比例调整
        sharded_shape = self._compute_sharded_shape(
            original_result.output,
            sharding_info,
            strategy,
        )
        
        return ShardedKernelResult(
            sharded_shape=sharded_shape,
            sharded_flops=sharded_flops,
            sharded_time=self._roofline_time(sharded_flops, sharded_shape),
        )

# llm_perf/kernels/backend/profiling.py

class ProfilingBackend(KernelBackend):
    def estimate_sharded_compute_time(self, original_result, sharding_info, strategy):
        # 用切分后的 shape 查找实测数据
        sharded_shape = self._compute_sharded_shape(
            original_result.output,
            sharding_info,
            strategy,
        )
        
        # 查找实测数据
        measured_time = self._lookup_or_interpolate(
            kernel_name=original_result.name,
            shape=sharded_shape,
            dtype=original_result.dtype,
        )
        
        return ShardedKernelResult(
            sharded_shape=sharded_shape,
            sharded_flops=...,  # 可从实测推断
            sharded_time=measured_time,
        )

# llm_perf/kernels/backend/microarch.py

class MicroarchBackend(KernelBackend):
    def estimate_sharded_compute_time(self, original_result, sharding_info, strategy):
        # 用切分后的 shape 和切分方式建模
        sharded_shape = self._compute_sharded_shape(...)
        
        # 基于微架构特性建模
        # 考虑 SM 数量、warp 效率、内存层级等
        modeled_time = self._model_from_microarch(
            sharded_shape,
            sharding_info,
            strategy,
        )
        
        return ShardedKernelResult(...)
```

## 实现路径

### Phase 1: Model 层扩展
1. BaseModel 添加 `build_sharded_layers()` 抽象方法
2. LayerConfig 添加 `sharding_info` 字段
3. 创建 ShardingInfo、ShardableDim、CommPattern 数据结构
4. LlamaModel 实现 `build_sharded_layers()`

### Phase 2: Kernel 层扩展
1. KernelBackend 添加 `estimate_sharded_compute_time()` 方法
2. TheoryBackend 实现切分 FLOPs 计算
3. ProfilingBackend 实现切分 shape 查找
4. MicroarchBackend 实现切分建模（预留框架）

### Phase 3: Analyzer 层重构
1. TrainingAnalyzer 使用 `build_sharded_layers()`
2. 统一计算逻辑，移除重复代码
3. InferenceAnalyzer 同步重构

### Phase 4: 测试验证
1. 单 GPU (tp=1) 结果应与原结果一致
2. 多 GPU (tp=8) 结果应正确
3. breakdown FLOPs 应反映切分后的值

## 优点

1. **分层清晰**：Model 提供"生成切分视图"的能力，不直接依赖 Strategy
2. **统一建模**：计算和通信在 Analyzer 层统一建模
3. **Backend 支持完整**：各 Backend 能获得切分后 shape，支持 Profiling/Microarch
4. **向后兼容**：原 `build_layers()` 保持不变，新增 `build_sharded_layers()`

## 待讨论问题

1. **MoE 模型切分**：EP 切分的专家如何建模？
2. **SP/CP 切分**：序列切分如何处理？
3. **Pipeline 切分**：PP 切分后每 stage 的层如何建模？
4. **组合切分**：TP+PP+DP 组合如何处理？