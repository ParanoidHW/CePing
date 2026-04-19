"""Breakdown classes for performance analysis."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class KernelBreakdown:
    """Breakdown by kernel type."""

    name: str
    time_sec: float
    flops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "time_sec": self.time_sec,
            "time_ms": self.time_sec * 1000,
            "flops": self.flops,
        }


@dataclass
class LayerBreakdown:
    """Breakdown by layer."""

    name: str
    kernels: List[KernelBreakdown]
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_time_sec": self.total_time,
            "kernels": [k.to_dict() for k in self.kernels],
        }


@dataclass
class ModuleBreakdown:
    """分层模块分解结果.

    支持计算、通信、内存开销的详细分解，帮助理解瓶颈。

    Attributes:
        name: 模块名称
        module_type: 模块类型 (embedding, transformer_block, attention, ffn, rms_norm, lm_head等)
        compute_time_sec: 计算时间（秒，每卡）
        compute_flops: 计算量（每卡）
        memory_activations_gb: 激活内存（GB，每卡）
        communication_bytes: 通信量（字节，每卡）
        sub_modules: 子模块分解列表
    """

    name: str
    module_type: str
    compute_time_sec: float = 0.0
    compute_flops: int = 0
    memory_activations_gb: float = 0.0
    communication_bytes: int = 0
    sub_modules: List["ModuleBreakdown"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，包含计算、通信、内存分解."""
        return {
            "name": self.name,
            "module_type": self.module_type,
            "compute": {
                "time_sec": self.compute_time_sec,
                "time_ms": self.compute_time_sec * 1000,
                "flops": self.compute_flops,
                "flops_gflops": self.compute_flops / 1e9 if self.compute_flops else 0,
            },
            "memory": {
                "activations_gb": self.memory_activations_gb,
                "activations_mb": self.memory_activations_gb * 1024,
            },
            "communication": {
                "total_bytes": self.communication_bytes,
                "total_gb": self.communication_bytes / 1e9 if self.communication_bytes else 0,
                "total_mb": self.communication_bytes / 1e6 if self.communication_bytes else 0,
            },
            "sub_modules": [sm.to_dict() for sm in self.sub_modules],
        }

    def add_sub_module(self, sub: "ModuleBreakdown"):
        """添加子模块."""
        self.sub_modules.append(sub)

    def aggregate_from_subs(self):
        """从子模块聚合指标."""
        if not self.sub_modules:
            return

        self.compute_time_sec = sum(sm.compute_time_sec for sm in self.sub_modules)
        self.compute_flops = sum(sm.compute_flops for sm in self.sub_modules)
        self.memory_activations_gb = sum(sm.memory_activations_gb for sm in self.sub_modules)
        self.communication_bytes = sum(sm.communication_bytes for sm in self.sub_modules)


@dataclass
class BreakdownSummary:
    """分解结果汇总."""

    total_compute_time_sec: float = 0.0
    total_memory_gb: float = 0.0
    total_communication_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_compute_time_sec": self.total_compute_time_sec,
            "total_memory_gb": self.total_memory_gb,
            "total_communication_gb": self.total_communication_gb,
        }


def generate_module_breakdown(
    phases: List[Any],
    workload_type: str,
    global_comm_bytes: int = 0,
) -> Dict[str, Any]:
    """生成模块分解结果.

    Args:
        phases: PhaseResult列表
        workload_type: workload类型
        global_comm_bytes: 全局通信量（避免重复累计）

    Returns:
        Dict containing:
        - by_module: 模块级别分解
        - by_block_type: 模块类型级别分解
        - summary: 总体汇总（使用全局通信量）
    """
    by_module: Dict[str, ModuleBreakdown] = {}
    by_block_type: Dict[str, ModuleBreakdown] = {}

    for phase in phases:
        for sm in phase.submodules:
            module_key = sm.name

            if module_key not in by_module:
                by_module[module_key] = ModuleBreakdown(
                    name=module_key,
                    module_type=sm.submodule_type,
                    compute_time_sec=0.0,
                    compute_flops=0,
                    memory_activations_gb=0.0,
                    communication_bytes=0,
                )

            by_module[module_key].compute_time_sec += sm.time_sec
            by_module[module_key].compute_flops += sm.flops
            by_module[module_key].memory_activations_gb += sm.memory_gb
            by_module[module_key].communication_bytes += sm.communication_bytes

            block_type = sm.submodule_type
            if block_type not in by_block_type:
                by_block_type[block_type] = ModuleBreakdown(
                    name=block_type,
                    module_type=block_type,
                    compute_time_sec=0.0,
                    compute_flops=0,
                    memory_activations_gb=0.0,
                    communication_bytes=0,
                )

            by_block_type[block_type].compute_time_sec += sm.time_sec
            by_block_type[block_type].compute_flops += sm.flops
            by_block_type[block_type].memory_activations_gb += sm.memory_gb
            by_block_type[block_type].communication_bytes += sm.communication_bytes

    summary = BreakdownSummary(
        total_compute_time_sec=sum(m.compute_time_sec for m in by_module.values()),
        total_memory_gb=sum(m.memory_activations_gb for m in by_module.values()),
        total_communication_gb=global_comm_bytes / 1e9,
    )

    return {
        "by_module": {k: v.to_dict() for k, v in by_module.items()},
        "by_block_type": {k: v.to_dict() for k, v in by_block_type.items()},
        "summary": summary.to_dict(),
    }
