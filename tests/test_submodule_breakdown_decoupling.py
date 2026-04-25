"""测试子模块分解解耦架构.

验证：
1. 新增模型无需注册即可分解
2. _submodule_name 属性正确获取
3. 类名推断 fallback 正确工作
"""

import pytest
from llm_perf.modeling.module import ShardedModule, ShardedTensor, ShardedParameter
from llm_perf.modeling.layers import ShardedAttention, ShardedFFN, ShardedMoE, ShardedLinearAttention
from llm_perf.modeling.mla import ShardedMLA
from llm_perf.analyzer.unified import infer_submodule_name_from_class, UnifiedAnalyzer
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig


class ShardedNewAttention(ShardedModule):
    """新增的 Attention 类（无需注册）."""

    _submodule_name = "new_attention"

    def __init__(self, hidden_size: int, num_heads: int, dtype: str = "fp16"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.weight = ShardedParameter(
            shape=(hidden_size, hidden_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="weight",
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        output = hidden @ self.weight
        self._activations["output"] = output
        return output


class ShardedNewFFN(ShardedModule):
    """新增的 FFN 类（测试 fallback：从类名推断）."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: str = "fp16"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.weight = ShardedParameter(
            shape=(hidden_size, intermediate_size),
            shardable={1: "tp"},
            dtype=dtype,
            name="weight",
        )

    def forward(self, hidden: ShardedTensor) -> ShardedTensor:
        output = hidden @ self.weight
        self._activations["output"] = output
        return output


class TestSubmoduleBreakdownDecoupling:
    """测试子模块分解解耦."""

    def test_submodule_name_from_attribute(self):
        """测试从 _submodule_name 获取名称."""
        assert ShardedAttention._submodule_name == "attention"
        assert ShardedLinearAttention._submodule_name == "linear_attention"
        assert ShardedMLA._submodule_name == "mla"
        assert ShardedFFN._submodule_name == "ffn"
        assert ShardedMoE._submodule_name == "moe"

    def test_submodule_name_from_class_name(self):
        """测试从类名推断名称（fallback）."""
        assert infer_submodule_name_from_class("ShardedAttention") == "attention"
        assert infer_submodule_name_from_class("ShardedLinearAttention") == "linear_attention"
        assert infer_submodule_name_from_class("ShardedMLA") == "mla"
        assert infer_submodule_name_from_class("ShardedFFN") == "ffn"
        assert infer_submodule_name_from_class("ShardedMoE") == "moe"
        assert infer_submodule_name_from_class("ShardedEmbedding") == "embedding"
        assert infer_submodule_name_from_class("ShardedLMHead") == "lm_head"
        assert infer_submodule_name_from_class("ShardedTransformerBlock") == "transformer_block"
        assert infer_submodule_name_from_class("ShardedRMSNorm") == "rms_norm"
        assert infer_submodule_name_from_class("ShardedLayerNorm") == "layer_norm"

    def test_new_model_with_submodule_name(self):
        """测试新模型类（有 _submodule_name）自动识别."""
        assert ShardedNewAttention._submodule_name == "new_attention"

        inferred = infer_submodule_name_from_class("ShardedNewAttention")
        assert inferred == "new_attention"

    def test_new_model_without_submodule_name(self):
        """测试新模型类（无 _submodule_name）使用类名推断."""
        inferred = infer_submodule_name_from_class("ShardedNewFFN")
        assert inferred == "new_ffn"

    def test_infer_submodule_type_priority(self):
        """测试 _infer_submodule_type 优先级：_submodule_name > 类名推断."""
        device = Device.from_preset("H100-SXM-80GB")
        cluster = Cluster(
            devices=[device],
            topology=NetworkTopology.create_2tier_simple(
                intra_node_bw_gbps=400.0,
                inter_node_bw_gbps=100.0,
            ),
            devices_per_node=8,
        )
        strategy = StrategyConfig(tp_degree=8, dp_degree=1, pp_degree=1)

        new_attn = ShardedNewAttention(hidden_size=4096, num_heads=32)

        class MockModuleInstance:
            def __init__(self, module):
                self.module = module

        mock_inst = MockModuleInstance(new_attn)

        analyzer = UnifiedAnalyzer(new_attn, device, cluster, strategy)
        inferred_type = analyzer._infer_submodule_type("test_attn", mock_inst)

        assert inferred_type == "new_attention"

    def test_camel_to_snake_conversion(self):
        """测试 CamelCase 到 snake_case 转换."""
        assert infer_submodule_name_from_class("ShardedQwen3_5MoEBlock") == "qwen3_5_moe"
        assert infer_submodule_name_from_class("ShardedWanDiTBlock") == "wan_dit"
        assert infer_submodule_name_from_class("ShardedViTBlock") == "vit"