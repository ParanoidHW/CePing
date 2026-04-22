"""Tests for Phase 1 core data structures."""

import pytest
from llm_perf.modeling import (
    ShardedTensor,
    ShardedParameter,
    ShardedModule,
    ParallelContext,
    SPType,
)


class TestShardedTensor:
    """Test ShardedTensor basic operations."""

    def test_tensor_creation(self):
        """Test creating a ShardedTensor."""
        tensor = ShardedTensor(
            shape=(4096, 4096),
            shardable={0: "tp"},
            dtype="fp16",
            name="test_weight",
        )

        assert tensor.shape == (4096, 4096)
        assert tensor.shardable == {0: "tp"}
        assert tensor.dtype == "fp16"
        assert tensor.name == "test_weight"
        assert tensor.ndim == 2
        assert tensor.numel() == 4096 * 4096

    def test_tensor_size(self):
        """Test tensor.size() method."""
        tensor = ShardedTensor(shape=(2, 3, 4))

        assert tensor.size() == (2, 3, 4)
        assert tensor.size(0) == 2
        assert tensor.size(1) == 3
        assert tensor.size(-1) == 4

    def test_get_physical_shape(self):
        """Test physical shape derivation."""
        tensor = ShardedTensor(
            shape=(4096, 4096),
            shardable={0: "tp"},
            dtype="fp16",
        )

        parallel_degrees = {"tp": 8}
        physical_shape = tensor.get_physical_shape(parallel_degrees)

        assert physical_shape == (512, 4096)

    def test_matmul_basic(self):
        """Test basic matmul operation."""
        input_tensor = ShardedTensor(
            shape=(1, 4096),
            shardable={},
            dtype="fp16",
            name="input",
        )

        weight_tensor = ShardedTensor(
            shape=(4096, 11008),
            shardable={1: "tp"},
            dtype="fp16",
            name="weight",
        )

        output = input_tensor @ weight_tensor

        assert output.shape == (1, 11008)
        assert 1 in output.shardable
        assert output.shardable[1] == "tp"

    def test_matmul_allreduce_case(self):
        """Test matmul that triggers AllReduce."""
        input_tensor = ShardedTensor(
            shape=(4096, 4096),
            shardable={0: "tp"},
            dtype="fp16",
        )

        weight_tensor = ShardedTensor(
            shape=(4096, 4096),
            shardable={},
            dtype="fp16",
        )

        output = input_tensor @ weight_tensor

        assert output.shape == (4096, 4096)
        assert output.shardable == {}

    def test_view_operation(self):
        """Test view/reshape operation."""
        tensor = ShardedTensor(
            shape=(1, 4096, 4096),
            shardable={1: "sp", 2: "tp"},
            dtype="fp16",
        )

        reshaped = tensor.view(1, 4096, 32, 128)

        assert reshaped.shape == (1, 4096, 32, 128)
        assert 1 in reshaped.shardable
        assert reshaped.shardable[1] == "sp"

    def test_transpose_operation(self):
        """Test transpose operation."""
        tensor = ShardedTensor(
            shape=(1, 32, 4096, 128),
            shardable={1: "tp", 2: "sp"},
            dtype="fp16",
        )

        transposed = tensor.transpose(1, 2)

        assert transposed.shape == (1, 4096, 32, 128)
        assert transposed.shardable == {1: "sp", 2: "tp"}

    def test_add_mul_operations(self):
        """Test element-wise operations."""
        tensor1 = ShardedTensor(
            shape=(4096, 4096),
            shardable={0: "tp"},
        )

        tensor2 = ShardedTensor(
            shape=(4096, 4096),
            shardable={0: "tp"},
        )

        add_result = tensor1 + tensor2
        mul_result = tensor1 * tensor2

        assert add_result.shape == (4096, 4096)
        assert mul_result.shape == (4096, 4096)
        assert add_result.shardable == {0: "tp"}
        assert mul_result.shardable == {0: "tp"}

    def test_to_dict(self):
        """Test to_dict serialization."""
        tensor = ShardedTensor(
            shape=(4096, 4096),
            shardable={0: "tp"},
            dtype="fp16",
            name="test",
        )

        result = tensor.to_dict()

        assert result["shape"] == (4096, 4096)
        assert result["shardable"] == {0: "tp"}
        assert result["dtype"] == "fp16"
        assert result["name"] == "test"
        assert result["numel"] == 4096 * 4096


class TestParallelContext:
    """Test ParallelContext."""

    def test_context_creation(self):
        """Test creating a ParallelContext."""
        ctx = ParallelContext(
            tp_degree=8,
            pp_degree=4,
            sp_degree=2,
            ep_degree=1,
            dp_degree=16,
            sp_type=SPType.ULYSSES,
            dtype="fp16",
        )

        assert ctx.tp_degree == 8
        assert ctx.pp_degree == 4
        assert ctx.sp_degree == 2
        assert ctx.ep_degree == 1
        assert ctx.dp_degree == 16
        assert ctx.sp_type == SPType.ULYSSES
        assert ctx.dtype == "fp16"

    def test_get_degree(self):
        """Test get_degree method."""
        ctx = ParallelContext(tp_degree=8, sp_degree=4)

        assert ctx.get_degree("tp") == 8
        assert ctx.get_degree("sp") == 4
        assert ctx.get_degree("pp") == 1

    def test_get_total_devices(self):
        """Test total device count."""
        ctx = ParallelContext(
            tp_degree=8,
            pp_degree=4,
            sp_degree=2,
            ep_degree=1,
            dp_degree=16,
        )

        total = ctx.get_total_devices()
        assert total == 8 * 4 * 2 * 1 * 16

    def test_to_dict(self):
        """Test to_dict serialization."""
        ctx = ParallelContext(tp_degree=8, dtype="fp16")

        result = ctx.to_dict()

        assert result["tp_degree"] == 8
        assert result["dtype"] == "fp16"
        assert "total_devices" in result


class TestShardedModule:
    """Test ShardedModule base class."""

    def test_module_creation(self):
        """Test creating a simple module."""

        class SimpleModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.weight = ShardedParameter(
                    shape=(4096, 4096),
                    shardable={0: "tp"},
                )

        module = SimpleModule()

        assert "weight" in module._weights
        assert module.params_count() == 4096 * 4096

    def test_params_count(self):
        """Test params_count method."""

        class TestModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.w1 = ShardedParameter(shape=(4096, 4096))
                self.w2 = ShardedParameter(shape=(4096, 11008))

        module = TestModule()

        assert module.params_count() == 4096 * 4096 + 4096 * 11008

    def test_params_count_breakdown(self):
        """Test params_count_breakdown method."""

        class TestModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.w1 = ShardedParameter(shape=(100, 100))
                self.w2 = ShardedParameter(shape=(100, 200))

        module = TestModule()
        breakdown = module.params_count_breakdown()

        assert breakdown["w1"] == 10000
        assert breakdown["w2"] == 20000

    def test_submodule_registration(self):
        """Test submodule auto-registration."""

        class SubModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.weight = ShardedParameter(shape=(100, 100))

        class ParentModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.sub = SubModule()

        module = ParentModule()

        assert "sub" in module._submodules
        weights = module.get_weights()
        assert "sub.weight" in weights

    def test_forward_not_implemented(self):
        """Test that forward raises NotImplementedError."""
        module = ShardedModule()

        with pytest.raises(NotImplementedError):
            module.forward()


class TestModuleInstance:
    """Test ModuleInstance."""

    def test_instance_creation(self):
        """Test creating a ModuleInstance."""

        class SimpleModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.weight = ShardedParameter(
                    shape=(4096, 4096),
                    shardable={0: "tp"},
                )

            def forward(self, x):
                return x @ self.weight

        module = SimpleModule()
        ctx = ParallelContext(tp_degree=8)

        instance = module.bind(ctx)

        assert instance.params_count_logical == 4096 * 4096
        assert instance.params_count_physical == (4096 // 8) * 4096

    def test_physical_shape(self):
        """Test physical shape in instance."""

        class TestModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.weight = ShardedParameter(
                    shape=(4096, 11008),
                    shardable={1: "tp"},
                )

        module = TestModule()
        ctx = ParallelContext(tp_degree=8)

        instance = module.bind(ctx)
        weight_inst = instance._weight_instances["weight"]

        assert weight_inst.physical_shape == (4096, 11008 // 8)

    def test_mode_forward_only(self):
        """Test forward-only mode."""

        class SimpleModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.w = ShardedParameter(shape=(100, 100))

        module = SimpleModule()
        ctx = ParallelContext()

        instance = module.bind(ctx, mode="forward")

        assert instance.mode == "forward"

    def test_to_dict(self):
        """Test to_dict serialization."""

        class SimpleModule(ShardedModule):
            def __init__(self):
                super().__init__()
                self.weight = ShardedParameter(
                    shape=(100, 100),
                    shardable={0: "tp"},
                )

        module = SimpleModule()
        module._name = "test_module"
        ctx = ParallelContext(tp_degree=2)

        instance = module.bind(ctx)
        result = instance.to_dict()

        assert "module_name" in result
        assert "params" in result
        assert result["params"]["logical"] == 10000
        assert result["params"]["physical"] == 5000


class TestOperationHistory:
    """Test operation history recording."""

    def test_matmul_history(self):
        """Test matmul operation history."""
        input_tensor = ShardedTensor(shape=(1, 4096))
        weight_tensor = ShardedTensor(shape=(4096, 11008))

        output = input_tensor @ weight_tensor

        assert len(output._op_history) == 1
        op = output._op_history[0]
        assert op.kernel_name == "linear"

    def test_view_history(self):
        """Test view operation history."""
        tensor = ShardedTensor(shape=(1, 4096, 4096))
        reshaped = tensor.view(1, 4096, 32, 128)

        assert len(reshaped._op_history) == 1

    def test_transpose_history(self):
        """Test transpose operation history."""
        tensor = ShardedTensor(shape=(1, 32, 4096, 128))
        transposed = tensor.transpose(1, 2)

        assert len(transposed._op_history) == 1

    def test_chain_operations(self):
        """Test chained operations."""
        input_tensor = ShardedTensor(shape=(1, 4096))
        w1 = ShardedTensor(shape=(4096, 11008))
        w2 = ShardedTensor(shape=(11008, 4096))

        hidden = input_tensor @ w1
        output = hidden @ w2

        assert len(output._op_history) == 2
