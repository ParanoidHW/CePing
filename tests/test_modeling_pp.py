"""Tests for Pipeline Parallelism Strategy and Model.

Phase 6 tests for PP framework:
- PPStrategy creation and bubble ratio calculation
- PPSchedule generation and visualization
- PPModel creation and stage division
- PPStageModule functionality
- Integration with LlamaModel
"""

import pytest
from llm_perf.modeling import (
    PPStrategy,
    PPSchedule,
    PPModel,
    PPStageModule,
    LlamaModel,
    ShardedTensor,
    ParallelContext,
)


class TestPPStrategy:
    """Tests for PPStrategy class."""

    def test_pp_strategy_creation(self):
        """Test basic PPStrategy creation."""
        pp_strategy = PPStrategy(
            num_stages=4,
            num_virtual_stages=1,
            schedule="1f1b",
            num_micro_batches=8,
            micro_batch_size=4,
        )

        assert pp_strategy.num_stages == 4
        assert pp_strategy.num_virtual_stages == 1
        assert pp_strategy.schedule == "1f1b"
        assert pp_strategy.num_micro_batches == 8
        assert pp_strategy.micro_batch_size == 4

    def test_pp_strategy_default_values(self):
        """Test default values."""
        pp_strategy = PPStrategy()

        assert pp_strategy.num_stages == 1
        assert pp_strategy.num_virtual_stages == 1
        assert pp_strategy.schedule == "1f1b"
        assert pp_strategy.num_micro_batches == 1
        assert pp_strategy.micro_batch_size == 1

    def test_pp_strategy_validation_invalid_stages(self):
        """Test validation for invalid num_stages."""
        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            PPStrategy(num_stages=0)

    def test_pp_strategy_validation_invalid_schedule(self):
        """Test validation for invalid schedule."""
        with pytest.raises(ValueError, match="Invalid schedule"):
            PPStrategy(schedule="invalid")

    def test_pp_strategy_validation_vpp_schedule_mismatch(self):
        """Test validation for vpp without interleaved schedule."""
        with pytest.raises(ValueError, match="requires 'interleaved' or 'vpp' schedule"):
            PPStrategy(num_virtual_stages=2, schedule="1f1b")

    def test_bubble_ratio_gpipe(self):
        """Test bubble ratio for GPipe schedule."""
        pp_strategy = PPStrategy(
            num_stages=4,
            schedule="gpipe",
            num_micro_batches=8,
        )

        expected = (4 - 1) / 8
        assert pp_strategy.get_bubble_ratio() == pytest.approx(expected, rel=1e-6)

    def test_bubble_ratio_1f1b(self):
        """Test bubble ratio for 1F1B schedule."""
        pp_strategy = PPStrategy(
            num_stages=4,
            schedule="1f1b",
            num_micro_batches=8,
        )

        expected = (4 - 1) / (4 + 8 - 1)
        assert pp_strategy.get_bubble_ratio() == pytest.approx(expected, rel=1e-6)

    def test_bubble_ratio_interleaved(self):
        """Test bubble ratio for Interleaved schedule."""
        pp_strategy = PPStrategy(
            num_stages=8,
            num_virtual_stages=2,
            schedule="interleaved",
            num_micro_batches=16,
        )

        effective_stages = 8 * 2
        expected = (8 - 1) / (effective_stages + 16 - 1)
        assert pp_strategy.get_bubble_ratio() == pytest.approx(expected, rel=1e-6)

    def test_bubble_ratio_zero_micro_batches(self):
        """Test bubble ratio with invalid micro_batches (should raise error)."""
        with pytest.raises(ValueError, match="num_micro_batches must be >= 1"):
            PPStrategy(num_stages=4, num_micro_batches=0)

    def test_assign_layers_balanced(self):
        """Test balanced layer assignment."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        assignment = pp_strategy.assign_layers(model, method="balanced")

        assert len(assignment) == 8

        for layer_name, stage_idx in assignment.items():
            assert layer_name.startswith("layers.")
            assert 0 <= stage_idx < 4

    def test_assign_layers_custom(self):
        """Test custom layer assignment."""
        custom_assignment = {
            "layers.0": 0,
            "layers.1": 0,
            "layers.2": 1,
            "layers.3": 1,
        }

        pp_strategy = PPStrategy(
            num_stages=4,
            stage_assignment=custom_assignment,
        )

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
        )

        assignment = pp_strategy.assign_layers(model, method="custom")

        assert assignment == custom_assignment

    def test_assign_layers_memory_balanced(self):
        """Test memory balanced layer assignment."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        assignment = pp_strategy.assign_layers(model, method="memory_balanced")

        assert len(assignment) == 8

        for layer_name, stage_idx in assignment.items():
            assert layer_name.startswith("layers.")
            assert 0 <= stage_idx < 4

    def test_pp_strategy_to_dict(self):
        """Test to_dict method."""
        pp_strategy = PPStrategy(
            num_stages=4,
            num_micro_batches=8,
            schedule="1f1b",
        )

        result = pp_strategy.to_dict()

        assert result["num_stages"] == 4
        assert result["num_micro_batches"] == 8
        assert result["schedule"] == "1f1b"
        assert "bubble_ratio" in result
        assert "stage_assignment" in result


class TestPPSchedule:
    """Tests for PPSchedule class."""

    def test_generate_gpipe_schedule(self):
        """Test GPipe schedule generation."""
        schedules = PPSchedule.generate_gpipe_schedule(4, 8)

        assert len(schedules) == 4

        for stage_ops in schedules:
            assert len(stage_ops) == 16

            forward_ops = [op for op in stage_ops if op.startswith("F")]
            backward_ops = [op for op in stage_ops if op.startswith("B")]

            assert len(forward_ops) == 8
            assert len(backward_ops) == 8

    def test_generate_1f1b_schedule(self):
        """Test 1F1B schedule generation."""
        schedules = PPSchedule.generate_1f1b_schedule(4, 8)

        assert len(schedules) == 4

        for stage_ops in schedules:
            forward_ops = [op for op in stage_ops if op.startswith("F")]
            backward_ops = [op for op in stage_ops if op.startswith("B")]

            assert len(forward_ops) == 8
            assert len(backward_ops) == 8

    def test_generate_interleaved_schedule(self):
        """Test Interleaved schedule generation."""
        schedules = PPSchedule.generate_interleaved_schedule(8, 2, 16)

        assert len(schedules) == 4

        for gpu_ops in schedules:
            forward_ops = [op for op in gpu_ops if op.startswith("F")]
            backward_ops = [op for op in gpu_ops if op.startswith("B")]

            assert len(forward_ops) > 0
            assert len(backward_ops) > 0

    def test_visualize_schedule(self):
        """Test schedule visualization."""
        schedules = PPSchedule.generate_1f1b_schedule(4, 8)
        visualization = PPSchedule.visualize_schedule(schedules)

        assert "PP Schedule Visualization" in visualization
        assert "Stage 0" in visualization
        assert "F0" in visualization

    def test_count_operations(self):
        """Test operation counting."""
        schedules = PPSchedule.generate_1f1b_schedule(4, 8)
        counts = PPSchedule.count_operations(schedules)

        assert counts["forward"] == 32
        assert counts["backward"] == 32


class TestPPModel:
    """Tests for PPModel class."""

    def test_pp_model_creation(self):
        """Test basic PPModel creation."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        assert pp_model.original_model == model
        assert pp_model.pp_strategy == pp_strategy

    def test_pp_model_get_stage(self):
        """Test get_stage method."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        for i in range(4):
            stage = pp_model.get_stage(i)
            assert stage is not None
            assert stage.stage_idx == i

    def test_pp_model_get_all_stages(self):
        """Test get_all_stages method."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        stages = pp_model.get_all_stages()

        assert len(stages) == 4
        for stage in stages:
            assert stage is not None
            assert isinstance(stage, PPStageModule)

    def test_pp_model_custom_assignment(self):
        """Test PPModel with custom stage assignment."""
        custom_assignment = {
            "layers.0": 0,
            "layers.1": 0,
            "layers.2": 1,
            "layers.3": 1,
            "layers.4": 2,
            "layers.5": 2,
            "layers.6": 3,
            "layers.7": 3,
        }

        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(
            num_stages=4,
            stage_assignment=custom_assignment,
        )
        pp_model = PPModel(model, pp_strategy)

        assignment = pp_model.get_stage_assignment()

        assert assignment == custom_assignment

    def test_pp_model_to_dict(self):
        """Test PPModel to_dict."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4, num_micro_batches=8)
        pp_model = PPModel(model, pp_strategy)

        result = pp_model.to_dict()

        assert result["num_stages"] == 4
        assert result["num_micro_batches"] == 8
        assert "bubble_ratio" in result
        assert "stage_assignment" in result


class TestPPStageModule:
    """Tests for PPStageModule class."""

    def test_pp_stage_creation(self):
        """Test PPStageModule creation."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        stage = pp_model.get_stage(0)

        assert stage is not None
        assert stage.stage_idx == 0
        assert len(stage.get_layers()) > 0

    def test_pp_stage_get_layers(self):
        """Test get_layers method."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        stage = pp_model.get_stage(0)
        layers = stage.get_layers()

        assert len(layers) > 0
        for layer in layers:
            assert hasattr(layer, "forward")

    def test_pp_stage_params_count_breakdown(self):
        """Test params_count_breakdown."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        stage = pp_model.get_stage(0)
        breakdown = stage.params_count_breakdown()

        assert len(breakdown) > 0
        for name, count in breakdown.items():
            assert count > 0


class TestPPIntegration:
    """Integration tests for PP with models."""

    def test_llama_bind_with_pp_strategy(self):
        """Test LlamaModel.bind with pp_strategy."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        ctx = ParallelContext(
            tp_degree=8,
            pp_degree=4,
        )

        pp_strategy = PPStrategy(
            num_stages=4,
            num_micro_batches=8,
            micro_batch_size=4,
        )

        pp_model = model.bind(ctx, pp_strategy=pp_strategy)

        assert pp_model is not None
        assert isinstance(pp_model, PPModel)

    def test_llama_bind_without_pp_strategy(self):
        """Test LlamaModel.bind without pp_strategy."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        ctx = ParallelContext(
            tp_degree=8,
            pp_degree=4,
        )

        instance = model.bind(ctx)

        assert instance is not None

    def test_pp_model_bind_stage(self):
        """Test bind_stage method."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        ctx = ParallelContext(
            tp_degree=8,
            pp_degree=4,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        stage_instance = pp_model.bind_stage(0, ctx)

        assert stage_instance is not None
        assert stage_instance.pp_stage == 0

    def test_pp_model_estimate_pp_time(self):
        """Test estimate_pp_time method."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        ctx = ParallelContext(
            tp_degree=8,
            pp_degree=4,
        )

        pp_strategy = PPStrategy(
            num_stages=4,
            num_micro_batches=8,
            micro_batch_size=4,
        )

        pp_model = PPModel(model, pp_strategy)

        input_ids = ShardedTensor(shape=(1, 128), shardable={}, dtype="fp16", name="input_ids")
        model(input_ids)

        backend = None
        result = pp_model.estimate_pp_time(ctx, backend)

        assert "bubble_ratio" in result
        assert result["bubble_ratio"] == pytest.approx(pp_strategy.get_bubble_ratio(), rel=1e-6)

    def test_vpp_integration(self):
        """Test VPP (Virtual Pipeline Parallelism) integration."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=16,
            num_heads=32,
        )

        ctx = ParallelContext(
            tp_degree=8,
            pp_degree=4,
        )

        vpp_strategy = PPStrategy(
            num_stages=8,
            num_virtual_stages=2,
            schedule="interleaved",
            num_micro_batches=16,
        )

        pp_model = PPModel(model, vpp_strategy)

        assert pp_model.pp_strategy.num_virtual_stages == 2

        bubble_ratio = vpp_strategy.get_bubble_ratio()
        assert bubble_ratio < 0.5


class TestPPForward:
    """Tests for PP forward pass."""

    def test_pp_stage_forward(self):
        """Test PPStageModule forward."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        input_ids = ShardedTensor(shape=(1, 128), shardable={}, dtype="fp16", name="input_ids")

        hidden = model.embedding(input_ids)

        stage = pp_model.get_stage(0)
        output = stage(hidden)

        assert output is not None
        assert isinstance(output, ShardedTensor)

    def test_pp_model_forward(self):
        """Test PPModel forward through all stages."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=8,
            num_heads=32,
        )

        pp_strategy = PPStrategy(num_stages=4)
        pp_model = PPModel(model, pp_strategy)

        input_ids = ShardedTensor(shape=(1, 128), shardable={}, dtype="fp16", name="input_ids")

        hidden = model.embedding(input_ids)
        output = pp_model(hidden)

        assert output is not None
        assert isinstance(output, ShardedTensor)
