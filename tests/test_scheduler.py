"""Tests for scheduler module."""

import pytest

from llm_perf.scheduler import (
    SchedulerConfig,
    SchedulerModel,
    SchedulerResult,
    OverlapFeature,
    PipelineBubbleFeature,
    ChunkingFeature,
    PrefetchFeature,
)
from llm_perf.strategy.base import StrategyConfig as StrategyConfigBase


class TestSchedulerConfig:
    """Test SchedulerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SchedulerConfig()
        assert config.enabled_features == []
        assert config.overlap_enabled == False
        assert config.overlap_efficiency == 0.8
        assert config.chunk_size is None
        assert config.prefetch_enabled == False
        assert config.pipeline_bubble_ratio == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = SchedulerConfig(
            enabled_features=["overlap", "chunking"],
            overlap_enabled=True,
            overlap_efficiency=0.9,
            chunk_size=1024,
            prefetch_enabled=True,
            prefetch_overlap_ratio=0.6,
            pipeline_bubble_ratio=0.2,
        )
        assert config.enabled_features == ["overlap", "chunking"]
        assert config.overlap_enabled == True
        assert config.overlap_efficiency == 0.9
        assert config.chunk_size == 1024
        assert config.prefetch_enabled == True
        assert config.prefetch_overlap_ratio == 0.6
        assert config.pipeline_bubble_ratio == 0.2

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = SchedulerConfig(
            overlap_enabled=True,
            overlap_efficiency=0.85,
        )
        data = config.to_dict()
        assert "overlap" in data
        assert data["overlap"]["enabled"] == True
        assert data["overlap"]["efficiency"] == 0.85

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "overlap": {"enabled": True, "efficiency": 0.75},
            "chunking": {"chunk_size": 512},
            "pipeline_bubble": {"ratio": 0.15},
        }
        config = SchedulerConfig.from_dict(data)
        assert config.overlap_enabled == True
        assert config.overlap_efficiency == 0.75
        assert config.chunk_size == 512
        assert config.pipeline_bubble_ratio == 0.15


class TestSchedulerResult:
    """Test SchedulerResult."""

    def test_default_result(self):
        """Test default result."""
        result = SchedulerResult()
        assert result.compute_time == 0.0
        assert result.comm_time == 0.0
        assert result.memory_bytes == 0
        assert result.overlapped_time == 0.0

    def test_custom_result(self):
        """Test custom result."""
        result = SchedulerResult(
            compute_time=1.5,
            comm_time=0.5,
            memory_bytes=10_000_000_000,
        )
        assert result.compute_time == 1.5
        assert result.comm_time == 0.5
        assert result.memory_bytes == 10_000_000_000

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = SchedulerResult(
            compute_time=2.0,
            comm_time=1.0,
            memory_bytes=20_000_000_000,
            overlapped_time=2.5,
        )
        data = result.to_dict()
        assert data["compute_time"] == 2.0
        assert data["comm_time"] == 1.0
        assert data["memory_bytes"] == 20_000_000_000
        assert data["memory_gb"] == 20.0
        assert data["overlapped_time"] == 2.5


class TestOverlapFeature:
    """Test OverlapFeature."""

    def test_init(self):
        """Test initialization."""
        config = SchedulerConfig(overlap_efficiency=0.9)
        feature = OverlapFeature(config)
        assert feature.name == "overlap"
        assert feature.efficiency == 0.9

    def test_apply_overhead(self):
        """Test overhead application."""
        config = SchedulerConfig()
        feature = OverlapFeature(config)
        assert feature.apply_overhead(1.0) == 1.0
        assert feature.apply_overhead(2.5) == 2.5

    def test_apply_overlap_full_efficiency(self):
        """Test overlap with full efficiency."""
        config = SchedulerConfig(overlap_efficiency=1.0)
        feature = OverlapFeature(config)
        effective = feature.apply_overlap(1.0, 0.5)
        assert effective == 1.0
        effective = feature.apply_overlap(0.5, 1.0)
        assert effective == 1.0

    def test_apply_overlap_partial_efficiency(self):
        """Test overlap with partial efficiency."""
        config = SchedulerConfig(overlap_efficiency=0.8)
        feature = OverlapFeature(config)
        effective = feature.apply_overlap(1.0, 0.5)
        expected = 1.0 + (0.5 - 0.5 * 0.8)
        assert abs(effective - expected) < 0.001

    def test_apply_overlap_zero_time(self):
        """Test overlap with zero time."""
        config = SchedulerConfig()
        feature = OverlapFeature(config)
        assert feature.apply_overlap(0.0, 1.0) == 1.0
        assert feature.apply_overlap(1.0, 0.0) == 1.0
        assert feature.apply_overlap(0.0, 0.0) == 0.0

    def test_apply_memory_optimization(self):
        """Test memory optimization."""
        config = SchedulerConfig()
        feature = OverlapFeature(config)
        assert feature.apply_memory_optimization(1000) == 1000


class TestPipelineBubbleFeature:
    """Test PipelineBubbleFeature."""

    def test_init(self):
        """Test initialization."""
        config = SchedulerConfig(pipeline_bubble_ratio=0.15)
        feature = PipelineBubbleFeature(config)
        assert feature.name == "pipeline_bubble"
        assert feature.bubble_ratio == 0.15

    def test_apply_overhead(self):
        """Test bubble overhead application."""
        config = SchedulerConfig(pipeline_bubble_ratio=0.1)
        feature = PipelineBubbleFeature(config)
        result = feature.apply_overhead(1.0)
        assert result == 1.1
        result = feature.apply_overhead(2.0)
        assert result == 2.2

    def test_apply_overlap(self):
        """Test overlap application."""
        config = SchedulerConfig()
        feature = PipelineBubbleFeature(config)
        assert feature.apply_overlap(1.0, 0.5) == 1.5

    def test_apply_memory_optimization(self):
        """Test memory optimization."""
        config = SchedulerConfig()
        feature = PipelineBubbleFeature(config)
        assert feature.apply_memory_optimization(1000) == 1000

    def test_calculate_bubble_ratio_gpipe(self):
        """Test bubble ratio calculation for GPipe."""
        config = SchedulerConfig()
        feature = PipelineBubbleFeature(config)
        ratio = feature.calculate_bubble_ratio(4, 8, "gpipe")
        expected = (4 - 1) / 8
        assert abs(ratio - expected) < 0.001

    def test_calculate_bubble_ratio_1f1b(self):
        """Test bubble ratio calculation for 1F1B."""
        config = SchedulerConfig()
        feature = PipelineBubbleFeature(config)
        ratio = feature.calculate_bubble_ratio(4, 8, "1f1b")
        expected = (4 - 1) / (4 + 8 - 1)
        assert abs(ratio - expected) < 0.001

    def test_calculate_bubble_ratio_interleaved(self):
        """Test bubble ratio calculation for interleaved."""
        config = SchedulerConfig()
        feature = PipelineBubbleFeature(config)
        ratio = feature.calculate_bubble_ratio(4, 8, "interleaved")
        base_bubble = (4 - 1) / (4 + 8 - 1)
        expected = base_bubble / 2
        assert abs(ratio - expected) < 0.001

    def test_calculate_bubble_ratio_pp1(self):
        """Test bubble ratio with pp_degree=1."""
        config = SchedulerConfig()
        feature = PipelineBubbleFeature(config)
        ratio = feature.calculate_bubble_ratio(1, 8, "1f1b")
        assert ratio == 0.0


class TestChunkingFeature:
    """Test ChunkingFeature."""

    def test_init(self):
        """Test initialization."""
        config = SchedulerConfig(chunk_size=1024)
        feature = ChunkingFeature(config)
        assert feature.name == "chunking"
        assert feature.chunk_size == 1024

    def test_init_no_chunk_size(self):
        """Test initialization without chunk size."""
        config = SchedulerConfig()
        feature = ChunkingFeature(config)
        assert feature.chunk_size is None

    def test_apply_overhead_with_chunking(self):
        """Test overhead with chunking enabled."""
        config = SchedulerConfig(chunk_size=512)
        feature = ChunkingFeature(config)
        result = feature.apply_overhead(1.0)
        assert result == 1.05
        result = feature.apply_overhead(2.0)
        assert result == 2.1

    def test_apply_overhead_no_chunking(self):
        """Test overhead without chunking."""
        config = SchedulerConfig()
        feature = ChunkingFeature(config)
        assert feature.apply_overhead(1.0) == 1.0

    def test_apply_overlap(self):
        """Test overlap application."""
        config = SchedulerConfig()
        feature = ChunkingFeature(config)
        assert feature.apply_overlap(1.0, 0.5) == 1.5

    def test_apply_memory_optimization_with_chunking(self):
        """Test memory optimization with chunking."""
        config = SchedulerConfig(chunk_size=512)
        feature = ChunkingFeature(config)
        result = feature.apply_memory_optimization(10_000_000_000)
        assert result == int(10_000_000_000 * 0.7)

    def test_apply_memory_optimization_no_chunking(self):
        """Test memory optimization without chunking."""
        config = SchedulerConfig()
        feature = ChunkingFeature(config)
        assert feature.apply_memory_optimization(1000) == 1000

    def test_calculate_memory_reduction_ratio(self):
        """Test memory reduction ratio calculation."""
        config = SchedulerConfig(chunk_size=512)
        feature = ChunkingFeature(config)
        ratio = feature.calculate_memory_reduction_ratio(2048)
        assert ratio < 1.0
        ratio_full = feature.calculate_memory_reduction_ratio(2048, 2048)
        assert ratio_full == 1.0


class TestPrefetchFeature:
    """Test PrefetchFeature."""

    def test_init(self):
        """Test initialization."""
        config = SchedulerConfig(prefetch_enabled=True, prefetch_overlap_ratio=0.6)
        feature = PrefetchFeature(config)
        assert feature.name == "prefetch"
        assert feature.enabled == True
        assert feature.overlap_ratio == 0.6

    def test_apply_overhead(self):
        """Test overhead application."""
        config = SchedulerConfig(prefetch_enabled=True)
        feature = PrefetchFeature(config)
        assert feature.apply_overhead(1.0) == 1.0

    def test_apply_overlap_enabled(self):
        """Test overlap with prefetch enabled."""
        config = SchedulerConfig(prefetch_enabled=True, prefetch_overlap_ratio=0.5)
        feature = PrefetchFeature(config)
        effective = feature.apply_overlap(1.0, 0.5)
        max_time = 1.0
        min_time = 0.5
        expected = max_time + (min_time - min_time * 0.5)
        assert abs(effective - expected) < 0.001

    def test_apply_overlap_disabled(self):
        """Test overlap with prefetch disabled."""
        config = SchedulerConfig(prefetch_enabled=False)
        feature = PrefetchFeature(config)
        assert feature.apply_overlap(1.0, 0.5) == 1.5

    def test_apply_memory_optimization_enabled(self):
        """Test memory optimization with prefetch enabled."""
        config = SchedulerConfig(prefetch_enabled=True)
        feature = PrefetchFeature(config)
        result = feature.apply_memory_optimization(1000)
        assert result == int(1000 * 1.05)

    def test_apply_memory_optimization_disabled(self):
        """Test memory optimization with prefetch disabled."""
        config = SchedulerConfig(prefetch_enabled=False)
        feature = PrefetchFeature(config)
        assert feature.apply_memory_optimization(1000) == 1000

    def test_estimate_prefetch_effectiveness(self):
        """Test prefetch effectiveness estimation."""
        config = SchedulerConfig(prefetch_enabled=True, prefetch_overlap_ratio=0.5)
        feature = PrefetchFeature(config)
        eff = feature.estimate_prefetch_effectiveness(1.0, 0.5)
        assert eff == 0.5
        eff_partial = feature.estimate_prefetch_effectiveness(0.3, 1.0)
        assert eff_partial < 0.5


class TestSchedulerModel:
    """Test SchedulerModel."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        model = SchedulerModel()
        assert model.config.enabled_features == []
        assert len(model._features) == 0

    def test_init_with_config(self):
        """Test initialization with config."""
        config = SchedulerConfig(
            enabled_features=["overlap"],
            overlap_efficiency=0.8,
        )
        model = SchedulerModel(config)
        assert len(model._features) == 1
        assert model.get_feature("overlap") is not None

    def test_init_with_overlap_enabled(self):
        """Test initialization with overlap enabled."""
        config = SchedulerConfig(overlap_enabled=True)
        model = SchedulerModel(config)
        assert model.get_feature("overlap") is not None

    def test_add_feature(self):
        """Test adding feature."""
        model = SchedulerModel()
        config = SchedulerConfig()
        feature = OverlapFeature(config)
        model.add_feature(feature)
        assert model.get_feature("overlap") is not None
        assert len(model._features) == 1

    def test_apply_all_no_features(self):
        """Test apply_all without features."""
        model = SchedulerModel()
        result = SchedulerResult(
            compute_time=1.0,
            comm_time=0.5,
            memory_bytes=1000,
        )
        modified = model.apply_all(result)
        assert modified.overlapped_time == 1.5
        assert modified.compute_time == 1.0
        assert modified.memory_bytes == 1000

    def test_apply_all_with_overlap(self):
        """Test apply_all with overlap feature."""
        config = SchedulerConfig(
            enabled_features=["overlap"],
            overlap_efficiency=1.0,
        )
        model = SchedulerModel(config)
        result = SchedulerResult(
            compute_time=1.0,
            comm_time=0.5,
            memory_bytes=1000,
        )
        modified = model.apply_all(result)
        assert modified.overlapped_time == 1.0

    def test_apply_all_with_pipeline_bubble(self):
        """Test apply_all with pipeline bubble."""
        config = SchedulerConfig(
            enabled_features=["pipeline_bubble"],
            pipeline_bubble_ratio=0.1,
        )
        model = SchedulerModel(config)
        result = SchedulerResult(
            compute_time=1.0,
            comm_time=0.5,
            memory_bytes=1000,
        )
        modified = model.apply_all(result)
        assert modified.compute_time == 1.1

    def test_apply_all_with_chunking(self):
        """Test apply_all with chunking."""
        config = SchedulerConfig(
            enabled_features=["chunking"],
            chunk_size=512,
        )
        model = SchedulerModel(config)
        result = SchedulerResult(
            compute_time=1.0,
            comm_time=0.5,
            memory_bytes=1000,
        )
        modified = model.apply_all(result)
        assert modified.compute_time == 1.05
        assert modified.memory_bytes == int(1000 * 0.7)

    def test_apply_overlap_method(self):
        """Test apply_overlap method."""
        config = SchedulerConfig(
            overlap_enabled=True,
            overlap_efficiency=1.0,
        )
        model = SchedulerModel(config)
        effective = model.apply_overlap(1.0, 0.5)
        assert effective == 1.0

    def test_apply_memory_optimization(self):
        """Test apply_memory_optimization method."""
        config = SchedulerConfig(
            enabled_features=["chunking"],
            chunk_size=512,
        )
        model = SchedulerModel(config)
        memory = model.apply_memory_optimization(1000)
        assert memory == int(1000 * 0.7)

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = SchedulerConfig(enabled_features=["overlap"])
        model = SchedulerModel(config)
        data = model.to_dict()
        assert "config" in data
        assert "features" in data
        assert len(data["features"]) == 1


class TestStrategyConfigIntegration:
    """Test StrategyConfig integration with scheduler."""

    def test_strategy_config_scheduler_default(self):
        """Test StrategyConfig with default scheduler."""
        strategy = StrategyConfigBase()
        assert strategy.scheduler == {}

    def test_strategy_config_scheduler_custom(self):
        """Test StrategyConfig with custom scheduler."""
        scheduler_config = SchedulerConfig(
            enabled_features=["overlap"],
            overlap_efficiency=0.85,
        )
        strategy = StrategyConfigBase(scheduler=scheduler_config.to_dict())
        assert strategy.scheduler["enabled_features"] == ["overlap"]
        assert strategy.scheduler["overlap"]["efficiency"] == 0.85

    def test_strategy_config_to_dict(self):
        """Test StrategyConfig to_dict includes scheduler."""
        scheduler_config = SchedulerConfig(overlap_enabled=True)
        strategy = StrategyConfigBase(scheduler=scheduler_config.to_dict())
        data = strategy.to_dict()
        assert "scheduler" in data
        assert data["scheduler"]["overlap"]["enabled"] == True

    def test_strategy_config_from_dict(self):
        """Test StrategyConfig from_dict with scheduler."""
        data = {
            "scheduler": {
                "overlap": {"enabled": True, "efficiency": 0.9},
                "chunking": {"chunk_size": 1024},
            }
        }
        strategy = StrategyConfigBase.from_dict(data)
        assert strategy.scheduler["overlap"]["enabled"] == True
        assert strategy.scheduler["overlap"]["efficiency"] == 0.9
        assert strategy.scheduler["chunking"]["chunk_size"] == 1024


class TestAnalyzerIntegration:
    """Test analyzer integration with scheduler."""

    def test_apply_scheduler_features_method_exists(self):
        """Test that analyzer has scheduler features method."""
        from llm_perf.analyzer.base import BaseAnalyzer

        assert hasattr(BaseAnalyzer, "_apply_scheduler_features")

    def test_scheduler_model_property_exists(self):
        """Test that analyzer has scheduler_model property."""
        from llm_perf.analyzer.base import BaseAnalyzer

        assert hasattr(BaseAnalyzer, "scheduler_model")

    def test_scheduler_model_lazy_init(self):
        """Test scheduler_model lazy initialization."""
        from llm_perf.scheduler.base import SchedulerModel, SchedulerConfig

        strategy = StrategyConfigBase(scheduler={"enabled_features": ["overlap"], "overlap": {"enabled": True}})
        scheduler_config = SchedulerConfig.from_dict(strategy.scheduler)
        model = SchedulerModel(scheduler_config)

        assert model.get_feature("overlap") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
