"""Tests for sequence parallelism communication kernels and analyzers."""

import unittest
from llm_perf.models.llama import LlamaConfig, LlamaModel
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster, NetworkConfig
from llm_perf.strategy.base import StrategyConfig, SPType
from llm_perf.kernels.communication import CommKernelRegistry
from llm_perf.analyzer.training import TrainingAnalyzer
from llm_perf.analyzer.inference import InferenceAnalyzer


class TestSPCommKernels(unittest.TestCase):
    """Test SP communication kernel creation."""

    def setUp(self):
        self.device = Device.from_preset("Ascend-910B2")
        self.network = NetworkConfig(
            intra_node_bandwidth_gbps=200.0,
            inter_node_bandwidth_gbps=100.0,
        )
        self.cluster = Cluster.create_homogeneous(
            self.device.config, 8, self.network, 8
        )
        self.registry = CommKernelRegistry(self.cluster)

    def test_create_ulysses_alltoall(self):
        """Test Ulysses-SP all-to-all kernel creation."""
        kernel = self.registry.create_sp_ulysses_alltoall(
            "layer0", 1024, [0, 1, 2, 3]
        )
        self.assertEqual(kernel.name, "sp_ulysses_alltoall_layer0")
        self.assertEqual(kernel.collective_type, "alltoall")
        self.assertEqual(kernel.num_bytes, 1024)
        self.assertEqual(kernel.participating_ranks, [0, 1, 2, 3])
        time = kernel.estimate_time()
        self.assertGreaterEqual(time, 0.0)

    def test_create_ring_p2p(self):
        """Test Ring-SP P2P kernel creation."""
        kernel = self.registry.create_sp_ring_p2p(
            "layer0", 512, [0, 1, 2, 3]
        )
        self.assertEqual(kernel.name, "sp_ring_p2p_layer0")
        self.assertEqual(kernel.collective_type, "broadcast")
        # 4 ranks -> (4-1) * 512 = 1536 bytes
        self.assertEqual(kernel.num_bytes, 1536)
        time = kernel.estimate_time()
        self.assertGreaterEqual(time, 0.0)

    def test_create_ring_allgather(self):
        """Test Ring-SP allgather kernel creation."""
        kernel = self.registry.create_sp_ring_allgather(
            "layer0", 1024, [0, 1, 2, 3]
        )
        self.assertEqual(kernel.name, "sp_ring_allgather_layer0")
        self.assertEqual(kernel.collective_type, "allgather")
        self.assertEqual(kernel.num_bytes, 1024)
        time = kernel.estimate_time()
        self.assertGreaterEqual(time, 0.0)

    def test_create_unified_2d(self):
        """Test Unified 2D-SP kernel creation."""
        kernels = self.registry.create_sp_unified_2d(
            "layer0", 1024, 512, [0, 1], [0, 1, 2, 3]
        )
        self.assertEqual(len(kernels), 2)
        self.assertEqual(kernels[0].name, "sp_ulysses_alltoall_layer0")
        self.assertEqual(kernels[1].name, "sp_ring_p2p_layer0")

    def test_create_unified_2d_allgather(self):
        """Test Unified 2D-SP with allgather ring."""
        kernels = self.registry.create_sp_unified_2d(
            "layer0", 1024, 512, [0, 1], [0, 1, 2, 3], use_ring_allgather=True
        )
        self.assertEqual(len(kernels), 2)
        self.assertEqual(kernels[0].name, "sp_ulysses_alltoall_layer0")
        self.assertEqual(kernels[1].name, "sp_ring_allgather_layer0")


class TestSPTrainingAnalyzer(unittest.TestCase):
    """Test SP communication estimation in training analyzer."""

    def setUp(self):
        self.model_config = LlamaConfig(
            name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            dtype="fp16",
        )
        self.model = LlamaModel(self.model_config)
        self.device = Device.from_preset("Ascend-910B2")
        self.network = NetworkConfig(
            intra_node_bandwidth_gbps=200.0,
        )
        self.cluster = Cluster.create_homogeneous(
            self.device.config, 8, self.network, 8
        )

    def _run_with_sp_type(self, sp_type: SPType, sp_degree: int = 4):
        strategy = StrategyConfig(
            tp_degree=1,
            pp_degree=1,
            dp_degree=1,
            sp_degree=sp_degree,
            sp_type=sp_type,
        )
        analyzer = TrainingAnalyzer(
            self.model, self.device, self.cluster, strategy
        )
        return analyzer.analyze(batch_size=8, seq_len=4096)

    def test_ulysses_training(self):
        """Test Ulysses-SP training analysis."""
        result = self._run_with_sp_type(SPType.ULYSSES, 4)
        self.assertGreater(result.tokens_per_sec, 0)
        self.assertGreater(result.memory_per_gpu_gb, 0)

    def test_ring_p2p_training(self):
        """Test Ring-SP P2P training analysis."""
        result = self._run_with_sp_type(SPType.RING_P2P, 4)
        self.assertGreater(result.tokens_per_sec, 0)
        self.assertGreater(result.memory_per_gpu_gb, 0)

    def test_ring_allgather_training(self):
        """Test Ring-SP allgather training analysis."""
        result = self._run_with_sp_type(SPType.RING_ALLGATHER, 4)
        self.assertGreater(result.tokens_per_sec, 0)
        self.assertGreater(result.memory_per_gpu_gb, 0)

    def test_unified_2d_training(self):
        """Test Unified 2D-SP training analysis."""
        strategy = StrategyConfig(
            tp_degree=1,
            pp_degree=1,
            dp_degree=1,
            sp_degree=8,
            sp_type=SPType.UNIFIED_2D,
            ulysses_degree=2,
            ring_degree=4,
        )
        analyzer = TrainingAnalyzer(
            self.model, self.device, self.cluster, strategy
        )
        result = analyzer.analyze(batch_size=8, seq_len=4096)
        self.assertGreater(result.tokens_per_sec, 0)
        self.assertGreater(result.memory_per_gpu_gb, 0)

    def test_sp_memory_reduction(self):
        """Test that SP reduces activation memory."""
        result_sp = self._run_with_sp_type(SPType.ULYSSES, 4)
        result_no_sp = self._run_with_sp_type(SPType.ULYSSES, 1)
        self.assertLess(
            result_sp.memory_per_gpu_gb,
            result_no_sp.memory_per_gpu_gb,
        )


class TestSPInferenceAnalyzer(unittest.TestCase):
    """Test SP communication estimation in inference analyzer."""

    def setUp(self):
        self.model_config = LlamaConfig(
            name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            dtype="fp16",
        )
        self.model = LlamaModel(self.model_config)
        self.device = Device.from_preset("Ascend-910B2")
        self.network = NetworkConfig(
            intra_node_bandwidth_gbps=200.0,
        )
        self.cluster = Cluster.create_homogeneous(
            self.device.config, 8, self.network, 8
        )

    def _run_inference_with_sp(self, sp_type: SPType, sp_degree: int = 4):
        strategy = StrategyConfig(
            tp_degree=1,
            pp_degree=1,
            dp_degree=1,
            sp_degree=sp_degree,
            sp_type=sp_type,
        )
        analyzer = InferenceAnalyzer(
            self.model, self.device, self.cluster, strategy
        )
        return analyzer.analyze(
            batch_size=1,
            prompt_len=1024,
            generation_len=128,
        )

    def test_ulysses_inference(self):
        """Test Ulysses-SP inference analysis."""
        result = self._run_inference_with_sp(SPType.ULYSSES, 4)
        self.assertGreater(result.prefill_time_sec, 0)
        self.assertGreater(result.decode_tokens_per_sec, 0)

    def test_ring_p2p_inference(self):
        """Test Ring-SP P2P inference analysis."""
        result = self._run_inference_with_sp(SPType.RING_P2P, 4)
        self.assertGreater(result.prefill_time_sec, 0)
        self.assertGreater(result.decode_tokens_per_sec, 0)

    def test_ring_allgather_inference(self):
        """Test Ring-SP allgather inference analysis."""
        result = self._run_inference_with_sp(SPType.RING_ALLGATHER, 4)
        self.assertGreater(result.prefill_time_sec, 0)
        self.assertGreater(result.decode_tokens_per_sec, 0)

    def test_unified_2d_inference(self):
        """Test Unified 2D-SP inference analysis."""
        strategy = StrategyConfig(
            tp_degree=1,
            pp_degree=1,
            dp_degree=1,
            sp_degree=8,
            sp_type=SPType.UNIFIED_2D,
            ulysses_degree=2,
            ring_degree=4,
        )
        analyzer = InferenceAnalyzer(
            self.model, self.device, self.cluster, strategy
        )
        result = analyzer.analyze(
            batch_size=1,
            prompt_len=1024,
            generation_len=128,
        )
        self.assertGreater(result.prefill_time_sec, 0)
        self.assertGreater(result.decode_tokens_per_sec, 0)


class TestStrategyConfigSP(unittest.TestCase):
    """Test StrategyConfig SP serialization."""

    def test_sp_type_serialization(self):
        """Test SP type round-trip through dict."""
        config = StrategyConfig(
            sp_degree=4,
            sp_type=SPType.RING_P2P,
            ulysses_degree=2,
            ring_degree=2,
        )
        data = config.to_dict()
        self.assertEqual(data["sequence_parallelism"]["sp_type"], "ring_p2p")
        self.assertEqual(data["sequence_parallelism"]["ulysses_degree"], 2)
        self.assertEqual(data["sequence_parallelism"]["ring_degree"], 2)

        restored = StrategyConfig.from_dict(data)
        self.assertEqual(restored.sp_type, SPType.RING_P2P)
        self.assertEqual(restored.ulysses_degree, 2)
        self.assertEqual(restored.ring_degree, 2)

    def test_sp_type_default(self):
        """Test default SP type when missing from dict."""
        config = StrategyConfig.from_dict({})
        self.assertEqual(config.sp_type, SPType.ULYSSES)
        self.assertEqual(config.ulysses_degree, 1)
        self.assertEqual(config.ring_degree, 1)


if __name__ == "__main__":
    unittest.main()
