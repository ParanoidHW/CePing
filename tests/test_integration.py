"""Integration tests for the full evaluation pipeline."""

import unittest
from llm_perf.models.llama import LlamaConfig, LlamaModel
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster, NetworkConfig
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.training import TrainingAnalyzer
from llm_perf.analyzer.inference import InferenceAnalyzer


class TestAscendIntegration(unittest.TestCase):
    """Integration tests with Ascend NPUs."""
    
    def test_llama7b_training_on_ascend(self):
        """Test Llama-7B training analysis on Ascend 910B."""
        # Create model
        model_config = LlamaConfig(
            name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            dtype="fp16",
        )
        model = LlamaModel(model_config)
        
        # Create Ascend cluster
        device = Device.from_preset("Ascend-910B2")
        network = NetworkConfig(
            intra_node_bandwidth_gbps=200.0,  # HCCS
            inter_node_bandwidth_gbps=100.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, network, 8)
        
        # Create strategy
        strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)
        
        # Analyze
        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(batch_size=32, seq_len=4096)
        
        # Verify results
        self.assertGreater(result.tokens_per_sec, 0)
        self.assertGreater(result.memory_per_gpu_gb, 0)
        self.assertLess(result.memory_per_gpu_gb, device.config.memory_gb)
    
    def test_llama7b_inference_on_ascend(self):
        """Test Llama-7B inference analysis on Ascend 910B."""
        model_config = LlamaConfig(
            name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            dtype="fp16",
        )
        model = LlamaModel(model_config)
        
        device = Device.from_preset("Ascend-910B2")
        network = NetworkConfig(
            intra_node_bandwidth_gbps=200.0,
        )
        cluster = Cluster.create_homogeneous(device.config, 8, network, 8)
        
        strategy = StrategyConfig(tp_degree=8)
        
        analyzer = InferenceAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            batch_size=8,
            prompt_len=1024,
            generation_len=128,
        )
        
        # Verify results
        self.assertGreater(result.prefill_time_sec, 0)
        self.assertGreater(result.decode_time_per_step_sec, 0)
        self.assertGreater(result.decode_tokens_per_sec, 0)
        self.assertGreater(result.memory_per_gpu_gb, 0)
    
    def test_ascend_vs_nvidia_performance(self):
        """Compare performance between Ascend and NVIDIA for same model."""
        model_config = LlamaConfig(
            name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            dtype="fp16",
        )
        model = LlamaModel(model_config)
        
        network = NetworkConfig(
            intra_node_bandwidth_gbps=400.0,
        )
        
        # Ascend analysis
        ascend_device = Device.from_preset("Ascend-910B2")
        ascend_cluster = Cluster.create_homogeneous(ascend_device.config, 8, network, 8)
        ascend_strategy = StrategyConfig(tp_degree=8)
        ascend_analyzer = TrainingAnalyzer(model, ascend_device, ascend_cluster, ascend_strategy)
        ascend_result = ascend_analyzer.analyze(batch_size=16, seq_len=4096)
        
        # NVIDIA analysis
        nvidia_device = Device.from_preset("H100-SXM-80GB")
        nvidia_cluster = Cluster.create_homogeneous(nvidia_device.config, 8, network, 8)
        nvidia_strategy = StrategyConfig(tp_degree=8)
        nvidia_analyzer = TrainingAnalyzer(model, nvidia_device, nvidia_cluster, nvidia_strategy)
        nvidia_result = nvidia_analyzer.analyze(batch_size=16, seq_len=4096)
        
        # H100 should have higher throughput than 910B2
        # (H100: 989 TFLOPS vs 910B2: 376 TFLOPS)
        self.assertGreater(nvidia_result.tokens_per_sec, ascend_result.tokens_per_sec)
        
        # Both should complete without OOM
        self.assertLess(ascend_result.memory_per_gpu_gb, ascend_device.config.memory_gb)
        self.assertLess(nvidia_result.memory_per_gpu_gb, nvidia_device.config.memory_gb)


class TestDifferentAscendVersions(unittest.TestCase):
    """Test different Ascend NPU versions."""
    
    def test_ascend_910a_vs_910b(self):
        """Compare 910A and 910B performance."""
        model_config = LlamaConfig(
            name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            dtype="fp16",
        )
        model = LlamaModel(model_config)
        
        network = NetworkConfig(intra_node_bandwidth_gbps=200.0)
        strategy = StrategyConfig(tp_degree=8)
        
        # 910A analysis
        device_a = Device.from_preset("Ascend-910A")
        cluster_a = Cluster.create_homogeneous(device_a.config, 8, network, 8)
        analyzer_a = TrainingAnalyzer(model, device_a, cluster_a, strategy)
        result_a = analyzer_a.analyze(batch_size=16, seq_len=4096)
        
        # 910B2 analysis
        device_b = Device.from_preset("Ascend-910B2")
        cluster_b = Cluster.create_homogeneous(device_b.config, 8, network, 8)
        analyzer_b = TrainingAnalyzer(model, device_b, cluster_b, strategy)
        result_b = analyzer_b.analyze(batch_size=16, seq_len=4096)
        
        # 910B2 should be faster than 910A
        # (910B2: 376 TFLOPS vs 910A: 256 TFLOPS)
        self.assertGreater(result_b.tokens_per_sec, result_a.tokens_per_sec)
    
    def test_ascend_910c_performance(self):
        """Test newer 910C performance."""
        model_config = LlamaConfig(
            name="llama-13b",
            vocab_size=32000,
            hidden_size=5120,
            num_layers=40,
            num_attention_heads=40,
            dtype="fp16",
        )
        model = LlamaModel(model_config)
        
        device = Device.from_preset("Ascend-910C")
        network = NetworkConfig(intra_node_bandwidth_gbps=400.0)
        cluster = Cluster.create_homogeneous(device.config, 8, network, 8)
        strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)
        
        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(batch_size=8, seq_len=4096)
        
        self.assertGreater(result.tokens_per_sec, 0)
        # 910C should handle 13B model
        self.assertLess(result.memory_per_gpu_gb, device.config.memory_gb)


if __name__ == "__main__":
    unittest.main()
