"""测试推理场景的子模块分解."""

import unittest

from llm_perf.hardware import Device
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.hardware.cluster import Cluster
from llm_perf.modeling import create_model_from_config
from llm_perf.strategy import StrategyConfig
from llm_perf.analyzer import UnifiedAnalyzer


class TestInferenceBreakdown(unittest.TestCase):
    """测试推理场景的子模块分解."""

    @classmethod
    def setUpClass(cls):
        cls.model = create_model_from_config({"type": "llama-7b"})
        cls.device = Device.from_preset("H100-SXM-80GB")
        cls.topology = NetworkTopology.create_2tier_simple(
            inter_node_bw_gbps=100, intra_node_bw_gbps=200
        )
        cls.cluster = Cluster.create_homogeneous(
            cls.device.config, num_devices=8, topology=cls.topology
        )
        cls.strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)

    def test_inference_prefill_has_submodule_breakdown(self):
        """测试 prefill 阶段有子模块分解."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        prefill = result.get_phase("prefill")
        self.assertIsNotNone(prefill)
        self.assertTrue(len(prefill.submodules) > 0)

        transformer_blocks = [
            s for s in prefill.submodules if s.submodule_type == "transformer_block"
        ]
        self.assertTrue(len(transformer_blocks) > 0)

    def test_inference_decode_has_submodule_breakdown(self):
        """测试 decode 阅段有子模块分解."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        decode = result.get_phase("decode")
        self.assertIsNotNone(decode)
        self.assertTrue(len(decode.submodules) > 0)

        transformer_blocks = [
            s for s in decode.submodules if s.submodule_type == "transformer_block"
        ]
        self.assertTrue(len(transformer_blocks) > 0)

    def test_inference_transformer_block_has_nested_breakdown(self):
        """测试 transformer_block 有嵌套分解（attention + ffn）."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        detailed = result.detailed_breakdown
        by_sub = detailed.get("by_submodule_type", {})
        self.assertIn("transformer_block", by_sub)

        transformer_data = by_sub["transformer_block"]
        self.assertIn("nested_breakdown", transformer_data)

        nested = transformer_data["nested_breakdown"]
        self.assertIn("attention", nested)
        self.assertIn("ffn", nested)

    def test_inference_nested_breakdown_has_activations_gb(self):
        """测试嵌套分解有 activations_gb 别名."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        detailed = result.detailed_breakdown
        by_sub = detailed.get("by_submodule_type", {})
        transformer_data = by_sub.get("transformer_block", {})
        nested = transformer_data.get("nested_breakdown", {})

        for nested_type, nested_data in nested.items():
            memory = nested_data.get("memory", {})
            self.assertIn("activation_gb", memory)
            self.assertIn("activations_gb", memory)
            self.assertEqual(memory["activation_gb"], memory["activations_gb"])

    def test_inference_time_breakdown_consistency(self):
        """测试推理时间分解的一致性（compute + comm 不重复计算）."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        prefill = result.get_phase("prefill")
        decode = result.get_phase("decode")

        for phase in [prefill, decode]:
            for s in phase.submodules:
                if s.submodule_type == "transformer_block":
                    nested_sum = sum(ns.time_sec for ns in s.nested_submodules)

                    self.assertTrue(
                        s.time_sec >= nested_sum * 0.5,
                        f"{s.name}: compute time should be >= nested compute sum",
                    )

    def test_inference_compute_time_only(self):
        """测试 compute.time_sec 不包含通信时间."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        detailed = result.detailed_breakdown
        by_sub = detailed.get("by_submodule_type", {})

        for submodule_type, data in by_sub.items():
            compute_time = data.get("compute", {}).get("time_sec", 0)
            comm_time = data.get("communication", {}).get("time_sec", 0)

            if submodule_type == "transformer_block":
                total_time = compute_time + comm_time
                self.assertTrue(
                    total_time > compute_time,
                    f"{submodule_type}: total should > compute when comm > 0",
                )

    def test_inference_phase_submodule_consistency(self):
        """测试 phase 的 submodule 时间与 by_submodule_type 一致."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        detailed = result.detailed_breakdown
        by_sub = detailed.get("by_submodule_type", {})

        total_compute = sum(
            v.get("compute", {}).get("time_sec", 0) for v in by_sub.values()
        )
        total_comm = sum(
            v.get("communication", {}).get("time_sec", 0) for v in by_sub.values()
        )

        self.assertTrue(total_compute > 0)
        self.assertTrue(total_comm >= 0)


class TestInferencePerformanceBreakdown(unittest.TestCase):
    """测试推理场景的性能分解."""

    @classmethod
    def setUpClass(cls):
        cls.model = create_model_from_config({"type": "llama-7b"})
        cls.device = Device.from_preset("H100-SXM-80GB")
        cls.topology = NetworkTopology.create_2tier_simple(
            inter_node_bw_gbps=100, intra_node_bw_gbps=200
        )
        cls.cluster = Cluster.create_homogeneous(
            cls.device.config, num_devices=8, topology=cls.topology
        )
        cls.strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)

    def test_inference_has_performance_breakdown(self):
        """测试推理有性能分解."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        breakdown = result.breakdown
        self.assertIsNotNone(breakdown)
        self.assertIn("inference_breakdown", breakdown)

    def test_inference_breakdown_has_prefill_time(self):
        """测试推理分解有 prefill 时间."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        inf_breakdown = result.breakdown.get("inference_breakdown", {})
        self.assertIn("prefill_sec", inf_breakdown)
        self.assertGreater(inf_breakdown["prefill_sec"], 0)
        self.assertIn("prefill_percent", inf_breakdown)

    def test_inference_breakdown_has_decode_time(self):
        """测试推理分解有 decode 时间."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        inf_breakdown = result.breakdown.get("inference_breakdown", {})
        self.assertIn("decode_sec", inf_breakdown)
        self.assertGreater(inf_breakdown["decode_sec"], 0)
        self.assertIn("decode_percent", inf_breakdown)
        self.assertIn("decode_per_token_sec", inf_breakdown)

    def test_inference_breakdown_percentage_sum(self):
        """测试推理分解占比总和接近100%."""
        analyzer = UnifiedAnalyzer(self.model, self.device, self.cluster, self.strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        inf_breakdown = result.breakdown.get("inference_breakdown", {})
        total_percent = (
            inf_breakdown.get("prefill_percent", 0)
            + inf_breakdown.get("decode_percent", 0)
            + inf_breakdown.get("communication_percent", 0)
            + inf_breakdown.get("kv_cache_percent", 0)
        )
        self.assertAlmostEqual(total_percent, 100.0, delta=1.0)

    def test_inference_breakdown_http_api(self):
        """测试 HTTP API 返回推理性能分解."""
        from web.app import app

        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-inference",
                "batch_size": 1,
                "prompt_len": 512,
                "generation_len": 128,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])

        result = data["result"]
        breakdown = result.get("breakdown", {})
        self.assertIn("inference_breakdown", breakdown)

        inf_breakdown = breakdown["inference_breakdown"]
        self.assertIn("prefill_sec", inf_breakdown)
        self.assertIn("decode_sec", inf_breakdown)
        self.assertIn("decode_per_token_sec", inf_breakdown)
        self.assertIn("communication_sec", inf_breakdown)


class TestInferenceBreakdownHTTPAPI(unittest.TestCase):
    """测试推理场景的 HTTP API 分解."""

    def test_http_api_inference_breakdown(self):
        """测试 HTTP API 返回推理分解数据."""
        from web.app import app

        client = app.test_client()

        response = client.post(
            "/api/evaluate",
            json={
                "model": {"type": "llama-7b"},
                "device": "H100-SXM-80GB",
                "num_devices": 8,
                "devices_per_node": 8,
                "topology": {"type": "2-Tier Simple", "intra_node_bw_gbps": 200},
                "strategy": {"tp": 8, "pp": 1, "dp": 1},
                "workload": "llm-inference",
                "batch_size": 1,
                "prompt_len": 512,
                "generation_len": 128,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])

        result = data["result"]
        detailed = result.get("detailed_breakdown", {})
        by_sub = detailed.get("by_submodule_type", {})

        self.assertIn("transformer_block", by_sub)

        transformer_data = by_sub["transformer_block"]
        self.assertIn("nested_breakdown", transformer_data)
        nested = transformer_data["nested_breakdown"]
        self.assertIn("attention", nested)
        self.assertIn("ffn", nested)

        for nested_type, nested_data in nested.items():
            memory = nested_data.get("memory", {})
            self.assertIn("activations_gb", memory)


class TestTransformerBlockNestedBreakdown(unittest.TestCase):
    """测试 transformer_block 嵌套分解与模型类型无关."""

    def test_llama_has_attention_ffn_breakdown(self):
        """测试 LLaMA transformer_block 有 attention + ffn 分解."""
        model = create_model_from_config({"type": "llama-7b"})
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology.create_2tier_simple(
            inter_node_bw_gbps=100, intra_node_bw_gbps=200
        )
        cluster = Cluster.create_homogeneous(
            device.config, num_devices=8, topology=topology
        )
        strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        detailed = result.detailed_breakdown
        by_sub = detailed.get("by_submodule_type", {})
        self.assertIn("transformer_block", by_sub)

        transformer_data = by_sub["transformer_block"]
        self.assertIn("nested_breakdown", transformer_data)

        nested = transformer_data["nested_breakdown"]
        self.assertIn("attention", nested)
        self.assertIn("ffn", nested)

    def test_qwen35_moe_has_attention_moe_breakdown(self):
        """测试 Qwen3.5 MoE transformer_block 有 attention + moe 分解."""
        from llm_perf.modeling.qwen3_5 import Qwen3_5MoEModel

        model = Qwen3_5MoEModel(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            num_experts=256,
            num_experts_per_token=8,
            shared_expert_intermediate=512,
        )
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology.create_2tier_simple(
            inter_node_bw_gbps=100, intra_node_bw_gbps=200
        )
        cluster = Cluster.create_homogeneous(
            device.config, num_devices=8, topology=topology
        )
        strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1, ep_degree=2)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        detailed = result.detailed_breakdown
        by_sub = detailed.get("by_submodule_type", {})
        self.assertIn("transformer_block", by_sub)

        transformer_data = by_sub["transformer_block"]
        self.assertIn("nested_breakdown", transformer_data)

        nested = transformer_data["nested_breakdown"]
        self.assertIn("attention", nested)
        self.assertIn("moe", nested)

    def test_qwen35_dense_has_attention_ffn_breakdown(self):
        """测试 Qwen3.5 Dense transformer_block 有 attention + ffn 分解."""
        from llm_perf.modeling.qwen3_5 import Qwen3_5Model

        model = Qwen3_5Model(
            vocab_size=248320,
            hidden_size=2048,
            num_layers=4,
            num_heads=16,
            intermediate_size=6144,
        )
        device = Device.from_preset("H100-SXM-80GB")
        topology = NetworkTopology.create_2tier_simple(
            inter_node_bw_gbps=100, intra_node_bw_gbps=200
        )
        cluster = Cluster.create_homogeneous(
            device.config, num_devices=8, topology=topology
        )
        strategy = StrategyConfig(tp_degree=8, pp_degree=1, dp_degree=1)
        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            "llm-inference", batch_size=1, prompt_len=512, generation_len=128
        )

        detailed = result.detailed_breakdown
        by_sub = detailed.get("by_submodule_type", {})
        self.assertIn("transformer_block", by_sub)

        transformer_data = by_sub["transformer_block"]
        self.assertIn("nested_breakdown", transformer_data)

        nested = transformer_data["nested_breakdown"]
        self.assertIn("attention", nested)
        self.assertIn("ffn", nested)


if __name__ == "__main__":
    unittest.main()