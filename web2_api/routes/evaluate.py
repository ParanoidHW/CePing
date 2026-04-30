"""Evaluate API routes.

Endpoints:
- POST /api/evaluate - Submit evaluation request

Design: Routes parse HTTP request and call llm_perf/workload engine.
All core logic is in EvaluationEngine.
"""

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from llm_perf.workload import (
    EvaluationEngine,
    EvaluationRequest,
    EvaluationResult,
    HardwareSchema,
    StrategySchema,
    get_loader,
    get_workload_registry,
    get_model_registry,
)
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.strategy.base import StrategyConfig
from llm_perf.modeling import create_model_from_config

logger = logging.getLogger(__name__)

evaluate_bp = Blueprint("evaluate", __name__)

TOPOLOGY_TYPES = {
    "2-Tier Simple": "create_2tier_simple",
    "3-Tier Clos": "create_clos_3tier",
    "Fat-Tree": "create_fat_tree",
    "CloudMatrix Supernode": "create_cloudmatrix_supernode",
}


def _create_topology(topology_config: Dict[str, Any]) -> NetworkTopology:
    """Create network topology from config."""
    topology_type = topology_config.get("type", "2-Tier Simple")

    if topology_type == "2-Tier Simple":
        return NetworkTopology.create_2tier_simple(
            topology_config.get("intra_node_bw_gbps", 900),
            topology_config.get("inter_node_bw_gbps", 200),
        )
    elif topology_type == "3-Tier Clos":
        return NetworkTopology.create_clos_3tier(
            topology_config.get("node_bw_gbps", 900),
            topology_config.get("rack_bw_gbps", 200),
            topology_config.get("cluster_bw_gbps", 100),
        )
    elif topology_type == "Fat-Tree":
        return NetworkTopology.create_fat_tree(
            topology_config.get("core_bw_gbps", 100),
            topology_config.get("agg_bw_gbps", 400),
            topology_config.get("edge_bw_gbps", 800),
            topology_config.get("oversubscription", 4.0),
        )
    elif topology_type == "CloudMatrix Supernode":
        return NetworkTopology.create_cloudmatrix_supernode(
            topology_config.get("num_npus", 384),
            topology_config.get("ub_bw_gbps", 3136),
            topology_config.get("ub_latency_us", 2.0),
            topology_config.get("rdma_bw_gbps", 400),
        )
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")


@evaluate_bp.route("/evaluate", methods=["POST"])
def evaluate() -> Dict[str, Any]:
    """Submit evaluation request.

    Request body:
        {
            "workload_name": "inference/autoregressive",
            "model_name": "llama-7b",
            "hardware": {
                "device_preset": "H100-SXM-80GB",
                "num_devices": 8,
                "topology": {...}
            },
            "strategy": {
                "tp_degree": 8,
                "pp_degree": 1,
                "dp_degree": 1,
                "ep_degree": 1,
                "sp_degree": 1,
                "activation_checkpointing": false,
                "zero_stage": 0
            },
            "params": {
                "batch_size": 32,
                "seq_len": 4096
            }
        }

    Returns:
        {
            "success": true,
            "validation": {...},
            "result": {...}
        }
    """
    try:
        data = request.json
        logger.info(f"Evaluate request: workload={data.get('workload_name')}, model={data.get('model_name')}")

        workload_name = data.get("workload_name")
        model_name = data.get("model_name")

        workload_registry = get_workload_registry()
        model_registry = get_model_registry()

        if not workload_registry.is_valid_workload(workload_name):
            return jsonify({
                "success": False,
                "error": "Invalid workload",
                "workload_name": workload_name,
            }), 400

        if not model_registry.is_valid_model(model_name):
            return jsonify({
                "success": False,
                "error": "Invalid model",
                "model_name": model_name,
            }), 400

        hardware_data = data.get("hardware", {})
        strategy_data = data.get("strategy", {})
        params = data.get("params", {})

        hardware = HardwareSchema(
            device_preset=hardware_data.get("device_preset", "H100-SXM-80GB"),
            num_devices=hardware_data.get("num_devices", 1),
            topology_type=hardware_data.get("topology", {}).get("type", "2-Tier Simple"),
            custom_topology=hardware_data.get("topology"),
        )

        strategy = StrategySchema(
            tp_degree=strategy_data.get("tp_degree", 1),
            pp_degree=strategy_data.get("pp_degree", 1),
            dp_degree=strategy_data.get("dp_degree", 1),
            ep_degree=strategy_data.get("ep_degree", 1),
            sp_degree=strategy_data.get("sp_degree", 1),
            activation_checkpointing=strategy_data.get("activation_checkpointing", False),
            zero_stage=strategy_data.get("zero_stage", 0),
        )

        request_obj = EvaluationRequest(
            workload_name=workload_name,
            model_name=model_name,
            hardware=hardware,
            strategy=strategy,
            params=params,
        )

        loader = get_loader()
        model_config = loader.load_model_yaml(model_name)

        workload_config = loader.load_workload_yaml(workload_name)
        workload_type = workload_config.get("workload_type", "inference")

        model = create_model_from_config(model_config.get("config", {}))

        engine = EvaluationEngine()
        result = engine.evaluate(request_obj, model)

        return jsonify(result.to_dict())

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500