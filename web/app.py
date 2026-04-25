#!/usr/bin/env python3
"""
LLM Performance Evaluator - Web Service
A local HTTPS web interface for interactive performance evaluation.
"""

import ipaddress
import logging
import os
import ssl
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_perf.analyzer import UnifiedAnalyzer, get_workload, infer_workload, list_workloads
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.device import Device
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.modeling import create_model_from_config, get_model_presets, get_presets_by_sparse_type, get_presets_by_workload, get_presets_by_workload_grouped
from llm_perf.scenarios import ColocateAnalyzer, ModelAllocation
from llm_perf.strategy.base import StrategyConfig
from llm_perf.strategy.parallel_context import ParallelContext
from llm_perf.validation import validate_all

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
CORS(app)

DEVICE_PRESETS = {
    "NVIDIA": [
        "H100-SXM-80GB",
        "H100-NVL-94GB",
        "H200-SXM-141GB",
        "A100-SXM-80GB",
        "A100-SXM-40GB",
        "L40S",
    ],
    "AMD": ["MI300X"],
    "Huawei": [
        "Ascend-910C",
        "Ascend-910B2",
        "Ascend-910B3",
        "Ascend-950-DT",
        "Ascend-950-PR",
        "Ascend-960",
        "Ascend-970",
    ],
}

TOPOLOGY_TYPES = {
    "2-Tier Simple": "create_2tier_simple",
    "3-Tier Clos": "create_clos_3tier",
    "Fat-Tree": "create_fat_tree",
    "CloudMatrix Supernode": "create_cloudmatrix_supernode",
}


SCENARIO_TO_WORKLOAD_MAP = {
    "training": "training",
    "inference": "autoregressive-inference",
    "diffusion": "diffusion-pipeline",
    "rl_training": "rl-ppo",
    "pd_disagg": "autoregressive-inference",
}


def create_topology(topology_config: dict) -> NetworkTopology:
    """Create network topology from configuration."""
    topology_type = topology_config["type"]

    if topology_type == "2-Tier Simple":
        return NetworkTopology.create_2tier_simple(
            topology_config["intra_node_bw_gbps"],
            topology_config.get("inter_node_bw_gbps", 200),
        )
    elif topology_type == "3-Tier Clos":
        return NetworkTopology.create_clos_3tier(
            topology_config["node_bw_gbps"],
            topology_config["rack_bw_gbps"],
            topology_config["cluster_bw_gbps"],
        )
    elif topology_type == "Fat-Tree":
        return NetworkTopology.create_fat_tree(
            topology_config["core_bw_gbps"],
            topology_config["agg_bw_gbps"],
            topology_config["edge_bw_gbps"],
            topology_config.get("oversubscription", 4.0),
        )
    else:
        return NetworkTopology.create_cloudmatrix_supernode(
            topology_config.get("num_npus", 384),
            ub_bw_gbps=topology_config.get("ub_bw_gbps", 3136),
        )


def create_model_from_registry(model_type: str, config: dict, workload_type: Optional[str] = None):
    """Create a model instance using the registry.
    
    Args:
        model_type: Model type string
        config: Model configuration dict
        workload_type: Workload type (training, inference, diffusion, etc.)
    """
    full_config = dict(config)
    full_config.setdefault("type", model_type)
    return create_model_from_config(full_config, workload_type=workload_type)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/devices", methods=["GET"])
def get_devices():
    return jsonify(
        {
            "devices": DEVICE_PRESETS,
            "device_info": {
                name: Device.from_preset(name).to_dict() for vendor in DEVICE_PRESETS.values() for name in vendor
            },
        }
    )


@app.route("/api/models", methods=["GET"])
def get_registered_models():
    workload_type = request.args.get("workload_type", None)
    
    if workload_type:
        return jsonify({
            "presets": get_presets_by_workload(workload_type),
            "by_sparse_type": get_presets_by_workload_grouped(workload_type),
        })
    
    return jsonify(
        {
            "presets": get_model_presets(),
            "by_sparse_type": get_presets_by_sparse_type(),
        }
    )


@app.route("/api/model/presets", methods=["GET"])
def get_model_presets_endpoint():
    return jsonify(
        {
            "presets": get_model_presets(),
            "by_sparse_type": get_presets_by_sparse_type(),
        }
    )


@app.route("/api/workloads", methods=["GET"])
def get_workloads():
    """Get available workload presets."""
    return jsonify(
        {
            "workloads": list_workloads(),
        }
    )


@app.route("/api/topology/presets", methods=["GET"])
def get_topology_presets():
    return jsonify(
        {
            "types": list(TOPOLOGY_TYPES.keys()),
            "defaults": {
                "2-Tier Simple": {
                    "intra_node_bw_gbps": 900,
                    "inter_node_bw_gbps": 200,
                },
                "3-Tier Clos": {
                    "node_bw_gbps": 900,
                    "rack_bw_gbps": 200,
                    "cluster_bw_gbps": 100,
                    "devices_per_node": 8,
                    "nodes_per_rack": 16,
                },
                "Fat-Tree": {
                    "core_bw_gbps": 100,
                    "agg_bw_gbps": 400,
                    "edge_bw_gbps": 800,
                    "oversubscription": 4.0,
                },
                "CloudMatrix Supernode": {
                    "num_npus": 384,
                    "ub_bw_gbps": 3136,
                    "ub_latency_us": 2.0,
                    "rdma_bw_gbps": 400,
                },
            },
        }
    )


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Unified evaluation endpoint."""
    try:
        data = request.json
        logger.info(
            f"[REQUEST] model={data.get('model')}, device={data.get('device')}, "
            f"strategy={data.get('strategy')}, workload={data.get('workload')}, "
            f"batch_size={data.get('batch_size')}, seq_len={data.get('seq_len')}"
        )

        model_config = data.get("model", {})
        preset_name = model_config.get("preset")
        
        workload_data = data.get("workload")
        workload_type = None
        if workload_data:
            if isinstance(workload_data, str):
                workload_type = "training" if "training" in workload_data else "inference"
            elif isinstance(workload_data, dict):
                scenario = workload_data.get("scenario", "training")
                workload_type = scenario
        
        if preset_name:
            model = create_model_from_registry(preset_name, model_config, workload_type=workload_type)
            model_type = preset_name
        else:
            model_type = model_config.get("type", "llama")
            model = create_model_from_registry(model_type, model_config, workload_type=workload_type)

        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        strategy = StrategyConfig(
            tp_degree=data["strategy"]["tp"],
            pp_degree=data["strategy"]["pp"],
            dp_degree=data["strategy"]["dp"],
            ep_degree=data["strategy"].get("ep", 1),
            activation_checkpointing=data["strategy"].get("activation_checkpointing", False),
            zero_stage=data["strategy"].get("zero_stage", 0),
        )

        workload_data = data.get("workload")
        workload = None
        mode = "inference"

        if workload_data:
            if isinstance(workload_data, str):
                workload = workload_data
                mode = "training" if workload.endswith("-training") or "training" in workload else "inference"
            elif isinstance(workload_data, dict):
                scenario = workload_data.get("scenario", "training")
                workload = SCENARIO_TO_WORKLOAD_MAP.get(scenario, "autoregressive-inference")
                mode = "training" if scenario == "training" or scenario == "rl_training" else "inference"

        if not workload:
            mode = data.get("mode", "inference")
            workload = infer_workload(model_type, mode)

        logger.info(
            f"[PARSED_PARAMS] model_type={model_type}, device={device.config.name}, "
            f"tp={strategy.tp_degree}, pp={strategy.pp_degree}, "
            f"dp={strategy.dp_degree}, ep={strategy.ep_degree}"
        )

        ctx = ParallelContext().build_from_strategy(strategy, cluster)

        seq_len = data.get("seq_len", model_config.get("max_seq_len", 4096))
        batch_size = data.get("batch_size", 1)

        pre_errors = validate_all(
            ctx=ctx,
            num_devices=data["num_devices"],
            vocab_size=model_config.get("vocab_size", 32000),
            hidden_size=model_config.get("hidden_size", 4096),
            num_heads=model_config.get("num_attention_heads", model_config.get("num_heads", 32)),
            intermediate_size=model_config.get("intermediate_size", model_config.get("hidden_size", 4096) * 4),
            seq_len=seq_len,
            num_kv_heads=model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)),
            mode=mode,
            model_type=model_type,
            num_experts=model_config.get("num_experts"),
            global_batch_size=batch_size * strategy.dp_degree,
            micro_batch_size=batch_size,
        )

        if pre_errors.has_errors():
            return jsonify({
                "success": False,
                "error": "配置验证失败",
                "validation": pre_errors.to_dict(),
            }), 400

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)

        params = {}
        if "batch_size" in data:
            params["batch_size"] = data["batch_size"]
        if "seq_len" in data:
            params["seq_len"] = data["seq_len"]
        if "prompt_len" in data:
            params["prompt_len"] = data["prompt_len"]
        if "generation_len" in data:
            params["generation_len"] = data["generation_len"]
        if "num_frames" in data:
            params["num_frames"] = data["num_frames"]
        if "height" in data:
            params["height"] = data["height"]
        if "width" in data:
            params["width"] = data["width"]
        if "num_inference_steps" in data:
            params["num_inference_steps"] = data["num_inference_steps"]
        
        if workload_data and isinstance(workload_data, dict):
            workload_params = workload_data
            scenario = workload_params.get("scenario", "training")

            SCENARIO_PARAM_MAP = {
                "pd_disagg": {
                    "input_tokens": ("input_tokens", 1000),
                    "output_tokens": ("output_tokens", 100),
                    "prefill_devices": ("prefill_devices", 32),
                    "decode_devices": ("decode_devices", 32),
                },
                "rl_training": {
                    "seq_len": ("seq_len", 4096),
                    "num_rollouts": ("num_rollouts", 100),
                    "ppo_epochs": ("ppo_epochs", 4),
                },
                "diffusion": {
                    "generation_mode": ("generation_mode", "T2I"),
                    "diffusion_steps": ("diffusion_steps", 50),
                    "prompt_tokens": ("prompt_tokens", None),
                    "input_image_width": ("input_image_width", None),
                    "input_image_height": ("input_image_height", None),
                    "input_video_frames": ("input_video_frames", None),
                    "input_video_width": ("input_video_width", None),
                    "input_video_height": ("input_video_height", None),
                    "output_image_width": ("output_image_width", None),
                    "output_image_height": ("output_image_height", None),
                    "output_video_frames": ("output_video_frames", None),
                    "output_video_width": ("output_video_width", None),
                    "output_video_height": ("output_video_height", None),
                    "height": ("output_image_height", None),
                    "width": ("output_image_width", None),
                    "num_frames": ("output_video_frames", None),
                    "num_steps": ("diffusion_steps", 50),
                },
                "inference": {
                    "prompt_len": ("input_tokens", 512),
                    "generation_len": ("output_tokens", 100),
                    "generated_tokens": ("output_tokens", 100),
                },
                "training": {
                    "batch_size": ("batch_size", None),
                    "seq_len": ("seq_len", None),
                },
            }

            param_mapping = SCENARIO_PARAM_MAP.get(scenario, {})
            for param_key, (source_key, default) in param_mapping.items():
                value = workload_params.get(source_key, default)
                if value is not None:
                    params[param_key] = value

            if scenario == "pd_disagg":
                if data.get("strategy_prefill"):
                    params["strategy_prefill"] = data["strategy_prefill"]
                if data.get("strategy_decode"):
                    params["strategy_decode"] = data["strategy_decode"]

            if scenario == "rl_training":
                if data.get("strategy_train"):
                    params["strategy_train"] = data["strategy_train"]
                if data.get("strategy_infer"):
                    params["strategy_infer"] = data["strategy_infer"]

            if "batch_size" in workload_params:
                params["batch_size"] = workload_params["batch_size"]

        result = analyzer.analyze(workload, **params)

        response = {
            "success": True,
            "result": result.to_dict(),
        }

        detailed = result.to_dict().get("detailed_breakdown", {})
        memory_by_type = detailed.get("memory", {}).get("by_type", {})
        device_memory_gb = device.config.memory_gb

        post_errors = validate_all(
            ctx=ctx,
            num_devices=data["num_devices"],
            vocab_size=model_config.get("vocab_size", 32000),
            hidden_size=model_config.get("hidden_size", 4096),
            num_heads=model_config.get("num_attention_heads", model_config.get("num_heads", 32)),
            intermediate_size=model_config.get("intermediate_size", model_config.get("hidden_size", 4096) * 4),
            seq_len=seq_len,
            num_kv_heads=model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)),
            weight_memory_gb=memory_by_type.get("weight"),
            activation_memory_gb=memory_by_type.get("activation"),
            device_memory_gb=device_memory_gb,
            gradient_memory_gb=memory_by_type.get("gradient"),
            optimizer_memory_gb=memory_by_type.get("optimizer"),
            mode=mode,
            model_type=model_type,
        )

        if post_errors.has_errors() or post_errors.has_warnings():
            response["validation"] = post_errors.to_dict()

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluate/training", methods=["POST"])
def evaluate_training():
    """Evaluate training performance (legacy endpoint)."""
    try:
        data = request.json

        model_config = data["model"]
        model_type = model_config.get("type", "llama")
        model = create_model_from_registry(model_type, model_config, workload_type="training")

        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        strategy = StrategyConfig(
            tp_degree=data["strategy"]["tp"],
            pp_degree=data["strategy"]["pp"],
            dp_degree=data["strategy"]["dp"],
            ep_degree=data["strategy"].get("ep", 1),
            activation_checkpointing=data["strategy"].get("activation_checkpointing", False),
            zero_stage=data["strategy"].get("zero_stage", 0),
        )

        workload = infer_workload(model_type, "training")

        ctx = ParallelContext().build_from_strategy(strategy, cluster)

        training_params = dict(data["training"])
        seq_len = training_params.get("seq_len", model_config.get("max_seq_len", 4096))
        batch_size = training_params.get("batch_size", 1)

        pre_errors = validate_all(
            ctx=ctx,
            num_devices=data["num_devices"],
            vocab_size=model_config.get("vocab_size", 32000),
            hidden_size=model_config.get("hidden_size", 4096),
            num_heads=model_config.get("num_attention_heads", model_config.get("num_heads", 32)),
            intermediate_size=model_config.get("intermediate_size", model_config.get("hidden_size", 4096) * 4),
            seq_len=seq_len,
            num_kv_heads=model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)),
            mode="training",
            model_type=model_type,
            num_experts=model_config.get("num_experts"),
            global_batch_size=batch_size * strategy.dp_degree,
            micro_batch_size=batch_size,
        )

        if pre_errors.has_errors():
            return jsonify({
                "success": False,
                "error": "配置验证失败",
                "validation": pre_errors.to_dict(),
            }), 400

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            workload,
            batch_size=training_params.get("batch_size", 1),
            **{k: v for k, v in training_params.items() if k != "batch_size"},
        )

        response = {
            "success": True,
            "result": result.to_dict(),
        }

        detailed = result.to_dict().get("detailed_breakdown", {})
        memory_by_type = detailed.get("memory", {}).get("by_type", {})
        device_memory_gb = device.config.memory_gb

        post_errors = validate_all(
            ctx=ctx,
            num_devices=data["num_devices"],
            vocab_size=model_config.get("vocab_size", 32000),
            hidden_size=model_config.get("hidden_size", 4096),
            num_heads=model_config.get("num_attention_heads", model_config.get("num_heads", 32)),
            intermediate_size=model_config.get("intermediate_size", model_config.get("hidden_size", 4096) * 4),
            seq_len=seq_len,
            num_kv_heads=model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)),
            weight_memory_gb=memory_by_type.get("weight"),
            activation_memory_gb=memory_by_type.get("activation"),
            device_memory_gb=device_memory_gb,
            gradient_memory_gb=memory_by_type.get("gradient"),
            optimizer_memory_gb=memory_by_type.get("optimizer"),
            mode="training",
            model_type=model_type,
        )

        if post_errors.has_errors() or post_errors.has_warnings():
            response["validation"] = post_errors.to_dict()

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluate/inference", methods=["POST"])
def evaluate_inference():
    """Evaluate inference performance (legacy endpoint)."""
    try:
        data = request.json

        model_config = data["model"]
        model_type = model_config.get("type", "llama")
        model = create_model_from_registry(model_type, model_config, workload_type="inference")

        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        strategy = StrategyConfig(
            tp_degree=data["strategy"]["tp"],
            pp_degree=data["strategy"]["pp"],
            dp_degree=data["strategy"]["dp"],
            ep_degree=data["strategy"].get("ep", 1),
        )

        workload = infer_workload(model_type, "inference")

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            workload,
            batch_size=data["inference"]["batch_size"],
            prompt_len=data["inference"]["prompt_len"],
            generation_len=data["inference"]["generation_len"],
        )

        return jsonify(
            {
                "success": True,
                "result": result.to_dict(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluate/pipeline/<pipeline_name>", methods=["POST"])
def evaluate_pipeline(pipeline_name: str):
    """Evaluate a registered pipeline."""
    try:
        data = request.json

        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        strategy = StrategyConfig(
            tp_degree=data["strategy"]["tp"],
            pp_degree=data["strategy"]["pp"],
            dp_degree=data["strategy"]["dp"],
            ep_degree=data["strategy"].get("ep", 1),
        )

        if pipeline_name == "diffusion-video":
            models = {
                "encoder": create_model_from_config({"type": "wan-text-encoder"}),
                "backbone": create_model_from_config({"type": "wan-dit"}),
                "decoder": create_model_from_config({"type": "wan-vae"}),
            }

            analyzer = UnifiedAnalyzer(models, device, cluster, strategy)
            workload = get_workload("diffusion-pipeline")
            result = analyzer.analyze(
                workload,
                num_frames=data.get("num_frames", 81),
                height=data.get("height", 720),
                width=data.get("width", 1280),
                num_steps=data.get("num_inference_steps", 50),
                use_cfg=data.get("use_cfg", True),
            )

            return jsonify(
                {
                    "success": True,
                    "result": result.to_dict(),
                }
            )

        elif pipeline_name == "inference":
            model_config = data.get("model", {})
            model_type = model_config.get("type", "llama")
            model = create_model_from_registry(model_type, model_config, workload_type="inference")

            analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
            result = analyzer.analyze(
                "llm-inference",
                batch_size=data.get("batch_size", 1),
                prompt_len=data.get("prompt_len", 512),
                generation_len=data.get("generation_len", 128),
            )

            return jsonify(
                {
                    "success": True,
                    "result": result.to_dict(),
                }
            )

        else:
            return jsonify(
                {
                    "success": False,
                    "error": f"Pipeline '{pipeline_name}' not implemented",
                }
            ), 501

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluate/colocate", methods=["POST"])
def evaluate_colocate():
    """Evaluate colocate scenario."""
    try:
        data = request.json

        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        allocations = []
        for model_config in data.get("models", []):
            model = create_model_from_config(model_config.get("model", {}))
            strategy = StrategyConfig(
                tp_degree=model_config.get("strategy", {}).get("tp", 1),
                pp_degree=model_config.get("strategy", {}).get("pp", 1),
                dp_degree=model_config.get("strategy", {}).get("dp", 1),
            )
            allocations.append(
                ModelAllocation(
                    name=model_config.get("name", "model"),
                    model=model,
                    strategy=strategy,
                    workload=model_config.get("workload", "training"),
                )
            )

        analyzer = ColocateAnalyzer(device, cluster)
        result = analyzer.analyze(allocations)

        return jsonify(
            {
                "success": True,
                "result": result.to_dict(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def create_ssl_context():
    """Create SSL context with local certificates."""
    cert_dir = Path(__file__).parent / "certs"
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"

    if not cert_file.exists() or not key_file.exists():
        print("SSL certificates not found. Generating self-signed certificates...")
        generate_self_signed_cert(cert_file, key_file)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    return context


def generate_self_signed_cert(cert_file: Path, key_file: Path):
    """Generate self-signed SSL certificates."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LLM Perf Evaluator"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_file.parent.mkdir(parents=True, exist_ok=True)
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    print(f"Certificates generated: {cert_file}, {key_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Performance Evaluator Web Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8443, help="Port to bind")
    parser.add_argument("--http", action="store_true", help="Use HTTP instead of HTTPS")
    args = parser.parse_args()

    if args.http:
        print(f"Starting HTTP server on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        ssl_context = create_ssl_context()
        print(f"Starting HTTPS server on https://{args.host}:{args.port}")
        print("Note: You may need to accept the self-signed certificate")
        app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=False)


if __name__ == "__main__":
    main()
