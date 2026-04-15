#!/usr/bin/env python3
"""
LLM Performance Evaluator - Web Service
A local HTTPS web interface for interactive performance evaluation.
"""

import sys
import ssl
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ipaddress

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import registries - this registers all models and pipelines
from llm_perf.core.registry import ModelRegistry, PipelineRegistry
from llm_perf.pipelines.registry import get_pipeline_presets
from llm_perf.models.registry import get_model_presets, get_presets_by_sparse_type, create_model_from_config

# Import base classes for type checking
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import (
    NetworkTopology,
)
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.training import TrainingAnalyzer
from llm_perf.analyzer.inference import InferenceAnalyzer

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
CORS(app)

# ==================== Data Models ====================

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

# Initialize registries
model_registry = ModelRegistry()
pipeline_registry = PipelineRegistry()


# ==================== Helper Functions ====================


def create_topology(topology_config: dict) -> NetworkTopology:
    """Create network topology from configuration.

    Args:
        topology_config: Topology configuration dictionary

    Returns:
        NetworkTopology instance
    """
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
    else:  # CloudMatrix
        return NetworkTopology.create_cloudmatrix_supernode(
            topology_config.get("num_npus", 384),
            ub_bw_gbps=topology_config.get("ub_bw_gbps", 3136),
        )


def create_model_from_registry(model_type: str, config: dict):
    """Create a model instance using the ModelRegistry.

    Uses the unified create_model_from_config factory function.

    Args:
        model_type: Model type identifier
        config: Model configuration

    Returns:
        Model instance
    """
    # Add type to config if not present
    full_config = dict(config)
    full_config.setdefault("type", model_type)
    return create_model_from_config(full_config)


# ==================== API Routes ====================


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/api/devices", methods=["GET"])
def get_devices():
    """Get available device presets."""
    return jsonify(
        {
            "devices": DEVICE_PRESETS,
            "device_info": {
                name: Device.from_preset(name).to_dict()
                for vendor in DEVICE_PRESETS.values()
                for name in vendor
            },
        }
    )




@app.route("/api/models", methods=["GET"])
def get_registered_models():
    """Get all registered models from ModelRegistry.

    Returns:
        JSON with all registered models grouped by sparse_type and architecture
    """
    return jsonify(
        {
            "models": model_registry.to_dict(),
            "by_sparse_type": model_registry.list_by_sparse_type(),
            "by_architecture": model_registry.list_by_architecture(),
            "categories": model_registry.list_by_category(),  # Backward compatibility
        }
    )


@app.route("/api/model/presets", methods=["GET"])
def get_model_presets_endpoint():
    """Get model presets grouped by sparse_type for UI display.

    Returns:
        JSON with presets grouped by: dense, sparse_standard_moe, sparse_deepseek_moe
    """
    return jsonify({
        "presets": get_model_presets(),
        "by_sparse_type": get_presets_by_sparse_type(),
    })


@app.route("/api/models/refresh", methods=["POST"])
def refresh_models():
    """Refresh model registry."""
    try:
        from llm_perf.models import registry as _registry_module  # noqa: F401
        return jsonify({
            "success": True,
            "models": model_registry.to_dict(),
            "by_sparse_type": model_registry.list_by_sparse_type(),
            "by_architecture": model_registry.list_by_architecture(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pipelines", methods=["GET"])
def get_registered_pipelines():
    """Get all registered pipelines."""
    return jsonify({
        "pipelines": pipeline_registry.to_dict(),
        "presets": get_pipeline_presets(),
    })


@app.route("/api/pipelines/refresh", methods=["POST"])
def refresh_pipelines():
    """Refresh pipeline registry."""
    try:
        from llm_perf.pipelines import registry as _pipeline_registry_module  # noqa: F401
        return jsonify({
            "success": True,
            "pipelines": pipeline_registry.to_dict(),
            "presets": get_pipeline_presets(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/topology/presets", methods=["GET"])
def get_topology_presets():
    """Get topology presets."""
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


@app.route("/api/evaluate/training", methods=["POST"])
def evaluate_training():
    """Evaluate training performance using ModelRegistry."""
    try:
        data = request.json

        # Create model using registry
        model_config = data["model"]
        model_type = model_config.get("type", "llama")
        model = create_model_from_registry(model_type, model_config)

        # Create device and cluster
        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        # Create strategy
        strategy = StrategyConfig(
            tp_degree=data["strategy"]["tp"],
            pp_degree=data["strategy"]["pp"],
            dp_degree=data["strategy"]["dp"],
            ep_degree=data["strategy"].get("ep", 1),
            activation_checkpointing=data["strategy"].get("activation_checkpointing", False),
            zero_stage=data["strategy"].get("zero_stage", 0),
        )

        # Analyze
        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            batch_size=data["training"]["batch_size"],
            seq_len=data["training"]["seq_len"],
            num_steps=data["training"].get("num_steps", 1000),
        )

        return jsonify(
            {
                "success": True,
                "result": result.to_dict(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluate/inference", methods=["POST"])
def evaluate_inference():
    """Evaluate inference performance using ModelRegistry."""
    try:
        data = request.json

        # Create model using registry
        model_config = data["model"]
        model_type = model_config.get("type", "llama")
        model = create_model_from_registry(model_type, model_config)

        # Create device and cluster
        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        # Create strategy
        strategy = StrategyConfig(
            tp_degree=data["strategy"]["tp"],
            pp_degree=data["strategy"]["pp"],
            dp_degree=data["strategy"]["dp"],
            ep_degree=data["strategy"].get("ep", 1),
        )

        # Analyze
        analyzer = InferenceAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
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
    """Evaluate a registered pipeline.

    Args:
        pipeline_name: Name of the registered pipeline

    Returns:
        Pipeline evaluation results
    """
    try:
        data = request.json

        # Check if pipeline is registered
        if not pipeline_registry.is_registered(pipeline_name):
            return jsonify(
                {"success": False, "error": f"Pipeline '{pipeline_name}' not found"}
            ), 404

        # Create device and cluster
        device = Device.from_preset(data["device"])
        topology = create_topology(data["topology"])

        cluster = Cluster.create_homogeneous(
            device.config,
            data["num_devices"],
            topology,
            data.get("devices_per_node", 8),
        )

        # Create strategy
        strategy = StrategyConfig(
            tp_degree=data["strategy"]["tp"],
            pp_degree=data["strategy"]["pp"],
            dp_degree=data["strategy"]["dp"],
            ep_degree=data["strategy"].get("ep", 1),
        )

        # For video generation pipeline
        if pipeline_name == "diffusion-video":
            from llm_perf.pipelines.diffusion_video import create_wan_t2v_pipeline

            pipeline = create_wan_t2v_pipeline(
                device=device,
                cluster=cluster,
                strategy=strategy,
                num_frames=data.get("num_frames", 81),
                height=data.get("height", 720),
                width=data.get("width", 1280),
                num_inference_steps=data.get("num_inference_steps", 50),
                dtype=data.get("dtype", "bf16"),
            )

            result = pipeline.run(
                {
                    "num_frames": data.get("num_frames", 81),
                    "height": data.get("height", 720),
                    "width": data.get("width", 1280),
                    "use_cfg": data.get("use_cfg", True),
                }
            )

            return jsonify(
                {
                    "success": True,
                    "result": result.to_dict(),
                }
            )

        # For inference pipeline
        elif pipeline_name == "inference":
            from llm_perf.pipelines.base import InferencePipeline

            # Create model
            model_config = data.get("model", {})
            model_type = model_config.get("type", "llama")
            model = create_model_from_registry(model_type, model_config)

            pipeline = InferencePipeline(
                model=model,
                device=device,
                cluster=cluster,
                strategy=strategy,
                batch_size=data.get("batch_size", 1),
                prompt_len=data.get("prompt_len", 512),
                generation_len=data.get("generation_len", 128),
            )

            result = pipeline.run()

            return jsonify(
                {
                    "success": True,
                    "result": {
                        "total_time_sec": result.total_time_sec,
                        "step_times": result.step_times,
                        "step_results": result.step_results,
                        "memory_peak_gb": result.memory_peak_gb,
                        "throughput": result.throughput,
                    },
                }
            )

        else:
            return jsonify(
                {
                    "success": False,
                    "error": f"Evaluation not implemented for pipeline '{pipeline_name}'",
                }
            ), 501

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== Main Entry ====================


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
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    # Generate private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Generate certificate
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

    # Write files
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
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8443, help="Port to bind (default: 8443)"
    )
    parser.add_argument("--http", action="store_true", help="Use HTTP instead of HTTPS")
    args = parser.parse_args()

    if args.http:
        print(f"Starting HTTP server on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        ssl_context = create_ssl_context()
        print(f"Starting HTTPS server on https://{args.host}:{args.port}")
        print("Note: You may need to accept the self-signed certificate in your browser")
        app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=False)


if __name__ == "__main__":
    main()
