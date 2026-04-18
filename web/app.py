#!/usr/bin/env python3
"""
LLM Performance Evaluator - Web Service
A local HTTPS web interface for interactive performance evaluation.
"""

import ipaddress
import ssl
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_perf.analyzer import UnifiedAnalyzer, get_workload, infer_workload, list_workloads
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.device import Device
from llm_perf.hardware.topology import NetworkTopology
from llm_perf.modeling import create_model_from_config, get_model_presets, get_presets_by_sparse_type
from llm_perf.strategy.base import StrategyConfig

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


def create_model_from_registry(model_type: str, config: dict):
    """Create a model instance using the registry."""
    full_config = dict(config)
    full_config.setdefault("type", model_type)
    return create_model_from_config(full_config)


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

        model_config = data.get("model", {})
        model_type = model_config.get("type", "llama")
        model = create_model_from_registry(model_type, model_config)

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

        workload = data.get("workload")
        if not workload:
            mode = data.get("mode", "inference")
            workload = infer_workload(model_type, mode)

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

        result = analyzer.analyze(workload, **params)

        return jsonify(
            {
                "success": True,
                "result": result.to_dict(),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/evaluate/training", methods=["POST"])
def evaluate_training():
    """Evaluate training performance (legacy endpoint)."""
    try:
        data = request.json

        model_config = data["model"]
        model_type = model_config.get("type", "llama")
        model = create_model_from_registry(model_type, model_config)

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

        analyzer = UnifiedAnalyzer(model, device, cluster, strategy)
        training_params = dict(data["training"])
        result = analyzer.analyze(
            workload,
            batch_size=training_params.get("batch_size", 1),
            **{k: v for k, v in training_params.items() if k != "batch_size"},
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
    """Evaluate inference performance (legacy endpoint)."""
    try:
        data = request.json

        model_config = data["model"]
        model_type = model_config.get("type", "llama")
        model = create_model_from_registry(model_type, model_config)

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
            model = create_model_from_registry(model_type, model_config)

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
