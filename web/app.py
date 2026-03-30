#!/usr/bin/env python3
"""
LLM Performance Evaluator - Web Service
A local HTTPS web interface for interactive performance evaluation.
"""

import os
import sys
import json
import ssl
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_perf.models.llama import LlamaConfig, LlamaModel
from llm_perf.models.moe import MoEConfig, MoEModel
from llm_perf.hardware.device import Device
from llm_perf.hardware.cluster import Cluster
from llm_perf.hardware.topology import (
    NetworkTopology, 
    TopologyType,
    ClosTopologyBuilder
)
from llm_perf.strategy.base import StrategyConfig
from llm_perf.analyzer.training import TrainingAnalyzer
from llm_perf.analyzer.inference import InferenceAnalyzer

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)

# ==================== Data Models ====================

DEVICE_PRESETS = {
    "NVIDIA": ["H100-SXM-80GB", "H100-NVL-94GB", "H200-SXM-141GB", 
               "A100-SXM-80GB", "A100-SXM-40GB", "L40S"],
    "AMD": ["MI300X"],
    "Huawei": ["Ascend-910C", "Ascend-910B2", "Ascend-910B3", 
               "Ascend-950-DT", "Ascend-950-PR", "Ascend-960", "Ascend-970"]
}

TOPOLOGY_TYPES = {
    "2-Tier Simple": "create_2tier_simple",
    "3-Tier Clos": "create_clos_3tier", 
    "Fat-Tree": "create_fat_tree",
    "CloudMatrix Supernode": "create_cloudmatrix_supernode"
}

# ==================== API Routes ====================

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get available device presets."""
    return jsonify({
        "devices": DEVICE_PRESETS,
        "device_info": {name: Device.from_preset(name).to_dict() 
                       for vendor in DEVICE_PRESETS.values() 
                       for name in vendor}
    })


@app.route('/api/model/presets', methods=['GET'])
def get_model_presets():
    """Get model presets."""
    presets = {
        "llama-7b": {
            "type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "max_seq_len": 4096,
            "dtype": "fp16"
        },
        "llama-70b": {
            "type": "llama",
            "vocab_size": 32000,
            "hidden_size": 8192,
            "num_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
            "max_seq_len": 4096,
            "dtype": "fp16"
        },
        "mixtral-8x7b": {
            "type": "moe",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 8,
            "num_experts_per_token": 2
        },
        "deepseek-v3": {
            "type": "moe",
            "vocab_size": 32000,
            "hidden_size": 7168,
            "num_layers": 61,
            "num_attention_heads": 64,
            "intermediate_size": 18432,
            "max_seq_len": 32768,
            "dtype": "fp16",
            "num_experts": 256,
            "num_experts_per_token": 8
        }
    }
    return jsonify(presets)


@app.route('/api/topology/presets', methods=['GET'])
def get_topology_presets():
    """Get topology presets."""
    return jsonify({
        "types": list(TOPOLOGY_TYPES.keys()),
        "defaults": {
            "2-Tier Simple": {
                "intra_node_bw_gbps": 900,
                "inter_node_bw_gbps": 200
            },
            "3-Tier Clos": {
                "node_bw_gbps": 900,
                "rack_bw_gbps": 200,
                "cluster_bw_gbps": 100,
                "devices_per_node": 8,
                "nodes_per_rack": 16
            },
            "Fat-Tree": {
                "core_bw_gbps": 100,
                "agg_bw_gbps": 400,
                "edge_bw_gbps": 800,
                "oversubscription": 4.0
            },
            "CloudMatrix Supernode": {
                "num_npus": 384,
                "ub_bw_gbps": 3136,
                "ub_latency_us": 2.0,
                "rdma_bw_gbps": 400
            }
        }
    })


@app.route('/api/evaluate/training', methods=['POST'])
def evaluate_training():
    """Evaluate training performance."""
    try:
        data = request.json
        
        # Create model
        model_config = data['model']
        if model_config['type'] == 'llama':
            cfg = LlamaConfig(**{k: v for k, v in model_config.items() if k != 'type'})
            model = LlamaModel(cfg)
        else:
            cfg = MoEConfig(**{k: v for k, v in model_config.items() if k != 'type'})
            model = MoEModel(cfg)
        
        # Create device and cluster
        device = Device.from_preset(data['device'])
        topology_config = data['topology']
        topology_type = topology_config['type']
        
        if topology_type == "2-Tier Simple":
            topology = NetworkTopology.create_2tier_simple(
                topology_config['intra_node_bw_gbps'],
                topology_config.get('inter_node_bw_gbps', 200)
            )
        elif topology_type == "3-Tier Clos":
            topology = NetworkTopology.create_clos_3tier(
                topology_config['node_bw_gbps'],
                topology_config['rack_bw_gbps'],
                topology_config['cluster_bw_gbps']
            )
        elif topology_type == "Fat-Tree":
            topology = NetworkTopology.create_fat_tree(
                topology_config['core_bw_gbps'],
                topology_config['agg_bw_gbps'],
                topology_config['edge_bw_gbps'],
                topology_config.get('oversubscription', 4.0)
            )
        else:  # CloudMatrix
            topology = NetworkTopology.create_cloudmatrix_supernode(
                topology_config.get('num_npus', 384),
                ub_bw_gbps=topology_config.get('ub_bw_gbps', 3136)
            )
        
        cluster = Cluster.create_homogeneous(
            device.config, 
            data['num_devices'],
            topology,
            data.get('devices_per_node', 8)
        )
        
        # Create strategy
        strategy = StrategyConfig(
            tp_degree=data['strategy']['tp'],
            pp_degree=data['strategy']['pp'],
            dp_degree=data['strategy']['dp'],
            ep_degree=data['strategy'].get('ep', 1),
            activation_checkpointing=data['strategy'].get('activation_checkpointing', False),
            zero_stage=data['strategy'].get('zero_stage', 0)
        )
        
        # Analyze
        analyzer = TrainingAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            batch_size=data['training']['batch_size'],
            seq_len=data['training']['seq_len'],
            num_steps=data['training'].get('num_steps', 1000)
        )
        
        return jsonify({
            "success": True,
            "result": {
                "samples_per_sec": result.samples_per_sec,
                "tokens_per_sec": result.tokens_per_sec,
                "time_per_step_sec": result.time_per_step_sec,
                "memory_per_gpu_gb": result.memory_per_gpu_gb,
                "breakdown": result.breakdown.to_dict() if result.breakdown else None
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/evaluate/inference', methods=['POST'])
def evaluate_inference():
    """Evaluate inference performance."""
    try:
        data = request.json
        
        # Create model
        model_config = data['model']
        if model_config['type'] == 'llama':
            cfg = LlamaConfig(**{k: v for k, v in model_config.items() if k != 'type'})
            model = LlamaModel(cfg)
        else:
            cfg = MoEConfig(**{k: v for k, v in model_config.items() if k != 'type'})
            model = MoEModel(cfg)
        
        # Create device and cluster
        device = Device.from_preset(data['device'])
        topology_config = data['topology']
        topology_type = topology_config['type']
        
        if topology_type == "2-Tier Simple":
            topology = NetworkTopology.create_2tier_simple(
                topology_config['intra_node_bw_gbps'],
                topology_config.get('inter_node_bw_gbps', 200)
            )
        elif topology_type == "3-Tier Clos":
            topology = NetworkTopology.create_clos_3tier(
                topology_config['node_bw_gbps'],
                topology_config['rack_bw_gbps'],
                topology_config['cluster_bw_gbps']
            )
        elif topology_type == "Fat-Tree":
            topology = NetworkTopology.create_fat_tree(
                topology_config['core_bw_gbps'],
                topology_config['agg_bw_gbps'],
                topology_config['edge_bw_gbps'],
                topology_config.get('oversubscription', 4.0)
            )
        else:  # CloudMatrix
            topology = NetworkTopology.create_cloudmatrix_supernode(
                topology_config.get('num_npus', 384),
                ub_bw_gbps=topology_config.get('ub_bw_gbps', 3136)
            )
        
        cluster = Cluster.create_homogeneous(
            device.config,
            data['num_devices'],
            topology,
            data.get('devices_per_node', 8)
        )
        
        # Create strategy
        strategy = StrategyConfig(
            tp_degree=data['strategy']['tp'],
            pp_degree=data['strategy']['pp'],
            dp_degree=data['strategy']['dp'],
            ep_degree=data['strategy'].get('ep', 1)
        )
        
        # Analyze
        analyzer = InferenceAnalyzer(model, device, cluster, strategy)
        result = analyzer.analyze(
            batch_size=data['inference']['batch_size'],
            prompt_len=data['inference']['prompt_len'],
            generation_len=data['inference']['generation_len']
        )
        
        return jsonify({
            "success": True,
            "result": {
                "prefill_time_sec": result.prefill_time_sec,
                "decode_time_per_step_sec": result.decode_time_per_step_sec,
                "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
                "decode_tokens_per_sec": result.decode_tokens_per_sec,
                "total_time_sec": result.total_time_sec,
                "memory_per_gpu_gb": result.memory_per_gpu_gb,
                "kv_cache_memory_gb": result.kv_cache_memory_gb
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== Main Entry ====================

def create_ssl_context():
    """Create SSL context with local certificates."""
    cert_dir = Path(__file__).parent / 'certs'
    cert_file = cert_dir / 'server.crt'
    key_file = cert_dir / 'server.key'
    
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
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LLM Perf Evaluator"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]),
        critical=False
    ).sign(key, hashes.SHA256())
    
    # Write files
    cert_file.parent.mkdir(parents=True, exist_ok=True)
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))
    print(f"Certificates generated: {cert_file}, {key_file}")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='LLM Performance Evaluator Web Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8443, help='Port to bind (default: 8443)')
    parser.add_argument('--http', action='store_true', help='Use HTTP instead of HTTPS')
    args = parser.parse_args()
    
    if args.http:
        print(f"Starting HTTP server on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        ssl_context = create_ssl_context()
        print(f"Starting HTTPS server on https://{args.host}:{args.port}")
        print("Note: You may need to accept the self-signed certificate in your browser")
        app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=False)


if __name__ == '__main__':
    main()
