"""Hardware API routes.

Endpoints:
- GET /api/hardware - List hardware device presets
- GET /api/hardware/<device_name> - Get device details
- GET /api/hardware/topologies - List topology types

Design: Routes only do HTTP adaptation, all logic is in llm_perf/hardware.
"""

import logging
from typing import Any, Dict, List

from flask import Blueprint, jsonify

from llm_perf.hardware.device import Device

logger = logging.getLogger(__name__)

hardware_bp = Blueprint("hardware", __name__)

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


@hardware_bp.route("/hardware", methods=["GET"])
def list_hardware() -> Dict[str, Any]:
    """List hardware device presets by vendor.

    Returns:
        {
            "devices": {
                "NVIDIA": ["H100-SXM-80GB", ...],
                "AMD": ["MI300X"],
                "Huawei": ["Ascend-910C", ...]
            },
            "device_details": {...}
        }
    """
    device_details = {}
    for vendor_devices in DEVICE_PRESETS.values():
        for name in vendor_devices:
            try:
                device = Device.from_preset(name)
                device_details[name] = device.to_dict()
            except Exception as e:
                logger.warning(f"Failed to load device {name}: {e}")

    return jsonify({
        "devices": DEVICE_PRESETS,
        "device_details": device_details,
    })


@hardware_bp.route("/hardware/<device_name>", methods=["GET"])
def get_hardware(device_name: str) -> Dict[str, Any]:
    """Get device details.

    Args:
        device_name: Like "H100-SXM-80GB", "Ascend-910C"

    Returns:
        Device config dict with:
        - name, memory_gb, memory_bandwidth_gbps
        - fp16_tflops_cube, bf16_tflops_cube, etc.
        - nvlink_bandwidth_gbps, hccs_bandwidth_gbps
    """
    try:
        device = Device.from_preset(device_name)
        return jsonify(device.to_dict())
    except Exception as e:
        logger.error(f"Failed to load device {device_name}: {e}")
        return jsonify({
            "error": "Device not found",
            "device_name": device_name,
            "message": str(e),
        }), 404


@hardware_bp.route("/hardware/topologies", methods=["GET"])
def list_topologies() -> Dict[str, Any]:
    """List topology types.

    Returns:
        {
            "topologies": ["2-Tier Simple", "3-Tier Clos", ...],
            "topology_details": {...}
        }
    """
    return jsonify({
        "topologies": list(TOPOLOGY_TYPES.keys()),
        "topology_methods": TOPOLOGY_TYPES,
    })