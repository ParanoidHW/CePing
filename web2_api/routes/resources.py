"""Resources API routes.

Endpoints:
- GET /api/resources - List all hardware resources (devices + topologies)

Design: Routes only do HTTP adaptation, combining hardware and topology info.
"""

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify

from llm_perf.hardware.device import Device

logger = logging.getLogger(__name__)

resources_bp = Blueprint("resources", __name__)

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


@resources_bp.route("/resources", methods=["GET"])
def list_resources() -> Dict[str, Any]:
    """List all hardware resources (devices + topologies).

    Returns:
        {
            "devices": {
                "NVIDIA": [...],
                "AMD": [...],
                "Huawei": [...]
            },
            "device_details": {...},
            "topologies": [...],
            "topology_methods": {...}
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
        "topologies": list(TOPOLOGY_TYPES.keys()),
        "topology_methods": TOPOLOGY_TYPES,
    })