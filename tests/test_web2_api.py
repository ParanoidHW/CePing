"""Tests for web2_api module.

Test coverage:
- GET /api/workloads - list workload categories
- GET /api/workload/<name> - get workload schema
- GET /api/models - list models
- GET /api/model/<name> - get model schema
- GET /api/hardware - list hardware presets
- GET /api/hardware/<name> - get device details
- GET /api/resources - combined resources
- POST /api/evaluate - evaluation request
"""

import pytest
from flask import Flask
from flask.testing import FlaskClient

from web2_api.app import create_app


@pytest.fixture
def app() -> Flask:
    """Create Flask app for testing."""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create test client."""
    return app.test_client()


class TestWorkloadsAPI:
    """Test workload API endpoints."""

    def test_list_workloads(self, client: FlaskClient) -> None:
        """Test GET /api/workloads returns categories."""
        response = client.get("/api/workloads")
        assert response.status_code == 200

        data = response.get_json()
        assert "categories" in data
        assert "total" in data
        assert data["total"] > 0

        categories = data["categories"]
        assert "training" in categories
        assert "inference" in categories

    def test_get_workload_schema(self, client: FlaskClient) -> None:
        """Test GET /api/workload/<name> returns schema."""
        response = client.get("/api/workload/inference/autoregressive")
        assert response.status_code == 200

        data = response.get_json()
        assert "name" in data
        assert "workload_name" in data
        assert "description" in data
        assert "category" in data
        assert "stages" in data
        assert "parameters" in data

        assert data["workload_name"] == "inference/autoregressive"
        assert data["category"] == "inference"

    def test_get_workload_not_found(self, client: FlaskClient) -> None:
        """Test GET /api/workload/<invalid> returns 404."""
        response = client.get("/api/workload/invalid/workload")
        assert response.status_code == 404

        data = response.get_json()
        assert "error" in data
        assert data["error"] == "Workload not found"


class TestModelsAPI:
    """Test model API endpoints."""

    def test_list_models(self, client: FlaskClient) -> None:
        """Test GET /api/models returns models."""
        response = client.get("/api/models")
        assert response.status_code == 200

        data = response.get_json()
        assert "models" in data
        assert "total" in data
        assert data["total"] > 0

        models = data["models"]
        assert len(models) > 0
        assert "name" in models[0]
        assert "architecture" in models[0]

    def test_list_models_with_workload_filter(self, client: FlaskClient) -> None:
        """Test GET /api/models?workload=training filters models."""
        response = client.get("/api/models?workload=training")
        assert response.status_code == 200

        data = response.get_json()
        assert "models" in data
        assert "total" in data

    def test_get_model_schema(self, client: FlaskClient) -> None:
        """Test GET /api/model/<name> returns schema."""
        response = client.get("/api/model/llama-7b")
        assert response.status_code == 200

        data = response.get_json()
        assert "name" in data
        assert "description" in data
        assert "architecture" in data
        assert "config" in data

        assert data["name"] == "llama-7b"

    def test_get_model_not_found(self, client: FlaskClient) -> None:
        """Test GET /api/model/<invalid> returns 404."""
        response = client.get("/api/model/invalid-model")
        assert response.status_code == 404

        data = response.get_json()
        assert "error" in data
        assert data["error"] == "Model not found"


class TestHardwareAPI:
    """Test hardware API endpoints."""

    def test_list_hardware(self, client: FlaskClient) -> None:
        """Test GET /api/hardware returns presets."""
        response = client.get("/api/hardware")
        assert response.status_code == 200

        data = response.get_json()
        assert "devices" in data
        assert "device_details" in data

        devices = data["devices"]
        assert "NVIDIA" in devices
        assert "Huawei" in devices
        assert len(devices["NVIDIA"]) > 0

    def test_get_hardware_details(self, client: FlaskClient) -> None:
        """Test GET /api/hardware/<name> returns device config."""
        response = client.get("/api/hardware/H100-SXM-80GB")
        assert response.status_code == 200

        data = response.get_json()
        assert "name" in data
        assert "memory_gb" in data
        assert "memory_bandwidth_gbps" in data

        assert data["name"] == "H100-SXM-80GB"

    def test_get_hardware_not_found(self, client: FlaskClient) -> None:
        """Test GET /api/hardware/<invalid> returns 404."""
        response = client.get("/api/hardware/invalid-device")
        assert response.status_code == 404

        data = response.get_json()
        assert "error" in data

    def test_list_topologies(self, client: FlaskClient) -> None:
        """Test GET /api/hardware/topologies returns topology types."""
        response = client.get("/api/hardware/topologies")
        assert response.status_code == 200

        data = response.get_json()
        assert "topologies" in data
        assert "topology_methods" in data

        topologies = data["topologies"]
        assert "2-Tier Simple" in topologies
        assert "3-Tier Clos" in topologies


class TestResourcesAPI:
    """Test resources API endpoints."""

    def test_list_resources(self, client: FlaskClient) -> None:
        """Test GET /api/resources returns combined resources."""
        response = client.get("/api/resources")
        assert response.status_code == 200

        data = response.get_json()
        assert "devices" in data
        assert "device_details" in data
        assert "topologies" in data
        assert "topology_methods" in data


class TestEvaluateAPI:
    """Test evaluate API endpoints."""

    def test_evaluate_invalid_workload(self, client: FlaskClient) -> None:
        """Test POST /api/evaluate with invalid workload returns 400."""
        response = client.post(
            "/api/evaluate",
            json={
                "workload_name": "invalid/workload",
                "model_name": "llama-7b",
                "hardware": {"device_preset": "H100-SXM-80GB", "num_devices": 1},
                "strategy": {"tp_degree": 1, "pp_degree": 1, "dp_degree": 1},
                "params": {"batch_size": 1},
            },
        )
        assert response.status_code == 400

        data = response.get_json()
        assert data["success"] is False
        assert "error" in data

    def test_evaluate_invalid_model(self, client: FlaskClient) -> None:
        """Test POST /api/evaluate with invalid model returns 400."""
        response = client.post(
            "/api/evaluate",
            json={
                "workload_name": "inference/autoregressive",
                "model_name": "invalid-model",
                "hardware": {"device_preset": "H100-SXM-80GB", "num_devices": 1},
                "strategy": {"tp_degree": 1, "pp_degree": 1, "dp_degree": 1},
                "params": {"batch_size": 1},
            },
        )
        assert response.status_code == 400

        data = response.get_json()
        assert data["success"] is False
        assert "error" in data

    @pytest.mark.skip(reason="Requires full evaluation setup, test in integration")
    def test_evaluate_success(self, client: FlaskClient) -> None:
        """Test POST /api/evaluate returns result (integration test)."""
        pass