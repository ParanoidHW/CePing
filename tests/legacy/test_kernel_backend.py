"""Tests for Kernel Backend Layer."""

import pytest
import json
import tempfile
import os

from llm_perf.kernels.backend.base import KernelBackend, BackendConfig
from llm_perf.kernels.backend.theory import TheoryBackend
from llm_perf.kernels.backend.profiling import ProfilingBackend
from llm_perf.kernels.backend.microarch import MicroarchBackend
from llm_perf.kernels.backend.registry import KernelBackendRegistry
from llm_perf.kernels.functional import linear
from llm_perf.hardware.device import Device


class TestTheoryBackend:
    """Tests for TheoryBackend."""

    @pytest.fixture
    def device(self):
        return Device.from_preset("H100-SXM-80GB")

    @pytest.fixture
    def backend(self, device):
        config = BackendConfig(name="theory", device=device)
        backend = TheoryBackend(config)
        backend.initialize()
        return backend

    def test_backend_initialization(self, backend):
        assert backend.is_initialized()
        assert backend.name == "theory"
        assert backend.get_backend_type() == "theory"

    def test_estimate_linear_compute_time(self, backend, device):
        input_shapes = [(4096,), (5120, 4096)]
        output_shape = (5120,)
        time = backend.estimate_compute_time("linear", input_shapes, output_shape, "fp16", device)
        assert time > 0
        assert time < 1.0

    def test_estimate_attention_compute_time(self, backend, device):
        input_shapes = [
            (1, 32, 4096, 128),
            (1, 32, 4096, 128),
            (1, 32, 4096, 128),
        ]
        output_shape = (1, 32, 4096, 128)
        time = backend.estimate_compute_time("attention", input_shapes, output_shape, "fp16", device, is_causal=True)
        assert time > 0

    def test_estimate_comm_time_allreduce(self, backend):
        time = backend.estimate_comm_time("allreduce", 1024 * 1024 * 1024, 8, 100.0)
        assert time > 0

    def test_estimate_memory(self, backend):
        input_shapes = [(4096,), (5120, 4096)]
        memory = backend.estimate_memory("linear", input_shapes, (5120,), "fp16")
        assert memory > 0

    def test_compute_time_from_result(self, backend, device):
        result = linear((4096,), (5120, 4096), dtype="fp16")
        time = backend.estimate_compute_time_from_result(result, device)
        assert time > 0
        assert time < 1.0

    def test_training_time_from_result(self, backend, device):
        result = linear((4096,), (5120, 4096), dtype="fp16")
        time = backend.estimate_training_time_from_result(result, device)
        assert time > 0

    def test_kernel_result_computation(self, backend):
        result = backend.compute_kernel_result("linear", [(4096,), (5120, 4096)], "fp16")
        assert result is not None
        assert result.flops > 0

    def test_memory_bound_detection(self, backend, device):
        result = linear((4096,), (5120, 4096), dtype="fp16")
        is_mem_bound = backend.is_memory_bound(result.flops, result.bytes_accessed, "fp16", device)
        assert isinstance(is_mem_bound, bool)


class TestProfilingBackend:
    """Tests for ProfilingBackend."""

    @pytest.fixture
    def profiling_data(self):
        return {
            "linear": {
                "4096_5120_4096_5120": {
                    "forward_time_ms": 0.123,
                    "backward_time_ms": 0.246,
                    "memory_bytes": 8192000,
                    "dtype": "fp16",
                    "device": "H100-SXM-80GB",
                }
            }
        }

    @pytest.fixture
    def device(self):
        return Device.from_preset("H100-SXM-80GB")

    @pytest.fixture
    def backend(self, profiling_data):
        config = BackendConfig(name="profiling")
        backend = ProfilingBackend(config)
        backend.load_data(profiling_data)
        backend.initialize()
        return backend

    def test_backend_initialization(self, backend):
        assert backend.is_initialized()
        assert backend.get_backend_type() == "profiling"

    def test_exact_lookup(self, backend, device):
        input_shapes = [(4096,), (5120, 4096)]
        output_shape = (5120,)
        time = backend.estimate_compute_time("linear", input_shapes, output_shape, "fp16", device)
        assert time == pytest.approx(0.123 * 1e-3, rel=0.01)

    def test_load_from_file(self, profiling_data):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(profiling_data, f)
            temp_path = f.name

        try:
            config = BackendConfig(name="profiling", profiling_data_path=temp_path)
            backend = ProfilingBackend(config)
            backend.initialize()
            assert backend.is_initialized()
            assert "linear" in backend.get_available_kernels()
        finally:
            os.unlink(temp_path)

    def test_save_to_file(self, backend):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            backend.save_to_file(temp_path)
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert "linear" in data
        finally:
            os.unlink(temp_path)


class TestMicroarchBackend:
    """Tests for MicroarchBackend."""

    @pytest.fixture
    def device(self):
        return Device.from_preset("H100-SXM-80GB")

    @pytest.fixture
    def backend(self, device):
        config = BackendConfig(name="microarch", device=device)
        backend = MicroarchBackend(config)
        backend.initialize()
        return backend

    def test_backend_initialization(self, backend):
        assert backend.is_initialized()
        assert backend.get_backend_type() == "microarch"

    def test_estimate_compute_time(self, backend, device):
        input_shapes = [(4096,), (5120, 4096)]
        time = backend.estimate_compute_time("linear", input_shapes, (5120,), "fp16", device)
        assert time > 0

    def test_get_microarch_config(self, backend):
        config = backend.get_microarch_config("H100-SXM-80GB")
        assert config.num_sm > 0
        assert config.warp_size > 0


class TestKernelBackendRegistry:
    """Tests for KernelBackendRegistry."""

    @pytest.fixture
    def registry(self):
        return KernelBackendRegistry()

    def test_registry_initialization(self, registry):
        assert "theory" in registry.list_backends()
        assert "profiling" in registry.list_backends()
        assert "microarch" in registry.list_backends()

    def test_default_backend(self, registry):
        assert registry.get_default_backend_name() == "theory"
        backend = registry.get_default_backend()
        assert isinstance(backend, TheoryBackend)

    def test_get_backend(self, registry):
        backend = registry.get_backend("theory")
        assert isinstance(backend, TheoryBackend)
        assert backend.is_initialized()

    def test_create_backend(self, registry):
        device = Device.from_preset("A100-SXM-80GB")
        backend = registry.create_backend("theory", device=device)
        assert isinstance(backend, TheoryBackend)
        assert backend.config.device == device

    def test_set_default_backend(self, registry):
        registry.set_default_backend("profiling")
        assert registry.get_default_backend_name() == "profiling"
        registry.set_default_backend("theory")

    def test_has_backend(self, registry):
        assert registry.has_backend("theory")
        assert not registry.has_backend("unknown")


class TestBackendIntegration:
    """Integration tests for backend with existing code."""

    @pytest.fixture
    def device(self):
        return Device.from_preset("H100-SXM-80GB")

    def test_theory_backend_matches_existing_kernel(self, device):
        from llm_perf.kernels.compute import ComputeKernelRegistry

        backend = TheoryBackend(BackendConfig(name="theory", device=device))
        backend.initialize()

        result = linear((4096,), (5120, 4096), dtype="fp16")
        backend_time = backend.estimate_compute_time_from_result(result, device)

        compute_registry = ComputeKernelRegistry(device)
        kernel = compute_registry.get_or_create_matmul(1, 5120, 4096, "fp16")
        kernel_time = kernel.estimate_time((4096,), (5120,), "fp16")

        ratio = backend_time / kernel_time
        assert 0.5 < ratio < 2.0

    def test_backend_to_dict(self, device):
        backend = TheoryBackend(BackendConfig(name="theory", device=device))
        backend.initialize()
        info = backend.to_dict()
        assert info["name"] == "theory"
        assert info["initialized"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
