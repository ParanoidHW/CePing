#!/usr/bin/env python3
"""Standalone test script for Kernel Backend Layer."""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing Kernel Backend Layer...")

try:
    print("\n1. Testing imports...")
    from llm_perf.kernels.functional import (
        KernelResult,
        linear,
        bmm,
        scaled_dot_product_attention,
        flash_attention,
        layer_norm,
        rms_norm,
        silu,
        gelu,
        relu,
        softmax,
        conv2d,
        conv3d,
        embedding,
    )
    from llm_perf.kernels.base import KernelConfig
    from llm_perf.hardware.device import Device, ComputeUnitType

    print("   ✓ Functional imports OK")

    from llm_perf.kernels.backend.base import KernelBackend, BackendConfig

    print("   ✓ Backend base imports OK")

    from llm_perf.kernels.backend.theory import TheoryBackend

    print("   ✓ TheoryBackend imports OK")

    from llm_perf.kernels.backend.profiling import ProfilingBackend

    print("   ✓ ProfilingBackend imports OK")

    from llm_perf.kernels.backend.microarch import MicroarchBackend, MicroarchConfig

    print("   ✓ MicroarchBackend imports OK")

    from llm_perf.kernels.backend.registry import KernelBackendRegistry

    print("   ✓ KernelBackendRegistry imports OK")

    print("\n2. Testing TheoryBackend...")
    device = Device.from_preset("H100-SXM-80GB")
    config = BackendConfig(name="theory", device=device)
    backend = TheoryBackend(config)
    backend.initialize()

    assert backend.is_initialized(), "Backend should be initialized"
    assert backend.name == "theory", "Backend name should be 'theory'"
    assert backend.get_backend_type() == "theory", "Backend type should be 'theory'"
    print("   ✓ TheoryBackend initialization OK")

    input_shapes = [(4096,), (5120, 4096)]
    output_shape = (5120,)
    time = backend.estimate_compute_time("linear", input_shapes, output_shape, "fp16", device)
    assert time > 0, "Compute time should be positive"
    assert time < 1.0, "Compute time should be reasonable"
    print(f"   ✓ Linear compute time: {time * 1e6:.2f} µs")

    result = backend.compute_kernel_result("linear", [(4096,), (5120, 4096)], "fp16")
    assert result is not None, "Kernel result should not be None"
    assert result.flops > 0, "FLOPs should be positive"
    print(f"   ✓ Linear FLOPs: {result.flops / 1e9:.2f} GFLOPs")

    comm_time = backend.estimate_comm_time("allreduce", 1024 * 1024 * 1024, 8, 100.0)
    assert comm_time > 0, "Communication time should be positive"
    print(f"   ✓ AllReduce time: {comm_time * 1e3:.2f} ms")

    memory = backend.estimate_memory("linear", [(4096,), (5120, 4096)], (5120,), "fp16")
    assert memory > 0, "Memory should be positive"
    print(f"   ✓ Linear memory: {memory / 1e6:.2f} MB")

    print("\n3. Testing ProfilingBackend...")
    profiling_data = {
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

    config = BackendConfig(name="profiling")
    backend = ProfilingBackend(config)
    backend.load_data(profiling_data)
    backend.initialize()

    assert backend.is_initialized(), "Backend should be initialized"
    assert backend.get_backend_type() == "profiling", "Backend type should be 'profiling'"
    print("   ✓ ProfilingBackend initialization OK")

    time = backend.estimate_compute_time("linear", [(4096,), (5120, 4096)], (5120,), "fp16", device)
    expected_time = 0.123 * 1e-3
    assert abs(time - expected_time) < 0.01 * expected_time, f"Time should be ~{expected_time}s"
    print(f"   ✓ Exact lookup time: {time * 1e6:.2f} µs (expected: {expected_time * 1e6:.2f} µs)")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profiling_data, f)
        temp_path = f.name

    try:
        config = BackendConfig(name="profiling", profiling_data_path=temp_path)
        backend2 = ProfilingBackend(config)
        backend2.initialize()
        assert backend2.is_initialized(), "Backend should be initialized from file"
        print("   ✓ ProfilingBackend load from file OK")
    finally:
        os.unlink(temp_path)

    print("\n4. Testing MicroarchBackend...")
    config = BackendConfig(name="microarch", device=device)
    backend = MicroarchBackend(config)
    backend.initialize()

    assert backend.is_initialized(), "Backend should be initialized"
    assert backend.get_backend_type() == "microarch", "Backend type should be 'microarch'"
    print("   ✓ MicroarchBackend initialization OK")

    time = backend.estimate_compute_time("linear", [(4096,), (5120, 4096)], (5120,), "fp16", device)
    assert time > 0, "Compute time should be positive"
    print(f"   ✓ Linear compute time: {time * 1e6:.2f} µs")

    microarch_config = backend.get_microarch_config("H100-SXM-80GB")
    assert microarch_config.num_sm > 0, "SM count should be positive"
    print(f"   ✓ H100 microarch: {microarch_config.num_sm} SMs, warp_size={microarch_config.warp_size}")

    print("\n5. Testing KernelBackendRegistry...")
    registry = KernelBackendRegistry()

    assert "theory" in registry.list_backends(), "theory backend should be registered"
    assert "profiling" in registry.list_backends(), "profiling backend should be registered"
    assert "microarch" in registry.list_backends(), "microarch backend should be registered"
    print("   ✓ Registry initialization OK")

    assert registry.get_default_backend_name() == "theory", "Default backend should be theory"
    backend = registry.get_default_backend()
    assert isinstance(backend, TheoryBackend), "Default backend should be TheoryBackend"
    print("   ✓ Default backend retrieval OK")

    registry.set_default_backend("profiling")
    assert registry.get_default_backend_name() == "profiling", "Default backend should be profiling"
    registry.set_default_backend("theory")
    print("   ✓ Set default backend OK")

    backend = registry.create_backend("theory", device=device)
    assert isinstance(backend, TheoryBackend), "Created backend should be TheoryBackend"
    print("   ✓ Create backend with device OK")

    print("\n6. Testing functional API integration...")
    result = linear((4096,), (5120, 4096), dtype="fp16")
    print(f"   ✓ linear() result: FLOPs={result.flops / 1e9:.2f}G, bytes={result.bytes_accessed / 1e6:.2f}MB")

    backend = registry.get_backend("theory")
    time = backend.estimate_compute_time_from_result(result, device)
    assert time > 0, "Time from result should be positive"
    print(f"   ✓ Time from KernelResult: {time * 1e6:.2f} µs")

    training_time = backend.estimate_training_time_from_result(result, device)
    assert training_time > time, "Training time should be greater than forward time"
    print(f"   ✓ Training time: {training_time * 1e6:.2f} µs")

    print("\n✅ All tests passed!")
    print("\nSummary:")
    print("- Backend base class interface defined")
    print("- TheoryBackend: FLOPs/Roofline model estimation")
    print("- ProfilingBackend: Lookup table with interpolation")
    print("- MicroarchBackend: Placeholder for future microarchitecture modeling")
    print("- KernelBackendRegistry: Backend management")
    print("- Integration with functional API works")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
