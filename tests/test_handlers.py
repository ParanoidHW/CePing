"""Tests for ModelType Handler mechanism."""

import pytest

from llm_perf.analyzer.base import WorkloadType
from llm_perf.analyzer.handlers import get_handler
from llm_perf.analyzer.handlers.base_handler import BaseModelHandler
from llm_perf.analyzer.handlers.diffusion_handler import DiffusionHandler
from llm_perf.analyzer.handlers.llm_handler import LLMHandler
from llm_perf.analyzer.handlers.vision_handler import VisionHandler
from llm_perf.modeling import ShardedModule


class MockLLMModule(ShardedModule):
    """Mock LLM module for testing."""
    
    def __init__(self):
        self.hidden_size = 4096
        self.vocab_size = 32000
    
    def __call__(self, *args, **kwargs):
        pass


class MockDiffusionModule(ShardedModule):
    """Mock Diffusion module for testing."""
    
    def __init__(self):
        self.latent_channels = 16
        self.vae_compression_ratio = 16
    
    def __call__(self, *args, **kwargs):
        pass


class MockVisionEncoder(ShardedModule):
    """Mock Vision encoder for testing."""
    
    def __init__(self):
        self.in_channels = 3
    
    def __call__(self, *args, **kwargs):
        pass


class MockVisionDecoder(ShardedModule):
    """Mock Vision decoder for testing."""
    
    def __init__(self):
        self.latent_channels = 16
        self.block_out_channels = (128, 256, 512, 512)
    
    def __call__(self, *args, **kwargs):
        pass


class TestHandlerRegistry:
    """Test handler registry and dispatch."""
    
    def test_get_handler_training(self):
        """Test get_handler returns LLMHandler for TRAINING."""
        handler = get_handler(WorkloadType.TRAINING)
        assert isinstance(handler, LLMHandler)
    
    def test_get_handler_inference(self):
        """Test get_handler returns LLMHandler for INFERENCE."""
        handler = get_handler(WorkloadType.INFERENCE)
        assert isinstance(handler, LLMHandler)
    
    def test_get_handler_diffusion(self):
        """Test get_handler returns DiffusionHandler for DIFFUSION."""
        handler = get_handler(WorkloadType.DIFFUSION)
        assert isinstance(handler, DiffusionHandler)
    
    def test_get_handler_mixed(self):
        """Test get_handler returns default LLMHandler for MIXED."""
        handler = get_handler(WorkloadType.MIXED)
        assert isinstance(handler, LLMHandler)
    
    def test_handler_registry_complete(self):
        """Test all workload types have handlers."""
        for wt in WorkloadType:
            handler = get_handler(wt)
            assert isinstance(handler, BaseModelHandler)


class TestLLMHandler:
    """Test LLMHandler seq_len and input creation."""
    
    def test_seq_len_prefill(self):
        """Test LLMHandler seq_len for prefill phase."""
        handler = LLMHandler()
        params = {"prompt_len": 1024, "seq_len": 512}
        
        seq_len = handler.get_seq_len(MockLLMModule(), params, "prefill")
        
        assert seq_len == 1024
    
    def test_seq_len_decode(self):
        """Test LLMHandler seq_len for decode phase."""
        handler = LLMHandler()
        params = {"prompt_len": 1024, "seq_len": 512}
        
        seq_len = handler.get_seq_len(MockLLMModule(), params, "decode")
        
        assert seq_len == 1
    
    def test_seq_len_other_phase(self):
        """Test LLMHandler seq_len for other phases."""
        handler = LLMHandler()
        params = {"seq_len": 2048}
        
        seq_len = handler.get_seq_len(MockLLMModule(), params, "forward")
        
        assert seq_len == 2048
    
    def test_create_inputs_with_vocab_size(self):
        """Test LLMHandler creates token input for vocab models."""
        handler = LLMHandler()
        component = MockLLMModule()
        
        inputs = handler.create_inputs(component, batch_size=2, seq_len=512, params={})
        
        assert len(inputs) == 1
        assert inputs[0].shape == (2, 512)
    
    def test_create_inputs_without_vocab_size(self):
        """Test LLMHandler creates hidden input for non-vocab models."""
        handler = LLMHandler()
        component = MockLLMModule()
        delattr(component, "vocab_size")
        
        inputs = handler.create_inputs(component, batch_size=2, seq_len=512, params={})
        
        assert len(inputs) == 1
        assert inputs[0].shape == (2, 512, 4096)


class TestDiffusionHandler:
    """Test DiffusionHandler seq_len and input creation."""
    
    def test_seq_len_latent_spatial(self):
        """Test DiffusionHandler seq_len is latent spatial size."""
        handler = DiffusionHandler()
        params = {"height": 720, "width": 1280}
        component = MockDiffusionModule()
        
        seq_len = handler.get_seq_len(component, params, "denoise")
        
        latent_h = 720 // 16
        latent_w = 1280 // 16
        expected = latent_h * latent_w
        
        assert seq_len == expected
    
    def test_seq_len_custom_vae_ratio(self):
        """Test DiffusionHandler with custom vae_compression_ratio."""
        handler = DiffusionHandler()
        params = {"height": 720, "width": 1280}
        component = MockDiffusionModule()
        component.vae_compression_ratio = 8
        
        seq_len = handler.get_seq_len(component, params, "denoise")
        
        latent_h = 720 // 8
        latent_w = 1280 // 8
        expected = latent_h * latent_w
        
        assert seq_len == expected
    
    def test_create_inputs_latent_timestep(self):
        """Test DiffusionHandler creates latent + timestep."""
        handler = DiffusionHandler()
        component = MockDiffusionModule()
        
        inputs = handler.create_inputs(component, batch_size=2, seq_len=3600, params={})
        
        assert len(inputs) == 2
        assert inputs[0].shape == (2, 3600, 16)
        assert inputs[1].shape == (2, 256)


class TestVisionHandler:
    """Test VisionHandler seq_len and input creation."""
    
    def test_seq_len_pixel_count(self):
        """Test VisionHandler seq_len is pixel count."""
        handler = VisionHandler()
        params = {"num_frames": 81, "height": 720, "width": 1280}
        
        seq_len = handler.get_seq_len(MockVisionEncoder(), params, "encode")
        
        expected = 81 * 720 * 1280
        assert seq_len == expected
    
    def test_create_inputs_encoder(self):
        """Test VisionHandler creates encoder input."""
        handler = VisionHandler()
        component = MockVisionEncoder()
        params = {"num_frames": 81, "height": 720, "width": 1280}
        
        inputs = handler.create_inputs(component, batch_size=2, seq_len=0, params=params)
        
        assert len(inputs) == 1
        assert inputs[0].shape == (2, 3, 81, 720, 1280)
    
    def test_create_inputs_decoder(self):
        """Test VisionHandler creates decoder latent input."""
        handler = VisionHandler()
        component = MockVisionDecoder()
        params = {"num_frames": 81, "height": 720, "width": 1280}
        
        inputs = handler.create_inputs(component, batch_size=2, seq_len=0, params=params)
        
        assert len(inputs) == 1
        latent_t = (81 - 1) // 4 + 1
        latent_h = 720 // 8
        latent_w = 1280 // 8
        assert inputs[0].shape == (2, 16, latent_t, latent_h, latent_w)


class TestBaseHandlerForward:
    """Test BaseModelHandler forward method."""
    
    def test_forward_success(self):
        """Test forward executes successfully."""
        handler = LLMHandler()
        component = MockLLMModule()
        inputs = handler.create_inputs(component, batch_size=1, seq_len=10, params={})
        
        result = handler.forward(component, inputs)
        
        assert result is None
    
    def test_forward_failure_raises_runtime_error(self):
        """Test forward failure raises RuntimeError with clear message."""
        handler = LLMHandler()
        
        class FailingModule(ShardedModule):
            def __init__(self):
                self.hidden_size = 4096
            
            def __call__(self, *args, **kwargs):
                raise ValueError("Simulated forward error")
        
        component = FailingModule()
        inputs = handler.create_inputs(component, batch_size=1, seq_len=10, params={})
        
        with pytest.raises(RuntimeError) as exc_info:
            handler.forward(component, inputs)
        
        assert "Forward pass failed" in str(exc_info.value)
        assert "FailingModule" in str(exc_info.value)