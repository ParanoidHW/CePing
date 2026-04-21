"""Tests for validation module."""

import unittest
from llm_perf.strategy.parallel_context import ParallelContext, SPType
from llm_perf.validation import (
    ValidationErrors,
    ValidationError,
    ValidationLevel,
    ValidationCategory,
    validate_all,
    validate_strategy,
    validate_model,
    validate_sequence,
    validate_memory,
    validate_special,
    validate_vpp,
)


class TestValidationError(unittest.TestCase):
    """Test ValidationError dataclass."""

    def test_error_creation(self):
        """Test creating an error."""
        error = ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="TEST_ERROR",
            message="Test error message",
        )
        self.assertEqual(error.level, ValidationLevel.ERROR)
        self.assertEqual(error.category, ValidationCategory.STRATEGY)
        self.assertEqual(error.code, "TEST_ERROR")

    def test_error_to_dict(self):
        """Test error serialization."""
        error = ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="TEST_ERROR",
            message="Test message",
            suggestion="Fix suggestion",
            details={"key": "value"},
        )
        data = error.to_dict()
        self.assertEqual(data["level"], "error")
        self.assertEqual(data["category"], "strategy")
        self.assertEqual(data["suggestion"], "Fix suggestion")
        self.assertEqual(data["details"]["key"], "value")


class TestValidationErrors(unittest.TestCase):
    """Test ValidationErrors collection."""

    def test_empty_errors(self):
        """Test empty validation errors."""
        errors = ValidationErrors()
        self.assertFalse(errors.has_errors())
        self.assertFalse(errors.has_warnings())
        self.assertFalse(errors)

    def test_add_error(self):
        """Test adding errors."""
        errors = ValidationErrors()
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="TEST",
            message="Test",
        ))
        self.assertTrue(errors.has_errors())
        self.assertFalse(errors.has_warnings())

    def test_add_warning(self):
        """Test adding warnings."""
        errors = ValidationErrors()
        errors.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.STRATEGY,
            code="TEST",
            message="Test",
        ))
        self.assertFalse(errors.has_errors())
        self.assertTrue(errors.has_warnings())

    def test_merge(self):
        """Test merging errors."""
        errors1 = ValidationErrors()
        errors1.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="E1",
            message="Error 1",
        ))
        errors2 = ValidationErrors()
        errors2.add_error(ValidationError(
            level=ValidationLevel.WARNING,
            category=ValidationCategory.MODEL,
            code="W1",
            message="Warning 1",
        ))
        errors1.merge(errors2)
        self.assertEqual(len(errors1.errors), 1)
        self.assertEqual(len(errors1.warnings), 1)

    def test_to_dict(self):
        """Test errors serialization."""
        errors = ValidationErrors()
        errors.add_error(ValidationError(
            level=ValidationLevel.ERROR,
            category=ValidationCategory.STRATEGY,
            code="TEST",
            message="Test",
        ))
        data = errors.to_dict()
        self.assertTrue(data["has_errors"])
        self.assertFalse(data["has_warnings"])
        self.assertEqual(len(data["errors"]), 1)


class TestStrategyValidator(unittest.TestCase):
    """Test StrategyValidator."""

    def test_valid_strategy(self):
        """Test valid parallel strategy."""
        ctx = ParallelContext(tp_degree=2, pp_degree=2, dp_degree=2, sp_degree=1)
        errors = validate_strategy(ctx, num_gpus=8)
        self.assertFalse(errors.has_errors())

    def test_parallel_product_mismatch(self):
        """Test parallel product mismatch."""
        ctx = ParallelContext(tp_degree=2, pp_degree=2, dp_degree=1, sp_degree=1)
        errors = validate_strategy(ctx, num_gpus=8)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "PARALLEL_PRODUCT_MISMATCH")

    def test_ep_exceeds_tp(self):
        """Test EP exceeding TP."""
        ctx = ParallelContext(tp_degree=2, ep_degree=4, dp_degree=1, sp_degree=1)
        errors = validate_strategy(ctx, num_gpus=8)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "EP_EXCEEDS_TP")

    def test_invalid_parallel_degree(self):
        """Test invalid parallel degree (less than 1)."""
        ctx = ParallelContext(tp_degree=0, dp_degree=1, pp_degree=1, ep_degree=1, sp_degree=1)
        errors = validate_strategy(ctx, num_gpus=1)
        self.assertTrue(errors.has_errors())
        error_codes = [e.code for e in errors.errors]
        self.assertIn("INVALID_PARALLEL_DEGREE", error_codes)


class TestModelValidator(unittest.TestCase):
    """Test ModelValidator."""

    def test_valid_model(self):
        """Test valid model specification."""
        ctx = ParallelContext(tp_degree=8)
        errors = validate_model(
            ctx,
            vocab_size=32000,
            hidden_size=4096,
            num_heads=32,
            intermediate_size=11008,
        )
        self.assertFalse(errors.has_errors())

    def test_vocab_size_not_divisible(self):
        """Test vocab_size not divisible by TP."""
        ctx = ParallelContext(tp_degree=7)
        errors = validate_model(ctx, vocab_size=32000, hidden_size=4096, num_heads=32, intermediate_size=11008)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "VOCAB_SIZE_NOT_DIVISIBLE")

    def test_hidden_size_not_divisible(self):
        """Test hidden_size not divisible by TP."""
        ctx = ParallelContext(tp_degree=3)
        errors = validate_model(ctx, vocab_size=30000, hidden_size=4096, num_heads=30, intermediate_size=12000)
        self.assertTrue(errors.has_errors())
        error_codes = [e.code for e in errors.errors]
        self.assertIn("HIDDEN_SIZE_NOT_DIVISIBLE", error_codes)

    def test_num_heads_not_divisible(self):
        """Test num_heads not divisible by TP."""
        ctx = ParallelContext(tp_degree=5)
        errors = validate_model(ctx, vocab_size=32000, hidden_size=5000, num_heads=32, intermediate_size=15000)
        self.assertTrue(errors.has_errors())
        error_codes = [e.code for e in errors.errors]
        self.assertIn("NUM_HEADS_NOT_DIVISIBLE", error_codes)

    def test_mqa_warning(self):
        """Test MQA detection warning."""
        ctx = ParallelContext(tp_degree=8)
        errors = validate_model(ctx, vocab_size=32000, hidden_size=4096, num_heads=32, intermediate_size=11008, num_kv_heads=1)
        self.assertFalse(errors.has_errors())
        self.assertTrue(errors.has_warnings())
        self.assertEqual(errors.warnings[0].code, "MQA_DETECTED")


class TestSequenceValidator(unittest.TestCase):
    """Test SequenceValidator."""

    def test_valid_sp(self):
        """Test valid sequence parallelism."""
        ctx = ParallelContext(tp_degree=8, sp_degree=4)
        errors = validate_sequence(ctx, seq_len=8192, num_heads=32)
        self.assertFalse(errors.has_errors())

    def test_sp_exceeds_tp(self):
        """Test SP exceeding TP."""
        ctx = ParallelContext(tp_degree=4, sp_degree=8)
        errors = validate_sequence(ctx, seq_len=4096)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "SP_EXCEEDS_TP")

    def test_seq_len_not_divisible(self):
        """Test seq_len not divisible by SP."""
        ctx = ParallelContext(tp_degree=8, sp_degree=4)
        errors = validate_sequence(ctx, seq_len=4095)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "SEQ_LEN_NOT_DIVISIBLE_BY_SP")

    def test_ulysses_valid(self):
        """Test valid Ulysses SP."""
        ctx = ParallelContext(tp_degree=4, ulysses_degree=2, sp_degree=2)
        errors = validate_sequence(ctx, seq_len=4096, num_heads=32)
        self.assertFalse(errors.has_errors())

    def test_ulysses_exceeds_heads(self):
        """Test Ulysses exceeding heads."""
        ctx = ParallelContext(tp_degree=8, ulysses_degree=8, sp_degree=8)
        errors = validate_sequence(ctx, seq_len=4096, num_heads=32)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "ULYSSES_TP_EXCEEDS_HEADS")

    def test_megatron_sp_valid(self):
        """Test valid Megatron-SP."""
        ctx = ParallelContext(tp_degree=8, sp_degree=8, sp_type=SPType.MEGATRON)
        errors = validate_sequence(ctx, seq_len=4096)
        self.assertFalse(errors.has_errors())

    def test_megatron_sp_mismatch(self):
        """Test Megatron-SP mismatch."""
        ctx = ParallelContext(tp_degree=8, sp_degree=4, sp_type=SPType.MEGATRON)
        errors = validate_sequence(ctx, seq_len=4096)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "MEGATRON_SP_MISMATCH")


class TestMemoryValidator(unittest.TestCase):
    """Test MemoryValidator."""

    def test_valid_memory(self):
        """Test valid memory configuration."""
        ctx = ParallelContext(tp_degree=8, dp_degree=1, pp_degree=1, ep_degree=1, sp_degree=1)
        errors = validate_memory(
            ctx,
            weight_memory_gb=20.0,
            activation_memory_gb=5.0,
            device_memory_gb=80.0,
            gradient_memory_gb=20.0,
            optimizer_memory_gb=20.0,
            mode="training",
        )
        self.assertFalse(errors.has_errors())

    def test_weight_memory_exceeded(self):
        """Test weight memory exceeded."""
        ctx = ParallelContext(tp_degree=1)
        errors = validate_memory(
            ctx,
            weight_memory_gb=168.0,
            activation_memory_gb=10.0,
            device_memory_gb=80.0,
            mode="inference",
        )
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "WEIGHT_MEMORY_EXCEEDED")

    def test_training_memory_exceeded(self):
        """Test training memory exceeded."""
        ctx = ParallelContext(tp_degree=8)
        errors = validate_memory(
            ctx,
            weight_memory_gb=21.0,
            activation_memory_gb=30.0,
            device_memory_gb=80.0,
            gradient_memory_gb=21.0,
            optimizer_memory_gb=42.0,
            mode="training",
        )
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "TRAINING_MEMORY_EXCEEDED")

    def test_activation_threshold_warning(self):
        """Test activation threshold warning."""
        ctx = ParallelContext(tp_degree=8)
        errors = validate_memory(
            ctx,
            weight_memory_gb=10.0,
            activation_memory_gb=45.0,
            device_memory_gb=80.0,
            mode="training",
        )
        self.assertTrue(errors.has_warnings())
        self.assertEqual(errors.warnings[0].code, "ACTIVATION_MEMORY_HIGH")


class TestSpecialValidator(unittest.TestCase):
    """Test SpecialValidator."""

    def test_dit_valid(self):
        """Test valid DiT configuration."""
        ctx = ParallelContext(tp_degree=8, ulysses_degree=2, sp_degree=2)
        errors = validate_special(
            ctx,
            model_type="dit",
            num_heads=40,
            image_height=1024,
            image_width=1024,
            patch_size=16,
        )
        self.assertFalse(errors.has_errors())

    def test_dit_heads_exceeded(self):
        """Test DiT heads exceeded."""
        ctx = ParallelContext(tp_degree=8, ulysses_degree=8, sp_degree=8)
        errors = validate_special(
            ctx,
            model_type="dit",
            num_heads=40,
        )
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "DIT_HEADS_EXCEEDED")

    def test_image_not_divisible(self):
        """Test image not divisible by patch."""
        ctx = ParallelContext(tp_degree=8)
        errors = validate_special(
            ctx,
            model_type="dit",
            image_height=1023,
            patch_size=16,
        )
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "IMAGE_HEIGHT_NOT_DIVISIBLE")

    def test_vpp_valid(self):
        """Test valid VPP configuration."""
        ctx = ParallelContext(pp_degree=4)
        errors = validate_vpp(ctx, num_layers=32, vpp_degree=2)
        self.assertFalse(errors.has_errors())

    def test_vpp_layers_not_divisible(self):
        """Test VPP layers not divisible."""
        ctx = ParallelContext(pp_degree=4)
        errors = validate_vpp(ctx, num_layers=30, vpp_degree=2)
        self.assertTrue(errors.has_errors())
        self.assertEqual(errors.errors[0].code, "LAYERS_NOT_DIVISIBLE_BY_VPP")


class TestValidateAll(unittest.TestCase):
    """Test validate_all unified entry."""

    def test_valid_all(self):
        """Test all validations pass."""
        ctx = ParallelContext(tp_degree=8)
        errors = validate_all(
            ctx,
            num_gpus=8,
            vocab_size=32000,
            hidden_size=4096,
            num_heads=32,
            intermediate_size=11008,
            seq_len=4096,
            weight_memory_gb=20.0,
            device_memory_gb=80.0,
            mode="inference",
        )
        self.assertFalse(errors.has_errors())

    def test_multiple_errors(self):
        """Test multiple validation errors."""
        ctx = ParallelContext(tp_degree=3)
        errors = validate_all(
            ctx,
            num_gpus=8,
            vocab_size=32000,
            hidden_size=4096,
            num_heads=32,
            intermediate_size=11008,
            seq_len=4096,
        )
        self.assertTrue(errors.has_errors())
        self.assertGreater(len(errors.errors), 1)


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for validation module."""

    def test_llama_7b_config(self):
        """Test Llama-7B configuration."""
        ctx = ParallelContext(tp_degree=2, pp_degree=2, dp_degree=2, sp_degree=1)
        errors = validate_all(
            ctx,
            num_gpus=8,
            vocab_size=32000,
            hidden_size=4096,
            num_heads=32,
            intermediate_size=11008,
            seq_len=2048,
        )
        self.assertFalse(errors.has_errors())

    def test_dsv3_config(self):
        """Test DeepSeek-V3-like configuration."""
        ctx = ParallelContext(tp_degree=8, ep_degree=8, dp_degree=1, sp_degree=1)
        errors = validate_strategy(ctx, num_gpus=64)
        self.assertFalse(errors.has_errors())

        errors = validate_model(
            ctx,
            vocab_size=129280,
            hidden_size=7168,
            num_heads=128,
            intermediate_size=18432,
        )
        self.assertFalse(errors.has_errors())


if __name__ == "__main__":
    unittest.main()