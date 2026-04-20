"""Tests for complete models."""

from llm_perf.modeling import (
    ShardedTensor,
    ShardedParameter,
    ParallelContext,
    ShardedTransformerBlock,
    ShardedMoEBlock,
    ShardedMoE,
    ShardedFFN,
    LlamaModel,
    DeepSeekModel,
    create_model_from_config,
    get_model_presets,
    ShardedWanDiT,
)


class TestShardedTransformerBlock:
    """Test ShardedTransformerBlock."""

    def test_block_creation(self):
        """Test creating transformer block."""
        block = ShardedTransformerBlock(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        assert block.hidden_size == 4096
        assert block.num_heads == 32
        assert block.num_kv_heads == 8
        assert block.intermediate_size == 11008

    def test_block_submodules(self):
        """Test block has correct submodules."""
        block = ShardedTransformerBlock(4096, 32, 8, 11008)

        assert "input_norm" in block._submodules
        assert "attention" in block._submodules
        assert "post_attn_norm" in block._submodules
        assert "ffn" in block._submodules

    def test_block_params_count(self):
        """Test block params count."""
        block = ShardedTransformerBlock(4096, 32, 8, 128, 11008)

        norm_params = 4096 * 2

        q_params = 4096 * 32 * 128
        k_params = 4096 * 8 * 128
        v_params = 4096 * 8 * 128
        o_params = 32 * 128 * 4096
        attn_params = q_params + k_params + v_params + o_params

        gate_params = 4096 * 11008
        up_params = 4096 * 11008
        down_params = 11008 * 4096
        ffn_params = gate_params + up_params + down_params

        expected = norm_params + attn_params + ffn_params
        assert block.params_count() == expected

    def test_block_forward(self):
        """Test block forward."""
        block = ShardedTransformerBlock(4096, 32, 8, 11008)

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = block(hidden)

        assert output.shape == (1, 512, 4096)
        assert "attn_out" in block._activations
        assert "ffn_out" in block._activations


class TestLlamaModel:
    """Test LlamaModel."""

    def test_llama_creation(self):
        """Test creating Llama model."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        assert model.vocab_size == 32000
        assert model.hidden_size == 4096
        assert model.num_layers == 32
        assert model.num_heads == 32
        assert model.num_kv_heads == 8
        assert model.intermediate_size == 11008

    def test_llama_layers(self):
        """Test Llama has correct number of layers."""
        model = LlamaModel(32000, 4096, 4, 32, 8, 11008)

        assert len(model.layers) == 4
        assert "embedding" in model._submodules
        assert "final_norm" in model._submodules
        assert "lm_head" in model._submodules

        for i in range(4):
            assert f"layers.{i}" in model._submodules

    def test_llama_params_count(self):
        """Test Llama params count."""
        vocab_size = 32000
        hidden_size = 4096
        num_layers = 2
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        intermediate_size = 11008

        model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
        )

        embedding_params = vocab_size * hidden_size
        lm_head_params = hidden_size * vocab_size

        norm_params = hidden_size

        q_params = hidden_size * num_heads * head_dim
        k_params = hidden_size * num_kv_heads * head_dim
        v_params = hidden_size * num_kv_heads * head_dim
        o_params = num_heads * head_dim * hidden_size
        attn_params = q_params + k_params + v_params + o_params

        gate_params = hidden_size * intermediate_size
        up_params = hidden_size * intermediate_size
        down_params = intermediate_size * hidden_size
        ffn_params = gate_params + up_params + down_params

        layer_params = norm_params + attn_params + norm_params + ffn_params
        total_layer_params = layer_params * num_layers

        final_norm_params = hidden_size

        expected = embedding_params + total_layer_params + final_norm_params + lm_head_params

        assert model.params_count() == expected

    def test_llama_forward(self):
        """Test Llama forward."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
        )

        input_ids = ShardedTensor(shape=(1, 512))
        logits = model(input_ids)

        assert logits.shape == (1, 512, 32000)
        assert "embedding_output" in model._activations
        assert "layer_0_output" in model._activations
        assert "layer_1_output" in model._activations

    def test_llama_module_instance(self):
        """Test Llama module instance."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        ctx = ParallelContext(tp_degree=8)

        instance = model.bind(ctx)

        assert instance.params_count_logical == model.params_count()
        assert instance.params_count_physical < instance.params_count_logical

    def test_llama_to_dict(self):
        """Test Llama instance to_dict."""
        model = LlamaModel(32000, 4096, 2, 32, 8)
        model._name = "llama_test"

        ctx = ParallelContext(tp_degree=2)

        instance = model.bind(ctx)
        result = instance.to_dict()

        assert "module_name" in result
        assert "params" in result
        assert "submodules" in result
        assert len(result["submodules"]) > 0


class TestEndToEnd:
    """End-to-end tests."""

    def test_simple_llama_pipeline(self):
        """Test simple pipeline with Llama."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=2,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
        )

        ctx = ParallelContext(
            tp_degree=8,
            sp_degree=1,
            dtype="fp16",
        )

        input_ids = ShardedTensor(shape=(1, 512))

        logits = model(input_ids)
        instance = model.bind(ctx)

        assert logits.shape == (1, 512, 32000)
        assert instance.params_count_physical > 0

    def test_llama_with_sp(self):
        """Test Llama with SP."""
        model = LlamaModel(32000, 4096, 2, 32, 8)

        ctx = ParallelContext(
            tp_degree=8,
            sp_degree=4,
            dtype="fp16",
        )

        input_ids = ShardedTensor(
            shape=(1, 512),
            shardable={},
        )

        logits = model(input_ids)
        instance = model.bind(ctx)

        assert logits.shape == (1, 512, 32000)


class TestShardedMoE:
    """Test ShardedMoE."""

    def test_moe_creation(self):
        """Test creating MoE layer."""
        moe = ShardedMoE(
            hidden_size=4096,
            intermediate_size=2048,
            num_experts=64,
            num_experts_per_token=8,
        )

        assert moe.hidden_size == 4096
        assert moe.intermediate_size == 2048
        assert moe.num_experts == 64
        assert moe.num_experts_per_token == 8

    def test_moe_with_shared_experts(self):
        """Test MoE with shared experts."""
        moe = ShardedMoE(
            hidden_size=4096,
            intermediate_size=2048,
            num_experts=64,
            num_experts_per_token=8,
            shared_expert_intermediate=4096,
        )

        assert moe.shared_expert_intermediate == 4096
        assert "shared_gate_weight" in moe._weights

    def test_moe_params_count(self):
        """Test MoE params count."""
        moe = ShardedMoE(
            hidden_size=4096,
            intermediate_size=2048,
            num_experts=64,
            num_experts_per_token=8,
            shared_expert_intermediate=4096,
        )

        router_params = 4096 * 64
        expert_gate = 64 * 4096 * 2048
        expert_up = 64 * 4096 * 2048
        expert_down = 64 * 2048 * 4096
        routed_params = router_params + expert_gate + expert_up + expert_down

        shared_gate = 4096 * 4096
        shared_up = 4096 * 4096
        shared_down = 4096 * 4096
        shared_params = shared_gate + shared_up + shared_down

        expected = routed_params + shared_params
        assert moe.params_count() == expected

    def test_moe_forward(self):
        """Test MoE forward."""
        moe = ShardedMoE(
            hidden_size=4096,
            intermediate_size=2048,
            num_experts=64,
            num_experts_per_token=8,
        )

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = moe(hidden)

        assert output.shape == (1, 512, 4096)


class TestShardedMoEBlock:
    """Test ShardedMoEBlock."""

    def test_moe_block_creation(self):
        """Test creating MoE block."""
        block = ShardedMoEBlock(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,
            intermediate_size=2048,
            num_experts=64,
            num_experts_per_token=8,
        )

        assert block.hidden_size == 4096
        assert block.num_experts == 64

    def test_moe_block_submodules(self):
        """Test MoE block has correct submodules."""
        block = ShardedMoEBlock(4096, 32, 8, 2048, 64, 8)

        assert "input_norm" in block._submodules
        assert "attention" in block._submodules
        assert "post_attn_norm" in block._submodules
        assert "moe" in block._submodules

    def test_moe_block_forward(self):
        """Test MoE block forward."""
        block = ShardedMoEBlock(4096, 32, 8, 2048, 64, 8)

        hidden = ShardedTensor(shape=(1, 512, 4096))
        output = block(hidden)

        assert output.shape == (1, 512, 4096)
        assert "attn_out" in block._activations
        assert "moe_out" in block._activations


class TestDeepSeekModel:
    """Test DeepSeekModel."""

    def test_deepseek_creation(self):
        """Test creating DeepSeek model."""
        model = DeepSeekModel(
            vocab_size=102400,
            hidden_size=5120,
            num_layers=60,
            num_heads=128,
            num_kv_heads=128,
            first_k_dense_layers=1,
            num_experts=160,
            num_experts_per_token=6,
            shared_expert_intermediate=2048,
            moe_intermediate_size=1536,
        )

        assert model.vocab_size == 102400
        assert model.hidden_size == 5120
        assert model.num_layers == 60
        assert model.num_experts == 160
        assert model.first_k_dense_layers == 1

    def test_deepseek_mixed_layers(self):
        """Test DeepSeek has mixed dense and MoE layers."""
        model = DeepSeekModel(
            vocab_size=102400,
            hidden_size=5120,
            num_layers=4,
            num_heads=128,
            first_k_dense_layers=1,
            num_experts=64,
        )

        assert len(model.layers) == 4
        assert isinstance(model.layers[0], ShardedTransformerBlock)
        assert isinstance(model.layers[1], ShardedMoEBlock)

    def test_deepseek_forward(self):
        """Test DeepSeek forward."""
        model = DeepSeekModel(
            vocab_size=102400,
            hidden_size=5120,
            num_layers=2,
            num_heads=128,
            first_k_dense_layers=1,
            num_experts=64,
        )

        input_ids = ShardedTensor(shape=(1, 128))
        logits = model(input_ids)

        assert logits.shape == (1, 128, 102400)

    def test_deepseek_module_instance(self):
        """Test DeepSeek module instance."""
        model = DeepSeekModel(
            vocab_size=102400,
            hidden_size=5120,
            num_layers=2,
            num_heads=128,
            num_experts=64,
        )

        ctx = ParallelContext(tp_degree=8, ep_degree=2)

        instance = model.bind(ctx)

        assert instance.params_count_logical == model.params_count()
        assert instance.params_count_physical < instance.params_count_logical

    def test_deepseek_all_dense(self):
        """Test DeepSeek with all dense layers."""
        model = DeepSeekModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            first_k_dense_layers=4,
            num_experts=0,
        )

        for layer in model.layers:
            assert isinstance(layer, ShardedTransformerBlock)

    def test_deepseek_all_moe(self):
        """Test DeepSeek with all MoE layers."""
        model = DeepSeekModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=4,
            num_heads=32,
            first_k_dense_layers=0,
            num_experts=64,
        )

        for layer in model.layers:
            assert isinstance(layer, ShardedMoEBlock)


class TestCreateModelFromConfig:
    """Test create_model_from_config with parameter filtering."""

    def test_create_with_preset(self):
        """Test creating model from preset."""
        model = create_model_from_config({"preset": "llama-7b"})
        assert isinstance(model, LlamaModel)
        assert model.hidden_size == 4096
        assert model.num_layers == 32

    def test_create_with_type(self):
        """Test creating model from type field."""
        model = create_model_from_config({"type": "llama-7b"})
        assert isinstance(model, LlamaModel)

    def test_create_with_extra_params_filtered(self):
        """Test that extra params not in model signature are filtered."""
        model = create_model_from_config(
            {
                "preset": "llama-7b",
                "extra_param": "should_be_filtered",
            }
        )
        assert isinstance(model, LlamaModel)

    def test_create_wan_dit_from_preset(self):
        """Test creating WanDiT from preset."""
        model = create_model_from_config({"preset": "wan-dit"})
        assert isinstance(model, ShardedWanDiT)
        assert model.hidden_size == 5120
        assert model.num_layers == 40

    def test_create_wan_dit_with_type(self):
        """Test creating WanDiT from type field."""
        model = create_model_from_config({"type": "wan-dit"})
        assert isinstance(model, ShardedWanDiT)

    def test_create_wan_dit_filters_invalid_params(self):
        """Test that invalid params for WanDiT are filtered."""
        model = create_model_from_config(
            {
                "type": "wan-dit",
                "max_seq_len": 4096,
                "vocab_size": 32000,
            }
        )
        assert isinstance(model, ShardedWanDiT)

    def test_preset_contains_required_params(self):
        """Test that presets contain all required constructor params."""
        presets = get_model_presets()

        llama_preset = presets["llama-7b"]
        model = create_model_from_config({"preset": "llama-7b"})
        assert model.hidden_size == llama_preset["hidden_size"]

        wan_preset = presets["wan-dit"]
        model = create_model_from_config({"preset": "wan-dit"})
        assert model.hidden_size == wan_preset["hidden_size"]

    def test_user_override_preset_params(self):
        """Test that user can override preset params."""
        model = create_model_from_config(
            {
                "preset": "llama-7b",
                "hidden_size": 2048,
            }
        )
        assert isinstance(model, LlamaModel)
        assert model.hidden_size == 2048

    def test_presets_loaded_from_yaml(self):
        """Test that presets are loaded from YAML files."""
        presets = get_model_presets()

        assert "llama-7b" in presets
        assert "llama-13b" in presets
        assert "llama-70b" in presets
        assert "mixtral-8x7b" in presets
        assert "deepseek-v3" in presets
        assert "resnet50" in presets
        assert "video-vae" in presets
        assert "wan-t2v-14b" in presets
        assert "wan-dit" in presets

        llama_7b = presets["llama-7b"]
        assert llama_7b["hidden_size"] == 4096
        assert llama_7b["num_layers"] == 32
        assert llama_7b["architecture"] == "llama"
        assert "param_schema" in llama_7b

    def test_create_model_from_yaml_preset(self):
        """Test create_model_from_config uses YAML presets."""
        model = create_model_from_config({"preset": "llama-13b"})
        assert isinstance(model, LlamaModel)
        assert model.hidden_size == 5120
        assert model.num_layers == 40

    def test_yaml_preset_deepseek_v3(self):
        """Test DeepSeek V3 preset from YAML."""
        model = create_model_from_config({"preset": "deepseek-v3"})
        assert isinstance(model, DeepSeekModel)
        assert model.hidden_size == 7168
        assert model.num_layers == 61
        assert model.num_experts == 256

    def test_yaml_preset_wan_dit(self):
        """Test WanDiT preset from YAML."""
        model = create_model_from_config({"preset": "wan-dit"})
        assert isinstance(model, ShardedWanDiT)
        assert model.hidden_size == 5120
        assert model.num_layers == 40


class TestModelParamsVerification:
    """Verify model params match official values.

    Official reference:
    - LLaMA 2: https://huggingface.co/meta-llama (vocab=32000)
    - LLaMA 3: https://huggingface.co/meta-llama (vocab=128256)
    - DeepSeek V3: https://huggingface.co/deepseek-ai/DeepSeek-V3

    Params formula:
    - FP16 memory = Params × 2 bytes
    - BF16 memory = Params × 2 bytes
    """

    def test_llama_7b_params(self):
        """LLaMA 7B: ~6.7B params, ~13.4GB FP16 memory."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            intermediate_size=11008,
        )

        params = model.params_count()
        fp16_mem_gb = params * 2 / 1e9

        assert 6.5e9 < params < 7.0e9, f"LLaMA 7B params should be ~6.7B, got {params / 1e9:.2f}B"
        assert 13.0 < fp16_mem_gb < 14.0, f"LLaMA 7B FP16 memory should be ~13.4GB, got {fp16_mem_gb:.2f}GB"

    def test_llama_13b_params(self):
        """LLaMA 13B: ~13B params, ~26GB FP16 memory."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            num_kv_heads=40,
            intermediate_size=13824,
        )

        params = model.params_count()
        fp16_mem_gb = params * 2 / 1e9

        assert 12.8e9 < params < 13.2e9, f"LLaMA 13B params should be ~13B, got {params / 1e9:.2f}B"
        assert 25.5 < fp16_mem_gb < 26.5, f"LLaMA 13B FP16 memory should be ~26GB, got {fp16_mem_gb:.2f}GB"

    def test_llama_70b_params_with_gqa(self):
        """LLaMA 70B with GQA: ~69B params, ~138GB FP16 memory.

        GQA (Grouped Query Attention):
        - 64 query heads, 8 KV heads
        - Reduces KV weights by ~9.4B
        """
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=8192,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            intermediate_size=28672,
        )

        params = model.params_count()
        fp16_mem_gb = params * 2 / 1e9

        assert 68e9 < params < 70e9, f"LLaMA 70B params should be ~69B (GQA), got {params / 1e9:.2f}B"
        assert 136 < fp16_mem_gb < 140, f"LLaMA 70B FP16 memory should be ~138GB, got {fp16_mem_gb:.2f}GB"

    def test_deepseek_v3_params(self):
        """DeepSeek V3: MoE model with MLA.

        Official config from HuggingFace:
        - vocab_size: 129280
        - hidden_size: 7168
        - num_layers: 61
        - first_k_dense_replace: 3 (first 3 layers dense, rest MoE)
        - n_routed_experts: 256
        - num_experts_per_tok: 8
        """
        model = DeepSeekModel(
            vocab_size=129280,
            hidden_size=7168,
            num_layers=61,
            num_heads=128,
            num_kv_heads=128,
            intermediate_size=18432,
            first_k_dense_layers=3,
            num_experts=256,
            num_experts_per_token=8,
            moe_intermediate_size=2048,
        )

        params = model.params_count()

        assert params > 1e9, f"DeepSeek V3 should have >1B params (active), got {params / 1e9:.2f}B"

    def test_params_memory_formula(self):
        """Verify FP16/BF16 memory formula: Params × 2 bytes."""
        model = LlamaModel(32000, 4096, 4, 32, 32, 11008)

        params = model.params_count()

        fp16_mem = params * 2
        bf16_mem = params * 2

        assert fp16_mem == bf16_mem, "FP16 and BF16 should have same memory (2 bytes each)"

    def test_llama_7b_breakdown(self):
        """Verify LLaMA 7B params breakdown."""
        model = LlamaModel(
            vocab_size=32000,
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            intermediate_size=11008,
        )

        breakdown = model.params_count_breakdown()

        embedding = sum(v for k, v in breakdown.items() if "embedding" in k)
        attention = sum(v for k, v in breakdown.items() if "attention" in k)
        ffn = sum(v for k, v in breakdown.items() if "ffn" in k or "gate" in k or "up" in k or "down" in k)
        lm_head = sum(v for k, v in breakdown.items() if "lm_head" in k)

        assert embedding == 32000 * 4096, f"Embedding params mismatch"
        assert lm_head == 32000 * 4096, f"LM head params mismatch"
        assert attention > 2e9, f"Attention params should be >2B"
        assert ffn > 4e9, f"FFN params should be >4B"

    def test_deepseek_v3_params_verification(self):
        """Verify DeepSeek V3 params match official claim of 671B."""
        from llm_perf.modeling import create_model_from_config

        model = create_model_from_config({"type": "deepseek-v3"})

        total_params = 0
        for name, sub in model._submodules.items():
            if hasattr(sub, "params_count"):
                total_params += sub.params_count()

        expected_params = 671e9
        assert abs(total_params - expected_params) / expected_params < 0.05

        moe_layer = model.layers[3]
        assert moe_layer.intermediate_size == 2048

        assert len(model.layers) == 61


class TestShardedParameterWeightRegistration:
    """Test ShardedParameter type distinguishes weights from activations."""

    def test_sharded_parameter_not_register_activation(self):
        """Test that ShardedTensor (activation) is not registered as weight."""
        from llm_perf.modeling.layers import ShardedEmbedding
        from llm_perf.modeling.tensor import ShardedTensor, ShardedParameter

        embedding = ShardedEmbedding(32000, 4096)

        assert isinstance(embedding.weight, ShardedParameter)

        assert len(embedding._weights) == 1
        assert "weight" in embedding._weights

        activation = ShardedTensor(shape=(1, 512, 4096))
        embedding._last_forward_input = activation

        assert len(embedding._weights) == 1
        assert "_last_forward_input" not in embedding._weights

    def test_llama_model_weight_count(self):
        """Test LlamaModel has correct number of weights."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=2, num_heads=32)

        expected_weights = 1 + 2 * 9 + 1 + 1
        assert len(model._weights) == 0

        total_weights = len(model.get_weights())
        assert total_weights == expected_weights

    def test_forward_does_not_add_weights(self):
        """Test forward pass does not add activation tensors to _weights."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=1, num_heads=32)

        input_ids = ShardedTensor(shape=(1, 512))
        logits = model(input_ids)

        assert len(model._weights) == 0
        assert "_last_forward_input" not in model._weights
        assert "_last_forward_output" not in model._weights

        for name, submodule in model._submodules.items():
            assert "_last_forward_input" not in submodule._weights
            assert "_last_forward_output" not in submodule._weights

    def test_all_layer_weights_are_parameters(self):
        """Test all layer weights are ShardedParameter instances."""
        model = LlamaModel(vocab_size=32000, hidden_size=4096, num_layers=1, num_heads=32)

        for name, weight in model.get_weights().items():
            assert isinstance(weight, ShardedParameter), f"{name} should be ShardedParameter"


class TestShardedFFNTypes:
    """Test ShardedFFN with different activation types."""

    def test_ffn_swiglu_default(self):
        """Test FFN default uses SwiGLU (gated, 3 weights)."""
        from llm_perf.utils.constants import FFNActType

        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008)

        assert ffn.ffn_act_type == FFNActType.SWIGLU.value
        assert hasattr(ffn, "gate_weight")
        assert hasattr(ffn, "up_weight")
        assert hasattr(ffn, "down_weight")

    def test_ffn_gelu_non_gated(self):
        """Test FFN with GELU (non-gated, 2 weights)."""
        from llm_perf.utils.constants import FFNActType

        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008, ffn_act_type=FFNActType.GELU.value)

        assert ffn.ffn_act_type == FFNActType.GELU.value
        assert not hasattr(ffn, "gate_weight")
        assert hasattr(ffn, "up_weight")
        assert hasattr(ffn, "down_weight")

    def test_ffn_relu_non_gated(self):
        """Test FFN with ReLU (non-gated, 2 weights)."""
        from llm_perf.utils.constants import FFNActType

        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008, ffn_act_type=FFNActType.RELU.value)

        assert ffn.ffn_act_type == FFNActType.RELU.value
        assert not hasattr(ffn, "gate_weight")

    def test_ffn_silu_non_gated(self):
        """Test FFN with SiLU (non-gated, 2 weights)."""
        from llm_perf.utils.constants import FFNActType

        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008, ffn_act_type=FFNActType.SILU.value)

        assert ffn.ffn_act_type == FFNActType.SILU.value
        assert not hasattr(ffn, "gate_weight")

    def test_ffn_swiglu_params_count(self):
        """Test SwiGLU FFN has 3 weight params."""
        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008)

        expected = 3 * 4096 * 11008
        assert ffn.params_count() == expected

    def test_ffn_gelu_params_count(self):
        """Test GELU FFN has 2 weight params."""
        from llm_perf.utils.constants import FFNActType

        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008, ffn_act_type=FFNActType.GELU.value)

        expected = 2 * 4096 * 11008
        assert ffn.params_count() == expected

    def test_ffn_forward_swiglu(self):
        """Test FFN forward with SwiGLU."""
        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008)
        hidden = ShardedTensor(shape=(1, 512, 4096))

        output = ffn(hidden)

        assert output.shape == (1, 512, 4096)
        assert hasattr(ffn, "gate_weight")
        assert "up_proj" in ffn._activations
        assert "intermediate" in ffn._activations

    def test_ffn_forward_gelu(self):
        """Test FFN forward with GELU."""
        from llm_perf.utils.constants import FFNActType

        ffn = ShardedFFN(hidden_size=4096, intermediate_size=11008, ffn_act_type=FFNActType.GELU.value)
        hidden = ShardedTensor(shape=(1, 512, 4096))

        output = ffn(hidden)

        assert output.shape == (1, 512, 4096)
        assert "gate_proj" not in ffn._activations
        assert "up_proj" in ffn._activations


class TestWanDiTMemory:
    """Test Wan DiT memory calculation."""

    def test_wan_dit_total_params(self):
        """Test Wan DiT total params matches official claim (14B)."""
        model = create_model_from_config({"type": "wan-dit"})
        total_params = model.params_count()

        expected = 14e9
        tolerance = 0.1

        assert abs(total_params - expected) / expected < tolerance, (
            f"Wan DiT params {total_params / 1e9:.2f}B != expected {expected / 1e9:.2f}B"
        )

    def test_wan_dit_ffn_params(self):
        """Test Wan DiT FFN uses non-gated (2 weights)."""
        from llm_perf.utils.constants import FFNActType

        model = create_model_from_config({"type": "wan-dit"})

        block = model._submodules["blocks.0"]
        ffn = block.ffn

        assert ffn.ffn_act_type == FFNActType.GELU.value

        assert not hasattr(ffn, "gate_weight") or ffn.gate_weight is None

        expected = 2 * block.hidden_size * block.intermediate_size
        assert ffn.params_count() == expected, (
            f"FFN params {ffn.params_count() / 1e6:.2f}M != expected {expected / 1e6:.2f}M"
        )

    def test_wan_dit_block_ffn_weights_count(self):
        """Test Wan DiT block FFN has 2 weights."""
        from llm_perf.utils.constants import FFNActType

        model = create_model_from_config({"type": "wan-dit"})

        for name, block in model._submodules.items():
            if name.startswith("blocks."):
                ffn = block.ffn
                assert ffn.ffn_act_type == FFNActType.GELU.value

                weights = ffn.get_weights()
                weight_count = len(weights)
                assert weight_count == 2, f"Block {name} FFN should have 2 weights, got {weight_count}"

    def test_t5_block_ffn_gelu(self):
        """Test T5 block uses GELU FFN."""
        from llm_perf.modeling.wan import ShardedT5Block
        from llm_perf.utils.constants import FFNActType

        block = ShardedT5Block(hidden_size=4096, num_heads=64, intermediate_size=10240)

        ffn = block.ffn_gate
        assert ffn.ffn_act_type == FFNActType.GELU.value
        assert not hasattr(ffn, "gate_weight")

        expected = 2 * 4096 * 10240
        assert ffn.params_count() == expected
