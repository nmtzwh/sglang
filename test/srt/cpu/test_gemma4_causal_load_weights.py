import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_gemma4_causal(monkeypatch):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    transformers = types.ModuleType("transformers")
    transformers.Gemma4TextConfig = object
    transformers.PretrainedConfig = object

    class PreTrainedModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

    transformers.PreTrainedModel = PreTrainedModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    distributed = types.ModuleType("sglang.srt.distributed")
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)

    fused_ops = types.ModuleType("sglang.srt.layers.gemma4_fused_ops")
    fused_ops.gemma_rmsnorm_residual_scalar = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.gemma4_fused_ops", fused_ops)

    layernorm = types.ModuleType("sglang.srt.layers.layernorm")
    layernorm.Gemma4RMSNorm = nn.Identity
    layernorm.RMSNorm = nn.Identity
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.layernorm", layernorm)

    linear = types.ModuleType("sglang.srt.layers.linear")
    linear.QKVParallelLinear = object
    linear.ReplicatedLinear = object
    linear.RowParallelLinear = object
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.linear", linear)

    logits_processor = types.ModuleType("sglang.srt.layers.logits_processor")
    logits_processor.LogitsProcessor = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.logits_processor", logits_processor
    )

    moe_layer = types.ModuleType("sglang.srt.layers.moe.ep_moe.layer")
    moe_layer.get_moe_impl_class = lambda quant_config: object
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.ep_moe.layer", moe_layer)

    topk = types.ModuleType("sglang.srt.layers.moe.topk")
    topk.TopK = object
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", topk)

    quantization = types.ModuleType("sglang.srt.layers.quantization.base_config")
    quantization.QuantizationConfig = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.quantization.base_config", quantization
    )

    radix_attention = types.ModuleType("sglang.srt.layers.radix_attention")
    radix_attention.RadixAttention = object
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.radix_attention", radix_attention)

    rotary = types.ModuleType("sglang.srt.layers.rotary_embedding")
    rotary.get_rope = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.rotary_embedding", rotary)

    vocab_embedding = types.ModuleType("sglang.srt.layers.vocab_parallel_embedding")
    vocab_embedding.ParallelLMHead = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.vocab_parallel_embedding", vocab_embedding
    )

    forward_batch = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_batch.ForwardBatch = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.model_executor.forward_batch_info", forward_batch
    )

    weight_utils = types.ModuleType("sglang.srt.model_loader.weight_utils")
    weight_utils.default_weight_loader = lambda *args, **kwargs: None
    weight_utils.maybe_remap_kv_scale_name = lambda name, params_dict: name
    monkeypatch.setitem(sys.modules, "sglang.srt.model_loader.weight_utils", weight_utils)

    gemma3 = types.ModuleType("sglang.srt.models.gemma3_causal")
    gemma3.Gemma3MLP = object
    gemma3.Gemma3TextScaledWordEmbedding = object
    monkeypatch.setitem(sys.modules, "sglang.srt.models.gemma3_causal", gemma3)

    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.get_global_server_args = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    utils = types.ModuleType("sglang.srt.utils")
    utils.add_prefix = lambda name, prefix: f"{prefix}.{name}" if prefix else name
    utils.make_layers = lambda *args, **kwargs: (0, [])
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)

    module_path = REPO_ROOT / "python/sglang/srt/models/gemma4_causal.py"
    spec = importlib.util.spec_from_file_location("gemma4_causal_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_dense_gemma4_load_weights_does_not_require_num_experts(monkeypatch):
    gemma4_causal = _load_gemma4_causal(monkeypatch)
    model = SimpleNamespace(
        config=SimpleNamespace(layer_types=[]),
        _get_k_eq_v_layers=lambda: set(),
        named_parameters=lambda: iter(()),
        named_buffers=lambda: iter(()),
        named_modules=lambda: iter(()),
    )

    loaded = gemma4_causal.Gemma4ForCausalLM.load_weights(model, [])

    assert loaded == set()


def test_gemma4_causal_load_weights_remaps_conditional_text_prefixes(monkeypatch):
    gemma4_causal = _load_gemma4_causal(monkeypatch)
    loaded_names = []

    param = nn.Parameter(torch.zeros(2, 3))
    param.weight_loader = lambda param, weight: loaded_names.append(weight.clone())
    model = SimpleNamespace(
        config=SimpleNamespace(layer_types=[]),
        _get_k_eq_v_layers=lambda: set(),
        named_parameters=lambda: iter([("model.embed_tokens.weight", param)]),
        named_buffers=lambda: iter(()),
        named_modules=lambda: iter(()),
    )
    loaded_weight = torch.ones(2, 3)

    loaded = gemma4_causal.Gemma4ForCausalLM.load_weights(
        model,
        [
            ("language_model.embed_tokens.weight", loaded_weight),
            ("model.vision_tower.patch_embedder.weight", torch.ones(2, 3)),
        ],
    )

    assert loaded == {"model.embed_tokens.weight"}
    assert len(loaded_names) == 1
    torch.testing.assert_close(loaded_names[0], loaded_weight)


def test_gemma4_causal_load_weights_remaps_nested_conditional_text_prefix(monkeypatch):
    gemma4_causal = _load_gemma4_causal(monkeypatch)
    loaded_names = []

    param = nn.Parameter(torch.zeros(2, 3))
    param.weight_loader = lambda param, weight: loaded_names.append(weight.clone())
    model = SimpleNamespace(
        config=SimpleNamespace(layer_types=[]),
        _get_k_eq_v_layers=lambda: set(),
        named_parameters=lambda: iter([("model.embed_tokens.weight", param)]),
        named_buffers=lambda: iter(()),
        named_modules=lambda: iter(()),
    )
    loaded_weight = torch.ones(2, 3)

    loaded = gemma4_causal.Gemma4ForCausalLM.load_weights(
        model, [("model.language_model.model.embed_tokens.weight", loaded_weight)]
    )

    assert loaded == {"model.embed_tokens.weight"}
    assert len(loaded_names) == 1
    torch.testing.assert_close(loaded_names[0], loaded_weight)


def test_gemma4_causal_load_weights_slices_attention_norm_to_head_dim(monkeypatch):
    gemma4_causal = _load_gemma4_causal(monkeypatch)
    loaded_values = []

    param = nn.Parameter(torch.zeros(256))
    param.weight_loader = lambda param, weight: loaded_values.append(weight.clone())
    model = SimpleNamespace(
        config=SimpleNamespace(layer_types=[]),
        _get_k_eq_v_layers=lambda: set(),
        named_parameters=lambda: iter([("model.layers.0.self_attn.q_norm.weight", param)]),
        named_buffers=lambda: iter(()),
        named_modules=lambda: iter(()),
    )
    loaded_weight = torch.arange(512, dtype=torch.float32)

    loaded = gemma4_causal.Gemma4ForCausalLM.load_weights(
        model, [("language_model.model.layers.0.self_attn.q_norm.weight", loaded_weight)]
    )

    assert loaded == {"model.layers.0.self_attn.q_norm.weight"}
    torch.testing.assert_close(loaded_values[0], loaded_weight[:256])


def test_gemma4_causal_load_weights_skips_nonlearned_v_norm(monkeypatch):
    gemma4_causal = _load_gemma4_causal(monkeypatch)

    param = torch.zeros(256)
    model = SimpleNamespace(
        config=SimpleNamespace(layer_types=[]),
        _get_k_eq_v_layers=lambda: set(),
        named_parameters=lambda: iter(()),
        named_buffers=lambda: iter([("model.layers.0.self_attn.v_norm.weight", param)]),
        named_modules=lambda: iter(()),
    )

    loaded = gemma4_causal.Gemma4ForCausalLM.load_weights(
        model, [("language_model.model.layers.0.self_attn.v_norm.weight", torch.ones(512))]
    )

    assert loaded == set()


def test_gemma4_causal_load_weights_reports_mismatched_name(monkeypatch):
    gemma4_causal = _load_gemma4_causal(monkeypatch)

    param = nn.Parameter(torch.zeros(2, 3))
    param.weight_loader = lambda param, weight: (_ for _ in ()).throw(
        AssertionError(f"{param.shape} != {weight.shape}")
    )
    model = SimpleNamespace(
        config=SimpleNamespace(layer_types=[]),
        _get_k_eq_v_layers=lambda: set(),
        named_parameters=lambda: iter([("model.embed_tokens.weight", param)]),
        named_buffers=lambda: iter(()),
        named_modules=lambda: iter(()),
    )

    with pytest.raises(RuntimeError, match="language_model.embed_tokens.weight"):
        gemma4_causal.Gemma4ForCausalLM.load_weights(
            model, [("language_model.embed_tokens.weight", torch.ones(3, 2))]
        )
