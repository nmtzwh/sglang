import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
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
