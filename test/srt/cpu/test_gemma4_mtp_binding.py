import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]


def _install_gemma4_mtp_stubs(monkeypatch):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        pass

    class PreTrainedModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

    transformers.PretrainedConfig = PretrainedConfig
    transformers.PreTrainedModel = PreTrainedModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    linear = types.ModuleType("sglang.srt.layers.linear")
    linear.ReplicatedLinear = object
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.linear", linear)

    logits_processor = types.ModuleType("sglang.srt.layers.logits_processor")
    logits_processor.LogitsMetadata = object
    logits_processor.LogitsProcessor = object
    logits_processor.LogitsProcessorOutput = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.logits_processor", logits_processor
    )

    quantization = types.ModuleType("sglang.srt.layers.quantization.base_config")
    quantization.QuantizationConfig = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.quantization.base_config", quantization
    )

    memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.KVCache = object
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.memory_pool", memory_pool)

    forward_batch = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_batch.ForwardBatch = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.model_executor.forward_batch_info", forward_batch
    )

    gemma4_causal = types.ModuleType("sglang.srt.models.gemma4_causal")
    gemma4_causal.Gemma4ForCausalLM = PreTrainedModel
    gemma4_causal.Gemma4TextModel = object
    monkeypatch.setitem(sys.modules, "sglang.srt.models.gemma4_causal", gemma4_causal)

    frozen_kv_mtp_info = types.ModuleType("sglang.srt.speculative.frozen_kv_mtp_info")

    @dataclass(frozen=True)
    class FrozenKVMTPContext:
        target_token_to_kv_pool: object
        physical_layer_ids: dict[int, int]

        def get_physical_layer_id(self, idx: int) -> int:
            return self.physical_layer_ids[idx]

    frozen_kv_mtp_info.FrozenKVMTPContext = FrozenKVMTPContext
    monkeypatch.setitem(
        sys.modules, "sglang.srt.speculative.frozen_kv_mtp_info", frozen_kv_mtp_info
    )

    utils = types.ModuleType("sglang.srt.utils")
    utils.add_prefix = lambda name, prefix: f"{prefix}.{name}" if prefix else name
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)


def _load_gemma4_mtp(monkeypatch):
    _install_gemma4_mtp_stubs(monkeypatch)
    module_path = REPO_ROOT / "python/sglang/srt/models/gemma4_mtp.py"
    spec = importlib.util.spec_from_file_location("gemma4_mtp_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _target_layer(is_shared=False, owner=None):
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            is_kv_shared_layer=is_shared,
            kv_shared_layer_index=owner,
        )
    )


def test_build_frozen_kv_mtp_context_collapses_target_kv_shared_layers(monkeypatch):
    gemma4_mtp = _load_gemma4_mtp(monkeypatch)
    assistant = gemma4_mtp.Gemma4AssistantForCausalLM.__new__(
        gemma4_mtp.Gemma4AssistantForCausalLM
    )
    assistant.config = SimpleNamespace(
        layer_types=["full_attention", "sliding_attention"]
    )

    target_layers = [
        _target_layer(),
        _target_layer(),
        _target_layer(is_shared=True, owner=0),
        _target_layer(is_shared=True, owner=1),
    ]
    target_model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        ),
        model=SimpleNamespace(layers=target_layers),
    )
    kv_pool = object()

    ctx = assistant.build_frozen_kv_mtp_context(target_model, kv_pool)

    assert ctx.target_token_to_kv_pool is kv_pool
    assert ctx.physical_layer_ids == {0: 1, 1: 0}


def test_bind_frozen_kv_context_rewrites_assistant_attention_layers(monkeypatch):
    gemma4_mtp = _load_gemma4_mtp(monkeypatch)
    assistant = gemma4_mtp.Gemma4AssistantForCausalLM.__new__(
        gemma4_mtp.Gemma4AssistantForCausalLM
    )
    assistant_layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                is_kv_shared_layer=False,
                kv_shared_layer_index=None,
                attn=SimpleNamespace(layer_id=0),
                layer_id=0,
            )
        ),
        SimpleNamespace(
            self_attn=SimpleNamespace(
                is_kv_shared_layer=False,
                kv_shared_layer_index=None,
                attn=SimpleNamespace(layer_id=1),
                layer_id=1,
            )
        ),
    ]
    assistant.model = SimpleNamespace(layers=assistant_layers)
    ctx = gemma4_mtp.FrozenKVMTPContext(
        target_token_to_kv_pool=object(),
        physical_layer_ids={0: 7, 1: 3},
    )

    assistant.bind_frozen_kv_context(ctx)

    assert assistant.kv_context is ctx
    assert [layer.self_attn.is_kv_shared_layer for layer in assistant_layers] == [
        True,
        True,
    ]
    assert [layer.self_attn.kv_shared_layer_index for layer in assistant_layers] == [
        7,
        3,
    ]
    assert [layer.self_attn.attn.layer_id for layer in assistant_layers] == [7, 3]
    assert [layer.self_attn.layer_id for layer in assistant_layers] == [0, 1]


def test_build_frozen_kv_mtp_context_rejects_nested_kv_shared_owner(monkeypatch):
    gemma4_mtp = _load_gemma4_mtp(monkeypatch)
    assistant = gemma4_mtp.Gemma4AssistantForCausalLM.__new__(
        gemma4_mtp.Gemma4AssistantForCausalLM
    )
    assistant.config = SimpleNamespace(layer_types=["sliding_attention"])

    target_layers = [
        _target_layer(is_shared=True, owner=0),
        _target_layer(is_shared=True, owner=0),
        _target_layer(is_shared=True, owner=0),
    ]
    target_model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=3,
            layer_types=["sliding_attention"] * 3,
        ),
        model=SimpleNamespace(layers=target_layers),
    )

    with pytest.raises(RuntimeError, match="itself KV-shared"):
        assistant.build_frozen_kv_mtp_context(target_model, object())
