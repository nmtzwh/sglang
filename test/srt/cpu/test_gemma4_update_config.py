import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


class _Config(SimpleNamespace):
    def get_total_num_kv_heads(self):
        return self.hf_text_config.num_key_value_heads


def _load_update_config(monkeypatch):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    vocab_parallel_embedding = types.ModuleType(
        "sglang.srt.layers.vocab_parallel_embedding"
    )
    vocab_parallel_embedding.pad_vocab_size = (
        lambda size, pad_to: ((size + pad_to - 1) // pad_to) * pad_to
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.vocab_parallel_embedding",
        vocab_parallel_embedding,
    )

    module_path = REPO_ROOT / "python/sglang/srt/configs/update_config.py"
    spec = importlib.util.spec_from_file_location(
        "update_config_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "may_get_weight_block_size", lambda *args: None)
    return module


def _real_gemma4_text_config():
    # Key fields from tmp/config.json.  The HF normalizer keeps Gemma4 full
    # attention fields on the base config and stores sliding-window values as
    # swa_* overrides.
    return SimpleNamespace(
        architectures=["Gemma4ForCausalLM"],
        model_type="gemma4_text",
        hidden_size=1536,
        intermediate_size=6144,
        num_attention_heads=8,
        num_key_value_heads=1,
        original_num_key_value_heads=1,
        head_dim=512,
        global_head_dim=512,
        swa_head_dim=256,
        swa_num_key_value_heads=1,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        vision_config=None,
    )


def test_gemma4_cpu_tp_adjusts_swa_kv_heads_from_real_dense_config(monkeypatch):
    update_config = _load_update_config(monkeypatch)
    text_config = _real_gemma4_text_config()
    model_config = _Config(
        hf_config=text_config,
        hf_text_config=text_config,
        hidden_size=text_config.hidden_size,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
    )

    adjusted = update_config.adjust_config_with_unaligned_cpu_tp(
        model_config, load_config=None, tp_size=2
    )

    assert adjusted.num_key_value_heads == 2
    assert adjusted.num_attention_heads == 16
    assert adjusted.hf_text_config.num_key_value_heads == 2
    assert adjusted.hf_text_config.swa_num_key_value_heads == 2
    assert adjusted.hf_text_config.original_total_num_kv_heads == 1


def test_gemma4_cpu_tp_ignores_dense_moe_none_fields(monkeypatch):
    update_config = _load_update_config(monkeypatch)
    text_config = _real_gemma4_text_config()
    text_config.num_experts = None
    text_config.moe_intermediate_size = None
    text_config.expert_intermediate_size = None
    model_config = _Config(
        hf_config=text_config,
        hf_text_config=text_config,
        hidden_size=text_config.hidden_size,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
    )

    adjusted = update_config.adjust_config_with_unaligned_cpu_tp(
        model_config, load_config=None, tp_size=2
    )

    assert adjusted.hf_text_config.moe_intermediate_size is None
    assert adjusted.hf_text_config.intermediate_size == 6144


def test_gemma4_cpu_tp_rejects_mixed_full_and_swa_kv_head_padding(monkeypatch):
    update_config = _load_update_config(monkeypatch)
    text_config = _real_gemma4_text_config()
    text_config.num_key_value_heads = 4
    text_config.swa_num_key_value_heads = 1
    model_config = _Config(
        hf_config=text_config,
        hf_text_config=text_config,
        hidden_size=text_config.hidden_size,
        num_attention_heads=16,
        num_key_value_heads=text_config.num_key_value_heads,
    )

    try:
        update_config.adjust_config_with_unaligned_cpu_tp(
            model_config, load_config=None, tp_size=8
        )
    except ValueError as exc:
        assert "different full-attention and sliding-window KV head counts" in str(exc)
    else:
        raise AssertionError("Expected unsupported mixed Gemma4 KV heads to fail")


def test_cpu_tp_siglip_vision_adjust_uses_head_dim(monkeypatch):
    update_config = _load_update_config(monkeypatch)
    text_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=4096,
        head_dim=128,
    )
    hf_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=4096,
        head_dim=128,
        vision_config=SimpleNamespace(
            model_type="siglip_vision_model",
            num_attention_heads=7,
            hidden_size=448,
            intermediate_size=2048,
        ),
    )
    model_config = _Config(
        hf_config=hf_config,
        hf_text_config=text_config,
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
    )

    adjusted = update_config.adjust_config_with_unaligned_cpu_tp(
        model_config, load_config=None, tp_size=4
    )

    assert adjusted.hf_config.vision_config.num_attention_heads == 8
    assert adjusted.hf_config.vision_config.head_dim == 64
