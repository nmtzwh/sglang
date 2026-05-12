import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


class _EnvValue:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value


def _load_model_config(monkeypatch):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    transformers = types.ModuleType("transformers")
    transformers.PretrainedConfig = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(SGLANG_EXTERNAL_MM_MODEL_ARCH=_EnvValue(None))
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", environ)

    quantization = types.ModuleType("sglang.srt.layers.quantization")
    quantization.QUANTIZATION_METHODS = {}
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization", quantization)

    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    utils = types.ModuleType("sglang.srt.utils")
    utils.is_hip = lambda: False
    utils.is_sm100_supported = lambda: False
    utils.retry = lambda *args, **kwargs: (lambda fn: fn)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)

    hf_utils = types.ModuleType("sglang.srt.utils.hf_transformers_utils")
    hf_utils.get_config = lambda *args, **kwargs: None
    hf_utils.get_context_length = lambda *args, **kwargs: 0
    hf_utils.get_generation_config = lambda *args, **kwargs: None
    hf_utils.get_hf_text_config = lambda config: getattr(config, "text_config", config)
    hf_utils.get_sparse_attention_config = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.hf_transformers_utils", hf_utils)

    sglang_utils = types.ModuleType("sglang.utils")
    sglang_utils.is_in_ci = lambda: False
    monkeypatch.setitem(sys.modules, "sglang.utils", sglang_utils)

    module_path = REPO_ROOT / "python/sglang/srt/configs/model_config.py"
    spec = importlib.util.spec_from_file_location(
        "model_config_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_gemma4_assistant_is_hybrid_swa_model(monkeypatch):
    model_config = _load_model_config(monkeypatch)

    assert model_config.is_hybrid_swa_model(["Gemma4AssistantForCausalLM"])


def test_gemma4_assistant_hybrid_layer_ids_follow_layer_types(monkeypatch):
    model_config = _load_model_config(monkeypatch)
    cfg = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )

    swa_ids, full_ids = model_config.get_hybrid_layer_ids(
        ["Gemma4AssistantForCausalLM"], cfg
    )

    assert swa_ids == [0, 2]
    assert full_ids == [1, 3]
