import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


class _DummyConfig:
    model_type = "dummy"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class _FlatGemma4Config(SimpleNamespace):
    def update(self, values):
        for key, value in values.items():
            setattr(self, key, value)


def _load_hf_utils(monkeypatch, config):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    hf_hub = types.ModuleType("huggingface_hub")
    hf_hub.snapshot_download = lambda *args, **kwargs: ""
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_hub)

    transformers = types.ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def register(*args, **kwargs):
            return None

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return config

    transformers.AutoConfig = AutoConfig
    transformers.GenerationConfig = object
    transformers.AutoProcessor = object
    transformers.AutoTokenizer = object
    transformers.PretrainedConfig = object
    transformers.PreTrainedTokenizer = object
    transformers.PreTrainedTokenizerBase = object
    transformers.PreTrainedTokenizerFast = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    modeling_auto = types.ModuleType("transformers.models.auto.modeling_auto")
    modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {}
    monkeypatch.setitem(
        sys.modules, "transformers.models.auto.modeling_auto", modeling_auto
    )

    configs = types.ModuleType("sglang.srt.configs")
    for name in [
        "AfmoeConfig",
        "BailingHybridConfig",
        "ChatGLMConfig",
        "DbrxConfig",
        "DeepseekVL2Config",
        "DotsOCRConfig",
        "DotsVLMConfig",
        "ExaoneConfig",
        "FalconH1Config",
        "GraniteMoeHybridConfig",
        "InternVLChatConfig",
        "JetNemotronConfig",
        "JetVLMConfig",
        "KimiK25Config",
        "KimiLinearConfig",
        "KimiVLConfig",
        "LongcatFlashConfig",
        "MultiModalityConfig",
        "NemotronH_Nano_VL_V2_Config",
        "NemotronHConfig",
        "Olmo3Config",
        "Qwen3NextConfig",
        "Qwen3_5Config",
        "Qwen3_5MoeConfig",
        "Step3VLConfig",
        "Step3p5Config",
    ]:
        setattr(configs, name, type(name, (_DummyConfig,), {"model_type": name}))
    monkeypatch.setitem(sys.modules, "sglang.srt.configs", configs)

    deepseek_ocr = types.ModuleType("sglang.srt.configs.deepseek_ocr")
    deepseek_ocr.DeepseekVLV2Config = _DummyConfig
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.deepseek_ocr", deepseek_ocr)

    internvl = types.ModuleType("sglang.srt.configs.internvl")
    internvl.InternVLChatConfig = getattr(configs, "InternVLChatConfig")
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.internvl", internvl)

    connector = types.ModuleType("sglang.srt.connector")
    connector.create_remote_connector = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sglang.srt.connector", connector)

    customized_mm = types.ModuleType(
        "sglang.srt.multimodal.customized_mm_processor_utils"
    )
    customized_mm._CUSTOMIZED_MM_PROCESSOR = {}
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.multimodal.customized_mm_processor_utils",
        customized_mm,
    )

    utils = types.ModuleType("sglang.srt.utils")
    utils.get_bool_env_var = lambda *args, **kwargs: False
    utils.is_remote_url = lambda *args, **kwargs: False
    utils.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    utils.lru_cache_frozenset = lambda *args, **kwargs: (lambda fn: fn)
    utils.mistral_utils = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)

    patch_tokenizer = types.ModuleType("sglang.srt.utils.patch_tokenizer")
    patch_tokenizer.patch_tokenizer = lambda tokenizer: tokenizer
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.patch_tokenizer", patch_tokenizer)

    module_path = REPO_ROOT / "python/sglang/srt/utils/hf_transformers_utils.py"
    spec = importlib.util.spec_from_file_location(
        "hf_transformers_utils_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _load_model_loader_utils(monkeypatch, supported_archs):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    transformers = types.ModuleType("transformers")
    transformers.Gemma4ForCausalLM = object
    transformers.Gemma4AssistantForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    dynamic_module_utils = types.ModuleType("transformers.dynamic_module_utils")
    dynamic_module_utils.get_class_from_dynamic_module = lambda *args, **kwargs: object
    monkeypatch.setitem(
        sys.modules, "transformers.dynamic_module_utils", dynamic_module_utils
    )

    model_config_module = types.ModuleType("sglang.srt.configs.model_config")

    class ModelImpl:
        AUTO = "auto"
        TRANSFORMERS = "transformers"
        MINDSPORE = "mindspore"

    model_config_module.ModelConfig = object
    model_config_module.ModelImpl = ModelImpl
    monkeypatch.setitem(
        sys.modules, "sglang.srt.configs.model_config", model_config_module
    )

    deep_gemm_wrapper = SimpleNamespace(
        ENABLE_JIT_DEEPGEMM=False,
        DEEPGEMM_SCALE_UE8M0=False,
    )
    layers = types.ModuleType("sglang.srt.layers")
    layers.deep_gemm_wrapper = deep_gemm_wrapper
    monkeypatch.setitem(sys.modules, "sglang.srt.layers", layers)

    registry_module = types.ModuleType("sglang.srt.models.registry")

    class ModelRegistry:
        @staticmethod
        def get_supported_archs():
            return set(supported_archs)

        @staticmethod
        def resolve_model_cls(architectures):
            arch = architectures[0]
            return object, arch

    registry_module.ModelRegistry = ModelRegistry
    monkeypatch.setitem(sys.modules, "sglang.srt.models.registry", registry_module)

    module_path = REPO_ROOT / "python/sglang/srt/model_loader/utils.py"
    spec = importlib.util.spec_from_file_location(
        "model_loader_utils_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, ModelImpl


def test_flat_gemma4_assistant_config_remaps_swa_fields(monkeypatch):
    config = _FlatGemma4Config(
        architectures=["Gemma4AssistantForCausalLM"],
        model_type="gemma4_assistant",
        head_dim=128,
        global_head_dim=512,
        num_attention_heads=16,
        num_key_value_heads=1,
        num_global_key_value_heads=4,
    )
    hf_utils = _load_hf_utils(monkeypatch, config)

    normalized = hf_utils.get_config("draft", trust_remote_code=False)

    assert normalized is config
    assert normalized.head_dim == 512
    assert normalized.v_head_dim == 512
    assert normalized.swa_head_dim == 128
    assert normalized.swa_v_head_dim == 128
    assert normalized.num_key_value_heads == 4
    assert normalized.swa_num_key_value_heads == 1


def test_flat_gemma4_assistant_config_allows_empty_architectures(monkeypatch):
    config = _FlatGemma4Config(
        architectures=[],
        model_type="gemma4_assistant",
        head_dim=128,
        global_head_dim=512,
        num_attention_heads=16,
        num_key_value_heads=1,
        num_global_key_value_heads=4,
    )
    hf_utils = _load_hf_utils(monkeypatch, config)

    normalized = hf_utils.get_config("draft", trust_remote_code=False)

    assert normalized is config
    assert normalized.architectures == ["Gemma4AssistantForCausalLM"]
    assert normalized.head_dim == 512
    assert normalized.swa_head_dim == 128


def test_flat_gemma4_target_config_allows_empty_architectures(monkeypatch):
    config = _FlatGemma4Config(
        architectures=[],
        model_type="gemma4",
        head_dim=128,
        global_head_dim=512,
        num_attention_heads=16,
        num_key_value_heads=1,
        num_global_key_value_heads=4,
    )
    hf_utils = _load_hf_utils(monkeypatch, config)

    normalized = hf_utils.get_config("target", trust_remote_code=False)

    assert normalized is config
    assert normalized.architectures == ["Gemma4ForCausalLM"]
    assert normalized.head_dim == 512
    assert normalized.swa_head_dim == 128


def test_flat_gemma4_target_config_ignores_none_mm_configs(monkeypatch):
    config = _FlatGemma4Config(
        architectures=[],
        model_type="gemma4",
        head_dim=128,
        global_head_dim=512,
        num_attention_heads=16,
        num_key_value_heads=1,
        num_global_key_value_heads=4,
        vision_config=None,
        audio_config=None,
    )
    hf_utils = _load_hf_utils(monkeypatch, config)

    normalized = hf_utils.get_config("target", trust_remote_code=False)

    assert normalized is config
    assert normalized.architectures == ["Gemma4ForCausalLM"]


def test_flat_gemma4_text_config_allows_empty_architectures(monkeypatch):
    config = _FlatGemma4Config(
        architectures=[],
        model_type="gemma4_text",
        head_dim=128,
        global_head_dim=512,
        num_attention_heads=16,
        num_key_value_heads=1,
        num_global_key_value_heads=4,
    )
    hf_utils = _load_hf_utils(monkeypatch, config)

    normalized = hf_utils.get_config("target", trust_remote_code=False)

    assert normalized is config
    assert normalized.architectures == ["Gemma4ForCausalLM"]
    assert normalized.head_dim == 512
    assert normalized.swa_head_dim == 128


def test_multimodal_gemma4_config_allows_empty_architectures(monkeypatch):
    config = _FlatGemma4Config(
        architectures=[],
        model_type="gemma4",
        text_config=SimpleNamespace(
            head_dim=128,
            global_head_dim=512,
            num_attention_heads=16,
            num_key_value_heads=1,
            num_global_key_value_heads=4,
        ),
        vision_config=SimpleNamespace(),
    )
    hf_utils = _load_hf_utils(monkeypatch, config)

    normalized = hf_utils.get_config("target", trust_remote_code=False)

    assert normalized is config
    assert normalized.architectures == ["Gemma4ForConditionalGeneration"]
    assert normalized.text_config.head_dim == 512
    assert normalized.text_config.swa_head_dim == 128


def test_config_without_model_type_skips_gemma4_normalization(monkeypatch):
    config = _FlatGemma4Config(
        architectures=["OtherForCausalLM"],
        head_dim=128,
        num_attention_heads=16,
    )
    hf_utils = _load_hf_utils(monkeypatch, config)

    normalized = hf_utils.get_config("other", trust_remote_code=False)

    assert normalized is config
    assert normalized.architectures == ["OtherForCausalLM"]
    assert not hasattr(normalized, "swa_head_dim")


def test_normalized_gemma4_architectures_select_native_loader(monkeypatch):
    loader_utils, ModelImpl = _load_model_loader_utils(
        monkeypatch,
        supported_archs={"Gemma4ForCausalLM", "Gemma4AssistantForCausalLM"},
    )

    for arch in ["Gemma4ForCausalLM", "Gemma4AssistantForCausalLM"]:
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch]),
            quantization=None,
            model_impl=ModelImpl.AUTO,
        )

        _, resolved_arch = loader_utils.get_model_architecture(model_config)

        assert resolved_arch == arch
