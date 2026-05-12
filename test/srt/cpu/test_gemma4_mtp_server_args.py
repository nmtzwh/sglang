import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


class _EnvValue:
    def __init__(self, value=None):
        self.value = value
        self.is_set_value = False

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        self.is_set_value = True

    def is_set(self):
        return self.is_set_value


def _install_server_args_stubs(monkeypatch):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    connector = types.ModuleType("sglang.srt.connector")
    connector.ConnectorType = object
    monkeypatch.setitem(sys.modules, "sglang.srt.connector", connector)

    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(
        SGLANG_ENABLE_SPEC_V2=_EnvValue(False),
        SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=_EnvValue(False),
        SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=_EnvValue(None),
        SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK=_EnvValue(None),
        SGLANG_SYMM_MEM_PREALLOC_GB_SIZE=_EnvValue(None),
        SGLANG_NSA_FORCE_MLA=_EnvValue(False),
        SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=_EnvValue(False),
        SGLANG_EMBEDDINGS_SPARSE_HEAD=_EnvValue(False),
        SGLANG_MOE_NVFP4_DISPATCH=_EnvValue(False),
        SGLANG_ENABLE_TORCH_COMPILE=_EnvValue(False),
        SGLANG_MAMBA_SSM_DTYPE=_EnvValue(None),
        SGLANG_DISABLE_OUTLINES_DISK_CACHE=_EnvValue(False),
        SGLANG_ENABLE_DETERMINISTIC_INFERENCE=_EnvValue(False),
        SGLANG_SPEC_NAN_DETECTION=_EnvValue(False),
        SGLANG_SPEC_OOB_DETECTION=_EnvValue(False),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", environ)

    function_call_parser = types.ModuleType(
        "sglang.srt.function_call.function_call_parser"
    )
    function_call_parser.FunctionCallParser = SimpleNamespace(
        get_parsers=lambda: [],
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.function_call.function_call_parser",
        function_call_parser,
    )

    chunk_delta_h = types.ModuleType("sglang.srt.layers.attention.fla.chunk_delta_h")
    chunk_delta_h.CHUNK_SIZE = 64
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.attention.fla.chunk_delta_h", chunk_delta_h
    )

    lora_registry = types.ModuleType("sglang.srt.lora.lora_registry")
    lora_registry.LoRARef = object
    monkeypatch.setitem(sys.modules, "sglang.srt.lora.lora_registry", lora_registry)

    reasoning_parser = types.ModuleType("sglang.srt.parser.reasoning_parser")
    reasoning_parser.ReasoningParser = SimpleNamespace(get_parsers=lambda: [])
    monkeypatch.setitem(
        sys.modules, "sglang.srt.parser.reasoning_parser", reasoning_parser
    )

    common = types.ModuleType("sglang.srt.utils.common")
    common.LORA_TARGET_ALL_MODULES = "all"
    common.SUPPORTED_LORA_TARGET_MODULES = []
    common.configure_ipv6 = lambda *args, **kwargs: None
    common.cpu_has_amx_support = lambda: True
    common.get_bool_env_var = lambda *args, **kwargs: False
    common.get_device = lambda: "cpu"
    common.get_device_memory_capacity = lambda *args, **kwargs: 0
    common.get_device_name = lambda *args, **kwargs: "cpu"
    common.get_device_sm = lambda *args, **kwargs: 0
    common.get_free_port = lambda *args, **kwargs: 30000
    common.get_int_env_var = lambda *args, **kwargs: None
    common.get_quantization_config = lambda *args, **kwargs: None
    common.is_blackwell_supported = lambda: False
    common.is_cpu = lambda: True
    common.is_cuda = lambda: False
    common.is_flashinfer_available = lambda: False
    common.is_hip = lambda: False
    common.is_hopper_with_cuda_12_3 = lambda: False
    common.is_host_cpu_arm64 = lambda: False
    common.is_no_spec_infer_or_topk_one = lambda *args, **kwargs: True
    common.is_npu = lambda: False
    common.is_remote_url = lambda *args, **kwargs: False
    common.is_sm90_supported = lambda: False
    common.is_sm100_supported = lambda: False
    common.is_sm120_supported = lambda: False
    common.is_triton_kernels_available = lambda: False
    common.is_valid_ipv6_address = lambda *args, **kwargs: False
    common.json_list_type = lambda value: value
    common.nullable_str = lambda value: value
    common.parse_connector_type = lambda value: value
    common.torch_release = "0"
    common.wait_port_available = lambda *args, **kwargs: None
    common.xpu_has_xmx_support = lambda: False
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.common", common)

    hf_utils = types.ModuleType("sglang.srt.utils.hf_transformers_utils")
    hf_utils.check_gguf_file = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.hf_transformers_utils", hf_utils)

    sglang_utils = types.ModuleType("sglang.utils")
    sglang_utils.is_in_ci = lambda: False
    monkeypatch.setitem(sys.modules, "sglang.utils", sglang_utils)

    moe_utils = types.ModuleType("sglang.srt.layers.moe.utils")

    class MoeRunnerBackend:
        def __init__(self, value):
            self.value = value

        def is_flashinfer_trtllm(self):
            return self.value == "flashinfer_trtllm"

    moe_utils.MoeRunnerBackend = MoeRunnerBackend
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.utils", moe_utils)


def _load_server_args(monkeypatch):
    _install_server_args_stubs(monkeypatch)
    module_path = REPO_ROOT / "python/sglang/srt/server_args.py"
    spec = importlib.util.spec_from_file_location(
        "server_args_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_gemma4_assistant_draft_promotes_nextn_and_eagle(monkeypatch):
    server_args = _load_server_args(monkeypatch)

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(architectures=["Gemma4AssistantForCausalLM"])

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert (
        server_args._resolve_speculative_algorithm_alias("NEXTN", "draft")
        == "FROZEN_KV_MTP"
    )
    assert (
        server_args._resolve_speculative_algorithm_alias("EAGLE", "draft")
        == "FROZEN_KV_MTP"
    )
    with pytest.raises(ValueError, match="EAGLE3"):
        server_args._resolve_speculative_algorithm_alias("EAGLE3", "draft")


def test_gemma4_assistant_draft_promotes_by_model_type(monkeypatch):
    server_args = _load_server_args(monkeypatch)

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(architectures=[], model_type="gemma4_assistant")

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert (
        server_args._resolve_speculative_algorithm_alias("NEXTN", "draft")
        == "FROZEN_KV_MTP"
    )
    assert (
        server_args._resolve_speculative_algorithm_alias("EAGLE", "draft")
        == "FROZEN_KV_MTP"
    )
    with pytest.raises(ValueError, match="EAGLE3"):
        server_args._resolve_speculative_algorithm_alias("EAGLE3", "draft")


def _cpu_frozen_kv_args(server_args):
    args = server_args.ServerArgs.__new__(server_args.ServerArgs)
    args.speculative_draft_model_path = None
    args.speculative_draft_model_revision = None
    args.speculative_moe_runner_backend = None
    args.moe_runner_backend = "auto"
    args.speculative_algorithm = "FROZEN_KV_MTP"
    args.trust_remote_code = False
    args.speculative_skip_dp_mlp_sync = True
    args.device = "cpu"
    args.attention_backend = "intel_amx"
    args.speculative_eagle_topk = None
    args.speculative_num_steps = 3
    args.speculative_num_draft_tokens = None
    args.max_running_requests = None
    args.enable_mixed_chunk = True
    args.disable_overlap_schedule = False
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Gemma4ForCausalLM"],
            enable_moe_block=False,
        )
    )
    return args


def test_cpu_frozen_kv_mtp_validation_normalizes_topk_and_draft_tokens(monkeypatch):
    server_args = _load_server_args(monkeypatch)
    args = _cpu_frozen_kv_args(server_args)

    server_args.ServerArgs._handle_speculative_decoding(args)

    assert args.speculative_eagle_topk == 1
    assert args.speculative_num_draft_tokens == 4
    assert args.disable_overlap_schedule is True
    assert args.enable_mixed_chunk is False
    assert args.max_running_requests == 48


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("attention_backend", "torch_native", "attention-backend intel_amx"),
        ("speculative_eagle_topk", 2, "speculative-eagle-topk 1"),
        ("speculative_num_steps", None, "speculative-num-steps"),
    ],
)
def test_cpu_frozen_kv_mtp_validation_rejects_unsupported_options(
    monkeypatch, field, value, match
):
    server_args = _load_server_args(monkeypatch)
    args = _cpu_frozen_kv_args(server_args)
    setattr(args, field, value)

    with pytest.raises(ValueError, match=match):
        server_args.ServerArgs._handle_speculative_decoding(args)


@pytest.mark.parametrize(
    ("arch", "enable_moe_block", "match"),
    [
        ("Gemma4ForConditionalGeneration", False, "text-only Gemma4ForCausalLM"),
        ("Gemma4ForCausalLM", True, "dense Gemma4 only"),
    ],
)
def test_cpu_frozen_kv_mtp_validation_rejects_multimodal_and_moe(
    monkeypatch, arch, enable_moe_block, match
):
    server_args = _load_server_args(monkeypatch)
    args = _cpu_frozen_kv_args(server_args)
    args.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[arch],
            enable_moe_block=enable_moe_block,
        )
    )

    with pytest.raises(ValueError, match=match):
        server_args.ServerArgs._handle_speculative_decoding(args)
