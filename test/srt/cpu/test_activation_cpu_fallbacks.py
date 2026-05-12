import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_activation(monkeypatch):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    transformers = types.ModuleType("transformers")
    transformers.PretrainedConfig = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    distributed = types.ModuleType("sglang.srt.distributed")
    distributed.divide = lambda a, b: a // b
    distributed.get_tensor_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)

    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(
        SGLANG_NPU_FORWARD_NATIVE_GELUTANH=SimpleNamespace(get=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", environ)

    quant_base = types.ModuleType("sglang.srt.layers.quantization.base_config")
    quant_base.QuantizationConfig = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.quantization.base_config", quant_base
    )

    layer_utils = types.ModuleType("sglang.srt.layers.utils")

    class MultiPlatformOp(nn.Module):
        pass

    layer_utils.MultiPlatformOp = MultiPlatformOp
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.utils", layer_utils)

    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.get_global_server_args = lambda: SimpleNamespace(
        rl_on_policy_target=None
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    srt_utils = types.ModuleType("sglang.srt.utils")
    srt_utils.cpu_has_amx_support = lambda: False
    srt_utils.is_cpu = lambda: True
    srt_utils.is_cuda = lambda: False
    srt_utils.is_hip = lambda: False
    srt_utils.is_npu = lambda: False
    srt_utils.is_xpu = lambda: False
    srt_utils.set_weight_attrs = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", srt_utils)

    sglang_utils = types.ModuleType("sglang.utils")
    sglang_utils.resolve_obj_by_qualname = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sglang.utils", sglang_utils)

    module_path = REPO_ROOT / "python/sglang/srt/layers/activation.py"
    spec = importlib.util.spec_from_file_location("activation_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_gelu_and_mul_cpu_uses_torch_fallback_when_sgl_kernel_op_missing(monkeypatch):
    activation = _load_activation(monkeypatch)
    monkeypatch.setattr(activation, "_get_sgl_kernel_cpu_op", lambda name: None)

    x = torch.randn(3, 8, dtype=torch.float32)

    out = activation.GeluAndMul("tanh").forward_cpu(x)
    ref = F.gelu(x[..., :4], approximate="tanh") * x[..., 4:]

    torch.testing.assert_close(out, ref)


def test_silu_and_mul_cpu_uses_torch_fallback_when_sgl_kernel_op_missing(monkeypatch):
    activation = _load_activation(monkeypatch)
    monkeypatch.setattr(activation, "_get_sgl_kernel_cpu_op", lambda name: None)

    x = torch.randn(3, 8, dtype=torch.float32)

    out = activation.SiluAndMul().forward_cpu(x)
    ref = F.silu(x[..., :4]) * x[..., 4:]

    torch.testing.assert_close(out, ref)
