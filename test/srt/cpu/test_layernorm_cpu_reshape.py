import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_layernorm(monkeypatch):
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
    monkeypatch.setitem(sys.modules, "sglang", sglang_pkg)

    batch_invariant_ops = types.ModuleType("sglang.srt.batch_invariant_ops")
    batch_invariant_ops.is_batch_invariant_mode_enabled = lambda: False
    batch_invariant_ops.rms_norm_batch_invariant = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules, "sglang.srt.batch_invariant_ops", batch_invariant_ops
    )

    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM=None)
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", environ)

    layer_utils = types.ModuleType("sglang.srt.layers.utils")

    class MultiPlatformOp(nn.Module):
        def __init__(self):
            super().__init__()

    layer_utils.MultiPlatformOp = MultiPlatformOp
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.utils", layer_utils)

    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.get_global_server_args = lambda: SimpleNamespace(
        rl_on_policy_target=None
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    utils = types.ModuleType("sglang.srt.utils")
    utils.cpu_has_amx_support = lambda: True
    utils.get_bool_env_var = lambda *args, **kwargs: False
    utils.is_cpu = lambda: True
    utils.is_cuda = lambda: False
    utils.is_flashinfer_available = lambda: False
    utils.is_hip = lambda: False
    utils.is_npu = lambda: False
    utils.is_xpu = lambda: False
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)

    module_path = REPO_ROOT / "python/sglang/srt/layers/layernorm.py"
    spec = importlib.util.spec_from_file_location("layernorm_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_rmsnorm_cpu_flattens_higher_rank_input_for_kernel(monkeypatch):
    layernorm = _load_layernorm(monkeypatch)
    seen_shape = None

    def rmsnorm_cpu(x, weight, eps):
        nonlocal seen_shape
        seen_shape = x.shape
        assert x.dim() == 2
        return x + 1

    monkeypatch.setattr(torch.ops.sgl_kernel, "rmsnorm_cpu", rmsnorm_cpu, raising=False)

    x = torch.zeros(2, 3, 4)
    out = layernorm.RMSNorm(4).forward_cpu(x)

    assert seen_shape == torch.Size([6, 4])
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.ones_like(x))


def test_gemma_rmsnorm_cpu_flattens_higher_rank_input_for_kernel(monkeypatch):
    layernorm = _load_layernorm(monkeypatch)
    seen_shape = None

    def gemma_rmsnorm_cpu(x, weight, eps):
        nonlocal seen_shape
        seen_shape = x.shape
        assert x.dim() == 2
        return x + 1

    monkeypatch.setattr(
        torch.ops.sgl_kernel, "gemma_rmsnorm_cpu", gemma_rmsnorm_cpu, raising=False
    )

    x = torch.zeros(2, 3, 4)
    out = layernorm.GemmaRMSNorm(4).forward_cpu(x)

    assert seen_shape == torch.Size([6, 4])
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.ones_like(x))


def test_gemma3_rmsnorm_cpu_flattens_higher_rank_input_for_kernel(monkeypatch):
    layernorm = _load_layernorm(monkeypatch)
    seen_shape = None

    def gemma3_rmsnorm_cpu(x, weight, eps):
        nonlocal seen_shape
        seen_shape = x.shape
        assert x.dim() == 2
        return x + 1

    monkeypatch.setattr(
        torch.ops.sgl_kernel, "gemma3_rmsnorm_cpu", gemma3_rmsnorm_cpu, raising=False
    )

    x = torch.zeros(2, 3, 4)
    out = layernorm.Gemma3RMSNorm(4).forward_cpu(x)

    assert seen_shape == torch.Size([6, 4])
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.ones_like(x))


def test_gemma4_rmsnorm_cpu_flattens_higher_rank_input_for_kernel(monkeypatch):
    layernorm = _load_layernorm(monkeypatch)
    seen_shape = None

    def rmsnorm_cpu(x, weight, eps):
        nonlocal seen_shape
        seen_shape = x.shape
        assert x.dim() == 2
        return x + 1

    monkeypatch.setattr(torch.ops.sgl_kernel, "rmsnorm_cpu", rmsnorm_cpu, raising=False)

    x = torch.zeros(2, 3, 4)
    out = layernorm.Gemma4RMSNorm(4).forward_cpu(x)

    assert seen_shape == torch.Size([6, 4])
    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.ones_like(x))


def test_gemma4_rmsnorm_cpu_uses_gemma_shift_kernel(monkeypatch):
    layernorm = _load_layernorm(monkeypatch)
    called = False

    def gemma_rmsnorm_cpu(x, weight, eps):
        nonlocal called
        called = True
        assert x.dim() == 2
        return x + 2

    monkeypatch.setattr(
        torch.ops.sgl_kernel, "gemma_rmsnorm_cpu", gemma_rmsnorm_cpu, raising=False
    )

    x = torch.zeros(2, 4)
    out = layernorm.Gemma4RMSNorm(4, scale_shift=1.0).forward_cpu(x)

    assert called
    torch.testing.assert_close(out, torch.full_like(x, 2))
