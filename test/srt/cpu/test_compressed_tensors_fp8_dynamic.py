from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.quant_args import QuantizationType

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
)
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 as fp8_scheme_module
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe as fp8_moe_scheme_module
from sglang.srt.layers.amx_utils import CPUQuantMethod
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.srt.utils import cpu_has_amx_support, is_host_cpu_arm64


requires_native_fp8 = pytest.mark.skipif(
    not (cpu_has_amx_support() or is_host_cpu_arm64()),
    reason="native CPU FP8 requires AVX512/AMX or SVE",
)


def _channel_weight_args():
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
        dynamic=False,
    )


def _dynamic_token_args():
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )


def _block_weight_args():
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.BLOCK,
        block_structure=[128, 128],
        dynamic=False,
    )


def _dynamic_group_args():
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
        dynamic=True,
    )


def test_cpu_fp8_dynamic_selects_native_channel_scheme():
    config = CompressedTensorsConfig({}, [], "float-quantized", {}, [])

    scheme = config._get_scheme_from_parts(
        _channel_weight_args(), _dynamic_token_args()
    )

    assert isinstance(scheme, CompressedTensorsW8A8Fp8)
    assert scheme.use_cpu_fp8_kernel
    assert scheme.strategy == QuantizationStrategy.CHANNEL


def test_fp8_channel_weight_only_does_not_initialize_marlin():
    config = CompressedTensorsConfig({}, [], "float-quantized", {}, [])

    with pytest.raises(NotImplementedError, match="FP8_DYNAMIC"):
        config._get_scheme_from_parts(_channel_weight_args(), None)


def test_cpu_fp8_dynamic_process_preserves_channel_scales(monkeypatch):
    scheme = CompressedTensorsW8A8Fp8(
        weight_quant=_channel_weight_args(),
        input_quant=_dynamic_token_args(),
        is_static_input_scheme=False,
        use_cpu_fp8_kernel=True,
    )
    calls = []
    monkeypatch.setattr(fp8_scheme_module, "_is_cpu_amx_available", True)
    monkeypatch.setattr(
        fp8_scheme_module,
        "_amx_process_weight_after_loading",
        lambda layer, names: calls.append((layer, names)),
    )
    scales = torch.rand(64, 1, dtype=torch.float32)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.randn(64, 128).to(torch.float8_e4m3fn), requires_grad=False
        ),
        weight_scale=torch.nn.Parameter(scales.clone(), requires_grad=False),
    )

    scheme.process_weights_after_loading(layer)

    assert calls == [(layer, ["weight"])]
    torch.testing.assert_close(layer.weight_scale, scales)
    assert layer.weight.shape == (64, 128)
    assert layer.input_scale is None


def test_cpu_fp8_dynamic_moe_bypasses_gpu_runner(monkeypatch):
    scheme = CompressedTensorsW8A8Fp8MoE(
        _channel_weight_args(),
        _dynamic_token_args(),
        use_cpu_fp8_kernel=True,
    )
    scheme.create_moe_runner(
        SimpleNamespace(),
        SimpleNamespace(no_combine=False, apply_router_weight_on_input=False),
    )
    assert scheme.runner is None

    captured = {}

    def fake_fused_experts(*args):
        captured["args"] = args
        return torch.zeros_like(args[0])

    monkeypatch.setattr(
        torch.ops.sgl_kernel,
        "fused_experts_cpu",
        fake_fused_experts,
        raising=False,
    )
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 128, 64, dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty(2, 64, 64, dtype=torch.float8_e4m3fn),
        w13_weight_scale=torch.ones(2, 128, 1),
        w2_weight_scale=torch.ones(2, 64, 1),
    )
    dispatch = SimpleNamespace(
        hidden_states=x,
        topk_output=(topk_weights, topk_ids, None),
    )

    combined = scheme.apply_weights(layer, dispatch)

    assert combined.hidden_states.shape == x.shape
    assert captured["args"][6] == CPUQuantMethod.FP8_W8A16_CHANNEL
    assert captured["args"][11] is None
    assert captured["args"][12] is True


def test_cpu_fp8_block_moe_preserves_native_loading_and_dispatch(monkeypatch):
    scheme_dict = {
        "weights": _block_weight_args(),
        "input_activations": _dynamic_group_args(),
        "format": "float-quantized",
    }
    config = CompressedTensorsConfig(
        {"Linear": scheme_dict}, [], "float-quantized", {}, []
    )
    scheme = config.get_moe_scheme(
        torch.nn.Linear(128, 128), "model.layers.0.mlp.experts"
    )
    assert isinstance(scheme, CompressedTensorsW8A8Fp8MoE)
    assert scheme.use_cpu_fp8_kernel
    assert scheme.weight_block_size == [128, 128]

    pack_calls = []
    monkeypatch.setattr(
        fp8_moe_scheme_module,
        "_amx_process_weight_after_loading",
        lambda layer, names: pack_calls.append((layer, names)),
    )
    loaded_layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.empty(2, 256, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.empty(2, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.rand(2, 2, 1), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.rand(2, 1, 1), requires_grad=False
        ),
    )
    w13_scales = loaded_layer.w13_weight_scale.detach().clone()
    w2_scales = loaded_layer.w2_weight_scale.detach().clone()
    scheme.process_weights_after_loading(loaded_layer)
    assert pack_calls == [(loaded_layer, ["w13_weight", "w2_weight"])]
    torch.testing.assert_close(loaded_layer.w13_weight_scale, w13_scales)
    torch.testing.assert_close(loaded_layer.w2_weight_scale, w2_scales)

    scheme.create_moe_runner(
        SimpleNamespace(),
        SimpleNamespace(no_combine=False, apply_router_weight_on_input=False),
    )
    captured = {}

    def fake_fused_experts(*args):
        captured["args"] = args
        return torch.zeros_like(args[0])

    monkeypatch.setattr(
        torch.ops.sgl_kernel,
        "fused_experts_cpu",
        fake_fused_experts,
        raising=False,
    )
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 256, 128, dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty(2, 128, 128, dtype=torch.float8_e4m3fn),
        w13_weight_scale=torch.ones(2, 2, 1),
        w2_weight_scale=torch.ones(2, 1, 1),
    )
    dispatch = SimpleNamespace(
        hidden_states=x,
        topk_output=(
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
            None,
        ),
    )

    scheme.apply_weights(layer, dispatch)

    assert captured["args"][6] == CPUQuantMethod.FP8_W8A16
    assert captured["args"][11] == [128, 128]
    assert captured["args"][12] is True


def test_qwen3_next_loads_unfused_expert_weights_and_channel_scales():
    calls = []

    def make_param(name):
        param = torch.nn.Parameter(torch.empty(1), requires_grad=False)

        def weight_loader(param, loaded_weight, weight_name, shard_id, expert_id):
            calls.append(
                (name, loaded_weight.clone(), weight_name, shard_id, expert_id)
            )

        param.weight_loader = weight_loader
        return param

    prefix = "model.layers.0.mlp."
    params = {
        prefix + "experts.w13_weight": make_param("w13_weight"),
        prefix + "experts.w13_weight_scale": make_param("w13_weight_scale"),
        prefix + "experts.w2_weight": make_param("w2_weight"),
        prefix + "experts.w2_weight_scale": make_param("w2_weight_scale"),
    }
    fake_model = SimpleNamespace(
        config=SimpleNamespace(num_experts=1),
        named_parameters=lambda: params.items(),
    )
    weights = [
        (prefix + "experts.0.gate_proj.weight", torch.tensor([1.0])),
        (prefix + "experts.0.gate_proj.weight_scale", torch.tensor([2.0])),
        (prefix + "experts.0.down_proj.weight", torch.tensor([3.0])),
        (prefix + "experts.0.down_proj.weight_scale", torch.tensor([4.0])),
    ]

    loaded = Qwen3NextForCausalLM.load_weights(fake_model, weights)

    assert [call[0] for call in calls] == [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ]
    assert [call[3:] for call in calls] == [
        ("w1", 0),
        ("w1", 0),
        ("w2", 0),
        ("w2", 0),
    ]
    assert prefix + "experts.w13_weight_scale" in loaded
    assert prefix + "experts.w2_weight_scale" in loaded


@pytest.mark.parametrize("prepack", [False, True])
@pytest.mark.parametrize("m", [1, 11])
@pytest.mark.parametrize("has_bias", [False, True])
@requires_native_fp8
def test_fp8_channelwise_scaled_mm_cpu(prepack, m, has_bias):
    n, k = 64, 256
    x = (torch.randn(m, k) / k**0.5).to(torch.bfloat16)
    weight = (torch.randn(n, k) / k**0.5).to(torch.float8_e4m3fn)
    scales = torch.rand(n, 1, dtype=torch.float32) * 0.5 + 0.5
    bias = torch.randn(n, dtype=torch.float32) if has_bias else None
    reference = x.float().matmul(weight.float().t())
    reference.mul_(scales.view(1, n))
    if bias is not None:
        reference.add_(bias)
    reference = reference.to(torch.bfloat16)

    kernel_weight = (
        torch.ops.sgl_kernel.convert_weight_packed(weight) if prepack else weight
    )
    output = torch.ops.sgl_kernel.fp8_channelwise_scaled_mm_cpu(
        x,
        kernel_weight,
        scales,
        bias,
        torch.bfloat16,
        prepack,
    )

    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("prepack", [False, True])
@pytest.mark.parametrize("topk", [1, 2])
@requires_native_fp8
def test_fp8_channelwise_fused_moe_cpu(prepack, topk):
    m, n, k, experts = 7, 64, 128, 3
    x = (torch.randn(m, k) / k**0.5).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * n, k) / k**0.5).to(
        torch.float8_e4m3fn
    )
    w2 = (torch.randn(experts, k, n) / n**0.5).to(torch.float8_e4m3fn)
    w1_scale = torch.rand(experts, 2 * n, 1) * 0.5 + 0.5
    w2_scale = torch.rand(experts, k, 1) * 0.5 + 0.5
    topk_ids = torch.randint(experts, (m, topk), dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(m, topk), dim=-1)

    reference = torch.zeros_like(x, dtype=torch.float32)
    for token in range(m):
        for route in range(topk):
            expert = int(topk_ids[token, route])
            fc1 = x[token].float().matmul(w1[expert].float().t())
            fc1.mul_(w1_scale[expert, :, 0])
            hidden = F.silu(fc1[:n]) * fc1[n:]
            output = hidden.matmul(w2[expert].float().t())
            output.mul_(w2_scale[expert, :, 0])
            reference[token].add_(output * topk_weights[token, route])

    kernel_w1 = (
        torch.ops.sgl_kernel.convert_weight_packed(w1) if prepack else w1
    )
    kernel_w2 = (
        torch.ops.sgl_kernel.convert_weight_packed(w2) if prepack else w2
    )
    output = torch.ops.sgl_kernel.fused_experts_cpu(
        x,
        kernel_w1,
        kernel_w2,
        topk_weights,
        topk_ids,
        False,
        CPUQuantMethod.FP8_W8A16_CHANNEL,
        w1_scale,
        w2_scale,
        None,
        None,
        None,
        prepack,
    )

    torch.testing.assert_close(
        output, reference.to(torch.bfloat16), atol=4e-2, rtol=4e-2
    )
