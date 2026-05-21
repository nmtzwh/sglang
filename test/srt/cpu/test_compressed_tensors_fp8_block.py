from types import SimpleNamespace

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.quant_args import QuantizationType
from torch.nn import Parameter

from sglang.srt.configs.model_config import _copy_quant_config_to_text_config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.model_loader.weight_utils import get_quant_config
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 as fp8_scheme_module


def _fp8_block_weight_args():
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.BLOCK,
        block_structure=[128, 128],
        dynamic=False,
    )


def _fp8_group_activation_args():
    return QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
        dynamic=True,
    )


def _fp8_block_compression_config():
    return {
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": _fp8_block_weight_args().model_dump(),
            }
        },
    }


def test_compressed_tensors_config_overrides_missing_quant_method():
    assert (
        CompressedTensorsConfig.override_quantization_method(
            _fp8_block_compression_config(), None
        )
        == "compressed-tensors"
    )


def test_gemma4_text_only_preserves_top_level_quantization_config():
    quantization_config = _fp8_block_compression_config()
    quantization_config["quant_method"] = "compressed-tensors"
    hf_config = SimpleNamespace(quantization_config=quantization_config)
    hf_text_config = SimpleNamespace()

    _copy_quant_config_to_text_config(hf_config, hf_text_config)

    assert hf_text_config.quantization_config is quantization_config


def test_text_config_compression_config_is_loaded():
    model_config = SimpleNamespace(
        quantization="compressed-tensors",
        hf_config=SimpleNamespace(
            quantization_config=None,
            compression_config=None,
            text_config=SimpleNamespace(
                quantization_config=None,
                compression_config=_fp8_block_compression_config(),
            ),
        ),
    )
    load_config = SimpleNamespace()

    config = get_quant_config(model_config, load_config, {"qkv_proj": ["q_proj"]})

    assert isinstance(config, CompressedTensorsConfig)
    assert "Linear" in config.target_scheme_map
    assert config.packed_modules_mapping == {"qkv_proj": ["q_proj"]}


def test_dict_text_config_compression_config_is_loaded():
    model_config = SimpleNamespace(
        quantization="compressed_tensors",
        hf_config={
            "text_config": {
                "compression_config": _fp8_block_compression_config(),
            },
        },
    )
    load_config = SimpleNamespace()

    config = get_quant_config(model_config, load_config, {})

    assert isinstance(config, CompressedTensorsConfig)
    assert "Linear" in config.target_scheme_map


def test_cpu_fp8_block_selects_native_amx_scheme():
    config = CompressedTensorsConfig({}, [], "float-quantized", {}, [])

    scheme = config._get_scheme_from_parts(
        _fp8_block_weight_args(), _fp8_group_activation_args()
    )

    assert isinstance(scheme, CompressedTensorsW8A8Fp8)
    assert scheme.use_cpu_fp8_kernel


def test_cpu_fp8_block_weight_only_selects_native_amx_scheme():
    config = CompressedTensorsConfig({}, [], "float-quantized", {}, [])

    scheme = config._get_scheme_from_parts(_fp8_block_weight_args(), None)

    assert isinstance(scheme, CompressedTensorsW8A8Fp8)
    assert scheme.use_cpu_fp8_kernel
    assert scheme.input_quant is None


def test_cpu_fp8_block_falls_back_for_unmatched_linear_target():
    fp8_block_scheme = {
        "weights": _fp8_block_weight_args(),
        "input_activations": None,
        "format": "float-quantized",
    }
    config = CompressedTensorsConfig(
        {"model.layers.*.self_attn.q_proj": fp8_block_scheme},
        [],
        "float-quantized",
        {},
        [],
    )
    layer = torch.nn.Linear(128, 128, bias=False)

    scheme_dict = config.get_scheme_dict(
        layer, "model.layers.0.self_attn.qkv_proj"
    )
    scheme = config.get_linear_scheme(layer, "model.layers.0.self_attn.qkv_proj")

    assert scheme_dict is fp8_block_scheme
    assert isinstance(scheme, CompressedTensorsW8A8Fp8)
    assert scheme.use_cpu_fp8_kernel


def test_cpu_fp8_block_process_aliases_native_scale(monkeypatch):
    scheme = CompressedTensorsW8A8Fp8(
        weight_quant=_fp8_block_weight_args(),
        input_quant=_fp8_group_activation_args(),
        is_static_input_scheme=False,
        use_cpu_fp8_kernel=True,
    )
    calls = []

    def fake_amx_process(layer, names):
        calls.append((layer, names))

    monkeypatch.setattr(fp8_scheme_module, "_is_cpu_amx_available", True)
    monkeypatch.setattr(
        fp8_scheme_module, "_amx_process_weight_after_loading", fake_amx_process
    )
    layer = SimpleNamespace(
        weight=Parameter(torch.empty((128, 128), dtype=torch.float8_e4m3fn)),
        weight_scale=Parameter(torch.ones((1, 1), dtype=torch.float32)),
        input_scale=Parameter(torch.ones((), dtype=torch.float32)),
    )
    layer.weight_scale.format_ue8m0 = False

    scheme.process_weights_after_loading(layer)

    assert calls == [(layer, ["weight"])]
    assert layer.input_scale is None
    assert torch.equal(layer.weight_scale_inv, layer.weight_scale)
    assert not layer.weight_scale_inv.requires_grad
    assert layer.weight_scale_inv.format_ue8m0 is False


def test_cpu_fp8_block_apply_uses_native_amx_kernel(monkeypatch):
    scheme = CompressedTensorsW8A8Fp8(
        weight_quant=_fp8_block_weight_args(),
        input_quant=_fp8_group_activation_args(),
        is_static_input_scheme=False,
        use_cpu_fp8_kernel=True,
    )
    captured = {}

    def fake_fp8_scaled_mm_cpu(
        input_tensor,
        weight,
        scales,
        block_size,
        bias,
        out_dtype,
        is_vnni,
    ):
        captured.update(
            input_tensor=input_tensor,
            weight=weight,
            scales=scales,
            block_size=block_size,
            bias=bias,
            out_dtype=out_dtype,
            is_vnni=is_vnni,
        )
        return torch.zeros((input_tensor.shape[0], weight.shape[0]), dtype=out_dtype)

    monkeypatch.setattr(
        torch.ops.sgl_kernel, "fp8_scaled_mm_cpu", fake_fp8_scaled_mm_cpu, raising=False
    )
    layer = SimpleNamespace(
        weight=torch.empty((128, 128), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.ones((1, 1), dtype=torch.float32),
    )
    x = torch.ones((2, 128), dtype=torch.bfloat16)

    out = scheme.apply_weights(layer, x)

    assert out.shape == (2, 128)
    assert captured["input_tensor"] is x
    assert captured["weight"] is layer.weight
    assert captured["scales"] is layer.weight_scale_inv
    assert captured["block_size"] == [128, 128]
    assert captured["out_dtype"] is torch.bfloat16
    assert captured["is_vnni"] is True
