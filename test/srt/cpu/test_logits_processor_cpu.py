import torch

from sglang.srt.layers.logits_processor import fused_softcap


def test_fused_softcap_uses_torch_fallback_on_cpu():
    logits = torch.tensor(
        [[-60.0, -1.0, 0.0, 1.0, 60.0]], dtype=torch.bfloat16, device="cpu"
    )
    expected = 30.0 * torch.tanh(logits.float() / 30.0)

    output = fused_softcap(logits, 30.0)

    assert output is logits
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=0.125)


def test_fused_softcap_uses_inplace_float_cpu_path():
    logits = torch.tensor([[-60.0, -1.0, 0.0, 1.0, 60.0]], dtype=torch.float32)
    expected = 30.0 * torch.tanh(logits / 30.0)

    output = fused_softcap(logits, 30.0)

    assert output is logits
    torch.testing.assert_close(output, expected)
