import torch

import sgl_kernel  # noqa: F401


def test_sample_logits_cpu_selects_dominant_token():
    logits = torch.tensor(
        [[-100.0, 100.0, -100.0], [100.0, -100.0, -100.0]],
        dtype=torch.float32,
    )
    temperatures = torch.ones((2, 1), dtype=torch.float32)

    out = torch.ops.sgl_kernel.sample_logits_cpu(logits, temperatures)

    assert out.dtype == torch.int32
    torch.testing.assert_close(out, torch.tensor([1, 0], dtype=torch.int32))


def test_sample_logits_cpu_supports_bfloat16_and_flat_temperatures():
    logits = torch.tensor(
        [[-100.0, -100.0, 100.0], [-100.0, 100.0, -100.0]],
        dtype=torch.bfloat16,
    )
    temperatures = torch.ones(2, dtype=torch.float32)

    out = torch.ops.sgl_kernel.sample_logits_cpu(logits, temperatures)

    torch.testing.assert_close(out, torch.tensor([2, 1], dtype=torch.int32))


def test_sample_top_k_top_p_logits_cpu_applies_top_k():
    logits = torch.tensor([[10.0, 9.0, 8.0, 100.0]], dtype=torch.float32)
    temperatures = torch.ones((1, 1), dtype=torch.float32)
    top_ks = torch.tensor([1], dtype=torch.int32)
    top_ps = torch.ones(1, dtype=torch.float32)

    out = torch.ops.sgl_kernel.sample_top_k_top_p_logits_cpu(
        logits, temperatures, top_ks, top_ps
    )

    torch.testing.assert_close(out, torch.tensor([3], dtype=torch.int32))


def test_sample_top_k_top_p_logits_cpu_applies_top_p():
    # The first token alone crosses top_p, so all lower-ranked tokens are removed.
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16).repeat(64, 1)
    temperatures = torch.ones((64, 1), dtype=torch.float32)
    top_ks = torch.full((64,), 2**30, dtype=torch.int32)
    top_ps = torch.full((64,), 0.5, dtype=torch.float32)

    out = torch.ops.sgl_kernel.sample_top_k_top_p_logits_cpu(
        logits, temperatures, top_ks, top_ps
    )

    assert torch.all(out == 0)


def test_sample_top_k_top_p_logits_cpu_applies_filters_jointly():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]], dtype=torch.float32).repeat(64, 1)
    temperatures = torch.full((64,), 0.7, dtype=torch.float32)
    top_ks = torch.full((64,), 2, dtype=torch.int32)
    top_ps = torch.full((64,), 0.9, dtype=torch.float32)

    out = torch.ops.sgl_kernel.sample_top_k_top_p_logits_cpu(
        logits, temperatures, top_ks, top_ps
    )

    assert torch.all((out == 0) | (out == 1))
