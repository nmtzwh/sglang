"""Test GPTQ int4 CPU kernel against naive torch dequantization using real model weights.

Loads 1 layer from Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4 and verifies
the CPU GPTQ kernel output matches naive dequantize + matmul.
"""

import torch
import sgl_kernel  # registers torch.ops.sgl_kernel CPU ops
from safetensors.torch import load_file


# ── Naive reference implementation ──────────────────────────────────────────


def naive_dequantize_matmul(x, qweight, qzeros, scales, g_idx, bit=4):
    """Dequantize GPTQ-packed weights to fp16, then matmul. Reference."""
    pack_factor = 32 // bit
    K_pack, N = qweight.shape
    K = K_pack * pack_factor
    num_groups, N_pack = qzeros.shape
    assert N_pack * pack_factor == N

    # unpack qweight: (K//8, N) -> (K, N) as int32
    uw = torch.zeros(K, N, dtype=torch.int32, device=qweight.device)
    for i in range(pack_factor):
        uw[i::pack_factor, :] = (qweight >> (i * bit)) & ((1 << bit) - 1)

    # unpack qzeros: (num_groups, N//8) -> (num_groups, N) as int32
    uz = torch.zeros(num_groups, N, dtype=torch.int32, device=qzeros.device)
    for i in range(pack_factor):
        uz[:, i::pack_factor] = (qzeros >> (i * bit)) & ((1 << bit) - 1)

    # GPTQ v1 format: stored zero = actual_zero - 1
    uz = uz + 1

    # dequant in groups to avoid large intermediate allocation
    # fp_weight[i, :] = (qweight[i, :] - zero[g_idx[i], :]) * scale[g_idx[i], :]
    uw_f = uw.to(scales.dtype)
    uz_f = uz.to(scales.dtype)

    fp_w = torch.empty(K, N, dtype=scales.dtype, device=qweight.device)
    for g in range(num_groups):
        mask = g_idx == g
        fp_w[mask] = (uw_f[mask] - uz_f[g].unsqueeze(0)) * scales[g].unsqueeze(0)

    return x @ fp_w


# ── CPU kernel path (mirrors GPTQLinearMethod._process_weights_for_cpu) ────


def gptq_to_awq(qweight, qzeros):
    """Repack GPTQ int32-packed weights into AWQ int32-packed format."""
    K_pack, N = qweight.shape
    K = K_pack * 8
    num_groups, N_pack = qzeros.shape

    uw = torch.zeros(K, N, dtype=torch.int32, device=qweight.device)
    for i in range(8):
        uw[i::8, :] = (qweight >> (i * 4)) & 0x0F

    uz = torch.zeros(num_groups, N, dtype=torch.int32, device=qzeros.device)
    for i in range(8):
        uz[:, i::8] = (qzeros >> (i * 4)) & 0x0F

    # GPTQ v1 +1
    uz = uz + 1

    aw = torch.zeros(K, N // 8, dtype=torch.int32, device=qweight.device)
    az = torch.zeros(num_groups, N // 8, dtype=torch.int32, device=qzeros.device)
    shifts = [0, 16, 4, 20, 8, 24, 12, 28]
    for i in range(8):
        aw |= (uw[:, i::8] & 0xF) << shifts[i]
        az |= (uz[:, i::8] & 0xF) << shifts[i]

    return aw, az


def kernel_dequantize_matmul(x, qweight, qzeros, scales):
    """CPU GPTQ kernel path: AWQ repack -> convert -> int4_scaled_mm_cpu."""
    awq_w, awq_z = gptq_to_awq(qweight, qzeros)
    pw, pz, ps = torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
        awq_w, awq_z, scales
    )
    reshaped_x = x.reshape(-1, x.shape[-1])
    out = torch.ops.sgl_kernel.int4_scaled_mm_cpu(reshaped_x, pw, pz, ps, None)
    out_dim = ps.shape[0] * ps.shape[-1]  # N = Nc * block_n
    return out.reshape(x.shape[:-1] + (out_dim,))


# ── Main test ───────────────────────────────────────────────────────────────


def test_layer(ckpt_path, full_name, x):
    """Load one layer's weights from checkpoint, compare naive vs kernel."""
    sd = load_file(ckpt_path)

    qw = sd[f"{full_name}.qweight"]  # int32
    qz = sd[f"{full_name}.qzeros"]  # int32
    sc = sd[f"{full_name}.scales"]  # float16
    gi = sd[f"{full_name}.g_idx"]  # int32

    print(f"  qweight  {tuple(qw.shape)}  {qw.dtype}")
    print(f"  qzeros   {tuple(qz.shape)}  {qz.dtype}")
    print(f"  scales   {tuple(sc.shape)}  {sc.dtype}")
    print(f"  g_idx    {tuple(gi.shape)}  {gi.dtype}")

    # Show qzeros samples
    uz_sample = torch.zeros(1, 8, dtype=torch.int32)
    for i in range(8):
        uz_sample[0, i] = (qz[0, 0] >> (i * 4)) & 0x0F
    print(f"  qzeros[0,0] unpacked = {uz_sample.tolist()}  raw = {qz[0, 0].item()}")

    # reference
    ref = naive_dequantize_matmul(x, qw, qz, sc, gi)

    # kernel
    out = kernel_dequantize_matmul(x, qw, qz, sc)

    print(f"  ref shape {tuple(ref.shape)}")
    print(f"  out shape {tuple(out.shape)}")

    diff = (ref - out).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    max_idx_tuple = tuple(
        i.item() for i in torch.unravel_index(diff.argmax(), diff.shape)
    )
    ref_val = ref[max_idx_tuple].item()
    out_val = out[max_idx_tuple].item()
    print(f"  max_err={max_err:.6f}  mean_err={mean_err:.6f}")
    print(f"  worst elem ref={ref_val:.6f}  out={out_val:.6f}  at {max_idx_tuple}")

    torch.testing.assert_close(ref, out, rtol=1e-1, atol=1e-1)
    print(f"  PASSED")


def main():
    ckpt = "/home/nmtzwh/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct-GPTQ-Int4/snapshots/a3610ad4cf54e09d73ce14801beade7568287d5f/model.safetensors"

    layers = [
        # (layer_name, input_size, output_size)
        ("self_attn.q_proj", 2048, 2048),
        ("self_attn.k_proj", 2048, 256),
        ("self_attn.v_proj", 2048, 256),
        ("self_attn.o_proj", 2048, 2048),
        ("mlp.gate_proj", 2048, 11008),
        ("mlp.up_proj", 2048, 11008),
        ("mlp.down_proj", 11008, 2048),
    ]

    # Use layer 0
    M = 4  # small batch for quick test

    for name, K, N in layers:
        full_name = f"model.layers.0.{name}"
        print(f"\n── {full_name}  (M={M}, K={K}, N={N}) ──")
        x = torch.randn(M, K, dtype=torch.float16)
        test_layer(ckpt, full_name, x)

    # Test 3D input (batch, seq_len, hidden) to verify reshape fix
    print(f"\n── 3D input test (gate_proj, batch=2, seq=8, K=2048, N=11008) ──")
    x3d = torch.randn(2, 8, 2048, dtype=torch.float16)
    test_layer(ckpt, "model.layers.0.mlp.gate_proj", x3d)

    print("\n✅ All tests passed.")


if __name__ == "__main__":
    main()
