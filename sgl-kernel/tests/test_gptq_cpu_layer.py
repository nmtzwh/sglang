"""Run a full Qwen2 layer (Attention + FFN) on CPU with GPTQ int4 kernel and check for NaNs."""

import torch
import sgl_kernel
from safetensors.torch import load_file


# ── GPTQ CPU helpers (mirror _process_weights_for_cpu) ──────────────────────


def process_gptq_for_cpu(qweight, qzeros, scales):
    """Repack GPTQ weights into the format expected by int4_scaled_mm_cpu."""
    K_pack, N = qweight.shape
    K = K_pack * 8
    num_groups, N_pack = qzeros.shape

    uw = torch.zeros(K, N, dtype=torch.int32)
    for i in range(8):
        uw[i::8, :] = (qweight >> (i * 4)) & 0x0F

    uz = torch.zeros(num_groups, N, dtype=torch.int32)
    for i in range(8):
        uz[:, i::8] = (qzeros >> (i * 4)) & 0x0F
    uz += 1  # GPTQ v1

    aw = torch.zeros(K, N // 8, dtype=torch.int32)
    az = torch.zeros(num_groups, N // 8, dtype=torch.int32)
    shifts = [0, 16, 4, 20, 8, 24, 12, 28]
    for i in range(8):
        aw |= (uw[:, i::8] & 0xF) << shifts[i]
        az |= (uz[:, i::8] & 0xF) << shifts[i]

    pw, pz, ps = torch.ops.sgl_kernel.convert_weight_packed_scale_zp(aw, az, scales)
    return pw, pz, ps


def gptq_linear(x, pw, pz, ps, bias=None):
    """GPTQ int4 CPU linear layer (matches GPTQLinearMethod.apply)."""
    reshaped_x = x.reshape(-1, x.shape[-1])
    output_dim = ps.shape[0] * ps.shape[-1]
    out = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
        reshaped_x,
        pw,
        pz,
        ps,
        bias.float() if bias is not None else None,
    )
    return out.reshape(x.shape[:-1] + (output_dim,))


# ── RMSNorm ─────────────────────────────────────────────────────────────────


def rms_norm(x, weight, eps=1e-6):
    orig_dtype = x.dtype
    xf = x.float()
    variance = (xf * xf).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return (xf * weight.float()).to(orig_dtype)


# ── RoPE ─────────────────────────────────────────────────────────────────────


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding. q, k: (B, S, H, D)"""
    q_r = q.float().reshape(*q.shape[:-1], -1, 2)
    k_r = k.float().reshape(*k.shape[:-1], -1, 2)
    q0, q1 = q_r[..., 0], q_r[..., 1]
    k0, k1 = k_r[..., 0], k_r[..., 1]
    cos_v = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    sin_v = sin.unsqueeze(0).unsqueeze(2)
    q_out = torch.stack([q0 * cos_v - q1 * sin_v, q0 * sin_v + q1 * cos_v], dim=-1)
    k_out = torch.stack([k0 * cos_v - k1 * sin_v, k0 * sin_v + k1 * cos_v], dim=-1)
    return q_out.flatten(-2).to(q.dtype), k_out.flatten(-2).to(k.dtype)


# ── Attention ────────────────────────────────────────────────────────────────


def attention(x, layer_sd, cos, sin, num_heads=16, num_kv_heads=2, head_dim=128):
    B, S, D = x.shape

    # Q, K, V projections
    q = gptq_linear(x, *layer_sd["q_proj"], layer_sd.get("q_bias"))
    k = gptq_linear(x, *layer_sd["k_proj"], layer_sd.get("k_bias"))
    v = gptq_linear(x, *layer_sd["v_proj"], layer_sd.get("v_bias"))

    q = q.reshape(B, S, num_heads, head_dim)
    k = k.reshape(B, S, num_kv_heads, head_dim)
    v = v.reshape(B, S, num_kv_heads, head_dim)

    # RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # GQA: repeat KV heads
    repeats = num_heads // num_kv_heads
    k = (
        k.unsqueeze(3)
        .expand(-1, -1, -1, repeats, -1)
        .reshape(B, S, num_heads, head_dim)
    )
    v = (
        v.unsqueeze(3)
        .expand(-1, -1, -1, repeats, -1)
        .reshape(B, S, num_heads, head_dim)
    )

    # Transpose to (B, H, S, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Scaled dot-product attention
    scale = head_dim**-0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn.float(), dim=-1).to(x.dtype)
    out = torch.matmul(attn, v)  # (B, H, S, D)

    # Transpose back and reshape
    out = out.transpose(1, 2).reshape(B, S, D)

    # Output projection
    out = gptq_linear(out, *layer_sd["o_proj"])
    return out


# ── FFN (SwiGLU MLP) ───────────────────────────────────────────────────────


def ffn(x, layer_sd):
    gate = gptq_linear(x, *layer_sd["gate_proj"])
    up = gptq_linear(x, *layer_sd["up_proj"])
    intermediate = torch.nn.functional.silu(gate) * up
    return gptq_linear(intermediate, *layer_sd["down_proj"])


# ── Full Layer ──────────────────────────────────────────────────────────────


def forward_layer(x, layer_sd, ln_weight, ln2_weight, cos, sin):
    """One Qwen2 transformer layer: Attention + FFN with residuals."""
    # Pre-attention norm
    h = rms_norm(x, ln_weight)
    # Attention + residual
    h = attention(h, layer_sd, cos, sin)
    x = x + h

    # Pre-FFN norm
    h = rms_norm(x, ln2_weight)
    # FFN + residual
    h = ffn(h, layer_sd)
    x = x + h
    return x


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ckpt = "/home/nmtzwh/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct-GPTQ-Int4/snapshots/a3610ad4cf54e09d73ce14801beade7568287d5f/model.safetensors"
    sd = load_file(ckpt)

    B, S = 2, 16
    D = 2048
    layer_idx = 0
    prefix = f"model.layers.{layer_idx}."

    # Process all GPTQ linear weights for CPU
    linear_names = {
        "q_proj": f"{prefix}self_attn.q_proj",
        "k_proj": f"{prefix}self_attn.k_proj",
        "v_proj": f"{prefix}self_attn.v_proj",
        "o_proj": f"{prefix}self_attn.o_proj",
        "gate_proj": f"{prefix}mlp.gate_proj",
        "up_proj": f"{prefix}mlp.up_proj",
        "down_proj": f"{prefix}mlp.down_proj",
    }

    layer_sd = {}
    for key, full_name in linear_names.items():
        qw = sd[f"{full_name}.qweight"]
        qz = sd[f"{full_name}.qzeros"]
        sc = sd[f"{full_name}.scales"]
        pw, pz, ps = process_gptq_for_cpu(qw, qz, sc)
        layer_sd[key] = (pw, pz, ps)
        print(
            f"  {key:12s}  qweight={tuple(qw.shape)}  packed=({ps.shape[0]}, {ps.shape[1]}, {ps.shape[2]})  N={ps.shape[0] * ps.shape[-1]}"
        )

    # Check for biases (only q, k, v have them)
    for proj in ["q_proj", "k_proj", "v_proj"]:
        bias_key = f"{prefix}self_attn.{proj}.bias"
        if bias_key in sd:
            layer_sd[f"{proj[0]}_bias"] = sd[bias_key]
            print(f"  {proj} bias: {tuple(sd[bias_key].shape)}")

    # Layer norms
    ln_weight = sd[f"{prefix}input_layernorm.weight"]
    ln2_weight = sd[f"{prefix}post_attention_layernorm.weight"]
    print(f"  input_layernorm:           {tuple(ln_weight.shape)}")
    print(f"  post_attention_layernorm:  {tuple(ln2_weight.shape)}")

    # Build RoPE cos/sin (Qwen2 style)
    rope_theta = 1000000.0
    head_dim = D // 16  # 128
    pos = torch.arange(S, dtype=torch.float32)
    freqs = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).to(torch.float16)
    sin = torch.sin(angles).to(torch.float16)

    # Input
    torch.manual_seed(42)
    x = torch.randn(B, S, D, dtype=torch.float16)

    print(f"\nInput:  shape={tuple(x.shape)}  has_nan={x.isnan().any()}")

    # Forward
    y = forward_layer(x, layer_sd, ln_weight, ln2_weight, cos, sin)

    print(
        f"Output: shape={tuple(y.shape)}  has_nan={y.isnan().any()}  has_inf={y.isinf().any()}"
    )
    print(f"  range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"  mean:  {y.mean().item():.4f}")
    print(f"  std:   {y.std().item():.4f}")

    if y.isnan().any():
        nan_count = y.isnan().sum().item()
        print(f"  NaN count: {nan_count}/{y.numel()}")
        print("  FAILED")
    else:
        print("  PASSED")


if __name__ == "__main__":
    main()
