import argparse
import time
import torch
from torch.nn.functional import scaled_dot_product_attention
import sgl_kernel

# Reference implementation based on test/srt/cpu/test_extend.py
def sdpa_extend_ref(
    query,
    output,
    k_cache,
    v_cache,
    req_to_token,
    req_pool_indices,
    seq_lens,
    extend_prefix_lens,
    extend_seq_lens,
    scaling=None,
    enable_gqa=False,
    causal=True,
):
    # query: [num_tokens, num_heads, head_size]
    # k_cache/v_cache: [max_total_tokens, num_heads_kv, head_size]
    
    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query_heads_first = query.movedim(0, 1)

    start_q = 0
    for seq_idx in range(seq_lens.shape[0]):
        extend_seq_len_q = extend_seq_lens[seq_idx].item()
        prefill_seq_len_q = extend_prefix_lens[seq_idx].item()
        seq_len_kv = seq_lens[seq_idx].item()
        
        end_q = start_q + extend_seq_len_q
        
        per_req_query = query_heads_first[:, start_q:end_q, :]
        # SDPA requires full sequence length for causal masking to work correctly in this layout
        per_req_query_redundant = torch.zeros(
            (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
            dtype=per_req_query.dtype,
            device=per_req_query.device,
        )
        per_req_query_redundant[:, prefill_seq_len_q:, :] = per_req_query

        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        # [seq_len_kv, heads_kv, d] -> [heads_kv, seq_len_kv, d]
        per_req_key = k_cache[per_req_tokens].movedim(0, 1)
        per_req_value = v_cache[per_req_tokens].movedim(0, 1)

        # SDPA expects [batch, heads, seq, dim]
        # Using enable_gqa=enable_gqa to handle H_Q != H_KV
        per_req_out_redundant = (
            scaled_dot_product_attention(
                per_req_query_redundant.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=scaling,
                enable_gqa=enable_gqa,
            )
            .squeeze(0)
            .movedim(1, 0) # [heads, seq, dim] -> [seq, heads, dim]
        )
        output[start_q:end_q, :, :] = per_req_out_redundant[prefill_seq_len_q:, :, :]
        start_q += extend_seq_len_q
        
    return output

def benchmark_extend(B, N_CTX, H_Q, H_KV, D, num_iters=100):
    device = "cpu"
    dtype = torch.bfloat16
    
    # Setup lengths: mix of prefix and extend
    b_seq_len_prefix = torch.full((B,), N_CTX // 2, dtype=torch.int32)
    b_seq_len_extend = torch.full((B,), N_CTX // 2, dtype=torch.int32)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend
    max_len_in_batch = torch.max(b_seq_len).item()
    
    # Metadata
    b_req_idx = torch.arange(B, dtype=torch.int64)
    req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    
    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    
    # Fill req_to_tokens with dummy indices
    for i in range(B):
        start = i * max_len_in_batch
        req_to_tokens[i, :b_seq_len[i]] = torch.arange(start, start + b_seq_len[i], dtype=torch.int32)

    # Buffers
    k_cache = torch.randn((B * max_len_in_batch, H_KV, D), dtype=dtype)
    v_cache = torch.randn((B * max_len_in_batch, H_KV, D), dtype=dtype)
    
    q_extend = torch.randn((extend_token_num, H_Q, D), dtype=dtype)
    k_extend = torch.randn((extend_token_num, H_KV, D), dtype=dtype)
    v_extend = torch.randn((extend_token_num, H_KV, D), dtype=dtype)
    
    # CRITICAL: Synchronize k_cache/v_cache with k_extend/v_extend for the extend portion
    for i in range(B):
        prefix_len = b_seq_len_prefix[i].item()
        extend_len = b_seq_len_extend[i].item()
        
        # Extend part in cache
        cache_indices = req_to_tokens[i, prefix_len : prefix_len + extend_len]
        
        # Extend part in *_extend tensors
        extend_start = b_start_loc_extend[i].item()
        extend_end = extend_start + extend_len
        
        k_cache[cache_indices] = k_extend[extend_start:extend_end]
        v_cache[cache_indices] = v_extend[extend_start:extend_end]
    
    o_sgl = torch.empty((extend_token_num, H_Q, D), dtype=dtype)
    o_ref = torch.empty((extend_token_num, H_Q, D), dtype=dtype)
    
    sm_scale = 1.0 / (D**0.5)
    logit_cap = 0.0
    max_len_extend = torch.max(b_seq_len_extend).item()
    enable_gqa = (H_Q != H_KV)

    # Warmup Native
    for _ in range(2):
        sdpa_extend_ref(q_extend, o_ref, k_cache, v_cache, req_to_tokens, b_req_idx, b_seq_len, b_seq_len_prefix, b_seq_len_extend, scaling=sm_scale, enable_gqa=enable_gqa)

    # Benchmark Native
    start_time = time.perf_counter()
    for _ in range(num_iters):
        sdpa_extend_ref(q_extend, o_ref, k_cache, v_cache, req_to_tokens, b_req_idx, b_seq_len, b_seq_len_prefix, b_seq_len_extend, scaling=sm_scale, enable_gqa=enable_gqa)
    torch_time = (time.perf_counter() - start_time) / num_iters

    # Warmup SGL
    for _ in range(2):
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend, k_extend, v_extend, o_sgl, k_cache, v_cache,
            req_to_tokens, b_req_idx, b_seq_len.to(torch.int64), b_seq_len_extend,
            b_start_loc_extend, max_len_extend, sm_scale, logit_cap
        )

    # Benchmark SGL
    start_time = time.perf_counter()
    for _ in range(num_iters):
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend, k_extend, v_extend, o_sgl, k_cache, v_cache,
            req_to_tokens, b_req_idx, b_seq_len.to(torch.int64), b_seq_len_extend,
            b_start_loc_extend, max_len_extend, sm_scale, logit_cap
        )
    sgl_time = (time.perf_counter() - start_time) / num_iters

    # Correctness
    try:
        torch.testing.assert_close(o_ref, o_sgl, atol=2e-2, rtol=2e-2)
        correct = "PASS"
    except Exception as e:
        correct = "FAIL"
        if not hasattr(benchmark_extend, "error_printed"):
            print(f"\nExample failure (B={B}, N_CTX={N_CTX}, H_Q={H_Q}, H_KV={H_KV}): {str(e)[:500]}...")
            benchmark_extend.error_printed = True

    return torch_time, sgl_time, correct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    args = parser.parse_args()

    print(f"Device: CPU")
    print(f"Torch version: {torch.__version__}")
    print(f"SGL Kernel version: {sgl_kernel.__version__ if hasattr(sgl_kernel, '__version__') else 'unknown'}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("-" * 110)

    # Configurations: (B, N_CTX, H_Q, H_KV, D)
    configs = [
        # Small for debug
        (1, 16, 1, 1, 128),
        (1, 128, 32, 32, 128),
        (4, 128, 32, 32, 128),
        # Medium
        (1, 1024, 32, 32, 128),
        (4, 1024, 32, 32, 128),
        # GQA
        (1, 1024, 32, 8, 128),
        (4, 1024, 32, 8, 128),
    ]

    header = f"{'B':>4} | {'SeqLen':>6} | {'H_Q':>4} | {'H_KV':>4} | {'D':>4} | {'Native (ms)':>12} | {'SGL (ms)':>12} | {'Speedup':>10} | {'Correct'}"
    print(header)
    print("-" * 110)

    for B, N_CTX, H_Q, H_KV, D in configs:
        t_torch, t_sgl, correct = benchmark_extend(B, N_CTX, H_Q, H_KV, D, args.num_iters)
        speedup = t_torch / t_sgl if t_sgl > 0 else 0
        print(f"{B:4d} | {N_CTX:6d} | {H_Q:4d} | {H_KV:4d} | {D:4d} | {t_torch*1000:12.4f} | {t_sgl*1000:12.4f} | {speedup:10.2f}x | {correct}")
