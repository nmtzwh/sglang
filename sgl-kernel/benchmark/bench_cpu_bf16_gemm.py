import argparse
import time
import torch
import torch.nn.functional as F
import sgl_kernel

# Precision dictionary for correctness check
precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}

def benchmark_matmul(M, N, K, has_bias, num_iters=100):
    device = "cpu"
    dtype = torch.bfloat16

    # Input and weights
    input_tensor = torch.randn(M, K, dtype=dtype, device=device)
    weight_tensor = torch.randn(N, K, dtype=dtype, device=device)
    bias_tensor = torch.randn(N, dtype=torch.float32, device=device) if has_bias else None

    # Warmup Native Torch
    for _ in range(10):
        _ = F.linear(input_tensor, weight_tensor, bias_tensor.to(dtype) if has_bias else None)
        
    # Benchmark Native Torch
    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = F.linear(input_tensor, weight_tensor, bias_tensor.to(dtype) if has_bias else None)
    torch_time = (time.perf_counter() - start_time) / num_iters

    # Prepare SGL Packed Weight
    # convert_weight_packed is used to pack the weight for sgl_kernel
    packed_weight = torch.ops.sgl_kernel.convert_weight_packed(weight_tensor)

    # Warmup SGL
    for _ in range(10):
        _ = torch.ops.sgl_kernel.weight_packed_linear(
            input_tensor, packed_weight, bias_tensor if has_bias else None, True
        )

    # Benchmark SGL Packed Linear
    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.ops.sgl_kernel.weight_packed_linear(
            input_tensor, packed_weight, bias_tensor if has_bias else None, True
        )
    sgl_time = (time.perf_counter() - start_time) / num_iters

    # Correctness check (once)
    native_out = torch.matmul(input_tensor.float(), weight_tensor.float().t())
    if has_bias:
        native_out.add_(bias_tensor)
    native_out = native_out.to(dtype)
    
    sgl_out = torch.ops.sgl_kernel.weight_packed_linear(
        input_tensor, packed_weight, bias_tensor if has_bias else None, True
    )
    
    # Calculate GFLOPS
    # GEMM ops = 2 * M * N * K
    gflops_val = (2 * M * N * K) / 1e9
    torch_gflops = gflops_val / torch_time
    sgl_gflops = gflops_val / sgl_time

    atol = rtol = precision[dtype]
    try:
        torch.testing.assert_close(native_out, sgl_out, atol=atol, rtol=rtol)
        correct = "PASS"
    except Exception as e:
        correct = f"FAIL"
        # Print a snippet of error for the first failure
        if not hasattr(benchmark_matmul, "error_printed"):
            print(f"\nExample failure (M={M}, N={N}, K={K}): {str(e)[:200]}...")
            benchmark_matmul.error_printed = True

    return torch_time, sgl_time, torch_gflops, sgl_gflops, correct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()

    print(f"Device: CPU")
    print(f"Torch version: {torch.__version__}")
    print(f"SGL Kernel version: {sgl_kernel.__version__ if hasattr(sgl_kernel, '__version__') else 'unknown'}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("-" * 125)

    # (M, N, K)
    # K: Hidden size, N: Intermediate size or Projection
    # Common sizes for Llama-3-8B: Hidden=4096, Intermediate=14336
    # Common sizes for Llama-3-70B: Hidden=8192, Intermediate=28672
    
    shapes = [
        # Decode sizes (small M = batch size)
        (1, 4096, 4096),
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        
        # Prefill sizes (M = batch size * seq length)
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),

        # Large hidden size (e.g. QKV or Output projection in 70B)
        (1, 8192, 8192),
        (128, 8192, 8192),
        
        # Intermediate layer (e.g. Gate/Up projection in 8B)
        (1, 14336, 4096),
        (128, 14336, 4096),
    ]

    print(f"{'M':>6} | {'N':>6} | {'K':>6} | {'Native (ms)':>12} | {'SGL (ms)':>12} | {'Speedup':>10} | {'Native GFLOPS':>15} | {'SGL GFLOPS':>15} | {'Correct'}")
    print("-" * 125)

    for M, N, K in shapes:
        # Benchmark without bias
        t_torch, t_sgl, g_torch, g_sgl, correct = benchmark_matmul(M, N, K, False, args.num_iters)
        speedup = t_torch / t_sgl
        print(f"{M:6d} | {N:6d} | {K:6d} | {t_torch*1000:12.4f} | {t_sgl*1000:12.4f} | {speedup:10.2f}x | {g_torch:15.2f} | {g_sgl:15.2f} | {correct}")
