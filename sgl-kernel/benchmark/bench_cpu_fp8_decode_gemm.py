import argparse
import math
import time

import torch
import sgl_kernel


BLOCK_N = 64
BLOCK_K = 128
FP8_MAX = 448.0


def convert_weight(weight: torch.Tensor, scale_block_size: list[int]):
    n, k = weight.size()
    scale_block_size_n, scale_block_size_k = scale_block_size

    pad_n = (scale_block_size_n - (n % scale_block_size_n)) % scale_block_size_n
    pad_k = (scale_block_size_k - (k % scale_block_size_k)) % scale_block_size_k

    if pad_n > 0 or pad_k > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n))

    weight_blocks = weight.view(
        math.ceil(n / scale_block_size_n),
        scale_block_size_n,
        math.ceil(k / scale_block_size_k),
        scale_block_size_k,
    )
    weight_blocks = weight_blocks.permute(0, 2, 1, 3).contiguous()

    abs_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True)
    scales = abs_max / FP8_MAX
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    q_fp8 = (weight_blocks / scales).to(torch.float8_e4m3fn)
    q_fp8 = q_fp8.permute(0, 2, 1, 3).contiguous()

    if pad_n > 0 or pad_k > 0:
        q_fp8 = q_fp8.view(n + pad_n, k + pad_k)[:n, :k].contiguous()
    else:
        q_fp8 = q_fp8.view(n, k)

    scales = scales.view(math.ceil(n / scale_block_size_n), math.ceil(k / scale_block_size_k))
    return q_fp8, scales.to(torch.float32)


def estimate_kernel_bytes(m: int, n: int, k: int, has_bias: bool) -> int:
    scale_bytes = math.ceil(n / BLOCK_N) * math.ceil(k / BLOCK_K) * 4
    bias_bytes = n * 4 if has_bias else 0
    return m * k * 2 + n * k + scale_bytes + bias_bytes + m * n * 2


def measure_dram_bandwidth(size_mb: int, iters: int, threads: int) -> float:
    torch.set_num_threads(threads)
    numel = size_mb * 1024 * 1024 // 4
    a = torch.randn(numel, dtype=torch.float32)
    b = torch.randn(numel, dtype=torch.float32)
    c = torch.empty_like(a)

    for _ in range(3):
        torch.add(a, b, out=c)

    start = time.perf_counter()
    for _ in range(iters):
        torch.add(a, b, out=c)
    elapsed = time.perf_counter() - start
    bytes_moved = iters * numel * 4 * 3
    return bytes_moved / elapsed / 1e9


def get_eviction_buffer_bytes(
    fp8_weight: torch.Tensor, scales: torch.Tensor, min_flush_mb: int
) -> int:
    min_flush_bytes = min_flush_mb * 1024 * 1024
    cold_weight_bytes = fp8_weight.numel() * fp8_weight.element_size() + scales.numel() * scales.element_size()
    return max(min_flush_bytes, cold_weight_bytes * 2)


def build_eviction_buffer(num_bytes: int) -> torch.Tensor:
    numel = math.ceil(num_bytes / 4)
    return torch.randn(numel, dtype=torch.float32)


def evict_cpu_cache(evict_buffer: torch.Tensor) -> None:
    torch.add(evict_buffer, 1.0, out=evict_buffer)


def bench_fp8_decode(
    m: int,
    n: int,
    k: int,
    has_bias: bool,
    warmup: int,
    iters: int,
    threads: int,
    flush_cache: bool,
    flush_size_mb: int,
):
    torch.set_num_threads(threads)
    dtype = torch.bfloat16
    block_size = [BLOCK_N, BLOCK_K]

    x = torch.randn(m, k, dtype=dtype)
    weight = torch.randn(n, k, dtype=dtype)
    fp8_weight, scales = convert_weight(weight, block_size)
    fp8_weight = torch.ops.sgl_kernel.convert_weight_packed(fp8_weight)
    bias = torch.randn(n, dtype=torch.float32) if has_bias else None
    evict_buffer = None
    if flush_cache:
        evict_buffer = build_eviction_buffer(
            get_eviction_buffer_bytes(fp8_weight, scales, flush_size_mb)
        )

    for _ in range(warmup):
        if evict_buffer is not None:
            evict_cpu_cache(evict_buffer)
        torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
            x, fp8_weight, scales, block_size, bias, dtype, True
        )

    elapsed = 0.0
    for _ in range(iters):
        if evict_buffer is not None:
            evict_cpu_cache(evict_buffer)
        start = time.perf_counter()
        out = torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
            x, fp8_weight, scales, block_size, bias, dtype, True
        )
        elapsed += time.perf_counter() - start
    ms = elapsed * 1000 / iters
    gbps = estimate_kernel_bytes(m, n, k, has_bias) / (elapsed / iters) / 1e9
    return ms, gbps, out


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="1,2,4")
    parser.add_argument("--n", default="4096,7168,8192,16384")
    parser.add_argument("--k", default="4096,7168,8192")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument("--measure-dram", action="store_true")
    parser.add_argument("--dram-size-mb", type=int, default=512)
    parser.add_argument("--dram-iters", type=int, default=50)
    parser.add_argument("--flush-size-mb", type=int, default=256)
    parser.add_argument("--hot-cache", action="store_true")
    args = parser.parse_args()

    if args.measure_dram:
        dram_gbps = measure_dram_bandwidth(args.dram_size_mb, args.dram_iters, args.threads)
        print(f"Measured DRAM bandwidth: {dram_gbps:.2f} GB/s")
        print()

    print(f"Threads: {args.threads}")
    print("M,N,K,has_bias,cold_weight,latency_ms,effective_gbps")
    for m in parse_int_list(args.m):
        for n in parse_int_list(args.n):
            for k in parse_int_list(args.k):
                ms, gbps, _ = bench_fp8_decode(
                    m=m,
                    n=n,
                    k=k,
                    has_bias=args.has_bias,
                    warmup=args.warmup,
                    iters=args.iters,
                    threads=args.threads,
                    flush_cache=not args.hot_cache,
                    flush_size_mb=args.flush_size_mb,
                )
                print(f"{m},{n},{k},{args.has_bias},{not args.hot_cache},{ms:.4f},{gbps:.2f}")


if __name__ == "__main__":
    main()
