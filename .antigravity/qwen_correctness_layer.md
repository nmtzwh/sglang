# Walkthrough: Layer-by-Layer Debug Printing for Qwen3.5

## Summary

Added tensor-statistics debug printing to `qwen3_5.py` and `qwen3_next.py` on a dedicated `debug/qwen3-layer-print` git branch. Every forward pass prints the shape, dtype, mean, std, min, max, and first 8 values of key intermediate tensors to `stderr`.

## Files Changed

| File | Change |
|---|---|
| [debug_utils.py](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/python/sglang/srt/models/debug_utils.py) | **NEW** — `dbg(tag, tensor)` and `dbg_sep(label)` helpers |
| [qwen3_5.py](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/python/sglang/srt/models/qwen3_5.py) | Instrumented `Qwen3_5ForCausalLM.forward`, both decoder-layer `forward()` methods, `self_attention`, and `Qwen3_5GatedDeltaNet.forward` |
| [qwen3_next.py](file://wsl.localhost/Ubuntu/home/tom/workspace/sglang/python/sglang/srt/models/qwen3_next.py) | Instrumented `Qwen3NextModel.forward`, both hybrid decoder-layer `forward()` methods, `self_attention`, and `Qwen3GatedDeltaNet.forward` |

## What Gets Printed

For each forward pass, the output (to `stderr`) looks like:

```
[DBG] ============= Qwen3_5ForCausalLM forward #1  input_ids=(4,) =============
[DBG] embed_tokens | shape=(4, 2048) dtype=torch.bfloat16 | mean=0.000012 std=0.019531 min=-0.031250 max=0.031250 | samples=[...]
[DBG] L00[linear].in_hidden | shape=(4, 2048) ...
[DBG] L00[linear].after_input_norm | shape=(4, 2048) ...
[DBG] L00[linear].linear_attn.input | shape=(4, 2048) ...
[DBG] L00[linear].linear_attn.mixed_qkv | ...
[DBG] L00[linear].linear_attn.z | ...
[DBG] L00[linear].linear_attn.b | ...
[DBG] L00[linear].linear_attn.a | ...
[DBG] L00[linear].linear_attn.core_attn_out | ...
[DBG] L00[linear].linear_attn.out_proj | ...
[DBG] L00[linear].after_linear_attn | ...
[DBG] L00[linear].after_post_attn_norm | ...
[DBG] L00[linear].after_mlp | ...
[DBG] L01[attn].in_hidden | ...
[DBG] L01[attn].qkv_proj | ...
[DBG] L01[attn].q_after_norm | ...
[DBG] L01[attn].k_after_norm | ...
[DBG] L01[attn].q_after_rope | ...
[DBG] L01[attn].k_after_rope | ...
[DBG] L01[attn].attn_out | ...
[DBG] L01[attn].attn_out_gated | ...   (only if attn_output_gate=True)
[DBG] L01[attn].o_proj | ...
...
[DBG] final_norm_out | shape=(4, 2048) ...
[DBG] ============= Qwen3_5ForCausalLM forward #1 END =============
```

## Usage

**Step 1 — Switch to the debug branch:**
```bash
git checkout debug/qwen3-layer-print
```

**Step 2 — Run inference, capture stderr:**
```bash
# x86 reference machine
python -m sglang.launch_server --model Qwen/Qwen3.5-... --device cpu \
  2>&1 1>/dev/null | grep '\[DBG\]' > x86_layers.txt
```

**Step 3 — Repeat on aarch64 SVE, then diff:**
```bash
diff x86_layers.txt aarch64_layers.txt
```

**Tip — Filter to a single tensor for closer inspection:**
```bash
grep 'L05\[attn\].attn_out' x86_layers.txt
grep 'L05\[attn\].attn_out' aarch64_layers.txt
```

**Tolerances:**
- `mean` / `std` differences < 1e-3 relative are normal for bf16 across platforms.
- Differences > 1e-2 relative indicate a probable implementation bug in the SVE kernel for that operator.
