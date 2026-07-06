# Qwen3.5 and Gemma4 CPU/AArch64 Porting Summary

This document summarizes the work completed in this branch to make Qwen3.5 and
Gemma4 usable on the SGLang CPU runtime, including speculative decoding and
quantized model support. The work was developed in an AArch64-oriented branch,
but some optimized execution paths remain specific to Intel AMX. The platform
status is called out explicitly below.

## Scope

The port focused on:

- text-only dense Qwen3.5 and Gemma4 execution;
- CPU-safe model loading and configuration handling;
- CPU implementations for GPU-only operators and speculative metadata kernels;
- Qwen3.5 MTP through the EAGLE speculative pipeline;
- Gemma4 MTP through the Frozen-KV speculative pipeline;
- greedy and sampled speculative verification;
- multi-token target verification rather than single-token emulation;
- BF16 and block-wise FP8 model execution;
- correctness tests and manual end-to-end validation.

Multimodal Gemma4, Gemma4 MoE, and production CPU graph execution for MTP were
not part of the completed scope.

## Platform Status

The implementation contains two classes of CPU support:

1. Architecture-neutral PyTorch and CPU-runtime fallbacks. These cover model
   configuration, loading, RMSNorm reshaping, activation and softcap fallbacks,
   speculative tree metadata, verification, and sampling. They are applicable
   to AArch64 when the underlying PyTorch operators are available.
2. Intel AMX optimized paths. The `intel_amx` attention backend, packed FP8
   GEMM, and the currently enabled Qwen3.5 CPU EAGLE launch path require an
   AMX-capable x86 CPU.

Therefore, the text-only model and native PyTorch operator work is portable to
AArch64, while the optimized speculative attention and FP8 paths still require
an AArch64-specific backend before they can replace `intel_amx`. The current
server guardrails intentionally prevent silently selecting an unsupported
backend.

## Qwen3.5

### Base CPU Model Support

The upstream Qwen3.5 CPU implementation was integrated and adapted around the
hybrid attention model:

- CPU configuration padding for attention heads and intermediate dimensions;
- CPU GDN/Mamba state handling;
- Qwen3.5-specific CPU model kernels and fused norm/gating integration;
- CPU-safe weight loading and linear dispatch;
- normal prefill and decode attention through the CPU attention backend;
- FP8 module-skip matching fixes for quantized checkpoints.

An include dependency on `sgl-kernel/csrc/cpu/vec.h` was also restored in
`gemm.h`, fixing builds that use the `sgl_vec` namespace.

### EAGLE MTP Scheduling

The initial CPU EAGLE path added an `intel_amx` draft backend and CPU-safe
dispatch for both draft decode and draft extension. Server validation restricts
this path to supported combinations instead of allowing a later kernel failure:

- `--device cpu`;
- `--attention-backend intel_amx`;
- `--speculative-algorithm EAGLE`;
- `--speculative-eagle-topk 1`;
- eager, non-overlap execution.

Spec-v2 environment settings are tolerated where required by the Qwen3.5 model
configuration, but CPU uses the non-overlap worker.

### CPU Speculative Metadata

GPU/Triton-only metadata operations received CPU implementations, including:

- draft cache-location assignment;
- request-to-token-pool updates;
- extend-after-decode positions and verified-token construction;
- page-aligned eviction masks;
- top-k=1 chain-tree construction;
- greedy tree verification.

The CPU tree implementation deliberately rejects unsupported branching modes
instead of producing an invalid verification layout.

### Correctness Fixes

Several failures were caused by CPU metadata being shaped or typed differently
from CUDA assumptions. The fixes included:

- synthesizing target-verification extend metadata for `intel_amx`;
- deriving verification sequence lengths from prefix and draft lengths;
- normalizing request lengths to the integer type required by CPU kernels;
- allocating Mamba and intermediate state pools on the selected device instead
  of hard-coding CUDA;
- adapting CPU causal-convolution and fused GDN calls to their actual ABI;
- preserving the recurrent state transition for every verified draft token;
- fixing cache locations used by draft and target passes;
- guarding CUDA cache and synchronization calls in the MTP model wiring.

These changes resolved the earlier behavior where generated text diverged after
the first few tokens despite apparently valid model weights.

### Multi-Token Verification

The first correctness fallback only evaluated the root draft token and filled
later verification outputs with zeros. That was functionally conservative but
removed the main performance benefit of MTP.

The final CPU path performs a sequential convolution/state chain over all draft
tokens in each request. Each draft step reads the state produced by the previous
step and stores its intermediate convolution window. This preserves the model's
recurrent semantics while allowing the target model to verify multiple tokens
in one forward batch.

### Sampling

CPU EAGLE was extended beyond greedy decoding with CPU implementations of:

- top-k probability renormalization;
- top-p probability renormalization;
- target-only speculative tree sampling;
- residual-distribution sampling after rejection.

Requests with temperature greater than zero therefore use the speculative
sampling contract instead of crashing the server or silently falling back to
greedy verification.

### CPU Graph Decision

CPU graph/`torch.compile` execution was investigated for EAGLE MTP. Timing
showed that normal decode improved under graph mode, but target verification
regressed from roughly 200 ms to roughly 1000 ms, with the graph call itself
accounting for the regression. CPU graph mode is consequently disabled for
EAGLE MTP until target verification can be compiled without this overhead.

## Gemma4

### Upstream Integration and Dependencies

Gemma4 model and MTP support were brought in from newer SGLang development,
including the corresponding Transformers and auxiliary dependency updates. The
port then concentrated on the text-only dense model so that CPU execution did
not depend on unavailable multimodal processors.

When multimodal execution is not requested,
`Gemma4ForConditionalGeneration` is routed through `Gemma4ForCausalLM`, and the
embedded `Gemma4TextConfig` is used as the runtime model configuration.

### Configuration and Weight Loading

Gemma4 exposed several assumptions that were valid for MoE or GPU checkpoints
but not for dense CPU models. The port fixed:

- dense configurations without `num_experts` or MoE intermediate sizes;
- CPU tensor-parallel padding when optional configuration attributes are absent;
- full-attention and sliding-window KV-head configuration;
- Q/K/V fused-weight loading and K-equals-V checkpoints;
- oversized Q/K normalization weights;
- assistant-model configuration discovery;
- text-only routing without constructing a multimodal processor.

The loader now validates the real tensor shapes and handles the Gemma4-specific
name mapping instead of relying on generic fused-layer assumptions.

### CPU Operator Coverage

CPU fallbacks and kernels were added or corrected for operators encountered in
Gemma4 inference:

- `GeluAndMul` native PyTorch fallback;
- Gemma4 RMSNorm CPU kernel;
- flatten/restore handling when RMSNorm receives tensors with more than two
  dimensions;
- final-logit softcapping through PyTorch when the Triton kernel is unavailable;
- dense MLP and attention paths without CUDA-only dispatch;
- hybrid sliding-window attention and shared-KV handling.

These changes moved several operators onto explicit CPU implementations and
prevented CPU execution from entering Triton-only logits processing. Profiling
also showed that attention-backend selection can still expose TensorIterator
hotspots, so backend-level performance remains an area for further work.

### Frozen-KV MTP

Gemma4 uses a Frozen-KV MTP assistant rather than the Qwen3.5 EAGLE draft-model
contract. The CPU work added:

- detection of `Gemma4AssistantForCausalLM` and `gemma4_assistant` configs;
- promotion of `NEXTN` or `EAGLE` requests to `FROZEN_KV_MTP`;
- dense Gemma4 launch guardrails;
- top-k=1 Frozen-KV draft and verification scheduling;
- target and per-step draft cache-location binding;
- sliding-window/shared-KV translation for CPU attention;
- compatibility with renamed speculative result fields and timing APIs;
- CPU-safe Frozen-KV input construction;
- sampled as well as greedy requests.

The supported production configuration is eager execution. CPU graph mode,
overlap/spec-v2 scheduling, top-k greater than one, Gemma4 MoE, and multimodal
Gemma4 remain unsupported for this path.

### Block-Wise FP8

Compressed-tensors FP8_BLOCK checkpoints required a separate fix. The real
RedHatAI Gemma4 configuration stores `quantization_config` on the outer Gemma4
config, while text-only routing replaces that object with `text_config`. Losing
the outer field caused FP8 tensors to be instantiated with
`UnquantizedLinearMethod`, so raw FP8 values were multiplied without block
scales and activations exploded in the first QKV projection.

The final implementation:

- preserves top-level `quantization_config` and `compression_config` when
  switching to text-only Gemma4;
- recognizes both `compressed-tensors` and `compressed_tensors` spellings;
- loads nested text compression configurations where present;
- selects the CPU FP8_BLOCK scheme even for weight-only metadata;
- creates and loads block-scale parameters for fused linear modules;
- aliases the loaded scale to the native CPU kernel's `weight_scale_inv` name;
- packs weights for the CPU FP8 kernel and dispatches
  `fp8_scaled_mm_cpu` on the AMX path.

This fixed both non-speculative output correctness and the zero-acceptance MTP
failure seen with compressed-tensors Gemma4 checkpoints.

## Validation

### Automated Tests

The focused CPU tests cover:

- speculative cache and request-pool metadata;
- extend-after-decode construction and eviction alignment;
- top-k=1 tree construction and greedy verification;
- multi-token GDN convolution/state progression;
- CPU top-k/top-p renormalization and speculative sampling;
- Gemma4 configuration routing and assistant detection;
- dense Gemma4 weight loading;
- Frozen-KV scheduler contracts and cache binding;
- Intel AMX shared/sliding KV translation;
- Gemma4 RMSNorm shape handling and logit softcapping;
- compressed-tensors FP8_BLOCK selection, scale loading, and kernel dispatch.

Representative commands:

```bash
source $HOME/virtualenv/venv_sglang/bin/activate
export SGLANG_USE_CPU_ENGINE=1

PYTHONPATH=python:. python3 -m pytest -q \
  test/srt/cpu/test_speculative_cpu_fallbacks.py \
  test/srt/cpu/test_frozen_kv_mtp_worker.py \
  test/srt/cpu/test_intel_amx_shared_kv_fallback.py \
  test/srt/cpu/test_gemma4_causal_load_weights.py \
  test/srt/cpu/test_gemma4_update_config.py \
  test/srt/cpu/test_layernorm_cpu_reshape.py \
  test/srt/cpu/test_logits_processor_cpu.py \
  test/srt/cpu/test_compressed_tensors_fp8_block.py
```

### Qwen3.5 AMX Launch

```bash
SGLANG_ENABLE_SPEC_V2=1 SGLANG_USE_CPU_ENGINE=1 \
python -m sglang.launch_server \
  --model <QWEN3_5_TARGET_MODEL_OR_PATH> \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --device cpu \
  --attention-backend intel_amx \
  --disable-cuda-graph \
  --disable-overlap-schedule
```

Qwen3.5 MTP requires `SGLANG_ENABLE_SPEC_V2=1` for model-side setup. CPU still
selects the non-overlap EAGLE worker when this environment variable is present.

### Gemma4 AMX Launch

```bash
SGLANG_USE_CPU_ENGINE=1 python -m sglang.launch_server \
  --model <GEMMA4_TEXT_TARGET_MODEL_OR_PATH> \
  --speculative-draft-model-path <GEMMA4_ASSISTANT_MODEL_OR_PATH> \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --device cpu \
  --attention-backend intel_amx \
  --disable-cuda-graph \
  --disable-overlap-schedule
```

The manual Gemma4 harness is available at
`test/manual/models/test_gemma4_mtp_cpu.py`. It compares greedy output with a
non-speculative baseline, exercises sampled requests, and checks for nonzero
speculative acceptance.

## Main Commits

### Qwen3.5

- `10fd0facc` - upstream Qwen3.5 CPU model and kernel optimization;
- `a46973f53` - FP8 module-skip matching fix;
- `506f6801d` - initial CPU EAGLE MTP support;
- `1e4ecf4dd` - CPU EAGLE correctness and state/cache fixes;
- `9b4ea6812` - performance-ready multi-token target verification;
- `a38ea4158` - CPU speculative sampling;
- `440caaf4e` - disable the regressing CPU graph path for EAGLE MTP.

### Gemma4

- `4db8ab6c9`, `7561d3d64` - upstream Gemma4 MTP model integration;
- `3caf48932` - dependency updates for Gemma4;
- `b4d724c21` - dense Gemma4 MTP CPU support;
- `e0cb2d7cb` - sliding-attention support for Gemma4 CPU MTP;
- `1300426ca` - text-only Gemma4 CPU MTP loading;
- `72d5d4409`, `5ee511107`, `23c6f69eb` - dense loading and RMSNorm fixes;
- `a0c9ffa41` - consolidated eager Gemma4 CPU MTP execution fixes;
- `ee6237ec2` - compressed-tensors FP8_BLOCK CPU support.

## Remaining Work

- Add an optimized AArch64 attention backend for speculative draft and target
  verification so MTP no longer depends on `intel_amx`.
- Add AArch64-native BF16/FP8 GEMM kernels or integrate a proven backend.
- Revisit CPU graph mode after isolating the target-verification compilation
  regression.
- Support top-k greater than one without Python tree traversal.
- Extend Gemma4 support to MoE and multimodal execution.
- Add repeatable AArch64 end-to-end performance baselines alongside AMX results.
