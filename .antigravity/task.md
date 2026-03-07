# Fix aarch64 CPU Dispatch Guards

- [x] Fix `MultiPlatformOp.dispatch_forward()` in `multi_platform.py`
- [x] Fix `RMSNorm.forward_cpu` and `GemmaRMSNorm.forward_cpu` in `layernorm.py`
- [x] Fix `SiluAndMul.forward_cpu` and `GeluAndMul.forward_cpu` in `activation.py`
- [x] Fix `RotaryEmbedding.forward_cpu` in `rotary_embedding/base.py`
- [x] Fix `_use_cpu` guard in `fla/layernorm_gated.py`
- [x] Fix `topk.py` module-level function selection guard
