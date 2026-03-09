# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
# ============================================================
"""
debug_utils.py — Layer-by-layer tensor diagnostic helpers.

This module is intentionally only imported inside the  debug/qwen3-layer-print
branch and should never be merged to main.

Usage example (inside a forward method):
    from sglang.srt.models.debug_utils import dbg
    dbg("layer0.attn_out", hidden_states)
"""

import sys


# --------------------------------------------------------------------------- #
#  Core printer                                                                #
# --------------------------------------------------------------------------- #

def dbg(tag: str, t, *, n_samples: int = 8) -> None:
    """Print a single-line diagnostic for tensor *t* with the given *tag*.

    Output format (one line, easy to grep and diff):
        [DBG] <tag> | shape=(...) dtype=<> | mean=<> std=<> min=<> max=<> | samples=<first n values>

    Args:
        tag:       Human-readable identifier, e.g. "L03.attn_out".
        t:         A torch.Tensor.
        n_samples: How many values to sample from the flattened tensor.
    """
    import torch  # local import – keeps module importable without torch

    if not isinstance(t, torch.Tensor):
        print(f"[DBG] {tag} | (not a tensor: {type(t).__name__})", flush=True, file=sys.stderr)
        return

    # cast to float32 for numerically stable stats (handles bf16/fp16)
    tf = t.detach().float()
    flat = tf.reshape(-1)
    actual_n = min(n_samples, flat.numel())
    samples = flat[:actual_n].tolist()
    samples_str = " ".join(f"{v:.6f}" for v in samples)

    mean = flat.mean().item()
    std  = flat.std().item()
    mn   = flat.min().item()
    mx   = flat.max().item()

    print(
        f"[DBG] {tag} | shape={tuple(t.shape)} dtype={t.dtype} | "
        f"mean={mean:.6f} std={std:.6f} min={mn:.6f} max={mx:.6f} | "
        f"samples=[{samples_str}]",
        flush=True,
        file=sys.stderr,
    )


def dbg_sep(label: str) -> None:
    """Print a visible separator line for readability."""
    print(f"[DBG] {'='*60} {label} {'='*60}", flush=True, file=sys.stderr)
