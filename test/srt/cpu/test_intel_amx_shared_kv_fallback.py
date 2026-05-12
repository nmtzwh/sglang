import enum
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _load_backend():
    try:
        from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend
        from sglang.srt.layers.radix_attention import AttentionType
    except ModuleNotFoundError:
        _install_sglang_stubs()
        module_path = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/layers/attention/intel_amx_backend.py"
        )
        spec = importlib.util.spec_from_file_location(
            "intel_amx_backend_under_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        from sglang.srt.layers.radix_attention import AttentionType

        IntelAMXAttnBackend = module.IntelAMXAttnBackend

    return IntelAMXAttnBackend, AttentionType


def _install_sglang_stubs():
    module_names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.base_attn_backend",
        "sglang.srt.layers.radix_attention",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.forward_batch_info",
    ]
    for name in module_names:
        sys.modules.setdefault(name, types.ModuleType(name))

    class AttentionBackend:
        pass

    class AttentionType(enum.Enum):
        DECODER = "decoder"
        DECODER_BIDIRECTIONAL = "decoder_bidirectional"
        ENCODER_ONLY = "encoder_only"

    sys.modules[
        "sglang.srt.layers.attention.base_attn_backend"
    ].AttentionBackend = AttentionBackend
    sys.modules["sglang.srt.layers.radix_attention"].AttentionType = AttentionType
    sys.modules["sglang.srt.model_executor.forward_batch_info"].ForwardBatch = object


class _KVPool:
    def __init__(self, key_buffer, value_buffer, full_to_swa=None):
        self.key_buffer = key_buffer
        self.value_buffer = value_buffer
        self.full_to_swa = full_to_swa
        self.layers_mapping = {0: (0, True)} if full_to_swa is not None else None

    def get_key_buffer(self, layer_id):
        assert layer_id == 0
        return self.key_buffer

    def get_value_buffer(self, layer_id):
        assert layer_id == 0
        return self.value_buffer

    def translate_loc_from_full_to_swa(self, kv_indices):
        return self.full_to_swa[kv_indices.to(torch.long)].to(torch.int32)


def _batch(
    key_buffer,
    value_buffer,
    req_to_token,
    req_pool_indices,
    full_to_swa=None,
):
    return SimpleNamespace(
        token_to_kv_pool=_KVPool(key_buffer, value_buffer, full_to_swa),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        req_pool_indices=req_pool_indices,
    )


def _layer(
    attention_type,
    num_q_heads=4,
    num_kv_heads=2,
    q_head_dim=3,
    v_head_dim=5,
    sliding_window_size=-1,
):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=num_q_heads,
        tp_k_head_num=num_kv_heads,
        qk_head_dim=q_head_dim,
        v_head_dim=v_head_dim,
        scaling=0.5,
        is_cross_attention=False,
        attn_type=attention_type.DECODER,
        sliding_window_size=sliding_window_size,
    )


def _sdpa_expected(
    q,
    k_cache,
    v_cache,
    token_indices,
    scaling,
    prefix_len=None,
    sliding_window_size=-1,
):
    q = q.transpose(0, 1).unsqueeze(0)
    k = k_cache[token_indices].transpose(0, 1).unsqueeze(0)
    v = v_cache[token_indices].transpose(0, 1).unsqueeze(0)
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    attn_mask = None
    if prefix_len is not None:
        q_pos = torch.arange(prefix_len, prefix_len + q.shape[2])
        kv_pos = torch.arange(k.shape[2])
        attn_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    elif sliding_window_size > -1:
        q_pos = torch.full((q.shape[2],), k.shape[2] - 1)
        kv_pos = torch.arange(k.shape[2])
        attn_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

    if sliding_window_size > -1:
        if attn_mask is None:
            q_pos = torch.full((q.shape[2],), k.shape[2] - 1)
            kv_pos = torch.arange(k.shape[2])
            attn_mask = torch.ones((q.shape[2], k.shape[2]), dtype=torch.bool)
        attn_mask = attn_mask & (
            kv_pos.unsqueeze(0) >= q_pos.unsqueeze(1) - sliding_window_size
        )

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scaling
    ).squeeze(0).transpose(0, 1)


def test_shared_kv_decode_fallback_reads_cache_without_new_kv():
    IntelAMXAttnBackend, AttentionType = _load_backend()
    torch.manual_seed(0)
    layer = _layer(AttentionType)
    key_buffer = torch.randn(8, layer.tp_k_head_num, layer.qk_head_dim)
    value_buffer = torch.randn(8, layer.tp_k_head_num, layer.v_head_dim)
    req_to_token = torch.tensor([[0, 2, 4, 0], [1, 3, 0, 0]], dtype=torch.int64)
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
    seq_lens = torch.tensor([3, 2], dtype=torch.int64)
    q = torch.randn(2, layer.tp_q_head_num, layer.qk_head_dim)
    o = torch.empty(2, layer.tp_q_head_num, layer.v_head_dim)

    backend = IntelAMXAttnBackend.__new__(IntelAMXAttnBackend)
    backend._forward_cached_kv_torch(
        q.reshape(2, -1),
        o.reshape(2, -1),
        layer,
        _batch(key_buffer, value_buffer, req_to_token, req_pool_indices),
        seq_lens,
    )

    expected0 = _sdpa_expected(
        q[:1], key_buffer, value_buffer, req_to_token[0, :3], layer.scaling
    )
    expected1 = _sdpa_expected(
        q[1:2], key_buffer, value_buffer, req_to_token[1, :2], layer.scaling
    )
    expected = torch.cat([expected0, expected1], dim=0)
    torch.testing.assert_close(o, expected)


def test_shared_kv_extend_fallback_uses_prefix_aligned_causal_mask():
    IntelAMXAttnBackend, AttentionType = _load_backend()
    torch.manual_seed(1)
    layer = _layer(AttentionType)
    key_buffer = torch.randn(6, layer.tp_k_head_num, layer.qk_head_dim)
    value_buffer = torch.randn(6, layer.tp_k_head_num, layer.v_head_dim)
    req_to_token = torch.tensor([[0, 1, 2, 3, 4, 0]], dtype=torch.int64)
    req_pool_indices = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([5], dtype=torch.int64)
    extend_seq_lens = torch.tensor([3], dtype=torch.int64)
    extend_start_loc = torch.tensor([0], dtype=torch.int64)
    q = torch.randn(3, layer.tp_q_head_num, layer.qk_head_dim)
    o = torch.empty(3, layer.tp_q_head_num, layer.v_head_dim)

    backend = IntelAMXAttnBackend.__new__(IntelAMXAttnBackend)
    backend._forward_cached_kv_torch(
        q.reshape(3, -1),
        o.reshape(3, -1),
        layer,
        _batch(key_buffer, value_buffer, req_to_token, req_pool_indices),
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
    )

    expected = _sdpa_expected(
        q, key_buffer, value_buffer, req_to_token[0, :5], layer.scaling, prefix_len=2
    )
    torch.testing.assert_close(o, expected)


def test_shared_kv_fallback_translates_swa_req_to_token_indices():
    IntelAMXAttnBackend, AttentionType = _load_backend()
    torch.manual_seed(2)
    layer = _layer(AttentionType)
    key_buffer = torch.randn(3, layer.tp_k_head_num, layer.qk_head_dim)
    value_buffer = torch.randn(3, layer.tp_k_head_num, layer.v_head_dim)
    req_to_token = torch.tensor([[5, 3, 0]], dtype=torch.int64)
    full_to_swa = torch.tensor([2, 2, 2, 1, 2, 0, -1], dtype=torch.int64)
    req_pool_indices = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([2], dtype=torch.int64)
    q = torch.randn(1, layer.tp_q_head_num, layer.qk_head_dim)
    o = torch.empty(1, layer.tp_q_head_num, layer.v_head_dim)

    backend = IntelAMXAttnBackend.__new__(IntelAMXAttnBackend)
    backend._forward_cached_kv_torch(
        q.reshape(1, -1),
        o.reshape(1, -1),
        layer,
        _batch(
            key_buffer,
            value_buffer,
            req_to_token,
            req_pool_indices,
            full_to_swa=full_to_swa,
        ),
        seq_lens,
    )

    expected = _sdpa_expected(
        q,
        key_buffer,
        value_buffer,
        torch.tensor([0, 1], dtype=torch.int64),
        layer.scaling,
    )
    torch.testing.assert_close(o, expected)


def test_swa_cache_loc_translation_uses_pool_mapping():
    IntelAMXAttnBackend, AttentionType = _load_backend()
    layer = _layer(AttentionType)
    key_buffer = torch.empty(3, layer.tp_k_head_num, layer.qk_head_dim)
    value_buffer = torch.empty(3, layer.tp_k_head_num, layer.v_head_dim)
    full_to_swa = torch.tensor([2, 2, 2, 1, 2, 0, -1], dtype=torch.int64)
    forward_batch = _batch(
        key_buffer,
        value_buffer,
        req_to_token=torch.tensor([[5, 3]], dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        full_to_swa=full_to_swa,
    )
    forward_batch.out_cache_loc = torch.tensor([5, 3], dtype=torch.int64)

    backend = IntelAMXAttnBackend.__new__(IntelAMXAttnBackend)
    torch.testing.assert_close(
        backend._get_cache_loc_for_layer(layer, forward_batch),
        torch.tensor([0, 1], dtype=torch.int64),
    )


def test_decode_attn_logits_resized_for_layer_value_dim():
    IntelAMXAttnBackend, AttentionType = _load_backend()
    layer = _layer(AttentionType, v_head_dim=7)
    forward_batch = SimpleNamespace(batch_size=2)
    backend = IntelAMXAttnBackend.__new__(IntelAMXAttnBackend)
    backend.forward_metadata = (
        torch.zeros((2, layer.tp_q_head_num, 8, 5), dtype=torch.float32),
        None,
    )

    attn_logits = backend._get_decode_attn_logits(layer, forward_batch)
    assert attn_logits.shape == (2, layer.tp_q_head_num, 8, layer.v_head_dim + 1)


def test_sliding_window_decode_fallback_masks_old_tokens():
    IntelAMXAttnBackend, AttentionType = _load_backend()
    torch.manual_seed(3)
    layer = _layer(AttentionType, sliding_window_size=1)
    key_buffer = torch.randn(4, layer.tp_k_head_num, layer.qk_head_dim)
    value_buffer = torch.randn(4, layer.tp_k_head_num, layer.v_head_dim)
    req_to_token = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    req_pool_indices = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([4], dtype=torch.int64)
    q = torch.randn(1, layer.tp_q_head_num, layer.qk_head_dim)
    o = torch.empty(1, layer.tp_q_head_num, layer.v_head_dim)

    backend = IntelAMXAttnBackend.__new__(IntelAMXAttnBackend)
    backend._forward_cached_kv_torch(
        q.reshape(1, -1),
        o.reshape(1, -1),
        layer,
        _batch(key_buffer, value_buffer, req_to_token, req_pool_indices),
        seq_lens,
    )

    expected = _sdpa_expected(
        q,
        key_buffer,
        value_buffer,
        req_to_token[0, :4],
        layer.scaling,
        sliding_window_size=1,
    )
    torch.testing.assert_close(o, expected)


def test_sliding_window_extend_fallback_combines_causal_and_window_masks():
    IntelAMXAttnBackend, AttentionType = _load_backend()
    torch.manual_seed(4)
    layer = _layer(AttentionType, sliding_window_size=1)
    key_buffer = torch.randn(5, layer.tp_k_head_num, layer.qk_head_dim)
    value_buffer = torch.randn(5, layer.tp_k_head_num, layer.v_head_dim)
    req_to_token = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    req_pool_indices = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([5], dtype=torch.int64)
    extend_seq_lens = torch.tensor([3], dtype=torch.int64)
    extend_start_loc = torch.tensor([0], dtype=torch.int64)
    q = torch.randn(3, layer.tp_q_head_num, layer.qk_head_dim)
    o = torch.empty(3, layer.tp_q_head_num, layer.v_head_dim)

    backend = IntelAMXAttnBackend.__new__(IntelAMXAttnBackend)
    backend._forward_cached_kv_torch(
        q.reshape(3, -1),
        o.reshape(3, -1),
        layer,
        _batch(key_buffer, value_buffer, req_to_token, req_pool_indices),
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
    )

    expected = _sdpa_expected(
        q,
        key_buffer,
        value_buffer,
        req_to_token[0, :5],
        layer.scaling,
        prefix_len=2,
        sliding_window_size=1,
    )
    torch.testing.assert_close(o, expected)
