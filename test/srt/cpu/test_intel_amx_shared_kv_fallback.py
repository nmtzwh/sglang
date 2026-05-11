from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("IPython.display")
pytest.importorskip("pydantic")

from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend
from sglang.srt.layers.radix_attention import AttentionType


class _KVPool:
    def __init__(self, key_buffer, value_buffer):
        self.key_buffer = key_buffer
        self.value_buffer = value_buffer

    def get_key_buffer(self, layer_id):
        assert layer_id == 0
        return self.key_buffer

    def get_value_buffer(self, layer_id):
        assert layer_id == 0
        return self.value_buffer


def _batch(key_buffer, value_buffer, req_to_token, req_pool_indices):
    return SimpleNamespace(
        token_to_kv_pool=_KVPool(key_buffer, value_buffer),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        req_pool_indices=req_pool_indices,
    )


def _layer(num_q_heads=4, num_kv_heads=2, q_head_dim=3, v_head_dim=5):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=num_q_heads,
        tp_k_head_num=num_kv_heads,
        qk_head_dim=q_head_dim,
        v_head_dim=v_head_dim,
        scaling=0.5,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
    )


def _sdpa_expected(q, k_cache, v_cache, token_indices, scaling, prefix_len=None):
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

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scaling
    ).squeeze(0).transpose(0, 1)


def test_shared_kv_decode_fallback_reads_cache_without_new_kv():
    torch.manual_seed(0)
    layer = _layer()
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
    torch.manual_seed(1)
    layer = _layer()
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
