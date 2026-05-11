from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class IntelAMXAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        import sgl_kernel  # noqa: F401

        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

        self.num_head = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )

        # [NB]: `layer_id` set to 0 for qwen3-next models, as not all attn layers require kv pool
        # using "full_attention_layer_id_mapping" to map which layer needs kv pool
        layer_id = 0
        if hasattr(model_runner.token_to_kv_pool, "full_attention_layer_id_mapping"):
            layer_id = [*model_runner.token_to_kv_pool.full_attention_layer_id_mapping][
                0
            ]
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            layer_id
        ).shape[-1]
        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        self._maybe_init_spec_extend_metadata(forward_batch)

        bs = forward_batch.batch_size
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
        else:
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
        self.forward_metadata = (attn_logits, max_extend_len)

    def _maybe_init_spec_extend_metadata(self, forward_batch: ForwardBatch):
        if (
            forward_batch.extend_seq_lens is not None
            or not forward_batch.forward_mode.is_target_verify()
            or forward_batch.spec_info is None
        ):
            return

        draft_token_num = getattr(forward_batch.spec_info, "draft_token_num", None)
        if draft_token_num is None:
            return

        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        index_dtype = forward_batch.req_to_token_pool.req_to_token.dtype
        extend_seq_lens = torch.full(
            (bs,), draft_token_num, dtype=index_dtype, device=device
        )
        extend_start_loc = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=index_dtype,
            device=device,
        )

        forward_batch.extend_seq_lens = extend_seq_lens
        forward_batch.extend_start_loc = extend_start_loc
        forward_batch.extend_num_tokens = bs * draft_token_num
        forward_batch.extend_prefix_lens = forward_batch.seq_lens.to(dtype=index_dtype)
        forward_batch.extend_seq_lens_cpu = [draft_token_num] * bs
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.extend_prefix_lens_cpu = forward_batch.seq_lens_cpu.tolist()
        else:
            forward_batch.extend_prefix_lens_cpu = forward_batch.seq_lens.cpu().tolist()
        forward_batch.extend_logprob_start_lens_cpu = (
            forward_batch.extend_prefix_lens_cpu
        )

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def _forward_cached_kv_torch(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor | None = None,
        extend_start_loc: torch.Tensor | None = None,
    ):
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
        kv_repeat = layer.tp_q_head_num // layer.tp_k_head_num if use_gqa else 1

        for seq_idx in range(seq_lens.shape[0]):
            seq_len_kv = int(seq_lens[seq_idx].item())
            if seq_len_kv == 0:
                continue

            if extend_seq_lens is None:
                start_q = seq_idx
                seq_len_q = 1
                prefix_len = seq_len_kv
            else:
                start_q = int(extend_start_loc[seq_idx].item())
                seq_len_q = int(extend_seq_lens[seq_idx].item())
                prefix_len = seq_len_kv - seq_len_q
                if seq_len_q == 0:
                    continue

            end_q = start_q + seq_len_q
            req_pool_idx = req_pool_indices[seq_idx]
            token_indices = req_to_token[req_pool_idx, :seq_len_kv].to(torch.long)

            per_req_q = q[start_q:end_q].transpose(0, 1).unsqueeze(0)
            per_req_k = k_buffer[token_indices].transpose(0, 1).unsqueeze(0)
            per_req_v = v_buffer[token_indices].transpose(0, 1).unsqueeze(0)

            if use_gqa:
                per_req_k = per_req_k.repeat_interleave(kv_repeat, dim=1)
                per_req_v = per_req_v.repeat_interleave(kv_repeat, dim=1)

            attn_mask = None
            if (
                extend_seq_lens is not None
                and not layer.is_cross_attention
                and layer.attn_type != AttentionType.ENCODER_ONLY
            ):
                q_pos = torch.arange(
                    prefix_len,
                    prefix_len + seq_len_q,
                    device=q.device,
                )
                kv_pos = torch.arange(seq_len_kv, device=q.device)
                attn_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

            per_req_o = F.scaled_dot_product_attention(
                per_req_q,
                per_req_k,
                per_req_v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=layer.scaling,
            )
            o[start_q:end_q].copy_(per_req_o.squeeze(0).transpose(0, 1))

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        max_extend_len = None
        self.forward_metadata = (attn_logits, max_extend_len)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata

        seq_lens = forward_batch.seq_lens
        if forward_batch.forward_mode.is_target_verify():
            seq_lens = (
                forward_batch.extend_prefix_lens + forward_batch.extend_seq_lens
            ).to(dtype=forward_batch.seq_lens.dtype)

        if k is None or v is None:
            self._forward_cached_kv_torch(
                q,
                o,
                layer,
                forward_batch,
                seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
            )
        else:
            self.extend_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k,
                v,
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
                max_extend_len,
                layer.scaling,
                layer.logit_cap,
            )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        attn_logits, _ = self.forward_metadata

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if k is None or v is None:
            self._forward_cached_kv_torch(
                q, o, layer, forward_batch, forward_batch.seq_lens
            )
        else:
            self.decode_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                k,
                v,
                forward_batch.out_cache_loc,
                attn_logits,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
            )

        return o

    def support_triton(self):
        return False


class IntelAMXMultiStepDraftBackend:
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        if topk != 1:
            raise ValueError(
                "Intel AMX EAGLE draft decode only supports speculative_eagle_topk=1."
            )
        self.attn_backends = [
            IntelAMXAttnBackend(model_runner) for _ in range(speculative_num_steps - 1)
        ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata(forward_batch)
