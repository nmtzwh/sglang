import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
elif is_cpu():
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu

    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )


def cpu_target_verify_gdn_chain(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = query_start_loc.shape[0] - 1
    seq_len = q.shape[1]
    if seq_len % batch_size != 0:
        raise RuntimeError(f"Invalid CPU GDN MTP verify shape: {seq_len=} {batch_size=}")
    draft_token_num = seq_len // batch_size

    num_q_heads = q.shape[2]
    head_q_dim = q.shape[3]
    num_v_heads = v.shape[2]
    head_v_dim = v.shape[3]

    q_mtp = q.view(batch_size, draft_token_num, num_q_heads, head_q_dim)
    k_mtp = k.view(batch_size, draft_token_num, num_q_heads, head_q_dim)
    v_mtp = v.view(batch_size, draft_token_num, num_v_heads, head_v_dim)
    a_mtp = a.view(batch_size, draft_token_num, num_v_heads)
    b_mtp = b.view(batch_size, draft_token_num, num_v_heads)

    scratch_indices = torch.arange(
        batch_size, dtype=torch.int32, device=cache_indices.device
    )
    scratch_ssm_states = ssm_states[cache_indices[:batch_size].to(torch.int64)].clone()
    scratch_query_start_loc = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=q.device
    )

    outputs = []
    for step in range(draft_token_num):
        step_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q_mtp[:, step : step + 1].transpose(0, 1).contiguous(),
            k=k_mtp[:, step : step + 1].transpose(0, 1).contiguous(),
            v=v_mtp[:, step : step + 1].transpose(0, 1).contiguous(),
            a=a_mtp[:, step].contiguous(),
            b=b_mtp[:, step].contiguous(),
            initial_state_source=scratch_ssm_states,
            initial_state_indices=scratch_indices,
            cu_seqlens=scratch_query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        intermediate_states_buffer[:, step].index_copy_(
            0,
            intermediate_state_indices[:batch_size].to(torch.int64),
            scratch_ssm_states,
        )
        outputs.append(step_out.squeeze(0))

    return torch.stack(outputs, dim=1).reshape(1, seq_len, num_v_heads, head_v_dim)


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        if is_npu() or is_cpu():
            recurrent_state = ssm_states[cache_indices]
            recurrent_state_indices_args = {}
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **recurrent_state_indices_args,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if is_cpu():
            if retrieve_parent_token is not None:
                raise RuntimeError("CPU GDN MTP verify only supports topk=1.")

            return cpu_target_verify_gdn_chain(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
            )

        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            # target_verify specific parameters
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
