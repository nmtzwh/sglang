import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sglang_pkg = types.ModuleType("sglang")
sglang_pkg.__path__ = [str(REPO_ROOT / "python" / "sglang")]
sys.modules.setdefault("sglang", sglang_pkg)

base_grammar_backend = types.ModuleType(
    "sglang.srt.constrained.base_grammar_backend"
)
base_grammar_backend.BaseGrammarObject = object
sys.modules.setdefault(
    "sglang.srt.constrained.base_grammar_backend", base_grammar_backend
)

parallel_state = types.ModuleType("sglang.srt.distributed.parallel_state")
parallel_state.GroupCoordinator = object
parallel_state.patch_tensor_parallel_group = lambda *args, **kwargs: None
sys.modules.setdefault("sglang.srt.distributed.parallel_state", parallel_state)


class _EnvValue:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


environ = types.ModuleType("sglang.srt.environ")
environ.envs = types.SimpleNamespace(
    SGLANG_SIMULATE_ACC_LEN=_EnvValue(-1.0),
    SGLANG_SIMULATE_ACC_METHOD=_EnvValue(""),
)
sys.modules.setdefault("sglang.srt.environ", environ)

schedule_batch = types.ModuleType("sglang.srt.managers.schedule_batch")
schedule_batch.Req = object
sys.modules.setdefault("sglang.srt.managers.schedule_batch", schedule_batch)

mem_cache_common = types.ModuleType("sglang.srt.mem_cache.common")
mem_cache_common.get_last_loc = lambda *args, **kwargs: None
sys.modules.setdefault("sglang.srt.mem_cache.common", mem_cache_common)

server_args = types.ModuleType("sglang.srt.server_args")
server_args.ServerArgs = object
server_args.get_global_server_args = lambda: types.SimpleNamespace(
    enable_multi_layer_eagle=False,
    speculative_accept_threshold_single=1.0,
    speculative_accept_threshold_acc=1.0,
)
sys.modules.setdefault("sglang.srt.server_args", server_args)

from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    verify_tree_greedy_func,
)
from sglang.srt.speculative.spec_utils import (
    align_evict_mask_to_page_size,
    assign_draft_cache_locs,
    create_extend_after_decode_spec_info,
)
from sglang.srt.utils import next_power_of_2


def _install_linear_mtp_import_stubs(monkeypatch):
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    from sglang.srt.utils.common import is_cpu

    is_cpu.cache_clear()

    kernel_backend = types.ModuleType(
        "sglang.srt.layers.attention.linear.kernels.kernel_backend"
    )
    kernel_backend.LinearAttnKernelBase = object
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.linear.kernels.kernel_backend",
        kernel_backend,
    )

    sgl_kernel_mod = types.ModuleType("sgl_kernel")
    sgl_kernel_mamba = types.ModuleType("sgl_kernel.mamba")
    sgl_kernel_mamba.chunk_gated_delta_rule_cpu = lambda *args, **kwargs: None
    sgl_kernel_mamba.causal_conv1d_fn_cpu = lambda *args, **kwargs: None
    sgl_kernel_mamba.causal_conv1d_update_cpu = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "sgl_kernel", sgl_kernel_mod)
    monkeypatch.setitem(sys.modules, "sgl_kernel.mamba", sgl_kernel_mamba)
    torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu = (
        lambda **kwargs: None
    )
    torch.ops.sgl_kernel.fused_gdn_gating_cpu = lambda *args, **kwargs: None

    fused_gdn_gating = types.ModuleType(
        "sglang.srt.layers.attention.fla.fused_gdn_gating"
    )
    fused_gdn_gating.fused_gdn_gating = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.fla.fused_gdn_gating",
        fused_gdn_gating,
    )

    hybrid_backend = types.ModuleType(
        "sglang.srt.layers.attention.hybrid_linear_attn_backend"
    )
    hybrid_backend.MambaAttnBackendBase = object
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.hybrid_linear_attn_backend",
        hybrid_backend,
    )

    linear_utils = types.ModuleType("sglang.srt.layers.attention.linear.utils")
    linear_utils.LinearAttnKernelBackend = object
    linear_utils.get_linear_attn_decode_backend = lambda: None
    linear_utils.get_linear_attn_prefill_backend = lambda: None
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.attention.linear.utils", linear_utils
    )

    causal_conv = types.ModuleType(
        "sglang.srt.layers.attention.mamba.causal_conv1d_triton"
    )
    causal_conv.causal_conv1d_fn = lambda *args, **kwargs: None
    causal_conv.causal_conv1d_update = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.attention.mamba.causal_conv1d_triton",
        causal_conv,
    )

    radix = types.ModuleType("sglang.srt.layers.radix_linear_attention")
    radix.RadixLinearAttention = object
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.radix_linear_attention", radix)

    memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.MambaPool = types.SimpleNamespace(SpeculativeState=object)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.memory_pool", memory_pool)

    forward_batch_info = types.ModuleType(
        "sglang.srt.model_executor.forward_batch_info"
    )
    forward_batch_info.ForwardBatch = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.model_executor.forward_batch_info", forward_batch_info
    )

    model_runner = types.ModuleType("sglang.srt.model_executor.model_runner")
    model_runner.ModelRunner = object
    monkeypatch.setitem(
        sys.modules, "sglang.srt.model_executor.model_runner", model_runner
    )


def test_assign_draft_cache_locs_cpu_topk1_page_size_1():
    num_seqs = 2
    topk = 1
    speculative_num_steps = 3
    page_size = 1
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int32)
    req_to_token = torch.zeros((num_seqs, 16), dtype=torch.int32)
    seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    extend_lens = torch.tensor([3, 3], dtype=torch.int32)
    out_cache_loc = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32)

    assign_draft_cache_locs[(num_seqs,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        extend_lens,
        torch.empty((0,), dtype=torch.int32),
        out_cache_loc,
        None,
        None,
        None,
        0,
        req_to_token.shape[1],
        topk,
        speculative_num_steps,
        page_size,
        next_power_of_2(num_seqs),
        next_power_of_2(speculative_num_steps + page_size),
    )

    assert req_to_token[0, 2:5].tolist() == [10, 11, 12]
    assert req_to_token[1, 4:7].tolist() == [20, 21, 22]
    assert out_cache_loc.tolist() == [10, 11, 12, 20, 21, 22]


def test_assign_draft_cache_locs_cpu_topk1_page_size_gt_1():
    num_seqs = 2
    topk = 1
    speculative_num_steps = 3
    page_size = 4
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int32)
    req_to_token = torch.zeros((num_seqs, 16), dtype=torch.int32)
    seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    extend_lens = torch.tensor([3, 3], dtype=torch.int32)
    out_cache_loc = torch.tensor([30, 31, 32, 40, 41, 42], dtype=torch.int32)

    assign_draft_cache_locs[(num_seqs,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        extend_lens,
        torch.empty((0,), dtype=torch.int32),
        out_cache_loc,
        None,
        None,
        None,
        0,
        req_to_token.shape[1],
        topk,
        speculative_num_steps,
        page_size,
        next_power_of_2(num_seqs),
        next_power_of_2(speculative_num_steps + page_size),
    )

    assert req_to_token[0, 3:6].tolist() == [30, 31, 32]
    assert req_to_token[1, 5:8].tolist() == [40, 41, 42]


def test_create_extend_after_decode_spec_info_cpu():
    verified_id = torch.tensor([101, 102, 201, 202, 203], dtype=torch.int64)
    seq_lens = torch.tensor([8, 11], dtype=torch.int32)
    accept_lens = torch.tensor([2, 3], dtype=torch.int32)
    positions = torch.empty((5,), dtype=torch.int64)
    new_verified_id = torch.empty((2,), dtype=torch.int32)

    create_extend_after_decode_spec_info[(2,)](
        verified_id,
        seq_lens,
        accept_lens,
        positions,
        new_verified_id,
        next_power_of_2(5),
    )

    assert positions.tolist() == [6, 7, 8, 9, 10]
    assert new_verified_id.tolist() == [102, 203]


def test_align_evict_mask_to_page_size_cpu():
    seq_lens = torch.tensor([5, 8], dtype=torch.int32)
    evict_mask = torch.tensor(
        [
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
        ],
        dtype=torch.bool,
    )

    align_evict_mask_to_page_size[(2,)](
        seq_lens,
        evict_mask,
        4,
        4,
        next_power_of_2(4),
    )

    assert evict_mask.view(2, 4).tolist() == [
        [False, False, False, True],
        [False, False, False, False],
    ]


def test_build_tree_kernel_efficient_cpu_topk1_chain():
    verified_id = torch.tensor([7, 9], dtype=torch.int32)
    parent_list = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    top_scores_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    draft_tokens = torch.tensor([[11, 12], [21, 22]], dtype=torch.int64)
    seq_lens = torch.tensor([5, 8], dtype=torch.int64)

    (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        flat_tokens,
    ) = build_tree_kernel_efficient(
        verified_id,
        parent_list,
        top_scores_index,
        draft_tokens,
        seq_lens,
        int(seq_lens.sum().item()),
        topk=1,
        spec_steps=2,
        num_verify_tokens=3,
        tree_mask_mode=TreeMaskMode.QLEN_ONLY,
    )

    assert tree_mask.dtype == torch.bool
    assert positions.tolist() == [5, 6, 7, 8, 9, 10]
    assert retrive_index.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert retrive_next_token.tolist() == [[1, 2, -1], [1, 2, -1]]
    assert retrive_next_sibling.tolist() == [[-1, -1, -1], [-1, -1, -1]]
    assert flat_tokens.tolist() == [7, 11, 12, 9, 21, 22]


def test_verify_tree_greedy_func_cpu_topk1():
    candidates = torch.tensor([[7, 11, 12], [9, 21, 22]], dtype=torch.int64)
    retrive_index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    retrive_next_token = torch.tensor([[1, 2, -1], [1, 2, -1]], dtype=torch.int64)
    retrive_next_sibling = torch.full((2, 3), -1, dtype=torch.int64)
    target_predict = torch.tensor([[11, 99, 0], [21, 22, 23]], dtype=torch.int64)
    predicts = torch.empty((7,), dtype=torch.int32)
    accept_index = torch.full((2, 3), -1, dtype=torch.int32)
    accept_token_num = torch.empty((2,), dtype=torch.int32)

    predicts, accept_index, accept_token_num = verify_tree_greedy_func(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        topk=1,
    )

    assert accept_index.tolist() == [[0, 1, -1], [3, 4, 5]]
    assert accept_token_num.tolist() == [1, 2]
    assert predicts[:6].tolist() == [11, 99, -1, 21, 22, 23]


def test_cpu_target_verify_conv1d_chain_saves_all_steps(monkeypatch):
    _install_linear_mtp_import_stubs(monkeypatch)
    from sglang.srt.layers.attention.linear import gdn_backend

    def fake_causal_conv1d_update(
        x, conv_states, conv_weights, bias, activation, conv_state_indices
    ):
        state = conv_states.index_select(0, conv_state_indices.to(torch.int64))
        state.add_(x.unsqueeze(-1))
        conv_states.index_copy_(0, conv_state_indices.to(torch.int64), state)
        return state.sum(dim=-1)

    monkeypatch.setattr(gdn_backend, "causal_conv1d_update", fake_causal_conv1d_update)

    batch_size = 2
    draft_token_num = 3
    mixed_qkv = torch.arange(
        batch_size * draft_token_num * 4, dtype=torch.float32
    ).view(batch_size * draft_token_num, 4)
    conv_states = torch.zeros((8, 4, 2), dtype=torch.float32)
    conv_states[3].fill_(10)
    conv_states[5].fill_(20)
    live_conv_states = conv_states.clone()
    intermediate = torch.empty((batch_size, draft_token_num, 4, 2))

    out = gdn_backend.cpu_target_verify_conv1d_chain(
        mixed_qkv,
        conv_states,
        torch.empty((4, 3)),
        None,
        "silu",
        torch.tensor([3, 5], dtype=torch.int32),
        intermediate,
        torch.arange(batch_size, dtype=torch.int32),
        batch_size,
        draft_token_num,
    )

    scratch = live_conv_states[torch.tensor([3, 5])]
    expected_out = []
    expected_intermediate = torch.empty_like(intermediate)
    for step in range(draft_token_num):
        x = mixed_qkv.view(batch_size, draft_token_num, 4)[:, step]
        scratch.add_(x.unsqueeze(-1))
        expected_intermediate[:, step].copy_(scratch)
        expected_out.append(scratch.sum(dim=-1).clone())

    torch.testing.assert_close(
        out, torch.stack(expected_out, dim=1).reshape(batch_size * draft_token_num, 4)
    )
    torch.testing.assert_close(intermediate, expected_intermediate)
    torch.testing.assert_close(conv_states, live_conv_states)


def test_cpu_target_verify_gdn_chain_saves_all_steps(monkeypatch):
    _install_linear_mtp_import_stubs(monkeypatch)

    def fake_fused_sigmoid_gating_delta_rule_update(
        *,
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        softplus_beta,
        softplus_threshold,
    ):
        indices = initial_state_indices.to(torch.int64)
        state = initial_state_source.index_select(0, indices)
        state.add_(v.squeeze(0).unsqueeze(2))
        initial_state_source.index_copy_(0, indices, state)
        return state[:, :, 0, :].unsqueeze(0)

    torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu = (
        fake_fused_sigmoid_gating_delta_rule_update
    )

    from sglang.srt.layers.attention.linear.kernels import gdn_triton

    monkeypatch.setattr(
        gdn_triton,
        "fused_sigmoid_gating_delta_rule_update",
        fake_fused_sigmoid_gating_delta_rule_update,
    )

    batch_size = 2
    draft_token_num = 3
    num_heads = 2
    head_dim = 4
    q = torch.zeros((1, batch_size * draft_token_num, num_heads, head_dim))
    k = torch.zeros_like(q)
    v = torch.arange(
        batch_size * draft_token_num * num_heads * head_dim,
        dtype=torch.float32,
    ).view(1, batch_size * draft_token_num, num_heads, head_dim)
    a = torch.zeros((batch_size * draft_token_num, num_heads))
    b = torch.zeros_like(a)
    ssm_states = torch.zeros((8, num_heads, 2, head_dim))
    ssm_states[3].fill_(10)
    ssm_states[5].fill_(20)
    live_ssm_states = ssm_states.clone()
    intermediate = torch.empty((batch_size, draft_token_num, num_heads, 2, head_dim))

    out = gdn_triton.cpu_target_verify_gdn_chain(
        A_log=torch.empty((num_heads,)),
        dt_bias=torch.empty((num_heads,)),
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        ssm_states=ssm_states,
        cache_indices=torch.tensor([3, 5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=torch.arange(batch_size, dtype=torch.int32),
    )

    scratch = live_ssm_states[torch.tensor([3, 5])]
    expected_out = []
    expected_intermediate = torch.empty_like(intermediate)
    v_by_step = v.view(batch_size, draft_token_num, num_heads, head_dim)
    for step in range(draft_token_num):
        scratch.add_(v_by_step[:, step].unsqueeze(2))
        expected_intermediate[:, step].copy_(scratch)
        expected_out.append(scratch[:, :, 0, :].clone())

    torch.testing.assert_close(
        out, torch.stack(expected_out, dim=1).reshape_as(out)
    )
    torch.testing.assert_close(intermediate, expected_intermediate)
    torch.testing.assert_close(ssm_states, live_ssm_states)
