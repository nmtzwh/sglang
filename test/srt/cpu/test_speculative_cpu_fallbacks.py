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
