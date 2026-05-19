from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPVerifyInput
from sglang.srt.speculative.frozen_kv_mtp_utils import select_last_verified_seed
from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker


def _worker(topk=1, steps=3, device="cpu"):
    worker = FrozenKVMTPWorker.__new__(FrozenKVMTPWorker)
    worker.topk = topk
    worker.speculative_num_steps = steps
    worker.device = device
    worker.page_size = 1
    return worker


def test_last_target_cache_locs_uses_committed_slots():
    worker = _worker(topk=1)
    req_to_token = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=torch.int64,
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([2, 4], dtype=torch.int64),
        req_pool_indices=torch.tensor([0, 2], dtype=torch.int64),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )

    assert worker._last_target_cache_locs(batch).tolist() == [11, 33]


def test_last_target_cache_locs_repeats_for_topk():
    worker = _worker(topk=2)
    req_to_token = torch.tensor([[10, 11, 12]], dtype=torch.int64)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([3], dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )

    assert worker._last_target_cache_locs(batch).tolist() == [12, 12]


def test_last_target_cache_locs_ignores_prefill_shaped_out_cache_loc():
    worker = _worker(topk=1)
    req_to_token = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([4], dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        out_cache_loc=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
    )

    loc = worker._last_target_cache_locs(batch)

    assert loc.tolist() == [13]
    assert loc.numel() == 1


def test_draft_step_cache_locs_reuses_target_locs_per_step():
    worker = _worker(steps=3)
    forward_batch = SimpleNamespace(
        batch_size=2,
        out_cache_loc=torch.tensor([11, 33], dtype=torch.int64),
    )

    assert worker._draft_step_cache_locs(forward_batch).tolist() == [
        [11, 33],
        [11, 33],
        [11, 33],
    ]


def test_draft_step_cache_locs_supports_step_major_allocated_layout():
    worker = _worker(steps=3)
    forward_batch = SimpleNamespace(
        batch_size=2,
        out_cache_loc=torch.arange(6, dtype=torch.int64),
    )

    assert worker._draft_step_cache_locs(forward_batch).tolist() == [
        [0, 3],
        [1, 4],
        [2, 5],
    ]


def test_draft_step_cache_locs_rejects_bad_loc_count():
    worker = _worker(steps=3)
    forward_batch = SimpleNamespace(
        batch_size=2,
        out_cache_loc=torch.arange(5, dtype=torch.int64),
    )

    with pytest.raises(RuntimeError, match="draft cache locations"):
        worker._draft_step_cache_locs(forward_batch)


def test_frozen_kv_verify_input_uses_local_eagle_field_names():
    verify_input = FrozenKVMTPVerifyInput(
        draft_token=torch.tensor([1, 2], dtype=torch.long),
        custom_mask=torch.ones(2, dtype=torch.bool),
        positions=torch.tensor([4, 5], dtype=torch.int64),
        retrive_index=torch.tensor([[0, 1]], dtype=torch.long),
        retrive_next_token=torch.tensor([[1, -1]], dtype=torch.long),
        retrive_next_sibling=torch.tensor([[-1, -1]], dtype=torch.long),
        retrive_cum_len=None,
        spec_steps=2,
        topk=1,
        draft_token_num=2,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=4,
        seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
    )

    assert verify_input.retrive_index.tolist() == [[0, 1]]


def test_select_last_verified_seed_uses_accept_length_tensor():
    draft_input = SimpleNamespace(
        verified_id=torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64),
        hidden_states=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        accept_length=torch.tensor([1, 2], dtype=torch.int32),
        accept_length_cpu=None,
    )

    ids, hidden = select_last_verified_seed(draft_input)

    assert ids.tolist() == [11, 22]
    assert hidden.tolist() == [[2.0, 3.0], [8.0, 9.0]]


def test_select_last_verified_seed_uses_accept_length_cpu_list():
    draft_input = SimpleNamespace(
        verified_id=torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64),
        hidden_states=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        accept_length=None,
        accept_length_cpu=[1, 2],
    )

    ids, hidden = select_last_verified_seed(draft_input)

    assert ids.tolist() == [11, 22]
    assert hidden.tolist() == [[2.0, 3.0], [8.0, 9.0]]
