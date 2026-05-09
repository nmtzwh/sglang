from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import is_cuda, is_hip, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


if _is_cuda:
    from sgl_kernel import fast_topk
elif _is_hip:
    from sgl_kernel import fast_topk
else:
    from sglang.srt.utils.common import fast_topk


logger = logging.getLogger(__name__)


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
TREE_SPEC_KERNEL_AVAILABLE = _is_cuda  # This kernel is only available for CUDA now


def top_k_renorm_prob_cpu(
    probs: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    if probs.device.type != "cpu":
        raise ValueError("top_k_renorm_prob_cpu expects CPU tensors.")

    probs = probs.float()
    top_k = top_k.to(device=probs.device, dtype=torch.int64).view(-1)
    renorm_probs = torch.zeros_like(probs)
    vocab_size = probs.shape[-1]

    for i in range(probs.shape[0]):
        k = int(top_k[i].item())
        if k <= 0 or k >= vocab_size:
            row = probs[i]
            denom = row.sum()
            renorm_probs[i] = row / denom if denom > 0 else row
            continue

        values, indices = torch.topk(probs[i], k=k, dim=-1)
        denom = values.sum()
        if denom > 0:
            renorm_probs[i].scatter_(0, indices, values / denom)

    return renorm_probs


def top_p_renorm_prob_cpu(
    probs: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    if probs.device.type != "cpu":
        raise ValueError("top_p_renorm_prob_cpu expects CPU tensors.")

    probs = probs.float()
    top_p = top_p.to(device=probs.device, dtype=torch.float32).view(-1)
    renorm_probs = torch.zeros_like(probs)

    for i in range(probs.shape[0]):
        p = float(top_p[i].item())
        if p >= 1.0:
            row = probs[i]
            denom = row.sum()
            renorm_probs[i] = row / denom if denom > 0 else row
            continue

        sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative_probs <= p
        if keep.numel() > 0:
            keep[0] = True
            first_over = torch.nonzero(cumulative_probs > p, as_tuple=False)
            if first_over.numel() > 0:
                keep[int(first_over[0].item())] = True

        kept_probs = sorted_probs[keep]
        kept_indices = sorted_indices[keep]
        denom = kept_probs.sum()
        if denom > 0:
            renorm_probs[i].scatter_(0, kept_indices, kept_probs / denom)

    return renorm_probs


def _sample_from_probs_cpu(probs: torch.Tensor, coin: float) -> int:
    total = float(probs.sum().item())
    if total <= 0:
        return probs.shape[0] - 1

    threshold = coin * total
    cumulative = torch.cumsum(probs, dim=-1)
    selected = torch.nonzero(cumulative > threshold, as_tuple=False)
    if selected.numel() == 0:
        return probs.shape[0] - 1
    return int(selected[0].item())


def tree_speculative_sampling_target_only_cpu(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    del deterministic
    if target_probs.device.type != "cpu":
        raise ValueError("tree_speculative_sampling_target_only_cpu expects CPU tensors.")

    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]
    threshold_acc = max(float(threshold_acc), 1e-9)

    accept_index.fill_(-1)
    accept_token_num.zero_()

    for bx in range(batch_size):
        prob_acc = 0.0
        cur_prob_index = 0
        coin = float(uniform_samples[bx, 0].item())
        last_accepted_retrive_idx = int(retrive_index[bx, 0].item())
        accept_index[bx, 0] = last_accepted_retrive_idx
        num_accepted_tokens = 0
        cur_index = 0

        for _ in range(1, num_speculative_tokens):
            cur_index = int(retrive_next_token[bx, cur_index].item())
            while cur_index != -1:
                draft_index = int(retrive_index[bx, cur_index].item())
                draft_token_id = int(candidates[bx, cur_index].item())
                target_prob_single = float(
                    target_probs[bx, cur_prob_index, draft_token_id].item()
                )
                prob_acc += target_prob_single

                if (
                    coin <= prob_acc / threshold_acc
                    or target_prob_single >= threshold_single
                ):
                    prob_acc = 0.0
                    cur_prob_index = cur_index
                    coin = float(uniform_samples[bx, cur_index].item())
                    predicts[last_accepted_retrive_idx] = draft_token_id
                    num_accepted_tokens += 1
                    accept_index[bx, num_accepted_tokens] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break

                draft_probs[bx, cur_prob_index, draft_token_id] = target_probs[
                    bx, cur_prob_index, draft_token_id
                ]
                cur_index = int(retrive_next_sibling[bx, cur_index].item())

            if cur_index == -1:
                break

        accept_token_num[bx] = num_accepted_tokens

        if num_accepted_tokens == num_speculative_tokens - 1:
            final_probs = target_probs[bx, cur_prob_index]
        else:
            final_probs = torch.clamp(
                target_probs[bx, cur_prob_index] - draft_probs[bx, cur_prob_index],
                min=0,
            )
        sampled_id = _sample_from_probs_cpu(
            final_probs, float(uniform_samples_for_final_sampling[bx].item())
        )
        predicts[last_accepted_retrive_idx] = sampled_id


class _DeviceDispatchKernel:
    def __init__(self, triton_kernel, cpu_func):
        self.triton_kernel = triton_kernel
        self.cpu_func = cpu_func

    def __getitem__(self, grid):
        triton_launcher = self.triton_kernel[grid]

        def launcher(*args, **kwargs):
            first_tensor = next(
                (arg for arg in args if isinstance(arg, torch.Tensor)), None
            )
            if first_tensor is not None and first_tensor.device.type == "cpu":
                return self.cpu_func(*args, **kwargs)
            return triton_launcher(*args, **kwargs)

        return launcher


def spec_need_hidden_states(server_args: Optional[ServerArgs] = None) -> bool:
    if server_args is None:
        server_args = get_global_server_args()

    # TODO(lsyin): also skip when 1) step = 1 or 2) standalone draft model
    return not server_args.enable_multi_layer_eagle


@triton.jit
def _create_extend_after_decode_spec_info_kernel(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    accept_length = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_length
    tl.store(positions_ptr + offsets, seq_length - accept_length + offsets, mask)

    accept_len_cumsum += accept_length - 1
    verified_id_data = tl.load(verified_id + accept_len_cumsum)
    tl.store(new_verified_id + pid, verified_id_data)


def _create_extend_after_decode_spec_info_cpu(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
    bs_upper,
):
    del bs_upper
    offset = 0
    for i in range(seq_lens.numel()):
        accept_len = int(accept_lens[i].item())
        seq_len = int(seq_lens[i].item())
        positions[offset : offset + accept_len] = torch.arange(
            seq_len - accept_len,
            seq_len,
            dtype=positions.dtype,
            device=positions.device,
        )
        new_verified_id[i] = verified_id[offset + accept_len - 1]
        offset += accept_len


create_extend_after_decode_spec_info = _DeviceDispatchKernel(
    _create_extend_after_decode_spec_info_kernel,
    _create_extend_after_decode_spec_info_cpu,
)


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    if req_pool_indices.device.type == "cpu":
        out_offset = 0
        for i in range(batch_size):
            start = int(start_offset[i].item())
            end = int(end_offset[i].item())
            req_idx = int(req_pool_indices[i].item())
            length = end - start
            req_to_token[req_idx, start:end] = out_cache_loc[
                out_offset : out_offset + length
            ].to(req_to_token.dtype)
            out_offset += length
        return

    assign_req_to_token_pool[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(batch_size),
    )


@triton.jit
def _assign_draft_cache_locs_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len: tl.constexpr,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(tl.load(extend_lens + bs_offset, mask=bs_offset < pid))
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)
    if page_size != 1 and topk != 1 and duplicate_cache_len > 0:
        # Part 2: Copy indices into source_cache_loc and target_cache_loc
        # Expected output: src:[8,9,10,8,9,10...] tgt:[16,17,18,24,25,26...]
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        offsets = tl.arange(0, page_size)
        mask = offsets < last_page_len
        num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
        prefix_base = token_pool + prefix_len - last_page_len
        src_indices = tl.load(prefix_base + offsets, mask=mask)
        last_page_lens_cumsum_ = tl.load(last_page_lens_cumsum + pid)
        # Skip the first one since no copy is needed
        for topk_id in range(1, topk):
            tl.store(
                source_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                src_indices,
                mask=mask,
            )
            tgt_indices = tl.load(
                prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
                mask=mask,
            )
            tl.store(
                target_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                tgt_indices,
                mask=mask,
            )
        # Part 3: Copy and remove the used indices for duplication
        # speculative_num_steps=5, page_size=4, num_new_pages_per_topk_=2, last_page_len=1
        #  - xxxxx .. | - xxxxx .. |
        #   topk=0        topk=1
        #  "-" means prefix tokens
        #  "x" means speculative draft tokens
        #  "." means padded tokens
        # we only want to copy the "x" part.
        iter_offset = tl.arange(0, iter_upper)
        for topk_id in range(topk):
            mask_upper = iter_offset < (speculative_num_steps + last_page_len)
            mask_lower = iter_offset >= last_page_len
            combined_mask = mask_upper & mask_lower
            indices = tl.load(
                prefix_base
                + topk_id * num_new_pages_per_topk_ * page_size
                + iter_offset,
                mask=combined_mask,
                other=0,
            )
            # Shift from previous batches
            ptr_offset = pid * speculative_num_steps * topk
            # Subtract last_page_len to fill the gap of duplicated last page tokens.
            # For example, token pool is (1, 2, 3, 4 ,5) and last page is 1,
            # we write 2, 3, 4 to the front of out_cache_loc.
            tl.store(
                out_cache_loc
                + ptr_offset
                + topk_id * speculative_num_steps
                - last_page_len
                + iter_offset,
                indices,
                mask=combined_mask,
            )


def _assign_draft_cache_locs_cpu(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len,
    pool_len,
    topk,
    speculative_num_steps,
    page_size,
    bs_upper,
    iter_upper,
):
    del duplicate_cache_len, pool_len, bs_upper, iter_upper
    num_seqs = req_pool_indices.numel()
    out_offset = 0
    original_out_cache_loc = out_cache_loc.clone()

    for pid in range(num_seqs):
        req_idx = int(req_pool_indices[pid].item())
        kv_start = int(seq_lens[pid].item())
        if page_size == 1 or topk == 1:
            copy_len = topk * speculative_num_steps
            out_start = pid * topk * speculative_num_steps
        else:
            copy_len = int(extend_lens[pid].item())
            out_start = out_offset
            out_offset += copy_len

        req_to_token[req_idx, kv_start : kv_start + copy_len] = original_out_cache_loc[
            out_start : out_start + copy_len
        ].to(req_to_token.dtype)

        if page_size == 1 or topk == 1:
            continue

        last_page_len = kv_start % page_size
        if last_page_len == 0:
            continue

        num_new_pages = int(num_new_pages_per_topk[pid].item())
        prefix_base = kv_start - last_page_len
        src_indices = req_to_token[
            req_idx, prefix_base : prefix_base + last_page_len
        ].to(source_cache_loc.dtype)
        last_page_lens_cumsum_cur = int(last_page_lens_cumsum[pid].item())
        dup_base = (topk - 1) * (last_page_lens_cumsum_cur - last_page_len)

        for topk_id in range(1, topk):
            dup_start = dup_base + (topk_id - 1) * last_page_len
            source_cache_loc[dup_start : dup_start + last_page_len] = src_indices
            tgt_start = prefix_base + topk_id * num_new_pages * page_size
            target_cache_loc[dup_start : dup_start + last_page_len] = req_to_token[
                req_idx, tgt_start : tgt_start + last_page_len
            ].to(target_cache_loc.dtype)

        ptr_offset = pid * speculative_num_steps * topk
        for topk_id in range(topk):
            src_start = (
                prefix_base
                + topk_id * num_new_pages * page_size
                + last_page_len
            )
            dst_start = ptr_offset + topk_id * speculative_num_steps
            out_cache_loc[dst_start : dst_start + speculative_num_steps] = req_to_token[
                req_idx, src_start : src_start + speculative_num_steps
            ].to(out_cache_loc.dtype)


assign_draft_cache_locs = _DeviceDispatchKernel(
    _assign_draft_cache_locs_kernel,
    _assign_draft_cache_locs_cpu,
)


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid, other=0)
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        extend_data = tl.load(
            token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        prefix_len = seq_len
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def _align_evict_mask_to_page_size_kernel(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


def _align_evict_mask_to_page_size_cpu(
    seq_lens,
    evict_mask,
    page_size,
    num_draft_tokens,
    BLOCK_SIZE,
):
    del BLOCK_SIZE
    evict_mask_2d = evict_mask.view(seq_lens.numel(), num_draft_tokens)
    for bid in range(seq_lens.numel()):
        seq_len = int(seq_lens[bid].item())
        num_trues = int(evict_mask_2d[bid].sum().item())
        num_false = num_draft_tokens - num_trues
        start = ((seq_len + num_false - 1) // page_size) * page_size - seq_len
        for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
            evict_mask_2d[bid, i] = False


align_evict_mask_to_page_size = _DeviceDispatchKernel(
    _align_evict_mask_to_page_size_kernel,
    _align_evict_mask_to_page_size_cpu,
)


@triton.jit
def get_target_cache_loc(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(accept_length + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


@torch.compile(dynamic=True, disable=_is_npu)
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    tgt_cache_loc = torch.empty_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = torch.minimum(
        (seq_lens + accept_length + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@triton.jit
def filter_finished_cache_loc_kernel(
    out_cache_loc,
    tgt_cache_loc,
    accept_length,
    accept_length_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    accept_length_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    old_start = tl.sum(accept_length_all) + bid

    accept_length_filter_all = tl.load(
        accept_length_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(accept_length_filter_all)

    copy_len = tl.load(accept_length_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


@torch.compile(dynamic=True, disable=_is_npu)
def create_accept_length_filter(
    accept_length: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    accept_length_filter = torch.zeros_like(accept_length)
    accept_length_filter[unfinished_index_device] = (
        accept_length[unfinished_index_device] + 1
    )
    seq_lens.add_(accept_length + 1)
    return accept_length_filter


@torch.compile(dynamic=True, disable=_is_npu)
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        if hidden_states is not None:
            hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device=topk_index.device
            ).repeat_interleave(topk)
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def generate_simulated_accept_index(
    accept_index,
    predict,
    accept_length,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
):
    assert simulate_acc_len > 0.0

    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and self.spec_steps
        simulated_values = torch.clamp(simulated_values, min=1.0, max=spec_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len = max(1.0, min(spec_steps + 1, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < spec_steps + 1 else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    accept_length.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            if vocab_size and curr_token_id >= vocab_size:
                accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                accepted = (
                    parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))
                ) != 0

        if accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        retrieve_next_token[curr],
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                retrieve_next_sibling[curr],
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    verify_input.grammar = grammar
    return allocate_token_bitmask


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        cache_dir = snapshot_download(
            os.path.dirname(token_map_path),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        token_map_path = os.path.join(cache_dir, os.path.basename(token_map_path))
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with patch_tensor_parallel_group(tp_group):
        yield


def maybe_detect_nan(tensor: torch.Tensor, msg: str = ""):
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_SPEC_NAN_DETECTION.get():
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_oob(indices: torch.Tensor, low: int, high: int, msg: str):
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_SPEC_OOB_DETECTION.get():
        return
    if indices.numel() == 0:
        return
    torch._assert_async(
        (indices.min() >= low) & (indices.max() < high),
        f"OOB indices not in [{low}, {high}): {msg}",
    )


# Disable torch.compile for this function because it will be
# even slower.
# @torch.compile(dynamic=True)
def get_last_loc_large_page_size_large_top_k(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    speculative_num_steps: int,
    topk: int,
    page_size: int,
):
    prefix_lens = seq_lens
    last_page_lens = prefix_lens % page_size
    num_new_pages_per_topk = (
        last_page_lens + speculative_num_steps + page_size - 1
    ) // page_size
    seq_lens = prefix_lens // page_size * page_size + num_new_pages_per_topk * (
        page_size * topk
    )
    extend_lens = seq_lens - prefix_lens
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )

    return (
        prefix_lens,
        seq_lens,
        last_loc,
        num_new_pages_per_topk,
        extend_lens,
        last_page_lens,
    )
