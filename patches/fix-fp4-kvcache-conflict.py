#!/usr/bin/env python3
"""Resolve cherry-pick conflicts for PR #21601 (FP4 KV cache SM120).

Two files conflict:
1. flashinfer_backend.py — main added piecewise CUDA graph padding (extra_kv),
   PR added custom_kv_indices branch. Merge: keep both.
2. trtllm_mha_backend.py — main added skip_softmax_threshold_scale_factor,
   PR added kv_cache_sf/mask and changed out_dtype. Merge: keep both params.
"""
import re
import sys


def resolve_conflicts(path):
    """Remove git conflict markers by keeping both sides merged."""
    with open(path) as f:
        src = f.read()

    if '<<<<<<<' not in src:
        print(f'  {path}: no conflicts')
        return

    # Generic conflict resolver: find all conflict blocks
    pattern = r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n'
    conflicts = list(re.finditer(pattern, src, re.DOTALL))
    if not conflicts:
        print(f'ERROR: found <<<<<<< but regex did not match in {path}', file=sys.stderr)
        sys.exit(1)

    print(f'  {path}: {len(conflicts)} conflict(s)')
    return conflicts, src


def fix_flashinfer():
    path = '/opt/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py'
    with open(path) as f:
        src = f.read()

    if '<<<<<<<' not in src:
        print(f'  flashinfer_backend.py: no conflicts')
        return

    # The conflict: main has extra_kv for CUDA graph padding, PR has custom_kv_indices branch.
    # Merged: if custom_kv_indices, use it; else use main's version with extra_kv.
    OLD = re.search(
        r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n',
        src, re.DOTALL
    )
    if not OLD:
        print('ERROR: conflict pattern not found in flashinfer_backend.py', file=sys.stderr)
        sys.exit(1)

    head = OLD.group(1)  # main's version
    pr = OLD.group(2)    # PR's version

    # Merged version: custom_kv_indices branch wrapping main's version
    merged = '''
            if custom_kv_indices is not None:
                kv_indices = custom_kv_indices
            else:
                # Reserve extra space for piecewise CUDA graph dummy request
                fwd_ctx = get_forward_context()
                pcg_num_tokens = fwd_ctx.num_tokens if fwd_ctx is not None else None
                extra_kv = pcg_num_tokens if pcg_num_tokens is not None else 0
                kv_indices = torch.empty(
                    paged_kernel_lens_sum + extra_kv + 256,
                    dtype=torch.int32,
                    device=req_pool_indices.device,
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    paged_kernel_lens,
                    kv_indptr,
                    kv_start_idx,
                    kv_indices,
                    self.req_to_token.shape[1],
                )
'''
    src = src[:OLD.start()] + merged + src[OLD.end():]

    with open(path, 'w') as f:
        f.write(src)
    print('  flashinfer_backend.py: resolved (custom_kv_indices + extra_kv)')


def fix_trtllm():
    path = '/opt/sglang/python/sglang/srt/layers/attention/trtllm_mha_backend.py'
    with open(path) as f:
        src = f.read()

    if '<<<<<<<' not in src:
        print(f'  trtllm_mha_backend.py: no conflicts')
        return

    # Conflict 1 (decode): merge skip_softmax + kv_cache_sf, use q.dtype
    src = src.replace(
        '<<<<<<< HEAD\n'
        '            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),\n'
        '            out_dtype=self.q_data_type,  # model_runner.dtype\n'
        '=======\n'
        '            out_dtype=q.dtype,  # model_runner.dtype\n'
        '            kv_cache_sf=kv_cache_block_scales,\n',
        '            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),\n'
        '            out_dtype=q.dtype,  # model_runner.dtype\n'
        '            kv_cache_sf=kv_cache_block_scales,\n'
    )
    # Remove the trailing >>>>>>> marker from conflict 1
    src = re.sub(r'            kv_cache_sf=kv_cache_block_scales,\n>>>>>>> [^\n]*\n',
                 '            kv_cache_sf=kv_cache_block_scales,\n', src, count=1)

    # Conflict 2 (extend with spec): merge skip_softmax + mask + q_seq_len
    src = src.replace(
        '<<<<<<< HEAD\n'
        '                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),\n'
        '                out_dtype=self.q_data_type,  # model_runner.dtype\n'
        '                q_len_per_req=self.forward_metadata.max_seq_len_q,\n'
        '=======\n'
        '                out_dtype=q.dtype,  # fp4 kv kernel doesn\'t support bf16 output\n'
        '                q_len_per_req=q_seq_len,\n'
        '                mask=spec_mask,\n',
        '                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),\n'
        '                out_dtype=q.dtype,  # fp4 kv kernel doesn\'t support bf16 output\n'
        '                q_len_per_req=q_seq_len,\n'
        '                mask=spec_mask,\n'
    )
    src = re.sub(r'                mask=spec_mask,\n>>>>>>> [^\n]*\n',
                 '                mask=spec_mask,\n', src, count=1)

    # Conflict 3 (prefill/context): merge skip_softmax, keep self.q_data_type
    src = src.replace(
        '<<<<<<< HEAD\n'
        '                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),\n'
        '                out_dtype=self.q_data_type,  # model_runner.dtype\n'
        '=======\n'
        '                out_dtype=self.q_data_type,\n',
        '                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),\n'
        '                out_dtype=self.q_data_type,\n'
    )
    src = re.sub(r'                out_dtype=self\.q_data_type,\n>>>>>>> [^\n]*\n',
                 '                out_dtype=self.q_data_type,\n', src, count=1)

    with open(path, 'w') as f:
        f.write(src)

    # Verify no markers remain
    if '<<<<<<<' in src or '>>>>>>>' in src:
        print('ERROR: conflict markers still present in trtllm_mha_backend.py!', file=sys.stderr)
        # Show remaining markers for debugging
        for i, line in enumerate(src.split('\n'), 1):
            if '<<<<<<<' in line or '>>>>>>>' in line or '=======' in line:
                print(f'  line {i}: {line.strip()}', file=sys.stderr)
        sys.exit(1)

    print('  trtllm_mha_backend.py: resolved (3 conflicts, skip_softmax + kv_cache_sf/mask)')


print('Resolving PR #21601 cherry-pick conflicts...')
fix_flashinfer()
fix_trtllm()
print('OK: PR #21601 conflicts resolved')
