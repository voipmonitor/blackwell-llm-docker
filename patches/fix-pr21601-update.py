#!/usr/bin/env python3
"""Apply PR #21601 update: causal masking fix + piecewise CUDA graph disable.

Two changes:
1. flashinfer_backend.py: Fix dq_page_table to skip padding gaps (causal masking bug)
2. server_args.py: Disable piecewise CUDA graph for fp4_e2m1 KV cache
"""
from pathlib import Path

SGLANG = Path("/opt/sglang/python/sglang/srt")

# --- 1. flashinfer_backend.py: Replace dq_page_table construction ---

fb = SGLANG / "layers/attention/flashinfer_backend.py"
content = fb.read_text()

OLD_DQ_BLOCK = """\
                # [prefix_len, 256] -> [padded_prefix_len, 256] -> sum_tokens -> token_indices[page_size, ..., padde_prefix_len + 256 + page_size]
                paged_seq_lens_cpu.append(256)
                import numpy as np

                paged_seq_lens_cpu = np.array(paged_seq_lens_cpu)
                paged_seq_lens_cpu_padded = (
                    (paged_seq_lens_cpu + self.page_size - 1)
                    // self.page_size
                    * self.page_size
                )
                # Store page-aligned per-request lengths (excluding buffer)
                # for kv_indptr alignment in call_begin_forward
                self.dq_paged_kernel_lens = torch.tensor(
                    paged_seq_lens_cpu_padded[:-1].tolist(),
                    dtype=torch.int32,
                    device=forward_batch.req_pool_indices.device,
                )
                total_paged_tokens = sum(paged_seq_lens_cpu_padded)
                self.dq_page_table = torch.arange(
                    self.page_size,
                    total_paged_tokens + self.page_size,
                    device=forward_batch.req_pool_indices.device,
                    dtype=torch.int32,
                )"""

NEW_DQ_BLOCK = """\
                # dq_buffer layout is page-aligned: each request occupies
                # ceil(seq_len/page_size)*page_size slots, starting after a
                # page_size dummy prefix.  We build dq_page_table (kv_indices)
                # that maps only the *actual* token positions (skipping padding
                # gaps), and dq_paged_kernel_lens stores the real seq_lens so
                # that kv_indptr reflects exact lengths.  This ensures the
                # flashinfer causal offset = seq_len - q_len (correct), not
                # page_align(seq_len) - q_len (wrong, leaks future tokens).
                paged_seq_lens_cpu.append(256)
                import numpy as np

                seq_lens_arr = np.array(paged_seq_lens_cpu)
                padded_lens = (
                    (seq_lens_arr + self.page_size - 1)
                    // self.page_size
                    * self.page_size
                )

                # Compute page-aligned start offsets in dq_buffer
                # (skip page_size dummy at the beginning)
                starts = np.empty(len(seq_lens_arr) + 1, dtype=np.int64)
                starts[0] = self.page_size
                np.cumsum(padded_lens, out=starts[1:])
                starts[1:] += self.page_size

                # Build kv_indices: for each request, only include the
                # actual data positions [start, start+actual_len), skipping
                # the padding [start+actual_len, start+padded_len).
                indices = []
                for i in range(len(seq_lens_arr)):
                    indices.append(np.arange(starts[i], starts[i] + seq_lens_arr[i]))
                self.dq_page_table = torch.tensor(
                    np.concatenate(indices),
                    device=forward_batch.req_pool_indices.device,
                    dtype=torch.int32,
                )

                # Use actual (non-padded) lengths for kv_indptr so causal
                # masking computes the correct offset.  Exclude the trailing
                # 256-token buffer entry.
                self.dq_paged_kernel_lens = torch.tensor(
                    seq_lens_arr[:-1].tolist(),
                    dtype=torch.int32,
                    device=forward_batch.req_pool_indices.device,
                )"""

if OLD_DQ_BLOCK in content:
    content = content.replace(OLD_DQ_BLOCK, NEW_DQ_BLOCK)
    print("OK: flashinfer_backend.py — replaced dq_page_table construction (causal fix)")
else:
    print("WARNING: dq_page_table block not found — may already be patched or changed")

# Also fix the kv_indptr comment in prefill updater
OLD_COMMENT = """\
            # When using custom_kv_indices (NVFP4 dq_page_table), the dq buffer
            # layout is page-aligned per request. Use page-aligned lengths for
            # kv_indptr so that each request's region in dq_page_table matches
            # the actual dq buffer layout. Causal masking ensures padding tokens
            # at the end of each request's region are never attended to."""

NEW_COMMENT = """\
            # When using custom_kv_indices (NVFP4 dq_page_table), use the
            # actual (non-padded) seq_lens stored in dq_paged_kernel_lens for
            # kv_indptr so causal masking computes correct offsets."""

if OLD_COMMENT in content:
    content = content.replace(OLD_COMMENT, NEW_COMMENT)
    print("OK: flashinfer_backend.py — updated kv_indptr comment")

fb.write_text(content)

# --- 2. server_args.py: Disable piecewise CUDA graph for NVFP4 KV cache ---

sa = SGLANG / "server_args.py"
sa_content = sa.read_text()

ANCHOR = "        # 16. Expert distribution recorder"
INSERT = """        # 17. NVFP4 KV cache: piecewise captures extend batches which invoke
        #     trtllm_batch_decode_with_kv_cache with kv_cache_sf (requires
        #     flashinfer >= 0.6.7).  Keep disabled until flashinfer is updated.
        if self.kv_cache_dtype == "fp4_e2m1":
            self.disable_piecewise_cuda_graph = True
"""

if "fp4_e2m1" not in sa_content:
    # Find the anchor and insert after its block
    idx = sa_content.find(ANCHOR)
    if idx >= 0:
        # Find the end of the block (next blank line or next comment)
        block_end = sa_content.find("\n\n", idx)
        if block_end < 0:
            block_end = sa_content.find("\n    def ", idx)
        if block_end >= 0:
            sa_content = sa_content[:block_end] + "\n" + INSERT + sa_content[block_end:]
            print("OK: server_args.py — added piecewise CUDA graph disable for fp4_e2m1")
        else:
            print("WARNING: could not find insertion point in server_args.py")
    else:
        print("WARNING: anchor not found in server_args.py")
else:
    print("OK: server_args.py — fp4_e2m1 check already present")

sa.write_text(sa_content)
