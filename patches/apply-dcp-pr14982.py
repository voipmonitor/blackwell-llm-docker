#!/usr/bin/env python3
"""
Apply DCP (Decode Context Parallel) patch from SGLang PR #14982.
Ported to current SGLang main (post-commit a0862f00c).

DCP splits KV cache across TP ranks (interleaved tokens), reducing per-GPU
KV memory usage. Uses --dcp-size N --enable-symm-mem.
"""

import os
import sys
import re
import textwrap


def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  PATCHED: {path}")


def patch_server_args(sglang_root):
    """Add --dcp-size argument to ServerArgs."""
    path = os.path.join(sglang_root, "srt/server_args.py")
    src = read_file(path)

    if "dcp_size" in src:
        print(f"  SKIP: {path} (already has dcp_size)")
        return

    # Add field after tp_size (handle both old and new signatures)
    if "    tp_size: int = 1\n    pp_size: int = 1" in src:
        src = src.replace(
            "    tp_size: int = 1\n    pp_size: int = 1",
            "    tp_size: int = 1\n    dcp_size: int = 1\n    pp_size: int = 1",
        )
    elif "    tp_size: int = 1\n" in src:
        src = src.replace(
            "    tp_size: int = 1\n",
            "    tp_size: int = 1\n    dcp_size: int = 1\n",
            1,
        )

    # Add CLI argument after --tensor-parallel-size block
    # Find the --tensor-parallel-size argument and add --dcp-size after it
    tp_arg_pattern = r'(        parser\.add_argument\(\s*"--tensor-parallel-size".*?help="The tensor parallelism size\."\s*,\s*\))'
    tp_match = re.search(tp_arg_pattern, src, re.DOTALL)
    if tp_match:
        insert_pos = tp_match.end()
        dcp_arg = '''
        parser.add_argument(
            "--decode-context-parallel-size",
            "--dcp-size",
            type=int,
            default=ServerArgs.dcp_size,
            help="The decode context parallel size.",
        )'''
        src = src[:insert_pos] + dcp_arg + src[insert_pos:]

    # Add dcp_size to from_cli_args
    src = src.replace(
        "        args.tp_size = args.tensor_parallel_size\n        args.pp_size",
        "        args.tp_size = args.tensor_parallel_size\n        args.dcp_size = args.decode_context_parallel_size\n        args.pp_size",
    )

    write_file(path, src)


def patch_parallel_state(sglang_root):
    """Add DCP group coordinator."""
    path = os.path.join(sglang_root, "srt/distributed/parallel_state.py")
    src = read_file(path)

    if "_DCP" in src:
        print(f"  SKIP: {path} (already has _DCP)")
        return

    # Add reduce_scatter_along_dim method to GroupCoordinator
    # Find the reduce_scatter method and add after it
    reduce_scatter_end = src.find("        return output\n", src.find("def reduce_scatter("))
    if reduce_scatter_end == -1:
        # Try alternative pattern
        reduce_scatter_marker = "torch.distributed.reduce_scatter(output, input_list, group=self.device_group)\n        return output"
        reduce_scatter_end = src.find(reduce_scatter_marker)
        if reduce_scatter_end != -1:
            reduce_scatter_end += len(reduce_scatter_marker)

    if reduce_scatter_end != -1:
        new_method = '''

    def reduce_scatter_along_dim(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            dim += input_.dim()
        with self.use_symmetric_memory(self):
            input_tensor = input_.movedim(0, dim).contiguous()
        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]
        with self.use_symmetric_memory(self):
            output_tensor = torch.empty(
                output_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
        self.reduce_scatter_tensor(output_tensor, input_tensor)
        return output_tensor.movedim(0, dim).contiguous()
'''
        src = src[:reduce_scatter_end] + new_method + src[reduce_scatter_end:]

    # Add _DCP global and getter after _PP getter
    pp_getter = "def get_pp_group() -> GroupCoordinator:"
    pp_getter_end = src.find("\n\n", src.find(pp_getter))
    if pp_getter_end != -1:
        # Find the full function including return
        pp_block_end = src.find("\n\n\n", src.find(pp_getter))
        if pp_block_end == -1:
            pp_block_end = pp_getter_end
        dcp_code = '''


_DCP: Optional[GroupCoordinator] = None


def get_dcp_group() -> GroupCoordinator:
    assert _DCP is not None, "decode context parallel group is not initialized"
    return _DCP

'''
        src = src[:pp_block_end] + dcp_code + src[pp_block_end:]

    # Patch graph_capture to include DCP group
    src = src.replace(
        "    ) as context, get_pp_group().graph_capture(context):",
        "    ) as context, get_pp_group().graph_capture(context), get_dcp_group().graph_capture(\n        context\n    ):",
    )

    # Add decode_context_parallel_size param to initialize_model_parallel
    # Handle both old and new function signatures
    if "    pipeline_model_parallel_size: int = 1,\n    backend: Optional[str] = None,\n    duplicate_tp_group" in src:
        src = src.replace(
            "    pipeline_model_parallel_size: int = 1,\n    backend: Optional[str] = None,\n    duplicate_tp_group",
            "    pipeline_model_parallel_size: int = 1,\n    decode_context_parallel_size: int = 1,\n    backend: Optional[str] = None,\n    duplicate_tp_group",
        )
    else:
        # New signature has more params between pipeline and backend
        import re as _re
        pattern = r'(    pipeline_model_parallel_size: int = 1,\n(?:    \w+.*\n)*?)(    backend: Optional\[str\] = None,\n    duplicate_tp_group)'
        repl = r'\g<1>    decode_context_parallel_size: int = 1,\n\g<2>'
        src = _re.sub(pattern, repl, src, count=1)

    # Add DCP group initialization in initialize_model_parallel
    # Find the moe_ep_size line and add DCP init before it
    moe_marker = "    moe_ep_size = expert_model_parallel_size"
    moe_pos = src.find(moe_marker)
    if moe_pos != -1:
        dcp_init = '''    # Build the decode context parallel groups.
    num_decode_context_parallel_groups: int = world_size // decode_context_parallel_size
    global _DCP
    assert _DCP is None, "decode context parallel group is already initialized"
    group_ranks = []
    for i in range(num_decode_context_parallel_groups):
        ranks = list(
            range(
                i * decode_context_parallel_size,
                (i + 1) * decode_context_parallel_size,
            )
        )
        group_ranks.append(ranks)
    _DCP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        group_name="dcp",
    )

'''
        src = src[:moe_pos] + dcp_init + src[moe_pos:]

    # Add decode_context_parallel_size to ensure_model_parallel_initialized
    ensure_fn = "def ensure_model_parallel_initialized("
    if ensure_fn in src and "decode_context_parallel_size" not in src[src.find(ensure_fn):src.find(ensure_fn)+500]:
        # Find the function and add the param before backend
        idx = src.find(ensure_fn)
        fn_body = src[idx:idx+500]
        if "    pipeline_model_parallel_size: int,\n    backend:" in fn_body:
            src = src[:idx] + fn_body.replace(
                "    pipeline_model_parallel_size: int,\n    backend:",
                "    pipeline_model_parallel_size: int,\n    decode_context_parallel_size: int,\n    backend:",
            ) + src[idx+500:]
        # Also pass it to initialize_model_parallel call inside
        src = src.replace(
            "            pipeline_model_parallel_size,\n            backend,\n        )\n        return",
            "            pipeline_model_parallel_size,\n            decode_context_parallel_size,\n            backend,\n        )\n        return",
        )

    # Add DCP cleanup in destroy_model_parallel
    tp_destroy = "    global _TP\n    if _TP:\n        _TP.destroy()\n    _TP = None"
    tp_destroy_pos = src.find(tp_destroy)
    if tp_destroy_pos != -1:
        dcp_destroy = '''

    global _DCP
    if _DCP:
        _DCP.destroy()
    _DCP = None
'''
        src = src[:tp_destroy_pos + len(tp_destroy)] + dcp_destroy + src[tp_destroy_pos + len(tp_destroy):]

    write_file(path, src)


def patch_pynccl_allocator(sglang_root):
    """Per-group memory pool map."""
    path = os.path.join(sglang_root, "srt/distributed/device_communicators/pynccl_allocator.py")
    src = read_file(path)

    if "_mem_pool_map" in src:
        print(f"  SKIP: {path} (already has _mem_pool_map)")
        return

    # Change _mem_pool to _mem_pool_map
    src = src.replace("_mem_pool = None\n", "_mem_pool_map = {}\n")

    # Split get_nccl_mem_pool into init + per-group pool
    old_fn = '''def get_nccl_mem_pool():
    global _allocator, _mem_pool, _cur_device
    if _mem_pool is None:'''
    if old_fn in src:
        # Find the full function
        fn_start = src.find(old_fn)
        # Find the return statement
        fn_return = src.find("    return _mem_pool\n", fn_start)
        if fn_return != -1:
            fn_end = fn_return + len("    return _mem_pool\n")
            new_fn = '''def init_pynccl_allocator():
    global _allocator, _cur_device
    if _allocator is None:'''

            # Get the body between the old function header and _mem_pool creation
            old_body_start = fn_start + len(old_fn)
            mem_pool_line = src.find("        _mem_pool = torch.cuda.MemPool(_allocator)\n", fn_start)
            cur_device_line = src.find("        _cur_device = torch.cuda.current_device()\n", fn_start)

            # Extract the setup code (imports, compilation, etc.)
            setup_code = src[old_body_start:mem_pool_line]

            new_code = new_fn + setup_code + "        _cur_device = torch.cuda.current_device()\n"
            new_code += "\n\n"
            new_code += '''def get_nccl_mem_pool(group_coordinator=None):
    global _allocator, _mem_pool_map
    key = group_coordinator.unique_name if group_coordinator is not None else "__default__"
    if key not in _mem_pool_map:
        init_pynccl_allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
        _mem_pool_map[key] = _mem_pool
    return _mem_pool_map[key]
'''
            src = src[:fn_start] + new_code + src[fn_end:]

    # Update calls to get_nccl_mem_pool to pass group_coordinator
    src = src.replace(
        "torch.cuda.use_mem_pool(get_nccl_mem_pool())",
        "torch.cuda.use_mem_pool(get_nccl_mem_pool(self.group_coordinator))",
    )

    write_file(path, src)


def patch_engine(sglang_root):
    """Set NCCL_GRAPH_MIXING_SUPPORT for DCP."""
    path = os.path.join(sglang_root, "srt/entrypoints/engine.py")
    src = read_file(path)

    if "NCCL_GRAPH_MIXING_SUPPORT" in src:
        print(f"  SKIP: {path} (already has NCCL_GRAPH_MIXING_SUPPORT)")
        return

    # Find a good insertion point
    marker = '    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"]'
    pos = src.find(marker)
    if pos != -1:
        insert = '''    if "NCCL_GRAPH_MIXING_SUPPORT" not in os.environ and server_args.dcp_size > 1:
        os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
'''
        src = src[:pos] + insert + src[pos:]

    write_file(path, src)


def patch_attention_utils(sglang_root):
    """Add DCP Triton kernels and correction functions."""
    path = os.path.join(sglang_root, "srt/layers/attention/utils.py")
    src = read_file(path)

    if "create_flashinfer_kv_indices_for_dcp_triton" in src:
        print(f"  SKIP: {path} (already has DCP kernels)")
        return

    # Add import
    if "from sglang.srt.distributed.parallel_state import GroupCoordinator" not in src:
        src = src.replace(
            "import triton.language as tl",
            "import triton.language as tl\n\nfrom sglang.srt.distributed.parallel_state import GroupCoordinator",
        )

    # Add DCP Triton kernel after create_flashinfer_kv_indices_triton
    marker = "        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)\n"
    # Find the last occurrence of this pattern (the one in the original kernel)
    pos = src.find(marker)
    if pos != -1:
        # Find the next blank line after the function
        next_fn = src.find("\n\ndef ", pos)
        if next_fn == -1:
            next_fn = src.find("\n\n@triton.jit", pos)
        if next_fn == -1:
            next_fn = len(src)

        dcp_kernel = '''


@triton.jit
def create_flashinfer_kv_indices_for_dcp_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    dcp_page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)
    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(dcp_page_kernel_lens_ptr + pid).to(tl.int32)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset * dcp_size
            + dcp_rank,
            mask=mask,
        )
        data = data / dcp_size
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)

'''
        # Insert right after the end of the original kernel function
        src = src[:next_fn] + dcp_kernel + src[next_fn:]

    # Add correction kernel and helper functions at the end
    dcp_attention_code = '''

# --- DCP (Decode Context Parallel) attention correction utilities ---
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.12.0/vllm/attention/ops/common.py

@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)
    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )
    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    lse_exp = tl.exp2(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    lse = tl.log2(lse_acc)
    lse += lse_max
    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor, lses: torch.Tensor, cp_rank: int, ctx: CPTritonContext
) -> tuple:
    if ctx is None:
        ctx = CPTritonContext()
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"
    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3
    B, H, D = out.shape
    N = lses.shape[0]
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )
    grid = (B, H, 1)
    regular_args = (
        out, out, lses, lse,
        o_sB, o_sH, o_sD,
        l_sN, l_sB, l_sH,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N}
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext = None,
    return_lse: bool = False,
):
    if ctx is None:
        ctx = CPTritonContext()
    assert cp_attn_lse.is_contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).view(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    out, lse = correct_attn_out(cp_attn_out, lses, cp_group.rank_in_group, ctx)
    assert out.is_contiguous()
    out = cp_group.all_reduce(out)
    cp_num_heads = lse.shape[1] // cp_group.world_size
    cp_rank = cp_group.rank_in_group
    out = out[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1), :]
    if return_lse:
        lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
        return out, lse
    return out
'''

    src += dcp_attention_code
    write_file(path, src)


def patch_flashinfer_backend(sglang_root):
    """Add DCP support to FlashInfer attention backend."""
    path = os.path.join(sglang_root, "srt/layers/attention/flashinfer_backend.py")
    src = read_file(path)

    if "dcp_size" in src:
        print(f"  SKIP: {path} (already has dcp_size)")
        return

    # Add imports
    if "from sglang.srt.distributed.parallel_state import get_dcp_group" not in src:
        src = src.replace(
            "from sglang.srt.layers.attention.base_attn_backend import AttentionBackend",
            "from sglang.srt.distributed.device_communicators.pynccl_allocator import (\n    use_symmetric_memory,\n)\nfrom sglang.srt.distributed.parallel_state import get_dcp_group\nfrom sglang.srt.layers.attention.base_attn_backend import AttentionBackend",
        )

    # Replace create_flashinfer_kv_indices_triton import with DCP version
    src = src.replace(
        "from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton",
        "from sglang.srt.layers.attention.utils import (\n    cp_lse_ag_out_rs,\n    create_flashinfer_kv_indices_for_dcp_triton,\n    create_flashinfer_kv_indices_triton,\n)",
    )

    # Add DCP size/rank initialization in __init__
    # Find the decode_use_tensor_cores line and add DCP init before it
    tensor_core_marker = "        self.decode_use_tensor_cores = should_use_tensor_core("
    pos = src.find(tensor_core_marker)
    if pos != -1:
        dcp_init = "        self.dcp_size = get_dcp_group().world_size\n        self.dcp_rank = get_dcp_group().rank_in_group\n\n"
        src = src[:pos] + dcp_init + src[pos:]

    # Modify decode_use_tensor_cores to account for DCP
    src = src.replace(
        "            num_attention_heads=model_runner.model_config.num_attention_heads\n            // get_attention_tp_size(),",
        "            num_attention_heads=model_runner.model_config.num_attention_heads\n            // get_attention_tp_size()\n            * self.dcp_size,",
    )

    # Add dcp_size/dcp_rank params to all update() calls
    # This is the trickiest part - need to find all call_begin_forward calls and add DCP params

    # For decode updater update calls, add dcp params
    src = src.replace(
        "                fixed_split_size=self.decode_split_tile_size,\n                disable_split_kv=False,",
        "                fixed_split_size=self.decode_split_tile_size,\n                disable_split_kv=False,\n                dcp_size=self.dcp_size,\n                dcp_rank=self.dcp_rank,",
    )

    src = src.replace(
        "                fixed_split_size=self.prefill_split_tile_size,\n                multi_item_params=multi_item_params,",
        "                fixed_split_size=self.prefill_split_tile_size,\n                multi_item_params=multi_item_params,\n                dcp_size=self.dcp_size,\n                dcp_rank=self.dcp_rank,",
    )

    # For cuda graph calls
    src = src.replace(
        "                fixed_split_size=None,\n                disable_split_kv=self.disable_cuda_graph_kv_split,\n            )\n            self.decode_cuda_graph_metadata",
        "                fixed_split_size=None,\n                disable_split_kv=self.disable_cuda_graph_kv_split,\n                dcp_size=self.dcp_size,\n                dcp_rank=self.dcp_rank,\n            )\n            self.decode_cuda_graph_metadata",
    )

    # For update_from_cuda_graph
    src = src.replace(
        "                fixed_split_size=None,\n                disable_split_kv=self.disable_cuda_graph_kv_split,\n            )\n        elif forward_mode.is_target_verify",
        "                fixed_split_size=None,\n                disable_split_kv=self.disable_cuda_graph_kv_split,\n                dcp_size=self.dcp_size,\n                dcp_rank=self.dcp_rank,\n            )\n        elif forward_mode.is_target_verify",
    )

    # Add DCP params to FlashInferIndicesUpdaterDecode.update() signature
    src = src.replace(
        "        fixed_split_size: Optional[int] = None,\n        disable_split_kv: Optional[bool] = None,\n    ):\n        decode_wrappers = decode_wrappers or self.decode_wrappers\n        self.call_begin_forward(",
        "        fixed_split_size: Optional[int] = None,\n        disable_split_kv: Optional[bool] = None,\n        dcp_size: Optional[int] = None,\n        dcp_rank: Optional[int] = None,\n    ):\n        decode_wrappers = decode_wrappers or self.decode_wrappers\n        self.call_begin_forward(",
    )

    # Pass dcp params through to call_begin_forward in decode updater
    src = src.replace(
        "            fixed_split_size=fixed_split_size,\n            disable_split_kv=disable_split_kv,\n        )",
        "            fixed_split_size=fixed_split_size,\n            disable_split_kv=disable_split_kv,\n            dcp_size=dcp_size,\n            dcp_rank=dcp_rank,\n        )",
        1,  # Only first occurrence (in decode updater update method)
    )

    # Add dcp params to call_begin_forward signature (decode)
    src = src.replace(
        "        use_sliding_window_kv_pool: bool = False,\n        fixed_split_size: Optional[int] = None,\n        disable_split_kv: Optional[bool] = None,\n    ):\n        if spec_info is None:",
        "        use_sliding_window_kv_pool: bool = False,\n        fixed_split_size: Optional[int] = None,\n        disable_split_kv: Optional[bool] = None,\n        dcp_size: Optional[int] = None,\n        dcp_rank: Optional[int] = None,\n    ):\n        if dcp_size is None:\n            dcp_size = 1\n            dcp_rank = 0\n        if spec_info is None:",
    )

    # Modify paged_kernel_lens for DCP in decode
    src = src.replace(
        "        if spec_info is None:\n            bs = len(req_pool_indices)\n            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)",
        "        if spec_info is None:\n            bs = len(req_pool_indices)\n            if dcp_size > 1:\n                paged_kernel_lens = paged_kernel_lens // dcp_size + (\n                    dcp_rank < paged_kernel_lens % dcp_size\n                )\n            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)",
    )

    # Replace create_flashinfer_kv_indices_triton with DCP version in decode
    src = src.replace(
        "            create_flashinfer_kv_indices_triton[(bs,)](\n                self.req_to_token,\n                req_pool_indices,\n                paged_kernel_lens,\n                kv_indptr,\n                kv_start_idx,\n                kv_indices,\n                self.req_to_token.shape[1],\n            )",
        "            create_flashinfer_kv_indices_for_dcp_triton[(bs,)](\n                self.req_to_token,\n                req_pool_indices,\n                paged_kernel_lens,\n                kv_indptr,\n                kv_start_idx,\n                kv_indices,\n                self.req_to_token.shape[1],\n                dcp_size,\n                dcp_rank,\n            )",
    )

    # Multiply num_qo_heads by dcp_size in begin_forward calls (decode)
    src = src.replace(
        "                self.num_qo_heads,\n                self.num_kv_heads,\n                self.head_dim,\n                1,",
        "                self.num_qo_heads * dcp_size,\n                self.num_kv_heads,\n                self.head_dim,\n                1,",
    )

    # Handle global_override_indptr_cpu for DCP
    old_global = "            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)"
    if old_global in src:
        new_global = """            if dcp_size > 1:
                dcp_seq_lens_cpu = seq_lens_cpu // dcp_size + (
                    dcp_rank < seq_lens_cpu % dcp_size
                )
                global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(
                    dcp_seq_lens_cpu, dim=0
                )
            else:
                global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(
                    seq_lens_cpu, dim=0
                )"""
        src = src.replace(old_global, new_global)

    # Add DCP params to prefill updater
    src = src.replace(
        "        fixed_split_size: Optional[int] = None,\n        multi_item_params: Optional[MultiItemScoringParams] = None,\n    ):\n        if use_ragged:",
        "        fixed_split_size: Optional[int] = None,\n        multi_item_params: Optional[MultiItemScoringParams] = None,\n        dcp_size: Optional[int] = None,\n        dcp_rank: Optional[int] = None,\n    ):\n        if use_ragged:",
    )

    # Pass DCP params in prefill updater
    src = src.replace(
        "            fixed_split_size=fixed_split_size,\n            multi_item_params=multi_item_params,\n        )",
        "            fixed_split_size=fixed_split_size,\n            multi_item_params=multi_item_params,\n            dcp_size=dcp_size,\n            dcp_rank=dcp_rank,\n        )",
    )

    # Add DCP params to prefill call_begin_forward signature
    src = src.replace(
        "        fixed_split_size: Optional[int] = None,\n        multi_item_params: Optional[MultiItemScoringParams] = None,\n    ):\n        bs = len(seq_lens)",
        "        fixed_split_size: Optional[int] = None,\n        multi_item_params: Optional[MultiItemScoringParams] = None,\n        dcp_size: Optional[int] = None,\n        dcp_rank: Optional[int] = None,\n    ):\n        if dcp_size is not None and dcp_size > 1:\n            if not use_ragged:\n                # Non-ragged (hybrid models) - fall back to no DCP\n                dcp_size = 1\n                dcp_rank = 0\n            else:\n                paged_kernel_lens = paged_kernel_lens // dcp_size + (\n                    dcp_rank < paged_kernel_lens % dcp_size\n                )\n        else:\n            dcp_size = 1\n            dcp_rank = 0\n\n        bs = len(seq_lens)",
    )

    # Replace create_flashinfer_kv_indices_triton in prefill
    old_prefill_kernel = "            create_flashinfer_kv_indices_triton[(bs,)](\n                self.req_to_token,\n                req_pool_indices,\n                paged_kernel_lens,\n                kv_indptr,\n                kv_start_idx,\n                kv_indices,\n                self.req_to_token.shape[1],\n            )"
    if old_prefill_kernel in src:
        new_prefill_kernel = "            create_flashinfer_kv_indices_for_dcp_triton[(bs,)](\n                self.req_to_token,\n                req_pool_indices,\n                paged_kernel_lens,\n                kv_indptr,\n                kv_start_idx,\n                kv_indices,\n                self.req_to_token.shape[1],\n                dcp_size,\n                dcp_rank,\n            )"
        src = src.replace(old_prefill_kernel, new_prefill_kernel)

    # Multiply num_qo_heads in prefill begin_forward (already handled above - check)

    # Now handle the actual attention forward methods
    # Decode forward: add DCP q all-gather and output correction
    old_decode_forward = """        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )"""
    if old_decode_forward in src:
        new_decode_forward = """        if self.dcp_size > 1:
            with use_symmetric_memory(get_dcp_group()):
                q = q.clone(memory_format=torch.contiguous_format)
            q = get_dcp_group().all_gather(q, dim=1)
        else:
            q = q.contiguous()

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if self.dcp_size > 1 and hasattr(forward_batch.token_to_kv_pool, 'k_buffer'):
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale,
                        dcp_kv_mask=forward_batch.dcp_kv_mask,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale,
                    )

        # Call the wrapped function
        with use_symmetric_memory(get_dcp_group()):
            o, s = decode_wrapper.forward_return_lse(
                q.view(-1, layer.tp_q_head_num * self.dcp_size, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        if self.dcp_size > 1:
            o = cp_lse_ag_out_rs(o, s, get_dcp_group())"""

        # Remove original KV cache set before the forward call
        old_kv_set_decode = """        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Call the wrapped function"""
        if old_kv_set_decode in src:
            src = src.replace(old_kv_set_decode, "        # Call the wrapped function")

        src = src.replace(old_decode_forward, new_decode_forward)

    # Prefill forward: DCP-aware set_kv_buffer (only for pools that support it)
    old_prefill_save = "            if save_kv_cache:\n                forward_batch.token_to_kv_pool.set_kv_buffer(\n                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale\n                )\n\n        return o.view(-1, layer.tp_q_head_num * layer.head_dim)"
    if old_prefill_save in src:
        new_prefill_save = """            if save_kv_cache:
                if self.dcp_size > 1 and hasattr(forward_batch.token_to_kv_pool, 'k_buffer'):
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale,
                        dcp_kv_mask=forward_batch.dcp_kv_mask,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale,
                    )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)"""
        src = src.replace(old_prefill_save, new_prefill_save)

    # NOTE: Do NOT add DCP assertion for non-ragged path.
    # Hybrid models (Qwen3.5 with Mamba) use non-ragged for some layers.
    # DCP simply doesn't activate for those layers (dcp_size handled in updater).

    # Add DCP q all-gather for prefill ragged path (paged wrapper)
    # Find the paged wrapper forward_return_lse call and add DCP
    old_paged = "                o2, s2 = prefill_wrapper_paged.forward_return_lse(\n                    q.view(-1, layer.tp_q_head_num, layer.head_dim),"
    if old_paged in src:
        new_paged = "                if self.dcp_size > 1:\n                    q = get_dcp_group().all_gather(q, dim=1)\n                o2, s2 = prefill_wrapper_paged.forward_return_lse(\n                    q.view(-1, layer.tp_q_head_num * self.dcp_size, layer.head_dim),"
        src = src.replace(old_paged, new_paged)

    # Add DCP correction after o2, s2
    old_merge = "                o, _ = merge_state(o1, s1, o2, s2)"
    if old_merge in src:
        new_merge = """                if self.dcp_size > 1:
                    o2, s2 = cp_lse_ag_out_rs(
                        o2,
                        s2,
                        get_dcp_group(),
                        return_lse=True,
                    )
                    o2 = o2.contiguous()
                    s2 = s2.contiguous()

                o, _ = merge_state(o1, s1, o2, s2)"""
        src = src.replace(old_merge, new_merge)

    write_file(path, src)


def patch_flashinfer_mla_backend(sglang_root):
    """Fix import path for create_flashinfer_kv_indices_triton."""
    path = os.path.join(sglang_root, "srt/layers/attention/flashinfer_mla_backend.py")
    src = read_file(path)

    # Fix import if it imports from flashinfer_backend
    old_import = "from sglang.srt.layers.attention.flashinfer_backend import (\n    create_flashinfer_kv_indices_triton,\n)"
    new_import = "from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton"
    if old_import in src:
        src = src.replace(old_import, new_import)
        write_file(path, src)
    else:
        print(f"  SKIP: {path} (import already correct or different pattern)")


def patch_memory_pool(sglang_root):
    """Add masked_set_kv_buffer_kernel for DCP."""
    path = os.path.join(sglang_root, "srt/mem_cache/memory_pool.py")
    src = read_file(path)

    if "masked_set_kv_buffer_kernel" in src:
        print(f"  SKIP: {path} (already has masked_set_kv_buffer_kernel)")
        return

    # Add dcp_kv_mask param to set_kv_buffer
    old_set_sig = "        k_scale: Optional[float] = None,\n        v_scale: Optional[float] = None,\n        layer_id_override: Optional[int] = None,\n    ):"
    if old_set_sig in src:
        new_set_sig = "        k_scale: Optional[float] = None,\n        v_scale: Optional[float] = None,\n        layer_id_override: Optional[int] = None,\n        dcp_kv_mask: Optional[torch.Tensor] = None,\n    ):"
        src = src.replace(old_set_sig, new_set_sig, 1)

    # Add DCP masked write path before the normal _set_kv_buffer_impl call
    old_set_impl = "        _set_kv_buffer_impl(\n            cache_k,\n            cache_v,\n            self.k_buffer[layer_id - self.start_layer],\n            self.v_buffer[layer_id - self.start_layer],\n            loc,"
    pos = src.find(old_set_impl)
    if pos != -1:
        # Find the full call including closing paren
        call_end = src.find("\n        )\n", pos)
        if call_end != -1:
            call_end += len("\n        )\n")
            old_call = src[pos:call_end]
            new_call = """        if dcp_kv_mask is not None:
            N, H, D = cache_k.shape
            grid = (N,)
            masked_set_kv_buffer_kernel[grid](
                cache_k,
                cache_v,
                self.k_buffer[layer_id - self.start_layer],
                self.v_buffer[layer_id - self.start_layer],
                loc,
                dcp_kv_mask,
                N,
                H,
                D,
                128,
                cache_k.stride(0),
                cache_k.stride(1),
                cache_v.stride(0),
                cache_v.stride(1),
            )
        else:
            """ + old_call.replace("\n        ", "\n            ")
            src = src[:pos] + new_call + src[call_end:]

    # Add the Triton kernel at the end
    kernel_code = '''


@triton.jit
def masked_set_kv_buffer_kernel(
    k_ptr,
    v_ptr,
    k_buffer_ptr,
    v_buffer_ptr,
    loc_ptr,
    mask_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    k_stride_B: tl.constexpr,
    k_stride_H: tl.constexpr,
    v_stride_B: tl.constexpr,
    v_stride_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    do_write = tl.load(mask_ptr + pid) != 0
    if not do_write:
        return
    loc = tl.load(loc_ptr + pid)
    total = H * D
    num_chunks = tl.cdiv(total, CHUNK)
    for c in range(num_chunks):
        offs = tl.arange(0, CHUNK)
        idx = c * CHUNK + offs
        mask = idx < total
        row = idx // D
        col = idx % D
        k_addr = k_ptr + pid * k_stride_B + row * k_stride_H + col
        key = tl.load(k_addr, mask=mask)
        key_buffer_addr = k_buffer_ptr + loc * H * D + idx
        tl.store(key_buffer_addr, key, mask=mask)
        v_addr = v_ptr + pid * v_stride_B + row * v_stride_H + col
        value = tl.load(v_addr, mask=mask)
        value_buffer_addr = v_buffer_ptr + loc * H * D + idx
        tl.store(value_buffer_addr, value, mask=mask)
'''

    # Check if triton imports exist
    if "import triton" not in src:
        # Add triton imports
        src = "import triton\nimport triton.language as tl\n" + src
    elif "import triton.language as tl" not in src:
        src = src.replace("import triton\n", "import triton\nimport triton.language as tl\n")

    src += kernel_code
    write_file(path, src)


def patch_scheduler(sglang_root):
    """Add dcp_size to scheduler."""
    path = os.path.join(sglang_root, "srt/managers/scheduler.py")
    src = read_file(path)

    if "self.dcp_size" in src:
        print(f"  SKIP: {path} (already has dcp_size)")
        return

    # Add dcp_size after tp_size
    src = src.replace(
        "        self.tp_size = server_args.tp_size\n",
        "        self.tp_size = server_args.tp_size\n        self.dcp_size = server_args.dcp_size\n",
    )

    # Multiply max_total_num_tokens in log message
    src = src.replace(
        'f"max_total_num_tokens={self.max_total_num_tokens}, "',
        'f"max_total_num_tokens={self.max_total_num_tokens * self.dcp_size}, "\n                f"{f\'dcp_size={self.dcp_size}, \' if self.dcp_size > 1 else \'\'}"',
    )

    # NOTE: Do NOT multiply page_size in tree cache or PrefillAdder.
    # MambaRadixCache (used by hybrid Qwen3.5) requires page_size=1.
    # The DCP page_size scaling is handled in model_runner_kv_cache_mixin.py instead.

    write_file(path, src)


def patch_scheduler_checker(sglang_root):
    """Adjust available/evictable sizes for DCP."""
    path = os.path.join(sglang_root, "srt/managers/scheduler_runtime_checker_mixin.py")
    src = read_file(path)

    if "dcp_size" in src:
        print(f"  SKIP: {path} (already has dcp_size)")
        return

    old_token_info = "        available_size = self.token_to_kv_pool_allocator.available_size()\n        evictable_size = self.tree_cache.evictable_size()"
    if old_token_info in src:
        new_token_info = "        available_size = (\n            self.token_to_kv_pool_allocator.available_size() // self.dcp_size\n        )\n        evictable_size = self.tree_cache.evictable_size() // self.dcp_size"
        src = src.replace(old_token_info, new_token_info)

    write_file(path, src)


def patch_model_runner(sglang_root):
    """Add DCP to model runner init."""
    path = os.path.join(sglang_root, "srt/model_executor/model_runner.py")
    src = read_file(path)

    if "self.dcp_size" in src:
        print(f"  SKIP: {path} (already has dcp_size)")
        return

    # Add dcp_size/dcp_rank after tp_size
    src = src.replace(
        "        self.tp_size = tp_size\n",
        "        self.tp_size = tp_size\n        self.dcp_size = server_args.dcp_size\n        self.dcp_rank = self.tp_rank % self.dcp_size\n",
    )

    # Pass decode_context_parallel_size to ensure_model_parallel_initialized
    src = src.replace(
        "                expert_model_parallel_size=self.moe_ep_size,\n",
        "                expert_model_parallel_size=self.moe_ep_size,\n                decode_context_parallel_size=self.dcp_size,\n",
    )

    # Pass dcp_size to GraphInputBuffers.create
    src = src.replace(
        "            pp_size=self.server_args.pp_size,\n",
        "            pp_size=self.server_args.pp_size,\n            dcp_size=self.server_args.dcp_size,\n",
    )

    write_file(path, src)


def patch_model_runner_kv_cache(sglang_root):
    """Force PagedTokenToKVPoolAllocator when DCP > 1."""
    path = os.path.join(sglang_root, "srt/model_executor/model_runner_kv_cache_mixin.py")
    src = read_file(path)

    if "self.dcp_size == 1" in src or "dcp_size" in src:
        print(f"  SKIP: {path} (already has dcp_size)")
        return

    # Change page_size == 1 check to also require dcp_size == 1
    src = src.replace(
        "                    if self.page_size == 1:\n",
        "                    if self.page_size == 1 and self.dcp_size == 1:\n",
    )

    # Multiply max_total_num_tokens and page_size by dcp_size
    src = src.replace(
        "                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(\n                            self.max_total_num_tokens,\n                            page_size=self.page_size,",
        "                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(\n                            self.max_total_num_tokens * self.dcp_size,\n                            page_size=self.page_size * self.dcp_size,",
    )

    write_file(path, src)


def patch_cuda_graph_runner(sglang_root):
    """Add DCP mask buffer to CUDA graph runner and DecodeInputBuffers."""
    path = os.path.join(sglang_root, "srt/model_executor/cuda_graph_runner.py")
    src = read_file(path)

    if "self.dcp_size" in src:
        print(f"  SKIP: {path} (already has dcp_size)")
        return

    # Add dcp_size attribute
    src = src.replace(
        "        self.tp_size = model_runner.server_args.tp_size\n",
        "        self.tp_size = model_runner.server_args.tp_size\n        self.dcp_size = model_runner.server_args.dcp_size\n",
    )

    # Add dcp_kv_mask to forward batch in capture
    old_forward_batch = "            global_forward_mode=self.capture_forward_mode,\n            lora_ids=lora_ids,"
    if old_forward_batch in src:
        new_forward_batch = old_forward_batch + "\n            dcp_kv_mask=buffers.dcp_kv_mask[:num_tokens] if self.dcp_size > 1 else None,"
        src = src.replace(old_forward_batch, new_forward_batch)

    # --- Patch DecodeInputBuffers (defined in this file) ---
    # Add dcp_kv_mask field to the class (check field section, not whole file)
    decode_cls_pos = src.find("class DecodeInputBuffers")
    create_pos = src.find("def create(", decode_cls_pos) if decode_cls_pos != -1 else -1
    fields_section = src[decode_cls_pos:create_pos] if decode_cls_pos != -1 and create_pos != -1 else ""
    if "class DecodeInputBuffers" in src and "dcp_kv_mask" not in fields_section:
        # Add field after pp_proxy_tensors
        old_field = "    pp_proxy_tensors: Optional[Dict[str, torch.Tensor]]"
        if old_field in src:
            src = src.replace(old_field, old_field + "\n    dcp_kv_mask: Optional[torch.Tensor]", 1)

        # Add dcp_size param to create()
        old_create = "        ne_token_table: Optional[torch.Tensor] = None,\n    ) -> \"DecodeInputBuffers\":"
        if old_create in src:
            src = src.replace(old_create,
                "        ne_token_table: Optional[torch.Tensor] = None,\n        dcp_size: int = 1,\n    ) -> \"DecodeInputBuffers\":")

        # Add dcp_kv_mask allocation (before ngram_embedding_info)
        old_ngram = "            ngram_embedding_info = ("
        if old_ngram in src:
            src = src.replace(old_ngram,
                "            if dcp_size > 1:\n"
                "                dcp_kv_mask = torch.zeros((max_num_token,), dtype=torch.bool)\n"
                "            else:\n"
                "                dcp_kv_mask = None\n\n"
                "            ngram_embedding_info = (", 1)

        # Add dcp_kv_mask to return
        old_ret = "            pp_proxy_tensors=pp_proxy_tensors,\n            ngram_embedding_info=ngram_embedding_info,"
        if old_ret in src:
            src = src.replace(old_ret,
                "            pp_proxy_tensors=pp_proxy_tensors,\n            dcp_kv_mask=dcp_kv_mask,\n            ngram_embedding_info=ngram_embedding_info,")

        # Add dcp_kv_mask copy in populate_from_forward_batch
        old_batch = "        # Batch all GPU copies, grouped by dtype pair.\n        _grouped_foreach_copy_(dsts, srcs)"
        if old_batch in src:
            src = src.replace(old_batch,
                "        # DCP KV mask\n"
                "        if self.dcp_kv_mask is not None and forward_batch.dcp_kv_mask is not None:\n"
                "            dsts.append(self.dcp_kv_mask[:raw_num_token])\n"
                "            srcs.append(forward_batch.dcp_kv_mask)\n\n"
                "        # Batch all GPU copies, grouped by dtype pair.\n"
                "        _grouped_foreach_copy_(dsts, srcs)")

    # Pass dcp_size to DecodeInputBuffers.create() calls
    # There are two call sites: in CudaGraphRunner.__init__ and in model_runner
    src = src.replace(
        "            pp_size=self.pp_size,\n            is_encoder_decoder",
        "            pp_size=self.pp_size,\n            dcp_size=self.dcp_size,\n            is_encoder_decoder",
    )

    write_file(path, src)


def patch_forward_batch_info(sglang_root):
    """Add dcp_kv_mask field to ForwardBatch."""
    path = os.path.join(sglang_root, "srt/model_executor/forward_batch_info.py")
    src = read_file(path)

    if "dcp_kv_mask" in src:
        print(f"  SKIP: {path} (already has dcp_kv_mask)")
        return

    # Add field
    old_field = "    # For hidden states before normal\n    return_hidden_states_before_norm: bool = False"
    if old_field in src:
        new_field = old_field + "\n\n    # For decode context parallel\n    dcp_kv_mask: Optional[torch.Tensor] = None"
        src = src.replace(old_field, new_field)
    else:
        # Try alternative placement - add before the @classmethod init_new
        src = src.replace(
            "    @classmethod\n    def init_new(",
            "    # For decode context parallel\n    dcp_kv_mask: Optional[torch.Tensor] = None\n\n    @classmethod\n    def init_new(",
        )

    # Add DCP mask computation in init_new
    old_lora = "            model_runner.lora_manager.prepare_lora_batch(ret)"
    if old_lora in src:
        dcp_code = """
        # For DCP
        if model_runner.dcp_size > 1:
            dcp_size = model_runner.dcp_size
            dcp_rank = model_runner.dcp_rank
            ret.dcp_kv_mask = ret.out_cache_loc % dcp_size == dcp_rank
            ret.out_cache_loc = ret.out_cache_loc // dcp_size
"""
        # Insert after lora_manager block
        insert_pos = src.find("\n", src.find(old_lora)) + 1
        src = src[:insert_pos] + dcp_code + src[insert_pos:]

    write_file(path, src)


def patch_input_buffers(sglang_root):
    """No-op: DecodeInputBuffers moved to cuda_graph_runner.py in current SGLang."""
    print(f"  SKIP: input_buffers.py (DecodeInputBuffers is in cuda_graph_runner.py now)")


def patch_models_utils(sglang_root):
    """Disable fused_set_kv_buffer when DCP is active."""
    path = os.path.join(sglang_root, "srt/models/utils.py")
    src = read_file(path)

    if "get_dcp_group" in src:
        print(f"  SKIP: {path} (already has get_dcp_group)")
        return

    # Add import
    src = src.replace(
        "from sglang.srt.environ import envs",
        "from sglang.srt.distributed.parallel_state import get_dcp_group\nfrom sglang.srt.environ import envs",
    )

    # Add DCP check to enable_fused_set_kv_buffer
    old_check = "        and not isinstance(forward_batch.token_to_kv_pool, SWAKVPool)\n    )"
    if old_check in src:
        new_check = "        and not isinstance(forward_batch.token_to_kv_pool, SWAKVPool)\n        and get_dcp_group().world_size == 1\n    )"
        src = src.replace(old_check, new_check)

    write_file(path, src)


def main():
    # Find sglang package root
    sglang_root = None
    candidates = [
        "/opt/sglang/python/sglang",
        "/opt/venv/lib/python3.12/site-packages/sglang",
    ]
    for c in candidates:
        if os.path.isdir(c):
            sglang_root = c
            break

    if sglang_root is None:
        print("ERROR: Cannot find sglang package")
        sys.exit(1)

    print(f"Applying DCP PR #14982 to: {sglang_root}")

    patch_server_args(sglang_root)
    patch_parallel_state(sglang_root)
    patch_pynccl_allocator(sglang_root)
    patch_engine(sglang_root)
    patch_attention_utils(sglang_root)
    patch_flashinfer_backend(sglang_root)
    patch_flashinfer_mla_backend(sglang_root)
    patch_memory_pool(sglang_root)
    patch_scheduler(sglang_root)
    patch_scheduler_checker(sglang_root)
    patch_model_runner(sglang_root)
    patch_model_runner_kv_cache(sglang_root)
    patch_cuda_graph_runner(sglang_root)
    patch_forward_batch_info(sglang_root)
    patch_input_buffers(sglang_root)
    patch_models_utils(sglang_root)

    print("\nDCP patch applied successfully!")
    print("Usage: python -m sglang.launch_server --model-path <model> --tp 8 --dcp 2 --enable-symm-mem")


if __name__ == "__main__":
    main()
