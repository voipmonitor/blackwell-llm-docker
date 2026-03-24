#!/usr/bin/env python3
"""Apply DCP to SGLang MLA attention backend."""

import os
import sys


def patch():
    path = '/opt/sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py'
    with open(path) as f:
        src = f.read()

    if 'dcp_size' in src:
        print('SKIP: already patched')
        return

    # 1. Add imports
    src = src.replace(
        "from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton",
        "from sglang.srt.distributed.device_communicators.pynccl_allocator import (\n"
        "    use_symmetric_memory,\n"
        ")\n"
        "from sglang.srt.distributed.parallel_state import get_dcp_group\n"
        "from sglang.srt.layers.attention.utils import (\n"
        "    cp_lse_ag_out_rs,\n"
        "    create_flashinfer_kv_indices_for_dcp_triton,\n"
        "    create_flashinfer_kv_indices_triton,\n"
        ")",
    )
    print('1. Added imports')

    # 2. Add dcp_size/dcp_rank to __init__
    src = src.replace(
        "        self.page_size = model_runner.page_size",
        "        self.page_size = model_runner.page_size\n"
        "\n"
        "        self.dcp_size = get_dcp_group().world_size\n"
        "        self.dcp_rank = get_dcp_group().rank_in_group",
    )
    print('2. Added dcp_size/dcp_rank')

    # 3. Modify forward_decode - add DCP path before existing code
    old_decode = (
        "        o = q_nope.new_empty(q_nope.shape)\n"
        "        # Direct call to run without the wrapper\n"
        "        o = decode_wrapper.run(\n"
        "            q_nope,\n"
        "            q_rope,\n"
        "            k_buffer[:, :, : layer.v_head_dim],\n"
        "            k_buffer[:, :, layer.v_head_dim :],\n"
        "            out=o,\n"
        "        )\n"
        "\n"
        "        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)"
    )

    new_decode = (
        "        if self.dcp_size > 1:\n"
        "            # DCP: All-gather Q, partial attention with local KV, merge with LSE\n"
        "            q_nope_ag = get_dcp_group().all_gather(q_nope.contiguous(), dim=1)\n"
        "            rope_dim = layer.head_dim - layer.v_head_dim\n"
        "            q_rope_ag = get_dcp_group().all_gather(q_rope.contiguous(), dim=1)\n"
        "            # MLA: NO interleaving - all Q heads attend to same KV latent\n"
        "\n"
        "            o = q_nope_ag.new_empty(q_nope_ag.shape)\n"
        "            o, s = decode_wrapper.run(\n"
        "                q_nope_ag,\n"
        "                q_rope_ag,\n"
        "                k_buffer[:, :, : layer.v_head_dim],\n"
        "                k_buffer[:, :, layer.v_head_dim :],\n"
        "                out=o,\n"
        "                return_lse=True,\n"
        "            )\n"
        "            o = cp_lse_ag_out_rs(o, s, get_dcp_group())\n"
        "            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)\n"
        "\n"
        "        o = q_nope.new_empty(q_nope.shape)\n"
        "        # Direct call to run without the wrapper\n"
        "        o = decode_wrapper.run(\n"
        "            q_nope,\n"
        "            q_rope,\n"
        "            k_buffer[:, :, : layer.v_head_dim],\n"
        "            k_buffer[:, :, layer.v_head_dim :],\n"
        "            out=o,\n"
        "        )\n"
        "\n"
        "        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)"
    )

    assert old_decode in src, "forward_decode pattern not found"
    src = src.replace(old_decode, new_decode)
    print('3. Modified forward_decode for DCP')

    # 4. DCP masked KV write
    old_kv = (
        "                if k_rope is not None:\n"
        "                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(\n"
        "                        layer,\n"
        "                        cache_loc,\n"
        "                        k,\n"
        "                        k_rope,\n"
        "                    )\n"
        "                else:\n"
        "                    forward_batch.token_to_kv_pool.set_kv_buffer(\n"
        "                        layer,\n"
        "                        cache_loc,\n"
        "                        k,\n"
        "                        v,\n"
        "                    )"
    )

    new_kv = (
        "                if k_rope is not None:\n"
        "                    if self.dcp_size > 1 and forward_batch.dcp_kv_mask is not None:\n"
        "                        mask = forward_batch.dcp_kv_mask\n"
        "                        forward_batch.token_to_kv_pool.set_mla_kv_buffer(\n"
        "                            layer, cache_loc[mask], k[mask], k_rope[mask],\n"
        "                        )\n"
        "                    else:\n"
        "                        forward_batch.token_to_kv_pool.set_mla_kv_buffer(\n"
        "                            layer,\n"
        "                            cache_loc,\n"
        "                            k,\n"
        "                            k_rope,\n"
        "                        )\n"
        "                else:\n"
        "                    if self.dcp_size > 1 and forward_batch.dcp_kv_mask is not None:\n"
        "                        mask = forward_batch.dcp_kv_mask\n"
        "                        forward_batch.token_to_kv_pool.set_kv_buffer(\n"
        "                            layer, cache_loc[mask], k[mask], v[mask],\n"
        "                        )\n"
        "                    else:\n"
        "                        forward_batch.token_to_kv_pool.set_kv_buffer(\n"
        "                            layer,\n"
        "                            cache_loc,\n"
        "                            k,\n"
        "                            v,\n"
        "                        )"
    )

    assert old_kv in src, "KV write pattern not found"
    src = src.replace(old_kv, new_kv)
    print('4. Added DCP masked KV write')

    # 5a. Decode indices updater - add dcp to update()
    old_upd = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "        **fast_decode_kwargs,\n"
        "    ):\n"
        "        decode_wrapper = decode_wrapper or self.decode_wrapper\n"
        "        self.call_begin_forward(\n"
        "            decode_wrapper,\n"
        "            req_pool_indices,\n"
        "            seq_lens,\n"
        "            seq_lens_sum,\n"
        "            self.q_indptr,\n"
        "            self.kv_indptr,\n"
        "            init_metadata_replay,\n"
        "            spec_info,\n"
        "            **fast_decode_kwargs,\n"
        "        )"
    )

    new_upd = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "        dcp_size: Optional[int] = None,\n"
        "        dcp_rank: Optional[int] = None,\n"
        "        **fast_decode_kwargs,\n"
        "    ):\n"
        "        decode_wrapper = decode_wrapper or self.decode_wrapper\n"
        "        self.call_begin_forward(\n"
        "            decode_wrapper,\n"
        "            req_pool_indices,\n"
        "            seq_lens,\n"
        "            seq_lens_sum,\n"
        "            self.q_indptr,\n"
        "            self.kv_indptr,\n"
        "            init_metadata_replay,\n"
        "            spec_info,\n"
        "            dcp_size=dcp_size,\n"
        "            dcp_rank=dcp_rank,\n"
        "            **fast_decode_kwargs,\n"
        "        )"
    )

    assert old_upd in src, "update() pattern not found"
    src = src.replace(old_upd, new_upd)
    print('5a. Added dcp to update()')

    # 5b. call_begin_forward - add dcp params and modify indices
    old_cbf = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "        **fast_decode_kwargs,\n"
        "    ):\n"
        "        bs = len(req_pool_indices)\n"
        "        q_indptr = q_indptr[: bs + 1]\n"
        "        kv_lens = paged_kernel_lens.to(torch.int32)\n"
        "        sm_scale = self.scaling\n"
        "        if spec_info is None:\n"
        "            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)\n"
        "            kv_indptr = kv_indptr[: bs + 1]\n"
        "            kv_indices = (\n"
        "                torch.empty(paged_kernel_lens_sum, dtype=torch.int32, device=\"cuda\")\n"
        "                if not init_metadata_replay\n"
        "                else fast_decode_kwargs[\"kv_indices\"]\n"
        "            )\n"
        "            create_flashinfer_kv_indices_triton[(bs,)](\n"
        "                self.req_to_token,\n"
        "                req_pool_indices,\n"
        "                paged_kernel_lens,\n"
        "                kv_indptr,\n"
        "                None,\n"
        "                kv_indices,\n"
        "                self.req_to_token.shape[1],\n"
        "            )"
    )

    new_cbf = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "        dcp_size: Optional[int] = None,\n"
        "        dcp_rank: Optional[int] = None,\n"
        "        **fast_decode_kwargs,\n"
        "    ):\n"
        "        if dcp_size is None:\n"
        "            dcp_size = 1\n"
        "            dcp_rank = 0\n"
        "        bs = len(req_pool_indices)\n"
        "        q_indptr = q_indptr[: bs + 1]\n"
        "        kv_lens = paged_kernel_lens.to(torch.int32)\n"
        "        sm_scale = self.scaling\n"
        "        if spec_info is None:\n"
        "            if dcp_size > 1:\n"
        "                paged_kernel_lens = paged_kernel_lens // dcp_size + (\n"
        "                    dcp_rank < paged_kernel_lens % dcp_size\n"
        "                )\n"
        "                paged_kernel_lens_sum = paged_kernel_lens.sum().item()\n"
        "            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)\n"
        "            kv_indptr = kv_indptr[: bs + 1]\n"
        "            kv_indices = (\n"
        "                torch.empty(paged_kernel_lens_sum, dtype=torch.int32, device=\"cuda\")\n"
        "                if not init_metadata_replay\n"
        "                else fast_decode_kwargs[\"kv_indices\"]\n"
        "            )\n"
        "            create_flashinfer_kv_indices_for_dcp_triton[(bs,)](\n"
        "                self.req_to_token,\n"
        "                req_pool_indices,\n"
        "                paged_kernel_lens,\n"
        "                kv_indptr,\n"
        "                None,\n"
        "                kv_indices,\n"
        "                self.req_to_token.shape[1],\n"
        "                dcp_size,\n"
        "                dcp_rank,\n"
        "            )"
    )

    assert old_cbf in src, "call_begin_forward pattern not found"
    src = src.replace(old_cbf, new_cbf)
    print('5b. Modified call_begin_forward for DCP')

    # 5c. num_local_heads * dcp_size in plan()
    old_plan = (
        "                self.num_local_heads,\n"
        "                self.kv_lora_rank,\n"
        "                self.qk_rope_head_dim,\n"
        "                1,\n"
        "                False,\n"
        "                sm_scale,\n"
        "                self.data_type,\n"
        "                self.data_type,\n"
        "            )\n"
        "        else:\n"
        "            wrapper.plan(\n"
        "                fast_decode_kwargs[\"qo_indptr_cpu\"],\n"
        "                fast_decode_kwargs[\"kv_indptr_cpu\"],\n"
        "                kv_indices,\n"
        "                fast_decode_kwargs[\"kv_len_arr_cpu\"],\n"
        "                self.num_local_heads,"
    )

    new_plan = (
        "                self.num_local_heads * dcp_size,\n"
        "                self.kv_lora_rank,\n"
        "                self.qk_rope_head_dim,\n"
        "                1,\n"
        "                False,\n"
        "                sm_scale,\n"
        "                self.data_type,\n"
        "                self.data_type,\n"
        "            )\n"
        "        else:\n"
        "            wrapper.plan(\n"
        "                fast_decode_kwargs[\"qo_indptr_cpu\"],\n"
        "                fast_decode_kwargs[\"kv_indptr_cpu\"],\n"
        "                kv_indices,\n"
        "                fast_decode_kwargs[\"kv_len_arr_cpu\"],\n"
        "                self.num_local_heads * dcp_size,"
    )

    assert old_plan in src, "plan() pattern not found"
    src = src.replace(old_plan, new_plan)
    print('5c. Multiplied num_local_heads by dcp_size')

    # 6. Pass dcp params from init_forward_metadata
    old_call = "self.indices_updater_decode.update("
    idx = src.find(old_call)
    if idx > 0:
        end = src.find(")", idx) + 1
        call = src[idx:end]
        if "dcp_size" not in call:
            new_call = call.rstrip(")") + "\n                dcp_size=self.dcp_size,\n                dcp_rank=self.dcp_rank,\n            )"
            src = src.replace(call, new_call)
            print('6. Added dcp to update() call site')

    with open(path, 'w') as f:
        f.write(src)
    print('\nMLA DCP patch complete!')


if __name__ == '__main__':
    patch()
