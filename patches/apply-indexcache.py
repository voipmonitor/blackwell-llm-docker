#!/usr/bin/env python3
"""
Apply IndexCache patch from THUDM/IndexCache to SGLang's DeepSeek MLA attention.

Adds skip_topk / next_skip_topk logic so that top-k indexing can be reused
across layers (controlled by config.index_topk_freq or config.index_topk_pattern).

See: https://github.com/THUDM/IndexCache
"""

import sys
from pathlib import Path

# Find SGLang source directory
candidates = [
    Path("/opt/sglang"),
    Path("/sgl-workspace/sglang"),
]
sglang_dir = None
for c in candidates:
    if (c / "python/sglang/srt/models/deepseek_v2.py").exists():
        sglang_dir = c
        break

if sglang_dir is None:
    print("ERROR: Could not find SGLang source directory")
    sys.exit(1)

base = sglang_dir / "python"
applied = 0

# ============================================================================
# Patch 1: forward_mla.py — add prev_topk_indices param + skip_topk logic
# ============================================================================
fmla = base / "sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py"
if fmla.exists():
    src = fmla.read_text()
    if "skip_topk" in src:
        print(f"Already patched: {fmla}")
    else:
        # 1a: Add prev_topk_indices parameter
        old1a = "        llama_4_scaling: Optional[torch.Tensor] = None,\n    ):"
        new1a = "        llama_4_scaling: Optional[torch.Tensor] = None,\n        prev_topk_indices: Optional[torch.Tensor] = None,\n    ):"
        if old1a in src:
            src = src.replace(old1a, new1a, 1)
        else:
            print(f"WARNING: param insertion pattern not found in {fmla}")

        # 1b: Replace the topk_indices block in the overlap branch (with alt_stream)
        old1b = """\
                topk_indices = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    topk_indices = self.indexer("""
        new1b = """\
                if not self.skip_topk:
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    topk_indices = prev_topk_indices
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    if not self.skip_topk:
                        topk_indices = self.indexer("""
        if old1b in src:
            src = src.replace(old1b, new1b, 1)
        else:
            print(f"WARNING: topk_indices block pattern not found in {fmla}")

        # 1c: Add else branch after the non-overlap indexer call
        old1c = """\
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
        else:
            q = self.q_proj(hidden_states)[0].view("""
        new1c = """\
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                    else:
                        topk_indices = prev_topk_indices
        else:
            q = self.q_proj(hidden_states)[0].view("""
        if old1c in src:
            src = src.replace(old1c, new1c, 1)
        else:
            print(f"WARNING: else branch pattern not found in {fmla}")

        # 1d: Change return value to include topk_indices
        old1d = "        output, _ = self.o_proj(attn_bmm_output)\n\n        return output"
        new1d = "        output, _ = self.o_proj(attn_bmm_output)\n\n        if not self.next_skip_topk:\n            return output, None\n        else:\n            return output, topk_indices"
        if old1d in src:
            src = src.replace(old1d, new1d, 1)
        else:
            print(f"WARNING: return pattern not found in {fmla}")

        fmla.write_text(src)
        applied += 1
        print(f"OK: patched {fmla}")
else:
    print(f"WARNING: {fmla} not found")

# ============================================================================
# Patch 2: deepseek_nextn.py — unpack topk_indices from decoder
# ============================================================================
nextn = base / "sglang/srt/models/deepseek_nextn.py"
if nextn.exists():
    src = nextn.read_text()
    old2 = "            hidden_states, residual = self.decoder("
    new2 = "            hidden_states, residual, topk_indices = self.decoder("
    if old2 in src and new2 not in src:
        src = src.replace(old2, new2, 1)
        nextn.write_text(src)
        applied += 1
        print(f"OK: patched {nextn}")
    elif new2 in src:
        print(f"Already patched: {nextn}")
    else:
        print(f"WARNING: pattern not found in {nextn}")
else:
    print(f"WARNING: {nextn} not found")

# ============================================================================
# Patch 3: deepseek_v2.py — skip_topk config, prev_topk_indices plumbing
# ============================================================================
dv2 = base / "sglang/srt/models/deepseek_v2.py"
if dv2.exists():
    src = dv2.read_text()
    if "skip_topk" in src and "index_topk_freq" in src:
        print(f"Already patched: {dv2}")
    else:
        # 3a: Add is_nextn param to __init__
        old3a = '        skip_rope: bool = False,\n    ) -> None:'
        new3a = '        skip_rope: bool = False,\n        is_nextn: bool = False,\n    ) -> None:'
        if old3a in src:
            src = src.replace(old3a, new3a, 1)

        # 3b: Add skip_topk/next_skip_topk config after indexer init
        old3b = """\
                layer_id=layer_id,
                alt_stream=alt_stream,
            )

        self.kv_b_proj"""
        new3b = """\
                layer_id=layer_id,
                alt_stream=alt_stream,
            )
            if is_nextn:
                self.skip_topk = False
                self.next_skip_topk = False
            else:
                self.index_topk_freq = getattr(config, "index_topk_freq", 1)
                self.index_topk_pattern = getattr(config, "index_topk_pattern", None)
                if self.index_topk_pattern is None:
                    self.skip_topk = (max(layer_id-1, 0) % self.index_topk_freq != 0)
                    self.next_skip_topk = (layer_id % self.index_topk_freq != 0)
                else:
                    self.skip_topk = self.index_topk_pattern[layer_id] == 'S'
                    if layer_id < len(self.index_topk_pattern) - 1:
                        self.next_skip_topk = self.index_topk_pattern[layer_id+1] == 'S'
                    else:
                        self.next_skip_topk = False
            print('layer_id {} DSA skip_topk {} next_skip_topk {} is_nextn {}'.format(layer_id, self.skip_topk, self.next_skip_topk, is_nextn))

        self.kv_b_proj"""
        if old3b in src:
            src = src.replace(old3b, new3b, 1)

        # 3c: Add prev_topk_indices to forward methods (multiple places)
        # Forward method 1
        old3c1 = "        llama_4_scaling: Optional[torch.Tensor] = None,\n    ):\n        s = self.forward_prepare("
        new3c1 = "        llama_4_scaling: Optional[torch.Tensor] = None,\n        prev_topk_indices: Optional[torch.Tensor] = None,\n    ):\n        s = self.forward_prepare("
        if old3c1 in src:
            src = src.replace(old3c1, new3c1, 1)

        # Add prev_topk_indices to forward_prepare call
        old3c2 = "            llama_4_scaling=llama_4_scaling,\n        )\n        return self.forward_core(s)"
        new3c2 = "            llama_4_scaling=llama_4_scaling,\n            prev_topk_indices=prev_topk_indices,\n        )\n        return self.forward_core(s)"
        if old3c2 in src:
            src = src.replace(old3c2, new3c2, 1)

        # Forward method 2 (forward_absorb)
        old3c3 = "        llama_4_scaling: Optional[torch.Tensor] = None,\n    ):\n        if self.attn_mha.kv_b_proj is None:"
        new3c3 = "        llama_4_scaling: Optional[torch.Tensor] = None,\n        prev_topk_indices: Optional[torch.Tensor] = None,\n    ):\n        if self.attn_mha.kv_b_proj is None:"
        if old3c3 in src:
            src = src.replace(old3c3, new3c3, 1)

        # Pass prev_topk_indices to forward_absorb_prepare
        old3c4 = "                positions, hidden_states, forward_batch, zero_allocator, llama_4_scaling\n            )"
        new3c4 = "                positions, hidden_states, forward_batch, zero_allocator, llama_4_scaling, prev_topk_indices\n            )"
        if old3c4 in src:
            src = src.replace(old3c4, new3c4, 1)

        # 3d: Add is_nextn to DeepseekV2DecoderLayer attn init
        old3d = '            reduce_results=False,\n            prefix=add_prefix("self_attn", prefix),\n            alt_stream=alt_stream,\n        )'
        new3d = '            reduce_results=False,\n            prefix=add_prefix("self_attn", prefix),\n            alt_stream=alt_stream,\n            is_nextn=is_nextn,\n        )'
        if old3d in src:
            src = src.replace(old3d, new3d, 1)

        # 3e: Add prev_topk_indices to DecoderLayer.forward
        old3e = "        llama_4_scaling: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:"
        new3e = "        llama_4_scaling: Optional[torch.Tensor] = None,\n        prev_topk_indices: Optional[torch.Tensor] = None,\n    ) -> torch.Tensor:"
        if old3e in src:
            src = src.replace(old3e, new3e, 1)

        # Pass prev_topk_indices to self_attn.forward
        old3e2 = "            llama_4_scaling=llama_4_scaling,\n            layer_scatter_modes=self.layer_scatter_modes,\n        )"
        new3e2 = "            llama_4_scaling=llama_4_scaling,\n            layer_scatter_modes=self.layer_scatter_modes,\n            prev_topk_indices=prev_topk_indices,\n        )"
        if old3e2 in src:
            src = src.replace(old3e2, new3e2, 1)

        # Unpack topk_indices from attention output
        old3e3 = "        hidden_states, residual = self.layer_communicator.prepare_mlp("
        new3e3 = "        if isinstance(hidden_states, tuple):\n            hidden_states, topk_indices = hidden_states\n        else:\n            topk_indices = None\n\n        hidden_states, residual = self.layer_communicator.prepare_mlp("
        if old3e3 in src:
            src = src.replace(old3e3, new3e3, 1)

        # Change return to include topk_indices
        old3e4 = "        return hidden_states, residual\n\n    def op_comm_prepare_attn("
        new3e4 = "        return hidden_states, residual, topk_indices\n\n    def op_comm_prepare_attn("
        if old3e4 in src:
            src = src.replace(old3e4, new3e4, 1)

        # 3f: Model forward — propagate topk_indices through layers
        old3f1 = "        aux_hidden_states = []\n        for i in range(normal_start_layer, normal_end_layer):"
        new3f1 = "        aux_hidden_states = []\n        topk_indices = None\n        for i in range(normal_start_layer, normal_end_layer):"
        if old3f1 in src:
            src = src.replace(old3f1, new3f1, 1)

        old3f2 = "                hidden_states, residual = layer("
        new3f2 = "                hidden_states, residual, topk_indices = layer("
        if old3f2 in src:
            src = src.replace(old3f2, new3f2, 1)

        # Add prev_topk_indices to layer call
        old3f3 = "                    gemm_output_zero_allocator,\n                    llama_4_scaling,\n                )"
        new3f3 = "                    gemm_output_zero_allocator,\n                    llama_4_scaling,\n                    prev_topk_indices=topk_indices,\n                )"
        if old3f3 in src:
            src = src.replace(old3f3, new3f3, 1)

        dv2.write_text(src)
        applied += 1
        print(f"OK: patched {dv2}")
else:
    print(f"WARNING: {dv2} not found")

if applied > 0:
    print(f"OK: applied IndexCache to {applied} file(s)")
else:
    print("No changes needed")
