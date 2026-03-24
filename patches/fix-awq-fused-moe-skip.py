#!/usr/bin/env python3
"""
Patch SGLang awq.py to respect modules_to_not_convert for FusedMoE layers.

Without this patch, AWQ checkpoints with modules_to_not_convert (like
QuantTrio/Qwen3.5-397B-A17B-AWQ which keeps layer 0 in BF16) fail with:
  KeyError: 'model.layers.0.mlp.experts.w13_weight'

The FusedMoE branch in get_quant_method() was missing the is_layer_skipped_awq()
check that LinearBase already had.

Upstream PR: https://github.com/sgl-project/sglang/pull/20439
"""

import sys
from pathlib import Path

# Find awq.py
candidates = [
    Path("/opt/sglang/python/sglang/srt/layers/quantization/awq.py"),
    *Path("/usr").rglob("sglang/srt/layers/quantization/awq.py"),
]
awq_path = None
for p in candidates:
    if p.exists():
        awq_path = p
        break

if awq_path is None:
    print("ERROR: Could not find sglang awq.py")
    sys.exit(1)

src = awq_path.read_text()

# Patch 1: AWQConfig.get_quant_method — NPU FusedMoE branch (line ~181)
old1 = """\
            elif isinstance(layer, FusedMoE):
                return AWQMoEAscendMethod(self)"""
new1 = """\
            elif isinstance(layer, FusedMoE):
                if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                    return None
                return AWQMoEAscendMethod(self)"""

# Patch 2: AWQMarlinConfig.get_quant_method — FusedMoE branch (line ~324)
old2 = """\
        elif isinstance(layer, FusedMoE):
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config"""
new2 = """\
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return None
            from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config"""

applied = 0
if old1 in src and new1 not in src:
    src = src.replace(old1, new1, 1)
    applied += 1
elif new1 in src:
    print("Patch 1 (AWQConfig/NPU FusedMoE): already applied")
else:
    print("WARNING: Patch 1 pattern not found — SGLang version may differ")

if old2 in src and new2 not in src:
    src = src.replace(old2, new2, 1)
    applied += 1
elif new2 in src:
    print("Patch 2 (AWQMarlinConfig FusedMoE): already applied")
else:
    print("WARNING: Patch 2 pattern not found — SGLang version may differ")

if applied > 0:
    awq_path.write_text(src)
    print(f"OK: applied {applied} patch(es) to {awq_path}")
else:
    print(f"No changes needed in {awq_path}")
