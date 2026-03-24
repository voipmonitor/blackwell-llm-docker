#!/usr/bin/env python3
"""
Allow flashinfer attention backend on Blackwell GPUs for hybrid GDN models.

The upstream assertion only allows triton, trtllm_mha, and fa4 backends.
This patch adds flashinfer to the allowed list.
"""

import sys
from pathlib import Path

targets = [
    "sglang/srt/layers/attention/attention_registry.py",
]

old = (
    '                    or runner.server_args.attention_backend == "fa4"\n'
    '                ), "triton or trtllm_mha or fa4 backend are the only supported backends on Blackwell GPUs for hybrid GDN models, use --attention-backend triton or --attention-backend trtllm_mha to specify the backend."'
)

new = (
    '                    or runner.server_args.attention_backend == "fa4"\n'
    '                    or runner.server_args.attention_backend == "flashinfer"\n'
    '                ), "triton, trtllm_mha, fa4, or flashinfer backend are the only supported backends on Blackwell GPUs for hybrid GDN models, use --attention-backend to specify the backend."'
)

applied = 0
for pattern in targets:
    candidates = [
        Path("/opt/sglang/python") / pattern,
        Path("/sgl-workspace/sglang/python") / pattern,
        *Path("/opt/venv").rglob(pattern),
    ]
    for p in candidates:
        if not p.exists():
            continue
        src = p.read_text()
        if old in src:
            src = src.replace(old, new, 1)
            p.write_text(src)
            applied += 1
            print(f"OK: patched {p}")
        elif 'flashinfer"' in src and "hybrid GDN" in src:
            print(f"Already patched: {p}")
        else:
            print(f"WARNING: pattern not found in {p}")
        break
    else:
        print(f"WARNING: {pattern} not found")

if applied > 0:
    print(f"OK: applied {applied} patch(es)")
else:
    print("No changes needed")
