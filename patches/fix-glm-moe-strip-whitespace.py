#!/usr/bin/env python3
"""
Patch SGLang GLM MoE detectors to not strip significant whitespace from
tool call argument values.

Both Glm4MoeDetector._parse_argument_pairs and Glm47MoeDetector._parse_argument_pairs
call arg_value.strip() which removes leading/trailing whitespace from string arguments.
This is lossy when whitespace is significant (e.g. source code with indentation).

See: https://github.com/sgl-project/sglang/issues/20542
"""

import sys
from pathlib import Path

targets = [
    "sglang/srt/function_call/glm4_moe_detector.py",
    "sglang/srt/function_call/glm47_moe_detector.py",
]

old = "            arg_value = arg_value.strip()\n"
new = ""

applied = 0
for pattern in targets:
    candidates = [
        Path("/opt/sglang/python") / pattern,
        *Path("/opt/venv").rglob(pattern),
        *Path("/usr").rglob(pattern),
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
        elif "arg_value.strip()" not in src:
            print(f"Already patched or pattern differs: {p}")
        break
    else:
        print(f"WARNING: {pattern} not found")

if applied > 0:
    print(f"OK: applied {applied} patch(es)")
else:
    print("No changes needed")
