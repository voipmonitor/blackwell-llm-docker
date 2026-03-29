#!/usr/bin/env python3
"""Fix merge conflicts in modelopt_quant.py from b12x + PR #21601 (FP4 KV cache).

Two conflicts:
1. fp4_gemm(): b12x check needs to come before cutlass scale fix
2. apply_weights(): condition for weight transpose needs both b12x and cutlass checks
"""
import re
import sys

path = '/opt/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py'
with open(path) as f:
    src = f.read()

if '<<<<<<< HEAD' not in src:
    print(f'OK: no conflicts in {path}')
    sys.exit(0)

# Conflict 1: fp4_gemm function — b12x check before cutlass
# HEAD has: cutlass scale dtype fix
# Incoming has: b12x dispatch
# Merged: b12x first, then cutlass with scale fix
CONFLICT1_PATTERN = r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n(\s+elif enable_flashinfer_fp4_gemm:)'
m1 = re.search(CONFLICT1_PATTERN, src, re.DOTALL)
if m1:
    head = m1.group(1)  # cutlass with scale fix
    incoming = m1.group(2)  # b12x dispatch
    after = m1.group(3)  # elif enable_flashinfer_fp4_gemm

    # Build merged version: b12x first, then cutlass with scale fix as elif
    merged = incoming.rstrip() + '\n'
    # Convert HEAD's "if" to "elif"
    head_as_elif = head.replace('if fp4_backend.is_cutlass()', 'elif fp4_backend.is_cutlass()', 1)
    merged += head_as_elif
    merged += after

    src = src[:m1.start()] + merged + src[m1.end():]
    print('OK: resolved conflict 1 (b12x before cutlass in fp4_gemm)')
else:
    # Try generic resolution
    print('WARNING: conflict 1 pattern not matched, trying generic resolution')

# Conflict 2: apply_weights — condition for weight transpose
# HEAD: enable_flashinfer_fp4_gemm and not is_cutlass()
# Incoming: not is_b12x() and enable_flashinfer_fp4_gemm
# Merged: not is_b12x() and enable_flashinfer_fp4_gemm and not is_cutlass()
CONFLICT2_PATTERN = r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n'
remaining = re.findall(CONFLICT2_PATTERN, src, re.DOTALL)
if remaining:
    for head, incoming in remaining:
        # This should be the weight transpose condition
        if 'is_b12x' in incoming and 'is_cutlass' in head:
            # Merge both conditions
            merged = '        if not fp4_backend.is_b12x() and enable_flashinfer_fp4_gemm and not get_fp4_gemm_runner_backend().is_cutlass():\n'
            src = re.sub(
                r'<<<<<<< HEAD\n.*?=======\n.*?>>>>>>> [^\n]+\n',
                merged, src, count=1, flags=re.DOTALL
            )
            print('OK: resolved conflict 2 (b12x + cutlass condition in apply_weights)')
        else:
            # Generic: keep both
            src = re.sub(
                r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n',
                lambda m: m.group(1) + m.group(2),
                src, count=1, flags=re.DOTALL
            )
            print('OK: resolved remaining conflict (kept both)')

# Verify no markers remain
if '<<<<<<< HEAD' in src:
    print(f'ERROR: unresolved conflict markers in {path}', file=sys.stderr)
    for i, line in enumerate(src.split('\n'), 1):
        if '<<<<<<<' in line or '>>>>>>>' in line:
            print(f'  line {i}: {line.strip()}', file=sys.stderr)
    sys.exit(1)

with open(path, 'w') as f:
    f.write(src)

# Verify syntax
try:
    compile(src, path, 'exec')
    print(f'OK: {path} is syntactically valid')
except SyntaxError as e:
    print(f'ERROR: {path} syntax error: {e}', file=sys.stderr)
    if e.lineno:
        lines = src.split('\n')
        for j in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
            print(f'  {j+1}: {lines[j]}', file=sys.stderr)
    sys.exit(1)
