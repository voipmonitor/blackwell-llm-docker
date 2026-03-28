#!/usr/bin/env python3
"""PR #19963 adapted: Fix CUDA arch suffix for Blackwell/Hopper JIT.

Original PR #19963 targeted _get_cuda_arch_list() which was refactored in main
to _init_jit_cuda_arch_once() using ArchInfo dataclass. This patch applies the
same fix (appending "a" suffix for CC >= 9.0) to the new code structure.

Without this fix, FlashInfer/TVM JIT compiles for "12.0" instead of "12.0a"
on Blackwell SM120, missing architecture-specific instructions.
"""
import sys

path = '/opt/sglang/python/sglang/jit_kernel/utils.py'
with open(path) as f:
    src = f.read()

OLD = '_CUDA_ARCH = ArchInfo(major, minor, "")'
NEW = '_CUDA_ARCH = ArchInfo(major, minor, "a" if major >= 9 else "")'

if OLD in src:
    src = src.replace(OLD, NEW)
    with open(path, 'w') as f:
        f.write(src)
    print(f'OK: applied arch suffix fix — "a" for CC >= 9.0 (Hopper/Blackwell)')
elif NEW in src:
    print('OK: arch suffix fix already applied')
else:
    print(f'ERROR: expected pattern not found in {path}', file=sys.stderr)
    print(f'  Looking for: {OLD}', file=sys.stderr)
    sys.exit(1)
