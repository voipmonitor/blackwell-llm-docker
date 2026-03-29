#!/usr/bin/env python3
"""Fix nvfp4 JIT dual-gencode bug: remove 10.3a from arch list.

On CUDA 13+, _get_nvfp4_cuda_arch_list() appends "10.3a" to match AOT fatbin
behavior. But SGL_CUDA_ARCH is a single compile-time value (e.g. 1200 for SM120),
so when nvcc compiles the SM103 variant, __CUDA_ARCH__=1030 != SGL_CUDA_ARCH=1200
causing a static assertion failure.

Fix: remove the 10.3a append — JIT should only compile for the actual device.
"""
import sys

path = '/opt/sglang/python/sglang/jit_kernel/nvfp4.py'
with open(path) as f:
    src = f.read()

OLD = '''    cuda_major, _cuda_minor = _parse_cuda_version()
    if cuda_major >= 13 and "10.3a" not in archs:
        # Match sgl-kernel AOT fatbin behavior on CUDA 13+ for Blackwell.
        archs.append("10.3a")'''

NEW = '''    # SM103 gencode removed: SGL_CUDA_ARCH is a single value, can not match both arches'''

if OLD in src:
    src = src.replace(OLD, NEW)
    with open(path, 'w') as f:
        f.write(src)
    print('OK: removed 10.3a from nvfp4 arch list')
elif '10.3a' not in src:
    print('OK: 10.3a already not present')
else:
    print(f'ERROR: expected pattern not found in {path}', file=sys.stderr)
    sys.exit(1)
