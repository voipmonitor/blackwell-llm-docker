#!/usr/bin/env python3
"""
Patch torch._inductor triton autotuner to skip configs exceeding SM120 shared memory.

SM120 (RTX PRO 6000 Blackwell) has 99 KB (101376 bytes) max shared memory per block.
torch._inductor's autotuner tries configs requiring up to 120 KB, which causes
CUDA illegal memory access errors that corrupt the entire CUDA context and crash
CUDA graph capture.

Fix: add a shared memory pre-check in CachingAutotuner.bench() that skips configs
exceeding the device's max shared memory BEFORE launching the kernel.
"""

import ast
import sys


def patch():
    try:
        import torch._inductor.runtime.triton_heuristics as mod
    except ImportError:
        print("SKIP: torch._inductor.runtime.triton_heuristics not found")
        return

    path = mod.__file__
    with open(path) as f:
        source = f.read()

    # Check if already patched
    if "SM120_SMEM_GUARD" in source:
        print("OK: already patched (SM120_SMEM_GUARD)")
        return

    # Find the bench() method and add shared memory check after the spill check
    marker = """self.benchmark_failure_reasons[launcher] = (
                BenchmarkFailureReason.REGISTER_SPILLING
            )
            return float("inf")"""

    if marker not in source:
        print("WARN: could not find spill check marker in bench(), skipping patch")
        return

    patch_code = """self.benchmark_failure_reasons[launcher] = (
                BenchmarkFailureReason.REGISTER_SPILLING
            )
            return float("inf")

        # SM120_SMEM_GUARD: skip configs that exceed device shared memory limit
        # to prevent CUDA illegal memory access errors during autotuning.
        # SM120 (RTX PRO 6000) has only 99 KB vs SM100's 228 KB.
        if launcher.shared is not None and hasattr(self, 'device_props'):
            try:
                _smem_limit = getattr(self, '_smem_limit', None)
                if _smem_limit is None:
                    dev = self.device_props
                    if hasattr(dev, 'index'):
                        _smem_limit = torch.cuda.get_device_properties(dev.index).shared_memory_per_block_optin
                    if not _smem_limit:
                        # Fallback: query via ctypes
                        import ctypes
                        _lib = ctypes.CDLL('libcuda.so')
                        _val = ctypes.c_int()
                        _lib.cuDeviceGetAttribute(ctypes.byref(_val), 97, 0)
                        _smem_limit = _val.value if _val.value > 0 else 0
                    self._smem_limit = _smem_limit
                if _smem_limit > 0 and launcher.shared > _smem_limit:
                    log.debug(
                        "Skip config %s: shared memory %d exceeds device limit %d",
                        launcher.config, launcher.shared, _smem_limit,
                    )
                    self.benchmark_failure_reasons[launcher] = (
                        BenchmarkFailureReason.INVALID_CONFIG
                    )
                    return float("inf")
            except Exception:
                pass  # If we can't check, let it proceed normally"""

    source = source.replace(marker, patch_code)

    with open(path, "w") as f:
        f.write(source)

    # Clear bytecache
    import pathlib
    cache_dir = pathlib.Path(path).parent / "__pycache__"
    if cache_dir.exists():
        for p in cache_dir.glob("triton_heuristics*"):
            p.unlink()

    print(f"OK: patched {path} (SM120_SMEM_GUARD)")


if __name__ == "__main__":
    patch()
