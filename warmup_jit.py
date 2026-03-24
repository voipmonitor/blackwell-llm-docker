"""Minimal JIT warmup: seeds Triton + torch caches on first container start."""

import torch
import triton
import triton.language as tl


@triton.jit
def _warmup_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.program_id(0) * 128 + tl.arange(0, 128)
    mask = idx < n
    x = tl.load(x_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, x * 2, mask=mask)


def main():
    print("Warming up torch.compile and Triton kernels...")
    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    out = torch.empty_like(x)
    _warmup_kernel[(n // 128,)](x, out, n)
    torch.cuda.synchronize()
    print("Basic Triton warmup done.")


if __name__ == "__main__":
    main()
