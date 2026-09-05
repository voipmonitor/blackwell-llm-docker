#!/usr/bin/env python3
"""Verify the DeepSeek Infernal Invocation CUDA 13.3 runtime contract."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import os
from pathlib import Path
import select
import tempfile
import time

import torch


CUDA_RUNTIME_WHEEL_PREFIXES = (
    "nvidia-cublas-",
    "nvidia-cuda-cupti-",
    "nvidia-cuda-nvrtc-",
    "nvidia-cuda-runtime-",
    "nvidia-cudnn-",
    "nvidia-cufft-",
    "nvidia-curand-",
    "nvidia-cusolver-",
    "nvidia-cusparse-",
    "nvidia-nccl-",
    "nvidia-nvjitlink-",
)
ALLOWED_CUDA_TOOLING_WHEELS = {"nvidia-cudnn-frontend"}


def _verify_lmcache_atomic_publication() -> None:
    """Require duplicate native stores to publish one complete cache object."""
    from lmcache import lmcache_fs

    with tempfile.TemporaryDirectory() as directory:
        client = lmcache_fs.LMCacheFSClient(directory, 2)
        try:
            payload_size = 8 * 1024 * 1024
            first = bytearray(b"a" * payload_size)
            second = bytearray(b"b" * payload_size)
            key = "model@00000000@runtime-atomic"
            future_id = client.submit_batch_set(
                [key, key],
                [memoryview(first), memoryview(second)],
            )

            poller = select.poll()
            poller.register(client.event_fd(), select.POLLIN)
            deadline = time.monotonic() + 10
            completions = []
            while time.monotonic() < deadline and not completions:
                if poller.poll(50):
                    completions = client.drain_completions()

            assert len(completions) == 1
            completed_id, ok, error, results = completions[0]
            assert completed_id == future_id
            assert ok, error
            if results is not None:
                assert results == [True, True]

            cache_path = Path(directory)
            published = (
                cache_path / "model@0x00000000@runtime-atomic.data"
            ).read_bytes()
            assert published in (first, second)
            assert list(cache_path.glob("*.tmp.*")) == []
        finally:
            client.close()


def _mapped_libraries(fragment: str) -> set[Path]:
    libraries: set[Path] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            path = line.rsplit(maxsplit=1)[-1]
            if path.startswith("/") and fragment in path:
                libraries.add(Path(path))
    return libraries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--flashinfer-version", required=True)
    parser.add_argument("--lmcache-version", required=True)
    parser.add_argument("--instanttensor-version", default="0.1.9")
    parser.add_argument("--nccl4py-version", default="0.3.1")
    parser.add_argument("--cutlass-dsl-version", default="4.6.2")
    parser.add_argument("--xgrammar-version", default="0.2.5")
    parser.add_argument("--vllm-source-root", default="/opt/infernal-invocation/vllm")
    parser.add_argument("--b12x-source-root", default="/opt/infernal-invocation/b12x")
    args = parser.parse_args()

    import cupy
    import deep_gemm
    import flashinfer
    import instanttensor
    import lmcache
    import nccl
    import vllm
    from b12x.attention import dense_mla
    from b12x.comm.pcie.pcie_dcp_a2a import SUPPORTED_WORLD_SIZES
    from cutlass import cute
    from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy
    from vllm.models.deepseek_v4.nvidia.ops import fused_indexer_q_cutedsl
    from vllm.vllm_flash_attn.cute import utils as flash_attn_cute_utils

    assert torch.__version__ == "2.13.0"
    assert torch.version.cuda == "13.3"
    assert torch.cuda.nccl.version() == (2, 31, 2)

    nccl_path = Path(os.environ["NCCL_LOCAL_INFERENCE_PATH"]).resolve(strict=True)
    nccl_library = ctypes.CDLL(str(nccl_path))
    version = ctypes.c_int()
    assert nccl_library.ncclGetVersion(ctypes.byref(version)) == 0
    assert version.value == 23102
    assert any(
        path.exists() and os.path.samefile(path, nccl_path)
        for path in _mapped_libraries("libnccl")
    )

    assert metadata.version("vllm") == args.vllm_version
    assert metadata.version("flashinfer-python") == args.flashinfer_version
    assert metadata.version("lmcache") == args.lmcache_version
    assert metadata.version("instanttensor") == args.instanttensor_version
    assert metadata.version("nccl4py") == args.nccl4py_version
    assert metadata.version("xgrammar") == args.xgrammar_version
    assert metadata.version("cupy-cuda13x") == "13.6.0"
    assert metadata.version("nvidia-cutlass-dsl") == args.cutlass_dsl_version
    assert metadata.version("nvidia-cutlass-dsl-libs-base") == args.cutlass_dsl_version
    assert metadata.version("nvidia-cutlass-dsl-libs-cu13") == args.cutlass_dsl_version
    assert os.environ["CUTLASS_DSL_VERSION"] == args.cutlass_dsl_version
    assert metadata.version("pytest") == "8.4.1"
    assert hasattr(cute.nvgpu.warp, "MmaMXF8Op")
    assert flashinfer.__file__ and instanttensor.__file__ and lmcache.__file__
    assert nccl.__file__
    assert deep_gemm.__file__ and cupy.__file__ and vllm.__file__
    assert fused_indexer_q_cutedsl.__file__ and flash_attn_cute_utils.__file__

    vllm_root = Path(args.vllm_source_root).resolve(strict=True)
    b12x_root = Path(args.b12x_source_root).resolve(strict=True)
    assert Path(vllm.__file__).resolve(strict=True).is_relative_to(vllm_root)
    assert Path(dense_mla.__file__).resolve(strict=True).is_relative_to(b12x_root)
    assert SUPPORTED_WORLD_SIZES == (2, 4, 8, 16)
    _verify_lmcache_atomic_publication()

    strategy = ParallelStrategy(
        mla_only=True,
        vllm_world_size=2,
        vllm_worker_id=1,
        tp_size=2,
        pp_size=1,
        n_servers=1,
        dcp_size=1,
    )
    assert (strategy.kv_world_size, strategy.kv_worker_id) == (1, 0)
    assert (strategy.kv_tp_size, strategy.num_kv_readers) == (2, 2)
    assert not strategy.is_kv_writer

    writer_strategy = ParallelStrategy(
        mla_only=True,
        vllm_world_size=2,
        vllm_worker_id=0,
        tp_size=2,
        pp_size=1,
        n_servers=1,
        dcp_size=1,
    )
    assert writer_strategy.is_kv_writer

    for launcher in (
        "/usr/local/bin/serve-ds4-flash.sh",
        "/usr/local/bin/serve-ds4-flash-spark.sh",
        "/usr/local/bin/serve-infernal-invocation.sh",
        "/usr/local/bin/serve-glm52-v19.sh",
        "/usr/local/bin/lmcache-mp-wrapper.sh",
    ):
        assert os.access(launcher, os.X_OK), launcher

    installed = {
        distribution.metadata["Name"].lower()
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }
    overlays = sorted(
        package
        for package in installed
        if package.startswith(CUDA_RUNTIME_WHEEL_PREFIXES)
        and package not in ALLOWED_CUDA_TOOLING_WHEELS
    )
    assert not overlays, f"pip CUDA runtime overlays are installed: {overlays}"

    print(
        "DeepSeek Infernal Invocation CUDA 13.3 runtime contract: PASS "
        f"torch={torch.__version__} cuda={torch.version.cuda} nccl={version.value} "
        f"vllm={metadata.version('vllm')} b12x={metadata.version('b12x')} "
        f"flashinfer={metadata.version('flashinfer-python')} "
        f"lmcache={metadata.version('lmcache')} "
        f"xgrammar={metadata.version('xgrammar')}"
    )


if __name__ == "__main__":
    main()
