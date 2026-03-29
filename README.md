# Blackwell LLM Docker

Docker images for LLM inference on NVIDIA Blackwell GPUs (SM120). Pre-built images are available on Docker Hub:

- **[voipmonitor/sglang:test-cu132](https://hub.docker.com/r/voipmonitor/sglang)**

## Image

| Image | Dockerfile | Stack |
|-------|-----------|-------|
| `voipmonitor/sglang:test-cu132` | `Dockerfile.sglang` | CUDA 13.2 base, PyTorch stable (cu130), FlashInfer nightly, Triton 3.6.0, SGLang from source |

Includes: CUTLASS 4.x DSL, sgl-kernel (SM120+SM90), PCIe allreduce, b12x NVFP4 backend, pre-tuned Triton MoE configs for RTX PRO 6000 Blackwell, JIT cache management.

## Run

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit -p 5000:5000 voipmonitor/sglang:test-cu132 \
  python -m sglang.launch_server --model-path <model> --tp 8 --host 0.0.0.0 --port 5000
```

## Build from source

```bash
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.sglang -t voipmonitor/sglang:test-cu132 .
```

## Hardware

- NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120) or compatible
- CUDA driver 575+
- 96 GB VRAM per GPU

## Changelog

### 2026-03-29 (v2)
- **New cherry-picks:**
  - [PR #20182](https://github.com/sgl-project/sglang/pull/20182) — fix mamba/GDN memory leak on request abort under concurrency (crash: `token_to_kv_pool_allocator memory leak detected!`)
  - [PR #20433](https://github.com/sgl-project/sglang/pull/20433) — async extra_buffer SSM state tracking for NEXTN spec-v2 (~15% MTP throughput improvement)
  - [PR #20445](https://github.com/sgl-project/sglang/pull/20445) — remove GPU sync points in mamba/GDN track metadata (perf)
  - [PR #21599](https://github.com/sgl-project/sglang/pull/21599) — adaptive speculative decoding for EAGLE topk=1 (dynamically adjusts `speculative_num_steps` based on acceptance rate, enable with `--speculative-adaptive`)
  - [PR #21601](https://github.com/sgl-project/sglang/pull/21601) — FP4 KV cache support for SM120 GPUs (`--kv-cache-dtype nvfp4`)
- **Conflict resolution scripts** for PRs with merge conflicts (no `|| true`):
  - `fix-mamba-leak-conflict.py` — resolves #20182 conflict (hisparse_coordinator in refactored helper)
  - `fix-fp4-kvcache-conflict.py` — resolves #21601 conflicts (flashinfer piecewise CUDA graph + trtllm skip_softmax params)

### 2026-03-29
- **KLD logit capture patch rewritten:**
  - Fixed MTP head contamination (doubled files, inflated KLD by ~18%)
  - Changed log-prob storage from float16 to float32 (matches vLLM PR #35961)
  - Auto-filters MTP/NextN speculative heads via call stack inspection
- **Added `fix-nvfp4-dual-gencode.py` patch** — removes 10.3a from nvfp4 JIT arch list to prevent SGL_CUDA_ARCH mismatch on SM120
- **Corrected KLD results** (Qwen3.5-397B-A17B vs FP8 reference):

  | Model | Mean KLD |
  |-------|----------|
  | AWQ (QuantTrio) | 0.024 |
  | nvidia/NVFP4 | 0.035 |
  | lukealonso/NVFP4 | 0.036 |

### 2026-03-28
- **FlashInfer switched from source build to nightly pip**
- **Triton updated from 3.5.1 to 3.6.0**
- **Cherry-picks cleaned up** — removed merged PRs, added syntax verification
- **PR #19963 adapted** for refactored ArchInfo code
- **Removed vLLM image**

### 2026-03-26
- Initial CUDA 13.2 build
