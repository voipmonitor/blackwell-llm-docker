# Blackwell LLM Docker

Docker images for LLM inference on NVIDIA Blackwell GPUs (SM120). Pre-built images are available on Docker Hub:

- **[voipmonitor/sglang:test-cu132](https://hub.docker.com/r/voipmonitor/sglang)**

## Image

| Image | Dockerfile | Stack |
|-------|-----------|-------|
| `voipmonitor/sglang:test-cu132` | `Dockerfile.sglang` | CUDA 13.2 base, PyTorch 2.11 stable (cu130), FlashInfer nightly, Triton 3.6.0, SGLang from source |

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

### 2026-03-29
- **KLD logit capture patch rewritten** — fixed two bugs that contaminated KLD measurements:
  1. **MTP head contamination**: LogitsProcessor.forward() was called by both the main model head and MTP speculative heads during prefill, doubling the captured files (200 instead of 100). MTP head logits (higher entropy, different distribution) inflated mean KLD by ~18%. Fixed by inspecting the call stack for MTP model files (`*mtp*.py`, `*nextn*.py`) and skipping those calls.
  2. **float16 precision loss**: log-probabilities were stored as float16. Changed to float32 (computed on CPU to avoid GPU OOM). Matches vLLM PR #35961 methodology.
  3. **DRAFT_EXTEND filter**: also skips post-decode MTP forward passes (forward_mode=DRAFT_EXTEND).
- **Added `fix-nvfp4-dual-gencode.py` patch** — on CUDA 13+, `_get_nvfp4_cuda_arch_list()` appends `"10.3a"` to the JIT arch list. With `SGL_CUDA_ARCH=1200` (SM120), nvcc compiles both SM120 and SM103 variants; the SM103 variant triggers a static assertion (`__CUDA_ARCH__=1030 != SGL_CUDA_ARCH=1200`). Fix: remove the 10.3a append — JIT should only compile for the actual device.
- **Corrected KLD results** (Qwen3.5-397B-A17B NVFP4 vs FP8 reference, main head only):

  | Backend | Mean KLD |
  |---------|----------|
  | AWQ (QuantTrio) | 0.024 |
  | flashinfer_cutlass | 0.036 |
  | cutedsl + cudnn | 0.036 |
  | cutlass MoE | 0.036 |

### 2026-03-28
- **PyTorch downgraded from 2.12 nightly to 2.11 stable (cu130)** — torch 2.12 has a CUDA graph memory management regression (`create_allocate_filter()`) that causes illegal memory access during MTP speculative decoding with CUDA graph padding. See [SGLang #21597](https://github.com/sgl-project/sglang/issues/21597). Torch 2.11 stable from `https://download.pytorch.org/whl/cu130` works on CUDA 13.2 base without issues.
- **FlashInfer switched from source build to nightly pip** — `pip install --pre flashinfer-python` from `https://flashinfer.ai/whl/nightly/` with pre-compiled cubins and JIT cache for cu130.
- **Triton updated from 3.5.1 to 3.6.0**
- **Cherry-picks cleaned up** — removed `|| true` from all cherry-picks (build fails loudly on conflicts), removed merged PRs (#20232, #20441), removed PRs with merge conflicts that are not needed for GLM-5 (#20074 Qwen3.5 DeltaNet, #20182/#20377/#20433/#20445 Mamba fixes).
- **PR #19963 adapted** — original cherry-pick conflicts with refactored `ArchInfo` code in main. Replaced with `fix-cuda-arch-suffix.py` patch that applies the same fix (append `"a"` suffix for CC >= 9.0) to the new code structure.
- **Added syntax verification** — all `.py` files are parsed with `ast.parse()` after cherry-picks to catch merge conflict artifacts.
- **Added `FLASHINFER_DISABLE_VERSION_CHECK=1`** to runtime env.
- **Removed vLLM image** — no longer maintained.

### 2026-03-26
- Initial CUDA 13.2 + PyTorch 2.12 nightly build
