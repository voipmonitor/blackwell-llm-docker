# Blackwell LLM Docker

Docker images for LLM inference on NVIDIA Blackwell GPUs (SM120, CUDA 13.2). Pre-built images are available on Docker Hub:

- **[voipmonitor/sglang:test-cu132](https://hub.docker.com/r/voipmonitor/sglang)**
- **[voipmonitor/vllm:test-cu132](https://hub.docker.com/r/voipmonitor/vllm)**

## Images

| Image | Dockerfile | Description |
|-------|-----------|-------------|
| `voipmonitor/sglang:test-cu132` | `Dockerfile.sglang` | SGLang from source, CUDA 13.2, PyTorch nightly cu132, FlashInfer from source |
| `voipmonitor/vllm:test-cu132` | `Dockerfile.vllm` | vLLM + SGLang from source, CUDA 13.2, PyTorch nightly cu132, Triton from source |

Both images include: CUTLASS 4.x DSL, DeepGEMM, pre-tuned Triton MoE configs for RTX PRO 6000 Blackwell, JIT cache management.

## Run

```bash
# SGLang
docker run --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit -p 5000:5000 voipmonitor/sglang:test-cu132 \
  python -m sglang.launch_server --model-path <model> --tp 8 --host 0.0.0.0 --port 5000

# vLLM
docker run --gpus all --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit -p 8000:8000 voipmonitor/vllm:test-cu132 \
  --model <model> --tensor-parallel-size 8 --host 0.0.0.0
```

## Build from source

```bash
# SGLang
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.sglang -t voipmonitor/sglang:test-cu132 .

# vLLM
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.vllm -t voipmonitor/vllm:test-cu132 .
```

## Hardware

- NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120) or compatible
- CUDA driver 575+
- 96 GB VRAM per GPU
