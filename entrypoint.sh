#!/bin/bash
set -euo pipefail

# =============================================================================
# Entrypoint for LLM Blackwell container (vLLM / SGLang)
# - Warms up JIT caches (Triton, torch) on first run
# - Then passes through to whatever command the user specified
# =============================================================================

source /usr/local/bin/jit-cache-invalidate.sh

WARMUP_MARKER="/cache/jit/.warmup-done-$(python -c 'import torch; print(torch.__version__)')"

warmup_jit() {
    echo "============================================="
    echo " JIT Warmup: compiling Triton/vLLM kernels..."
    echo " This runs only once per cache volume."
    echo "============================================="

    python /workspace/warmup_jit.py

    mkdir -p "$(dirname "$WARMUP_MARKER")"
    touch "$WARMUP_MARKER"

    echo "============================================="
    echo " JIT Warmup complete. Cached for next start."
    echo "============================================="
}

# Run warmup if cache marker doesn't exist
if [ ! -f "$WARMUP_MARKER" ]; then
    warmup_jit
else
    echo "JIT cache found ($(basename "$WARMUP_MARKER")), skipping warmup."
fi

# If no arguments passed, show help
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage:"
    echo "  docker run ... <image> --model <model-name> [vllm args...]"
    echo "  docker run ... <image> python3 -m sglang.launch_server --model-path <model> [sglang args...]"
    echo ""
    exec python -m vllm.entrypoints.openai.api_server --help
fi

# If first arg starts with "--", assume vLLM server mode
if [[ "$1" == --* ]]; then
    exec python -m vllm.entrypoints.openai.api_server "$@"
else
    # Otherwise pass through as a raw command (e.g. python3 -m sglang.launch_server ...)
    exec "$@"
fi
