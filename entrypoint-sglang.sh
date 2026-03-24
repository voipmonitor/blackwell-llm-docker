#!/bin/bash
set -euo pipefail

# =============================================================================
# Entrypoint for SGLang Blackwell container (no vLLM)
# - Invalidates JIT caches on image change
# - Passes through to whatever command the user specified
# =============================================================================

source /usr/local/bin/jit-cache-invalidate.sh

# If no arguments passed, show help
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage:"
    echo "  docker run ... <image> python -m sglang.launch_server --model-path <model> [sglang args...]"
    echo ""
    exec python -m sglang.launch_server --help
fi

# Pass through as a raw command
exec "$@"
