#!/bin/bash
set -euo pipefail

# =============================================================================
# Entrypoint for SGLang Blackwell container
#
# Usage:
#   # With model profile:
#   docker run --gpus all -e MODEL_PROFILE=qwen35-b12x <image>
#
#   # With profile + extra args (appended):
#   docker run --gpus all -e MODEL_PROFILE=qwen35-b12x <image> --port 8000
#
#   # Direct command (no profile):
#   docker run --gpus all <image> python -m sglang.launch_server --model ...
#
#   # Interactive shell:
#   docker run --gpus all -it --entrypoint /bin/bash <image>
# =============================================================================

source /usr/local/bin/jit-cache-invalidate.sh 2>/dev/null || true

# --- MODEL_PROFILE mode ---
if [ -n "${MODEL_PROFILE:-}" ]; then
    PROFILE="/profiles/${MODEL_PROFILE}.sh"
    if [ ! -f "$PROFILE" ]; then
        echo "ERROR: profile '${MODEL_PROFILE}' not found at ${PROFILE}"
        echo "Available profiles:"
        ls /profiles/*.sh 2>/dev/null | sed 's|/profiles/||;s|\.sh$||' | sed 's/^/  /'
        exit 1
    fi
    source "$PROFILE"
    echo "=== Profile: ${MODEL_PROFILE} ==="
    echo "=== Model: $(echo $SGLANG_ARGS | grep -oP '(?<=--model )\S+')"
    # Export env vars from profile
    for kv in ${SGLANG_ENV:-}; do
        export "$kv"
    done
    # Launch with profile args + any extra args from docker run
    exec python3 -m sglang.launch_server $SGLANG_ARGS "$@"
fi

# --- No profile: pass through or show help ---
if [ $# -eq 0 ]; then
    echo ""
    echo "Usage:"
    echo "  docker run -e MODEL_PROFILE=<profile> <image> [extra args...]"
    echo "  docker run <image> python -m sglang.launch_server --model <model> [args...]"
    echo ""
    echo "Available profiles:"
    ls /profiles/*.sh 2>/dev/null | sed 's|/profiles/||;s|\.sh$||' | sed 's/^/  /' || echo "  (none)"
    echo ""
    exec python -m sglang.launch_server --help
fi

exec "$@"
