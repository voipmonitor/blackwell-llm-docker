#!/bin/bash
# Invalidate JIT caches when image changes.
# Runs from entrypoint.sh and .bashrc to cover all launch modes.
#
# BUILD_ID sources (in priority order):
#   1. JIT_BUILD_ID env var (set via --build-arg BUILD_ID=...)
#   2. /etc/jit-build-id file (written during docker build)
#   3. Docker image ID from /proc/self/cgroup (fallback)
_BID="${JIT_BUILD_ID:-}"
if [ -z "$_BID" ] || [ "$_BID" = "unknown" ]; then
    _BID="$(cat /etc/jit-build-id 2>/dev/null || true)"
fi
if [ -z "$_BID" ] || [ "$_BID" = "unknown" ]; then
    # Derive from container's image hash — changes every rebuild
    _BID="$(cat /proc/self/cgroup 2>/dev/null | grep -oP 'docker/\K[a-f0-9]{12}' | head -1 || true)"
fi

BUILD_MARKER="/cache/jit/.build-id"
if [ -n "$_BID" ]; then
    if [ ! -f "$BUILD_MARKER" ] || [ "$(cat "$BUILD_MARKER" 2>/dev/null)" != "$_BID" ]; then
        echo "New image detected (build: $_BID). Clearing JIT caches..."
        rm -rf /cache/jit/flashinfer /cache/jit/tvm-ffi /cache/jit/triton /cache/jit/torch_extensions /cache/jit/vllm
        mkdir -p /cache/jit/triton /cache/jit/torch_extensions /cache/jit/vllm
        echo "$_BID" > "$BUILD_MARKER"
    fi
fi
unset _BID
