#!/bin/bash
# Build script for bleeding-edge LLM inference stack (RTX 5090 / sm_120f)
#
# Usage:
#   ./build.sh              # default build (64 parallel jobs)
#   ./build.sh --no-cache   # clean rebuild
#   MAX_JOBS=128 ./build.sh # more parallel jobs

set -euo pipefail

IMAGE_NAME="llm-pytorch-blackwell"
IMAGE_TAG="nightly"

MAX_JOBS="${MAX_JOBS:-128}"
NVCC_THREADS="${NVCC_THREADS:-8}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  MAX_JOBS=${MAX_JOBS}  NVCC_THREADS=${NVCC_THREADS}"
echo ""

BUILD_ID="$(date +%Y%m%d-%H%M%S)"
CACHEBUST="$(date +%s)"

DOCKER_BUILDKIT=1 docker build \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --build-arg NVCC_THREADS="${NVCC_THREADS}" \
    --build-arg BUILD_ID="${BUILD_ID}" \
    --build-arg CACHEBUST="${CACHEBUST}" \
    --progress=plain \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$@" \
    .

echo ""
echo "Done. Run with:"
echo "  docker run --gpus all --ipc=host --ulimit memlock=-1 -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG} --model <model-name>"
