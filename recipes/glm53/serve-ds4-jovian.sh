#!/usr/bin/env bash
set -euo pipefail

# Model-specific entrypoint over the same installed vLLM/B12X/LMCache packages
# as the GLM launcher. DS4_* controls isolate graph and CPU settings from the
# GLM defaults embedded in the image; native serving options remain available.
export PATH="/opt/venv/bin:${PATH}"
export PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
export DS4_MODEL_VARIANT=${DS4_MODEL_VARIANT:-text}
case "${DS4_MODEL_VARIANT}" in
  text) default_seqs=8; default_drafts=5 ;;
  vision) default_seqs=4; default_drafts=3 ;;
  *) echo 'DS4_MODEL_VARIANT must be text or vision.' >&2; exit 2 ;;
esac
export MODE=${MODE:-dspark} BACKEND=${BACKEND:-b12x-a8-dglin}
export TP_SIZE=${TP_SIZE:-${TP:-2}} DCP_SIZE=${DCP_SIZE:-${DCP:-1}}
export DSPARK_DEPTH_MODE=${DSPARK_DEPTH_MODE:-fixed}
export DSPARK_TOKENS=${DSPARK_TOKENS:-${default_drafts}}
export DRAFT_SAMPLE_METHOD=${DRAFT_SAMPLE_METHOD:-probabilistic}
export REJECTION_SAMPLE_METHOD=${REJECTION_SAMPLE_METHOD:-standard}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:--1}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-${default_seqs}}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export OMP_NUM_THREADS=${DS4_OMP_NUM_THREADS:-2}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${DS4_MAX_CUDAGRAPH_CAPTURE_SIZE:-auto}
export CUDAGRAPH_CAPTURE_SIZES=${DS4_CUDAGRAPH_CAPTURE_SIZES:-default}
export LOAD_FORMAT=${LOAD_FORMAT:-instanttensor}
export INSTANTTENSOR_BACKEND=${INSTANTTENSOR_BACKEND:-BUFFERED}
export LMCACHE_MODE=${LMCACHE_MODE:-off}
export LMCACHE_AUTO_TRANSFER_MODE=${LMCACHE_AUTO_TRANSFER_MODE:-engine_driven}
export LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-4096}
export LMCACHE_SEPARATE_OBJECT_GROUPS=${LMCACHE_SEPARATE_OBJECT_GROUPS:-1}
export LMCACHE_L1_GB=${LMCACHE_L1_GB:-24}
unset NCCL_GRAPH_FILE

exec /usr/local/bin/lmcache-mp-wrapper.sh \
  /opt/glm53-flash/vllm/serve-ds4-flash.sh "$@"
