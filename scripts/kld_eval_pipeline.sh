#!/usr/bin/env bash
# KLD Evaluation Pipeline for Quantized Models
#
# Automates the full KLD evaluation workflow: start server with logit capture,
# run eval client, compute results.
#
# Usage:
#   ./scripts/kld_eval_pipeline.sh ref                                    # FP8 reference
#   ./scripts/kld_eval_pipeline.sh test nvidia/Qwen3.5-397B-A17B-NVFP4   # test model
#   ./scripts/kld_eval_pipeline.sh compute                                # compute KLD
#   ./scripts/kld_eval_pipeline.sh all nvidia/Qwen3.5-397B-A17B-NVFP4    # full pipeline
#
# Environment variables:
#   KLD_IMAGE          Docker image (default: llm-pytorch-blackwell:nightly)
#   KLD_REF_MODEL      Reference model (default: Qwen/Qwen3.5-397B-A17B-FP8)
#   KLD_REF_TP         Reference TP size (default: 8)
#   KLD_TEST_TP        Test model TP size (default: 4)
#   KLD_VOCAB_SIZE     Vocabulary size (default: 152064)
#   KLD_PORT           Server port (default: 5000)
#   KLD_BASE_DIR       Base directory for logits (default: /tmp/kld)
#   KLD_HF_CACHE       HuggingFace cache path (default: /root/.cache/huggingface)
#   KLD_EXTRA_ARGS     Extra args for sglang server (default: "")

set -euo pipefail

# --- Configuration ---
IMAGE="${KLD_IMAGE:-llm-pytorch-blackwell:nightly}"
REF_MODEL="${KLD_REF_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
REF_TP="${KLD_REF_TP:-8}"
TEST_TP="${KLD_TEST_TP:-4}"
VOCAB_SIZE="${KLD_VOCAB_SIZE:-152064}"
PORT="${KLD_PORT:-5000}"
BASE_DIR="${KLD_BASE_DIR:-/tmp/kld}"
HF_CACHE="${KLD_HF_CACHE:-/root/.cache/huggingface}"
EXTRA_ARGS="${KLD_EXTRA_ARGS:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

REF_DIR="${BASE_DIR}/ref"
TOKENIZER="${REF_MODEL}"

# --- Functions ---

log() { echo -e "\n\033[1;36m=== $* ===\033[0m\n"; }
err() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

wait_for_server() {
    local url="http://localhost:${PORT}/health"
    local timeout=600
    local start=$SECONDS
    echo -n "Waiting for server at ${url}"
    while (( SECONDS - start < timeout )); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo " ready."
            return 0
        fi
        echo -n "."
        sleep 5
    done
    err "Server not ready after ${timeout}s"
}

start_container() {
    local logits_dir="$1"
    mkdir -p "$logits_dir"

    docker run --rm -d \
        --runtime nvidia --ipc host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -p "${PORT}:${PORT}" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v vllm-nightly-jit:/cache/jit \
        -v "${logits_dir}:${logits_dir}" \
        -v "${SCRIPT_DIR}/patches/sglang-kld-logit-capture.py:/workspace/sglang-kld-logit-capture.py:ro" \
        -v "${SCRIPT_DIR}/scripts/sglang_kld_eval.py:/workspace/sglang_kld_eval.py:ro" \
        "${IMAGE}" \
        sleep infinity
}

run_in_container() {
    local cid="$1"
    shift
    docker exec "$cid" "$@"
}

stop_container() {
    local cid="$1"
    echo "Stopping container ${cid:0:12}..."
    docker stop "$cid" > /dev/null 2>&1 || true
}

run_server_and_eval() {
    local cid="$1"
    local phase="$2"
    local logits_dir="$3"
    local model="$4"
    local tp="$5"
    shift 5
    local extra_server_args=("$@")

    # Apply patch
    log "Applying logit capture patch"
    run_in_container "$cid" python /workspace/sglang-kld-logit-capture.py

    # Start server in background inside container
    log "Starting SGLang server: ${model} (TP${tp})"
    run_in_container "$cid" bash -c "
        SGLANG_KLD_SAVE_DIR=${logits_dir} \
        SGLANG_KLD_VOCAB_SIZE=${VOCAB_SIZE} \
        SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0 \
        NCCL_P2P_LEVEL=SYS \
        python -m sglang.launch_server \
            --model ${model} \
            --tp ${tp} --trust-remote-code \
            --kv-cache-dtype fp8_e4m3 \
            --mem-fraction-static 0.85 \
            --disable-custom-all-reduce \
            --host 0.0.0.0 --port ${PORT} \
            ${extra_server_args[*]:-} &
        echo \$! > /tmp/sglang.pid
    " &

    # Wait for server
    sleep 10
    wait_for_server

    # Run eval
    log "Running eval: --phase ${phase}"
    run_in_container "$cid" \
        python /workspace/sglang_kld_eval.py \
            --phase "$phase" \
            --server-url "http://localhost:${PORT}" \
            --tokenizer "$TOKENIZER" \
            --logits-dir "$logits_dir"

    # Kill server
    run_in_container "$cid" bash -c 'kill $(cat /tmp/sglang.pid) 2>/dev/null; wait $(cat /tmp/sglang.pid) 2>/dev/null || true'
    sleep 3
}

# --- Sanitize model name for directory ---
model_dir_name() {
    echo "$1" | tr '/' '_'
}

# --- Commands ---

cmd_ref() {
    log "Phase: FP8 Reference (${REF_MODEL}, TP${REF_TP})"
    mkdir -p "$REF_DIR"

    local cid
    cid=$(start_container "$REF_DIR")
    echo "Container: ${cid:0:12}"

    trap "stop_container $cid" EXIT

    run_server_and_eval "$cid" ref "$REF_DIR" "$REF_MODEL" "$REF_TP"

    log "Reference logits saved to ${REF_DIR}"
    echo "Files: $(ls "$REF_DIR"/*.safetensors 2>/dev/null | wc -l)"
}

cmd_test() {
    local model="${1:?Usage: $0 test <model> [extra sglang args...]}"
    shift
    local extra_args=("$@")
    local dir_name
    dir_name=$(model_dir_name "$model")
    local test_dir="${BASE_DIR}/test_${dir_name}"

    log "Phase: Test (${model}, TP${TEST_TP})"
    mkdir -p "$test_dir"

    local cid
    cid=$(start_container "$test_dir")
    echo "Container: ${cid:0:12}"

    trap "stop_container $cid" EXIT

    run_server_and_eval "$cid" test "$test_dir" "$model" "$TEST_TP" "${extra_args[@]}"

    log "Test logits saved to ${test_dir}"
    echo "Files: $(ls "$test_dir"/*.safetensors 2>/dev/null | wc -l)"
}

cmd_compute() {
    log "Phase: Compute KLD"

    # Find all test dirs
    local test_dirs=()
    local test_names=()
    for d in "${BASE_DIR}"/test_*; do
        if [ -d "$d" ] && ls "$d"/*.safetensors > /dev/null 2>&1; then
            test_dirs+=("$d")
            # Extract model name from dir name
            local name
            name=$(basename "$d" | sed 's/^test_//' | tr '_' '/')
            test_names+=("$name")
        fi
    done

    if [ ${#test_dirs[@]} -eq 0 ]; then
        err "No test directories found in ${BASE_DIR}/test_*"
    fi

    echo "Reference: ${REF_DIR}"
    echo "Test models: ${test_names[*]}"

    # Run compute (needs GPU for tensor ops)
    local cid
    cid=$(start_container "$BASE_DIR")
    trap "stop_container $cid" EXIT

    # Mount all dirs
    run_in_container "$cid" python /workspace/sglang_kld_eval.py \
        --phase compute \
        --ref-dir "$REF_DIR" \
        --test-dirs "${test_dirs[@]}" \
        --test-names "${test_names[@]}"
}

cmd_all() {
    local model="${1:?Usage: $0 all <test-model> [extra sglang args...]}"
    shift

    cmd_ref
    cmd_test "$model" "$@"
    cmd_compute
}

# --- Main ---

case "${1:-help}" in
    ref)     cmd_ref ;;
    test)    shift; cmd_test "$@" ;;
    compute) cmd_compute ;;
    all)     shift; cmd_all "$@" ;;
    *)
        echo "KLD Evaluation Pipeline"
        echo ""
        echo "Usage:"
        echo "  $0 ref                              Generate FP8 reference logits"
        echo "  $0 test <model> [server args...]    Generate test model logits"
        echo "  $0 compute                          Compute KLD between ref and all tests"
        echo "  $0 all <model> [server args...]     Full pipeline (ref + test + compute)"
        echo ""
        echo "Examples:"
        echo "  $0 ref"
        echo "  $0 test nvidia/Qwen3.5-397B-A17B-NVFP4 --quantization modelopt_fp4 --attention-backend triton"
        echo "  $0 compute"
        echo ""
        echo "Environment variables: KLD_IMAGE, KLD_REF_MODEL, KLD_REF_TP, KLD_TEST_TP,"
        echo "  KLD_VOCAB_SIZE, KLD_PORT, KLD_BASE_DIR, KLD_HF_CACHE, KLD_EXTRA_ARGS"
        exit 1
        ;;
esac
