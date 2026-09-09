#!/usr/bin/env bash
set -euo pipefail

# DFlash K7 verifies eight target rows per request, while MTP3 verifies four.
# These capture sizes keep C1/C8/C12/C24 target batches on exact full CUDA
# graphs. Other request counts use vLLM's next captured padded size.
readonly base_launcher=/usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh
capture_sizes=${CUDAGRAPH_CAPTURE_SIZES:-'1 2 4 8 16 32 40 48 64 96 128 192 256'}

export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

require_nonnegative_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[0-9]+$ ]] ||
    fail "${name} must be a non-negative integer; got ${value}"
}

require_open_unit_interval() {
  local name=$1
  local value=$2
  require_positive_finite "${name}" "${value}"
  awk -v value="${value}" 'BEGIN { exit !(value < 1.0) }' ||
    fail "${name} must be greater than zero and less than one; got ${value}"
}

require_positive_finite() {
  local name=$1 value=$2
  [[ ${value} =~ ^[+]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ||
    ${value} =~ ^[+]?[0-9]+\.([eE][+-]?[0-9]+)?$ ]] &&
    awk -v value="${value}" 'BEGIN {
      exit !(value > 0 && tolower(sprintf("%.17g", value)) !~ /inf|nan/)
    }' || fail "${name} must be positive and finite; got ${value}"
}

has_cli_option() {
  local expected=$1
  shift
  local argument
  for argument in "$@"; do
    if [[ ${argument} == "${expected}" || ${argument} == "${expected}="* ]]; then
      return 0
    fi
  done
  return 1
}

runtime_args=()
launcher_args=("$@")

if (($# == 1)) && [[ $1 == --help || $1 == -h ]]; then
  printf '%s\n' 'GLM-5.3-Flash scheduler controls (environment -> vLLM CLI):
  PREFILL_COMPUTE_SHARE -> --prefill-compute-share: auto or 0 < float < 1
  PREFILL_COMPUTE_HALF_LIFE -> --prefill-compute-half-life: smooth, responsive, or positive finite seconds; requires auto share
  MAX_PARALLEL_PREFILLS -> --max-parallel-prefills: auto or positive integer
  PREFILL_POLICY -> --prefill-policy: round-robin or decode-aware
  DECODE_REFILL_TARGET -> --decode-refill-target: auto or positive integer

Explicit CLI values override environment values. Unset controls retain vLLM defaults.
The community image sets fixed share 0.4, interval 1, and retains one prefill lane.
Compute-share fairness requires --prefill-schedule-interval 1.
FAIRNESS_ENGINE=none disables inherited share; an explicit CLI share still applies.
FAIRNESS_ENGINE=micro_slicing is unsupported.
Use DRY_RUN=1 with CACHE_MODE=vram to inspect the complete serving command.'
  exit 0
fi

# Resolve the effective value before validation. Invalid shadowed environment
# values must not reject a valid command-line override.
resolve_option() {
  local option=$1 environment_value=$2 emit_environment=${3:-1}
  local index argument seen=0
  resolved_value=${environment_value}
  for ((index = 0; index < ${#launcher_args[@]}; index++)); do
    argument=${launcher_args[index]}
    if [[ ${argument} == "${option}" || ${argument} == "${option}="* ]]; then
      ((seen == 0)) || fail "Specify ${option} only once"
      seen=1
      if [[ ${argument} == "${option}" ]]; then
        ((index + 1 < ${#launcher_args[@]})) || fail "${option} requires a value"
        resolved_value=${launcher_args[index + 1]}
      else
        resolved_value=${argument#*=}
      fi
      [[ -n ${resolved_value} && ${resolved_value} != --* ]] ||
        fail "${option} requires a value"
    fi
  done
  if ((seen == 0 && emit_environment)) && [[ -n ${environment_value} ]]; then
    runtime_args+=("${option}" "${environment_value}")
  fi
}

# Compute-share fairness charges measured model-execution time to prefill and
# decode. ``PREFILL_COMPUTE_SHARE`` is the vLLM scheduler interface. The
# ``FAIRNESS_ENGINE`` environment variable remains a launcher compatibility
# control for deployments created before the scheduler exposed compute share
# directly.
fairness_engine=${FAIRNESS_ENGINE:-}
share_environment=${PREFILL_COMPUTE_SHARE:-}
case "${fairness_engine}" in
  '' | compute_share) ;;
  none) share_environment= ;;
  micro_slicing)
    fail 'FAIRNESS_ENGINE=micro_slicing is unsupported by this vLLM scheduler; use compute_share or none'
    ;;
  *)
    fail "FAIRNESS_ENGINE must be none, compute_share, or micro_slicing; got ${fairness_engine}"
    ;;
esac

resolve_option --prefill-compute-share "${share_environment}"
prefill_compute_share=${resolved_value}
if [[ ${fairness_engine} == compute_share && -z ${prefill_compute_share} ]]; then
  fail 'PREFILL_COMPUTE_SHARE or --prefill-compute-share is required for FAIRNESS_ENGINE=compute_share'
fi
if [[ -n ${prefill_compute_share} && ${prefill_compute_share} != auto ]]; then
  require_open_unit_interval PREFILL_COMPUTE_SHARE "${prefill_compute_share}"
fi
resolve_option --prefill-schedule-interval "${PREFILL_SCHEDULE_INTERVAL:-8}" 0
if [[ -n ${prefill_compute_share} && ${resolved_value} != 1 ]]; then
  fail 'PREFILL_SCHEDULE_INTERVAL/--prefill-schedule-interval must be 1 when compute-share fairness is enabled'
fi

resolve_option --prefill-compute-half-life "${PREFILL_COMPUTE_HALF_LIFE:-}"
if [[ -n ${resolved_value} ]]; then
  [[ ${prefill_compute_share} == auto ]] ||
    fail 'PREFILL_COMPUTE_HALF_LIFE requires effective PREFILL_COMPUTE_SHARE=auto'
  case "${resolved_value}" in
    smooth | responsive) ;;
    *) require_positive_finite PREFILL_COMPUTE_HALF_LIFE "${resolved_value}" ;;
  esac
fi

resolve_option --max-parallel-prefills "${MAX_PARALLEL_PREFILLS:-}"
if [[ -n ${resolved_value} && ${resolved_value} != auto ]]; then
  [[ ${resolved_value} =~ ^[1-9][0-9]*$ ]] ||
    fail "MAX_PARALLEL_PREFILLS must be auto or a positive integer; got ${resolved_value}"
fi
resolve_option --prefill-policy "${PREFILL_POLICY:-}"
case "${resolved_value}" in
  '' | round-robin | decode-aware) ;;
  *) fail "PREFILL_POLICY must be round-robin or decode-aware; got ${resolved_value}" ;;
esac
resolve_option --decode-refill-target "${DECODE_REFILL_TARGET:-}"
if [[ -n ${resolved_value} && ${resolved_value} != auto ]]; then
  [[ ${resolved_value} =~ ^[1-9][0-9]*$ ]] ||
    fail "DECODE_REFILL_TARGET must be auto or a positive integer; got ${resolved_value}"
fi

# A caller-supplied CLI value is authoritative and must not be duplicated.
if [[ ${capture_sizes} != none ]] &&
  ! has_cli_option --cudagraph-capture-sizes "$@"; then
  read -r -a capture_size_args <<<"${capture_sizes}"
  if ((${#capture_size_args[@]} == 0)); then
    fail 'CUDAGRAPH_CAPTURE_SIZES must contain positive integers or "none"'
  fi

  previous=0
  for size in "${capture_size_args[@]}"; do
    if [[ ! ${size} =~ ^[1-9][0-9]*$ ]]; then
      fail "Invalid CUDA graph capture size: ${size}"
    fi
    if ((size <= previous)); then
      fail "CUDAGRAPH_CAPTURE_SIZES must be strictly increasing; got ${size} after ${previous}"
    fi
    if ((size > MAX_CUDAGRAPH_CAPTURE_SIZE)); then
      fail "CUDA graph capture size ${size} exceeds MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE}"
    fi
    previous=${size}
  done
  runtime_args+=(--cudagraph-capture-sizes "${capture_size_args[@]}")
fi

# Mixed decode/prefill controls are disabled by default. Nonzero values opt in
# to bounded prefill quanta, partial-prefill admission, and decode-burst QoS.
max_num_prefill_tokens_per_step=${MAX_NUM_PREFILL_TOKENS_PER_STEP:-0}
max_num_partial_prefills=${MAX_NUM_PARTIAL_PREFILLS:-0}
decode_prefill_min_decode_steps=${DECODE_PREFILL_MIN_DECODE_STEPS:-0}
decode_prefill_max_wait_ms=${DECODE_PREFILL_MAX_WAIT_MS:-0}

require_nonnegative_integer MAX_NUM_PREFILL_TOKENS_PER_STEP \
  "${max_num_prefill_tokens_per_step}"
require_nonnegative_integer MAX_NUM_PARTIAL_PREFILLS \
  "${max_num_partial_prefills}"
require_nonnegative_integer DECODE_PREFILL_MIN_DECODE_STEPS \
  "${decode_prefill_min_decode_steps}"
require_nonnegative_integer DECODE_PREFILL_MAX_WAIT_MS \
  "${decode_prefill_max_wait_ms}"

if ((max_num_prefill_tokens_per_step > 0)) &&
  ! has_cli_option --max-num-prefill-tokens-per-step "$@"; then
  runtime_args+=(
    --max-num-prefill-tokens-per-step "${max_num_prefill_tokens_per_step}"
  )
fi
if ((max_num_partial_prefills > 0)) &&
  ! has_cli_option --max-num-partial-prefills "$@"; then
  runtime_args+=(--max-num-partial-prefills "${max_num_partial_prefills}")
fi
if ((decode_prefill_min_decode_steps > 0)) &&
  ! has_cli_option --decode-prefill-min-decode-steps "$@"; then
  runtime_args+=(
    --decode-prefill-min-decode-steps "${decode_prefill_min_decode_steps}"
  )
fi
if ((decode_prefill_max_wait_ms > 0)) &&
  ! has_cli_option --decode-prefill-max-wait-ms "$@"; then
  runtime_args+=(--decode-prefill-max-wait-ms "${decode_prefill_max_wait_ms}")
fi

exec "${base_launcher}" "$@" "${runtime_args[@]}"
