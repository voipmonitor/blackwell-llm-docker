#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "ERROR: lmcache-mp-wrapper.sh requires a model server command" >&2
  exit 2
fi

mode="${LMCACHE_MODE:-off}"
mode="${mode,,}"
case "${mode}" in
  off|0)
    exec "$@"
    ;;
  ram|memory|1)
    mode=ram
    ;;
  disk|ram-disk|memory-disk)
    mode=disk
    ;;
  *)
    echo "ERROR: LMCACHE_MODE must be off, ram, or disk; got ${mode}" >&2
    exit 2
    ;;
esac

command -v lmcache >/dev/null || {
  echo "ERROR: LMCACHE_MODE=${mode}, but the lmcache CLI is not installed" >&2
  exit 2
}

lmcache_expected_source_root="${LMCACHE_EXPECTED_SOURCE_ROOT:-}"
if [[ -n "${lmcache_expected_source_root}" ]]; then
  "${PYTHON_BIN:-python3}" - "${lmcache_expected_source_root}" <<'PY'
from pathlib import Path
import sys

import lmcache

expected = Path(sys.argv[1]).resolve()
loaded = Path(lmcache.__file__).resolve()
if not loaded.is_relative_to(expected):
    raise SystemExit(
        f"LMCache imported from {loaded}; expected a module below {expected}"
    )
PY
fi

service_port="${PORT:-8000}"
port_offset=0
if [[ "${service_port}" =~ ^[0-9]+$ ]] && (( service_port >= 8000 )); then
  port_offset=$((service_port - 8000))
fi

lmcache_host="${LMCACHE_HOST:-127.0.0.1}"
# Derived ports assume each service uses a unique PORT offset. Deployments with
# custom spacing must set both LMCACHE_PORT and LMCACHE_HTTP_PORT explicitly.
lmcache_port="${LMCACHE_PORT:-$((5555 + port_offset))}"
lmcache_http_port="${LMCACHE_HTTP_PORT:-$((8099 + port_offset))}"
lmcache_chunk_size="${LMCACHE_CHUNK_SIZE:-}"
if [[ -z "${lmcache_chunk_size}" ]]; then
  # LMCache chunks must align to every effective DCP paged-cache block.
  # TP6 uses 192-token (DCP3) or 384-token (DCP6) manager blocks; 512 is
  # invalid for both. Power-of-two DCP layouts retain the established 512.
  case "${DCP_SIZE:-${DCP:-1}}" in
    3|6) lmcache_chunk_size=384 ;;
    *) lmcache_chunk_size=512 ;;
  esac
fi
lmcache_l1_gb="${LMCACHE_L1_GB:-24}"
lmcache_l1_init_gb="${LMCACHE_L1_INIT_GB:-${lmcache_l1_gb}}"
# Every TP rank registers an independent GPU client, including at DCP1. Give
# each client its own affinity worker so rank transfers are not serialized.
# Constrained hosts can still override this explicitly.
lmcache_gpu_workers="${LMCACHE_MAX_GPU_WORKERS:-${TP_SIZE:-${TP:-1}}}"
lmcache_cpu_workers="${LMCACHE_MAX_CPU_WORKERS:-4}"
lmcache_log="${LMCACHE_LOG:-/tmp/lmcache-mp-${service_port}.log}"
lmcache_transfer_mode="${LMCACHE_TRANSFER_MODE:-auto}"
lmcache_transfer_mode="${lmcache_transfer_mode,,}"
case "${lmcache_transfer_mode}" in
  auto|lmcache_driven|engine_driven) ;;
  *)
    echo "ERROR: LMCACHE_TRANSFER_MODE must be auto, lmcache_driven, or engine_driven; got ${lmcache_transfer_mode}" >&2
    exit 2
    ;;
esac
lmcache_auto_transfer_mode="${LMCACHE_AUTO_TRANSFER_MODE:-auto}"
lmcache_auto_transfer_mode="${lmcache_auto_transfer_mode,,}"
case "${lmcache_auto_transfer_mode}" in
  auto|lmcache_driven|engine_driven) ;;
  *)
    echo "ERROR: LMCACHE_AUTO_TRANSFER_MODE must be auto, lmcache_driven, or engine_driven; got ${lmcache_auto_transfer_mode}" >&2
    exit 2
    ;;
esac
if [[ "${lmcache_transfer_mode}" == "auto" \
  && "${lmcache_auto_transfer_mode}" != "auto" ]]; then
  lmcache_transfer_mode="${lmcache_auto_transfer_mode}"
fi
# The downstream model launcher must classify the same effective transport
# that the cache server and vLLM connector receive.
export LMCACHE_TRANSFER_MODE="${lmcache_transfer_mode}"

lmcache_kv_load_failure_policy="${LMCACHE_KV_LOAD_FAILURE_POLICY:-recompute}"
lmcache_kv_load_failure_policy="${lmcache_kv_load_failure_policy,,}"
case "${lmcache_kv_load_failure_policy}" in
  recompute|fail) ;;
  *)
    echo "ERROR: LMCACHE_KV_LOAD_FAILURE_POLICY must be recompute or fail; got ${lmcache_kv_load_failure_policy}" >&2
    exit 2
    ;;
esac

lmcache_mq_timeout="${LMCACHE_MQ_TIMEOUT:-60}"
lmcache_heartbeat_interval="${LMCACHE_HEARTBEAT_INTERVAL:-10}"
lmcache_worker_reap_timeout="${LMCACHE_WORKER_REAP_TIMEOUT:-120}"
lmcache_worker_registration_grace="${LMCACHE_WORKER_REGISTRATION_GRACE:-3600}"

lmcache_separate_object_groups="${LMCACHE_SEPARATE_OBJECT_GROUPS:-0}"
lmcache_separate_object_groups="${lmcache_separate_object_groups,,}"
case "${lmcache_separate_object_groups}" in
  1|true|yes|on) lmcache_separate_object_groups=1 ;;
  0|false|no|off) lmcache_separate_object_groups=0 ;;
  *)
    echo "ERROR: LMCACHE_SEPARATE_OBJECT_GROUPS must be a boolean; got ${lmcache_separate_object_groups}" >&2
    exit 2
    ;;
esac

lmcache_shm_name="${LMCACHE_SHM_NAME-}"
lmcache_shm_name_is_explicit=0
if [[ -v LMCACHE_SHM_NAME ]]; then
  lmcache_shm_name_is_explicit=1
elif [[ "${lmcache_transfer_mode}" == engine_driven ]]; then
  # Host-network services cannot share a PORT. The port therefore gives each
  # concurrently runnable cache sidecar a stable, collision-free SHM arena.
  lmcache_shm_name="lmcache-${service_port}"
fi
if [[ -n "${lmcache_shm_name}" \
  && ! "${lmcache_shm_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: LMCACHE_SHM_NAME contains unsupported characters: ${lmcache_shm_name}" >&2
  exit 2
fi

server_args=(
  server
  --host "${lmcache_host}"
  --port "${lmcache_port}"
  --chunk-size "${lmcache_chunk_size}"
  --max-gpu-workers "${lmcache_gpu_workers}"
  --max-cpu-workers "${lmcache_cpu_workers}"
  --supported-transfer-mode "${lmcache_transfer_mode}"
  --l1-size-gb "${lmcache_l1_gb}"
  --l1-init-size-gb "${lmcache_l1_init_gb}"
  --l1-write-ttl-seconds 600
  --l1-read-ttl-seconds 300
  --eviction-policy LRU
  --eviction-trigger-watermark 0.90
  --eviction-ratio 0.10
  --l2-store-policy default
  --l2-prefetch-policy retain
  --http-port "${lmcache_http_port}"
  --worker-reap-timeout-seconds "${lmcache_worker_reap_timeout}"
  --worker-registration-grace-seconds "${lmcache_worker_registration_grace}"
)

# Engine-driven SHM exposes stable direct views into the complete L1 arena, so
# it requires a named, non-lazy pool. An explicitly empty name remains a
# diagnostic switch for the bounded pickle transport.
if [[ "${lmcache_transfer_mode}" == engine_driven \
  && -n "${lmcache_shm_name}" ]]; then
  server_args+=(--no-l1-use-lazy --shm-name "${lmcache_shm_name}")
else
  server_args+=(--l1-use-lazy)
  if [[ "${lmcache_shm_name_is_explicit}" == 1 ]]; then
    server_args+=(--shm-name "${lmcache_shm_name}")
  fi
fi
if [[ "${lmcache_separate_object_groups}" == 1 ]]; then
  server_args+=(--separate-object-groups)
fi

lmcache_l2_path=disabled
if [[ "${mode}" == "disk" ]]; then
  lmcache_l2_path="${LMCACHE_L2_PATH:-/cache/lmcache/${service_port}}"
  lmcache_l2_gb="${LMCACHE_L2_GB:-256}"
  lmcache_l2_workers="${LMCACHE_L2_WORKERS:-4}"
  mkdir -p "${lmcache_l2_path}"
  l2_config="$(
    LMCACHE_JSON_PATH="${lmcache_l2_path}" \
    LMCACHE_JSON_WORKERS="${lmcache_l2_workers}" \
    LMCACHE_JSON_CAPACITY_GB="${lmcache_l2_gb}" \
      python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "type": "fs_native",
            "base_path": os.environ["LMCACHE_JSON_PATH"],
            "num_workers": int(os.environ["LMCACHE_JSON_WORKERS"]),
            "use_odirect": False,
            "max_capacity_gb": float(os.environ["LMCACHE_JSON_CAPACITY_GB"]),
        },
        separators=(",", ":"),
    )
)
PY
  )"
  server_args+=(--l2-adapter "${l2_config}")
fi

transfer_config="$(
  LMCACHE_JSON_HOST="${lmcache_host}" \
  LMCACHE_JSON_PORT="${lmcache_port}" \
  LMCACHE_JSON_TRANSFER_MODE="${lmcache_transfer_mode}" \
  LMCACHE_JSON_LOAD_FAILURE_POLICY="${lmcache_kv_load_failure_policy}" \
  LMCACHE_JSON_MQ_TIMEOUT="${lmcache_mq_timeout}" \
  LMCACHE_JSON_HEARTBEAT_INTERVAL="${lmcache_heartbeat_interval}" \
  LMCACHE_JSON_WORKER_REAP_TIMEOUT="${lmcache_worker_reap_timeout}" \
  LMCACHE_JSON_WORKER_REGISTRATION_GRACE="${lmcache_worker_registration_grace}" \
    python3 - <<'PY'
import json
import math
import os

mq_timeout = float(os.environ["LMCACHE_JSON_MQ_TIMEOUT"])
heartbeat_interval = float(os.environ["LMCACHE_JSON_HEARTBEAT_INTERVAL"])
worker_reap_timeout = float(os.environ["LMCACHE_JSON_WORKER_REAP_TIMEOUT"])
worker_registration_grace = float(
    os.environ["LMCACHE_JSON_WORKER_REGISTRATION_GRACE"]
)
for name, value in (
    ("LMCACHE_MQ_TIMEOUT", mq_timeout),
    ("LMCACHE_HEARTBEAT_INTERVAL", heartbeat_interval),
    ("LMCACHE_WORKER_REAP_TIMEOUT", worker_reap_timeout),
    ("LMCACHE_WORKER_REGISTRATION_GRACE", worker_registration_grace),
):
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be finite; got {value}")
if mq_timeout <= 0:
    raise SystemExit(f"LMCACHE_MQ_TIMEOUT must be positive; got {mq_timeout}")
if heartbeat_interval <= 0:
    raise SystemExit(
        "LMCACHE_HEARTBEAT_INTERVAL must be positive; "
        f"got {heartbeat_interval}"
    )
if worker_reap_timeout != 0 and worker_reap_timeout < 30:
    raise SystemExit(
        "LMCACHE_WORKER_REAP_TIMEOUT must be 0 or at least 30 seconds; "
        f"got {worker_reap_timeout}"
    )
if worker_registration_grace < worker_reap_timeout:
    raise SystemExit(
        "LMCACHE_WORKER_REGISTRATION_GRACE must be at least "
        f"LMCACHE_WORKER_REAP_TIMEOUT; got {worker_registration_grace} "
        f"and {worker_reap_timeout}"
    )
if worker_reap_timeout and 3 * heartbeat_interval > worker_reap_timeout:
    raise SystemExit(
        "LMCACHE_WORKER_REAP_TIMEOUT must be at least three heartbeat "
        f"intervals; got {worker_reap_timeout} and {heartbeat_interval}"
    )

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            # A cache-tier failure is not a model failure. Recompute discards
            # the affected external blocks and preserves the API request.
            "kv_load_failure_policy": os.environ[
                "LMCACHE_JSON_LOAD_FAILURE_POLICY"
            ],
            "kv_connector_extra_config": {
                "lmcache.mp.host": f"tcp://{os.environ['LMCACHE_JSON_HOST']}",
                "lmcache.mp.port": int(os.environ["LMCACHE_JSON_PORT"]),
                "lmcache.mp.mq_timeout": mq_timeout,
                "lmcache.mp.heartbeat_interval": heartbeat_interval,
                "lmcache.mp.mp_transfer_mode": os.environ[
                    "LMCACHE_JSON_TRANSFER_MODE"
                ],
            },
        },
        separators=(",", ":"),
    )
)
PY
)"

# LMCache-driven transfers register KV storage by address, so expandable CUDA
# allocator segments could invalidate the registered addresses. Engine-driven
# transfers copy through vLLM-owned staging buffers and impose no stable-address
# requirement on the model allocator.
if [[ "${lmcache_transfer_mode}" != "engine_driven" ]]; then
  allocator_config="${PYTORCH_CUDA_ALLOC_CONF:-}"
  if [[ -z "${allocator_config}" ]]; then
    allocator_config="expandable_segments:False"
  elif [[ "${allocator_config}" =~ (^|,)expandable_segments:True(,|$) ]]; then
    allocator_config="${allocator_config//expandable_segments:True/expandable_segments:False}"
    echo "LMCache-driven transfers require PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False; overriding expandable_segments:True"
  fi
  export PYTORCH_CUDA_ALLOC_CONF="${allocator_config}"
fi

rm -f "${lmcache_log}"
export LMCACHE_DISABLE_BANNER="${LMCACHE_DISABLE_BANNER:-1}"
lmcache_server_command=(lmcache)
if [[ "${lmcache_transfer_mode}" == engine_driven ]]; then
  # GPU visibility is removed only from the standalone cache server. GPU
  # gather/scatter operations remain in the existing vLLM worker processes.
  lmcache_server_env="${LMCACHE_SERVER_ENV-CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY}"
  read -r -a lmcache_server_env_args <<< "${lmcache_server_env}"
  for assignment in "${lmcache_server_env_args[@]}"; do
    if [[ ! "${assignment}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
      echo "ERROR: invalid LMCACHE_SERVER_ENV assignment: ${assignment}" >&2
      exit 2
    fi
  done
  lmcache_server_command=(env "${lmcache_server_env_args[@]}" lmcache)
fi
"${lmcache_server_command[@]}" "${server_args[@]}" >"${lmcache_log}" 2>&1 &
lmcache_pid=$!
model_pid=""
shutdown_requested=0

stop_children() {
  if [[ -n "${model_pid}" ]] && kill -0 "${model_pid}" 2>/dev/null; then
    kill -TERM "${model_pid}" 2>/dev/null || true
  fi
  if kill -0 "${lmcache_pid}" 2>/dev/null; then
    kill -TERM "${lmcache_pid}" 2>/dev/null || true
  fi
}

request_shutdown() {
  shutdown_requested=1
  stop_children
}
trap request_shutdown INT TERM HUP

ready=0
for _ in $(seq 1 "${LMCACHE_START_TIMEOUT:-120}"); do
  if ! kill -0 "${lmcache_pid}" 2>/dev/null; then
    break
  fi
  if curl --fail --silent --show-error --max-time 1 \
      "http://127.0.0.1:${lmcache_http_port}/healthcheck" \
      >/dev/null 2>&1; then
    ready=1
    break
  fi
  if grep -Fq "${LMCACHE_READY_LOG_TEXT:-LMCache ZMQ cache server is running}" \
      "${lmcache_log}"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "ERROR: LMCache did not become ready; log follows" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${lmcache_pid}" 2>/dev/null || true
  exit 1
fi

if [[ "${lmcache_transfer_mode}" == engine_driven \
  && -n "${lmcache_shm_name}" ]]; then
  status_file="$(mktemp)"
  if ! curl --fail --silent --show-error --max-time 5 \
      "http://127.0.0.1:${lmcache_http_port}/status" >"${status_file}"; then
    echo "ERROR: LMCache did not expose engine-driven transfer status" >&2
    rm -f "${status_file}"
    stop_children
    wait "${lmcache_pid}" 2>/dev/null || true
    exit 1
  fi
  if ! "${PYTHON_BIN:-python3}" - "${status_file}" "${lmcache_l1_gb}" <<'PY'
import json
import sys

status_path, configured_gib = sys.argv[1:]
with open(status_path, encoding="utf-8") as stream:
    status = json.load(stream)
pool = status.get("engine_driven_shm_pool") or {}
name = pool.get("shm_name") or ""
size = int(pool.get("pool_size") or 0)
minimum = int(float(configured_gib) * 1024**3)
if not name or size < minimum:
    raise SystemExit(
        "engine-driven SHM is unavailable: "
        f"name={name!r}, pool_size={size}, required={minimum}"
    )
PY
  then
    echo "ERROR: LMCache replaced the requested engine-driven SHM transport with pickle" >&2
    rm -f "${status_file}"
    stop_children
    wait "${lmcache_pid}" 2>/dev/null || true
    exit 1
  fi
  rm -f "${status_file}"
fi

printf 'LMCache ready: mode=%s transfer=%s SHM=%s L1=%sGB chunk=%s L2=%s load_failure=%s heartbeat=%ss health=http://%s:%s/healthcheck metrics=http://%s:%s/metrics log=%s\n' \
  "${mode}" "${lmcache_transfer_mode}" "${lmcache_shm_name:-disabled}" \
  "${lmcache_l1_gb}" "${lmcache_chunk_size}" "${lmcache_l2_path}" \
  "${lmcache_kv_load_failure_policy}" \
  "${lmcache_heartbeat_interval}" "${lmcache_host}" "${lmcache_http_port}" \
  "${lmcache_host}" "${lmcache_http_port}" "${lmcache_log}"

"$@" --kv-transfer-config "${transfer_config}" &
model_pid=$!

set +e
completed_pid=""
wait -n -p completed_pid "${lmcache_pid}" "${model_pid}"
first_status=$?
set -e

if [[ "${shutdown_requested}" == 1 ]]; then
  set +e
  if [[ "${completed_pid:-}" == "${model_pid}" ]]; then
    model_status=${first_status}
  else
    wait "${model_pid}" 2>/dev/null
    model_status=$?
  fi
  if [[ "${completed_pid:-}" != "${lmcache_pid}" ]]; then
    wait "${lmcache_pid}" 2>/dev/null
  fi
  set -e
  exit "${model_status}"
fi

if [[ "${completed_pid:-}" == "${lmcache_pid}" ]]; then
  echo "ERROR: LMCache exited while the model server was running" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${model_pid}" 2>/dev/null || true
  exit 1
fi

stop_children
wait "${lmcache_pid}" 2>/dev/null || true
exit "${first_status}"
