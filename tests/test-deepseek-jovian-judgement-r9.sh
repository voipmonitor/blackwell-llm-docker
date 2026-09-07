#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-jovian-judgement-cu133-torch213.sh"
compose="${repo_root}/examples/docker-compose-ds4-dspark-jovian-judgement-r9.yml"
vision_compose="${repo_root}/examples/docker-compose-ds4-vision-jovian-judgement-r9.yml"
composition_root="${repo_root}/patches/releases/jovian-judgement-ds4-r9"

verify_lock() {
  local component=$1 expected_base=$2 expected_tree=$3 expected_prs=$4
  local lock="${composition_root}/${component}/integration.lock.json"
  local patch="${composition_root}/${component}/integration.patch"

  jq -e \
    --arg base "${expected_base}" \
    --arg tree "${expected_tree}" \
    --argjson expected_prs "${expected_prs}" '
    .schema_version == 1 and
    .composition_strategy == "cherry_pick" and
    .base.commit == $base and
    .result.tree == $tree and
    ([.pull_requests[].number] == $expected_prs) and
    all(.pull_requests[]; .head | test("^[0-9a-f]{40}$")) and
    (.source_patches | length) == 0 and
    (.result.patch_sha256 | test("^[0-9a-f]{64}$"))
  ' "${lock}" >/dev/null
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" |
    sha256sum -c - >/dev/null
}

verify_lock vllm \
  2a979314dc97b03173a0a76fc15664ec924db32b \
  f66599d9a90d57172fd26ca5b9116f381b582b94 \
  '[628,630,634,553,671,679,694,695]'
verify_lock b12x \
  06b4de7c723e6f166d65abf5909c5b7d0f8acc68 \
  15b6813011bd47e466b39f9b474b3bca0c48c8e8 \
  '[301]'
verify_lock lmcache \
  7ed4675404a31f4ffafd98975899dc83832ba965 \
  d85748de9bf985dabc00c044396a3b8de97f4ac1 \
  '[49,50,51,55,56]'

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'revision=r9' <<<"${output}"
grep -Fxq 'vllm_tree=f66599d9a90d57172fd26ca5b9116f381b582b94' \
  <<<"${output}"
grep -Fxq 'b12x_tree=15b6813011bd47e466b39f9b474b3bca0c48c8e8' \
  <<<"${output}"
grep -Fxq 'lmcache_tree=d85748de9bf985dabc00c044396a3b8de97f4ac1' \
  <<<"${output}"
grep -Fq 'jovian-judgement-vllmf66599d-b12x15b6813-fi803c466-cu133-torch213-20260907-r9' \
  <<<"${output}"

config="$(docker compose -f "${compose}" config)"
grep -Fq 'MODE: dspark' <<<"${config}"
grep -Fq 'BACKEND: b12x-a8-dglin' <<<"${config}"
grep -Fq 'TP_SIZE: "2"' <<<"${config}"
grep -Fq 'DSPARK_TOKENS: "5"' <<<"${config}"
grep -Fq 'MAX_MODEL_LEN: "1048576"' <<<"${config}"
grep -Fq 'MAX_NUM_BATCHED_TOKENS: "4096"' <<<"${config}"
grep -Fq 'LMCACHE_MODE: "off"' <<<"${config}"
grep -Fq 'LMCACHE_TRANSFER_MODE: engine_driven' <<<"${config}"
grep -Fq 'LMCACHE_CHUNK_SIZE: "4096"' <<<"${config}"
grep -Fq 'LMCACHE_SEPARATE_OBJECT_GROUPS: "1"' <<<"${config}"
grep -Fq 'LOAD_FORMAT: instanttensor' <<<"${config}"
grep -Fq 'INSTANTTENSOR_BACKEND: BUFFERED' <<<"${config}"
grep -Fq 'jovian-judgement-vllmf66599d-b12x15b6813-fi803c466' <<<"${config}"

vision_config="$(docker compose -f "${vision_compose}" config)"
grep -Fq 'DS4_MODEL_VARIANT: vision' <<<"${vision_config}"
grep -Fq 'MODEL: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp' <<<"${vision_config}"
grep -Fq 'MODEL_REVISION: 6821d6ad3681a4b137b066b76094fa82ebd0a380' \
  <<<"${vision_config}"
grep -Fq 'DSPARK_TOKENS: "3"' <<<"${vision_config}"
grep -Fq 'MAX_NUM_SEQS: "4"' <<<"${vision_config}"
grep -Fq 'MAX_NUM_BATCHED_TOKENS: "4096"' <<<"${vision_config}"
grep -Fq 'GPU_MEMORY_UTILIZATION: ""' <<<"${vision_config}"
grep -Fq 'LMCACHE_MODE: "off"' <<<"${vision_config}"
grep -Fq 'LMCACHE_TRANSFER_MODE: engine_driven' <<<"${vision_config}"
grep -Fq 'LOAD_FORMAT: instanttensor' <<<"${vision_config}"
grep -Fq 'INSTANTTENSOR_BACKEND: BUFFERED' <<<"${vision_config}"
grep -Fq 'jovian-judgement-vllmf66599d-b12x15b6813-fi803c466' \
  <<<"${vision_config}"

grep -Fq 'ARG LMCACHE_BUILD_VERSION=0.5.2+jj.ds4.r6' \
  "${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"
grep -Fq 'LMCACHE_AUTO_TRANSFER_MODE=engine_driven' \
  "${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"

printf 'DeepSeek Jovian Judgement r9 release contract: PASS\n'
