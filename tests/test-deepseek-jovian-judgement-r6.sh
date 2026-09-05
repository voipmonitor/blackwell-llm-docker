#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-jovian-judgement-cu133-torch213.sh"
compose="${repo_root}/examples/docker-compose-ds4-dspark-jovian-judgement-r6.yml"
composition_root="${repo_root}/patches/releases/jovian-judgement-ds4-r6-qualification"

verify_lock() {
  local component=$1 expected_base=$2 expected_tree=$3
  local lock="${composition_root}/${component}/integration.lock.json"
  local patch="${composition_root}/${component}/integration.patch"

  jq -e --arg base "${expected_base}" --arg tree "${expected_tree}" '
    .schema_version == 1 and
    .composition_strategy == "cherry_pick" and
    .base.commit == $base and
    .result.tree == $tree and
    (.pull_requests | length) == 0 and
    (.source_patches | length) == 1 and
    (.source_patches[0].sha256 | test("^[0-9a-f]{64}$")) and
    (.result.patch_sha256 | test("^[0-9a-f]{64}$"))
  ' "${lock}" >/dev/null
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" |
    sha256sum -c - >/dev/null
}

verify_lock vllm \
  b7e3d033676d5db46fb7d6cdd40d760365a1e239 \
  a67b59a4099457fbcdadce4476c88504fafaf083
verify_lock b12x \
  d27805aef99ae0ad092f79fc458aa1fae1a580e3 \
  aa76f044cbe43c191d33c0c9232e42193b16a544
verify_lock lmcache \
  7ed4675404a31f4ffafd98975899dc83832ba965 \
  86ee2a3bb5675cd3a25b09ad3e2f20dad4720f58

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'revision=r6' <<<"${output}"
grep -Fxq 'vllm_tree=a67b59a4099457fbcdadce4476c88504fafaf083' \
  <<<"${output}"
grep -Fxq 'b12x_tree=aa76f044cbe43c191d33c0c9232e42193b16a544' \
  <<<"${output}"
grep -Fxq 'lmcache_tree=86ee2a3bb5675cd3a25b09ad3e2f20dad4720f58' \
  <<<"${output}"
grep -Fq 'jovian-judgement-vllma67b59a-b12xaa76f04-fi803c466-cu133-torch213-20260905-r6' \
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
grep -Fq 'LOAD_FORMAT: fastsafetensors' <<<"${config}"
grep -Fq 'jovian-judgement-vllma67b59a-b12xaa76f04-fi803c466' <<<"${config}"

grep -Fq 'ARG LMCACHE_BUILD_VERSION=0.5.2+jj.ds4.r6' \
  "${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"
grep -Fq 'LMCACHE_AUTO_TRANSFER_MODE=engine_driven' \
  "${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"

printf 'DeepSeek Jovian Judgement r6 release contract: PASS\n'
