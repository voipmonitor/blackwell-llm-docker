#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-jovian-judgement-cu133-torch213.sh"
compose="${repo_root}/examples/docker-compose-ds4-dspark-jovian-judgement-r8.yml"
composition_root="${repo_root}/patches/releases/jovian-judgement-ds4-r8"

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
  7d66922a7bf0c9c7efe9a35a87df128f6b24e762 \
  8baad5e80636122c540eb659b54a8e89911abb7b \
  '[628,630,634,553,671,679]'
verify_lock b12x \
  a1bbd02781c7505754e7aa58a959c1a77891c690 \
  e58515a63b7b5d15bbc523258e1e338f49698ce3 \
  '[246,301]'
verify_lock lmcache \
  7ed4675404a31f4ffafd98975899dc83832ba965 \
  86ee2a3bb5675cd3a25b09ad3e2f20dad4720f58 \
  '[49,50,51,55,56]'

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'revision=r8' <<<"${output}"
grep -Fxq 'vllm_tree=8baad5e80636122c540eb659b54a8e89911abb7b' \
  <<<"${output}"
grep -Fxq 'b12x_tree=e58515a63b7b5d15bbc523258e1e338f49698ce3' \
  <<<"${output}"
grep -Fxq 'lmcache_tree=86ee2a3bb5675cd3a25b09ad3e2f20dad4720f58' \
  <<<"${output}"
grep -Fq 'jovian-judgement-vllm8baad5e-b12xe58515a-fi803c466-cu133-torch213-20260906-r8' \
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
grep -Fq 'jovian-judgement-vllm8baad5e-b12xe58515a-fi803c466' <<<"${config}"

grep -Fq 'ARG LMCACHE_BUILD_VERSION=0.5.2+jj.ds4.r6' \
  "${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"
grep -Fq 'LMCACHE_AUTO_TRANSFER_MODE=engine_driven' \
  "${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"

printf 'DeepSeek Jovian Judgement r8 release contract: PASS\n'
