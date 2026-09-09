#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
  echo "usage: $0 <source-bundle-directory> <image-tag>" >&2
  exit 2
fi
readonly bundles=$(realpath -e "$1") tag=$2
readonly recipe=$(cd -- "$(dirname -- "$0")" && pwd)
readonly lock="$bundles/source.lock"
value() { awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {if (!found) exit 1}' "$lock"; }
readonly parent=$(value runtime.parent.release-image)
readonly lock_sha=$(sha256sum "$lock" | cut -d' ' -f1)
labels=()
mapfile -d '' -t source_labels < <(
  docker image inspect "$parent" | "$bundles/uv" run --offline --no-project \
    --python 3.12 "$recipe/source_locked_image_labels.py" --source-lock "$lock"
)
((${#source_labels[@]} > 0)) || { echo 'Image label generation failed.' >&2; exit 1; }
for label in "${source_labels[@]}"; do labels+=(--label "$label"); done
labels+=(--label "local-inference.runtime.parent-image=$parent"
  --label "local-inference.runtime.default.reasoning-effort=high"
  --label "local-inference.vllm.version=$(value vllm.version)"
  --label 'local-inference.scheduler.max-parallel-prefills=1'
  --label 'local-inference.release.rootfs-format=two-layer FP8 runtime plus Python scheduler and launcher overlay')
docker buildx build --load --progress plain --tag "$tag" \
  --build-arg "SOURCE_LOCK_SHA256=$lock_sha" \
  --build-arg "CACHE_FINGERPRINT=$(value runtime.cache.fingerprint)" \
  "${labels[@]}" --build-context "source_bundles=$bundles" \
  -f "$recipe/Dockerfile.glm53-scheduler-overlay" "$recipe"
test "$(docker image inspect --format '{{len .RootFS.Layers}}' "$tag")" = 3
test "$(docker image inspect --format '{{index .Config.Labels "local-inference.runtime.source-lock.sha256"}}' "$tag")" = "$lock_sha"
docker image inspect --format '{{.Id}}' "$tag"
