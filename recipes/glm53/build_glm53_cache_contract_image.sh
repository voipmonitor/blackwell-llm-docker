#!/usr/bin/env bash
set -euo pipefail

if (($# != 3)); then
    echo "usage: $0 <source-bundle-directory> <FlashKDA-artifact-directory> <local-image-tag>" >&2
    exit 2
fi
readonly bundles=$(realpath -e "$1") native=$(realpath -e "$2") tag=$3
readonly directory=$(cd -- "$(dirname -- "$0")" && pwd)
readonly lock="$bundles/source.lock"
value() { awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {if (!found) exit 1}' "$lock"; }
readonly fingerprint=$(value runtime.cache.fingerprint)
readonly lock_sha=$(sha256sum "$lock" | cut -d' ' -f1)
labels=()
mapfile -d '' -t source_labels < <(
    docker image inspect "$(value runtime.base.image)" | \
        "$bundles/uv" run --no-project --python 3.12 "$directory/source_locked_image_labels.py" \
        --source-lock "$lock"
)
(( ${#source_labels[@]} > 0 )) || { echo 'Image metadata generation failed.' >&2; exit 1; }
for label in "${source_labels[@]}"; do labels+=(--label "$label"); done
docker buildx build --load --progress plain --tag "$tag" \
    --build-arg "SOURCE_LOCK_SHA256=$lock_sha" \
    --build-arg "LMCACHE_COMMIT=$(value lmcache.commit)" \
    --build-arg "LMCACHE_TREE=$(value lmcache.tree)" \
    --build-arg "LMCACHE_BUNDLE_SHA256=$(value lmcache.bundle.sha256)" \
    --build-arg "LMCACHE_VERSION=$(value lmcache.version)" \
    --build-arg "UV_SHA256=$(value build.uv.sha256)" \
    --build-arg "CACHE_FINGERPRINT=$fingerprint" "${labels[@]}" \
    --build-context "source_bundles=$bundles" --build-context "native_artifact=$native" \
    -f "$directory/Dockerfile.glm53-cache-contracts" "$directory"
test "$(docker image inspect --format '{{len .RootFS.Layers}}' "$tag")" = 2
test "$(docker image inspect --format '{{index .Config.Labels "local-inference.runtime.source-lock.sha256"}}' "$tag")" = "$lock_sha"
docker image inspect --format '{{.Id}}' "$tag"
