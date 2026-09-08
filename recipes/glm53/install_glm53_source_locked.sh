#!/usr/bin/env bash
set -euo pipefail

# Build-time mounts: /source-bundles, /build-inputs, /lmcache-artifacts,
# /flashkda-artifacts. No serving source-code mount is required at runtime.
readonly lock=/source-bundles/source.lock
value() { awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; found=1 } END { if (!found) exit 1 }' "$lock"; }
verify() { test "$(sha256sum "$1" | cut -d' ' -f1)" = "$2"; }
verify "$lock" "${SOURCE_LOCK_SHA256:?Supply the frozen lock hash}"
test "${CACHE_FINGERPRINT:?Supply the frozen cache namespace}" = "$(value runtime.cache.fingerprint)"
verify /source-bundles/uv "$(value build.uv.sha256)"
for name in vllm b12x lmcache; do
    verify "/source-bundles/$name.bundle" "$(value "$name.bundle.sha256")"
done
while IFS='=' read -r key checksum; do
    if [[ $key == input.*.sha256 ]]; then
        filename=${key#input.}; filename=${filename%.sha256}
        verify "/build-inputs/$filename" "$checksum"
    fi
done < "$lock"
verify /flashkda-artifacts/_flashkda_C.abi3.so "$(value flashkda.extension.sha256)"
test "$(cat /flashkda-artifacts/flashkda-base.commit)" = "$(value flashkda.base.commit)"
test "$(awk '{print $1}' /flashkda-artifacts/flashkda-patch.sha256)" = "$(value flashkda.patch.sha256)"

for name in vllm b12x lmcache; do
    if [[ $name == lmcache ]]; then destination=/opt/lmcache/source; else destination=/opt/glm53-flash/$name; fi
    bash /build-inputs/install_source_bundle.sh "/source-bundles/$name.bundle" \
        "$(value "$name.commit")" "$(value "$name.tree")" "$destination"
done
verify /opt/glm53-flash/vllm/cmake/external_projects/patches/flashkda-packed-checkpoints.patch \
    "$(value flashkda.patch.sha256)"

readonly python=/opt/venv/bin/python
readonly uv=/source-bundles/uv
readonly lmcache_package=/opt/venv/lib/python3.12/site-packages/lmcache
"$uv" run --no-project --python "$python" "$python" \
    /build-inputs/install_vllm_source_version.py /opt/glm53-flash/vllm \
    "$(value vllm.version)"
torch_before=$("$uv" run --no-project --python "$python" "$python" -c \
    'import torch; print(f"{torch.__version__}|{torch.version.cuda}|{int(torch._C._GLIBCXX_USE_CXX11_ABI)}")')
test "$torch_before" = '2.13.0|13.3|1'
"$uv" pip install --python "$python" --no-deps --requirement \
    /build-inputs/glm53-r18-lmcache-runtime-requirements.txt
"$uv" pip install --python "$python" --no-deps --reinstall /lmcache-artifacts/*.whl
# Wheels omit some package data. Install the complete committed package, while
# retaining the native extensions compiled in the dedicated build stage.
cp -a /opt/lmcache/source/lmcache/. "$lmcache_package/"
GIT_WORK_TREE=/opt/venv/lib/python3.12/site-packages \
    git --git-dir=/opt/lmcache/source/.git diff --quiet HEAD -- lmcache
install -Dm755 /flashkda-artifacts/_flashkda_C.abi3.so \
    /opt/glm53-flash/vllm/vllm/_flashkda_C.abi3.so
install -Dm755 /flashkda-artifacts/_flashkda_C.abi3.so \
    /opt/venv/lib/python3.12/site-packages/vllm/_flashkda_C.abi3.so
install -Dm755 /lmcache-artifacts/liblmcache_cumem_shareable.so \
    /opt/lmcache/lib/liblmcache_cumem_shareable.so
install -Dm644 /lmcache-artifacts/SHA256SUMS /opt/lmcache/native-artifacts.sha256

install -Dm755 /build-inputs/serve-glm53-flash-nvfp4-dflash2.sh /usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh
install -Dm755 /build-inputs/serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh /usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh
install -Dm755 /build-inputs/serve-glm53-flash-lmcache.sh /usr/local/bin/serve-glm53-flash-lmcache.sh
install -Dm755 /build-inputs/serve-glm53-flash-lmcache-cache-complete.sh /usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh
install -Dm755 /build-inputs/serve-glm53-flash-cache-complete.sh /usr/local/bin/serve-glm53-flash.sh
install -Dm755 /build-inputs/glm53_checkpoint_identity.py /usr/local/libexec/glm53_checkpoint_identity.py
install -Dm644 "$lock" /opt/glm53-flash/source.lock
CUDA_VISIBLE_DEVICES= "$uv" run --no-project --python "$python" "$python" -c '
import importlib.metadata as metadata
import torch, vllm, b12x, vllm._flashkda_C
import vllm.vllm_flash_attn.layers.rotary, vllm.third_party.triton_kernels.topk
import lmcache.cuda_ops, lmcache.lmcache_native, lmcache.lmcache_fs
from lmcache.integration.vllm.recurrent_checkpoint_connector import LMCacheRecurrentCheckpointConnector
assert (torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI) == ("2.13.0", "13.3", True)
print("LMCache wheel:", metadata.version("lmcache"))
print("Source imports:", vllm.__file__, b12x.__file__)
'
# The foundation exposes both venv and system site-packages. Check the exact
# interpreter search path used by serving, including the system Torch package.
"$uv" run --no-project --python "$python" "$python" -m pip check
for name in vllm b12x lmcache; do
    if [[ $name == lmcache ]]; then destination=/opt/lmcache/source; else destination=/opt/glm53-flash/$name; fi
    test "$(git -C "$destination" rev-parse HEAD)" = "$(value "$name.commit")"
    test "$(git -C "$destination" rev-parse 'HEAD^{tree}')" = "$(value "$name.tree")"
    git -C "$destination" diff --quiet HEAD
done
