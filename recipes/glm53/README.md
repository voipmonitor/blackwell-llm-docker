# GLM-5.3-Flash source-locked serving image

Status: **implemented; release qualification is tracked separately**.

The build installs complete committed vLLM, B12X and LMCache Git trees over an
immutable CUDA 13.3/PyTorch 2.13 runtime. It compiles LMCache native extensions
and installs a source-authenticated FlashKDA operator with packed recurrent
checkpoint exports. The result has two filesystem layers; serving does not
require source-code bind mounts.

The runtime foundation supplies the compatible PyTorch, CUDA, NCCL,
FlashInfer, InstantTensor and vLLM core native binaries. It is pinned by digest
in both Dockerfiles. Changing that foundation requires ABI and serving
qualification; this recipe does not rebuild those dependencies from scratch.

## Build inputs

Provide clean, complete Git checkouts as `vllm-source`, `b12x-source` and
`lmcache-source`. Use the release's published source refs and revisions, not
moving branch heads when reproducing a measured image. The image's
`/opt/glm53-flash/source.lock` records repository and package trees, build-input
hashes, Git-bundle hashes and native-artifact identity.

Host requirements: Git, Docker Buildx and `uv`; no host PyTorch or active GPU
is needed for image construction. Compilation uses the runtime foundation.

The GLM FP8 checkpoint qualification uses the following published integration
refs. They preserve contributor history and integration resolutions; simply
merging a list of open PRs does not reproduce those resolutions automatically.
These source refs are **implemented**, not by themselves release approval.

| Checkout | Repository | Commit |
|---|---|---|
| `vllm-source` | [voipmonitor/vllm](https://github.com/voipmonitor/vllm/tree/integration/glm-fp8-checkpoint-serving-20260908) | `2531689fa50b956d3e1156e1ab80d119aaf34c1e` |
| `b12x-source` | [voipmonitor/b12x](https://github.com/voipmonitor/b12x/tree/integration/glm-fp8-checkpoint-serving-20260908) | `3edbcbce70f491741b82f5eab9c1b30b39447228` |
| `lmcache-source` | [local-inference-lab/LMCache](https://github.com/local-inference-lab/LMCache/tree/integration/glm-fp8-checkpoint-serving-20260908) | `617a1b47a790de6b86eea92f59deb232a9eff87d` |

Clone the `integration/glm-fp8-checkpoint-serving-20260908` branch of each
repository into its checkout directory, then verify `git rev-parse HEAD`
against the table. Complete checkouts are required; do not use shallow clones
for the source bundles. The package trees and compiled FlashKDA artifact are
reproducible inputs; OCI timestamps and archive metadata are not promised to
produce a bit-identical image digest on a different builder.

From this recipe directory, build the FlashKDA operator:

```bash
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git cutlass-source
docker buildx build \
  --build-context vllm_source=./vllm-source \
  --build-context cutlass_source=./cutlass-source \
  --output type=local,dest=./flashkda-artifact \
  -f Dockerfile.glm53-managed-checkpoint-native .
```

The native build resolves the FlashKDA revision and CUTLASS submodule declared
by vLLM, applies its recorded checkpoint patch, and exports the binary with
source and checksum records. Do not supply an unpatched `FLASH_KDA_SRC_DIR`.

Freeze source bundles and install them:

```bash
uv run --no-project --python 3.12 prepare_glm53_source_bundles.py \
  --vllm ./vllm-source --b12x ./b12x-source --lmcache ./lmcache-source \
  --native-artifact ./flashkda-artifact --uv "$(command -v uv)" \
  --output ./source-bundles \
  --release-name jovian-judgement-community-source-locked \
  --release-version source-locked
bash build_glm53_cache_contract_image.sh \
  ./source-bundles ./flashkda-artifact local/glm53:source-locked
```

The bundle directory must not already exist. Uncommitted serving changes,
bundle/tree mismatches and native-patch mismatches fail the build. Git bundles
preserve contributor history; no source squashing or filesystem overlays are
used to impersonate a committed tree.

`source_locked_image_labels.py` clears inherited serving claims and replaces
them from the authenticated manifest. It preserves metadata for unchanged
runtime dependencies. An implementation label is not evidence of qualification.

Run the CPU-only installation and metadata contracts with:

```bash
uv run --no-project --with pytest python -m pytest -q tests/test_build_contract.py
```

## Serving contract

The installed launcher is `/usr/local/bin/serve-glm53-flash.sh`. Model defaults
are `local-inference-lab/GLM-5.3-Flash-NVFP4` and
`local-inference-lab/GLM-5.3-Flash-DFlash2`; revisions are resolved at startup.
The latter checkpoint contains offline MXFP8 draft weights. A changed model or
runtime identity cannot reuse an incompatible external recurrent checkpoint.

The production scheduler budget is 4096 tokens, OMP threads1, NCCL channels16
with 2 MiB buffers, and `FULL_AND_PIECEWISE` CUDA graphs. GPU-local attention
pages default to2048 tokens. Target operations use B12X; FlashKDA supplies
checkpoint-producing recurrent prefill. The private MTP vocabulary projection
is NVFP4, while the target vocabulary projection remains BF16.

Modes use `SPECULATOR=mtp MTP_DEPTH=0`, `SPECULATOR=mtp MTP_DEPTH=3`, or
`SPECULATOR=dflash2 DFLASH_DEPTH=7`. External cache is opt-in with
`CACHE_MODE=lmcache`; its default transfer is `engine_driven`. Semantic
checkpoints use immutable all-rank storage and asynchronous shared-memory
copies performed by vLLM workers, without a sidecar CUDA context.
Checkpoint payload namespaces are scoped to rank and storage group so that
unlocked pages remain evictable under RAM pressure. Complete manifests and
transfer leases enforce atomic retrieval. Version-1 semantic payloads are
incompatible with these version-2 keys and produce safe cache misses.

Mode-specific deployment instructions and measured conditions belong in the
[GLM-5.3-Flash wiki](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md).
Qwen source may be included in the selected Git trees, but this launcher serves
GLM. A source build alone establishes neither Qwen runtime qualification nor
NVFP4 target-KV qualification.
