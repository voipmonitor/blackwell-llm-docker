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

### Python scheduler overlay

Status: implemented; GPU qualification is recorded per image in the linked
model runbook. `Dockerfile.glm53-scheduler-overlay` applies scheduler Python
sources and launcher controls to the immutable R28 FP8 image without replacing
CUDA, PyTorch, B12X, FlashKDA, NCCL, or LMCache. It produces three filesystem
layers. The installed Git history remains complete; the incremental bundle
requires the parent image's authenticated source objects.

Extract the parent manifest and prepare the overlay from a clean vLLM checkout:

```bash
git -C vllm-source fetch https://github.com/voipmonitor/vllm.git \
  integration/glm-scheduler-hardening-20260908
git -C vllm-source checkout --detach 9ff42d83938e74018f9c255e8cfa7ca6df6921b0
docker run --rm --entrypoint /bin/cat \
  voipmonitor/vllm@sha256:f5f121e37fd2afbb6f8f036e7eb627435cfb736de0a4420306dc2a25b6631669 \
  /opt/glm53-flash/source.lock > r28.source.lock
uv run --no-project prepare_glm53_scheduler_overlay.py \
  --vllm ./vllm-source --parent-lock ./r28.source.lock \
  --uv "$(command -v uv)" --output ./scheduler-bundles \
  --release-name jovian-judgement-community-20260908-r28.1 \
  --release-version r28.1
bash build_glm53_scheduler_overlay.sh ./scheduler-bundles local/glm53:scheduler
```

The preparer rejects changes outside the scheduler and its tests. The installer
verifies the parent manifest, source trees, input hashes, and FlashKDA binary.
Generated vLLM version metadata identifies the installed Python source; native
component identities remain those in the manifest. A changed source lock
produces safe external-checkpoint misses across image versions. It does not
reinterpret persistent checkpoints written by another runtime identity.

### Scheduler controls

The image retains fixed prefill compute share `0.4`, schedule interval `1`, and
one prefill lane. Interleaving is opt-in. Explicit CLI values override the
corresponding environment variable and are emitted only once.

| Environment | vLLM option | Values |
|---|---|---|
| `PREFILL_COMPUTE_SHARE` | `--prefill-compute-share` | `auto` or finite float strictly between 0 and 1 |
| `PREFILL_COMPUTE_HALF_LIFE` | `--prefill-compute-half-life` | `smooth`, `responsive`, or positive finite seconds; requires auto share |
| `MAX_PARALLEL_PREFILLS` | `--max-parallel-prefills` | `auto` or positive integer |
| `PREFILL_POLICY` | `--prefill-policy` | `round-robin` or `decode-aware` |
| `DECODE_REFILL_TARGET` | `--decode-refill-target` | `auto` or positive integer |

Automatic lane count is `min(4, max_num_seqs)`, independently of attention,
recurrent, or LMCache object geometry. The scheduler token budget still limits
each forward pass. With automatic refill, the refill target follows the
effective lane count. A numeric fixed share rejects any half-life setting.
Compute-share scheduling requires interval `1`.

`FAIRNESS_ENGINE=none` suppresses the inherited environment share; an explicit
CLI share remains authoritative. `FAIRNESS_ENGINE=compute_share` is supported
for compatibility, while `micro_slicing` is rejected. `--help` lists the
controls without loading models. `CACHE_MODE=vram DRY_RUN=1` prints the full
command. Unit tests run with `uv run --no-project --with pytest pytest -q tests`.

### Models and cache

The installed launcher is `/usr/local/bin/serve-glm53-flash.sh`. Model defaults
are `local-inference-lab/GLM-5.3-Flash-NVFP4` and
`local-inference-lab/GLM-5.3-Flash-DFlash2`; revisions are resolved at startup.
The latter checkpoint contains offline MXFP8 draft weights. A changed model or
runtime identity cannot reuse an incompatible external recurrent checkpoint.

The GLM launcher supplies `--default-chat-template-kwargs
'{"reasoning_effort":"high"}'`. A chat request can override that default with
`"reasoning_effort":"max"`, `"high"`, or `"low"`; this does not change
`max_tokens` or the parser. Without a server/request override, the checkpoint's
chat template selects `max`. Direct `vllm serve` commands bypass this launcher's
defaults and must provide the option themselves.

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
