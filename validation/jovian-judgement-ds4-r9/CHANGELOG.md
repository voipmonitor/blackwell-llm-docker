# DeepSeek V4 Jovian Judgement r8 To r9

This is a historical release comparison, not the serving specification.
The image identities and measurements are recorded in [README.md](README.md).

## Correctness Changes

| Change | Practical effect | Evidence |
|---|---|---|
| B12X clears masked shared-memory value rows before compressed-attention MMA | NaN bytes in unused KV pages cannot contaminate valid output through zero attention probabilities. The prefill dispatcher also preserves the requested compute mode and compressed RoPE source. | Eight unused-page cases fail on r8; all 20 attention tests pass on r9. [B12X cfce3d64](https://github.com/local-inference-lab/b12x/commit/cfce3d64587dd4cff2e35acbed21dde29f98c701) |
| Auxiliary-stream outputs retain their consuming-stream allocator lifetime | A returned tensor cannot be recycled while the caller still reads it. Nested lists, tuples, and dictionaries are supported without disabling overlap or graph capture. | The r8 helper fails two lifetime cases; the corrected helper passes all six. [vLLM #695](https://github.com/local-inference-lab/vllm/pull/695) |
| Graph memory is counted once in post-capture KV recommendations | Estimated graph allocations no longer remain inside the activation figure when actual graph allocations are added. Initial KV admission is unchanged. | Fourteen focused tests, including direct post-capture recommendation assertions. [vLLM #694](https://github.com/local-inference-lab/vllm/pull/694) |
| Engine-driven LMCache retains locally pinned buffers after partial copy submission errors | A failed copy submission cannot release a buffer still referenced by an earlier queued transfer. | Nine asynchronous-copy lifetime tests. [LMCache #55](https://github.com/local-inference-lab/LMCache/pull/55/commits/078199fa3e08a0a515eae738194d29119d523b2f) |

The two reported multi-hour failure classes remain **research-only**:
Vision query-projection address errors and text-serving physical-memory
exhaustion. None has a locally reproduced production root cause. In particular,
the `compress_ratio=128` attention branch returns no tensor from its auxiliary
compressor callback, so the output-lifetime correction cannot be claimed as
a demonstrated fix for it.

## Source Integration

- vLLM base advances from `7d66922a` to `2a979314`. It includes loader-owned
  custom weight allocations and fences before model-internal preparation.
  The qualified DS4 loader remains InstantTensor BUFFERED, not the B12X loader.
- B12X base advances from `a1bbd027` to `06b4de7c`. It includes the attention
  correction above, packed checkpoint slice handling, capacity-based block-FP8
  dispatch, and MXFP8 scale-initialization elimination. These component changes
  are not independently attributed an end-to-end speedup.
- Generation-safe TP2 peer-push PR #246 is included through B12X master rather
  than applied twice. The r8 image already included an earlier head of that PR;
  peer-push itself is not introduced by r9.
- LMCache base stays at `7ed46754`; PR #55 advances to `078199fa` for the
  partial-submission lifetime case. PRs #49, #50, #51, and #56 remain included.
- vLLM PRs #628, #630, #634, #553, #671, and #679 and B12X PR #301 remain
  included. Maintainer review of open PRs is separate from image publication.

Complete canonical-base commit comparisons:
[vLLM](https://github.com/local-inference-lab/vllm/compare/7d66922a7bf0c9c7efe9a35a87df128f6b24e762...2a979314dc97b03173a0a76fc15664ec924db32b),
[B12X](https://github.com/local-inference-lab/b12x/compare/a1bbd02781c7505754e7aa58a959c1a77891c690...06b4de7c723e6f166d65abf5909c5b7d0f8acc68).
These bases contain other models' changes; DS4 qualification is not GLM or
Qwen qualification.

## Deployment Compatibility

CUDA 13.3, PyTorch 2.13, XGrammar 0.2.5, InstantTensor BUFFERED, fixed
probabilistic K5 for text and K3 for Vision, and GPU-only caching by default
are unchanged. Engine-driven LMCache remains opt-in. No fixed additional
GPU reserve, sampling change, or precision-reducing workaround is added.

Repeated image-prefix checks pass on both images when the prompt contains
complete reusable KV blocks. A 219-token request is shorter than the
256-token block and cannot establish a GPU-prefix-cache defect. No image-cache
patch was added for that report.

The benchmark receipts show no regression in the tested CC1 and approximately
32k-prefill controls. Sampling and draft acceptance vary; consult the serving
specification's conditions before interpreting token-rate differences.
