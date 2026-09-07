# DeepSeek V4 TP2 Runtime Qualification

This directory records operator correctness, serving-memory, prefix-reuse,
and bounded performance checks for the following image:

```text
voipmonitor/vllm:jovian-judgement-vllmf66599d-b12x15b6813-fi803c466-cu133-torch213-20260907-r9
image ID: sha256:b0b8af509dc8c3990a10c738ea0a2240a22e52af41af6c272e3945512b236139
registry digest: sha256:5bea088597980b299a1df8a6f3fc6d2d22c723276088ea8583b456f27043c0cd
```

The source locks are under
`patches/releases/jovian-judgement-ds4-r9/{vllm,b12x,lmcache}/`.
The deployment specification is
[DS4 Jovian Judgement r9](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4-jovian-judgement-r9.md).

## Conditions And Limits

- Hardware: GPUs 0/1, two 96 GiB RTX PRO 6000 Blackwell Workstation GPUs,
  600 W limits, same PCIe switch. Direct-root PCIe 4 is not qualified here.
- Text checkpoint: `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`, fixed probabilistic K5,
  MNS8, MNB4096, CUDA graph cap 48.
- Vision checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
  `6821d6ad3681a4b137b066b76094fa82ebd0a380`, fixed probabilistic K3,
  MNS4, MNB4096, CUDA graph cap 16.
- Both profiles: TP2/DCP1, 1,048,576-token request limit,
  `BACKEND=b12x-a8-dglin`, FP8 compressed KV, InstantTensor BUFFERED,
  GPU prefix caching, FULL target/decode and DSpark capture, PIECEWISE prefill.
- GPU-only utilization is 0.975; engine-driven LMCache uses 0.970.
- Synthetic token-ID workloads test execution and allocation contracts,
  not semantic answer quality. Passing them does not close a multi-hour
  production crash report. No production conversation or reporter archive
  is included.

The r8 control is
`voipmonitor/vllm:jovian-judgement-vllmd267ca7-b12xe58515a-fi803c466-cu133-torch213-20260906-r8`,
image ID `sha256:7e533ec88bcd04979f05c737768f75de0fb16fa80391c1b5d104565092f85579`.

## Artifact Map

| Artifact | Contract and result |
|---|---|
| `b12x-unused-page-r8.xml` | Eight expected reference failures: unused KV NaNs reach logical outputs |
| `b12x-compressed-mla-r9.xml` | Twenty passing attention tests, including those eight cases |
| `vllm-worker-stream-r9.xml` | Fourteen worker-accounting and six auxiliary-lifetime cases pass |
| `aux-stream-cuda-r8.xml`, `aux-stream-cuda-patched.xml` | Two reference lifetime failures; six passes with only the corrected helper mounted |
| `lmcache-copy-lifetime-r9.xml` | Nine asynchronous host-copy lifetime cases pass |
| `text-r{8,9}-c1-t1.json` | Sequential 30-second CC1 controls, temperature 1 |
| `vision-r{8,9}-c1-t1.json` | Sequential 30-second CC1 controls, temperature 1 |
| `text-r{8,9}-prefill32k-t1.json` | Four cold-prefill samples per image; approximately 32k actual tokens |
| `{text,vision}-r9-memory.json` | Exact 4,096- and 810,000-token memory-admission requests |
| `vision-r9-exact-resume-c1.json` | Six continuations: 270,731 prompt, 267,008 cache hits, 4,096 output tokens |
| `vision-r{8,9}-variable-resumes.json` | One prime plus 96 continuations with exactly 267,008 cache hits |
| `vision-r8-image-first-prefix-cache.json`, `vision-r9-image-prefix-cache.json` | Repeated 4,049-token image prompts yield 3,840 cached tokens |
| `text-r9-chat-smoke.json` | Correct country answer with a completed response |
| `text-r9-strict-tool-smoke.json` | Required strict tool call emits valid `weather(city="Tallinn")` arguments |
| `lmcache-r9-{memory,eviction,restore}.json` | 810k store, independent 400k eviction pressure, and successful host-tier restore |
| `lmcache-r9-{before,after}-restore{,-vllm}.metrics` | Zero local hits; 4,095 externally transferred tokens after restore |
| `lmcache-r9-transfer.log`, `lmcache-r9-status.json` | Both ranks retrieve a 4,096-token block; engine-driven 24 GiB shared-memory pool |

The benchmark's hybrid-KV token estimate is not the serving capacity. Use the
vLLM startup pool size in the deployment specification. Compare tokens/s,
steps/s, and draft acceptance together; these short trials do not establish
a statistically significant kernel speedup.

## Reproduction Commands

Start one profile from `examples/docker-compose-ds4-*-jovian-judgement-r9.yml`.
The following clients run outside vLLM and do not modify the server. Run
workloads sequentially; overlapping clients invalidate the throughput control.
Set `BASE` to the server URL and `MODEL` to its `/v1/models` identifier.

Memory admission:

```bash
python scripts/qualify_ds4_memory_admission.py \
  --base-url "$BASE" --model "$MODEL" --gpu-ids 0,1 \
  --long-prefill-tokens 810000 --output memory.json
```

Add `--vision` only on the Vision checkpoint. This submits ten 2048px images
during the long prefill. The memory client uses temperature 0 and one output
token to isolate execution, not to assess temperature-zero output quality.

Exact reported prefix-resume geometry:

```bash
python scripts/reproduce_ds4_prefix_resume.py \
  --base-url "$BASE" --model "$MODEL" --gpu-ids 0,1 \
  --prefix-tokens 267264 --expected-cached-tokens 267008 \
  --resume-tokens 3467 --concurrency 1 --waves 6 --output-tokens 4096 \
  --output prefix-resume.json
python validation/jovian-judgement-ds4-r9/probe_variable_prefix_resumes.py \
  --base-url "$BASE" --model "$MODEL" --gpu-ids 0,1 \
  --output variable-resumes.json
```

Image cache: install `requests` in the client environment and pass the
checkpoint's `inference/examples/images/corn.jpeg` file:

```bash
python validation/jovian-judgement-ds4-r9/probe_image_prefix_cache.py \
  --base "$BASE" --model "$MODEL" --image "$IMAGE_PATH" \
  --output image-prefix-cache.json
```

CC1 and cold prefill use
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
version 0.6.2, commit `ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f`:

```bash
python llm_decode_bench.py --host http://127.0.0.1 --port 8158 \
  --model "$MODEL" --concurrency 1 --contexts 0 --duration 30 \
  --max-tokens 16384 --temperature 1 --skip-prefill \
  --no-resume --no-hw-monitor --output c1.json
python llm_decode_bench.py --host http://127.0.0.1 --port 8158 \
  --model "$MODEL" --concurrency 1 --contexts 0 --prefill-only \
  --prefill-contexts 32768 --prefill-duration 10 --temperature 1 \
  --no-resume --no-hw-monitor --output prefill32k.json
```

The synthetic-prefix model fields use test aliases; they do not identify
different weights. Checkpoint revisions and source identities above define
the controls.

## Host-Tier Restoration

Start the text Compose with `LMCACHE_MODE=ram LMCACHE_L1_GB=24`. Run the
memory command with token ID 100 and 810,000 long-prefill tokens, then token
ID 200 and 400,000 long-prefill tokens. These independent histories exceed
the qualified GPU pool together. Repeat token ID 100 with
`--long-prefill-tokens 0` to request the original 4,096-token prefix.

Verify the counters before and after that request. A local prefix hit is not
evidence of host restoration. In the qualified run, local cache hits remain
zero; `external_prefix_cache_hits_total` and the `external_kv_transfer` prompt
source each increase by 4,095. LMCache retrieves all 4,096 tokens for both
ranks; vLLM recomputes the last token. The complete request takes 1.60 seconds,
including first-use indexer compilation, while each rank's logged retrieval
is 0.006 seconds. This is a functionality check, not a warmed transfer benchmark.
