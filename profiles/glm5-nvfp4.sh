# GLM-5 NVFP4 with MTP speculative decoding (TP=8)
SGLANG_ENV="SGLANG_ENABLE_SPEC_V2=True"
SGLANG_ARGS="
  --model festr2/GLM-5-NVFP4-MTP
  --served-model-name glm-5
  --reasoning-parser glm45
  --tool-call-parser glm47
  --tensor-parallel-size 8
  --quantization modelopt_fp4
  --kv-cache-dtype bf16
  --trust-remote-code
  --cuda-graph-max-bs 32
  --max-running-requests 64
  --mem-fraction-static 0.85
  --speculative-algo NEXTN
  --speculative-num-steps 5
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6
  --host 0.0.0.0
  --port 5000
  --disable-custom-all-reduce
  --enable-flashinfer-allreduce-fusion
  --enable-metrics
  --attention-backend flashinfer
  --fp4-gemm-backend b12x
  --moe-runner-backend b12x
"
