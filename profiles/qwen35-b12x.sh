# Qwen3.5-397B-A17B NVFP4 with b12x MoE backend (TP=4)
SGLANG_ENV="NCCL_P2P_LEVEL=SYS SGLANG_ENABLE_SPEC_V2=True"
SGLANG_ARGS="
  --model lukealonso/Qwen3.5-397B-A17B-NVFP4
  --served-model-name Qwen3.5
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --tensor-parallel-size 4
  --quantization modelopt_fp4
  --kv-cache-dtype fp8_e4m3
  --trust-remote-code
  --cuda-graph-max-bs 64
  --max-running-requests 64
  --chunked-prefill-size 16384
  --speculative-algo NEXTN
  --speculative-num-steps 5
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6
  --mamba-scheduler-strategy extra_buffer
  --mem-fraction-static 0.93
  --host 0.0.0.0
  --port 5000
  --enable-pcie-oneshot-allreduce
  --enable-pcie-oneshot-allreduce-fusion
  --enable-metrics
  --schedule-conservativeness 0.1
  --attention-backend flashinfer
  --fp4-gemm-backend b12x
  --moe-runner-backend b12x
  --sleep-on-idle
"
