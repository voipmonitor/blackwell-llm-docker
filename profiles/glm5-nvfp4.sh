# GLM-5 NVFP4 with MTP speculative decoding (TP=8)
SGLANG_ENV="SGLANG_ENABLE_SPEC_V2=True SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0 NCCL_GRAPH_FILE=/etc/nccl_graph_opt.xml NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_MIN_NCHANNELS=8 OMP_NUM_THREADS=8 SAFETENSORS_FAST_GPU=1"
SGLANG_ARGS="
  --model-path festr2/GLM-5-NVFP4-MTP
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
  --chunked-prefill-size 16384
  --speculative-algorithm NEXTN
  --speculative-num-steps 4
  --speculative-num-draft-tokens 6
  --speculative-eagle-topk 1
  --host 0.0.0.0
  --port 5000
  --disable-custom-all-reduce
  --enable-metrics
  --attention-backend flashinfer
  --fp4-gemm-backend b12x
  --moe-runner-backend b12x
  --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 16}'
  --json-model-override-args '{\"index_topk_pattern\": \"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS\"}'
  --sleep-on-idle
"
