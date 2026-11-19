#!/usr/bin/env bash
# JAX vllm-tpu, GPT-OSS 120B MXFP4 on one 4-chip TPU host (v5p or v7x), measured with `vllm bench serve` the same way
# as the section C runs (random fixed-length prompts, by default 1,024 tokens in and 1,024 out, EOS ignored), with
# 2 prompts per concurrent request and at least 16 prompts per load point.
#
#   CHIP=v5p TP=2 DP=2 MAX_NUM_SEQS=128 MOE_REQUANTIZE_WEIGHT_DTYPE=int8 MOE_REQUANTIZE_BLOCK_SIZE=32 OUT=./results/v5p_tp2_dp2 bash run_vllm_tpu_jax_v5p_vs_v7x.sh
#   CHIP=v7x TP=2 DP=4 MAX_NUM_SEQS=128 OUT=./results/v7x_tp2_dp4 bash run_vllm_tpu_jax_v5p_vs_v7x.sh
#
# MAX_NUM_SEQS is the running-request limit per replica (DP = number of replicas). A v5p chip shows up as one device,
# a v7x chip as two (one per TensorCore), so TP x DP is 4 on v5p and 8 on v7x.
# CHIP=v7x also sets the single-host TPU environment for a node that belongs to a multi-host node pool.
set -uo pipefail
CHIP="${CHIP:?set CHIP to v5p or v7x}"
MODEL_PATH="${MODEL_PATH:-/models/models/openai/gpt-oss-120b}"
if [ "$CHIP" = "v7x" ]; then
  TP="${TP:-2}"; DP="${DP:-4}"; MEM="${MEM:-0.86}"
  export MEGASCALE_COORDINATOR_ADDRESS='' MEGASCALE_NUM_SLICES='' MEGASCALE_PORT='' MEGASCALE_SLICE_ID=''
  export TPU_PROCESS_ADDRESSES='' TPU_PROCESS_PORT='' TPU_WORKER_HOSTNAMES='' TPU_WORKER_ID=0
  export TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2x1 TPU_ACCELERATOR_TYPE=tpu7x-8 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
  # GKE lists the node's 4 chips in TPU_VISIBLE_CHIPS; vLLM's data-parallel launcher then indexes 8 TensorCores into
  # a 4-entry list and fails, so leave it unset.
  unset TPU_VISIBLE_CHIPS
  MULTIPROCESS_DP=1
else
  TP="${TP:-4}"; DP="${DP:-1}"; MEM="${MEM:-0.90}"
  # The multi-process data-parallel launcher in this build assumes two devices per chip (true on v7x). A v5p chip is
  # one device, so it hands each replica the wrong chips and the TPU slice fails to start. Use single-process
  # data parallelism instead.
  MULTIPROCESS_DP=0
  # The MoE kernel multiplies BF16 activations by FP4 expert weights, which the v5p matrix unit does not support
  # ("Bad lhs/rhs type ... f4E2M1FN"). On v5p, requantize the experts at load time to a type it supports, e.g.
  #   MOE_REQUANTIZE_WEIGHT_DTYPE=int8 MOE_REQUANTIZE_BLOCK_SIZE=32
  # The block count along the 2,880-wide dimension must divide by TP (32 -> 90 blocks: TP=1 or 2; 256 -> 12: TP=4).
fi
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"; KV_DTYPE="${KV_DTYPE:-fp8}"
LOADS="${LOADS:-1 8 32 64 128 256}"
INPUT_LEN="${INPUT_LEN:-1024}"; OUTPUT_LEN="${OUTPUT_LEN:-1024}"
OUT="${OUT:?set OUT}"
export TPU_BACKEND_TYPE=jax MODEL_IMPL_TYPE=vllm TPU_MULTIPROCESS_DP="$MULTIPROCESS_DP" USE_BATCHED_RPA_KERNEL=1
export JAX_GC_POST_COMPILE=true JAX_THREE_PHASE_DEALLOCATION=true
export VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_XLA_CACHE_PATH=/dev/shm/xla-cache
mkdir -p "$OUT" /dev/shm/xla-cache
rm -f "$OUT/FAILED" "$OUT/server_failed.log" "$OUT/DONE"
{ python3 -c "import vllm; print('vllm', vllm.__version__)"; pip list 2>/dev/null | grep -E '^(jax|jaxlib|libtpu|tpu.inference) '; \
  echo "CHIP=$CHIP TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE:-unset} TPU_MULTIPROCESS_DP=$MULTIPROCESS_DP TP=$TP DP=$DP MAX_NUM_SEQS=$MAX_NUM_SEQS KV_DTYPE=$KV_DTYPE MEM=$MEM"; \
  echo "MOE_REQUANTIZE_WEIGHT_DTYPE=${MOE_REQUANTIZE_WEIGHT_DTYPE:-} MOE_REQUANTIZE_BLOCK_SIZE=${MOE_REQUANTIZE_BLOCK_SIZE:-}"; } > "$OUT/version.txt" 2>&1
python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 --port=8000 \
  --model="$MODEL_PATH" --served-model-name=openai/gpt-oss-120b \
  --tensor-parallel-size="$TP" --data-parallel-size="$DP" \
  --max-model-len=32768 --max-num-batched-tokens=16384 --max-num-seqs="$MAX_NUM_SEQS" \
  --enable-chunked-prefill --no-enable-prefix-caching --trust-remote-code \
  --kv-cache-dtype="$KV_DTYPE" --async-scheduling --gpu-memory-utilization="$MEM" \
  > /dev/shm/server.log 2>&1 &
SRV=$!
T0=$(date +%s)
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "server exited during startup" > "$OUT/FAILED"; tail -n 400 /dev/shm/server.log > "$OUT/server_failed.log"; exit 1
  fi
  sleep 10
done
echo "startup_s $(( $(date +%s) - T0 ))" >> "$OUT/version.txt"
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "openai/gpt-oss-120b", "prompt": "Compute 37 * 43 step by step:", "max_tokens": 64, "temperature": 0}' > "$OUT/sanity.json"
# The benchmark client must not touch the TPU that the server holds.
export JAX_PLATFORMS=cpu TPU_SKIP_MDS_QUERY=true TPU_VISIBLE_CHIPS="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bench() {  # isl osl concurrency prompts tag
  vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model openai/gpt-oss-120b --tokenizer "$MODEL_PATH" --trust-remote-code \
    --dataset-name random --random-input-len "$1" --random-output-len "$2" \
    --num-prompts "$4" --request-rate inf --max-concurrency "$3" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --seed 42 \
    --save-result --result-dir "$OUT" --result-filename "$5.json" > "$OUT/$5.client.log" 2>&1
}
bench 1024 1024 8 16 warmup_isl1024_osl1024_c8_n16
kt() { if [ $(( $1 % 1024 )) -eq 0 ]; then echo "$(( $1 / 1024 ))k"; else echo "$(( $1 / 1000 ))k"; fi; }  # 1024->1k, 22000->22k
K="$(kt "$INPUT_LEN")$(kt "$OUTPUT_LEN")"
for c in $LOADS; do
  n=$((2 * c)); [ "$n" -lt 16 ] && n=16
  bench "$INPUT_LEN" "$OUTPUT_LEN" "$c" "$n" "ref_${K}_isl${INPUT_LEN}_osl${OUTPUT_LEN}_c${c}_n${n}"
done
tail -n 400 /dev/shm/server.log > "$OUT/server_tail.log"
kill "$SRV"
echo done > "$OUT/DONE"
