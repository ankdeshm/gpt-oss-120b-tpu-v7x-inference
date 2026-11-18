#!/usr/bin/env bash
# JAX vllm-tpu, GPT-OSS 120B MXFP4 on one v7x host (4 chips), measured with `vllm bench serve`.
# Server settings are the single-host MXFP4 settings from section B (TP=2, DP=4, FP8 KV cache, async scheduling);
# only the running-request limit per replica (MAX_NUM_SEQS), the prompt and output lengths (INPUT_LEN, OUTPUT_LEN) and
# the load points (LOADS) change between runs:
#
#   bash run_vllm_tpu_jax_high_load.sh                                      # 64 per replica (section B settings)
#   MAX_NUM_SEQS=128 LOADS="256 512 1024" bash run_vllm_tpu_jax_high_load.sh
#   MAX_NUM_SEQS=512 LOADS="512 1024 2048" bash run_vllm_tpu_jax_high_load.sh
#   MAX_NUM_SEQS=128 INPUT_LEN=8192 LOADS="64 128 256 512" bash run_vllm_tpu_jax_high_load.sh
#   MAX_NUM_SEQS=128 INPUT_LEN=22000 OUTPUT_LEN=4000 LOADS="8 16 32 64" bash run_vllm_tpu_jax_high_load.sh
#
# Each load point sends 2x as many prompts as there are concurrent requests (INPUT_LEN tokens in, OUTPUT_LEN out).
# Run inside a vllm/vllm-tpu container on a v7x host. If the node belongs to a multi-host node pool, the exports at
# the top make the runtime use only this host's 4 chips.
set -uo pipefail
MODEL_PATH="${MODEL_PATH:-/models/models/openai/gpt-oss-120b}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
LOADS="${LOADS:-64 128 256 512}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
OUT="${OUT:-./results/vllm_tpu_jax_isl${INPUT_LEN}_${MAX_NUM_SEQS}_per_replica}"
export MEGASCALE_COORDINATOR_ADDRESS='' MEGASCALE_NUM_SLICES='' MEGASCALE_PORT='' MEGASCALE_SLICE_ID=''
export TPU_PROCESS_ADDRESSES='' TPU_PROCESS_PORT='' TPU_WORKER_HOSTNAMES='' TPU_WORKER_ID=0
export TPU_HOST_BOUNDS=1,1,1 TPU_TOPOLOGY=2x2x1 TPU_ACCELERATOR_TYPE=tpu7x-8 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
# GKE lists this node's 4 chips in TPU_VISIBLE_CHIPS; vLLM's data-parallel launcher then indexes 8 TensorCores into a
# 4-entry list and fails ("Error computing device indices for TPU_VISIBLE_CHIPS"), so leave it unset.
unset TPU_VISIBLE_CHIPS
export TPU_BACKEND_TYPE=jax MODEL_IMPL_TYPE=vllm TPU_MULTIPROCESS_DP=1 USE_BATCHED_RPA_KERNEL=1
export JAX_GC_POST_COMPILE=true JAX_THREE_PHASE_DEALLOCATION=true
export VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_XLA_CACHE_PATH=/dev/shm/xla-cache
mkdir -p "$OUT" /dev/shm/xla-cache
python3 -c "import vllm; print('vllm', vllm.__version__)" > "$OUT/version.txt" 2>&1
python3 -m vllm.entrypoints.openai.api_server \
  --host=0.0.0.0 --port=8000 \
  --model="$MODEL_PATH" --served-model-name=openai/gpt-oss-120b \
  --tensor-parallel-size=2 --data-parallel-size=4 \
  --max-model-len=32768 --max-num-batched-tokens=16384 --max-num-seqs="$MAX_NUM_SEQS" \
  --enable-chunked-prefill --no-enable-prefix-caching --trust-remote-code \
  --kv-cache-dtype=fp8 --async-scheduling --gpu-memory-utilization=0.86 \
  > /dev/shm/server.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
  kill -0 "$SRV" 2>/dev/null || { echo "server exited during startup; see /dev/shm/server.log"; exit 1; }
  sleep 10
done
# The benchmark client must not touch the TPU that the server holds.
export JAX_PLATFORMS=cpu TPU_SKIP_MDS_QUERY=true TPU_VISIBLE_CHIPS="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bench() {  # isl osl concurrency prompts tag
  vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --model openai/gpt-oss-120b --tokenizer "$MODEL_PATH" --trust-remote-code \
    --dataset-name random --random-input-len "$1" --random-output-len "$2" \
    --num-prompts "$4" --request-rate inf --max-concurrency "$3" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 --seed 42 \
    --save-result --result-dir "$OUT" --result-filename "$5.json"
}
bench 1024 1024 8 16 warmup_isl1024_osl1024_c8_n16
kt() { if [ $(( $1 % 1024 )) -eq 0 ]; then echo "$(( $1 / 1024 ))k"; else echo "$(( $1 / 1000 ))k"; fi; }  # 1024->1k, 22000->22k
K="$(kt "$INPUT_LEN")$(kt "$OUTPUT_LEN")"
for c in $LOADS; do
  bench "$INPUT_LEN" "$OUTPUT_LEN" "$c" $((2 * c)) "ref_${K}_isl${INPUT_LEN}_osl${OUTPUT_LEN}_c${c}_n$((2 * c))"
done
kill "$SRV"
