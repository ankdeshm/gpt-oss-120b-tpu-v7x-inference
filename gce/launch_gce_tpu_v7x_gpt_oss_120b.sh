#!/usr/bin/env bash
# Turnkey GCE TPU v7x (tpu7x-standard-4t) Serving & Benchmark Recipe
# Model: openai/gpt-oss-120b
# Supports 1 host (4 chips), 2 hosts (8 chips), or 4 hosts (16 chips) + c2-standard-16 Client VM.
set -euo pipefail

GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET=<YOUR_GCS_BUCKET>}"
MODEL_SUBPATH="${MODEL_SUBPATH:-models/openai/gpt-oss-120b}"
XLA_CACHE_SUBPATH="${XLA_CACHE_SUBPATH:-xla-cache/gpt-oss-120b}"
MOUNT_DIR="${MOUNT_DIR:-/model-vol-mount}"
MODE="${1:-help}"

mount_gcsfuse() {
  sudo mkdir -p "${MOUNT_DIR}"
  gcsfuse --implicit-dirs \
    -o allow_other \
    --stat-cache-max-size-mb=-1 \
    --type-cache-max-size-mb=-1 \
    --kernel-list-cache-ttl-secs=-1 \
    --enable-buffered-read \
    --read-ahead-kb=131072 \
    "${GCS_BUCKET}" "${MOUNT_DIR}"
  mkdir -p "${MOUNT_DIR}/${XLA_CACHE_SUBPATH}"
}

serve_vllm_tpu() {
  docker run --rm --privileged --net=host --shm-size=256g \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e TPU_BACKEND_TYPE=jax \
    -e MODEL_IMPL_TYPE=vllm \
    -e TPU_MULTIPROCESS_DP=1 \
    -e USE_BATCHED_RPA_KERNEL=1 \
    -e JAX_GC_POST_COMPILE=true \
    -e JAX_THREE_PHASE_DEALLOCATION=true \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}" \
    vllm/vllm-tpu:latest \
    python3 -m vllm.entrypoints.openai.api_server \
      --host=0.0.0.0 \
      --port=8000 \
      --model="/model-vol-mount/${MODEL_SUBPATH}" \
      --served-model-name=openai/gpt-oss-120b \
      --tensor-parallel-size=2 \
      --data-parallel-size=4 \
      --max-model-len=32768 \
      --max-num-batched-tokens=16384 \
      --max-num-seqs=64 \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --trust-remote-code \
      --kv-cache-dtype=fp8 \
      --async-scheduling \
      --gpu-memory-utilization=0.86
}

run_client_benchmarks() {
  local lb_host="${LB_HOST:-127.0.0.1}"
  local lb_port="${LB_PORT:-8000}"
  local out_dir="${OUT_DIR:-./results}"
  mkdir -p "${out_dir}"

  # 1. Smoke test (1024 in / 21 out, CON = 1, 8)
  for CON in 1 8; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=1024 --random-output-len=21 --num-prompts=32 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=openai/gpt-oss-120b --tokenizer="${MOUNT_DIR}/${MODEL_SUBPATH}" \
      --request-rate=inf --max-concurrency="${CON}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done

  # 2. Long-context Open-Loop Poisson RPS validation (22,000 in / 4,000 out, RPS = 1, 2, 6)
  for RPS in 1 2 6; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=openai/gpt-oss-120b --tokenizer="${MOUNT_DIR}/${MODEL_SUBPATH}" \
      --request-rate="${RPS}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done

  # 3. Long-context Closed-Loop Concurrency Sweep (22,000 in / 4,000 out, CON = 1, 6, 16, 32)
  for CON in 1 6 16 32; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=openai/gpt-oss-120b --tokenizer="${MOUNT_DIR}/${MODEL_SUBPATH}" \
      --request-rate=inf --max-concurrency="${CON}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done
}

case "${MODE}" in
  mount) mount_gcsfuse ;;
  serve-vllmtpu) serve_vllm_tpu ;;
  bench) run_client_benchmarks ;;
  *)
    echo "Usage: GCS_BUCKET=<YOUR_GCS_BUCKET> $0 {mount|serve-vllmtpu|bench}"
    exit 1
    ;;
esac
