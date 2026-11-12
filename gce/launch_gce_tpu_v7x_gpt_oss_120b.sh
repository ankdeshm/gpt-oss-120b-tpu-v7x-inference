#!/usr/bin/env bash
# Optional Single-Host (1 host = 4 chips) GCE TPU v7x (tpu7x-standard-4t) Helper
# Models:
#   - lmsys/gpt-oss-120b-bf16 (Pure BF16 weights, 234 GB -> TP=8, DP=1, EP)
#   - openai/gpt-oss-120b     (Native MXFP4 weights, 65.3 GB -> TP=2, DP=4 or TP=8, EP=8)
# NOTE: For DWS Calendar Mode reservations and Multi-Host (2-host / 8-chip or 4-host / 16-chip)
# deployments, use the tested GKE workflow in INFRA_SETUP.md and k8s/provision_gke_dws_calendar.sh.
set -euo pipefail

GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET=<YOUR_GCS_BUCKET>}"
MODEL_SUBPATH="${MODEL_SUBPATH:-models/openai/gpt-oss-120b}"
BF16_MODEL_SUBPATH="${BF16_MODEL_SUBPATH:-models/lmsys/gpt-oss-120b-bf16}"
XLA_CACHE_SUBPATH="${XLA_CACHE_SUBPATH:-xla-cache/gpt-oss-120b}"
MOUNT_DIR="${MOUNT_DIR:-/model-vol-mount}"
VLLM_TPU_IMAGE="${VLLM_TPU_IMAGE:-vllm/vllm-tpu:latest}"
TORCHTPU_IMAGE="${TORCHTPU_IMAGE:-us-central1-docker.pkg.dev/<YOUR_PROJECT_ID>/vllm/vllm-torchtpu-bench:v0.32.0}"
SGLANG_IMAGE="${SGLANG_IMAGE:-us-central1-docker.pkg.dev/<YOUR_PROJECT_ID>/sglang/sglang-torchtpu-diffusion:v0.2.0}"
MODE="${1:-help}"

install_deps() {
  sudo apt-get update && sudo apt-get install -y curl git jq fuse3 ca-certificates
  if ! command -v gcsfuse >/dev/null 2>&1; then
    local codename
    codename="$(lsb_release -c -s)"
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt gcsfuse-${codename} main" \
      | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc >/dev/null
    sudo apt-get update && sudo apt-get install -y gcsfuse
  fi
  if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
  fi
  sudo usermod -aG docker "${USER}" || true
  gcloud auth configure-docker us-central1-docker.pkg.dev --quiet || true
}

mount_gcsfuse() {
  sudo mkdir -p "${MOUNT_DIR}"
  sudo chown -R "${USER}:${USER}" "${MOUNT_DIR}" || true
  if mountpoint -q "${MOUNT_DIR}"; then
    echo "${MOUNT_DIR} is already mounted."
  else
    gcsfuse --implicit-dirs \
      -o allow_other \
      --stat-cache-max-size-mb=-1 \
      --type-cache-max-size-mb=-1 \
      --kernel-list-cache-ttl-secs=-1 \
      --enable-buffered-read \
      --read-ahead-kb=131072 \
      "${GCS_BUCKET}" "${MOUNT_DIR}"
  fi
  mkdir -p "${MOUNT_DIR}/${XLA_CACHE_SUBPATH}"
}

# ==============================================================================
# PROFILE 1: Pure BF16 Checkpoint (lmsys/gpt-oss-120b-bf16, 234 GB -> TP=8, EP)
# ==============================================================================

# 1A: Pure BF16 weights + BF16 KV cache on public vllm/vllm-tpu:latest (TP=8, DP=1, EP)
serve_vllm_tpu_bf16_tp8() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e TPU_BACKEND_TYPE=jax \
    -e MODEL_IMPL_TYPE=vllm \
    -e USE_BATCHED_RPA_KERNEL=1 \
    -e JAX_GC_POST_COMPILE=true \
    -e JAX_THREE_PHASE_DEALLOCATION=true \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-bf16-vllmtpu-bf16kv" \
    "${VLLM_TPU_IMAGE}" \
    python3 -m vllm.entrypoints.openai.api_server \
      --host=0.0.0.0 \
      --port=8000 \
      --model="/model-vol-mount/${BF16_MODEL_SUBPATH}" \
      --served-model-name=openai/gpt-oss-120b \
      --tensor-parallel-size=8 \
      --data-parallel-size=1 \
      --enable-expert-parallel \
      --max-model-len=32768 \
      --max-num-batched-tokens=16384 \
      --max-num-seqs=64 \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --trust-remote-code \
      --kv-cache-dtype=auto \
      --async-scheduling \
      --gpu-memory-utilization=0.88
}

# 1B: Native BF16 weights + FP8 KV cache on public vllm/vllm-tpu:latest (TP=8, DP=1, EP)
serve_vllm_tpu_bf16_fp8kv_tp8() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e TPU_BACKEND_TYPE=jax \
    -e MODEL_IMPL_TYPE=vllm \
    -e USE_BATCHED_RPA_KERNEL=1 \
    -e JAX_GC_POST_COMPILE=true \
    -e JAX_THREE_PHASE_DEALLOCATION=true \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-bf16-vllmtpu-fp8kv" \
    "${VLLM_TPU_IMAGE}" \
    python3 -m vllm.entrypoints.openai.api_server \
      --host=0.0.0.0 \
      --port=8000 \
      --model="/model-vol-mount/${BF16_MODEL_SUBPATH}" \
      --served-model-name=openai/gpt-oss-120b \
      --tensor-parallel-size=8 \
      --data-parallel-size=1 \
      --enable-expert-parallel \
      --max-model-len=32768 \
      --max-num-batched-tokens=16384 \
      --max-num-seqs=128 \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --trust-remote-code \
      --kv-cache-dtype=fp8 \
      --async-scheduling \
      --gpu-memory-utilization=0.88
}

# 1C: Pure BF16 weights + BF16 KV cache on sglang-torchtpu (TP=8, EP=8 — 693ms 22k TTFT, 4.49ms TPOT)
serve_sglang_bf16_tp8() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e PJRT_DEVICE=TPU \
    -e JAX_PLATFORMS=tpu \
    "${SGLANG_IMAGE}" \
    python3 -m sglang.launch_server \
      --model-path "/model-vol-mount/${BF16_MODEL_SUBPATH}" \
      --served-model-name openai/gpt-oss-120b \
      --host 0.0.0.0 \
      --port 8000 \
      --tp-size 8 \
      --ep-size 8 \
      --context-length 32768 \
      --max-prefill-tokens 32768 \
      --chunked-prefill-size 8192 \
      --max-running-requests 64 \
      --page-size 64 \
      --kv-cache-dtype bfloat16 \
      --attention-backend pallas_tpu_rpa_v3 \
      --mem-fraction-static 0.85 \
      --disable-radix-cache \
      --trust-remote-code
}

# 1D: Pure BF16 weights + BF16 KV cache on vllm-torchtpu (TP=8, DP=1, EP)
serve_torchtpu_bf16_tp8() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e PJRT_DEVICE=TPU \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}-bf16-torchtpu" \
    "${TORCHTPU_IMAGE}" \
    vllm serve "/model-vol-mount/${BF16_MODEL_SUBPATH}" \
      --served-model-name openai/gpt-oss-120b \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size=8 \
      --data-parallel-size=1 \
      --enable-expert-parallel \
      --attention-backend CUSTOM \
      --max-model-len=32768 \
      --max-num-batched-tokens=16384 \
      --max-num-seqs=64 \
      --async-scheduling \
      --no-enable-prefix-caching \
      --gpu-memory-utilization=0.88 \
      --kv-cache-dtype=auto \
      --block-size=256
}

# ==============================================================================
# PROFILE 2: Native MXFP4 Checkpoint (openai/gpt-oss-120b, 65.3 GB)
# ==============================================================================

# 2A: MXFP4 Throughput Champion on public vllm/vllm-tpu:latest (TP=2, DP=4, FP8 KV)
serve_vllm_tpu() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e TPU_BACKEND_TYPE=jax \
    -e MODEL_IMPL_TYPE=vllm \
    -e TPU_MULTIPROCESS_DP=1 \
    -e USE_BATCHED_RPA_KERNEL=1 \
    -e JAX_GC_POST_COMPILE=true \
    -e JAX_THREE_PHASE_DEALLOCATION=true \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    -e VLLM_XLA_CACHE_PATH="/model-vol-mount/${XLA_CACHE_SUBPATH}" \
    "${VLLM_TPU_IMAGE}" \
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

# 2B: MXFP4 Low-Latency Champion on sglang-torchtpu (TP=8, EP=8, FP8 KV)
serve_sglang_mxfp4_tp8() {
  sudo docker run --rm --privileged --net=host --shm-size=256g \
    -v /dev/vfio:/dev/vfio \
    -v "${MOUNT_DIR}:/model-vol-mount" \
    -e PJRT_DEVICE=TPU \
    -e JAX_PLATFORMS=tpu \
    "${SGLANG_IMAGE}" \
    python3 -m sglang.launch_server \
      --model-path "/model-vol-mount/${MODEL_SUBPATH}" \
      --served-model-name openai/gpt-oss-120b \
      --host 0.0.0.0 \
      --port 8000 \
      --tp-size 8 \
      --ep-size 8 \
      --context-length 32768 \
      --max-prefill-tokens 32768 \
      --chunked-prefill-size 8192 \
      --max-running-requests 64 \
      --page-size 64 \
      --kv-cache-dtype fp8_e4m3 \
      --attention-backend pallas_tpu_rpa_v3 \
      --mem-fraction-static 0.85 \
      --disable-radix-cache \
      --trust-remote-code
}

run_client_benchmarks() {
  local lb_host="${LB_HOST:-127.0.0.1}"
  local lb_port="${LB_PORT:-8000}"
  local out_dir="${OUT_DIR:-./results}"
  local tok_dir="${TOKENIZER_DIR:-${MOUNT_DIR}/${MODEL_SUBPATH}}"
  if [ ! -d "${tok_dir}" ] && [ -d "${MOUNT_DIR}/${BF16_MODEL_SUBPATH}" ]; then
    tok_dir="${MOUNT_DIR}/${BF16_MODEL_SUBPATH}"
  fi
  mkdir -p "${out_dir}"

  # 1. Smoke test (1024 in / 21 out, CON = 1, 8)
  for CON in 1 8; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=1024 --random-output-len=21 --num-prompts=32 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=openai/gpt-oss-120b --tokenizer="${tok_dir}" \
      --request-rate=inf --max-concurrency="${CON}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done

  # 2. Long-context Open-Loop Poisson RPS validation (22,000 in / 4,000 out, RPS = 1, 2, 4, 6)
  for RPS in 1 2 4 6; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=openai/gpt-oss-120b --tokenizer="${tok_dir}" \
      --request-rate="${RPS}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done

  # 3. Long-context Closed-Loop Concurrency Sweep (22,000 in / 4,000 out, CON = 1, 6, 16, 32)
  for CON in 1 6 16 32; do
    vllm bench serve --backend=vllm --dataset-name=random \
      --random-input-len=22000 --random-output-len=4000 --num-prompts=50 --ignore-eos \
      --host="${lb_host}" --port="${lb_port}" --endpoint=/v1/completions \
      --model=openai/gpt-oss-120b --tokenizer="${tok_dir}" \
      --request-rate=inf --max-concurrency="${CON}" \
      --percentile-metrics=ttft,tpot,itl,e2el --metric-percentiles=50,90,95,99 \
      --goodput ttft:500 tpot:10 e2el:60000 --save-result --result-dir="${out_dir}"
  done
}

case "${MODE}" in
  install-deps) install_deps ;;
  mount) mount_gcsfuse ;;
  serve-vllmtpu-bf16-tp8) serve_vllm_tpu_bf16_tp8 ;;
  serve-vllmtpu-bf16-fp8kv-tp8) serve_vllm_tpu_bf16_fp8kv_tp8 ;;
  serve-sglang-bf16-tp8) serve_sglang_bf16_tp8 ;;
  serve-torchtpu-bf16-tp8) serve_torchtpu_bf16_tp8 ;;
  serve-vllmtpu|serve-vllmtpu-mxfp4) serve_vllm_tpu ;;
  serve-sglang-mxfp4-tp8) serve_sglang_mxfp4_tp8 ;;
  bench) run_client_benchmarks ;;
  *)
    echo "Usage: GCS_BUCKET=<YOUR_GCS_BUCKET> $0 {install-deps|mount|serve-vllmtpu-bf16-tp8|serve-vllmtpu-bf16-fp8kv-tp8|serve-sglang-bf16-tp8|serve-torchtpu-bf16-tp8|serve-vllmtpu|serve-sglang-mxfp4-tp8|bench}"
    exit 1
    ;;
esac
