# Serving `openai/gpt-oss-120b` on Google Cloud Ironwood TPU (v7x)

Production-ready inference recipes, Kubernetes (GKE) and Compute Engine (GCE) deployment manifests, multi-host load-balancing tooling, and full-spectrum benchmark telemetry for serving **`openai/gpt-oss-120b`** (117B total parameters, 5.1B active parameters per token, native MXFP4 Mixture-of-Experts weights) on **Google Cloud Ironwood TPU (`tpu7x`)**.

---

## Hardware & Architecture Overview

| Attribute | Ironwood TPU (`tpu7x`) Specification |
|---|---|
| **Single Host (`tpu7x-standard-4t`)** | `4` physical Ironwood chips (`2x2x1` ICI mesh) = `8` TensorCores + SparseCores |
| **HBM Capacity & Bandwidth** | `192 GB` HBM3e per chip (`768 GB` HBM3e per 4-chip host) @ `7.4 TB/s` per chip |
| **2-Host Deployment (`8` Chips)** | `2 x tpu7x-standard-4t` (`16` TensorCores, `1.54 TB` aggregate HBM3e) |
| **4-Host Deployment (`16` Chips)** | `4 x tpu7x-standard-4t` (`32` TensorCores, `3.07 TB` aggregate HBM3e) |
| **Model Checkpoint** | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`~65.3 GB` native MXFP4 MoE + BF16 attention) |
| **Optional Draft Checkpoint** | [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) (`EAGLE3` speculative decoding) |

---

## Headline Performance Highlights (`1`, `2`, and `4` Hosts)

All benchmarks use `vllm bench serve` with uncached random token IDs (`--ignore-eos`, zero prefix-cache reuse) and track strict per-request SLA Goodput (`TTFT <= 500 ms`, `TPOT <= 10 ms`, `E2EL <= 60 s`).

| Scale | Chips (Cores) | Serving Engine & Topology | `1k / 1k` Peak Out tok/s | `1k / 1k` Peak Goodput | `8k / 1k` Peak Out tok/s (Total tok/s) | `22k / 4k` Peak Out tok/s | `22k / 4k` @ `6 RPS` P50 TPOT / E2E |
|---|---:|---|---:|---:|---:|---:|---:|
| **1 Host** | `4` (`8`) | `vllm-tpu` (JAX) `TP=2, DP=4`, FP8 KV | **`5,847.0`** | `3.340 req/s` | `2,696.6` (`24,269.5`) | `1,789.4` | `14.76 ms` / `71.17 s` |
| **1 Host** | `4` (`8`) | `sglang-torchtpu` `TP=8, EP=8`, FP8 KV | `2,587.5` | `1.104 req/s` | `1,630.1` (`14,670.9`) | `1,242.2` | `4.09 ms` TPOT / **`737.0 ms` TTFT** @ `CON=1` |
| **2 Hosts** | `8` (`16`) | `2x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`14,277.1`** | `5.054 req/s` | `6,533.7` (`58,803.6`) | `3,315.7` | `11.27 ms` / `50.51 s` |
| **4 Hosts** | `16` (`32`) | `4x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`20,366.9`** | **`8.584 req/s`** | **`9,820.8`** (**`88,387.3`**) | **`8,749.7`** | **`8.82 ms`** (`P95 9.96 ms`) / **`37.90 s`** |

See **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** for complete Wave 1 (`1-Host`) and Wave 2 (`2-Host` & `4-Host`) latency, throughput, and Poisson open-loop tables across all configurations.

---

## Choosing the Right Serving Stack on Ironwood TPU v7x

1. **Throughput & Multi-Host Scaling Champion — `vllm-tpu` (JAX `tpu-inference`, `TP=2, DP=4, --kv-cache-dtype=fp8`)**:
   - Runs **4 independent `TP=2` data-parallel engines per 4-chip Ironwood host** (`TPU_MULTIPROCESS_DP=1`, `USE_BATCHED_RPA_KERNEL=1`), cutting inter-core collective all-reduce traffic by `4x` while keeping the entire 65 GB MXFP4 checkpoint resident in each `TP=2` (`384 GB` HBM3e) pair with massive headroom for FP8 KV cache.
   - Scales horizontally via stateless single-host replicas behind a `least_conn` load balancer (`scripts/lb_proxy.py` or GKE Service/Gateway):
     - **`20,366.9` output tok/s** (`40,733.9` total tok/s) on `1,024 / 1,024` across 4 hosts (`16` chips).
     - **`9,820.8` output tok/s** (`88,387.3` total tok/s) on `8,192 / 1,024` across 4 hosts (`16` chips).
     - **`8,749.7` output tok/s** (`56,873.2` total tok/s, `E2E P95 = 58.44 s < 60 s`) on `22,000 / 4,000` at `CON=128` across 4 hosts (`16` chips), and **`5,564.6` output tok/s** (`TPOT P95 = 9.96 ms < 10 ms`, `E2E P95 = 41.11 s < 60 s`) under Open-Loop `6.0 RPS`.
2. **Ultra-Low Single-Stream Latency Champion — `sglang-torchtpu` (`TP=8, EP=8, fp8_e4m3 KV` + optional `EAGLE3`)**:
   - Shards prefill and MoE routing across all `8` TensorCores on the host (`TP=8, EP=8` with `pallas_tpu_rpa_v3`), reducing `22,000`-token single-request TTFT from **`1,279.3 ms`** (`TP=2`) down to **`737.0 ms`** (`746.7 ms` with EAGLE3) and delivering **`4.09 ms`** TPOT (`~244 tok/s` per stream, `E2E P50 = 17.08 s` for 4,000 generated tokens).

---

## Repository Structure

```text
.
├── README.md                                  # Architecture, quickstart, GKE & GCE deployment guide
├── BENCHMARK_RESULTS.md                       # Complete 1-host, 2-host, and 4-host benchmark tables & sizing
├── configs/                                   # Reproducible JSON configs for Wave 1 (1h) & Wave 2 (2h, 4h)
├── gce/
│   └── launch_gce_tpu_v7x_gpt_oss_120b.sh     # Turnkey GCE TPU VM mount, server launch & benchmark script
├── k8s/
│   ├── gpt-oss-120b-vllm-tpu-v7x.yaml         # GKE Deployment (vllm-tpu JAX TP=2, DP=4, FP8 KV) + GCSFuse PVC
│   └── gpt-oss-120b-sglang-eagle3-v7x.yaml    # GKE Deployment (sglang-torchtpu TP=8, EP=8 + EAGLE3)
├── results/                                   # Sanitized raw & summary JSON artifacts for all benchmark suites
└── scripts/
    ├── stage_hf_to_gcs.py                     # Multi-threaded HF -> GCS weight streamer (excludes original/*, metal/*)
    ├── run_v7x_bench.py                       # Unified server launcher, XLA warmup verifier & vllm bench serve runner
    ├── lb_proxy.py                            # Zero-buffering async least_conn HTTP load balancer for 2h/4h sweeps
    ├── bench_agent.py                         # Sidecar HTTP agent for CPU-isolated vllm bench serve execution
    ├── bench_client.py                        # Standalone async OpenAI streaming benchmark client
    ├── mock_metadata.py                       # Local GCE metadata proxy for 4-chip single-host isolation on GKE slices
    ├── metadata_hook.c                        # LD_PRELOAD connect() redirector pairing with mock_metadata.py
    └── summarize_results.py                   # Renders Markdown SLA & throughput tables from results/*/summary.json
```

---

## Step 1: Stage `openai/gpt-oss-120b` Weights to Google Cloud Storage

`openai/gpt-oss-120b` includes redundant `original/` and `metal/` subdirectories on Hugging Face (`~130 GB` extra). Use `scripts/stage_hf_to_gcs.py` with `--exclude 'original/*,metal/*'` to stream only the `~65.3 GB` serving weights directly to GCS with byte-level manifest verification:

```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
export GCS_BUCKET="<YOUR_GCS_BUCKET>"

python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos openai/gpt-oss-120b \
  --exclude 'original/*,metal/*' \
  --workers 8

# Optional: stage EAGLE3 draft model for speculative decoding
python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos lmsys/EAGLE3-gpt-oss-120b-bf16 \
  --workers 8
```

---

## Step 2: Deploy on Google Kubernetes Engine (GKE)

### 2.1 High-Throughput Deployment (`vllm-tpu` JAX `TP=2, DP=4, FP8 KV`)

1. Edit `k8s/gpt-oss-120b-vllm-tpu-v7x.yaml` and replace `<YOUR_GCS_BUCKET>` with your GCS bucket name.
2. Adjust `spec.replicas` to `1` (`4` chips), `2` (`8` chips), or `4` (`16` chips) to match your `tpu7x` node pool capacity:

```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-vllm-tpu-v7x.yaml | kubectl apply -f -
kubectl rollout status deployment/gpt-oss-120b-vllm-tpu-v7x --timeout=30m
```

> **Note on Multi-Host Slice Node Pools (`2x2x2` or `2x2x4`)**: When running independent 4-chip (`2x2x1`) pods on multi-host GKE TPU slice node pools, `scripts/run_v7x_bench.py` automatically launches `scripts/mock_metadata.py` + `scripts/metadata_hook.c` (`tpu_env_mode: "gke_single_host"`) so `libtpu` initializes each 4-chip host as a standalone `2x2x1` replica without cross-host ICI barrier lockups.

### 2.2 Low-Latency Deployment (`sglang-torchtpu` `TP=8, EP=8` + `EAGLE3`)

```bash
sed -e "s/<YOUR_PROJECT_ID>/<YOUR_PROJECT_ID>/g" \
    k8s/gpt-oss-120b-sglang-eagle3-v7x.yaml | kubectl apply -f -
```

---

## Step 3: Deploy on Google Compute Engine (GCE TPU VM)

On each `tpu7x-standard-4t` TPU VM:

```bash
# 1. Mount GCS bucket via gcsfuse with kernel metadata/list caching enabled
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh mount

# 2. Launch the vllm-tpu (TP=2, DP=4, FP8 KV) OpenAI API server on :8000
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh serve-vllmtpu
```

To load-balance `2` (`8` chips) or `4` (`16` chips) independent TPU hosts from a client VM (`c2-standard-16` or sidecar):

```bash
python3 scripts/lb_proxy.py \
  --listen 0.0.0.0:8080 \
  --backends http://10.0.0.11:8000,http://10.0.0.12:8000,http://10.0.0.13:8000,http://10.0.0.14:8000
```

---

## Step 4: Reproduce the Benchmarks

Run any single-host or multi-host configuration in `configs/` via `scripts/run_v7x_bench.py`:

```bash
# Single-host (4 chips) full suite (stages weights to /dev/shm, warms XLA cache, runs all cells)
python3 scripts/run_v7x_bench.py --config configs/gptoss120b_1h_4c_vllmtpu_fp8kv_tp2dp4.json

# 4-host (16 chips) load-balanced suite against lb_proxy.py (:8080)
python3 scripts/run_v7x_bench.py \
  --config configs/gptoss120b_4h_16c_vllmtpu_fp8kv_tp2dp4.json \
  --bench-only-url http://127.0.0.1:8080

# Regenerate Markdown summary tables from results/
python3 scripts/summarize_results.py results/
```
