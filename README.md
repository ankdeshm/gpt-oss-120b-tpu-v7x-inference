# Serving `GPT-OSS 120B` on Google Cloud Ironwood TPU (v7x)

Production-ready inference recipes, **Compute Engine (GCE / MIG) DWS Calendar Reservation** and Kubernetes (GKE) deployment manifests, multi-host load-balancing tooling, runtime patches, and full-spectrum benchmark telemetry for serving **`GPT-OSS 120B`** across both **Native MXFP4** (`openai/gpt-oss-120b`) and **Native Unquantized BF16** (`lmsys/gpt-oss-120b-bf16`) checkpoints on **Google Cloud Ironwood TPU (`tpu7x`)**.

> **First-Time GCP User or Using a DWS Calendar Reservation on GCE / MIG (Without GKE)?**
>
> **GKE is NOT required** to serve or benchmark `GPT-OSS 120B` on `1` host (`4` chips), `2` hosts (`8` chips), or `4` hosts (`16` chips). Each `tpu7x-standard-4t` GCE VM (`4` physical chips = `8` TensorCores, `768 GB` HBM3e) runs a self-contained OpenAI API server (`TP=8, EP` for `234 GB` Native BF16, or `TP=2, DP=4` / `TP=8, EP=8` for `65.3 GB` MXFP4), and `2` or `4` hosts scale linearly behind an NGINX or `scripts/lb_proxy.py` load balancer on a CPU Client VM.
>
> Follow **[`INFRA_SETUP.md`](INFRA_SETUP.md)** (or run [`gce/provision_dws_calendar_gce.sh`](gce/provision_dws_calendar_gce.sh)) for a complete, step-by-step guide from a blank GCP project: enabling APIs, IAM Service Account roles, Custom VPC / Subnet / Cloud NAT / Firewall Rules, Regional GCS Bucket & `gcsfuse` staging, binding `tpu7x-standard-4t` GCE VMs or a GCE MIG to your **DWS Calendar Mode reservation**, and running both **Pure `bfloat16`** and **Native `MXFP4`** recipes.

---

## Hardware & Architecture Overview

| Attribute | Ironwood TPU (`tpu7x`) Specification |
|---|---|
| **Single Host (`tpu7x-standard-4t`)** | `4` physical Ironwood chips (`2x2x1` ICI mesh) = `8` TensorCores + SparseCores |
| **HBM Capacity & Bandwidth** | `192 GB` HBM3e per chip (`768 GB` HBM3e per 4-chip host) @ `7.4 TB/s` per chip |
| **2-Host Deployment (`8` Chips)** | `2 x tpu7x-standard-4t` (`16` TensorCores, `1.54 TB` aggregate HBM3e) |
| **4-Host Deployment (`16` Chips)** | `4 x tpu7x-standard-4t` (`32` TensorCores, `3.07 TB` aggregate HBM3e) |
| **MXFP4 Model Checkpoint** | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`~65.3 GB` native MXFP4 MoE + BF16 attention) |
| **Native BF16 Checkpoint** | [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (`~233.7 GB` unquantized BF16 MoE -> `TP=8, DP=1, EP`, `58.5 GB`/chip) |
| **Draft Checkpoint** | [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) (`EAGLE3` speculative decoding) |

---

## Headline Performance Highlights

All benchmarks use `vllm bench serve` with uncached random token IDs (`--ignore-eos`, zero prefix-cache reuse) and track strict per-request SLA Goodput (`TTFT <= 500 ms`, `TPOT <= 10 ms`, `E2EL <= 60 s`).

### 1. Multi-Host Scaling (`openai/gpt-oss-120b` MXFP4)

| Scale | Chips (Cores) | Serving Engine & Topology | `1k / 1k` Peak Out tok/s | `1k / 1k` Peak Goodput | `8k / 1k` Peak Out tok/s (Total tok/s) | `22k / 4k` Peak Out tok/s | `22k / 4k` @ `6 RPS` P50 TPOT / E2E |
|---|---:|---|---:|---:|---:|---:|---:|
| **1 Host** | `4` (`8`) | `vllm-tpu` (JAX) `TP=2, DP=4`, FP8 KV | **`5,847.0`** | `3.340 req/s` | `2,696.6` (`24,269.5`) | `1,789.4` | `14.76 ms` / `71.17 s` |
| **1 Host** | `4` (`8`) | `sglang-torchtpu` `TP=8, EP=8`, FP8 KV | `2,587.5` | `1.104 req/s` | `1,630.1` (`14,670.9`) | `1,242.2` | `4.09 ms` TPOT / **`737.0 ms` TTFT** @ `CON=1` |
| **2 Hosts** | `8` (`16`) | `2x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`14,277.1`** | `5.054 req/s` | `6,533.7` (`58,803.6`) | `3,315.7` | `11.27 ms` / `50.51 s` |
| **4 Hosts** | `16` (`32`) | `4x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`20,366.9`** | **`8.584 req/s`** | **`9,820.8`** (**`88,387.3`**) | **`8,749.7`** | **`8.82 ms`** (`P95 9.96 ms`) / **`37.90 s`** |

### 2. Single-Host Native BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB` Weights)

| Serving Stack & KV Precision | `22k / 4k` `CON=1` TTFT P50 | `22k / 4k` `CON=1` TPOT P50 | `22k / 4k` `CON=1` E2E P50 | `22k / 4k` Peak Out tok/s | `1k / 1k` `CON=1` TPOT P50 | `1k / 1k` Peak Out tok/s (`CON=32`) |
|---|---:|---:|---:|---:|---:|---:|
| **`sglang-torchtpu` (BF16 KV)** | **`739.5 ms`** | **`4.97 ms`** | **`20.6 s`** | `1,150.1` | **`4.85 ms`** | **`2,530.8`** |
| **`sglang-torchtpu` + `EAGLE3` (BF16 KV)** | `762.8 ms` | `5.48 ms` | `22.7 s` | `977.7` | `5.32 ms` | `2,059.0` |
| **`vllm-torchtpu` (BF16 KV, Patched)** | `878.7 ms` | `5.95 ms` | `24.7 s` | `733.6` | `5.30 ms` | `2,243.0` |
| **`vllm-torchtpu` (FP8 KV, Patched)** | `877.1 ms` | `5.96 ms` | `24.7 s` | `739.8` | `5.30 ms` | `2,284.8` |
| **`vllm-tpu` (JAX, BF16 KV)** | `880.5 ms` | `8.54 ms` | `35.0 s` | `1,249.9` | `8.21 ms` | `2,463.7` |
| **`vllm-tpu` (JAX, FP8 KV)** | `880.3 ms` | `8.48 ms` | `34.8 s` | **`1,302.6`** | `8.26 ms` | **`2,540.3`** |

See **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** for complete 1-host, 2-host, and 4-host telemetry across all suites.

---

## Choosing the Right Serving Stack on Ironwood TPU v7x

1. **High-Throughput Multi-Host Champion — `vllm-tpu` (JAX `tpu-inference`)**:
   - **For MXFP4 (`openai/gpt-oss-120b`, `65.3 GB`)**: Runs **`4x TP=2` data-parallel engines per 4-chip host** (`192 GB` HBM3e per `TP=2` chip), scaling horizontally to **`20,366.9` output tok/s** across 4 hosts (`16` chips).
   - **For Native BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB`)**: Must use **`TP=8, DP=1, --enable-expert-parallel`** (`58.5 GB`/chip weights across 4 chips = `768 GB` HBM3e; `TP=2` will OOM on `234 GB` weights). Runs out-of-the-box on public `vllm/vllm-tpu:latest` with **zero patches**, sustaining **`2,540.3 tok/s`** on `1k/1k` and **`1,487.5 tok/s`** (`1,302.6 tok/s` closed-loop) on `22k/4k`.
2. **Lowest Single-Stream Latency Champion — `sglang-torchtpu` (PyTorch / Pallas)**:
   - Shards prefill across all `8` TensorCores (`TP=8, EP=8` with `pallas_tpu_rpa_v3`), delivering **`737.0 ms` TTFT** (MXFP4) / **`739.5 ms` TTFT** (Native BF16) and **`4.09 ms`** (MXFP4) / **`4.97 ms`** (Native BF16) decode TPOT (`> 201 tok/s` per stream).
   - Apply runtime patches via `python3 scripts/patches/apply_sglang_bf16_patches.py`.
3. **Pallas-Accelerated PyTorch Stack — `vllm-torchtpu` (PyTorch / XLA)**:
   - With `scripts/patches/apply_vllm_torchtpu_bf16_patches.py`, pads `hidden_size` (`2880 -> 2944`) to eliminate out-of-bounds lane reads in Pallas `gmm_v2`, dropping single-stream decode TPOT **`3.2x` from `18.8 ms` down to `5.95 ms`**.

---

## Repository Structure

```text
.
├── README.md                                  # Architecture, quickstart, GCE & GKE deployment guide
├── INFRA_SETUP.md                             # First-time GCP & DWS Calendar Reservation setup (GCE, MIG, GKE)
├── BENCHMARK_RESULTS.md                       # Complete 1h, 2h, and 4h benchmark tables & sizing guide
├── configs/                                   # Reproducible JSON benchmark configs
│   ├── bf16/                                  # Native BF16 configs across vllm-tpu, torchtpu, and sglang
│   └── ...                                    # MXFP4 configs across 1h, 2h, and 4h topologies
├── gce/
│   ├── provision_dws_calendar_gce.sh          # Turnkey VPC, NAT, Firewall, GCS & DWS Calendar GCE VM provisioner
│   └── launch_gce_tpu_v7x_gpt_oss_120b.sh     # Turnkey GCE TPU VM script (Pure BF16 TP=8 & MXFP4 TP=2xDP=4 / TP=8)
├── k8s/
│   ├── gpt-oss-120b-vllm-tpu-v7x.yaml         # GKE Deployment (vllm-tpu JAX TP=2, DP=4, FP8 KV)
│   ├── gpt-oss-120b-sglang-eagle3-v7x.yaml    # GKE Deployment (sglang-torchtpu TP=8, EP=8 + EAGLE3)
│   ├── gpt-oss-120b-bf16-vllm-tpu-v7x.yaml    # GKE Deployment (Native BF16 vllm-tpu JAX TP=8, EP=true)
│   └── gpt-oss-120b-bf16-sglang-v7x.yaml      # GKE Deployment (Native BF16 sglang-torchtpu TP=8, EP=8)
├── results/                                   # Sanitized raw & summary JSON artifacts for all runs
└── scripts/
    ├── patches/                               # Runtime patches for Native BF16 serving
    │   ├── apply_sglang_bf16_patches.py       # FusedMoE weight layout & swigluoai deinterleaving
    │   ├── apply_vllm_torchtpu_bf16_patches.py# Pallas hidden_size padding (3.2x decode speedup)
    │   └── README.md                          # Patch architectural documentation
    ├── stage_hf_to_gcs.py                     # Multi-threaded HF -> GCS weight streamer
    ├── run_v7x_bench.py                       # Unified server launcher, warmup verifier & bench runner
    ├── lb_proxy.py                            # Zero-buffering async least_conn HTTP load balancer
    ├── bench_agent.py                         # Sidecar HTTP agent for CPU-isolated benchmark execution
    ├── bench_client.py                        # Standalone async OpenAI streaming benchmark client
    ├── mock_metadata.py                       # Local GCE metadata proxy for single-host slice isolation
    ├── metadata_hook.c                        # LD_PRELOAD connect() redirector pairing with mock_metadata.py
    └── summarize_results.py                   # Renders Markdown SLA & throughput tables from summary.json
```

---

## Quickstart: Provision DWS Calendar GCE VMs, Stage Weights, Deploy & Benchmark

> For full step-by-step explanations of IAM roles, VPC/Firewall rules, GCE Managed Instance Groups (MIG), and NGINX setup, see **[`INFRA_SETUP.md`](INFRA_SETUP.md)**.

### 1. Provision GCP Network, GCS Bucket, Client VM & DWS-Reserved `tpu7x-standard-4t` VMs

```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export RESERVATION_NAME="<YOUR_DWS_RESERVATION_NAME>"
export GCS_BUCKET="${PROJECT_ID}-tpu-v7x-models"
export NUM_HOSTS=4   # 1 (4 chips), 2 (8 chips), or 4 (16 chips)

bash gce/provision_dws_calendar_gce.sh
```

### 2. Stage Checkpoints to GCS (from `tpu-bench-client`)

```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
export GCS_BUCKET="<YOUR_GCS_BUCKET>"

# Stage Native BF16 checkpoint (~234 GB) for Pure BF16 PoC benchmarks:
python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos lmsys/gpt-oss-120b-bf16 \
  --workers 8

# Stage Native MXFP4 checkpoint (~65.3 GB, excluding redundant original/* and metal/*):
python3 scripts/stage_hf_to_gcs.py \
  --bucket "${GCS_BUCKET}" \
  --prefix models \
  --repos openai/gpt-oss-120b \
  --exclude 'original/*,metal/*' \
  --workers 8
```

### 3. Deploy on GCE TPU VM (`tpu7x-standard-4t` — No GKE Required)

Run on each `tpu-v7x-node-{1..N}` VM:

```bash
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh install-deps
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh mount

# Option A — Pure BF16 weights + BF16 KV cache (lmsys/gpt-oss-120b-bf16, TP=8, DP=1, EP on public vllm/vllm-tpu:latest):
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh serve-vllmtpu-bf16-tp8
# Or Pure BF16 low-latency mode on sglang-torchtpu (TP=8, EP=8, 739.5ms TTFT / 4.97ms TPOT):
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh serve-sglang-bf16-tp8

# Option B — Native MXFP4 throughput champion (openai/gpt-oss-120b, TP=2, DP=4, FP8 KV on public vllm/vllm-tpu:latest):
GCS_BUCKET="<YOUR_GCS_BUCKET>" bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh serve-vllmtpu
```

### 4. Load Balance Multi-Host VMs & Run Reproducible Benchmarks (from `tpu-bench-client`)

```bash
python3 scripts/lb_proxy.py \
  --listen 0.0.0.0:8080 \
  --backends http://10.128.0.11:8000,http://10.128.0.12:8000,http://10.128.0.13:8000,http://10.128.0.14:8000 &

GCS_BUCKET="${GCS_BUCKET}" LB_HOST=127.0.0.1 LB_PORT=8080 \
  bash gce/launch_gce_tpu_v7x_gpt_oss_120b.sh bench
```
