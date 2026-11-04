# Serving `GPT-OSS 120B` on Google Cloud Ironwood TPU (v7x)

Production-ready inference recipes, **GKE + DWS Calendar Mode reservation** provisioning scripts and manifests, TorchTPU (`sglang-torchtpu` & `vllm-torchtpu`) and `vllm-tpu` (JAX) runtime configurations, and full-spectrum benchmark telemetry for serving **`GPT-OSS 120B`** across both **Native Unquantized BF16** (`lmsys/gpt-oss-120b-bf16`) and **Native MXFP4** (`openai/gpt-oss-120b`) checkpoints on **Google Cloud Ironwood TPU (`tpu7x`)**.

> **First-Time GCP User or Activating a DWS Calendar Reservation?**
> Start with **[`INFRA_SETUP.md`](INFRA_SETUP.md)** and **[`k8s/provision_gke_dws_calendar.sh`](k8s/provision_gke_dws_calendar.sh)** for a step-by-step guide to setting up your VPC, Cloud NAT, GCS bucket, Regional Workload Policy (`2x2x1`, `2x2x2`, or `2x2x4`), and GKE Node Pool bound to your **DWS Calendar Mode reservation (`--reservation-affinity=specific --reservation=<RESERVATION_NAME>`)**.

---

## Hardware & Architecture Overview

| Attribute | Ironwood TPU (`tpu7x`) Specification |
|---|---|
| **Single Host (`tpu7x-standard-4t`)** | `4` physical Ironwood chips (`2x2x1` ICI mesh) = `8` TensorCores + SparseCores |
| **HBM Capacity & Bandwidth** | `192 GB` HBM3e per chip (`768 GB` HBM3e per 4-chip host) @ `7.4 TB/s` per chip |
| **2-Host Deployment (`8` Chips)** | `2 × tpu7x-standard-4t` (`2x2x2` WorkloadPolicy, `16` TensorCores, `1.54 TB` HBM3e) |
| **4-Host Deployment (`16` Chips)** | `4 × tpu7x-standard-4t` (`2x2x4` WorkloadPolicy, `32` TensorCores, `3.07 TB` HBM3e) |
| **Native BF16 Checkpoint** | [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (`~233.7 GB` unquantized BF16 MoE — runs on **all 3 stacks**: `sglang-torchtpu`, `vllm-torchtpu`, `vllm-tpu`) |
| **MXFP4 Model Checkpoint** | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`~65.3 GB` native MXFP4 MoE + BF16 attention — runs on `sglang-torchtpu` & `vllm-tpu`) |
| **Draft Checkpoint** | [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) (`EAGLE3` speculative decoding) |

---

## Headline Performance Highlights

All benchmarks use `vllm bench serve` with uncached random token IDs (`--ignore-eos`, zero prefix-cache reuse) and track strict per-request SLA Goodput (`TTFT <= 500 ms`, `TPOT <= 10 ms`, `E2EL <= 60 s`).

### 1. Native Unquantized BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB` Weights) Across 1 & 2 Hosts

| Scale | Serving Stack & KV Precision | `22k / 4k` `CON=1` TTFT P50 | `22k / 4k` `CON=1` TPOT P50 | `22k / 4k` Peak Out tok/s | `1k / 1k` `CON=1` TPOT P50 | `1k / 1k` Peak Out tok/s | `8k / 1k` Peak Out tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| **1 Host (4 Chips)** | `sglang-torchtpu` (BF16 KV, Patched) | **`739.5 ms`** | **`4.97 ms`** | `1,150.1` | **`4.85 ms`** | `2,530.8` | `1,630.1` |
| **1 Host (4 Chips)** | `sglang-torchtpu` + `EAGLE3` (BF16 KV) | `762.8 ms` | `5.48 ms` | `977.7` | `5.32 ms` | `2,059.0` | `1,349.1` |
| **1 Host (4 Chips)** | `vllm-torchtpu` (BF16 KV, Patched) | `878.7 ms` | `5.95 ms` | `733.6` | `5.30 ms` | `2,243.0` | `1,600.2` |
| **1 Host (4 Chips)** | `vllm-torchtpu` (FP8 KV, Patched) | `877.1 ms` | `5.96 ms` | `739.8` | `5.30 ms` | `2,284.8` | `1,612.4` |
| **1 Host (4 Chips)** | `vllm-tpu` (JAX, BF16 KV, Unpatched) | `880.5 ms` | `8.54 ms` | `1,249.9` | `8.21 ms` | `2,463.7` | `1,557.4` |
| **1 Host (4 Chips)** | `vllm-tpu` (JAX, FP8 KV, Unpatched) | `880.3 ms` | `8.48 ms` | `1,302.6` | `8.26 ms` | `2,540.3` | `1,582.9` |
| **2 Hosts (8 Chips)**| `2x sglang-torchtpu` + `least_conn` LB | **`771.4 ms`** | **`5.05 ms`** | **`2,433.8`** (`2.12x`) | **`5.77 ms`** (`CON=8`) | **`5,084.5`** (`2.01x`) | **`3,281.0`** (`2.01x`) |

### 2. Multi-Host Scaling (`openai/gpt-oss-120b` MXFP4, `65.3 GB` Weights)

| Scale | Chips (Cores) | Serving Engine & Topology | `1k / 1k` Peak Out tok/s | `1k / 1k` Peak Goodput | `8k / 1k` Peak Out tok/s (Total tok/s) | `22k / 4k` Peak Out tok/s | `22k / 4k` @ `6 RPS` P50 TPOT / E2E |
|---|---:|---|---:|---:|---:|---:|---:|
| **1 Host** | `4` (`8`) | `vllm-tpu` (JAX) `TP=2, DP=4`, FP8 KV | **`5,847.0`** | `3.340 req/s` | `2,696.6` (`24,269.5`) | `1,789.4` | `14.76 ms` / `71.17 s` |
| **1 Host** | `4` (`8`) | `sglang-torchtpu` `TP=8, EP=8`, FP8 KV | `2,587.5` | `1.104 req/s` | `1,630.1` (`14,670.9`) | `1,242.2` | `4.09 ms` TPOT / **`737.0 ms` TTFT** @ `CON=1` |
| **2 Hosts** | `8` (`16`) | `2x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`14,277.1`** | `5.054 req/s` | `6,533.7` (`58,803.6`) | `3,315.7` | `11.27 ms` / `50.51 s` |
| **4 Hosts** | `16` (`32`) | `4x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`20,366.9`** | **`8.584 req/s`** | **`9,820.8`** (**`88,387.3`**) | **`8,749.7`** | **`8.82 ms`** (`P95 9.96 ms`) / **`37.90 s`** |

See **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** for complete 1-host, 2-host, and 4-host telemetry across all suites.

---

## TorchTPU (`sglang-torchtpu` & `vllm-torchtpu`) vs. `vllm-tpu` (JAX): Precision Support & Why Runtime Patches Are Included

### 1. How `openai/gpt-oss-120b` (MXFP4) vs. `lmsys/gpt-oss-120b-bf16` (Native BF16) Execute on Ironwood TPU (`v7x`)
- **Why `openai/gpt-oss-120b` (`gpt_oss_mxfp4`) runs on `sglang-torchtpu` and `vllm-tpu`, but not on `vllm-torchtpu`**:
  - `gpt_oss_mxfp4` is a 4-bit microscaling format (`32`-element blocks with `E8M0` shared scales) created by OpenAI for GPU/CUDA Triton kernels.
  - Ironwood TPU (`v7x`) Matrix Multiply Units (MXUs) execute natively in **BF16 (`2,307 TFLOPs/chip`)** and **FP8 (`4,614 TFLOPs/chip`)** (native hardware FP4 MXU arrives in TPU v8).
  - **`sglang-torchtpu`** and **`vllm-tpu` (JAX)** implement custom Pallas TPU kernels that unpack 4-bit `mxfp4` weights on the fly in the TPU Vector Processing Unit (VPU) into BF16/FP8 registers before feeding the MXU.
  - Upstream **`vllm-torchtpu`** has not yet registered a TPU kernel for `gpt_oss_mxfp4` (`vllm/model_executor/layers/quantization/mxfp4.py` only binds CUDA/ROCm Triton kernels) and raises `ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu`.
- **Why `lmsys/gpt-oss-120b-bf16` (`234 GB` Pure Unquantized BF16) is recommended for comparing all 3 stacks (`sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu`)**:
  - A single 4-chip Ironwood host has **`768 GB` of HBM3e**, so the entire `234 GB` unquantized BF16 checkpoint fits comfortably on 1 host (`TP=8, EP=8`) with `> 400 GB` of HBM3e left over for KV cache.

### 2. Why `vllm-torchtpu` and `sglang-torchtpu` Use a Lightweight Load-Time Alignment Patch on Native BF16 (`scripts/patches/`)
- **`vllm-tpu` (JAX)** runs `lmsys/gpt-oss-120b-bf16` out-of-the-box with **zero patches** (`2,540.3` out tok/s on `1k/1k`).
- On **`vllm-torchtpu`** and **`sglang-torchtpu`**, two upstream details in `gpt_oss.py` and the unquantized BF16 MoE layer are resolved automatically at startup by [`scripts/patches/apply_vllm_torchtpu_bf16_patches.py`](scripts/patches/apply_vllm_torchtpu_bf16_patches.py) and [`scripts/patches/apply_sglang_bf16_patches.py`](scripts/patches/apply_sglang_bf16_patches.py):
  1. **TPU MXU `128 × 128` Tile Alignment (`hidden_size = 2880 -> 2944`)**:
     - TPU MXUs and Pallas grouped-matmul (`gmm_v2`) kernels require matrix dimensions to be multiples of **`128`**.
     - `GPT-OSS 120B` has a non-standard `hidden_size = 2880` (`22.5 × 128`, leaving a remainder of `64` columns).
     - Upstream `vllm-torchtpu` padded `intermediate_size` to a multiple of `128`, **but forgot to pad `hidden_size` (`2880 -> 2944`, `+64` zero columns)** in its unquantized BF16 MoE layer (`VllmUnquantizedFusedMoEMethod`). Without that 6-line load-time padding, Pallas `gmm_v2` reads 64 uninitialized memory lanes past each row (`18.8 ms` TPOT). With the 6-line load-time padding, **`vllm-torchtpu` decode latency drops `3.2x` from `18.8 ms` down to `5.30–5.95 ms` TPOT (`2,284.8` out tok/s on `1k/1k`)**!
  2. **Standard Hugging Face BF16 Tensor Mapping & `swigluoai` Deinterleaving**:
     - Upstream `gpt_oss.py` loaders were originally written assuming MXFP4 block/scale tensor names. Our startup patch maps standard BF16 expert tensor names (`gate_up_proj`, `down_proj`), deinterleaves OpenAI's `swigluoai` gate/up channels (`[::2]` and `[1::2]`), and pads `2880 -> 2944` once during weight loading, enabling **`sglang-torchtpu` to achieve `739.5 ms` TTFT on `22k` input, `4.97 ms` TPOT, and `5,084.5` out tok/s across 2 hosts (`8` chips)**!

---

## Repository Structure

```text
.
├── README.md                                     # Architecture, benchmark summary & precision guide
├── INFRA_SETUP.md                                # Step-by-step GKE + DWS Calendar Mode Reservation guide
├── BENCHMARK_RESULTS.md                          # Complete 1h, 2h, and 4h benchmark tables & sizing guide
├── configs/                                      # Reproducible JSON benchmark configs
│   ├── bf16/                                     # Native BF16 configs across sglang-torchtpu, vllm-torchtpu & vllm-tpu
│   └── ...                                       # MXFP4 configs across 1h, 2h, and 4h topologies
├── k8s/
│   ├── provision_gke_dws_calendar.sh             # Automated GKE + WorkloadPolicy + DWS Calendar node-pool setup
│   ├── gpt-oss-120b-bf16-sglang-v7x.yaml         # GKE Deployment (Native BF16 sglang-torchtpu TP=8, EP=8)
│   ├── gpt-oss-120b-bf16-vllm-tpu-v7x.yaml       # GKE Deployment (Native BF16 vllm-tpu JAX TP=8, EP=true)
│   ├── gpt-oss-120b-vllm-tpu-v7x.yaml            # GKE Deployment (MXFP4 vllm-tpu JAX TP=2, DP=4, FP8 KV)
│   └── gpt-oss-120b-sglang-eagle3-v7x.yaml       # GKE Deployment (MXFP4 sglang-torchtpu TP=8, EP=8 + EAGLE3)
├── gce/
│   └── launch_gce_tpu_v7x_gpt_oss_120b.sh        # Optional Single-Host (1 host = 4 chips) GCE helper script
├── results/                                      # Sanitized raw & summary JSON artifacts for all runs
└── scripts/
    ├── patches/                                  # Runtime patches for Native BF16 serving on TorchTPU
    │   ├── apply_sglang_bf16_patches.py          # FusedMoE BF16 weight layout & swigluoai deinterleaving
    │   ├── apply_vllm_torchtpu_bf16_patches.py   # Pallas hidden_size (2880->2944) padding (3.2x decode speedup)
    │   └── README.md                             # Patch architectural documentation
    ├── stage_hf_to_gcs.py                        # Multi-threaded HF -> GCS weight streamer
    ├── run_v7x_bench.py                          # Unified server launcher, warmup verifier & bench runner
    ├── lb_proxy.py                               # Zero-buffering async least_conn HTTP load balancer
    ├── bench_agent.py                            # Sidecar HTTP agent for CPU-isolated benchmark execution
    ├── bench_client.py                           # Standalone async OpenAI streaming benchmark client
    ├── mock_metadata.py                          # Local GCE metadata proxy for single-host slice isolation
    ├── metadata_hook.c                           # LD_PRELOAD connect() redirector pairing with mock_metadata.py
    └── summarize_results.py                      # Renders Markdown SLA & throughput tables from summary.json
```
