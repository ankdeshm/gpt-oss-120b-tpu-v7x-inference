# Serving `GPT-OSS 120B` on Google Cloud Ironwood TPU (v7x)

Production-ready inference recipes, Kubernetes (GKE) and Compute Engine (GCE) deployment manifests, multi-host load-balancing tooling, runtime patches, and full-spectrum benchmark telemetry for serving **`GPT-OSS 120B`** across both **Native MXFP4** (`openai/gpt-oss-120b`) and **Native Unquantized BF16** (`lmsys/gpt-oss-120b-bf16`) checkpoints on **Google Cloud Ironwood TPU (`tpu7x`)**.

---

## Hardware & Architecture Overview

| Attribute | Ironwood TPU (`tpu7x`) Specification |
|---|---|
| **Single Host (`tpu7x-standard-4t`)** | `4` physical Ironwood chips (`2x2x1` ICI mesh) = `8` TensorCores + SparseCores |
| **HBM Capacity & Bandwidth** | `192 GB` HBM3e per chip (`768 GB` HBM3e per 4-chip host) @ `7.4 TB/s` per chip |
| **2-Host Deployment (`8` Chips)** | `2 x tpu7x-standard-4t` (`16` TensorCores, `1.54 TB` aggregate HBM3e) |
| **4-Host Deployment (`16` Chips)** | `4 x tpu7x-standard-4t` (`32` TensorCores, `3.07 TB` aggregate HBM3e) |
| **MXFP4 Model Checkpoint** | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`~65.3 GB` native MXFP4 MoE + BF16 attention) |
| **Native BF16 Checkpoint** | [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (`~233.7 GB` unquantized BF16 MoE) |
| **Draft Checkpoint** | [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) (`EAGLE3` speculative decoding) |

---

## Performance Highlights

All benchmarks use `vllm bench serve` with uncached random token IDs (`--ignore-eos`, zero prefix-cache reuse) and track strict per-request SLA Goodput (`TTFT <= 500 ms`, `TPOT <= 10 ms`, `E2EL <= 60 s`).

### 1. Multi-Host Scaling (`openai/gpt-oss-120b` MXFP4)

| Scale | Chips (Cores) | Serving Engine & Topology | `1k / 1k` Peak Out tok/s | `1k / 1k` Peak Goodput | `8k / 1k` Peak Out tok/s (Total tok/s) | `22k / 4k` Peak Out tok/s | `22k / 4k` @ `6 RPS` P50 TPOT / E2E |
|---|---:|---|---:|---:|---:|---:|---:|
| **1 Host** | `4` (`8`) | `vllm-tpu` (JAX) `TP=2, DP=4`, FP8 KV | **`5,847.0`** | `3.340 req/s` | `2,696.6` (`24,269.5`) | `1,789.4` | `14.76 ms` / `71.17 s` |
| **1 Host** | `4` (`8`) | `sglang-torchtpu` `TP=8, EP=8`, FP8 KV | `2,587.5` | `1.104 req/s` | `1,630.1` (`14,670.9`) | `1,242.2` | `4.09 ms` TPOT / **`737.0 ms` TTFT** @ `CON=1` |
| **2 Hosts** | `8` (`16`) | `2x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`14,277.1`** | `5.054 req/s` | `6,533.7` (`58,803.6`) | `3,315.7` | `11.27 ms` / `50.51 s` |
| **4 Hosts** | `16` (`32`) | `4x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB | **`20,366.9`** | **`8.584 req/s`** | **`9,820.8`** (**`88,387.3`**) | **`8,749.7`** | **`8.82 ms`** (`P95 9.96 ms`) / **`37.90 s`** |

### 2. Native BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB` Weights) Across 1 & 2 Hosts

| Scale | Serving Stack & KV Precision | `22k / 4k` `CON=1` TTFT P50 | `22k / 4k` `CON=1` TPOT P50 | `22k / 4k` Peak Out tok/s | `1k / 1k` `CON=1` TPOT P50 | `1k / 1k` Peak Out tok/s | `8k / 1k` Peak Out tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| **1 Host (4 Chips)** | `sglang-torchtpu` (BF16 KV) | **`739.5 ms`** | **`4.97 ms`** | `1,150.1` | **`4.85 ms`** | `2,530.8` | `1,630.1` |
| **1 Host (4 Chips)** | `sglang` + `EAGLE3` (BF16 KV) | `762.8 ms` | `5.48 ms` | `977.7` | `5.32 ms` | `2,059.0` | `1,349.1` |
| **1 Host (4 Chips)** | `vllm-torchtpu` (BF16 KV, Patched) | `878.7 ms` | `5.95 ms` | `733.6` | `5.30 ms` | `2,243.0` | `1,600.2` |
| **1 Host (4 Chips)** | `vllm-torchtpu` (FP8 KV, Patched) | `877.1 ms` | `5.96 ms` | `739.8` | `5.30 ms` | `2,284.8` | `1,612.4` |
| **1 Host (4 Chips)** | `vllm-tpu` (JAX, BF16 KV) | `880.5 ms` | `8.54 ms` | `1,249.9` | `8.21 ms` | `2,463.7` | `1,557.4` |
| **1 Host (4 Chips)** | `vllm-tpu` (JAX, FP8 KV) | `880.3 ms` | `8.48 ms` | `1,302.6` | `8.26 ms` | `2,540.3` | `1,582.9` |
| **2 Hosts (8 Chips)**| `2x sglang-torchtpu` + `least_conn` LB | **`771.4 ms`** | **`5.05 ms`** | **`2,433.8`** (`2.12x`) | **`5.77 ms`** (`CON=8`) | **`5,084.5`** (`2.01x`) | **`3,281.0`** (`2.01x`) |

See **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** for complete 1-host, 2-host, and 4-host telemetry across all suites.

---

## Choosing the Right Serving Stack on Ironwood TPU v7x

1. **High-Throughput Multi-Host Champion — `vllm-tpu` (JAX `tpu-inference`)**:
   - For MXFP4: Runs **`4x TP=2` data-parallel engines per 4-chip host** (`384 GB` per pair), scaling horizontally to **`20,366.9` output tok/s** across 4 hosts (`16` chips).
   - For Native BF16: Runs `TP=8, --enable-expert-parallel` out-of-the-box with **zero patches**, sustaining **`2,540.3 tok/s`** on `1k/1k` and **`1,302.6 tok/s`** on `22k/4k`.
2. **Lowest Single-Stream Latency Champion — `sglang-torchtpu` (PyTorch / Pallas)**:
   - Shards prefill across all `8` TensorCores (`TP=8, EP=8` with `pallas_tpu_rpa_v3`), delivering **`737.0 ms` TTFT** (MXFP4) / **`739.5 ms` TTFT** (Native BF16) and **`4.09 ms`** (MXFP4) / **`4.97 ms`** (Native BF16) decode TPOT (`> 201 tok/s` per stream).
   - Scales horizontally across 2 hosts (`8` physical chips) to **`5,084.5 out tok/s`** on `1k/1k` (`2.01x` linear scaling) and **`2,433.8 out tok/s`** on `22k/4k` (`2.12x` scaling).
   - Apply runtime patches via `python3 scripts/patches/apply_sglang_bf16_patches.py`.
3. **Pallas-Accelerated PyTorch Stack — `vllm-torchtpu` (PyTorch / XLA)**:
   - With `scripts/patches/apply_vllm_torchtpu_bf16_patches.py`, pads `hidden_size` (`2880 -> 2944`) to eliminate out-of-bounds lane reads in Pallas `gmm_v2`, dropping single-stream decode TPOT **`3.2x` from `18.8 ms` down to `5.95 ms`**.

---

## Repository Structure

```text
.
├── README.md                                  # Architecture, quickstart, GKE & GCE deployment guide
├── BENCHMARK_RESULTS.md                       # Complete 1h, 2h, and 4h benchmark tables & sizing guide
├── configs/                                   # Reproducible JSON benchmark configs
│   ├── bf16/                                  # Native BF16 configs across vllm-tpu, torchtpu, and sglang
│   └── ...                                    # MXFP4 configs across 1h, 2h, and 4h topologies
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
