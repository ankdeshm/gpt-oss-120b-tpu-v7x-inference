# GPT-OSS 120B Benchmark Results on Ironwood TPU (v7x)

Comprehensive single-host (`1 Host = 4 Chips`), dual-host (`2 Hosts = 8 Chips`), and quad-host (`4 Hosts = 16 Chips`) inference benchmark telemetry on **Google Cloud Ironwood TPU (`tpu7x`)** for:
1. **`openai/gpt-oss-120b`**: Native MXFP4 Mixture-of-Experts weights (`~65.3 GB`), 117B total parameters, 5.1B active parameters per token.
2. **`lmsys/gpt-oss-120b-bf16`**: Full unquantized `bfloat16` weights (`~233.7 GB`), 128 routed experts per layer with non-128-aligned dimensions (`hidden_size = 2880`, `intermediate_size = 2880`).

---

## 1. Benchmark Methodology & SLA Definitions

- **Harness**: `vllm bench serve` (`--endpoint /v1/completions`, `--dataset-name random`, `--ignore-eos`, `--trust-remote-code`).
- **Prefix Caching Disabled**: `--no-enable-prefix-caching` on every server and unique random token sequences on every request (`0%` prefix cache hit rate) so prefill compute and HBM bandwidth are measured honestly without synthetic cache amplification.
- **Weight Staging**: Checkpoints staged from GCS to host RAM (`/dev/shm/models/`) with persistent XLA/Pallas compilation caching.
- **Multi-Host Load Balancing (`2` and `4` Hosts)**: Stateless 4-chip single-host replicas fronted by `scripts/lb_proxy.py` (`least_conn` = minimum in-flight requests with round-robin tie-breaking, chunk-by-chunk SSE streaming, `0` failed requests across all runs).
- **Per-Request Goodput SLA Thresholds** (`--goodput ttft:500 tpot:10 e2el:60000`):
  - **TTFT**: `<= 500 ms`
  - **TPOT**: `<= 10 ms` (`>= 100 tok/s` per stream)
  - **End-to-End Latency (E2EL)**: `<= 60,000 ms` (`60 s`, including `4,000`-token long-form generation)
- **Precision Scope**: BF16 weights with a BF16 KV cache are the primary configuration. MXFP4-expert (`openai/gpt-oss-120b`) results and all FP8-KV results are optional (non-BF16) and are labeled as such.
- **BF16 Checkpoint**: `lmsys/gpt-oss-120b-bf16` is a BF16 conversion of the `openai/gpt-oss-120b` MXFP4 checkpoint (MoE expert weights dequantized to BF16).
- **Sample Size (`n`)**: `n` is `--num-prompts` for a cell; every request completed (`0` failed), so `n` also equals the completed-request count in the §6 tables. The customer workload specifies `n=50`; `22k / 4k` cells used `n=10–128` (1-host BF16 runs: `n=12 / 30 / 36 / 48` at `CON=1 / 6 / 16 / 32` and `n=28–40` open-loop).
- **Derived Values**: in the benchmark tables, ratios, percentages and per-chip throughput (out tok/s ÷ chips) are computed from the raw cells and are marked as derived; all other throughput and latency numbers are raw `vllm bench serve` fields from `results/`.

---

## 2. Optional (non-BF16): Multi-Host Horizontal Scaling Summary (`openai/gpt-oss-120b` MXFP4, `vllm-tpu` JAX `TP=2, DP=4, FP8 KV`)

Each `tpu7x` host (`4` physical chips = `8` TensorCores) runs `4` data-parallel `TP=2` instances (`16` total DP ranks at `4` hosts = `16` chips).

### 2.1 Peak Throughput & Goodput Scaling (`1` vs `2` vs `4` Hosts)

| Workload (`ISL : OSL`) | Metric | `1 Host` (`4` Chips) | `2 Hosts` (`8` Chips) | `4 Hosts` (`16` Chips) | `4-Host` / `1-Host` (derived, highest vs highest) |
|---|---|---:|---:|---:|---:|
| **`1,024 : 21`** | Request Throughput (`req/s`) | `20.88` (`CON=8`, `n=32`) | `58.19` (`CON=32`, `n=128`) | **`74.94`** (`CON=32`, `n=128`) | **`3.59x`** |
| **`1,024 : 21`** | SLA Goodput (`req/s`) | `14.36` (`CON=8`, `n=32`) | `31.73` (`CON=8`, `n=64`) | **`48.01`** (`CON=32`, `n=128`) | **`3.34x`** |
| **`1,024 : 1,024`** (Peak) | Output Throughput (`tok/s`) | `5,847.0` (`CON=64`, `n=128`) | `14,277.1` (`CON=256`, `n=512`) | **`20,366.9`** (`CON=256`, `n=512`) | **`3.48x`** |
| **`1,024 : 1,024`** (SLA) | Strict SLA Goodput (`req/s`) | `3.340` (`CON=32`, `n=128`) | `5.054` (`CON=64`, `n=192`) | **`8.584`** (`CON=128`, `n=256`) | **`2.57x`** |
| **`8,192 : 1,024`** (Peak) | Output / Total Throughput (`tok/s`) | `2,696.6` / `24,269.5` (`CON=64`, `n=96`) | `6,533.7` / `58,803.6` (`CON=128`, `n=256`) | **`9,820.8` / `88,387.3`** (`CON=128`, `n=256`) | **`3.64x`** |
| **`22,000 : 4,000`** (Closed-Loop Peak) | Output Throughput (`tok/s`) | `1,789.4` (`CON=32`, `n=50`) | `3,315.7` (`CON=64`, `n=96`) | **`8,749.7`** (`56,873.2` total) (`CON=128`, `n=128`) | **`4.89x`** |
| **`22,000 : 4,000`** (Open-Loop `6.0 RPS`) | Achieved `req/s` (`Out tok/s`) | `0.662` (`2,647.0`) (`n=50`) | `1.032` (`4,127.5`) (`n=60`) | **`1.391`** (**`5,564.6`**) (`n=60`) | **`2.10x`** |

### 2.2 Long-Context (`22,000` Input / `4,000` Output) Latency Reduction Across Scale

| Load Regime | `1 Host` (`4` Chips) TTFT P50 / TPOT P50 / E2E P50 | `2 Hosts` (`8` Chips) TTFT P50 / TPOT P50 / E2E P50 | `4 Hosts` (`16` Chips) TTFT P50 / TPOT P50 / E2E P50 | `n` (1 / 2 / 4 Hosts) |
|---|---|---|---|---:|
| **Closed-Loop `CON=6`** | `2,768.8 ms` / `6.74 ms` / `30.28 s` | `2,468.8 ms` / `5.76 ms` / `25.46 s` | **`1,288.3 ms` / `5.73 ms` / `25.17 s`** | `50 / 48 / 48` |
| **Closed-Loop `CON=16`** | `6,543.5 ms` / `8.24 ms` / `39.88 s` | `4,160.5 ms` / `6.59 ms` / `30.44 s` | **`2,491.0 ms` / `5.77 ms` / `25.48 s`** | `50 / 64 / 64` |
| **Closed-Loop `CON=32`** | `11,287.9 ms` / `11.66 ms` / `58.74 s` | `7,264.1 ms` / `8.24 ms` / `40.10 s` | **`4,163.1 ms` / `6.56 ms` / `30.43 s`** | `50 / 64 / 80` |
| **Open-Loop `RPS=1.0`** | `1,702.0 ms` / `16.32 ms` / `67.45 s` | `1,323.6 ms` / `11.72 ms` / `48.57 s` | **`1,287.3 ms` / `8.18 ms` (`P95 9.05 ms`) / `34.02 s`** | `50 / 48 / 48` |
| **Open-Loop `RPS=6.0`** | `13,247.6 ms` / `14.76 ms` / `71.17 s` | `3,600.3 ms` / `11.27 ms` / `50.51 s` | **`1,746.9 ms` / `8.82 ms` (`P95 9.96 ms`) / `37.90 s`** | `50 / 60 / 60` |

---

## 3. Optional (non-BF16): Single-Host (`4` Chips) MXFP4 Engine & Speculative Decoding Comparison (FP8 KV)

| Engine & Configuration (`1 Host = 4 Chips`) | `22k / 4k` `CON=1` TTFT P50 (ms) | `22k / 4k` `CON=1` TPOT P50 (ms) | `22k / 4k` `CON=1` E2E P50 (s) | `n` | `22k / 4k` Peak Out tok/s | `1k / 1k` `CON=1` TTFT / TPOT P50 | `1k / 1k` Peak Out tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`vllm-tpu` (JAX) `TP=2, DP=4, FP8 KV`** | `1,279.3` | `5.70` | `24.06` | `20` | `2,647.0` (`OL`, `RPS=6.0`, `n=50`) / `1,789.4` (`CL`, `CON=32`, `n=50`) | `51.3 ms` / `5.49 ms` (`n=32`) | `5,847.0` (`CON=64`, `n=128`) |
| **`sglang-torchtpu` `TP=8, EP=8, FP8 KV`** | `737.0` | `4.09` | `17.08` | `16` | `1,242.2` (`CL`, `CON=16`, `n=48`) | `44.5 ms` / `4.05 ms` (`n=32`) | `2,587.5` (`CON=32`, `n=128`) |
| **`sglang-torchtpu` `TP=8, EP=8` + `EAGLE3` (`steps=3, topk=1, draft=4`), FP8 KV** | `746.7` | `5.05` | `20.95` | `16` | `1,052.6` (`OL`, `RPS=2.0`, `n=40`) / `977.7` (`CL`, `CON=32`, `n=48`) | `44.4 ms` / `4.63 ms` (`n=32`) | `2,132.5` (`CON=32`, `n=128`) |

---

## 4. Single-Host (`4` Chips) & Dual-Host (`8` Chips) BF16-Weight (`lmsys/gpt-oss-120b-bf16`) Serving Stack Analysis

`lmsys/gpt-oss-120b-bf16` is a BF16 conversion of the `openai/gpt-oss-120b` MXFP4 checkpoint (MoE experts dequantized to BF16) and stores all `128` routed experts in unquantized 16-bit `bfloat16` (`~233.7 GB` total checkpoint footprint in TPU HBM3e). Because each 4-chip Ironwood host possesses **`768 GB` aggregate HBM3e** (`192 GB` per chip @ `7.4 TB/s`), the full unquantized model fits comfortably inside a single host with massive capacity remaining for KV cache.

### 4.1 Native BF16 Serving Stack Overview & Runtime Patches

1. **`vllm-tpu` (JAX / `tpu-inference`)**:
   - Runs natively at `TP=8, --enable-expert-parallel` with both `bfloat16` (`--kv-cache-dtype=auto`) and `fp8` (`--kv-cache-dtype=fp8`, optional non-BF16) KV caches.
   - Highest measured `1k/1k` output throughput at 2 hosts: **`5,567.6 tok/s`** (`CON=128`, BF16 KV). At 1 host: **`2,463.7 tok/s`** (`CON=32`, BF16 KV; `2,540.3 tok/s` with FP8 KV, optional).
2. **`vllm-torchtpu` (PyTorch / XLA with `scripts/patches/apply_vllm_torchtpu_bf16_patches.py`)**:
   - Upstream unquantized MoE reads non-128-aligned `hidden_size = 2880`. The runtime patch pads `hidden_size` (`2880 -> 2944`, multiple of 128) on `w13_weight`, `w2_weight`, and `w2_bias` in `VllmUnquantizedFusedMoEMethod.process_weights_after_loading`.
   - **Performance Impact**: Eliminates Pallas out-of-bounds lane reads. At `22k/4k` `CON=1` (`n=12`, BF16 KV, identical server arguments), TPOT P50 is **`18.83 ms`** without the patch (`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T074125Z`; every sanity prompt returned only `!` tokens) and **`5.95 ms`** with it (`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T100207Z`): **`3.2x`** lower (derived).
3. **`sglang-torchtpu` (PyTorch / Pallas with `scripts/patches/apply_sglang_bf16_patches.py`)**:
   - Upstream SGLang expects fused expert tensors transposed and deinterleaved. The runtime patch maps `gate_up_proj` and `down_proj`, transposes `[-2, -1]` to match `FusedMoE`, and deinterleaves `swigluoai` gate/up channels (`::2`, `1::2`) prior to 128-boundary tile padding.
   - **Latency Champion**: Achieves the fastest single-stream decode (**`4.97 ms` TPOT**, `201.2 tok/s` per stream) and fastest prefill (**`739.5 ms` TTFT** for `22,000` tokens).
4. **`sglang-torchtpu` + `EAGLE3` Speculative Decoding**:
   - Evaluates `lmsys/EAGLE3-gpt-oss-120b-bf16` speculative decoding draft head (`steps=3, draft=4`). Delivers `5.48 ms` single-stream TPOT and `762.8 ms` TTFT.

### 4.2 Cross-Stack Benchmark Comparison on 1 Host (`4` Chips / `8` TensorCores)

All configurations run on `1 Host = 4 Chips` (`tpu7x-standard-4t`, `768 GB` HBM3e) under identical uncached workloads.

| Stack | Layout · KV Cache | `22k / 4k` `CON=1` TTFT P50 (`n=12`) | `22k / 4k` `CON=1` TPOT P50 (`n=12`) | `22k / 4k` `CON=1` E2E P50 (`n=12`) | `22k / 4k` Closed-Loop Peak Out tok/s (`CON=32`, `n=48`) | `1k / 1k` `CON=1` TTFT / TPOT P50 (`n=24`) | `1k / 1k` Peak Out tok/s (`CON=32`, `n=96`) | `8k / 1k` Peak Out tok/s (`CON=32`, `n=64`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`sglang-torchtpu`** (patched) | `TP=8, EP=8` · BF16 KV | `739.5 ms` | `4.97 ms` | `20.6 s` | `1,150.1` | `67.7 ms` / `4.85 ms` | `2,530.8` | `1,641.9` |
| **`sglang-torchtpu` + `EAGLE3`** (patched) | `TP=8, EP=8` · BF16 KV | `762.8 ms` | `5.48 ms` | `22.7 s` | `934.1` | `66.8 ms` / `5.32 ms` | `2,059.0` | `1,333.7` |
| **`vllm-torchtpu`** (patched) | `TP=8, DP=1, EP=true` · BF16 KV | `878.7 ms` | `5.95 ms` | `24.7 s` | `733.6` | `45.7 ms` / `5.30 ms` | `2,243.0` | `1,223.2` |
| **`vllm-tpu`** (JAX) | `TP=8, DP=1, EP=true` · BF16 KV | `880.5 ms` | `8.54 ms` | `35.0 s` | `1,249.9` | `40.6 ms` / `8.21 ms` | `2,463.7` | `1,557.4` |
| **`vllm-torchtpu`** (patched) | `TP=8, DP=1, EP=true` · FP8 KV — optional (non-BF16) | `877.1 ms` | `5.96 ms` | `24.7 s` | `739.8` | `45.1 ms` / `5.30 ms` | `2,284.8` | `1,243.1` |
| **`vllm-tpu`** (JAX) | `TP=8, DP=1, EP=true` · FP8 KV — optional (non-BF16) | `880.3 ms` | `8.48 ms` | `34.8 s` | `1,302.6` | `41.1 ms` / `8.26 ms` | `2,540.3` | `1,582.9` |

### 4.3 Dual-Host (`2 Hosts = 8 Chips`) BF16-Weight Horizontal Scaling Analysis

Across `2 Hosts = 8 Chips` (two `tpu7x-standard-4t` hosts, `1.54 TB` aggregate HBM3e) fronted by `scripts/lb_proxy.py` (`least_conn`). Ratios compare cells at the same per-replica concurrency (2-host `CON` = 2 × 1-host `CON`) or at the same offered load, and are derived.

#### A. `sglang-torchtpu`, BF16 KV (1 vs 2 Replicas of `TP=8, EP=8`)

| Workload (`ISL : OSL`) | Metric | `1 Host` (`4` Chips) | `2 Hosts` (`8` Chips) | `2-Host` / `1-Host` (derived) |
|---|---|---:|---:|---:|
| **`1,024 : 1,024`** | Output tok/s, 32 in-flight requests per replica | `2,530.8` (`CON=32`, `n=96`) | `5,084.5` (`CON=64`, `n=192`) | `2.01x` |
| **`1,024 : 1,024`** | Total tok/s (same cells) | `5,061.7` | `10,169.0` | `2.01x` |
| **`8,192 : 1,024`** | Output tok/s, 32 in-flight requests per replica | `1,641.9` (`CON=32`, `n=64`) | `3,230.7` (`CON=64`, `n=128`) | `1.97x` |
| **`8,192 : 1,024`** | Total tok/s (same cells) | `14,776.7` | `29,076.1` | `1.97x` |
| **`22,000 : 4,000`** | Closed-loop output tok/s, 32 in-flight requests per replica | `1,150.1` (`CON=32`, `n=48`) | `2,234.1` (`CON=64`, `n=96`) | `1.94x` |
| **`22,000 : 4,000`** | Open-loop output tok/s, same offered load (`RPS=4.0`) | `1,223.2` (`n=40`) | `2,172.7` (`n=60`) | `1.78x` |
| `1,024 : 1,024` | Highest measured output tok/s | `2,530.8` (`CON=32`, `n=96`) | `5,084.5` (`CON=64`, `n=192`) | – |
| `8,192 : 1,024` | Highest measured output tok/s | `1,641.9` (`CON=32`, `n=64`) | `3,281.0` (`CON=128`, `n=256`) | – |
| `22,000 : 4,000` closed loop | Highest measured output tok/s | `1,150.1` (`CON=32`, `n=48`) | `2,433.8` (`CON=32`, `n=64`) | – |
| **`22,000 : 4,000`** | `CON=1` TTFT / TPOT P50 | `739.5 ms` / `4.97 ms` (`n=12`) | `771.4 ms` / `5.05 ms` (`n=10`) | – |

The 2-host run also covered `RPS=6.0` (`2,193.4` out tok/s, `n=60`) and `CON=128` cells; the 1-host run stops at `RPS=4.0` and `CON=32`, so those cells have no 1-host counterpart.

#### B. `vllm-tpu` (JAX), BF16 KV (1 vs 2 Replicas of `TP=8, DP=1, EP=true`)

| Workload (`ISL : OSL`) | Metric | `1 Host` (`4` Chips) | `2 Hosts` (`8` Chips) | `2-Host` / `1-Host` (derived) |
|---|---|---:|---:|---:|
| **`1,024 : 1,024`** | Output tok/s, 32 in-flight requests per replica | `2,463.7` (`CON=32`, `n=96`) | `4,437.3` (`CON=64`, `n=192`) | `1.80x` |
| **`1,024 : 1,024`** | Total tok/s (same cells) | `4,927.4` | `8,874.7` | `1.80x` |
| **`8,192 : 1,024`** | Output tok/s, 32 in-flight requests per replica | `1,557.4` (`CON=32`, `n=64`) | `2,450.3` (`CON=64`, `n=128`) | `1.57x` |
| **`8,192 : 1,024`** | Total tok/s (same cells) | `14,016.5` | `22,052.6` | `1.57x` |
| **`22,000 : 4,000`** | Closed-loop output tok/s, 32 in-flight requests per replica | `1,249.9` (`CON=32`, `n=48`) | `1,460.5` (`CON=64`, `n=96`) | `1.17x` |
| **`22,000 : 4,000`** | Open-loop output tok/s, same offered load (`RPS=4.0`) | `1,434.3` (`n=40`) | `1,511.4` (`n=60`) | `1.05x` |
| `1,024 : 1,024` | Highest measured output tok/s | `2,463.7` (`CON=32`, `n=96`) | `5,567.6` (`CON=128`, `n=256`) | – |
| `8,192 : 1,024` | Highest measured output tok/s | `1,557.4` (`CON=32`, `n=64`) | `2,814.4` (`CON=128`, `n=256`) | – |
| `22,000 : 4,000` closed loop | Highest measured output tok/s | `1,249.9` (`CON=32`, `n=48`) | `1,460.5` (`CON=64`, `n=96`) | – |
| **`22,000 : 4,000`** | `CON=1` TTFT / TPOT P50 | `880.5 ms` / `8.54 ms` (`n=12`) | `882.6 ms` / `5.99 ms` (`n=10`) | – |

The 2-host run also covered `RPS=6.0` (`1,507.1` out tok/s, `n=60`) and `CON=128` cells; the 1-host run stops at `RPS=4.0` and `CON=32`. The 2-host `CON=1` TPOT (`5.99 ms`) is lower than the 1-host value (`8.54 ms`); both are raw measurements.

### 4.4 BF16 Weights (`234 GB`) vs MXFP4 Experts (`65 GB`, optional non-BF16) Architectural Comparison

| Dimension | MXFP4 Experts (`openai/gpt-oss-120b`) — optional (non-BF16) | BF16 Weights (`lmsys/gpt-oss-120b-bf16`) | Architectural Takeaway |
|---|---|---|---|
| **Checkpoint Size** | `~65.3 GB` | `~233.7 GB` (`3.58x` larger) | Both fit comfortably on `1 Host` (`768 GB` HBM3e) |
| **Best 1-Host `22k/4k` Single-Stream TPOT** (`CON=1`) | **`4.09 ms`** (`sglang-torchtpu`, FP8 KV) | **`4.97 ms`** (`sglang-torchtpu`, BF16 KV) | BF16 TPOT is `1.22x` the MXFP4 TPOT (derived; the KV cache dtype also differs) |
| **Best 1-Host `22k/4k` Single-Stream TTFT** (`CON=1`) | **`737.0 ms`** (`sglang-torchtpu`, FP8 KV) | **`739.5 ms`** (`sglang-torchtpu`, BF16 KV) | Prefill is matrix-multiply compute bound: **identical TTFT** (`< 0.4%` delta, derived) |
| **Best 1-Host `1k/1k` Peak Out tok/s** | **`5,847.0`** (`vllm-tpu` `TP=2, DP=4`, FP8 KV, `CON=64`) | **`2,540.3`** (`vllm-tpu` `TP=8, EP=true`, FP8 KV — optional, `CON=32`); **`2,530.8`** with BF16 KV (`sglang-torchtpu` `TP=8, EP=8`, `CON=32`) | MXFP4 allows `4x` DP ranks on 1 Host; BF16 runs `TP=8` full-host sharding |
| **Best 2-Host `1k/1k` Peak Out tok/s** | **`14,277.1`** (`vllm-tpu` `2x TP=2, DP=4`, FP8 KV, `CON=256`) | **`5,567.6`** (`vllm-tpu` `2x TP=8, EP=true`, BF16 KV, `CON=128`) | At the same per-replica concurrency, 2-host / 1-host is `1.76x` (MXFP4, FP8 KV) and `1.80x` (BF16, BF16 KV) (derived) |
| **Best 1-Host `22k/4k` Peak Out tok/s** (closed loop) | **`1,789.4`** (`vllm-tpu` `TP=2, DP=4`, FP8 KV, `CON=32`) | **`1,302.6`** (`vllm-tpu` `TP=8, EP=true`, FP8 KV — optional, `CON=32`); **`1,249.9`** with BF16 KV (`CON=32`) | With FP8 KV on both, BF16 weights deliver `72.8%` of the MXFP4 closed-loop throughput (derived) |

---

## 5. Measured SLA Goodput at `22,000 / 4,000`

For each configuration: the highest measured goodput (requests that met `TTFT <= 500 ms`, `TPOT <= 10 ms` and `E2EL <= 60 s`) across the closed-loop (`CON`) and open-loop (`RPS`) `22k / 4k` cells, the cell where it was reached, and the P95 verdict for each SLA in that cell (E2E target `40–60 s`, evaluated as P95 `<= 60 s`). Goodput out tok/s/chip is derived (goodput req/s × `4,000` output tokens ÷ chips). Where no cell had a request that met all three SLAs, the verdicts are shown for the `CON=1` cell.

| Configuration | Chips | Max Goodput (`req/s`) | Goodput Out tok/s/chip (derived) | Reached At | TTFT P95 `<= 500 ms` | TPOT P95 `<= 10 ms` | E2E P95 `<= 60 s` |
|---|---:|---:|---:|---|---|---|---|
| `sglang-torchtpu` `TP=8, EP=8`, BF16 weights + BF16 KV | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=12`) | missed (`742.9 ms`) | met (`5.06 ms`) | met (`20.96 s`) |
| `sglang-torchtpu` `TP=8, EP=8` + `EAGLE3`, BF16 weights + BF16 KV | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=12`) | missed (`774.5 ms`) | met (`5.96 ms`) | met (`24.59 s`) |
| `vllm-torchtpu` `TP=8, DP=1, EP=true`, BF16 weights + BF16 KV | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=12`) | missed (`890.5 ms`) | met (`5.99 ms`) | met (`24.84 s`) |
| `vllm-tpu` `TP=8, DP=1, EP=true`, BF16 weights + BF16 KV | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=12`) | missed (`927.9 ms`) | met (`8.70 ms`) | met (`35.67 s`) |
| `2x sglang-torchtpu` `TP=8, EP=8`, BF16 weights + BF16 KV | `8` | `0` | `0` | none (verdicts at `CON=1`, `n=10`) | missed (`790.8 ms`) | met (`5.16 ms`) | met (`21.43 s`) |
| `2x vllm-tpu` `TP=8, DP=1, EP=true`, BF16 weights + BF16 KV | `8` | `0` | `0` | none (verdicts at `CON=1`, `n=10`) | missed (`889.0 ms`) | met (`6.05 ms`) | met (`25.06 s`) |
| BF16 weights + BF16 KV, 4 hosts | `16` | not measured | not measured | – | not measured | not measured | not measured |
| `vllm-torchtpu` `TP=8, DP=1, EP=true`, BF16 weights + FP8 KV — optional (non-BF16) | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=12`) | missed (`882.1 ms`) | met (`6.01 ms`) | met (`24.89 s`) |
| `vllm-tpu` `TP=8, DP=1, EP=true`, BF16 weights + FP8 KV — optional (non-BF16) | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=12`) | missed (`924.5 ms`) | met (`8.74 ms`) | met (`35.82 s`) |
| `vllm-tpu` `TP=2, DP=4`, MXFP4 experts + FP8 KV — optional (non-BF16) | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=20`) | missed (`1,283.1 ms`) | met (`5.71 ms`) | met (`24.14 s`) |
| `sglang-torchtpu` `TP=8, EP=8`, MXFP4 experts + FP8 KV — optional (non-BF16) | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=16`) | missed (`742.1 ms`) | met (`4.14 ms`) | met (`17.29 s`) |
| `sglang-torchtpu` `TP=8, EP=8` + `EAGLE3`, MXFP4 experts + FP8 KV — optional (non-BF16) | `4` | `0` | `0` | none (verdicts at `CON=1`, `n=16`) | missed (`756.8 ms`) | met (`6.07 ms`) | met (`25.00 s`) |
| `2x vllm-tpu` `TP=2, DP=4`, MXFP4 experts + FP8 KV — optional (non-BF16) | `8` | `0` | `0` | none (verdicts at `CON=1`, `n=10`) | missed (`1,400.8 ms`) | met (`5.71 ms`) | met (`24.18 s`) |
| `4x vllm-tpu` `TP=2, DP=4`, MXFP4 experts + FP8 KV — optional (non-BF16) | `16` | `0` | `0` | none (verdicts at `CON=1`, `n=10`) | missed (`1,397.1 ms`) | met (`5.72 ms`) | met (`24.25 s`) |

At `22k / 4k`, every measured configuration missed the `500 ms` TTFT SLA, including at `CON=1`, while TPOT and E2E were met at `CON=1`.

---

## 6. Raw Benchmark Telemetry & SLA Tables

### 6.1 Optional (non-BF16): MXFP4 Experts (`openai/gpt-oss-120b`) with FP8 KV

### `gptoss120b_1h_4c_vllmtpu_fp8kv_tp2dp4`
- **Model**: `openai/gpt-oss-120b` | **Framework**: `vllm-tpu (JAX / tpu-inference)`
- **Precision**: `mxfp4 moe / bf16 attn, fp8 kv` | **Topology**: `1 host (4 chips / 8 TensorCores), TP=2, DP=4` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `5.041` | `5.041` | **`105.9`** | **`26.5`** | `5,268.3` | `48.9 / 378.9 / 402.4` | `5.37 / 5.45 / 5.55` | `5.44 / 5.85` | `0.16 / 0.49` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `20.884` | `14.357` | **`438.6`** | **`109.6`** | `21,823.4` | `192.1 / 338.2 / 343.4` | `8.27 / 15.21 / 19.48` | `5.85 / 36.55` | `0.38 / 0.48` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `20` | `0.042` | `0.000` | **`166.3`** | **`41.6`** | `1,080.7` | `1,279.3 / 1,283.1 / 1,284.3` | `5.70 / 5.71 / 5.72` | `5.69 / 6.02` | `24.06 / 24.14` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `50` | `0.187` | `0.000` | **`749.1`** | **`187.3`** | `4,868.9` | `2,768.8 / 4,797.9 / 4,804.0` | `6.74 / 7.28 / 7.37` | `6.26 / 6.88` | `30.28 / 31.53` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `50` | `0.345` | `0.000` | **`1,379.5`** | **`344.9`** | `8,966.9` | `6,543.5 / 9,602.4 / 9,664.8` | `8.24 / 9.62 / 9.65` | `7.45 / 8.21` | `39.88 / 40.40` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `50` | `0.447` | `0.000` | **`1,789.4`** | **`447.4`** | `11,631.3` | `11,287.9 / 18,780.6 / 19,141.6` | `11.66 / 13.53 / 14.15` | `9.45 / 10.70` | `58.74 / 62.06` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `32` | `0.324` | `0.000` | **`1,295.3`** | **`323.8`** | `8,419.5` | `1,297.1 / 2,156.8 / 2,215.5` | `12.36 / 13.55 / 13.73` | `8.19 / 10.07` | `50.69 / 55.76` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `50` | `0.516` | `0.000` | **`2,064.5`** | **`516.1`** | `13,419.3` | `1,702.0 / 3,132.9 / 3,669.0` | `16.32 / 19.32 / 19.66` | `11.43 / 13.14` | `67.45 / 78.64` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `50` | `0.625` | `0.000` | **`2,498.5`** | **`624.6`** | `16,240.2` | `3,752.7 / 9,308.9 / 10,428.5` | `15.41 / 18.56 / 19.02` | `11.66 / 13.08` | `67.14 / 76.38` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | `50` | `0.662` | `0.000` | **`2,647.0`** | **`661.7`** | `17,205.3` | `13,247.6 / 17,157.9 / 18,038.1` | `14.76 / 17.70 / 18.32` | `11.81 / 13.78` | `71.17 / 74.38` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `32` | `0.176` | `0.176` | **`180.7`** | **`45.2`** | `361.4` | `51.3 / 52.5 / 52.7` | `5.49 / 5.52 / 5.52` | `5.48 / 5.84` | `5.67 / 5.69` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `64` | `1.263` | `1.263` | **`1,293.6`** | **`323.4`** | `2,587.3` | `71.2 / 176.1 / 215.6` | `6.03 / 6.44 / 6.48` | `5.91 / 6.81` | `6.25 / 6.71` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `128` | `3.562` | `3.340` | **`3,647.8`** | **`911.9`** | `7,295.6` | `217.5 / 601.8 / 632.7` | `8.50 / 8.96 / 9.00` | `8.00 / 9.13` | `9.02 / 9.30` | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | `128` | `5.710` | `0.000` | **`5,847.0`** | **`1,461.7`** | `11,694.0` | `443.8 / 966.3 / 968.4` | `10.35 / 10.64 / 10.80` | `9.80 / 11.33` | `11.10 / 11.56` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.167` | `0.167` | **`170.9`** | **`42.7`** | `1,538.5` | `300.5 / 303.2 / 303.5` | `5.56 / 5.58 / 5.59` | `5.55 / 5.90` | `5.99 / 6.01` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `48` | `0.998` | `0.416` | **`1,022.2`** | **`255.5`** | `9,199.6` | `620.9 / 1,273.8 / 1,298.3` | `7.14 / 7.78 / 7.80` | `6.10 / 6.99` | `7.98 / 8.38` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `2.420` | `0.000` | **`2,478.6`** | **`619.6`** | `22,307.1` | `2,663.5 / 4,379.9 / 4,381.6` | `10.29 / 12.37 / 12.48` | `8.60 / 9.72` | `13.11 / 14.38` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | `96` | `2.633` | `0.000` | **`2,696.6`** | **`674.2`** | `24,269.5` | `4,563.5 / 8,449.4 / 8,468.9` | `15.34 / 17.80 / 18.49` | `10.96 / 12.30` | `22.11 / 23.11` | Partial (E2E) |


### `gptoss120b_2h_8c_vllmtpu_fp8kv_tp2dp4`
- **Model**: `openai/gpt-oss-120b` | **Framework**: `vllm-tpu JAX (2x single-host replicas + least_conn LB)`
- **Precision**: `mxfp4 moe / bf16 attn, fp8 kv` | **Topology**: `2 hosts (8 chips / 16 TensorCores), 2x independent replicas` (`8` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `4.172` | `4.172` | **`87.6`** | **`11.0`** | `4,359.6` | `50.5 / 384.6 / 388.3` | `5.37 / 5.53 / 5.59` | `5.43 / 5.91` | `0.16 / 0.49` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `64` | `31.731` | `31.731` | **`666.3`** | **`83.3`** | `33,158.5` | `129.9 / 138.5 / 181.1` | `5.54 / 6.02 / 6.06` | `5.49 / 6.30` | `0.24 / 0.25` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=32` | `128` | `58.186` | `22.274` | **`1,221.9`** | **`152.7`** | `60,804.0` | `248.5 / 650.4 / 674.2` | `10.54 / 16.53 / 28.49` | `6.46 / 51.12` | `0.43 / 0.85` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `10` | `0.041` | `0.000` | **`165.9`** | **`20.7`** | `1,078.2` | `1,389.1 / 1,400.8 / 1,402.5` | `5.68 / 5.71 / 5.71` | `5.68 / 6.02` | `24.11 / 24.18` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `48` | `0.235` | `0.000` | **`940.9`** | **`117.6`** | `6,115.6` | `2,468.8 / 2,547.8 / 2,567.7` | `5.76 / 6.06 / 6.12` | `5.68 / 6.37` | `25.46 / 25.82` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `64` | `0.525` | `0.000` | **`2,099.3`** | **`262.4`** | `13,645.7` | `4,160.5 / 4,895.5 / 4,908.4` | `6.59 / 7.29 / 7.30` | `6.34 / 7.08` | `30.44 / 30.55` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `64` | `0.796` | `0.000` | **`3,182.0`** | **`397.8`** | `20,683.2` | `7,264.1 / 9,666.2 / 9,694.3` | `8.24 / 9.67 / 9.71` | `7.50 / 8.36` | `40.10 / 40.23` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=64` | `96` | `0.829` | `0.000` | **`3,315.7`** | **`414.5`** | `21,551.8` | `12,149.5 / 19,118.4 / 19,182.9` | `11.49 / 13.77 / 14.31` | `9.67 / 10.94` | `59.70 / 61.51` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `36` | `0.368` | `0.000` | **`1,472.3`** | **`184.0`** | `9,570.1` | `1,285.6 / 1,847.0 / 2,025.9` | `9.40 / 10.33 / 10.36` | `6.75 / 7.73` | `38.89 / 42.61` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `48` | `0.596` | `0.000` | **`2,382.3`** | **`297.8`** | `15,485.3` | `1,323.6 / 2,329.5 / 2,374.6` | `11.72 / 13.88 / 14.02` | `8.09 / 9.29` | `48.57 / 56.98` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `56` | `0.863` | `0.000` | **`3,450.7`** | **`431.3`** | `22,429.6` | `1,637.6 / 2,482.2 / 2,954.6` | `12.14 / 14.14 / 14.36` | `9.09 / 10.22` | `50.02 / 57.94` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `60` | `1.011` | `0.000` | **`4,042.8`** | **`505.3`** | `26,278.0` | `3,781.8 / 6,554.5 / 7,906.7` | `11.82 / 13.85 / 13.97` | `9.35 / 10.81` | `51.70 / 56.74` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | `60` | `1.032` | `0.000` | **`4,127.5`** | **`515.9`** | `26,828.6` | `3,600.3 / 9,928.5 / 10,478.3` | `11.27 / 13.81 / 13.94` | `9.38 / 10.64` | `50.51 / 56.83` | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `64` | `1.371` | `1.371` | **`1,403.5`** | **`175.4`** | `2,807.1` | `125.7 / 159.6 / 164.7` | `5.56 / 5.67 / 5.68` | `5.46 / 6.35` | `5.81 / 5.87` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `128` | `4.165` | `4.165` | **`4,265.0`** | **`533.1`** | `8,529.9` | `142.1 / 345.8 / 348.8` | `7.33 / 7.78 / 7.83` | `6.71 / 8.06` | `7.60 / 8.22` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | `192` | `6.882` | `5.054` | **`7,047.6`** | **`881.0`** | `14,095.3` | `283.3 / 626.6 / 634.6` | `8.42 / 9.53 / 9.63` | `7.90 / 9.55` | `9.17 / 9.89` | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | `256` | `10.057` | `0.000` | **`10,298.0`** | **`1,287.2`** | `20,595.9` | `563.8 / 1,149.2 / 1,164.4` | `10.39 / 13.49 / 13.61` | `9.85 / 13.63` | `11.42 / 14.02` | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=256` | `512` | `13.943` | `0.000` | **`14,277.1`** | **`1,784.6`** | `28,554.3` | `725.4 / 1,991.2 / 1,999.3` | `16.59 / 17.88 / 18.02` | `14.99 / 18.70` | `18.19 / 18.67` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `48` | `1.263` | `0.316` | **`1,293.1`** | **`161.6`** | `11,638.1` | `621.3 / 664.4 / 666.5` | `5.58 / 5.84 / 5.85` | `5.53 / 6.27` | `6.32 / 6.37` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `96` | `3.299` | `0.619` | **`3,377.9`** | **`422.2`** | `30,401.1` | `1,458.9 / 2,309.5 / 2,315.1` | `7.96 / 9.12 / 9.20` | `7.00 / 7.85` | `9.79 / 10.05` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | `128` | `4.798` | `0.000` | **`4,913.0`** | **`614.1`** | `44,217.4` | `2,507.4 / 4,393.5 / 4,409.4` | `10.51 / 12.41 / 12.58` | `8.60 / 9.73` | `13.16 / 14.61` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | `256` | `6.381` | `0.000` | **`6,533.7`** | **`816.7`** | `58,803.6` | `4,218.3 / 8,451.1 / 8,519.3` | `15.41 / 18.41 / 19.24` | `11.00 / 13.00` | `19.88 / 23.37` | Partial (E2E) |


### `gptoss120b_4h_16c_vllmtpu_fp8kv_tp2dp4`
- **Model**: `openai/gpt-oss-120b` | **Framework**: `vllm-tpu JAX (4x single-host replicas + least_conn LB)`
- **Precision**: `mxfp4 moe / bf16 attn, fp8 kv` | **Topology**: `4 hosts (16 chips / 32 TensorCores), 4x independent replicas` (`16` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `4.176` | `4.176` | **`87.7`** | **`5.5`** | `4,363.8` | `51.6 / 382.5 / 393.1` | `5.36 / 5.58 / 5.62` | `5.45 / 5.83` | `0.16 / 0.49` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `64` | `43.612` | `43.612` | **`915.9`** | **`57.2`** | `45,575.0` | `50.5 / 128.0 / 140.7` | `5.41 / 5.68 / 5.77` | `5.45 / 6.03` | `0.16 / 0.24` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=32` | `128` | `74.938` | `48.007` | **`1,573.7`** | **`98.4`** | `78,309.8` | `126.0 / 504.5 / 538.1` | `8.19 / 14.72 / 20.70` | `5.93 / 38.72` | `0.34 / 0.69` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `10` | `0.041` | `0.000` | **`165.7`** | **`10.4`** | `1,077.0` | `1,381.2 / 1,397.1 / 1,400.4` | `5.70 / 5.72 / 5.72` | `5.70 / 6.02` | `24.14 / 24.25` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `48` | `0.241` | `0.000` | **`963.5`** | **`60.2`** | `6,262.7` | `1,288.3 / 2,171.8 / 2,545.0` | `5.73 / 6.01 / 6.04` | `5.69 / 6.05` | `25.17 / 25.39` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `64` | `0.622` | `0.000` | **`2,486.5`** | **`155.4`** | `16,162.3` | `2,491.0 / 2,586.8 / 2,596.0` | `5.77 / 6.11 / 6.16` | `5.68 / 6.49` | `25.48 / 25.97` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `80` | `0.876` | `0.000` | **`3,505.3`** | **`219.1`** | `22,784.5` | `4,163.1 / 4,916.4 / 4,928.3` | `6.56 / 7.29 / 7.30` | `6.32 / 6.92` | `30.43 / 30.61` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=64` | `96` | `1.198` | `0.000` | **`4,793.8`** | **`299.6`** | `31,159.5` | `7,346.1 / 9,706.7 / 9,724.9` | `8.17 / 9.60 / 9.67` | `7.47 / 8.25` | `39.96 / 40.18` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=128` | `128` | `2.187` | `0.000` | **`8,749.7`** | **`546.9`** | `56,873.2` | `14,099.2 / 19,209.0 / 19,238.8` | `11.09 / 13.80 / 14.19` | `9.56 / 10.68` | `58.25 / 58.44` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `36` | `0.375` | `0.000` | **`1,498.2`** | **`93.6`** | `9,738.3` | `1,288.5 / 1,355.0 / 1,367.9` | `6.69 / 7.49 / 7.53` | `5.70 / 6.55` | `28.02 / 31.32` | Partial (TPOT+E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `48` | `0.648` | `0.000` | **`2,592.7`** | **`162.0`** | `16,852.8` | `1,287.3 / 1,336.8 / 1,387.1` | `8.18 / 9.05 / 9.11` | `6.29 / 7.26` | `34.02 / 37.46` | Partial (TPOT+E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `56` | `0.971` | `0.000` | **`3,884.5`** | **`242.8`** | `25,249.3` | `1,313.2 / 2,068.3 / 2,183.6` | `9.37 / 10.96 / 11.13` | `7.07 / 8.19` | `39.42 / 45.15` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `60` | `1.278` | `0.000` | **`5,110.2`** | **`319.4`** | `33,216.5` | `1,773.8 / 2,368.7 / 2,598.0` | `9.28 / 10.54 / 10.81` | `7.39 / 8.28` | `38.96 / 43.92` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | `60` | `1.391` | `0.000` | **`5,564.6`** | **`347.8`** | `36,169.7` | `1,746.9 / 3,584.3 / 3,907.6` | `8.82 / 9.96 / 10.08` | `7.43 / 8.28` | `37.90 / 41.11` | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `64` | `1.391` | `1.391` | **`1,423.9`** | **`89.0`** | `2,847.8` | `53.3 / 157.4 / 162.9` | `5.51 / 5.55 / 5.58` | `5.47 / 5.93` | `5.69 / 5.81` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `128` | `4.848` | `4.848` | **`4,964.5`** | **`310.3`** | `9,929.1` | `73.0 / 264.8 / 351.8` | `6.26 / 6.49 / 6.54` | `5.93 / 6.84` | `6.51 / 6.76` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | `192` | `8.111` | `7.477` | **`8,305.2`** | **`519.1`** | `16,610.4` | `177.6 / 510.7 / 650.4` | `7.43 / 7.95 / 8.04` | `6.68 / 8.28` | `7.70 / 8.37` | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | `256` | `13.734` | `8.584` | **`14,063.9`** | **`879.0`** | `28,127.8` | `295.5 / 659.7 / 666.2` | `8.33 / 9.18 / 9.34` | `7.89 / 9.55` | `9.07 / 9.54` | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=256` | `512` | `19.890` | `0.000` | **`20,366.9`** | **`1,272.9`** | `40,733.9` | `456.5 / 1,304.0 / 1,337.9` | `10.73 / 13.52 / 13.74` | `9.86 / 14.10` | `11.81 / 14.13` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `48` | `1.257` | `0.629` | **`1,287.5`** | **`80.5`** | `11,587.7` | `489.4 / 652.8 / 655.9` | `5.76 / 5.86 / 5.95` | `5.54 / 5.98` | `6.31 / 6.43` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `96` | `4.039` | `1.220` | **`4,136.0`** | **`258.5`** | `37,224.2` | `895.2 / 1,300.8 / 1,307.1` | `6.61 / 7.52 / 7.74` | `6.07 / 7.06` | `7.82 / 8.09` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | `128` | `6.604` | `0.929` | **`6,762.9`** | **`422.7`** | `60,866.0` | `1,765.1 / 2,337.0 / 2,342.0` | `7.93 / 9.00 / 9.07` | `7.00 / 7.86` | `9.58 / 10.02` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | `256` | `9.591` | `0.000` | **`9,820.8`** | **`613.8`** | `88,387.3` | `2,516.9 / 4,415.4 / 4,429.2` | `10.42 / 12.34 / 12.52` | `8.57 / 9.65` | `13.11 / 14.71` | Partial (E2E) |


### `gptoss120b_1h_4c_sglang_fp8kv_tp8ep8`
- **Model**: `openai/gpt-oss-120b` | **Framework**: `sglang-torchtpu`
- **Precision**: `mxfp4 moe / bf16 attn, fp8_e4m3 kv` | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `7.944` | `7.944` | **`166.8`** | **`41.7`** | `8,301.8` | `43.5 / 46.2 / 75.7` | `4.01 / 4.47 / 4.61` | `4.13 / 4.69` | `0.12 / 0.13` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `19.576` | `11.623` | **`411.1`** | **`102.8`** | `20,456.8` | `230.7 / 292.9 / 294.1` | `6.73 / 18.68 / 18.89` | `6.35 / 9.62` | `0.42 / 0.42` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `16` | `0.059` | `0.000` | **`234.0`** | **`58.5`** | `1,521.3` | `737.0 / 742.1 / 742.1` | `4.09 / 4.14 / 4.14` | `4.03 / 4.61` | `17.08 / 17.29` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `36` | `0.181` | `0.000` | **`724.3`** | **`181.1`** | `4,707.8` | `2,263.3 / 3,845.2 / 3,913.8` | `7.72 / 8.13 / 8.20` | `7.28 / 7.76` | `33.11 / 33.63` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `48` | `0.311` | `0.000` | **`1,242.2`** | **`310.6`** | `8,074.4` | `4,785.0 / 9,108.3 / 10,405.0` | `11.74 / 12.59 / 12.84` | `10.17 / 10.87` | `51.13 / 53.81` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.289` | `0.000` | **`1,154.8`** | **`288.7`** | `7,505.9` | `11,169.4 / 71,000.7 / 72,313.4` | `15.83 / 19.04 / 22.12` | `12.58 / 13.60` | `72.94 / 137.31` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `32` | `0.254` | `0.000` | **`1,014.8`** | **`253.7`** | `6,596.4` | `1,019.9 / 20,036.9 / 23,188.9` | `14.22 / 16.06 / 16.32` | `12.40 / 13.34` | `61.87 / 65.77` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `40` | `0.296` | `0.000` | **`1,184.5`** | **`296.1`** | `7,699.1` | `2,429.6 / 47,924.7 / 48,334.7` | `15.80 / 19.13 / 24.92` | `12.45 / 13.37` | `67.80 / 105.27` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `40` | `0.306` | `0.000` | **`1,225.5`** | **`306.4`** | `7,965.6` | `4,844.9 / 59,126.1 / 59,758.6` | `15.72 / 20.09 / 24.92` | `12.46 / 13.32` | `76.40 / 117.77` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `32` | `0.239` | `0.239` | **`244.8`** | **`61.2`** | `489.6` | `44.5 / 46.7 / 47.9` | `4.05 / 4.14 / 4.15` | `3.95 / 4.73` | `4.19 / 4.28` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `64` | `1.104` | `1.104` | **`1,130.5`** | **`282.6`** | `2,261.1` | `269.6 / 292.9 / 294.3` | `6.85 / 7.06 / 7.08` | `6.80 / 7.15` | `7.26 / 7.30` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `128` | `2.527` | `0.000` | **`2,587.5`** | **`646.9`** | `5,175.0` | `496.1 / 896.5 / 926.6` | `11.86 / 12.25 / 12.39` | `11.52 / 12.22` | `12.66 / 12.77` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.229` | `0.229` | **`235.0`** | **`58.7`** | `2,114.7` | `245.1 / 247.1 / 248.0` | `4.02 / 4.07 / 4.07` | `3.96 / 4.52` | `4.36 / 4.40` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `48` | `0.888` | `0.222` | **`909.7`** | **`227.4`** | `8,187.4` | `996.6 / 1,661.3 / 1,747.9` | `7.82 / 8.59 / 8.67` | `7.10 / 7.48` | `9.02 / 9.15` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.592` | `0.000` | **`1,630.1`** | **`407.5`** | `14,670.9` | `2,403.2 / 6,319.4 / 6,875.9` | `17.15 / 18.96 / 19.21` | `12.95 / 13.79` | `19.97 / 23.86` | Partial (E2E) |


### `gptoss120b_1h_4c_sglang_fp8kv_eagle3_tp8ep8`
- **Model**: `openai/gpt-oss-120b` | **Framework**: `sglang-torchtpu`
- **Precision**: `mxfp4 moe / bf16 attn, fp8_e4m3 kv + EAGLE3 (steps=3, topk=1, draft=4)` | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `6.788` | `6.788` | **`142.5`** | **`35.6`** | `7,093.1` | `44.9 / 46.1 / 52.1` | `5.25 / 5.79 / 6.13` | `6.62 / 7.55` | `0.15 / 0.16` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `16.507` | `0.516` | **`346.6`** | **`86.7`** | `17,249.8` | `94.7 / 236.3 / 236.4` | `18.06 / 25.20 / 26.71` | `14.46 / 83.18` | `0.46 / 0.62` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `16` | `0.047` | `0.000` | **`187.0`** | **`46.8`** | `1,215.6` | `746.7 / 756.8 / 765.5` | `5.05 / 6.07 / 6.09` | `6.86 / 7.38` | `20.95 / 25.00` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `36` | `0.120` | `0.000` | **`478.1`** | **`119.5`** | `3,107.4` | `749.3 / 2,886.3 / 3,890.7` | `11.96 / 13.51 / 14.21` | `15.30 / 15.82` | `48.57 / 56.67` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `48` | `0.243` | `0.000` | **`973.7`** | **`243.4`** | `6,329.1` | `770.6 / 9,201.6 / 10,425.0` | `15.21 / 17.93 / 18.91` | `18.34 / 19.32` | `63.42 / 76.14` | Above SLA |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.244` | `0.000` | **`977.7`** | **`244.4`** | `6,355.1` | `13,538.2 / 77,274.6 / 78,662.6` | `18.09 / 26.48 / 36.40` | `20.75 / 21.60` | `89.28 / 156.05` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `32` | `0.214` | `0.000` | **`855.3`** | **`213.8`** | `5,559.7` | `1,011.5 / 29,128.6 / 34,870.0` | `16.56 / 19.31 / 21.05` | `20.39 / 21.15` | `75.08 / 87.73` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `40` | `0.253` | `0.000` | **`1,011.8`** | **`253.0`** | `6,577.0` | `2,429.7 / 58,831.6 / 59,632.3` | `17.88 / 25.99 / 28.80` | `20.78 / 21.66` | `100.58 / 127.59` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `40` | `0.263` | `0.000` | **`1,052.6`** | **`263.2`** | `6,842.1` | `5,384.6 / 71,861.7 / 72,508.9` | `17.73 / 22.20 / 27.71` | `20.56 / 21.88` | `92.40 / 137.88` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `32` | `0.207` | `0.207` | **`212.1`** | **`53.0`** | `424.1` | `44.4 / 49.0 / 53.0` | `4.63 / 5.44 / 5.78` | `6.77 / 7.53` | `4.79 / 5.62` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `64` | `0.711` | `0.178` | **`728.5`** | **`182.1`** | `1,457.0` | `59.8 / 145.1 / 237.1` | `10.73 / 12.73 / 13.41` | `15.27 / 15.98` | `11.07 / 13.08` | Partial (TTFT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `128` | `2.082` | `0.016` | **`2,132.5`** | **`533.1`** | `4,264.9` | `69.4 / 757.3 / 846.6` | `13.98 / 16.79 / 17.21` | `19.73 / 21.00` | `14.49 / 17.51` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.187` | `0.187` | **`191.8`** | **`47.9`** | `1,726.0` | `246.0 / 246.9 / 247.2` | `5.00 / 5.46 / 5.59` | `6.79 / 7.28` | `5.36 / 5.83` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `48` | `0.586` | `0.012` | **`600.3`** | **`150.1`** | `5,402.8` | `255.8 / 1,272.7 / 1,678.1` | `12.21 / 14.83 / 15.24` | `15.60 / 16.24` | `12.91 / 15.65` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.318` | `0.000` | **`1,349.1`** | **`337.3`** | `12,142.2` | `578.8 / 6,431.3 / 6,990.7` | `20.12 / 26.88 / 28.13` | `21.32 / 22.41` | `22.04 / 30.33` | Partial (E2E) |


### 6.2 BF16 Weights (`lmsys/gpt-oss-120b-bf16`, BF16 Conversion of the MXFP4 Checkpoint)

### `gptoss120b_bf16_1h_4c_vllmtpu_bf16kv_tp8ep`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `vllm-tpu (JAX / tpu-inference)`
- **Precision**: `native bf16 weights (234 GB), bf16 kv` | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `4.625` | `4.625` | **`97.1`** | **`24.3`** | `4,832.9` | `40.1 / 41.5 / 259.5` | `8.30 / 8.42 / 8.49` | `8.50 / 8.88` | `0.21 / 0.21` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `16.582` | `11.400` | **`348.2`** | **`87.1`** | `17,328.7` | `246.9 / 414.2 / 414.5` | `8.55 / 18.68 / 20.70` | `8.59 / 9.33` | `0.47 / 0.59` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `12` | `0.029` | `0.000` | **`114.0`** | **`28.5`** | `741.1` | `880.5 / 927.9 / 968.8` | `8.54 / 8.70 / 8.74` | `8.56 / 9.10` | `35.02 / 35.67` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `30` | `0.155` | `0.000` | **`618.7`** | **`154.7`** | `4,021.3` | `2,992.8 / 4,573.5 / 4,594.6` | `8.95 / 9.51 / 9.55` | `8.51 / 9.17` | `38.88 / 39.15` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `36` | `0.233` | `0.000` | **`932.7`** | **`233.2`** | `6,062.3` | `5,749.2 / 10,923.8 / 11,807.2` | `13.39 / 14.38 / 14.56` | `11.74 / 12.58` | `59.32 / 64.10` | Above SLA |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.312` | `0.000` | **`1,249.9`** | **`312.5`** | `8,124.2` | `6,636.6 / 22,375.3 / 23,949.8` | `21.91 / 22.82 / 23.17` | `17.15 / 18.75` | `96.87 / 106.33` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `28` | `0.248` | `0.000` | **`991.1`** | **`247.8`** | `6,442.4` | `905.4 / 1,812.9 / 2,131.0` | `17.06 / 17.58 / 17.66` | `14.74 / 16.39` | `69.78 / 71.22` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `36` | `0.328` | `0.000` | **`1,311.6`** | **`327.9`** | `8,525.4` | `2,822.5 / 4,723.0 / 5,098.7` | `21.66 / 23.96 / 24.05` | `18.72 / 19.87` | `89.71 / 96.69` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `36` | `0.345` | `0.000` | **`1,381.1`** | **`345.3`** | `8,976.9` | `5,929.2 / 11,629.3 / 12,319.0` | `22.16 / 25.06 / 25.19` | `18.97 / 20.29` | `94.53 / 101.15` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `40` | `0.359` | `0.000` | **`1,434.3`** | **`358.6`** | `9,323.3` | `10,711.3 / 19,647.5 / 20,222.3` | `23.83 / 27.01 / 27.34` | `20.16 / 21.66` | `106.01 / 109.53` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `24` | `0.119` | `0.119` | **`122.0`** | **`30.5`** | `243.9` | `40.6 / 41.4 / 42.4` | `8.21 / 8.33 / 8.38` | `8.17 / 8.59` | `8.44 / 8.56` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `48` | `0.923` | `0.923` | **`945.1`** | **`236.3`** | `1,890.2` | `221.6 / 298.9 / 299.2` | `8.26 / 8.41 / 8.50` | `8.18 / 8.70` | `8.65 / 8.76` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `96` | `2.406` | `0.000` | **`2,463.7`** | **`615.9`** | `4,927.4` | `683.8 / 840.2 / 861.3` | `12.33 / 12.89 / 12.97` | `12.18 / 12.79` | `13.31 / 13.40` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.117` | `0.117` | **`119.9`** | **`30.0`** | `1,079.5` | `263.6 / 267.9 / 268.0` | `8.10 / 8.25 / 8.30` | `8.11 / 8.51` | `8.54 / 8.71` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `40` | `0.764` | `0.076` | **`782.7`** | **`195.7`** | `7,044.7` | `1,311.5 / 1,784.7 / 1,814.7` | `8.99 / 9.92 / 10.05` | `8.46 / 8.89` | `10.49 / 10.61` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.521` | `0.000` | **`1,557.4`** | **`389.3`** | `14,016.5` | `2,030.9 / 6,307.4 / 6,695.9` | `18.90 / 19.69 / 20.15` | `14.07 / 15.04` | `20.96 / 25.64` | Partial (E2E) |


### `gptoss120b_bf16_1h_4c_vllmtpu_fp8kv_tp8ep`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `vllm-tpu (JAX / tpu-inference)`
- **Precision**: `native bf16 weights (234 GB), fp8 kv` — optional (non-BF16) | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `4.578` | `4.578` | **`96.1`** | **`24.0`** | `4,783.7` | `40.3 / 41.7 / 264.3` | `8.37 / 8.64 / 8.81` | `8.54 / 9.32` | `0.21 / 0.22` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `18.065` | `13.549` | **`379.4`** | **`94.8`** | `18,878.0` | `225.4 / 406.9 / 407.3` | `8.59 / 17.69 / 19.49` | `8.76 / 9.70` | `0.40 / 0.58` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `12` | `0.029` | `0.000` | **`114.4`** | **`28.6`** | `743.7` | `880.3 / 924.5 / 963.4` | `8.48 / 8.74 / 8.76` | `8.51 / 9.16` | `34.84 / 35.82` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `30` | `0.151` | `0.000` | **`603.9`** | **`151.0`** | `3,925.2` | `2,999.1 / 4,567.9 / 4,599.0` | `9.19 / 9.76 / 9.82` | `8.76 / 9.46` | `39.73 / 40.25` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `36` | `0.235` | `0.000` | **`941.7`** | **`235.4`** | `6,120.8` | `5,743.0 / 10,913.9 / 11,805.9` | `13.00 / 14.11 / 14.28` | `11.44 / 12.19` | `58.36 / 62.19` | Above SLA |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.326` | `0.000` | **`1,302.6`** | **`325.6`** | `8,466.8` | `6,633.7 / 22,351.7 / 23,919.2` | `20.91 / 21.83 / 22.18` | `16.25 / 17.55` | `92.90 / 102.33` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `28` | `0.252` | `0.000` | **`1,008.4`** | **`252.1`** | `6,554.3` | `903.5 / 1,821.7 / 2,141.3` | `16.59 / 17.30 / 17.40` | `14.07 / 16.08` | `68.05 / 70.20` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `36` | `0.330` | `0.000` | **`1,321.9`** | **`330.5`** | `8,592.2` | `2,794.4 / 4,692.2 / 5,057.8` | `21.44 / 23.70 / 23.80` | `18.51 / 19.61` | `88.82 / 95.67` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `36` | `0.352` | `0.000` | **`1,406.4`** | **`351.6`** | `9,141.3` | `5,902.6 / 11,579.0 / 12,270.2` | `21.69 / 24.60 / 24.73` | `18.34 / 20.00` | `92.66 / 99.31` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `40` | `0.372` | `0.000` | **`1,487.5`** | **`371.9`** | `9,668.6` | `10,683.6 / 19,583.3 / 20,154.2` | `22.84 / 26.02 / 26.35` | `19.28 / 20.52` | `102.03 / 105.57` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `24` | `0.118` | `0.118` | **`120.7`** | **`30.2`** | `241.3` | `41.1 / 42.9 / 45.3` | `8.26 / 8.36 / 8.38` | `8.22 / 8.77` | `8.49 / 8.59` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `48` | `0.890` | `0.890` | **`911.8`** | **`227.9`** | `1,823.6` | `223.6 / 293.7 / 298.1` | `8.49 / 8.88 / 8.89` | `8.49 / 9.01` | `8.91 / 9.31` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `96` | `2.481` | `0.000` | **`2,540.3`** | **`635.1`** | `5,080.6` | `657.1 / 954.3 / 955.4` | `12.02 / 12.44 / 12.60` | `11.74 / 12.35` | `12.87 / 12.99` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.115` | `0.115` | **`117.5`** | **`29.4`** | `1,057.8` | `266.5 / 270.0 / 270.9` | `8.30 / 8.37 / 8.37` | `8.24 / 8.73` | `8.76 / 8.83` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `40` | `0.759` | `0.038` | **`777.3`** | **`194.3`** | `6,996.0` | `1,375.9 / 1,786.8 / 1,819.2` | `8.96 / 10.00 / 10.04` | `8.50 / 9.02` | `10.51 / 10.64` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.546` | `0.000` | **`1,582.9`** | **`395.7`** | `14,246.2` | `2,027.9 / 6,318.3 / 6,704.7` | `18.48 / 19.35 / 19.80` | `13.74 / 14.73` | `20.63 / 25.22` | Partial (E2E) |


### `gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `vllm-torchtpu`
- **Precision**: `native bf16 weights (234 GB), bf16 kv` | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `4.090` | `3.962` | **`85.9`** | **`21.5`** | `4,273.9` | `44.4 / 65.7 / 2,082.6` | `5.32 / 5.46 / 5.77` | `5.34 / 5.83` | `0.15 / 0.17` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `19.934` | `17.442` | **`418.6`** | **`104.7`** | `20,830.7` | `214.3 / 311.2 / 311.5` | `8.80 / 16.24 / 18.54` | `7.20 / 9.21` | `0.38 / 0.49` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `12` | `0.041` | `0.000` | **`162.1`** | **`40.5`** | `1,053.4` | `878.7 / 890.5 / 893.4` | `5.95 / 5.99 / 5.99` | `5.93 / 6.12` | `24.66 / 24.84` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `30` | `0.121` | `0.000` | **`485.6`** | **`121.4`** | `3,156.3` | `3,027.0 / 4,686.9 / 4,713.4` | `11.63 / 12.14 / 12.24` | `11.16 / 11.93` | `49.41 / 50.00` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `36` | `0.155` | `0.000` | **`620.6`** | **`155.1`** | `4,033.8` | `4,596.8 / 13,245.0 / 14,217.5` | `22.56 / 23.63 / 23.85` | `20.59 / 21.96` | `96.32 / 103.31` | Above SLA |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.183` | `0.000` | **`733.6`** | **`183.4`** | `4,768.6` | `6,774.0 / 25,399.0 / 27,035.7` | `39.73 / 41.20 / 41.48` | `34.49 / 37.14` | `171.33 / 180.47` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `28` | `0.166` | `0.000` | **`665.3`** | **`166.3`** | `4,324.1` | `937.5 / 1,773.4 / 2,111.9` | `30.56 / 31.66 / 31.74` | `31.17 / 32.31` | `123.93 / 127.61` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `36` | `0.184` | `0.000` | **`735.2`** | **`183.8`** | `4,778.7` | `2,982.8 / 6,387.4 / 7,857.2` | `43.13 / 45.05 / 45.22` | `40.04 / 42.45` | `176.06 / 181.68` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `36` | `0.189` | `0.000` | **`757.6`** | **`189.4`** | `4,924.4` | `6,404.9 / 12,783.1 / 13,480.1` | `43.47 / 46.20 / 46.49` | `40.37 / 42.61` | `180.23 / 186.16` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `40` | `0.194` | `0.000` | **`775.2`** | **`193.8`** | `5,038.7` | `11,318.3 / 21,035.5 / 21,741.1` | `47.36 / 50.59 / 50.84` | `43.91 / 46.28` | `200.73 / 203.85` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `24` | `0.183` | `0.183` | **`187.2`** | **`46.8`** | `374.4` | `45.7 / 46.5 / 48.6` | `5.30 / 5.33 / 5.33` | `5.27 / 5.47` | `5.47 / 5.49` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `48` | `0.983` | `0.983` | **`1,006.4`** | **`251.6`** | `2,012.8` | `213.9 / 228.3 / 228.6` | `7.70 / 7.89 / 8.00` | `7.75 / 8.10` | `8.09 / 8.30` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `96` | `2.190` | `0.000` | **`2,243.0`** | **`560.7`** | `4,485.9` | `550.7 / 861.2 / 862.0` | `13.72 / 14.01 / 14.24` | `13.46 / 14.41` | `14.63 / 14.90` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.165` | `0.165` | **`168.6`** | **`42.2`** | `1,517.4` | `236.8 / 239.6 / 239.6` | `5.71 / 5.75 / 5.75` | `5.69 / 5.98` | `6.07 / 6.11` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `40` | `0.712` | `0.000` | **`728.9`** | **`182.2`** | `6,560.5` | `1,176.8 / 1,664.8 / 1,693.5` | `9.83 / 10.72 / 10.84` | `9.33 / 10.18` | `11.21 / 11.38` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.195` | `0.000` | **`1,223.2`** | **`305.8`** | `11,008.8` | `1,913.8 / 5,902.1 / 6,290.9` | `24.66 / 25.32 / 25.73` | `19.88 / 22.95` | `26.69 / 31.13` | Partial (E2E) |


### `gptoss120b_bf16_1h_4c_torchtpu_fp8kv_tp8_ep`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `vllm-torchtpu`
- **Precision**: `native bf16 weights (234 GB), fp8 kv` — optional (non-BF16) | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `4.094` | `3.967` | **`86.0`** | **`21.5`** | `4,278.7` | `44.7 / 49.0 / 2,115.7` | `5.15 / 5.72 / 6.07` | `5.22 / 5.95` | `0.15 / 0.16` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `21.232` | `18.578` | **`445.9`** | **`111.5`** | `22,187.5` | `213.8 / 311.2 / 311.6` | `6.92 / 14.98 / 17.79` | `7.19 / 7.91` | `0.35 / 0.45` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `12` | `0.040` | `0.000` | **`161.8`** | **`40.4`** | `1,051.6` | `877.1 / 882.1 / 882.7` | `5.96 / 6.01 / 6.01` | `5.93 / 6.14` | `24.69 / 24.89` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `30` | `0.122` | `0.000` | **`489.1`** | **`122.3`** | `3,179.2` | `3,032.5 / 4,684.6 / 4,723.1` | `11.51 / 12.07 / 12.16` | `11.08 / 11.72` | `49.10 / 49.62` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `36` | `0.158` | `0.000` | **`631.6`** | **`157.9`** | `4,105.1` | `4,596.3 / 13,334.6 / 14,304.7` | `22.13 / 23.16 / 23.43` | `20.08 / 21.48` | `93.76 / 101.64` | Above SLA |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.185` | `0.000` | **`739.8`** | **`185.0`** | `4,808.8` | `6,785.8 / 25,454.4 / 27,089.6` | `39.25 / 40.73 / 41.01` | `34.08 / 36.46` | `169.47 / 178.59` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `28` | `0.169` | `0.000` | **`675.3`** | **`168.8`** | `4,389.6` | `950.7 / 1,862.1 / 2,194.3` | `29.89 / 31.03 / 31.11` | `30.44 / 31.57` | `121.29 / 125.08` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `36` | `0.186` | `0.000` | **`742.9`** | **`185.7`** | `4,828.8` | `3,007.2 / 6,401.4 / 7,866.9` | `42.61 / 44.52 / 44.69` | `39.55 / 42.01` | `174.04 / 179.57` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `36` | `0.191` | `0.000` | **`765.6`** | **`191.4`** | `4,976.6` | `6,392.3 / 12,763.8 / 13,460.5` | `42.97 / 45.71 / 46.00` | `39.82 / 42.31` | `178.24 / 184.18` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `40` | `0.196` | `0.000` | **`783.8`** | **`195.9`** | `5,094.5` | `11,314.3 / 21,013.6 / 21,718.0` | `46.80 / 50.03 / 50.28` | `43.27 / 45.72` | `198.47 / 201.61` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `24` | `0.183` | `0.183` | **`187.0`** | **`46.7`** | `373.9` | `45.1 / 46.0 / 47.3` | `5.30 / 5.36 / 5.38` | `5.26 / 5.50` | `5.47 / 5.53` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `48` | `0.985` | `0.985` | **`1,008.8`** | **`252.2`** | `2,017.6` | `214.4 / 233.2 / 233.5` | `7.73 / 7.87 / 7.96` | `7.73 / 8.13` | `8.11 / 8.21` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `96` | `2.231` | `0.000` | **`2,284.8`** | **`571.2`** | `4,569.6` | `557.1 / 865.5 / 866.1` | `13.47 / 13.76 / 13.91` | `13.24 / 13.93` | `14.30 / 14.65` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.171` | `0.171` | **`175.1`** | **`43.8`** | `1,575.5` | `238.7 / 241.7 / 241.9` | `5.48 / 5.52 / 5.53` | `5.45 / 5.65` | `5.85 / 5.88` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `40` | `0.712` | `0.000` | **`728.9`** | **`182.2`** | `6,560.4` | `1,222.1 / 1,668.6 / 1,696.2` | `9.81 / 10.73 / 10.79` | `9.29 / 10.13` | `11.24 / 11.36` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.214` | `0.000` | **`1,243.1`** | **`310.8`** | `11,187.5` | `1,916.1 / 5,908.6 / 6,295.4` | `24.33 / 24.83 / 25.29` | `19.44 / 22.52` | `26.24 / 30.80` | Partial (E2E) |


### `gptoss120b_bf16_1h_4c_sglang_bf16kv_tp8ep8`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `sglang-torchtpu`
- **Precision**: `native bf16 weights (234 GB), bf16 kv` | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `5.929` | `5.929` | **`124.5`** | **`31.1`** | `6,195.8` | `66.3 / 67.5 / 99.9` | `5.10 / 5.17 / 5.24` | `5.01 / 6.13` | `0.17 / 0.17` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `19.125` | `10.758` | **`401.6`** | **`100.4`** | `19,985.4` | `249.1 / 316.6 / 317.4` | `6.84 / 17.66 / 19.05` | `6.56 / 9.09` | `0.42 / 0.45` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `12` | `0.048` | `0.000` | **`193.7`** | **`48.4`** | `1,259.3` | `739.5 / 742.9 / 744.0` | `4.97 / 5.06 / 5.06` | `4.99 / 5.90` | `20.60 / 20.96` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `30` | `0.193` | `0.000` | **`773.6`** | **`193.4`** | `5,028.4` | `2,276.8 / 3,860.2 / 3,941.3` | `7.20 / 7.57 / 7.62` | `6.76 / 7.17` | `30.98 / 31.27` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `36` | `0.274` | `0.000` | **`1,095.8`** | **`274.0`** | `7,122.9` | `4,715.4 / 9,593.3 / 10,596.1` | `11.87 / 12.80 / 12.92` | `10.30 / 11.04` | `52.55 / 56.54` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.288` | `0.000` | **`1,150.1`** | **`287.5`** | `7,475.6` | `11,596.6 / 72,535.7 / 73,851.6` | `16.16 / 19.04 / 22.57` | `12.89 / 13.71` | `74.50 / 139.96` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `28` | `0.247` | `0.000` | **`990.0`** | **`247.5`** | `6,434.9` | `847.7 / 18,657.1 / 18,852.7` | `14.31 / 15.14 / 15.23` | `12.32 / 13.46` | `58.97 / 61.50` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `36` | `0.288` | `0.000` | **`1,150.3`** | **`287.6`** | `7,476.7` | `2,468.4 / 48,869.5 / 49,303.5` | `16.06 / 20.06 / 23.05` | `12.32 / 13.71` | `68.37 / 98.07` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `36` | `0.297` | `0.000` | **`1,186.9`** | **`296.7`** | `7,715.1` | `4,585.9 / 61,074.6 / 61,689.3` | `16.09 / 19.67 / 23.07` | `12.54 / 13.42` | `68.97 / 109.11` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `40` | `0.306` | `0.000` | **`1,223.2`** | **`305.8`** | `7,950.8` | `9,386.3 / 68,663.6 / 69,434.1` | `15.79 / 16.51 / 21.85` | `12.64 / 13.46` | `80.66 / 124.70` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `24` | `0.198` | `0.198` | **`203.1`** | **`50.8`** | `406.3` | `67.7 / 68.6 / 84.0` | `4.85 / 4.96 / 4.96` | `4.88 / 6.09` | `5.03 / 5.14` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `48` | `1.091` | `1.091` | **`1,117.7`** | **`279.4`** | `2,235.3` | `312.8 / 317.8 / 318.8` | `6.87 / 7.10 / 7.13` | `6.86 / 7.18` | `7.33 / 7.41` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `96` | `2.472` | `0.000` | **`2,530.8`** | **`632.7`** | `5,061.7` | `517.9 / 990.4 / 997.4` | `12.09 / 12.45 / 12.55` | `11.69 / 12.41` | `12.94 / 13.03` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.184` | `0.184` | **`188.5`** | **`47.1`** | `1,696.4` | `253.1 / 256.7 / 266.8` | `5.07 / 5.10 / 5.12` | `4.97 / 6.12` | `5.44 / 5.47` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `40` | `0.882` | `0.221` | **`903.7`** | **`225.9`** | `8,132.9` | `993.7 / 1,732.1 / 1,767.1` | `7.88 / 8.57 / 8.72` | `7.14 / 7.55` | `9.11 / 9.18` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.603` | `0.000` | **`1,641.9`** | **`410.5`** | `14,776.7` | `2,606.0 / 6,320.3 / 6,880.9` | `16.86 / 18.80 / 19.18` | `12.84 / 13.56` | `19.83 / 23.21` | Partial (E2E) |


### `gptoss120b_bf16_1h_4c_sglang_bf16kv_eagle3_tp8ep8`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `sglang-torchtpu`
- **Precision**: `native bf16 weights (234 GB), bf16 kv + EAGLE3 (steps=3, topk=1, draft=4)` | **Topology**: `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `5.509` | `5.509` | **`115.7`** | **`28.9`** | `5,757.0` | `66.3 / 68.5 / 77.2` | `5.84 / 7.23 / 7.72` | `7.33 / 8.38` | `0.18 / 0.21` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `32` | `13.970` | `0.437` | **`293.4`** | **`73.3`** | `14,598.9` | `117.7 / 279.4 / 313.3` | `21.18 / 30.15 / 32.40` | `14.46 / 70.14` | `0.55 / 0.74` | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `12` | `0.044` | `0.000` | **`177.8`** | **`44.5`** | `1,155.8` | `762.8 / 774.5 / 775.9` | `5.48 / 5.96 / 6.10` | `7.37 / 8.80` | `22.69 / 24.59` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `30` | `0.123` | `0.000` | **`493.7`** | **`123.4`** | `3,209.2` | `768.5 / 3,152.3 / 3,986.6` | `11.64 / 13.40 / 14.63` | `15.15 / 15.84` | `47.38 / 54.35` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `36` | `0.222` | `0.000` | **`887.3`** | **`221.8`** | `5,767.8` | `784.7 / 9,740.6 / 10,700.9` | `14.77 / 18.79 / 19.18` | `18.57 / 19.59` | `64.21 / 77.42` | Above SLA |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `48` | `0.234` | `0.000` | **`934.1`** | **`233.5`** | `6,071.4` | `14,938.9 / 80,261.6 / 83,620.6` | `19.19 / 31.16 / 38.06` | `21.71 / 22.64` | `94.85 / 166.42` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `28` | `0.201` | `0.000` | **`805.4`** | **`201.4`** | `5,235.4` | `983.2 / 30,503.0 / 30,725.3` | `17.21 / 19.65 / 19.88` | `21.15 / 22.43` | `72.56 / 86.53` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `36` | `0.238` | `0.000` | **`953.7`** | **`238.4`** | `6,199.2` | `2,547.4 / 62,060.4 / 63,307.9` | `18.36 / 26.78 / 29.77` | `21.41 / 22.50` | `87.75 / 121.60` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `36` | `0.239` | `0.000` | **`955.4`** | **`238.9`** | `6,210.1` | `4,622.1 / 73,373.0 / 73,425.5` | `18.42 / 28.82 / 32.79` | `21.16 / 22.44` | `91.59 / 137.10` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `40` | `0.257` | `0.000` | **`1,029.5`** | **`257.4`** | `6,691.5` | `9,074.1 / 79,939.4 / 80,877.8` | `17.68 / 34.19 / 35.61` | `21.22 / 22.19` | `94.64 / 145.32` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | `24` | `0.184` | `0.184` | **`188.0`** | **`47.0`** | `376.0` | `66.8 / 69.9 / 86.4` | `5.32 / 6.41 / 6.83` | `7.54 / 8.95` | `5.51 / 6.62` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `48` | `0.670` | `0.154` | **`686.2`** | **`171.6`** | `1,372.5` | `80.4 / 292.6 / 292.8` | `10.80 / 13.31 / 13.82` | `15.23 / 15.89` | `11.18 / 13.70` | Partial (TTFT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `96` | `2.011` | `0.021` | **`2,059.0`** | **`514.7`** | `4,118.0` | `91.6 / 915.4 / 916.1` | `14.23 / 17.83 / 19.09` | `20.30 / 21.60` | `14.90 / 18.61` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | `20` | `0.164` | `0.164` | **`167.6`** | **`41.9`** | `1,508.0` | `250.0 / 260.9 / 266.9` | `5.75 / 6.23 / 6.24` | `7.45 / 8.88` | `6.14 / 6.64` | PASS (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `40` | `0.603` | `0.045` | **`617.2`** | **`154.3`** | `5,554.4` | `269.5 / 1,374.3 / 1,707.8` | `12.51 / 14.27 / 14.86` | `15.39 / 15.98` | `13.14 / 14.97` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `64` | `1.302` | `0.000` | **`1,333.7`** | **`333.4`** | `12,003.6` | `670.9 / 6,470.8 / 7,016.2` | `20.91 / 26.03 / 27.28` | `22.05 / 23.15` | `22.73 / 30.45` | Partial (E2E) |


### `gptoss120b_bf16_2h_8c_sglang_bf16kv_tp8ep8`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `sglang-torchtpu (2x single-host replicas + least_conn LB)`
- **Precision**: `native bf16 weights (234 GB), bf16 kv` | **Topology**: `2 hosts (8 chips / 16 TensorCores), 2x independent TP=8 EP=8 replicas` (`8` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `5.160` | `4.999` | **`108.4`** | **`13.5`** | `5,392.0` | `65.7 / 210.3 / 542.0` | `5.08 / 5.56 / 5.57` | `5.21 / 6.20` | `0.17 / 0.30` | PASS (All) |
| `smoke` | `1,024 : 21` | `CON=8` | `64` | `28.220` | `21.606` | **`592.6`** | **`74.1`** | `29,490.3` | `118.7 / 170.5 / 171.9` | `8.11 / 10.77 / 10.90` | `5.70 / 8.12` | `0.28 / 0.29` | Partial (TTFT+E2E) |
| `smoke` | `1,024 : 21` | `CON=32` | `128` | `46.768` | `5.481` | **`982.1`** | **`122.8`** | `48,872.6` | `329.1 / 525.0 / 526.8` | `18.03 / 28.46 / 30.65` | `8.20 / 14.05` | `0.68 / 0.70` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `10` | `0.048` | `0.000` | **`192.3`** | **`24.0`** | `1,249.7` | `771.4 / 790.8 / 798.3` | `5.05 / 5.16 / 5.17` | `5.08 / 6.02` | `20.93 / 21.43` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `48` | `0.229` | `0.000` | **`916.2`** | **`114.5`** | `5,955.6` | `1,348.6 / 1,962.3 / 2,063.4` | `6.10 / 6.37 / 6.40` | `5.88 / 6.42` | `25.81 / 26.36` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `64` | `0.451` | `0.000` | **`1,802.4`** | **`225.3`** | `11,715.6` | `2,930.9 / 5,231.5 / 5,379.6` | `8.11 / 8.66 / 8.72` | `7.47 / 7.99` | `35.41 / 35.81` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `64` | `0.608` | `0.000` | **`2,433.8`** | **`304.2`** | `15,819.8` | `5,114.0 / 10,033.8 / 10,887.0` | `11.79 / 12.81 / 12.95` | `10.33 / 11.16` | `52.43 / 56.46` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=64` | `96` | `0.559` | `0.000` | **`2,234.1`** | **`279.3`** | `14,521.5` | `11,626.0 / 72,385.8 / 74,250.7` | `16.14 / 19.21 / 24.10` | `12.83 / 13.73` | `75.19 / 139.25` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `36` | `0.357` | `0.000` | **`1,428.4`** | **`178.5`** | `9,284.3` | `750.3 / 868.6 / 987.0` | `10.09 / 11.29 / 11.41` | `9.49 / 10.34` | `41.23 / 45.92` | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `48` | `0.485` | `0.000` | **`1,939.0`** | **`242.4`** | `12,603.3` | `752.1 / 8,121.0 / 14,362.1` | `13.83 / 15.07 / 15.28` | `12.47 / 13.39` | `56.14 / 61.13` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `56` | `0.538` | `0.000` | **`2,153.9`** | **`269.2`** | `14,000.6` | `1,488.8 / 46,458.6 / 47,876.6` | `16.04 / 18.30 / 19.69` | `12.81 / 13.77` | `66.96 / 81.86` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `60` | `0.543` | `0.000` | **`2,172.7`** | **`271.6`** | `14,122.4` | `4,613.9 / 57,736.0 / 58,086.2` | `15.83 / 17.29 / 19.65` | `12.59 / 13.46` | `67.41 / 97.96` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | `60` | `0.548` | `0.000` | **`2,193.4`** | **`274.2`** | `14,256.9` | `6,640.9 / 61,819.9 / 62,506.5` | `15.85 / 17.30 / 19.67` | `12.51 / 13.55` | `68.88 / 101.89` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `64` | `1.303` | `1.303` | **`1,333.9`** | **`166.7`** | `2,667.7` | `156.9 / 174.6 / 186.9` | `5.77 / 5.93 / 6.01` | `5.74 / 6.36` | `6.02 / 6.24` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `128` | `3.330` | `3.017` | **`3,409.6`** | **`426.2`** | `6,819.1` | `357.1 / 513.4 / 515.6` | `9.05 / 9.31 / 9.47` | `8.91 / 9.41` | `9.56 / 9.85` | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | `192` | `4.965` | `0.000` | **`5,084.5`** | **`635.6`** | `10,169.0` | `533.0 / 984.6 / 1,033.6` | `11.99 / 12.37 / 12.49` | `11.57 / 12.28` | `12.87 / 13.03` | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | `256` | `4.949` | `0.000` | **`5,067.3`** | **`633.4`** | `10,134.7` | `13,265.1 / 13,813.1 / 13,957.0` | `12.08 / 12.52 / 24.43` | `11.65 / 12.39` | `25.67 / 25.98` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `48` | `1.165` | `0.582` | **`1,192.8`** | **`149.1`** | `10,735.3` | `570.1 / 877.9 / 917.6` | `6.13 / 6.47 / 6.51` | `5.77 / 6.30` | `6.80 / 6.93` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `96` | `2.417` | `0.000` | **`2,475.2`** | **`309.4`** | `22,276.5` | `1,521.0 / 3,080.4 / 3,514.9` | `11.33 / 12.48 / 12.66` | `9.48 / 10.01` | `13.19 / 14.00` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | `128` | `3.155` | `0.000` | **`3,230.7`** | **`403.8`** | `29,076.1` | `2,551.3 / 6,397.8 / 7,003.6` | `17.23 / 18.92 / 19.29` | `12.91 / 13.78` | `19.97 / 23.90` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | `256` | `3.204` | `0.000` | **`3,281.0`** | **`410.1`** | `29,528.8` | `20,534.9 / 25,771.4 / 26,889.7` | `18.56 / 19.00 / 19.21` | `12.85 / 13.57` | `39.58 / 45.05` | Partial (E2E) |


### `gptoss120b_bf16_2h_8c_vllmtpu_bf16kv_tp8ep`
- **Model**: `lmsys/gpt-oss-120b-bf16` | **Framework**: `vllm-tpu (2x single-host replicas + least_conn LB)`
- **Precision**: `native bf16 weights (234 GB), bf16 kv` | **Topology**: `2 hosts (8 chips / 16 TensorCores), 2x independent TP=8 EP=true replicas` (`8` physical `tpu7x` chips)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip (derived) | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | SLA Status (P95) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke` | `1,024 : 21` | `CON=1` | `32` | `2.872` | `2.692` | **`60.3`** | **`7.5`** | `3,001.1` | `46.4 / 1,355.1 / 3,003.1` | `5.49 / 6.25 / 12.82` | `5.41 / 6.12` | `0.16 / 1.54` | Partial (TPOT+E2E) |
| `smoke` | `1,024 : 21` | `CON=8` | `64` | `27.405` | `20.554` | **`575.5`** | **`71.9`** | `28,638.4` | `147.4 / 275.5 / 276.7` | `6.14 / 10.98 / 16.93` | `6.20 / 6.95` | `0.27 / 0.44` | Partial (TTFT+E2E) |
| `smoke` | `1,024 : 21` | `CON=32` | `128` | `20.956` | `0.982` | **`440.1`** | **`55.0`** | `21,898.9` | `450.7 / 3,039.0 / 3,560.0` | `14.58 / 53.56 / 152.76` | `8.64 / 100.43` | `0.74 / 3.96` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | `10` | `0.040` | `0.000` | **`160.7`** | **`20.1`** | `1,044.4` | `882.6 / 889.0 / 890.2` | `5.99 / 6.05 / 6.05` | `5.96 / 6.15` | `24.85 / 25.06` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | `48` | `0.169` | `0.000` | **`677.7`** | **`84.7`** | `4,405.3` | `1,883.7 / 2,444.2 / 2,457.5` | `8.38 / 8.66 / 8.72` | `8.23 / 8.57` | `35.40 / 35.75` | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | `64` | `0.272` | `0.000` | **`1,087.8`** | **`136.0`** | `7,070.8` | `3,869.1 / 5,889.9 / 6,230.6` | `13.71 / 14.49 / 14.56` | `13.14 / 13.91` | `58.56 / 59.39` | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | `64` | `0.338` | `0.000` | **`1,351.0`** | **`168.9`** | `8,781.8` | `4,021.7 / 11,656.0 / 12,539.7` | `22.61 / 23.31 / 23.44` | `20.59 / 21.93` | `94.65 / 101.95` | Above SLA |
| `long_cl` | `22,000 : 4,000` | `CON=64` | `96` | `0.365` | `0.000` | **`1,460.5`** | **`182.6`** | `9,493.0` | `6,831.1 / 25,620.2 / 27,311.1` | `39.87 / 41.33 / 41.71` | `34.61 / 37.37` | `171.87 / 181.00` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | `36` | `0.266` | `0.000` | **`1,062.8`** | **`132.8`** | `6,908.2` | `902.6 / 1,009.2 / 1,192.2` | `19.50 / 21.13 / 21.34` | `18.48 / 22.65` | `78.90 / 85.40` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | `48` | `0.325` | `0.000` | **`1,300.1`** | **`162.5`** | `8,450.6` | `916.1 / 1,310.7 / 1,417.8` | `27.69 / 29.24 / 29.43` | `27.78 / 29.44` | `111.63 / 118.14` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | `56` | `0.363` | `0.000` | **`1,453.0`** | **`181.6`** | `9,444.5` | `1,729.4 / 3,408.6 / 3,673.2` | `34.31 / 36.09 / 36.27` | `32.04 / 33.86` | `140.12 / 145.33` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | `60` | `0.378` | `0.000` | **`1,511.4`** | **`188.9`** | `9,824.4` | `6,224.4 / 9,157.1 / 9,615.1` | `36.31 / 38.68 / 38.92` | `33.62 / 35.61` | `151.68 / 156.16` | Above SLA |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | `60` | `0.377` | `0.000` | **`1,507.1`** | **`188.4`** | `9,796.1` | `8,131.9 / 13,399.7 / 14,055.6` | `36.53 / 38.96 / 39.26` | `33.86 / 35.71` | `154.13 / 157.50` | Above SLA |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | `64` | `1.151` | `1.151` | **`1,178.5`** | **`147.3`** | `2,357.0` | `148.5 / 160.9 / 163.2` | `6.63 / 6.75 / 6.79` | `6.57 / 6.87` | `6.93 / 7.02` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | `128` | `3.113` | `2.967` | **`3,188.0`** | **`398.5`** | `6,376.1` | `378.4 / 466.4 / 565.6` | `9.65 / 9.87 / 10.00` | `9.56 / 10.12` | `10.27 / 10.32` | PASS (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | `192` | `4.333` | `0.000` | **`4,437.3`** | **`554.7`** | `8,874.7` | `653.6 / 937.8 / 949.8` | `13.85 / 14.28 / 14.35` | `13.53 / 14.31` | `14.71 / 15.12` | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | `256` | `5.437` | `0.000` | **`5,567.6`** | **`696.0`** | `11,135.2` | `958.5 / 4,011.6 / 4,027.7` | `20.91 / 23.78 / 23.88` | `20.21 / 21.67` | `23.38 / 25.31` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | `48` | `0.942` | `0.235` | **`964.6`** | **`120.6`** | `8,681.1` | `769.2 / 940.6 / 943.0` | `7.54 / 8.09 / 8.15` | `7.35 / 7.82` | `8.46 / 8.59` | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | `96` | `1.942` | `0.000` | **`1,988.8`** | **`248.6`** | `17,899.4` | `1,796.7 / 2,897.3 / 3,215.0` | `14.36 / 15.62 / 15.77` | `12.82 / 14.36` | `16.43 / 17.58` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | `128` | `2.393` | `0.000` | **`2,450.3`** | **`306.3`** | `22,052.6` | `1,914.9 / 5,960.2 / 6,308.9` | `24.41 / 25.26 / 25.70` | `19.80 / 22.98` | `26.65 / 30.93` | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | `256` | `2.748` | `0.000` | **`2,814.4`** | **`351.8`** | `25,329.7` | `2,002.3 / 11,546.0 / 12,653.7` | `43.73 / 44.08 / 44.67` | `33.00 / 39.29` | `46.20 / 56.58` | Partial (E2E) |

