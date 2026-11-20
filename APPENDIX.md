# Appendix: Stack Compatibility Deep Dive, Load-Time Patches, Ablations, and Raw Telemetry (GPT-OSS 120B on TPU v7x)

This appendix preserves the complete architectural analysis across `sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu` (JAX), load-time patch root causes (including the unpatched `vllm-torchtpu` correctness failure), `EAGLE3` speculative decoding ablations, multi-host scaling breakdowns, `22k/4k` latency-target verdicts, and all 13 complete per-suite benchmark tables. Section 4 covers the follow-up single-host MXFP4 runs at higher load (README section C). For headline numbers and reproduction commands, see [`README.md`](./README.md).

---

## 1. Serving Stack & Checkpoint Compatibility Analysis on TPU v7x

| Serving Stack | `openai/gpt-oss-120b` (`65.3 GB` Native MXFP4) | `lmsys/gpt-oss-120b-bf16` (`233.7 GB` Unquantized BF16) | Key Behavior on TPU v7x |
| :--- | :--- | :--- | :--- |
| `sglang-torchtpu` *(PyTorch + Pallas)* | Supported out of the box (`TP=8, EP=8`; VPU unpacks 4-bit MXFP4 → BF16/FP8 MXU; `737.0 ms` TTFT, `4.09 ms` TPOT at `22k/4k`, FP8 KV) | Supported via [`apply_sglang_bf16_patches.py`](./scripts/patches/apply_sglang_bf16_patches.py) (`739.5 ms` TTFT, `4.97 ms` TPOT on 1 host; `5,084.5` tok/s on 2 hosts, BF16 KV) | Lowest single-stream TTFT and TPOT on both BF16 and MXFP4 |
| `vllm-torchtpu` *(PyTorch / XLA + Pallas)* | Not supported upstream (`mxfp4.py` binds GPU/ROCm Triton kernels only; raises `ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu`) | Supported via [`apply_vllm_torchtpu_bf16_patches.py`](./scripts/patches/apply_vllm_torchtpu_bf16_patches.py) (`878.7 ms` TTFT, `5.95 ms` TPOT, `2,243.0` tok/s on `1k/1k`, BF16 KV) | Requires 128-tile `hidden_size` padding (`2880 → 2944`) for valid outputs |
| `vllm-tpu` *(JAX `tpu-inference`)* | Supported out of the box (`TP=2, DP=4` = 4 DP replicas/host; `5,847.0` tok/s on 1h → `20,366.9` tok/s on 4h, FP8 KV) | Supported out of the box with zero patches (`TP=8, EP=true`; `880.5 ms` TTFT, `8.54 ms` TPOT on 1h; `5,567.6` tok/s on 2h, BF16 KV) | Highest multi-host throughput on both MXFP4 (`4` hosts) and BF16 (`2` hosts) |

### A. Why `openai/gpt-oss-120b` (MXFP4) Works on `sglang-torchtpu` and `vllm-tpu`, But Fails on `vllm-torchtpu`
- `openai/gpt-oss-120b` stores its `128` routed MoE experts per layer in `gpt_oss_mxfp4` (`32`-element 4-bit `E2M1` blocks sharing an 8-bit `E8M0` power-of-two exponent scale).
- TPU `v7x` MXUs execute natively in BF16 (`2,307 TFLOP/s/chip`) and FP8 (`4,614 TFLOP/s/chip`). Both `sglang-torchtpu` and `vllm-tpu` (JAX) include Pallas kernels that keep expert weights compressed in HBM3e (`65.3 GB`, `3.58×` smaller than BF16) and unpack 4-bit nibbles + `E8M0` scales in the Vector Processing Unit (VPU) immediately before feeding the MXU.
- Upstream `vllm-torchtpu` registers only GPU/ROCm Triton kernels in `vllm/model_executor/layers/quantization/mxfp4.py`, failing validation at startup with `ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu`.

### B. Correctness Fix on `vllm-torchtpu`: Non-128-Aligned `hidden_size = 2880 → 2944`
- TPU MXUs and Pallas grouped-matmul (`gmm_v2`) MoE kernels operate on `128 × 128` tiles. While most LLMs use a `hidden_size` divisible by `128`, GPT-OSS 120B uses `hidden_size = 2880` (`22.5 × 128`, leaving a `64`-column remainder).
- Upstream `VllmUnquantizedFusedMoEMethod.process_weights_after_loading` padded `intermediate_size` to a multiple of `128` (`2880 → 2944` per gate/up half, `5888` combined), **but left `hidden_size = 2880` unpadded** (`w13=[16, 2880, 5888]`, `w2=[16, 2944, 2880]`).
- **Unpatched Failure Mode ([`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_unpatched/`](./results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_unpatched/)):** Without padding `H`, Pallas `gmm_v2` reads 64 uninitialized HBM lanes past each row, corrupting MoE activations and collapsing router logits onto a single expert so every sanity prompt emits only `!` tokens (also running `3.1×` slower at `18.83 ms` TPOT vs. `5.95 ms`; see [`server_log_excerpt.txt`](./results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_unpatched/server_log_excerpt.txt)).
- **Patched Fix ([`scripts/patches/apply_vllm_torchtpu_bf16_patches.py`](./scripts/patches/apply_vllm_torchtpu_bf16_patches.py)):** Zero-pads `w13_weight`, `w2_weight`, and `w2_bias` along `H` (`2880 → 2944`), producing `w13=[16, 2944, 5888]`, `w2=[16, 2944, 2944]` and coherent generations ([`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep/`](./results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep/)). All reported `vllm-torchtpu` performance numbers use the patched run.

### C. BF16 Loader & `swigluoai` Deinterleaving on `sglang-torchtpu`
- Upstream `sglang/srt/models/gpt_oss.py` only mapped MXFP4 block/scale tensor names, and OpenAI's `swigluoai` activation interleaves `gate` (`::2`) and `up` (`1::2`) channels whereas SGLang's TPU `FusedMoE` kernel expects contiguous `[gate, up]`.
- [`scripts/patches/apply_sglang_bf16_patches.py`](./scripts/patches/apply_sglang_bf16_patches.py) maps standard Hugging Face BF16 expert tensors, slices the `128` experts across `EP=8`, deinterleaves `swigluoai` channels once at load time, and pads `2880 → 2944`.

### D. `EAGLE3` Speculative Decoding Ablation (`steps=3, topk=1, draft=4`)
- Evaluated on 1-host `sglang-torchtpu` (`TP=8, EP=8`) with [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) on both BF16 weights + BF16 KV and MXFP4 weights + FP8 KV:
  - Because non-speculative `sglang-torchtpu` already achieves `4.09 ms` (MXFP4) and `4.97 ms` (BF16) single-stream TPOT (`5.1B` active parameters across `29.6 TB/s` host HBM3e bandwidth), 3-step `EAGLE3` drafting overhead outweighed verification savings: `CON=1` `22k/4k` TPOT increased from `4.97 ms → 5.48 ms` on BF16 (`4.09 ms → 5.05 ms` on MXFP4), and `1k/1k` `CON=32` output throughput dropped by `18.6%` on BF16 (`2,530.8 → 2,059.0` tok/s) and `17.6%` on MXFP4 (`2,587.5 → 2,132.5` tok/s).

---

## 2. Multi-Host Scaling & `22,000 / 4,000` Goodput Breakdown

### A. 2-Host (`8` Chips) vs. 1-Host (`4` Chips) BF16 Scaling at Equal Per-Replica Load

| Stack (`BF16` Weights + `BF16` KV) | Workload (`ISL : OSL`) | `1` Host (`CON=32`) | `2` Hosts (`CON=64`, `32` req/replica) | `2`-Host / `1`-Host Ratio |
| :--- | :--- | ---: | ---: | ---: |
| `sglang-torchtpu` (`TP=8, EP=8` / host) | `1,024 : 1,024` Out TPS | 2,530.8 (`n=96`) | **5,084.5** (`n=192`) | **2.01×** |
| `sglang-torchtpu` (`TP=8, EP=8` / host) | `8,192 : 1,024` Out TPS | 1,641.9 (`n=64`) | **3,230.7** (`n=128`; `3,281.0` at `CON=128`) | **1.97×** |
| `sglang-torchtpu` (`TP=8, EP=8` / host) | `22,000 : 4,000` CL Out TPS | 1,150.1 (`n=48`) | **2,234.1** (`n=96`; `2,433.8` at `CON=32`) | **1.94×** |
| `vllm-tpu` (JAX `TP=8, EP` / host) | `1,024 : 1,024` Out TPS | 2,463.7 (`n=96`) | 4,437.3 (`n=192`; **5,567.6** at `CON=128`) | 1.80× (`2.26×` peak) |
| `vllm-tpu` (JAX `TP=8, EP` / host) | `8,192 : 1,024` Out TPS | 1,557.4 (`n=64`) | 2,450.3 (`n=128`; 2,814.4 at `CON=128`) | 1.57× (`1.81×` peak) |
| `vllm-tpu` (JAX `TP=8, EP` / host) | `22,000 : 4,000` CL Out TPS | 1,249.9 (`n=48`) | 1,460.5 (`n=96`) | 1.17× |

### B. 1, 2 & 4 Host (`4`, `8`, `16` Chips) MXFP4 + FP8 KV Scaling (`vllm-tpu` JAX `TP=2, DP=4`)

| Workload (`ISL : OSL`) | Metric | `1` Host (`4` Chips) | `2` Hosts (`8` Chips) | `4` Hosts (`16` Chips) | `4`-Host / `1`-Host Ratio |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `1,024 : 21` (`smoke`) | Request TPS / Goodput (`req/s`) | 20.88 / 14.36 | 58.19 / 31.73 | **74.94 / 48.01** | **3.59× / 3.34×** |
| `1,024 : 1,024` | Peak Out TPS / Max Goodput (`req/s`) | 5,847.0 / 3.340 | 14,277.1 / 5.054 | **20,366.9 / 8.584** | **3.48× / 2.57×** |
| `8,192 : 1,024` | Peak Out / Total TPS (`tok/s`) | 2,696.6 / 24,269.5 | 6,533.7 / 58,803.6 | **9,820.8 / 88,387.3** | **3.64×** |
| `22,000 : 4,000` (Closed-Loop Peak) | Output TPS (`tok/s`) | 1,789.4 (`CON=32`) | 3,315.7 (`CON=64`) | **8,749.7** (`CON=128`) | **4.89×** |
| `22,000 : 4,000` (Open-Loop `6.0 RPS`) | Output TPS / TPOT P50 / E2E P50 | 2,647.0 / 14.76 ms / 71.17 s | 4,127.5 / 11.27 ms / 50.51 s | **5,564.6 / 8.82 ms / 37.90 s** | **2.10×** TPS; `-40.2%` TPOT |

### C. `22,000 / 4,000` Simultaneous Latency Target Verdicts (`TTFT <= 500 ms`, `TPOT <= 10 ms`, `E2E <= 60 s`)
- Across all 13 valid configurations (both BF16 and MXFP4, 1 to 4 hosts), **`22k / 4k` simultaneous goodput was `0.000 req/s`** because `22,000`-token prefill TTFT exceeded `500 ms` even at `CON=1` (best `CON=1` TTFT P50/P95: `737.0 / 742.1 ms` on MXFP4 `sglang-torchtpu`, `739.5 / 742.9 ms` on BF16 `sglang-torchtpu`, `878.7 / 890.5 ms` on BF16 `vllm-torchtpu`, `880.5 / 927.9 ms` on BF16 `vllm-tpu`, and `1,279.3 / 1,283.1 ms` on MXFP4 `vllm-tpu` `TP=2, DP=4`).
- Conversely, **every configuration met both `TPOT P95 <= 10 ms` (`4.14–8.74 ms`) and `E2E P95 <= 60 s` (`17.29–35.82 s`) at `CON=1`**, and 4-host MXFP4 `vllm-tpu` sustained `TPOT P95 = 9.96 ms <= 10 ms` and `E2E P95 = 41.11 s <= 60 s` at `6.0 RPS` open-loop (`5,564.6` out tok/s).
- Additional configs in [`configs/`](./configs/) (`tp4dp2`, `torchtpu` MXFP4/FP8-KV `tp2dp4`, `vllmtpu` MXFP4 `tp8`, 2h/4h `sglang` MXFP4) are included as reference benchmark specs without corresponding `results/` runs.

---

## 3. Complete Per-Suite Benchmark Tables (All 13 Valid Runs)

*Note:* `Out tok/s/chip` is derived (`Out tok/s` ÷ physical chips); all other columns are direct `vllm bench serve` fields from `results/*/summary.json`.

### 3.1 Unquantized BF16 Weights (`lmsys/gpt-oss-120b-bf16`, `233.7 GB`)

#### `gptoss120b_bf16_1h_4c_sglang_bf16kv_tp8ep8`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `sglang-torchtpu`
- **Precision:** `native bf16 weights (234 GB), bf16 kv` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_1h_4c_sglang_bf16kv_tp8ep8/summary.json`](./results/gptoss120b_bf16_1h_4c_sglang_bf16kv_tp8ep8/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 5.929 | 5.929 | 124.5 | 31.1 | 6,195.8 | 66.3 / 67.5 / 99.9 | 5.10 / 5.17 / 5.24 | 5.01 / 6.13 | 0.17 / 0.17 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 19.125 | 10.758 | 401.6 | 100.4 | 19,985.4 | 249.1 / 316.6 / 317.4 | 6.84 / 17.66 / 19.05 | 6.56 / 9.09 | 0.42 / 0.45 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 12 | 0.048 | 0.000 | 193.7 | 48.4 | 1,259.3 | 739.5 / 742.9 / 744.0 | 4.97 / 5.06 / 5.06 | 4.99 / 5.90 | 20.60 / 20.96 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 30 | 0.193 | 0.000 | 773.6 | 193.4 | 5,028.4 | 2,276.8 / 3,860.2 / 3,941.3 | 7.20 / 7.57 / 7.62 | 6.76 / 7.17 | 30.98 / 31.27 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 36 | 0.274 | 0.000 | 1,095.8 | 274.0 | 7,122.9 | 4,715.4 / 9,593.3 / 10,596.1 | 11.87 / 12.80 / 12.92 | 10.30 / 11.04 | 52.55 / 56.54 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.288 | 0.000 | 1,150.1 | 287.5 | 7,475.6 | 11,596.6 / 72,535.7 / 73,851.6 | 16.16 / 19.04 / 22.57 | 12.89 / 13.71 | 74.50 / 139.96 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 28 | 0.247 | 0.000 | 990.0 | 247.5 | 6,434.9 | 847.7 / 18,657.1 / 18,852.7 | 14.31 / 15.14 / 15.23 | 12.32 / 13.46 | 58.97 / 61.50 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 36 | 0.288 | 0.000 | 1,150.3 | 287.6 | 7,476.7 | 2,468.4 / 48,869.5 / 49,303.5 | 16.06 / 20.06 / 23.05 | 12.32 / 13.71 | 68.37 / 98.07 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 36 | 0.297 | 0.000 | 1,186.9 | 296.7 | 7,715.1 | 4,585.9 / 61,074.6 / 61,689.3 | 16.09 / 19.67 / 23.07 | 12.54 / 13.42 | 68.97 / 109.11 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 40 | 0.306 | 0.000 | 1,223.2 | 305.8 | 7,950.8 | 9,386.3 / 68,663.6 / 69,434.1 | 15.79 / 16.51 / 21.85 | 12.64 / 13.46 | 80.66 / 124.70 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 24 | 0.198 | 0.198 | 203.1 | 50.8 | 406.3 | 67.7 / 68.6 / 84.0 | 4.85 / 4.96 / 4.96 | 4.88 / 6.09 | 5.03 / 5.14 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 48 | 1.091 | 1.091 | 1,117.7 | 279.4 | 2,235.3 | 312.8 / 317.8 / 318.8 | 6.87 / 7.10 / 7.13 | 6.86 / 7.18 | 7.33 / 7.41 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 96 | 2.472 | 0.000 | **2,530.8** | 632.7 | 5,061.7 | 517.9 / 990.4 / 997.4 | 12.09 / 12.45 / 12.55 | 11.69 / 12.41 | 12.94 / 13.03 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.184 | 0.184 | 188.5 | 47.1 | 1,696.4 | 253.1 / 256.7 / 266.8 | 5.07 / 5.10 / 5.12 | 4.97 / 6.12 | 5.44 / 5.47 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 40 | 0.882 | 0.221 | 903.7 | 225.9 | 8,132.9 | 993.7 / 1,732.1 / 1,767.1 | 7.88 / 8.57 / 8.72 | 7.14 / 7.55 | 9.11 / 9.18 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.603 | 0.000 | 1,641.9 | 410.5 | 14,776.7 | 2,606.0 / 6,320.3 / 6,880.9 | 16.86 / 18.80 / 19.18 | 12.84 / 13.56 | 19.83 / 23.21 | Partial (E2E) |

#### `gptoss120b_bf16_1h_4c_sglang_bf16kv_eagle3_tp8ep8`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `sglang-torchtpu`
- **Precision:** `native bf16 weights (234 GB), bf16 kv + EAGLE3 (steps=3, topk=1, draft=4)` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_1h_4c_sglang_bf16kv_eagle3_tp8ep8/summary.json`](./results/gptoss120b_bf16_1h_4c_sglang_bf16kv_eagle3_tp8ep8/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 5.509 | 5.509 | 115.7 | 28.9 | 5,757.0 | 66.3 / 68.5 / 77.2 | 5.84 / 7.23 / 7.72 | 7.33 / 8.38 | 0.18 / 0.21 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 13.970 | 0.437 | 293.4 | 73.3 | 14,598.9 | 117.7 / 279.4 / 313.3 | 21.18 / 30.15 / 32.40 | 14.46 / 70.14 | 0.55 / 0.74 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 12 | 0.044 | 0.000 | 177.8 | 44.5 | 1,155.8 | 762.8 / 774.5 / 775.9 | 5.48 / 5.96 / 6.10 | 7.37 / 8.80 | 22.69 / 24.59 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 30 | 0.123 | 0.000 | 493.7 | 123.4 | 3,209.2 | 768.5 / 3,152.3 / 3,986.6 | 11.64 / 13.40 / 14.63 | 15.15 / 15.84 | 47.38 / 54.35 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 36 | 0.222 | 0.000 | 887.3 | 221.8 | 5,767.8 | 784.7 / 9,740.6 / 10,700.9 | 14.77 / 18.79 / 19.18 | 18.57 / 19.59 | 64.21 / 77.42 | Above target |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.234 | 0.000 | 934.1 | 233.5 | 6,071.4 | 14,938.9 / 80,261.6 / 83,620.6 | 19.19 / 31.16 / 38.06 | 21.71 / 22.64 | 94.85 / 166.42 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 28 | 0.201 | 0.000 | 805.4 | 201.4 | 5,235.4 | 983.2 / 30,503.0 / 30,725.3 | 17.21 / 19.65 / 19.88 | 21.15 / 22.43 | 72.56 / 86.53 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 36 | 0.238 | 0.000 | 953.7 | 238.4 | 6,199.2 | 2,547.4 / 62,060.4 / 63,307.9 | 18.36 / 26.78 / 29.77 | 21.41 / 22.50 | 87.75 / 121.60 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 36 | 0.239 | 0.000 | 955.4 | 238.9 | 6,210.1 | 4,622.1 / 73,373.0 / 73,425.5 | 18.42 / 28.82 / 32.79 | 21.16 / 22.44 | 91.59 / 137.10 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 40 | 0.257 | 0.000 | 1,029.5 | 257.4 | 6,691.5 | 9,074.1 / 79,939.4 / 80,877.8 | 17.68 / 34.19 / 35.61 | 21.22 / 22.19 | 94.64 / 145.32 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 24 | 0.184 | 0.184 | 188.0 | 47.0 | 376.0 | 66.8 / 69.9 / 86.4 | 5.32 / 6.41 / 6.83 | 7.54 / 8.95 | 5.51 / 6.62 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 48 | 0.670 | 0.154 | 686.2 | 171.6 | 1,372.5 | 80.4 / 292.6 / 292.8 | 10.80 / 13.31 / 13.82 | 15.23 / 15.89 | 11.18 / 13.70 | Partial (TTFT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 96 | 2.011 | 0.021 | **2,059.0** | 514.7 | 4,118.0 | 91.6 / 915.4 / 916.1 | 14.23 / 17.83 / 19.09 | 20.30 / 21.60 | 14.90 / 18.61 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.164 | 0.164 | 167.6 | 41.9 | 1,508.0 | 250.0 / 260.9 / 266.9 | 5.75 / 6.23 / 6.24 | 7.45 / 8.88 | 6.14 / 6.64 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 40 | 0.603 | 0.045 | 617.2 | 154.3 | 5,554.4 | 269.5 / 1,374.3 / 1,707.8 | 12.51 / 14.27 / 14.86 | 15.39 / 15.98 | 13.14 / 14.97 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.302 | 0.000 | 1,333.7 | 333.4 | 12,003.6 | 670.9 / 6,470.8 / 7,016.2 | 20.91 / 26.03 / 27.28 | 22.05 / 23.15 | 22.73 / 30.45 | Partial (E2E) |

#### `gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `vllm-torchtpu`
- **Precision:** `native bf16 weights (234 GB), bf16 kv` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep/summary.json`](./results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 4.090 | 3.962 | 85.9 | 21.5 | 4,273.9 | 44.4 / 65.7 / 2,082.6 | 5.32 / 5.46 / 5.77 | 5.34 / 5.83 | 0.15 / 0.17 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 19.934 | 17.442 | 418.6 | 104.7 | 20,830.7 | 214.3 / 311.2 / 311.5 | 8.80 / 16.24 / 18.54 | 7.20 / 9.21 | 0.38 / 0.49 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 12 | 0.041 | 0.000 | 162.1 | 40.5 | 1,053.4 | 878.7 / 890.5 / 893.4 | 5.95 / 5.99 / 5.99 | 5.93 / 6.12 | 24.66 / 24.84 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 30 | 0.121 | 0.000 | 485.6 | 121.4 | 3,156.3 | 3,027.0 / 4,686.9 / 4,713.4 | 11.63 / 12.14 / 12.24 | 11.16 / 11.93 | 49.41 / 50.00 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 36 | 0.155 | 0.000 | 620.6 | 155.1 | 4,033.8 | 4,596.8 / 13,245.0 / 14,217.5 | 22.56 / 23.63 / 23.85 | 20.59 / 21.96 | 96.32 / 103.31 | Above target |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.183 | 0.000 | 733.6 | 183.4 | 4,768.6 | 6,774.0 / 25,399.0 / 27,035.7 | 39.73 / 41.20 / 41.48 | 34.49 / 37.14 | 171.33 / 180.47 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 28 | 0.166 | 0.000 | 665.3 | 166.3 | 4,324.1 | 937.5 / 1,773.4 / 2,111.9 | 30.56 / 31.66 / 31.74 | 31.17 / 32.31 | 123.93 / 127.61 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 36 | 0.184 | 0.000 | 735.2 | 183.8 | 4,778.7 | 2,982.8 / 6,387.4 / 7,857.2 | 43.13 / 45.05 / 45.22 | 40.04 / 42.45 | 176.06 / 181.68 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 36 | 0.189 | 0.000 | 757.6 | 189.4 | 4,924.4 | 6,404.9 / 12,783.1 / 13,480.1 | 43.47 / 46.20 / 46.49 | 40.37 / 42.61 | 180.23 / 186.16 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 40 | 0.194 | 0.000 | 775.2 | 193.8 | 5,038.7 | 11,318.3 / 21,035.5 / 21,741.1 | 47.36 / 50.59 / 50.84 | 43.91 / 46.28 | 200.73 / 203.85 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 24 | 0.183 | 0.183 | 187.2 | 46.8 | 374.4 | 45.7 / 46.5 / 48.6 | 5.30 / 5.33 / 5.33 | 5.27 / 5.47 | 5.47 / 5.49 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 48 | 0.983 | 0.983 | 1,006.4 | 251.6 | 2,012.8 | 213.9 / 228.3 / 228.6 | 7.70 / 7.89 / 8.00 | 7.75 / 8.10 | 8.09 / 8.30 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 96 | 2.190 | 0.000 | **2,243.0** | 560.7 | 4,485.9 | 550.7 / 861.2 / 862.0 | 13.72 / 14.01 / 14.24 | 13.46 / 14.41 | 14.63 / 14.90 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.165 | 0.165 | 168.6 | 42.2 | 1,517.4 | 236.8 / 239.6 / 239.6 | 5.71 / 5.75 / 5.75 | 5.69 / 5.98 | 6.07 / 6.11 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 40 | 0.712 | 0.000 | 728.9 | 182.2 | 6,560.5 | 1,176.8 / 1,664.8 / 1,693.5 | 9.83 / 10.72 / 10.84 | 9.33 / 10.18 | 11.21 / 11.38 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.195 | 0.000 | 1,223.2 | 305.8 | 11,008.8 | 1,913.8 / 5,902.1 / 6,290.9 | 24.66 / 25.32 / 25.73 | 19.88 / 22.95 | 26.69 / 31.13 | Partial (E2E) |

#### `gptoss120b_bf16_1h_4c_torchtpu_fp8kv_tp8_ep`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `vllm-torchtpu`
- **Precision:** `native bf16 weights (234 GB), fp8 kv` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_1h_4c_torchtpu_fp8kv_tp8_ep/summary.json`](./results/gptoss120b_bf16_1h_4c_torchtpu_fp8kv_tp8_ep/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 4.094 | 3.967 | 86.0 | 21.5 | 4,278.7 | 44.7 / 49.0 / 2,115.7 | 5.15 / 5.72 / 6.07 | 5.22 / 5.95 | 0.15 / 0.16 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 21.232 | 18.578 | 445.9 | 111.5 | 22,187.5 | 213.8 / 311.2 / 311.6 | 6.92 / 14.98 / 17.79 | 7.19 / 7.91 | 0.35 / 0.45 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 12 | 0.040 | 0.000 | 161.8 | 40.4 | 1,051.6 | 877.1 / 882.1 / 882.7 | 5.96 / 6.01 / 6.01 | 5.93 / 6.14 | 24.69 / 24.89 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 30 | 0.122 | 0.000 | 489.1 | 122.3 | 3,179.2 | 3,032.5 / 4,684.6 / 4,723.1 | 11.51 / 12.07 / 12.16 | 11.08 / 11.72 | 49.10 / 49.62 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 36 | 0.158 | 0.000 | 631.6 | 157.9 | 4,105.1 | 4,596.3 / 13,334.6 / 14,304.7 | 22.13 / 23.16 / 23.43 | 20.08 / 21.48 | 93.76 / 101.64 | Above target |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.185 | 0.000 | 739.8 | 185.0 | 4,808.8 | 6,785.8 / 25,454.4 / 27,089.6 | 39.25 / 40.73 / 41.01 | 34.08 / 36.46 | 169.47 / 178.59 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 28 | 0.169 | 0.000 | 675.3 | 168.8 | 4,389.6 | 950.7 / 1,862.1 / 2,194.3 | 29.89 / 31.03 / 31.11 | 30.44 / 31.57 | 121.29 / 125.08 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 36 | 0.186 | 0.000 | 742.9 | 185.7 | 4,828.8 | 3,007.2 / 6,401.4 / 7,866.9 | 42.61 / 44.52 / 44.69 | 39.55 / 42.01 | 174.04 / 179.57 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 36 | 0.191 | 0.000 | 765.6 | 191.4 | 4,976.6 | 6,392.3 / 12,763.8 / 13,460.5 | 42.97 / 45.71 / 46.00 | 39.82 / 42.31 | 178.24 / 184.18 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 40 | 0.196 | 0.000 | 783.8 | 195.9 | 5,094.5 | 11,314.3 / 21,013.6 / 21,718.0 | 46.80 / 50.03 / 50.28 | 43.27 / 45.72 | 198.47 / 201.61 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 24 | 0.183 | 0.183 | 187.0 | 46.7 | 373.9 | 45.1 / 46.0 / 47.3 | 5.30 / 5.36 / 5.38 | 5.26 / 5.50 | 5.47 / 5.53 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 48 | 0.985 | 0.985 | 1,008.8 | 252.2 | 2,017.6 | 214.4 / 233.2 / 233.5 | 7.73 / 7.87 / 7.96 | 7.73 / 8.13 | 8.11 / 8.21 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 96 | 2.231 | 0.000 | **2,284.8** | 571.2 | 4,569.6 | 557.1 / 865.5 / 866.1 | 13.47 / 13.76 / 13.91 | 13.24 / 13.93 | 14.30 / 14.65 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.171 | 0.171 | 175.1 | 43.8 | 1,575.5 | 238.7 / 241.7 / 241.9 | 5.48 / 5.52 / 5.53 | 5.45 / 5.65 | 5.85 / 5.88 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 40 | 0.712 | 0.000 | 728.9 | 182.2 | 6,560.4 | 1,222.1 / 1,668.6 / 1,696.2 | 9.81 / 10.73 / 10.79 | 9.29 / 10.13 | 11.24 / 11.36 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.214 | 0.000 | 1,243.1 | 310.8 | 11,187.5 | 1,916.1 / 5,908.6 / 6,295.4 | 24.33 / 24.83 / 25.29 | 19.44 / 22.52 | 26.24 / 30.80 | Partial (E2E) |

#### `gptoss120b_bf16_1h_4c_vllmtpu_bf16kv_tp8ep`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `vllm-tpu (JAX / tpu-inference)`
- **Precision:** `native bf16 weights (234 GB), bf16 kv` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_1h_4c_vllmtpu_bf16kv_tp8ep/summary.json`](./results/gptoss120b_bf16_1h_4c_vllmtpu_bf16kv_tp8ep/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 4.625 | 4.625 | 97.1 | 24.3 | 4,832.9 | 40.1 / 41.5 / 259.5 | 8.30 / 8.42 / 8.49 | 8.50 / 8.88 | 0.21 / 0.21 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 16.582 | 11.400 | 348.2 | 87.1 | 17,328.7 | 246.9 / 414.2 / 414.5 | 8.55 / 18.68 / 20.70 | 8.59 / 9.33 | 0.47 / 0.59 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 12 | 0.029 | 0.000 | 114.0 | 28.5 | 741.1 | 880.5 / 927.9 / 968.8 | 8.54 / 8.70 / 8.74 | 8.56 / 9.10 | 35.02 / 35.67 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 30 | 0.155 | 0.000 | 618.7 | 154.7 | 4,021.3 | 2,992.8 / 4,573.5 / 4,594.6 | 8.95 / 9.51 / 9.55 | 8.51 / 9.17 | 38.88 / 39.15 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 36 | 0.233 | 0.000 | 932.7 | 233.2 | 6,062.3 | 5,749.2 / 10,923.8 / 11,807.2 | 13.39 / 14.38 / 14.56 | 11.74 / 12.58 | 59.32 / 64.10 | Above target |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.312 | 0.000 | 1,249.9 | 312.5 | 8,124.2 | 6,636.6 / 22,375.3 / 23,949.8 | 21.91 / 22.82 / 23.17 | 17.15 / 18.75 | 96.87 / 106.33 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 28 | 0.248 | 0.000 | 991.1 | 247.8 | 6,442.4 | 905.4 / 1,812.9 / 2,131.0 | 17.06 / 17.58 / 17.66 | 14.74 / 16.39 | 69.78 / 71.22 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 36 | 0.328 | 0.000 | 1,311.6 | 327.9 | 8,525.4 | 2,822.5 / 4,723.0 / 5,098.7 | 21.66 / 23.96 / 24.05 | 18.72 / 19.87 | 89.71 / 96.69 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 36 | 0.345 | 0.000 | 1,381.1 | 345.3 | 8,976.9 | 5,929.2 / 11,629.3 / 12,319.0 | 22.16 / 25.06 / 25.19 | 18.97 / 20.29 | 94.53 / 101.15 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 40 | 0.359 | 0.000 | 1,434.3 | 358.6 | 9,323.3 | 10,711.3 / 19,647.5 / 20,222.3 | 23.83 / 27.01 / 27.34 | 20.16 / 21.66 | 106.01 / 109.53 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 24 | 0.119 | 0.119 | 122.0 | 30.5 | 243.9 | 40.6 / 41.4 / 42.4 | 8.21 / 8.33 / 8.38 | 8.17 / 8.59 | 8.44 / 8.56 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 48 | 0.923 | 0.923 | 945.1 | 236.3 | 1,890.2 | 221.6 / 298.9 / 299.2 | 8.26 / 8.41 / 8.50 | 8.18 / 8.70 | 8.65 / 8.76 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 96 | 2.406 | 0.000 | **2,463.7** | 615.9 | 4,927.4 | 683.8 / 840.2 / 861.3 | 12.33 / 12.89 / 12.97 | 12.18 / 12.79 | 13.31 / 13.40 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.117 | 0.117 | 119.9 | 30.0 | 1,079.5 | 263.6 / 267.9 / 268.0 | 8.10 / 8.25 / 8.30 | 8.11 / 8.51 | 8.54 / 8.71 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 40 | 0.764 | 0.076 | 782.7 | 195.7 | 7,044.7 | 1,311.5 / 1,784.7 / 1,814.7 | 8.99 / 9.92 / 10.05 | 8.46 / 8.89 | 10.49 / 10.61 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.521 | 0.000 | 1,557.4 | 389.3 | 14,016.5 | 2,030.9 / 6,307.4 / 6,695.9 | 18.90 / 19.69 / 20.15 | 14.07 / 15.04 | 20.96 / 25.64 | Partial (E2E) |

#### `gptoss120b_bf16_1h_4c_vllmtpu_fp8kv_tp8ep`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `vllm-tpu (JAX / tpu-inference)`
- **Precision:** `native bf16 weights (234 GB), fp8 kv` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, DP=1, EP=true` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_1h_4c_vllmtpu_fp8kv_tp8ep/summary.json`](./results/gptoss120b_bf16_1h_4c_vllmtpu_fp8kv_tp8ep/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 4.578 | 4.578 | 96.1 | 24.0 | 4,783.7 | 40.3 / 41.7 / 264.3 | 8.37 / 8.64 / 8.81 | 8.54 / 9.32 | 0.21 / 0.22 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 18.065 | 13.549 | 379.4 | 94.8 | 18,878.0 | 225.4 / 406.9 / 407.3 | 8.59 / 17.69 / 19.49 | 8.76 / 9.70 | 0.40 / 0.58 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 12 | 0.029 | 0.000 | 114.4 | 28.6 | 743.7 | 880.3 / 924.5 / 963.4 | 8.48 / 8.74 / 8.76 | 8.51 / 9.16 | 34.84 / 35.82 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 30 | 0.151 | 0.000 | 603.9 | 151.0 | 3,925.2 | 2,999.1 / 4,567.9 / 4,599.0 | 9.19 / 9.76 / 9.82 | 8.76 / 9.46 | 39.73 / 40.25 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 36 | 0.235 | 0.000 | 941.7 | 235.4 | 6,120.8 | 5,743.0 / 10,913.9 / 11,805.9 | 13.00 / 14.11 / 14.28 | 11.44 / 12.19 | 58.36 / 62.19 | Above target |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.326 | 0.000 | 1,302.6 | 325.6 | 8,466.8 | 6,633.7 / 22,351.7 / 23,919.2 | 20.91 / 21.83 / 22.18 | 16.25 / 17.55 | 92.90 / 102.33 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 28 | 0.252 | 0.000 | 1,008.4 | 252.1 | 6,554.3 | 903.5 / 1,821.7 / 2,141.3 | 16.59 / 17.30 / 17.40 | 14.07 / 16.08 | 68.05 / 70.20 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 36 | 0.330 | 0.000 | 1,321.9 | 330.5 | 8,592.2 | 2,794.4 / 4,692.2 / 5,057.8 | 21.44 / 23.70 / 23.80 | 18.51 / 19.61 | 88.82 / 95.67 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 36 | 0.352 | 0.000 | 1,406.4 | 351.6 | 9,141.3 | 5,902.6 / 11,579.0 / 12,270.2 | 21.69 / 24.60 / 24.73 | 18.34 / 20.00 | 92.66 / 99.31 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 40 | 0.372 | 0.000 | 1,487.5 | 371.9 | 9,668.6 | 10,683.6 / 19,583.3 / 20,154.2 | 22.84 / 26.02 / 26.35 | 19.28 / 20.52 | 102.03 / 105.57 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 24 | 0.118 | 0.118 | 120.7 | 30.2 | 241.3 | 41.1 / 42.9 / 45.3 | 8.26 / 8.36 / 8.38 | 8.22 / 8.77 | 8.49 / 8.59 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 48 | 0.890 | 0.890 | 911.8 | 227.9 | 1,823.6 | 223.6 / 293.7 / 298.1 | 8.49 / 8.88 / 8.89 | 8.49 / 9.01 | 8.91 / 9.31 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 96 | 2.481 | 0.000 | **2,540.3** | 635.1 | 5,080.6 | 657.1 / 954.3 / 955.4 | 12.02 / 12.44 / 12.60 | 11.74 / 12.35 | 12.87 / 12.99 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.115 | 0.115 | 117.5 | 29.4 | 1,057.8 | 266.5 / 270.0 / 270.9 | 8.30 / 8.37 / 8.37 | 8.24 / 8.73 | 8.76 / 8.83 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 40 | 0.759 | 0.038 | 777.3 | 194.3 | 6,996.0 | 1,375.9 / 1,786.8 / 1,819.2 | 8.96 / 10.00 / 10.04 | 8.50 / 9.02 | 10.51 / 10.64 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.546 | 0.000 | 1,582.9 | 395.7 | 14,246.2 | 2,027.9 / 6,318.3 / 6,704.7 | 18.48 / 19.35 / 19.80 | 13.74 / 14.73 | 20.63 / 25.22 | Partial (E2E) |

#### `gptoss120b_bf16_2h_8c_sglang_bf16kv_tp8ep8`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `sglang-torchtpu (2x single-host replicas + least_conn LB)`
- **Precision:** `native bf16 weights (234 GB), bf16 kv` | **Topology:** `2 hosts (8 chips / 16 TensorCores), 2x independent TP=8 EP=8 replicas` (`8` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_2h_8c_sglang_bf16kv_tp8ep8/summary.json`](./results/gptoss120b_bf16_2h_8c_sglang_bf16kv_tp8ep8/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 5.160 | 4.999 | 108.4 | 13.5 | 5,392.0 | 65.7 / 210.3 / 542.0 | 5.08 / 5.56 / 5.57 | 5.21 / 6.20 | 0.17 / 0.30 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 64 | 28.220 | 21.606 | 592.6 | 74.1 | 29,490.3 | 118.7 / 170.5 / 171.9 | 8.11 / 10.77 / 10.90 | 5.70 / 8.12 | 0.28 / 0.29 | Partial (TTFT+E2E) |
| `smoke` | `1,024 : 21` | `CON=32` | 128 | 46.768 | 5.481 | 982.1 | 122.8 | 48,872.6 | 329.1 / 525.0 / 526.8 | 18.03 / 28.46 / 30.65 | 8.20 / 14.05 | 0.68 / 0.70 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 10 | 0.048 | 0.000 | 192.3 | 24.0 | 1,249.7 | 771.4 / 790.8 / 798.3 | 5.05 / 5.16 / 5.17 | 5.08 / 6.02 | 20.93 / 21.43 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 48 | 0.229 | 0.000 | 916.2 | 114.5 | 5,955.6 | 1,348.6 / 1,962.3 / 2,063.4 | 6.10 / 6.37 / 6.40 | 5.88 / 6.42 | 25.81 / 26.36 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 64 | 0.451 | 0.000 | 1,802.4 | 225.3 | 11,715.6 | 2,930.9 / 5,231.5 / 5,379.6 | 8.11 / 8.66 / 8.72 | 7.47 / 7.99 | 35.41 / 35.81 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 64 | 0.608 | 0.000 | 2,433.8 | 304.2 | 15,819.8 | 5,114.0 / 10,033.8 / 10,887.0 | 11.79 / 12.81 / 12.95 | 10.33 / 11.16 | 52.43 / 56.46 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=64` | 96 | 0.559 | 0.000 | 2,234.1 | 279.3 | 14,521.5 | 11,626.0 / 72,385.8 / 74,250.7 | 16.14 / 19.21 / 24.10 | 12.83 / 13.73 | 75.19 / 139.25 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 36 | 0.357 | 0.000 | 1,428.4 | 178.5 | 9,284.3 | 750.3 / 868.6 / 987.0 | 10.09 / 11.29 / 11.41 | 9.49 / 10.34 | 41.23 / 45.92 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 48 | 0.485 | 0.000 | 1,939.0 | 242.4 | 12,603.3 | 752.1 / 8,121.0 / 14,362.1 | 13.83 / 15.07 / 15.28 | 12.47 / 13.39 | 56.14 / 61.13 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 56 | 0.538 | 0.000 | 2,153.9 | 269.2 | 14,000.6 | 1,488.8 / 46,458.6 / 47,876.6 | 16.04 / 18.30 / 19.69 | 12.81 / 13.77 | 66.96 / 81.86 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 60 | 0.543 | 0.000 | 2,172.7 | 271.6 | 14,122.4 | 4,613.9 / 57,736.0 / 58,086.2 | 15.83 / 17.29 / 19.65 | 12.59 / 13.46 | 67.41 / 97.96 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | 60 | 0.548 | 0.000 | 2,193.4 | 274.2 | 14,256.9 | 6,640.9 / 61,819.9 / 62,506.5 | 15.85 / 17.30 / 19.67 | 12.51 / 13.55 | 68.88 / 101.89 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 64 | 1.303 | 1.303 | 1,333.9 | 166.7 | 2,667.7 | 156.9 / 174.6 / 186.9 | 5.77 / 5.93 / 6.01 | 5.74 / 6.36 | 6.02 / 6.24 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 128 | 3.330 | 3.017 | 3,409.6 | 426.2 | 6,819.1 | 357.1 / 513.4 / 515.6 | 9.05 / 9.31 / 9.47 | 8.91 / 9.41 | 9.56 / 9.85 | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | 192 | 4.965 | 0.000 | **5,084.5** | 635.6 | 10,169.0 | 533.0 / 984.6 / 1,033.6 | 11.99 / 12.37 / 12.49 | 11.57 / 12.28 | 12.87 / 13.03 | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | 256 | 4.949 | 0.000 | 5,067.3 | 633.4 | 10,134.7 | 13,265.1 / 13,813.1 / 13,957.0 | 12.08 / 12.52 / 24.43 | 11.65 / 12.39 | 25.67 / 25.98 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 48 | 1.165 | 0.582 | 1,192.8 | 149.1 | 10,735.3 | 570.1 / 877.9 / 917.6 | 6.13 / 6.47 / 6.51 | 5.77 / 6.30 | 6.80 / 6.93 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 96 | 2.417 | 0.000 | 2,475.2 | 309.4 | 22,276.5 | 1,521.0 / 3,080.4 / 3,514.9 | 11.33 / 12.48 / 12.66 | 9.48 / 10.01 | 13.19 / 14.00 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | 128 | 3.155 | 0.000 | 3,230.7 | 403.8 | 29,076.1 | 2,551.3 / 6,397.8 / 7,003.6 | 17.23 / 18.92 / 19.29 | 12.91 / 13.78 | 19.97 / 23.90 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | 256 | 3.204 | 0.000 | 3,281.0 | 410.1 | 29,528.8 | 20,534.9 / 25,771.4 / 26,889.7 | 18.56 / 19.00 / 19.21 | 12.85 / 13.57 | 39.58 / 45.05 | Partial (E2E) |

#### `gptoss120b_bf16_2h_8c_vllmtpu_bf16kv_tp8ep`
- **Model:** `lmsys/gpt-oss-120b-bf16` | **Framework:** `vllm-tpu (2x single-host replicas + least_conn LB)`
- **Precision:** `native bf16 weights (234 GB), bf16 kv` | **Topology:** `2 hosts (8 chips / 16 TensorCores), 2x independent TP=8 EP=true replicas` (`8` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_bf16_2h_8c_vllmtpu_bf16kv_tp8ep/summary.json`](./results/gptoss120b_bf16_2h_8c_vllmtpu_bf16kv_tp8ep/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 2.872 | 2.692 | 60.3 | 7.5 | 3,001.1 | 46.4 / 1,355.1 / 3,003.1 | 5.49 / 6.25 / 12.82 | 5.41 / 6.12 | 0.16 / 1.54 | Partial (TPOT+E2E) |
| `smoke` | `1,024 : 21` | `CON=8` | 64 | 27.405 | 20.554 | 575.5 | 71.9 | 28,638.4 | 147.4 / 275.5 / 276.7 | 6.14 / 10.98 / 16.93 | 6.20 / 6.95 | 0.27 / 0.44 | Partial (TTFT+E2E) |
| `smoke` | `1,024 : 21` | `CON=32` | 128 | 20.956 | 0.982 | 440.1 | 55.0 | 21,898.9 | 450.7 / 3,039.0 / 3,560.0 | 14.58 / 53.56 / 152.76 | 8.64 / 100.43 | 0.74 / 3.96 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 10 | 0.040 | 0.000 | 160.7 | 20.1 | 1,044.4 | 882.6 / 889.0 / 890.2 | 5.99 / 6.05 / 6.05 | 5.96 / 6.15 | 24.85 / 25.06 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 48 | 0.169 | 0.000 | 677.7 | 84.7 | 4,405.3 | 1,883.7 / 2,444.2 / 2,457.5 | 8.38 / 8.66 / 8.72 | 8.23 / 8.57 | 35.40 / 35.75 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 64 | 0.272 | 0.000 | 1,087.8 | 136.0 | 7,070.8 | 3,869.1 / 5,889.9 / 6,230.6 | 13.71 / 14.49 / 14.56 | 13.14 / 13.91 | 58.56 / 59.39 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 64 | 0.338 | 0.000 | 1,351.0 | 168.9 | 8,781.8 | 4,021.7 / 11,656.0 / 12,539.7 | 22.61 / 23.31 / 23.44 | 20.59 / 21.93 | 94.65 / 101.95 | Above target |
| `long_cl` | `22,000 : 4,000` | `CON=64` | 96 | 0.365 | 0.000 | 1,460.5 | 182.6 | 9,493.0 | 6,831.1 / 25,620.2 / 27,311.1 | 39.87 / 41.33 / 41.71 | 34.61 / 37.37 | 171.87 / 181.00 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 36 | 0.266 | 0.000 | 1,062.8 | 132.8 | 6,908.2 | 902.6 / 1,009.2 / 1,192.2 | 19.50 / 21.13 / 21.34 | 18.48 / 22.65 | 78.90 / 85.40 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 48 | 0.325 | 0.000 | 1,300.1 | 162.5 | 8,450.6 | 916.1 / 1,310.7 / 1,417.8 | 27.69 / 29.24 / 29.43 | 27.78 / 29.44 | 111.63 / 118.14 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 56 | 0.363 | 0.000 | 1,453.0 | 181.6 | 9,444.5 | 1,729.4 / 3,408.6 / 3,673.2 | 34.31 / 36.09 / 36.27 | 32.04 / 33.86 | 140.12 / 145.33 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 60 | 0.378 | 0.000 | 1,511.4 | 188.9 | 9,824.4 | 6,224.4 / 9,157.1 / 9,615.1 | 36.31 / 38.68 / 38.92 | 33.62 / 35.61 | 151.68 / 156.16 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | 60 | 0.377 | 0.000 | 1,507.1 | 188.4 | 9,796.1 | 8,131.9 / 13,399.7 / 14,055.6 | 36.53 / 38.96 / 39.26 | 33.86 / 35.71 | 154.13 / 157.50 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 64 | 1.151 | 1.151 | 1,178.5 | 147.3 | 2,357.0 | 148.5 / 160.9 / 163.2 | 6.63 / 6.75 / 6.79 | 6.57 / 6.87 | 6.93 / 7.02 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 128 | 3.113 | 2.967 | 3,188.0 | 398.5 | 6,376.1 | 378.4 / 466.4 / 565.6 | 9.65 / 9.87 / 10.00 | 9.56 / 10.12 | 10.27 / 10.32 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | 192 | 4.333 | 0.000 | 4,437.3 | 554.7 | 8,874.7 | 653.6 / 937.8 / 949.8 | 13.85 / 14.28 / 14.35 | 13.53 / 14.31 | 14.71 / 15.12 | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | 256 | 5.437 | 0.000 | **5,567.6** | 696.0 | 11,135.2 | 958.5 / 4,011.6 / 4,027.7 | 20.91 / 23.78 / 23.88 | 20.21 / 21.67 | 23.38 / 25.31 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 48 | 0.942 | 0.235 | 964.6 | 120.6 | 8,681.1 | 769.2 / 940.6 / 943.0 | 7.54 / 8.09 / 8.15 | 7.35 / 7.82 | 8.46 / 8.59 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 96 | 1.942 | 0.000 | 1,988.8 | 248.6 | 17,899.4 | 1,796.7 / 2,897.3 / 3,215.0 | 14.36 / 15.62 / 15.77 | 12.82 / 14.36 | 16.43 / 17.58 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | 128 | 2.393 | 0.000 | 2,450.3 | 306.3 | 22,052.6 | 1,914.9 / 5,960.2 / 6,308.9 | 24.41 / 25.26 / 25.70 | 19.80 / 22.98 | 26.65 / 30.93 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | 256 | 2.748 | 0.000 | 2,814.4 | 351.8 | 25,329.7 | 2,002.3 / 11,546.0 / 12,653.7 | 43.73 / 44.08 / 44.67 | 33.00 / 39.29 | 46.20 / 56.58 | Partial (E2E) |

### 3.2 Optional: Native MXFP4 Experts (`openai/gpt-oss-120b`, `65.3 GB`) with FP8 KV

#### `gptoss120b_1h_4c_vllmtpu_fp8kv_tp2dp4`
- **Model:** `openai/gpt-oss-120b` | **Framework:** `vllm-tpu (JAX / tpu-inference)`
- **Precision:** `mxfp4 moe / bf16 attn, fp8 kv` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=2, DP=4` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_1h_4c_vllmtpu_fp8kv_tp2dp4/summary.json`](./results/gptoss120b_1h_4c_vllmtpu_fp8kv_tp2dp4/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 5.041 | 5.041 | 105.9 | 26.5 | 5,268.3 | 48.9 / 378.9 / 402.4 | 5.37 / 5.45 / 5.55 | 5.44 / 5.85 | 0.16 / 0.49 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 20.884 | 14.357 | 438.6 | 109.6 | 21,823.4 | 192.1 / 338.2 / 343.4 | 8.27 / 15.21 / 19.48 | 5.85 / 36.55 | 0.38 / 0.48 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 20 | 0.042 | 0.000 | 166.3 | 41.6 | 1,080.7 | 1,279.3 / 1,283.1 / 1,284.3 | 5.70 / 5.71 / 5.72 | 5.69 / 6.02 | 24.06 / 24.14 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 50 | 0.187 | 0.000 | 749.1 | 187.3 | 4,868.9 | 2,768.8 / 4,797.9 / 4,804.0 | 6.74 / 7.28 / 7.37 | 6.26 / 6.88 | 30.28 / 31.53 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 50 | 0.345 | 0.000 | 1,379.5 | 344.9 | 8,966.9 | 6,543.5 / 9,602.4 / 9,664.8 | 8.24 / 9.62 / 9.65 | 7.45 / 8.21 | 39.88 / 40.40 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 50 | 0.447 | 0.000 | 1,789.4 | 447.4 | 11,631.3 | 11,287.9 / 18,780.6 / 19,141.6 | 11.66 / 13.53 / 14.15 | 9.45 / 10.70 | 58.74 / 62.06 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 32 | 0.324 | 0.000 | 1,295.3 | 323.8 | 8,419.5 | 1,297.1 / 2,156.8 / 2,215.5 | 12.36 / 13.55 / 13.73 | 8.19 / 10.07 | 50.69 / 55.76 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 50 | 0.516 | 0.000 | 2,064.5 | 516.1 | 13,419.3 | 1,702.0 / 3,132.9 / 3,669.0 | 16.32 / 19.32 / 19.66 | 11.43 / 13.14 | 67.45 / 78.64 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 50 | 0.625 | 0.000 | 2,498.5 | 624.6 | 16,240.2 | 3,752.7 / 9,308.9 / 10,428.5 | 15.41 / 18.56 / 19.02 | 11.66 / 13.08 | 67.14 / 76.38 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | 50 | 0.662 | 0.000 | 2,647.0 | 661.7 | 17,205.3 | 13,247.6 / 17,157.9 / 18,038.1 | 14.76 / 17.70 / 18.32 | 11.81 / 13.78 | 71.17 / 74.38 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 32 | 0.176 | 0.176 | 180.7 | 45.2 | 361.4 | 51.3 / 52.5 / 52.7 | 5.49 / 5.52 / 5.52 | 5.48 / 5.84 | 5.67 / 5.69 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 64 | 1.263 | 1.263 | 1,293.6 | 323.4 | 2,587.3 | 71.2 / 176.1 / 215.6 | 6.03 / 6.44 / 6.48 | 5.91 / 6.81 | 6.25 / 6.71 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 128 | 3.562 | 3.340 | 3,647.8 | 911.9 | 7,295.6 | 217.5 / 601.8 / 632.7 | 8.50 / 8.96 / 9.00 | 8.00 / 9.13 | 9.02 / 9.30 | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | 128 | 5.710 | 0.000 | **5,847.0** | 1,461.7 | 11,694.0 | 443.8 / 966.3 / 968.4 | 10.35 / 10.64 / 10.80 | 9.80 / 11.33 | 11.10 / 11.56 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.167 | 0.167 | 170.9 | 42.7 | 1,538.5 | 300.5 / 303.2 / 303.5 | 5.56 / 5.58 / 5.59 | 5.55 / 5.90 | 5.99 / 6.01 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 48 | 0.998 | 0.416 | 1,022.2 | 255.5 | 9,199.6 | 620.9 / 1,273.8 / 1,298.3 | 7.14 / 7.78 / 7.80 | 6.10 / 6.99 | 7.98 / 8.38 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 2.420 | 0.000 | 2,478.6 | 619.6 | 22,307.1 | 2,663.5 / 4,379.9 / 4,381.6 | 10.29 / 12.37 / 12.48 | 8.60 / 9.72 | 13.11 / 14.38 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | 96 | 2.633 | 0.000 | 2,696.6 | 674.2 | 24,269.5 | 4,563.5 / 8,449.4 / 8,468.9 | 15.34 / 17.80 / 18.49 | 10.96 / 12.30 | 22.11 / 23.11 | Partial (E2E) |

#### `gptoss120b_2h_8c_vllmtpu_fp8kv_tp2dp4`
- **Model:** `openai/gpt-oss-120b` | **Framework:** `vllm-tpu JAX (2x single-host replicas + least_conn LB)`
- **Precision:** `mxfp4 moe / bf16 attn, fp8 kv` | **Topology:** `2 hosts (8 chips / 16 TensorCores), 2x independent replicas` (`8` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_2h_8c_vllmtpu_fp8kv_tp2dp4/summary.json`](./results/gptoss120b_2h_8c_vllmtpu_fp8kv_tp2dp4/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 4.172 | 4.172 | 87.6 | 11.0 | 4,359.6 | 50.5 / 384.6 / 388.3 | 5.37 / 5.53 / 5.59 | 5.43 / 5.91 | 0.16 / 0.49 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 64 | 31.731 | 31.731 | 666.3 | 83.3 | 33,158.5 | 129.9 / 138.5 / 181.1 | 5.54 / 6.02 / 6.06 | 5.49 / 6.30 | 0.24 / 0.25 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=32` | 128 | 58.186 | 22.274 | 1,221.9 | 152.7 | 60,804.0 | 248.5 / 650.4 / 674.2 | 10.54 / 16.53 / 28.49 | 6.46 / 51.12 | 0.43 / 0.85 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 10 | 0.041 | 0.000 | 165.9 | 20.7 | 1,078.2 | 1,389.1 / 1,400.8 / 1,402.5 | 5.68 / 5.71 / 5.71 | 5.68 / 6.02 | 24.11 / 24.18 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 48 | 0.235 | 0.000 | 940.9 | 117.6 | 6,115.6 | 2,468.8 / 2,547.8 / 2,567.7 | 5.76 / 6.06 / 6.12 | 5.68 / 6.37 | 25.46 / 25.82 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 64 | 0.525 | 0.000 | 2,099.3 | 262.4 | 13,645.7 | 4,160.5 / 4,895.5 / 4,908.4 | 6.59 / 7.29 / 7.30 | 6.34 / 7.08 | 30.44 / 30.55 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 64 | 0.796 | 0.000 | 3,182.0 | 397.8 | 20,683.2 | 7,264.1 / 9,666.2 / 9,694.3 | 8.24 / 9.67 / 9.71 | 7.50 / 8.36 | 40.10 / 40.23 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=64` | 96 | 0.829 | 0.000 | 3,315.7 | 414.5 | 21,551.8 | 12,149.5 / 19,118.4 / 19,182.9 | 11.49 / 13.77 / 14.31 | 9.67 / 10.94 | 59.70 / 61.51 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 36 | 0.368 | 0.000 | 1,472.3 | 184.0 | 9,570.1 | 1,285.6 / 1,847.0 / 2,025.9 | 9.40 / 10.33 / 10.36 | 6.75 / 7.73 | 38.89 / 42.61 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 48 | 0.596 | 0.000 | 2,382.3 | 297.8 | 15,485.3 | 1,323.6 / 2,329.5 / 2,374.6 | 11.72 / 13.88 / 14.02 | 8.09 / 9.29 | 48.57 / 56.98 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 56 | 0.863 | 0.000 | 3,450.7 | 431.3 | 22,429.6 | 1,637.6 / 2,482.2 / 2,954.6 | 12.14 / 14.14 / 14.36 | 9.09 / 10.22 | 50.02 / 57.94 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 60 | 1.011 | 0.000 | 4,042.8 | 505.3 | 26,278.0 | 3,781.8 / 6,554.5 / 7,906.7 | 11.82 / 13.85 / 13.97 | 9.35 / 10.81 | 51.70 / 56.74 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | 60 | 1.032 | 0.000 | 4,127.5 | 515.9 | 26,828.6 | 3,600.3 / 9,928.5 / 10,478.3 | 11.27 / 13.81 / 13.94 | 9.38 / 10.64 | 50.51 / 56.83 | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 64 | 1.371 | 1.371 | 1,403.5 | 175.4 | 2,807.1 | 125.7 / 159.6 / 164.7 | 5.56 / 5.67 / 5.68 | 5.46 / 6.35 | 5.81 / 5.87 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 128 | 4.165 | 4.165 | 4,265.0 | 533.1 | 8,529.9 | 142.1 / 345.8 / 348.8 | 7.33 / 7.78 / 7.83 | 6.71 / 8.06 | 7.60 / 8.22 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | 192 | 6.882 | 5.054 | 7,047.6 | 881.0 | 14,095.3 | 283.3 / 626.6 / 634.6 | 8.42 / 9.53 / 9.63 | 7.90 / 9.55 | 9.17 / 9.89 | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | 256 | 10.057 | 0.000 | 10,298.0 | 1,287.2 | 20,595.9 | 563.8 / 1,149.2 / 1,164.4 | 10.39 / 13.49 / 13.61 | 9.85 / 13.63 | 11.42 / 14.02 | Partial (E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=256` | 512 | 13.943 | 0.000 | **14,277.1** | 1,784.6 | 28,554.3 | 725.4 / 1,991.2 / 1,999.3 | 16.59 / 17.88 / 18.02 | 14.99 / 18.70 | 18.19 / 18.67 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 48 | 1.263 | 0.316 | 1,293.1 | 161.6 | 11,638.1 | 621.3 / 664.4 / 666.5 | 5.58 / 5.84 / 5.85 | 5.53 / 6.27 | 6.32 / 6.37 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 96 | 3.299 | 0.619 | 3,377.9 | 422.2 | 30,401.1 | 1,458.9 / 2,309.5 / 2,315.1 | 7.96 / 9.12 / 9.20 | 7.00 / 7.85 | 9.79 / 10.05 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | 128 | 4.798 | 0.000 | 4,913.0 | 614.1 | 44,217.4 | 2,507.4 / 4,393.5 / 4,409.4 | 10.51 / 12.41 / 12.58 | 8.60 / 9.73 | 13.16 / 14.61 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | 256 | 6.381 | 0.000 | 6,533.7 | 816.7 | 58,803.6 | 4,218.3 / 8,451.1 / 8,519.3 | 15.41 / 18.41 / 19.24 | 11.00 / 13.00 | 19.88 / 23.37 | Partial (E2E) |

#### `gptoss120b_4h_16c_vllmtpu_fp8kv_tp2dp4`
- **Model:** `openai/gpt-oss-120b` | **Framework:** `vllm-tpu JAX (4x single-host replicas + least_conn LB)`
- **Precision:** `mxfp4 moe / bf16 attn, fp8 kv` | **Topology:** `4 hosts (16 chips / 32 TensorCores), 4x independent replicas` (`16` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_4h_16c_vllmtpu_fp8kv_tp2dp4/summary.json`](./results/gptoss120b_4h_16c_vllmtpu_fp8kv_tp2dp4/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 4.176 | 4.176 | 87.7 | 5.5 | 4,363.8 | 51.6 / 382.5 / 393.1 | 5.36 / 5.58 / 5.62 | 5.45 / 5.83 | 0.16 / 0.49 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 64 | 43.612 | 43.612 | 915.9 | 57.2 | 45,575.0 | 50.5 / 128.0 / 140.7 | 5.41 / 5.68 / 5.77 | 5.45 / 6.03 | 0.16 / 0.24 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=32` | 128 | 74.938 | 48.007 | 1,573.7 | 98.4 | 78,309.8 | 126.0 / 504.5 / 538.1 | 8.19 / 14.72 / 20.70 | 5.93 / 38.72 | 0.34 / 0.69 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 10 | 0.041 | 0.000 | 165.7 | 10.4 | 1,077.0 | 1,381.2 / 1,397.1 / 1,400.4 | 5.70 / 5.72 / 5.72 | 5.70 / 6.02 | 24.14 / 24.25 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 48 | 0.241 | 0.000 | 963.5 | 60.2 | 6,262.7 | 1,288.3 / 2,171.8 / 2,545.0 | 5.73 / 6.01 / 6.04 | 5.69 / 6.05 | 25.17 / 25.39 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 64 | 0.622 | 0.000 | 2,486.5 | 155.4 | 16,162.3 | 2,491.0 / 2,586.8 / 2,596.0 | 5.77 / 6.11 / 6.16 | 5.68 / 6.49 | 25.48 / 25.97 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 80 | 0.876 | 0.000 | 3,505.3 | 219.1 | 22,784.5 | 4,163.1 / 4,916.4 / 4,928.3 | 6.56 / 7.29 / 7.30 | 6.32 / 6.92 | 30.43 / 30.61 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=64` | 96 | 1.198 | 0.000 | 4,793.8 | 299.6 | 31,159.5 | 7,346.1 / 9,706.7 / 9,724.9 | 8.17 / 9.60 / 9.67 | 7.47 / 8.25 | 39.96 / 40.18 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=128` | 128 | 2.187 | 0.000 | 8,749.7 | 546.9 | 56,873.2 | 14,099.2 / 19,209.0 / 19,238.8 | 11.09 / 13.80 / 14.19 | 9.56 / 10.68 | 58.25 / 58.44 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 36 | 0.375 | 0.000 | 1,498.2 | 93.6 | 9,738.3 | 1,288.5 / 1,355.0 / 1,367.9 | 6.69 / 7.49 / 7.53 | 5.70 / 6.55 | 28.02 / 31.32 | Partial (TPOT+E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 48 | 0.648 | 0.000 | 2,592.7 | 162.0 | 16,852.8 | 1,287.3 / 1,336.8 / 1,387.1 | 8.18 / 9.05 / 9.11 | 6.29 / 7.26 | 34.02 / 37.46 | Partial (TPOT+E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 56 | 0.971 | 0.000 | 3,884.5 | 242.8 | 25,249.3 | 1,313.2 / 2,068.3 / 2,183.6 | 9.37 / 10.96 / 11.13 | 7.07 / 8.19 | 39.42 / 45.15 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=4.0` | 60 | 1.278 | 0.000 | 5,110.2 | 319.4 | 33,216.5 | 1,773.8 / 2,368.7 / 2,598.0 | 9.28 / 10.54 / 10.81 | 7.39 / 8.28 | 38.96 / 43.92 | Partial (E2E) |
| `long_ol` | `22,000 : 4,000` | `RPS=6.0` | 60 | 1.391 | 0.000 | 5,564.6 | 347.8 | 36,169.7 | 1,746.9 / 3,584.3 / 3,907.6 | 8.82 / 9.96 / 10.08 | 7.43 / 8.28 | 37.90 / 41.11 | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 64 | 1.391 | 1.391 | 1,423.9 | 89.0 | 2,847.8 | 53.3 / 157.4 / 162.9 | 5.51 / 5.55 / 5.58 | 5.47 / 5.93 | 5.69 / 5.81 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 128 | 4.848 | 4.848 | 4,964.5 | 310.3 | 9,929.1 | 73.0 / 264.8 / 351.8 | 6.26 / 6.49 / 6.54 | 5.93 / 6.84 | 6.51 / 6.76 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=64` | 192 | 8.111 | 7.477 | 8,305.2 | 519.1 | 16,610.4 | 177.6 / 510.7 / 650.4 | 7.43 / 7.95 / 8.04 | 6.68 / 8.28 | 7.70 / 8.37 | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=128` | 256 | 13.734 | 8.584 | 14,063.9 | 879.0 | 28,127.8 | 295.5 / 659.7 / 666.2 | 8.33 / 9.18 / 9.34 | 7.89 / 9.55 | 9.07 / 9.54 | Partial (TPOT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=256` | 512 | 19.890 | 0.000 | **20,366.9** | 1,272.9 | 40,733.9 | 456.5 / 1,304.0 / 1,337.9 | 10.73 / 13.52 / 13.74 | 9.86 / 14.10 | 11.81 / 14.13 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 48 | 1.257 | 0.629 | 1,287.5 | 80.5 | 11,587.7 | 489.4 / 652.8 / 655.9 | 5.76 / 5.86 / 5.95 | 5.54 / 5.98 | 6.31 / 6.43 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 96 | 4.039 | 1.220 | 4,136.0 | 258.5 | 37,224.2 | 895.2 / 1,300.8 / 1,307.1 | 6.61 / 7.52 / 7.74 | 6.07 / 7.06 | 7.82 / 8.09 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=64` | 128 | 6.604 | 0.929 | 6,762.9 | 422.7 | 60,866.0 | 1,765.1 / 2,337.0 / 2,342.0 | 7.93 / 9.00 / 9.07 | 7.00 / 7.86 | 9.58 / 10.02 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=128` | 256 | 9.591 | 0.000 | 9,820.8 | 613.8 | 88,387.3 | 2,516.9 / 4,415.4 / 4,429.2 | 10.42 / 12.34 / 12.52 | 8.57 / 9.65 | 13.11 / 14.71 | Partial (E2E) |

#### `gptoss120b_1h_4c_sglang_fp8kv_tp8ep8`
- **Model:** `openai/gpt-oss-120b` | **Framework:** `sglang-torchtpu`
- **Precision:** `mxfp4 moe / bf16 attn, fp8_e4m3 kv` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_1h_4c_sglang_fp8kv_tp8ep8/summary.json`](./results/gptoss120b_1h_4c_sglang_fp8kv_tp8ep8/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 7.944 | 7.944 | 166.8 | 41.7 | 8,301.8 | 43.5 / 46.2 / 75.7 | 4.01 / 4.47 / 4.61 | 4.13 / 4.69 | 0.12 / 0.13 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 19.576 | 11.623 | 411.1 | 102.8 | 20,456.8 | 230.7 / 292.9 / 294.1 | 6.73 / 18.68 / 18.89 | 6.35 / 9.62 | 0.42 / 0.42 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 16 | 0.059 | 0.000 | 234.0 | 58.5 | 1,521.3 | 737.0 / 742.1 / 742.1 | 4.09 / 4.14 / 4.14 | 4.03 / 4.61 | 17.08 / 17.29 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 36 | 0.181 | 0.000 | 724.3 | 181.1 | 4,707.8 | 2,263.3 / 3,845.2 / 3,913.8 | 7.72 / 8.13 / 8.20 | 7.28 / 7.76 | 33.11 / 33.63 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 48 | 0.311 | 0.000 | 1,242.2 | 310.6 | 8,074.4 | 4,785.0 / 9,108.3 / 10,405.0 | 11.74 / 12.59 / 12.84 | 10.17 / 10.87 | 51.13 / 53.81 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.289 | 0.000 | 1,154.8 | 288.7 | 7,505.9 | 11,169.4 / 71,000.7 / 72,313.4 | 15.83 / 19.04 / 22.12 | 12.58 / 13.60 | 72.94 / 137.31 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 32 | 0.254 | 0.000 | 1,014.8 | 253.7 | 6,596.4 | 1,019.9 / 20,036.9 / 23,188.9 | 14.22 / 16.06 / 16.32 | 12.40 / 13.34 | 61.87 / 65.77 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 40 | 0.296 | 0.000 | 1,184.5 | 296.1 | 7,699.1 | 2,429.6 / 47,924.7 / 48,334.7 | 15.80 / 19.13 / 24.92 | 12.45 / 13.37 | 67.80 / 105.27 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 40 | 0.306 | 0.000 | 1,225.5 | 306.4 | 7,965.6 | 4,844.9 / 59,126.1 / 59,758.6 | 15.72 / 20.09 / 24.92 | 12.46 / 13.32 | 76.40 / 117.77 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 32 | 0.239 | 0.239 | 244.8 | 61.2 | 489.6 | 44.5 / 46.7 / 47.9 | 4.05 / 4.14 / 4.15 | 3.95 / 4.73 | 4.19 / 4.28 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 64 | 1.104 | 1.104 | 1,130.5 | 282.6 | 2,261.1 | 269.6 / 292.9 / 294.3 | 6.85 / 7.06 / 7.08 | 6.80 / 7.15 | 7.26 / 7.30 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 128 | 2.527 | 0.000 | **2,587.5** | 646.9 | 5,175.0 | 496.1 / 896.5 / 926.6 | 11.86 / 12.25 / 12.39 | 11.52 / 12.22 | 12.66 / 12.77 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.229 | 0.229 | 235.0 | 58.7 | 2,114.7 | 245.1 / 247.1 / 248.0 | 4.02 / 4.07 / 4.07 | 3.96 / 4.52 | 4.36 / 4.40 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 48 | 0.888 | 0.222 | 909.7 | 227.4 | 8,187.4 | 996.6 / 1,661.3 / 1,747.9 | 7.82 / 8.59 / 8.67 | 7.10 / 7.48 | 9.02 / 9.15 | Partial (TPOT+E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.592 | 0.000 | 1,630.1 | 407.5 | 14,670.9 | 2,403.2 / 6,319.4 / 6,875.9 | 17.15 / 18.96 / 19.21 | 12.95 / 13.79 | 19.97 / 23.86 | Partial (E2E) |

#### `gptoss120b_1h_4c_sglang_fp8kv_eagle3_tp8ep8`
- **Model:** `openai/gpt-oss-120b` | **Framework:** `sglang-torchtpu`
- **Precision:** `mxfp4 moe / bf16 attn, fp8_e4m3 kv + EAGLE3 (steps=3, topk=1, draft=4)` | **Topology:** `1 host (4 chips / 8 TensorCores), TP=8, EP=8` (`4` physical `tpu7x` chips) | **Raw JSON:** [`results/gptoss120b_1h_4c_sglang_fp8kv_eagle3_tp8ep8/summary.json`](./results/gptoss120b_1h_4c_sglang_fp8kv_eagle3_tp8ep8/summary.json)

| Suite | ISL : OSL | Load | n | Req/s | Goodput (req/s) | Out tok/s | Out tok/s/chip | Total tok/s | TTFT P50 / P95 / P99 (ms) | TPOT P50 / P95 / P99 (ms) | ITL P50 / P95 (ms) | E2E P50 / P95 (s) | Latency Target (P95) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `smoke` | `1,024 : 21` | `CON=1` | 32 | 6.788 | 6.788 | 142.5 | 35.6 | 7,093.1 | 44.9 / 46.1 / 52.1 | 5.25 / 5.79 / 6.13 | 6.62 / 7.55 | 0.15 / 0.16 | Pass (All) |
| `smoke` | `1,024 : 21` | `CON=8` | 32 | 16.507 | 0.516 | 346.6 | 86.7 | 17,249.8 | 94.7 / 236.3 / 236.4 | 18.06 / 25.20 / 26.71 | 14.46 / 83.18 | 0.46 / 0.62 | Partial (TTFT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=1` | 16 | 0.047 | 0.000 | 187.0 | 46.8 | 1,215.6 | 746.7 / 756.8 / 765.5 | 5.05 / 6.07 / 6.09 | 6.86 / 7.38 | 20.95 / 25.00 | Partial (TPOT+E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=6` | 36 | 0.120 | 0.000 | 478.1 | 119.5 | 3,107.4 | 749.3 / 2,886.3 / 3,890.7 | 11.96 / 13.51 / 14.21 | 15.30 / 15.82 | 48.57 / 56.67 | Partial (E2E) |
| `long_cl` | `22,000 : 4,000` | `CON=16` | 48 | 0.243 | 0.000 | 973.7 | 243.4 | 6,329.1 | 770.6 / 9,201.6 / 10,425.0 | 15.21 / 17.93 / 18.91 | 18.34 / 19.32 | 63.42 / 76.14 | Above target |
| `long_cl` | `22,000 : 4,000` | `CON=32` | 48 | 0.244 | 0.000 | 977.7 | 244.4 | 6,355.1 | 13,538.2 / 77,274.6 / 78,662.6 | 18.09 / 26.48 / 36.40 | 20.75 / 21.60 | 89.28 / 156.05 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=0.5` | 32 | 0.214 | 0.000 | 855.3 | 213.8 | 5,559.7 | 1,011.5 / 29,128.6 / 34,870.0 | 16.56 / 19.31 / 21.05 | 20.39 / 21.15 | 75.08 / 87.73 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=1.0` | 40 | 0.253 | 0.000 | 1,011.8 | 253.0 | 6,577.0 | 2,429.7 / 58,831.6 / 59,632.3 | 17.88 / 25.99 / 28.80 | 20.78 / 21.66 | 100.58 / 127.59 | Above target |
| `long_ol` | `22,000 : 4,000` | `RPS=2.0` | 40 | 0.263 | 0.000 | 1,052.6 | 263.2 | 6,842.1 | 5,384.6 / 71,861.7 / 72,508.9 | 17.73 / 22.20 / 27.71 | 20.56 / 21.88 | 92.40 / 137.88 | Above target |
| `ref_1k1k` | `1,024 : 1,024` | `CON=1` | 32 | 0.207 | 0.207 | 212.1 | 53.0 | 424.1 | 44.4 / 49.0 / 53.0 | 4.63 / 5.44 / 5.78 | 6.77 / 7.53 | 4.79 / 5.62 | Pass (All) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=8` | 64 | 0.711 | 0.178 | 728.5 | 182.1 | 1,457.0 | 59.8 / 145.1 / 237.1 | 10.73 / 12.73 / 13.41 | 15.27 / 15.98 | 11.07 / 13.08 | Partial (TTFT+E2E) |
| `ref_1k1k` | `1,024 : 1,024` | `CON=32` | 128 | 2.082 | 0.016 | **2,132.5** | 533.1 | 4,264.9 | 69.4 / 757.3 / 846.6 | 13.98 / 16.79 / 17.21 | 19.73 / 21.00 | 14.49 / 17.51 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=1` | 20 | 0.187 | 0.187 | 191.8 | 47.9 | 1,726.0 | 246.0 / 246.9 / 247.2 | 5.00 / 5.46 / 5.59 | 6.79 / 7.28 | 5.36 / 5.83 | Pass (All) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=8` | 48 | 0.586 | 0.012 | 600.3 | 150.1 | 5,402.8 | 255.8 / 1,272.7 / 1,678.1 | 12.21 / 14.83 / 15.24 | 15.60 / 16.24 | 12.91 / 15.65 | Partial (E2E) |
| `ref_8k1k` | `8,192 : 1,024` | `CON=32` | 64 | 1.318 | 0.000 | 1,349.1 | 337.3 | 12,142.2 | 578.8 / 6,431.3 / 6,990.7 | 20.12 / 26.88 / 28.13 | 21.32 / 22.41 | 22.04 / 30.33 | Partial (E2E) |

---

## 4. Follow-Up: One Host at Higher Load (MXFP4 + FP8 KV)

These runs came after sections 1 to 3. They use the native MXFP4 checkpoint (`openai/gpt-oss-120b`) with FP8 KV cache on one v7x host (4 chips, 8 TensorCores) and push to higher load than before. Raw results are in [`results/gptoss120b_1h_4c_mxfp4_followup/`](./results/gptoss120b_1h_4c_mxfp4_followup/).

### 4.1 What is new compared with sections 1 to 3

- **`vllm-torchtpu` now serves MXFP4.** [`scripts/patches/apply_vllm_torchtpu_gptoss_mxfp4_patch.py`](./scripts/patches/apply_vllm_torchtpu_gptoss_mxfp4_patch.py) registers the `gpt_oss_mxfp4` method for TPU, packs the experts into native 4-bit K-major FP4, pads `hidden_size` 2880 to 2944 (the same alignment issue as in section 1.B), fixes the 64-wide attention head KV layout, and lets the DFlash draft model share the target model's embeddings. Every sanity prompt returned the correct answer (37 × 43 = 1591) and readable text.
- **`sglang-torchtpu` with data-parallel attention.** `--enable-dp-attention --dp-size 4` (or 8) splits attention into 4 groups of 2 TensorCores (or 8 single cores) while the experts stay split over all 8 cores (`EP=8`). With `TP=8`, sglang requires `--enable-dp-lm-head` for this.
- **Larger batches and request limits.** In sglang, `--cuda-graph-max-bs-decode` sets the largest decode batch with a compiled graph and `--max-running-requests` caps requests in flight. In the vLLM stacks, `--max-num-seqs` caps running requests per data-parallel replica. We raised these limits and measured up to 8,192 concurrent requests, 8k-token prompts at up to 2,048 requests, and 22k-token prompts with 4k outputs at 8 to 1,024 requests.
- **DFlash speculative decoding** with [`z-lab/gpt-oss-120b-DFlash`](https://huggingface.co/z-lab/gpt-oss-120b-DFlash), a small draft model that proposes 8 or 16 tokens at a time.
- **Newer container images** than sections 1 to 3. To check that the change of image does not explain the differences, we re-ran the section B JAX `vllm-tpu` setup (`TP=2, DP=4`, 64 running requests per replica) with the same client: 5,788.5 tok/s at 64 requests vs. 5,847.0 in section B (1% apart).

### 4.2 Clients

- `vllm-tpu` (JAX) and `vllm-torchtpu` were measured with `vllm bench serve`, like sections 1 to 3.
- `sglang-torchtpu` was measured with `sglang.bench_serving` (`--backend sglang-oai --random-range-ratio 1.0`).
- Both clients send random-token prompts of exactly the stated length, ignore end-of-sequence tokens so every request produces the full output length, and keep a fixed number of requests in flight. To check that the client does not change the comparison, we also measured one sglang setup with `vllm bench serve` (section 4.5).
- Most points are one run. We repeated the best high-load setup of each stack with a fresh server and the same loads:
  - `vllm-torchtpu`, 1,024 requests per replica: 19,672.4 and 19,667.6 tok/s at 2,048 requests, 24,357.1 and 23,875.7 at 4,096 (2.0% apart).
  - JAX `vllm-tpu`, 1,024 per replica: 22,952.2 and 22,059.4 at 2,048 (3.9% apart), 22,046.9 and 21,420.5 at 4,096 (2.8%).
  - JAX `vllm-tpu`, 512 per replica: 17,190.0 and 17,588.6 at 512 (2.3%), 22,163.6 and 21,688.5 at 1,024 (2.1%), 21,387.7 and 22,276.0 at 2,048 (4.2%).
  - `sglang-torchtpu`, 8 attention groups, decode batch 256, 16,384-token chunks: 16,246.4 and 16,029.4 at 1,024 (1.3%), 16,046.6 and 15,670.0 at 2,048 (2.3%).
  - `vllm-torchtpu`, 2,048 requests per replica: 21,451.4 and 21,580.3 tok/s at 4,096 requests (0.6% apart), 26,738.2 and 26,626.0 at 8,192 (0.4%).
  - JAX `vllm-tpu`, 22k in / 4k out at 512 requests (128 per replica): 5,648.1 and 5,640.1 tok/s (0.1%).
- Two runs of the same `vllm-torchtpu` layout at 128 requests, where neither run's request limit was reached, gave 5,313.7 and 5,806.9 tok/s, so differences under about 10% between separate runs should not be read as real.

### 4.3 Setups

`sglang-torchtpu` (all with `TP=8`, `EP=8`, FP8 KV cache, 26,624-token context limit, radix cache off; exact command lines in [`scripts/benchmark_sglang_mxfp4_single_host.py`](./scripts/benchmark_sglang_mxfp4_single_host.py)):

| Setup | Attention groups | `torch.compile` | Decode graphs up to batch | Running-request limit | DFlash draft tokens | Other | Result |
| :--- | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| `sglang_mxfp4_tp8_ep8_baseline` | 1 | no | – | default | – | memory fraction 0.84 | measured |
| `sglang_mxfp4_tp8_ep8_compiled_baseline` | 1 | yes | 8 | default | – | – | measured |
| `sglang_mxfp4_tp8_ep8_dflash8` | 1 | yes | 8 | default | 8 | – | measured |
| `sglang_mxfp4_tp8_ep8_dflash16` | 1 | yes | 8 | default | 16 | – | measured |
| `sglang_mxfp4_dp4_tp8_ep8_dflash8_exp` | 4 | yes | 8 | default | 8 | – | measured |
| `sglang_mxfp4_dp4_tp8_ep8_dflash16_exp` | 4 | yes | 8 | default | 16 | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_dflash8_exp` | 8 | yes | 8 | default | 8 | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_dflash16_exp` | 8 | yes | 8 | default | 16 | – | measured |
| `sglang_mxfp4_dp4_tp8_ep8_dflash8_bs16` | 4 | yes | 16 | default | 8 | – | measured |
| `sglang_mxfp4_dp4_tp8_ep8_compiled_bs32` | 4 | yes | 32 | default | – | – | measured |
| `sglang_mxfp4_dp4_tp8_ep8_compiled_bs64` | 4 | yes | 64 | 256 | – | – | measured |
| `sglang_mxfp4_dp4_tp8_ep8_compiled_bs128` | 4 | yes | 128 | 512 | – | – | measured |
| `sglang_mxfp4_dp4_tp8_ep8_compiled_bs256` | 4 | yes | 256 | 1,024 | – | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs32` | 8 | yes | 32 | 256 | – | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs64` | 8 | yes | 64 | 512 | – | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128` | 8 | yes | 128 | 1,024 | – | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256` | 8 | yes | 256 | 2,048 | – | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k` | 8 | yes | 128 | 1,024 | – | prefill chunk 16,384 | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_8k` | 8 | yes | 128 | 1,024 | – | – | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk32k` | 8 | yes | 128 | 1,024 | – | prefill chunk 32,768 | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_8k` | 8 | yes | 128 | 1,024 | – | prefill chunk 16,384 | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256_chunk16k` | 8 | yes | 256 | 2,048 | – | prefill chunk 16,384 | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_22k` | 8 | yes | 128 | 1,024 | – | prefill chunk 16,384 | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_22k_hi` | 8 | yes | 128 | 1,024 | – | prefill chunk 16,384 | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256_chunk16k_repeat` | 8 | yes | 256 | 2,048 | – | prefill chunk 16,384 | measured |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_bf16kv` | 8 | yes | 128 | 1,024 | – | prefill chunk 16,384 | not run |
| `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256_chunk16k_bf16kv` | 8 | yes | 256 | 2,048 | – | prefill chunk 16,384 | not run |
| `sglang_mxfp4_tp8_ep8_compiled_chunk32k` | 1 | yes | 8 | default | – | prefill chunk 32,768 | measured |
| `sglang_mxfp4_tp8_ep8_dflash16_chunk32k` | 1 | yes | 8 | default | 16 | prefill chunk 32,768 | not completed (pod evicted) |

`vllm-torchtpu` with the MXFP4 patch (all with FP8 KV cache, 26,624-token context limit, prefix cache off; exact flags in [`scripts/benchmark_vllm_torchtpu_mxfp4.py`](./scripts/benchmark_vllm_torchtpu_mxfp4.py)):

| Setup | Layout | Running requests per replica | Batched tokens per step | Memory fraction | Other flags | Result |
| :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| `vllm_torchtpu_mxfp4_tp8_dp1` | `TP=8, DP=1` | 128 | 16,384 | 0.88 | expert parallel | measured |
| `vllm_torchtpu_mxfp4_tp4_dp2_async` | `TP=4, DP=2` | 64 | 16,384 | 0.88 | expert parallel, async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async` | `TP=2, DP=4` | 64 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp8_dflash8` | `TP=8, DP=1` | 32 | 8,192 | 0.80 | expert parallel, DFlash, 8 draft tokens | measured |
| `vllm_torchtpu_mxfp4_tp8_dflash16` | `TP=8, DP=1` | 32 | 8,192 | 0.80 | expert parallel, DFlash, 16 draft tokens | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_dflash8` | `TP=2, DP=4` | 32 | 8,192 | 0.80 | async scheduling, DFlash, 8 draft tokens | failed to start |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs128` | `TP=2, DP=4` | 128 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256` | `TP=2, DP=4` | 256 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512` | `TP=2, DP=4` | 512 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024` | `TP=2, DP=4` | 1,024 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512_bt16384` | `TP=2, DP=4` | 512 | 16,384 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_8k` | `TP=2, DP=4` | 256 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512_8k` | `TP=2, DP=4` | 512 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_8k_bt16384` | `TP=2, DP=4` | 256 | 16,384 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024_8k` | `TP=2, DP=4` | 1,024 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k` | `TP=2, DP=4` | 256 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_hi` | `TP=2, DP=4` | 256 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024_repeat` | `TP=2, DP=4` | 1,024 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_c512` | `TP=2, DP=4` | 256 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp4_dp2_async_seqs2048` | `TP=4, DP=2` | 2,048 | 16,384 | 0.88 | expert parallel, async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_c1024` | `TP=2, DP=4` | 256 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs2048` | `TP=2, DP=4` | 2,048 | 8,192 | 0.88 | async scheduling | measured |
| `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs2048_repeat` | `TP=2, DP=4` | 2,048 | 8,192 | 0.88 | async scheduling | measured |

`vllm-tpu` (JAX): the single-host MXFP4 settings from section B ([`k8s/gpt-oss-120b-vllm-tpu-v7x.yaml`](./k8s/gpt-oss-120b-vllm-tpu-v7x.yaml)): `TP=2, DP=4`, 16,384 batched tokens per step, memory fraction 0.86, FP8 KV cache, async scheduling, chunked prefill, prefix cache off. Only the running-request limit per replica changed between runs (64, 128, 512 and 1,024); further runs used 8k-token prompts with 128 and 512 per replica, and 22k-token prompts with 4k outputs at 128 per replica. Script: [`scripts/run_vllm_tpu_jax_high_load.sh`](./scripts/run_vllm_tpu_jax_high_load.sh).

### 4.4 Full results

TTFT is time to first token and TPOT is time per output token (both medians). "Accepted tokens per step" is the average number of tokens each decode step produced with DFlash. "failed" means that point did not complete. When the number of concurrent requests is above the setup's running-request limit, the extra requests wait in a queue, which shows up as a long time to first token.

#### `sglang-torchtpu` (measured with `sglang.bench_serving`)

##### `sglang_mxfp4_tp8_ep8_baseline`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 8.4 | 16.9 | 285 | 118.16 |
| 1k / 1k, 8 | 62.3 | 124.6 | 5,197 | 123.50 |
| 1k / 1k, 32 | 225.7 | 451.5 | 12,856 | 129.35 |
| 1k / 1k, 64 | 244.0 | 488.0 | 128,137 | 126.87 |
| 8k / 1k, 1 request | 8.4 | 75.5 | 421 | 117.30 |
| 8k / 1k, 32 | 115.3 | 1,037.4 | 34,575 | 230.59 |
| 22k / 4k, 1 request | 8.1 | 52.8 | 979 | 122.23 |
| 22k / 4k, 8 | 59.1 | 384.3 | 19,463 | 130.73 |

##### `sglang_mxfp4_dp8_tp8_ep8_dflash8_exp`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 229.3 | 458.6 | 158 | 4.45 | 2.63 |
| 1k / 1k, 8 | 1,147.5 | 2,295.0 | 168 | 5.81 | 2.80 |
| 1k / 1k, 32 | 2,712.4 | 5,424.7 | 182 | 9.61 | 2.77 |
| 1k / 1k, 64 | 3,794.3 | 7,588.6 | 287 | 14.08 | 2.82 |
| 8k / 1k, 1 request | 130.7 | 1,176.2 | 1,163 | 5.55 | 3.12 |
| 8k / 1k, 32 | 983.8 | 8,854.1 | 1,246 | 28.61 | 2.47 |
| 22k / 4k, 1 request | 131.4 | 854.4 | 3,421 | 7.20 | 2.31 |
| 22k / 4k, 8 | 525.3 | 3,414.5 | 3,478 | 12.17 | 2.45 |

##### `sglang_mxfp4_dp8_tp8_ep8_dflash16_exp`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 189.5 | 379.0 | 160 | 5.19 | 2.48 |
| 1k / 1k, 8 | 864.8 | 1,729.5 | 170 | 7.69 | 2.47 |
| 1k / 1k, 32 | 1,916.7 | 3,833.3 | 199 | 13.70 | 2.84 |
| 1k / 1k, 64 | 2,815.5 | 5,631.0 | 276 | 19.45 | 2.53 |
| 8k / 1k, 1 request | 113.0 | 1,016.9 | 1,164 | 7.07 | 2.68 |
| 8k / 1k, 32 | 862.9 | 7,766.3 | 1,345 | 30.58 | 2.50 |
| 22k / 4k, 1 request | 117.9 | 766.4 | 3,424 | 7.28 | 2.41 |
| 22k / 4k, 8 | 434.1 | 2,821.7 | 3,501 | 13.90 | 2.19 |

##### `sglang_mxfp4_dp4_tp8_ep8_compiled_bs32`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 143.9 | 287.8 | 101 | 6.80 |
| 1k / 1k, 8 | 1,044.8 | 2,089.6 | 213 | 7.46 |
| 1k / 1k, 32 | 2,667.0 | 5,334.0 | 438 | 11.57 |
| 1k / 1k, 64 | 3,685.3 | 7,370.6 | 882 | 15.87 |
| 8k / 1k, 1 request | 132.9 | 1,196.0 | 696 | 6.83 |
| 8k / 1k, 32 | 1,573.5 | 14,161.4 | 3,523 | 16.78 |
| 22k / 4k, 1 request | 133.2 | 865.6 | 2,010 | 6.93 |
| 22k / 4k, 8 | 896.1 | 5,824.5 | 4,135 | 7.98 |
| 1k / 1k, 128 | 6,103.0 | 12,206.1 | 1,634 | 19.06 |

##### `sglang_mxfp4_dp4_tp8_ep8_compiled_bs64`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 143.4 | 286.8 | 101 | 6.83 |
| 1k / 1k, 8 | 1,037.3 | 2,074.5 | 203 | 7.47 |
| 1k / 1k, 32 | 2,657.7 | 5,315.5 | 460 | 11.60 |
| 1k / 1k, 64 | 4,177.8 | 8,355.6 | 777 | 14.55 |
| 8k / 1k, 1 request | 130.5 | 1,174.3 | 695 | 6.98 |
| 8k / 1k, 32 | 1,804.9 | 16,244.3 | 3,821 | 14.00 |
| 22k / 4k, 1 request | 135.7 | 882.0 | 2,007 | 6.88 |
| 22k / 4k, 8 | 896.6 | 5,827.7 | 4,141 | 7.95 |
| 1k / 1k, 128 | 6,073.0 | 12,145.9 | 1,315 | 19.53 |
| 1k / 1k, 256 | 8,114.1 | 16,228.3 | 3,174 | 28.00 |

##### `sglang_mxfp4_dp4_tp8_ep8_compiled_bs128`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 128 | 6,271.5 | 12,542.9 | 1,031 | 19.27 |
| 1k / 1k, 256 | 9,133.0 | 18,266.0 | 2,447 | 25.52 |
| 1k / 1k, 512 | 10,988.6 | 21,977.2 | 5,669 | 40.41 |

##### `sglang_mxfp4_dp4_tp8_ep8_compiled_bs256`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 256 | 9,354.2 | 18,708.3 | 2,381 | 24.79 |
| 1k / 1k, 512 | 11,619.3 | 23,238.5 | 4,172 | 39.04 |
| 1k / 1k, 1,024 | 13,499.9 | 26,999.7 | 10,492 | 65.16 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 512 | 12,400.4 | 24,800.8 | 2,937 | 37.54 |
| 1k / 1k, 1,024 | 13,869.7 | 27,739.4 | 10,500 | 62.88 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_8k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 8k / 1k, 64 | 2,482.9 | 22,345.9 | 5,858 | 20.03 |
| 8k / 1k, 128 | 3,165.4 | 28,488.2 | 9,026 | 31.21 |
| 8k / 1k, 256 | 3,783.8 | 34,054.4 | 12,654 | 52.82 |
| 8k / 1k, 512 | 3,823.3 | 34,409.5 | 15,006 | 108.58 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_8k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 8k / 1k, 64 | 2,590.7 | 23,316.4 | 5,558 | 19.41 |
| 8k / 1k, 128 | 3,123.5 | 28,111.4 | 4,574 | 33.79 |
| 8k / 1k, 256 | 3,725.8 | 33,532.0 | 6,539 | 57.86 |
| 8k / 1k, 512 | 3,937.3 | 35,435.8 | 10,076 | 111.85 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_22k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 22k / 4k, 8 | 903.0 | 5,869.6 | 3,777 | 7.91 |
| 22k / 4k, 16 | 1,246.1 | 8,099.7 | 6,623 | 11.25 |
| 22k / 4k, 32 | 1,758.1 | 11,427.9 | 9,605 | 15.86 |
| 22k / 4k, 64 | 2,350.2 | 15,276.5 | 15,792 | 23.54 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256_chunk16k_repeat`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1,024 | 16,029.4 | 32,058.7 | 5,422 | 58.08 |
| 1k / 1k, 2,048 | 15,670.0 | 31,340.1 | 14,372 | 116.45 |

##### `sglang_mxfp4_tp8_ep8_compiled_baseline`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 190.0 | 379.9 | 46 | 5.23 |
| 1k / 1k, 8 | 1,127.0 | 2,253.9 | 138 | 6.95 |
| 1k / 1k, 32 | 2,570.4 | 5,140.8 | 406 | 12.06 |
| 1k / 1k, 64 | 2,586.8 | 5,173.5 | 12,969 | 11.99 |
| 8k / 1k, 1 request | 180.6 | 1,625.7 | 187 | 5.37 |
| 8k / 1k, 32 | 1,741.7 | 15,675.0 | 1,500 | 16.70 |
| 22k / 4k, 1 request | 188.8 | 1,227.3 | 542 | 5.18 |
| 22k / 4k, 8 | 936.0 | 6,083.7 | 2,704 | 7.87 |

##### `sglang_mxfp4_tp8_ep8_dflash8`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 260.8 | 521.5 | 47 | 3.47 | 2.25 |
| 1k / 1k, 8 | 1,444.6 | 2,889.2 | 56 | 4.88 | 2.51 |
| 1k / 1k, 32 | 2,966.3 | 5,932.7 | 323 | 7.61 | 2.61 |
| 1k / 1k, 64 | 3,331.4 | 6,662.8 | 9,181 | 8.28 | 2.61 |
| 8k / 1k, 1 request | 228.6 | 2,057.3 | 191 | 3.96 | 2.59 |
| 8k / 1k, 32 | 1,591.3 | 14,321.5 | 1,751 | 13.05 | 2.47 |
| 22k / 4k, 1 request | 222.0 | 1,443.1 | 552 | 4.68 | 2.44 |
| 22k / 4k, 8 | 863.0 | 5,609.7 | 585 | 7.55 | 2.33 |

##### `sglang_mxfp4_tp8_ep8_dflash16`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 272.9 | 545.7 | 48 | 3.22 | 2.37 |
| 1k / 1k, 8 | 1,181.5 | 2,363.1 | 60 | 5.77 | 2.42 |
| 1k / 1k, 32 | 2,588.2 | 5,176.3 | 412 | 9.98 | 2.53 |
| 1k / 1k, 64 | 2,690.9 | 5,381.8 | 11,263 | 10.25 | 2.52 |
| 8k / 1k, 1 request | 217.4 | 1,956.8 | 191 | 4.32 | 2.50 |
| 8k / 1k, 32 | 1,331.1 | 11,979.6 | 1,954 | 17.22 | 2.37 |
| 22k / 4k, 1 request | 224.6 | 1,459.7 | 549 | 4.21 | 2.34 |
| 22k / 4k, 8 | 715.8 | 4,652.6 | 591 | 9.31 | 2.23 |

##### `sglang_mxfp4_dp4_tp8_ep8_dflash8_exp`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 205.5 | 410.9 | 103 | 4.94 | 2.32 |
| 1k / 1k, 8 | 1,258.2 | 2,516.3 | 112 | 5.72 | 2.30 |
| 1k / 1k, 32 | 2,827.2 | 5,654.4 | 143 | 9.28 | 2.54 |
| 1k / 1k, 64 | 4,004.3 | 8,008.6 | 220 | 13.10 | 2.56 |
| 8k / 1k, 1 request | 144.6 | 1,301.1 | 699 | 5.37 | 2.56 |
| 8k / 1k, 32 | 1,189.7 | 10,707.1 | 815 | 21.44 | 2.66 |
| 22k / 4k, 1 request | 114.5 | 744.5 | 2,018 | 7.98 | 2.63 |
| 22k / 4k, 8 | 636.0 | 4,134.2 | 2,045 | 10.03 | 2.44 |

##### `sglang_mxfp4_dp4_tp8_ep8_dflash16_exp`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 186.7 | 373.4 | 105 | 4.69 | 1.74 |
| 1k / 1k, 8 | 927.4 | 1,854.8 | 114 | 7.39 | 1.88 |
| 1k / 1k, 32 | 2,090.7 | 4,181.5 | 149 | 13.26 | 2.52 |
| 1k / 1k, 64 | 2,673.4 | 5,346.8 | 209 | 20.57 | 2.46 |
| 8k / 1k, 1 request | 122.3 | 1,101.1 | 702 | 7.07 | 2.54 |
| 8k / 1k, 32 | 996.5 | 8,968.3 | 854 | 25.01 | 2.40 |
| 22k / 4k, 1 request | 102.6 | 666.8 | 2,017 | 9.45 | 2.31 |
| 22k / 4k, 8 | 567.9 | 3,691.5 | 2,059 | 12.17 | 2.04 |

##### `sglang_mxfp4_dp4_tp8_ep8_dflash8_bs16`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) | Accepted tokens per step |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 205.4 | 410.8 | 103 | 4.90 | 2.78 |
| 1k / 1k, 8 | 1,125.1 | 2,250.1 | 109 | 5.51 | 2.92 |
| 1k / 1k, 32 | 2,839.7 | 5,679.3 | 136 | 8.93 | 2.73 |
| 1k / 1k, 64 | 3,922.7 | 7,845.4 | 213 | 13.38 | 2.57 |
| 8k / 1k, 1 request | 144.4 | 1,299.2 | 698 | 5.45 | 2.80 |
| 8k / 1k, 32 | 1,190.4 | 10,713.7 | 860 | 21.44 | 2.62 |
| 22k / 4k, 1 request | 116.2 | 755.4 | 2,020 | 7.82 | 2.41 |
| 22k / 4k, 8 | 611.9 | 3,977.3 | 2,043 | 10.32 | 2.28 |
| 1k / 1k, 128 | 5,132.3 | 10,264.6 | 1,559 | 16.92 | 2.48 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs32`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 149.4 | 298.7 | 153 | 6.54 |
| 1k / 1k, 8 | 1,051.1 | 2,102.2 | 248 | 7.36 |
| 1k / 1k, 32 | 2,591.6 | 5,183.2 | 411 | 11.93 |
| 1k / 1k, 64 | 4,139.8 | 8,279.7 | 685 | 14.77 |
| 8k / 1k, 1 request | 129.1 | 1,161.4 | 1,150 | 6.61 |
| 8k / 1k, 32 | 1,832.3 | 16,490.7 | 3,425 | 14.25 |
| 22k / 4k, 1 request | 130.3 | 846.7 | 3,412 | 6.82 |
| 22k / 4k, 8 | 909.8 | 5,913.6 | 3,653 | 7.90 |
| 1k / 1k, 128 | 5,685.2 | 11,370.4 | 1,000 | 20.65 |
| 1k / 1k, 256 | 8,296.2 | 16,592.4 | 2,577 | 27.71 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs64`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 128 | 6,213.2 | 12,426.3 | 867 | 19.15 |
| 1k / 1k, 256 | 8,688.0 | 17,376.0 | 1,494 | 27.13 |
| 1k / 1k, 512 | 11,369.1 | 22,738.1 | 5,557 | 39.25 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1,024 | 13,861.0 | 27,722.0 | 6,174 | 66.22 |
| 1k / 1k, 2,048 | 12,860.8 | 25,721.7 | 17,758 | 140.99 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 512 | 13,479.4 | 26,958.8 | 2,597 | 34.58 |
| 1k / 1k, 1,024 | 15,748.3 | 31,496.6 | 7,098 | 57.59 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk32k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 512 | 13,404.3 | 26,808.6 | 3,850 | 33.94 |
| 1k / 1k, 1,024 | 16,119.6 | 32,239.2 | 7,391 | 55.73 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256_chunk16k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1,024 | 16,246.4 | 32,492.7 | 6,702 | 56.07 |
| 1k / 1k, 2,048 | 16,046.6 | 32,093.1 | 14,608 | 113.22 |

##### `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_22k_hi`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 22k / 4k, 128 | 2,875.3 | 18,689.6 | 14,881 | 39.25 |
| 22k / 4k, 256 | 3,167.3 | 20,587.5 | 27,550 | 72.46 |

##### `sglang_mxfp4_tp8_ep8_compiled_chunk32k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 8k / 1k, 1 request | 183.1 | 1,648.0 | 182 | 5.27 |
| 22k / 4k, 1 request | 185.4 | 1,205.3 | 654 | 5.22 |
| 22k / 4k, 8 | 909.9 | 5,914.6 | 2,817 | 8.08 |

#### `vllm-torchtpu` with the MXFP4 patch (measured with `vllm bench serve`)

##### `vllm_torchtpu_mxfp4_tp8_dp1`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 226.2 | 452.4 | 35 | 4.39 |
| 1k / 1k, 8 | 1,287.3 | 2,574.7 | 163 | 6.06 |
| 1k / 1k, 32 | 2,549.8 | 5,099.6 | 617 | 10.15 |
| 1k / 1k, 64 | 3,511.8 | 7,023.5 | 645 | 16.36 |
| 1k / 1k, 128 | 4,481.1 | 8,962.3 | 1,008 | 26.24 |
| 8k / 1k, 1 request | 213.7 | 1,923.5 | 191 | 4.50 |
| 8k / 1k, 32 | 1,730.3 | 15,572.9 | 1,213 | 17.55 |
| 8k / 1k, 64 | 1,964.2 | 17,678.2 | 1,258 | 31.42 |
| 22k / 4k, 1 request | 202.7 | 1,317.6 | 707 | 4.76 |
| 22k / 4k, 8 | 787.1 | 5,116.2 | 3,116 | 9.38 |
| 22k / 4k, 32 | 1,124.4 | 7,308.7 | 5,418 | 25.51 |

##### `vllm_torchtpu_mxfp4_tp4_dp2_async`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 153.9 | 307.8 | 55 | 6.44 |
| 1k / 1k, 8 | 1,050.1 | 2,100.3 | 300 | 7.27 |
| 1k / 1k, 32 | 2,180.3 | 4,360.7 | 1,133 | 12.58 |
| 1k / 1k, 64 | 3,667.4 | 7,334.8 | 1,237 | 15.53 |
| 1k / 1k, 128 | 5,089.8 | 10,179.6 | 1,625 | 22.78 |
| 8k / 1k, 1 request | 147.4 | 1,326.6 | 352 | 6.44 |
| 8k / 1k, 32 | 1,624.2 | 14,618.2 | 2,461 | 17.51 |
| 8k / 1k, 64 | 1,975.7 | 17,781.3 | 2,505 | 30.11 |
| 22k / 4k, 1 request | 140.6 | 914.0 | 1,315 | 6.77 |
| 22k / 4k, 8 | 687.9 | 4,471.2 | 4,359 | 10.54 |
| 22k / 4k, 32 | 1,059.8 | 6,888.7 | 7,803 | 25.05 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 124.5 | 249.0 | 60 | 7.50 |
| 1k / 1k, 8 | 1,004.8 | 2,009.6 | 151 | 7.81 |
| 1k / 1k, 32 | 2,813.6 | 5,627.1 | 240 | 10.12 |
| 1k / 1k, 64 | 4,173.6 | 8,347.3 | 460 | 13.71 |
| 1k / 1k, 128 | 5,313.7 | 10,627.3 | 608 | 23.20 |
| 8k / 1k, 1 request | 127.9 | 1,150.9 | 322 | 7.54 |
| 8k / 1k, 32 | 2,116.6 | 19,049.4 | 1,626 | 13.23 |
| 8k / 1k, 64 | 2,797.7 | 25,178.9 | 2,345 | 20.64 |
| 22k / 4k, 1 request | 127.6 | 829.1 | 1,299 | 7.52 |
| 22k / 4k, 8 | 796.9 | 5,180.1 | 2,485 | 9.46 |
| 22k / 4k, 32 | 1,481.4 | 9,629.1 | 4,888 | 17.52 |

##### `vllm_torchtpu_mxfp4_tp8_dflash8`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 233.6 | 467.1 | 35 | 4.12 |
| 1k / 1k, 8 | 751.4 | 1,502.7 | 66 | 10.31 |
| 1k / 1k, 32 | 1,234.1 | 2,468.2 | 125 | 24.09 |
| 1k / 1k, 64 | 1,231.3 | 2,462.6 | 25,885 | 24.87 |
| 1k / 1k, 128 | 1,295.4 | 2,590.8 | 73,171 | 24.28 |
| 8k / 1k, 1 request | 186.1 | 1,674.7 | 186 | 5.09 |
| 8k / 1k, 32 | 696.3 | 6,266.5 | 607 | 42.13 |
| 8k / 1k, 64 | 727.9 | 6,551.2 | 43,484 | 41.15 |
| 22k / 4k, 1 request | 176.7 | 1,148.8 | 832 | 5.41 |
| 22k / 4k, 8 | 359.0 | 2,333.7 | 910 | 19.77 |
| 22k / 4k, 32 | 431.7 | 2,805.8 | 7,045 | 64.33 |

##### `vllm_torchtpu_mxfp4_tp8_dflash16`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | 214.3 | 428.6 | 35 | 4.87 |
| 1k / 1k, 8 | 606.6 | 1,213.1 | 73 | 12.09 |
| 1k / 1k, 32 | 976.4 | 1,952.7 | 165 | 31.17 |
| 1k / 1k, 64 | 1,009.6 | 2,019.2 | 31,741 | 29.92 |
| 1k / 1k, 128 | 991.9 | 1,983.8 | 94,694 | 31.73 |
| 8k / 1k, 1 request | 179.4 | 1,614.5 | 187 | 5.39 |
| 8k / 1k, 32 | 614.4 | 5,529.7 | 565 | 47.36 |
| 8k / 1k, 64 | 615.2 | 5,537.2 | 53,151 | 49.44 |
| 22k / 4k, 1 request | 159.6 | 1,037.5 | 830 | 5.98 |
| 22k / 4k, 8 | 313.4 | 2,037.1 | 920 | 24.50 |
| 22k / 4k, 32 | 394.8 | 2,566.4 | 7,056 | 73.30 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs128`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 64 | 3,337.2 | 6,674.4 | 3,178 | 16.68 |
| 1k / 1k, 128 | 5,806.9 | 11,613.8 | 575 | 20.11 |
| 1k / 1k, 256 | 7,849.5 | 15,699.0 | 716 | 31.00 |
| 1k / 1k, 512 | 11,426.5 | 22,853.0 | 901 | 42.74 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 256 | 7,451.6 | 14,903.2 | 861 | 30.71 |
| 1k / 1k, 512 | 11,394.8 | 22,789.7 | 1,114 | 42.76 |
| 1k / 1k, 1,024 | 14,867.1 | 29,734.3 | 1,188 | 62.70 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1,024 | 13,774.0 | 27,548.1 | 1,011 | 67.65 |
| 1k / 1k, 2,048 | 20,284.0 | 40,568.0 | 1,310 | 97.48 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 2,048 | 19,672.4 | 39,344.8 | 1,304 | 94.84 |
| 1k / 1k, 4,096 | 24,357.1 | 48,714.2 | 3,512 | 160.09 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512_bt16384`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 1,024 | 13,320.5 | 26,641.0 | 3,463 | 68.40 |
| 1k / 1k, 2,048 | 19,814.5 | 39,629.0 | 1,813 | 97.46 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_8k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 8k / 1k, 64 | 3,117.3 | 28,055.4 | 1,457 | 18.08 |
| 8k / 1k, 128 | 3,829.3 | 34,463.6 | 1,512 | 30.82 |
| 8k / 1k, 256 | 5,061.6 | 45,554.2 | 1,509 | 47.43 |
| 8k / 1k, 512 | 5,971.3 | 53,741.3 | 1,916 | 81.21 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512_8k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 8k / 1k, 1,024 | 6,881.7 | 61,935.4 | 3,023 | 136.37 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_8k_bt16384`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 8k / 1k, 256 | 4,630.8 | 41,677.6 | 2,803 | 49.79 |
| 8k / 1k, 512 | 5,962.8 | 53,665.2 | 2,318 | 82.20 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024_8k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 8k / 1k, 2,048 | 7,632.6 | 68,692.9 | 8,256 | 236.81 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 22k / 4k, 8 | 833.2 | 5,415.5 | 3,425 | 8.76 |
| 22k / 4k, 16 | 1,495.7 | 9,722.3 | 3,975 | 9.78 |
| 22k / 4k, 32 | 2,193.1 | 14,255.2 | 4,139 | 13.56 |
| 22k / 4k, 64 | 2,825.0 | 18,362.8 | 4,288 | 20.84 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_hi`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 22k / 4k, 128 | 3,479.5 | 22,616.8 | 3,524 | 34.78 |
| 22k / 4k, 256 | 3,974.8 | 25,836.2 | 5,472 | 61.88 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024_repeat`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 2,048 | 19,667.6 | 39,335.2 | 1,303 | 94.85 |
| 1k / 1k, 4,096 | 23,875.7 | 47,751.4 | 3,369 | 160.26 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_c512`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 22k / 4k, 512 | 4,488.9 | 29,177.7 | 5,775 | 108.02 |

##### `vllm_torchtpu_mxfp4_tp4_dp2_async_seqs2048`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 2,048 | 11,126.6 | 22,253.2 | 3,583 | 162.44 |
| 1k / 1k, 4,096 | 13,762.3 | 27,524.5 | 7,407 | 276.83 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_c1024`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 22k / 4k, 1,024 | 4,677.2 | 30,401.4 | 62,031 | 184.17 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs2048`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 4,096 | 21,451.4 | 42,902.8 | 3,705 | 167.95 |
| 1k / 1k, 8,192 | 26,738.2 | 53,476.3 | 15,144 | 275.85 |

##### `vllm_torchtpu_mxfp4_tp2_dp4_async_seqs2048_repeat`

| Input / output, concurrent requests | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 1k / 1k, 4,096 | 21,580.3 | 43,160.6 | 3,574 | 167.78 |
| 1k / 1k, 8,192 | 26,626.0 | 53,252.0 | 15,716 | 277.18 |

#### `vllm-tpu` (JAX), `TP=2, DP=4` (measured with `vllm bench serve`)

##### 64 running requests per replica (section B settings), newer image

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 5,788.5 | 11,577.0 | 515 | 10.49 |
| 128 | 256 | 7,589.8 | 15,179.6 | 857 | 15.80 |
| 256 | 512 | 9,434.6 | 18,869.2 | 1,304 | 22.04 |
| 512 | 1,024 | 10,134.9 | 20,269.9 | 24,874 | 22.93 |

##### 128 running requests per replica

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 512 | 11,283.7 | 22,567.4 | 1,434 | 21.01 |
| 512 | 1,024 | 15,876.6 | 31,753.2 | 2,367 | 29.80 |
| 1,024 | 2,048 | 15,784.4 | 31,568.7 | 33,394 | 31.31 |

##### 512 running requests per replica

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1,024 | 17,190.0 | 34,379.9 | 2,687 | 27.02 |
| 1,024 | 2,048 | 22,163.6 | 44,327.1 | 4,223 | 41.88 |
| 2,048 | 4,096 | 21,387.7 | 42,775.4 | 7,829 | 87.60 |

##### 512 running requests per replica, second run (fresh server, same settings)

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1,024 | 17,588.6 | 35,177.3 | 2,281 | 26.85 |
| 1,024 | 2,048 | 21,688.5 | 43,376.9 | 3,615 | 43.18 |
| 2,048 | 4,096 | 22,276.0 | 44,552.0 | 6,340 | 85.34 |

##### 1,024 running requests per replica

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4,096 | 22,952.2 | 45,904.5 | 6,962 | 81.91 |
| 4,096 | 8,192 | 22,046.9 | 44,093.7 | 12,022 | 171.79 |

##### 1,024 running requests per replica, second run (fresh server, same settings)

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4,096 | 22,059.4 | 44,118.9 | 6,634 | 85.76 |
| 4,096 | 8,192 | 21,420.5 | 42,840.9 | 12,088 | 176.96 |

##### `TP=1, DP=8` (one TensorCore per replica, single-process data parallelism), 1,024 running requests per replica

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1,024 | 12,713.4 | 25,426.9 | 1,974 | 33.66 |
| 1,024 | 2,048 | 17,174.0 | 34,348.0 | 3,595 | 53.32 |
| 2,048 | 4,096 | 22,261.0 | 44,521.9 | 6,169 | 84.33 |

##### 128 running requests per replica, 8k-token prompts (8k in / 1k out)

| Concurrent requests (8k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 4,023.4 | 36,210.6 | 3,609 | 12.28 |
| 128 | 256 | 4,133.2 | 37,199.0 | 7,911 | 23.36 |
| 256 | 512 | 6,035.6 | 54,320.8 | 4,260 | 37.28 |
| 512 | 1,024 | 7,447.1 | 67,023.7 | 3,957 | 64.05 |

##### 512 running requests per replica, 8k-token prompts (8k in / 1k out)

| Concurrent requests (8k in / 1k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2,048 | 8,798.0 | 79,182.4 | 8,892 | 106.57 |
| 2,048 | 4,096 | 9,743.8 | 87,694.1 | 6,405 | 194.71 |

##### 128 running requests per replica, 22k-token prompts (22k in / 4k out); the 128- and 256-request points are from a second server start with the same settings, and the 512-request point from a third

| Concurrent requests (22k in / 4k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 1,079.4 | 7,016.2 | 3,984 | 6.40 |
| 16 | 32 | 1,672.0 | 10,868.2 | 6,957 | 7.83 |
| 32 | 64 | 2,295.6 | 14,921.6 | 12,100 | 10.91 |
| 64 | 128 | 3,120.4 | 20,282.5 | 17,771 | 16.24 |
| 128 | 256 | 3,973.9 | 25,830.6 | 13,748 | 29.37 |
| 256 | 512 | 4,885.5 | 31,755.5 | 14,390 | 48.70 |
| 512 | 1,024 | 5,648.1 | 36,712.7 | 9,821 | 84.66 |

##### 128 running requests per replica, 22k-token prompts (22k in / 4k out), second run of the 512-request point (fresh server, same settings)

| Concurrent requests (22k in / 4k out) | Prompts | Output tok/s | Total tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1,024 | 5,640.1 | 36,660.5 | 9,838 | 84.81 |

### 4.5 Same-client check

`sglang-torchtpu` was measured with its own client and the two vLLM stacks with `vllm bench serve`. To see whether the client changes the numbers, we started the 8-attention-group, decode-batch-64 sglang setup again with the same flags on the same host and measured it with `vllm bench serve`, using exactly the client settings of the vLLM runs. Raw results are in [`results/gptoss120b_1h_4c_mxfp4_followup/sglang_measured_with_vllm_bench/`](./results/gptoss120b_1h_4c_mxfp4_followup/sglang_measured_with_vllm_bench/).

| Concurrent requests (1k in / 1k out) | `sglang.bench_serving`: output tok/s | `vllm bench serve`: output tok/s | Difference | Median TTFT (ms), sglang / vllm client | Median TPOT (ms), sglang / vllm client |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6,213.2 | 6,142.2 | −1.1% | 867 / 1,262 | 19.15 / 19.66 |
| 256 | 8,688.0 | 8,396.3 | −3.4% | 1,494 / 2,265 | 27.13 / 27.47 |
| 512 | 11,369.1 | 11,743.4 | +3.3% | 5,557 / 4,274 | 39.25 / 38.80 |

Output throughput agreed within 3.4% and time per output token within 2.7%, which is inside the run-to-run variation in section 4.2 (the two measurements are separate server launches). Time to first token did not agree: it differed by 23% to 52%, in both directions. So throughput and time per output token can be compared across the two clients, but time to first token under load should only be compared within one client.

### 4.6 What worked and what did not

- **JAX `vllm-tpu` gave the highest throughput from 64 to 2,048 concurrent requests,** without code patches. With 512 requests per replica it gave 17,190.0 tok/s at 512 requests and 22,163.6 at 1,024 (median 41.9 ms per output token); with 1,024 per replica it gave 22,952.2 at 2,048. The best `vllm-torchtpu` and `sglang-torchtpu` results at those loads were 11,426.5 and 13,479.4 (512 requests), 14,867.1 and 16,246.4 (1,024), and 20,284.0 and 16,046.6 (2,048). At 4,096 requests JAX gave 22,046.9 tok/s and `vllm-torchtpu` 24,357.1, and at 8,192 `vllm-torchtpu` gave 26,738.2 (2,048 per replica). At 1,024 requests JAX took longer to return the first token than `vllm-torchtpu` (median 4.2 s vs. 1.19 s, same client).
- **`torch.compile` is essential for `sglang-torchtpu`.** Without it (`sglang_mxfp4_tp8_ep8_baseline`) a single request ran at 8.4 tok/s (118 ms per output token), vs. 190.0 tok/s (5.23 ms) with `--enable-torch-compile`.
- **Data-parallel attention raises throughput under load and costs single-request speed.** At 64 requests, 4 attention groups with decode batch 64 gave 4,177.8 tok/s vs. 2,586.8 for plain `TP=8` (which only compiles decode graphs up to batch 8). With one request, time per output token went from 5.23 ms to 6.83 ms.
- **DFlash helps one request at a time and hurts under load.** With `TP=8`, DFlash-16 raised single-request throughput from 190.0 to 272.9 tok/s (3.22 ms per token). Under load the extra verification work costs more than it saves: with 4 attention groups at 128 requests, DFlash-8 (decode batch 16) gave 5,132.3 tok/s vs. 6,103.0 without DFlash (decode batch 32). On random prompts the draft model accepted between 1.7 and 3.1 tokens per step, mostly 2.3 to 2.8.
- **DFlash-8 beat DFlash-16 under load** (4,004.3 vs. 2,673.4 tok/s at 64 requests with 4 attention groups). Each DFlash-16 step verifies about twice as many tokens while acceptance was similar (2.56 vs. 2.46 tokens per step), which likely explains the gap.
- **DFlash on `vllm-torchtpu`** worked with `TP=8` (233.6 tok/s for one request) but was limited to 32 running requests, so it queued at 64 requests and above. With `TP=2, DP=4` it failed at startup; the saved error output only shows a compiler warning, so we did not find the cause.
- **`vllm-torchtpu` with one TensorCore per replica (`TP=1, DP=8`) is not supported** by this build ("No TPU topology found for count: 1").
- **`vllm-torchtpu` with `TP=4, DP=2` scaled worse at high load.** With 2,048 running requests per replica (expert parallel, 16,384 tokens per step) it gave 11,126.6 tok/s at 2,048 requests and 13,762.3 at 4,096, vs. 19,672.4 and 24,357.1 for `TP=2, DP=4` with 1,024 per replica.
- **JAX `vllm-tpu` with one TensorCore per replica (`TP=1, DP=8`) was not faster.** It needs single-process data parallelism (`TPU_MULTIPROCESS_DP=0`). With up to 1,024 running requests per replica it gave 12,713.4 tok/s at 512 requests, 17,174.0 at 1,024 and 22,261.0 at 2,048, vs. 17,190.0, 22,163.6 and 22,952.2 (22,059.4 in the second run) for `TP=2, DP=4`.
- **For `vllm-torchtpu`, a higher request limit helped only by letting more requests run at once.** At the same load the limit made no clear difference: at 512 requests, 128 and 256 per replica gave 11,426.5 and 11,394.8 tok/s; at 1,024 requests, 256 and 512 per replica gave 14,867.1 and 13,774.0; at 2,048 requests, 512 and 1,024 per replica gave 20,284.0 and 19,672.4. The one exception was 4,096 requests, where 2,048 per replica gave about 11% less than 1,024 per replica (21,451.4 and 21,580.3 tok/s in two runs, vs. 24,357.1 and 23,875.7). Throughput kept rising with load up to the highest load we tried: 24,357.1 tok/s at 4,096 requests (1,024 per replica, median 160.1 ms per output token) and 26,738.2 at 8,192 (2,048 per replica, median 275.9 ms per output token and 15.1 s to the first token).
- **The step budget does not explain why JAX was faster.** With 16,384 batched tokens per step (the JAX setting) instead of 8,192, `vllm-torchtpu` with 512 requests per replica gave 13,320.5 tok/s at 1,024 requests and 19,814.5 at 2,048, vs. 13,774.0 and 20,284.0 with 8,192, which is within run-to-run variation. With 8k prompts, 16,384 tokens per step gave 4,630.8 and 5,962.8 tok/s at 256 and 512 requests, vs. 5,061.6 and 5,971.3 with 8,192.
- **Larger prefill chunks helped `sglang-torchtpu` under heavy load.** With 8 attention groups and decode batch 128, `--chunked-prefill-size 16384` raised throughput at 1,024 requests from 13,869.7 to 15,748.3 tok/s (14% more) and cut the median time to first token there from 10.5 s to 7.1 s. 32,768-token chunks gave 16,119.6 tok/s at 1,024 requests (7.4 s to the first token), within run-to-run variation of the 16,384-token result. With decode batch 256 and 2,048 running requests, 16,384-token chunks gave 16,246.4 tok/s at 1,024 requests and 16,046.6 at 2,048, vs. 13,861.0 and 12,860.8 with the default chunks (17% and 25% more). At 512 requests the gain (12,400.4 to 13,479.4) is within run-to-run variation. With 8k prompts, 16,384-token chunks left throughput about the same (2,590.7 to 3,937.3 vs. 2,482.9 to 3,823.3 tok/s from 64 to 512 requests) and lowered the median time to first token at 128 to 512 requests from 9.0 to 15.0 s to 4.6 to 10.1 s. With the default setting, the server log shows prefill chunks of 512 tokens per attention group. The other high-load sglang setups (different decode batches and attention groups, default chunks) gave 10,988.6 to 12,400.4 tok/s at 512 requests.
- **`sglang-torchtpu` stopped gaining above 1,024 requests.** With 8 attention groups, decode batch 256 and 2,048 running requests, it gave 13,861.0 tok/s at 1,024 requests and 12,860.8 at 2,048 (median 17.8 s to the first token) with the default prefill chunks, and 16,246.4 and 16,046.6 with 16,384-token chunks (median 6.7 s and 14.6 s to the first token).
- **Under heavy load, `sglang-torchtpu` took much longer to return the first token.** At 1,024 requests its median time to first token was 6.2 to 10.5 s across setups (6.7 to 7.4 s with 16,384- and 32,768-token prefill chunks), vs. 1.19 s for `vllm-torchtpu` with 256 requests per replica. All these setups allowed at least 1,024 running requests. The client difference measured in section 4.5 (up to 52%) is much smaller than this gap.
- **Long prompts at higher load (8k in / 1k out).** JAX `vllm-tpu` (128 requests per replica) gave 4,023.4 to 7,447.1 tok/s from 64 to 512 requests, and with 512 per replica 8,798.0 at 1,024 and 9,743.8 at 2,048. `vllm-torchtpu` (256 per replica) gave 3,117.3 to 5,971.3, then 6,881.7 at 1,024 (512 per replica) and 7,632.6 at 2,048 (1,024 per replica, median 236.8 ms per output token). `sglang-torchtpu` (8 attention groups, decode batch 128) gave 2,482.9 to 3,823.3, with a median 5.9 to 15.0 s to the first token. Section B measured 2,696.6 tok/s at 64 requests with 8k prompts using 96 prompts and an older image; these runs used 2 prompts per concurrent request, so the numbers are not directly comparable.
- **22k-token prompts with 4k outputs at 8 to 1,024 requests.** JAX `vllm-tpu` (128 requests per replica) gave the most throughput at every load where it was measured: 1,079.4 tok/s at 8 requests, 3,120.4 at 64, 4,885.5 at 256 and 5,648.1 at 512. `vllm-torchtpu` (256 per replica) gave 833.2, 2,825.0, 3,974.8 and 4,488.9 at the same loads, and 4,677.2 at 1,024. `sglang-torchtpu` (8 attention groups, decode batch 128, 16,384-token prefill chunks; measured up to 256 requests) gave 903.0, 2,350.2 and 3,167.3 at 8, 64 and 256 requests. JAX had the lowest time per output token (6.4 ms at 8 requests, 16.2 ms at 64, 48.7 ms at 256 and 84.7 ms at 512, vs. 8.8, 20.8, 61.9 and 108.0 ms for `vllm-torchtpu`), but returned the first token later: a median 4.0 s at 8 requests and 9.8 to 17.8 s from 32 to 512 requests, vs. 3.4 to 5.8 s for `vllm-torchtpu` up to 512 requests (same client). At 1,024 requests, where all 1,024 prompts arrive at once and every running slot is in use, `vllm-torchtpu` took a median 62.0 s to the first token. The JAX points come from three server starts with the same settings (8 to 64, 128 and 256, and 512 requests); the first-token time at 64 requests in the first start (17.8 s) was higher than at 128 to 512 requests in the later starts (9.8 to 14.4 s). sglang took 3.8 s, 15.8 s and 27.5 s with its own client (section 4.5 explains why first-token times should not be compared across clients). At 128 and 256 requests the sglang server answered 2 of 256 and 3 of 512 requests with HTTP 400 (Bad Request); the throughput counts only completed requests, and the server log does not give the reason. All requests completed in the other 22k runs. None of these runs went above its running-request limit.
- **sglang's sliding-window KV pool follows the decode batch.** GPT-OSS alternates full-attention and sliding-window layers, and sglang keeps a separate, smaller KV pool for the sliding-window layers. Its server logs show that pool sized at roughly 390 to 420 tokens per decode-graph slot: per attention group, 13,312 tokens at decode batch 32, 25,600 at 64, 49,664 at 128 and 98,816 at 256 (8 attention groups, default prefill chunks), and 52,736 and 56,832 with 16,384- and 32,768-token chunks at decode batch 128. The full-attention pool was about 1.0 million tokens per group in all of these. The server logs of the five 8-attention-group runs that went up to 1,024 or 2,048 requests each contain "KV cache pool is full. Retract requests" (61 to 93 times per run); we did not check which of the two pools ran out.
- **Long prompts.** The 22k-in / 4k-out latency target from section 2.C (time to first token at most 500 ms) is still not met. The lowest median time to first token for one 22k prompt was 542.5 ms (`sglang-torchtpu`, `TP=8`, compiled). Prefilling the 22k prompt in one 32k chunk did not help (653.7 ms). Data-parallel attention raised it to 2.0 to 3.4 s, because each prompt is prefilled by a smaller group of cores.
- **Disk space on the host.** Kubernetes evicted one sglang pod for low disk space (its container was using 66 GiB), which cut short the DFlash-16 + 32k-chunk run; we did not repeat it. sglang keeps its `torch.compile` (inductor) cache under `/root/.cache/sglang` on the container's disk, and on a similar host that cache had reached 64 GiB. We now clear it between runs.
- **Cold starts.** The first attempt of the 128-requests-per-replica `vllm-torchtpu` run timed out after vLLM's default 600 s engine start limit while it was still reading weights from Cloud Storage (about 130 s per file). We raised the limit to 3,600 s for later runs.
- **JAX on a node from a multi-host node pool.** The first JAX attempt failed with "Error computing device indices for TPU_VISIBLE_CHIPS": GKE lists the node's 4 chips there, and vLLM's data-parallel launcher indexes 8 TensorCores into it. Unsetting `TPU_VISIBLE_CHIPS` fixed it ([`scripts/run_vllm_tpu_jax_high_load.sh`](./scripts/run_vllm_tpu_jax_high_load.sh)).
- **An unexplained low point:** the first point of the 128-requests-per-replica `vllm-torchtpu` run (64 requests: 3,337.2 tok/s, 3.2 s to first token) was lower than the same layout with 64 requests per replica (4,173.6). We did not investigate it.
- **Profiling:** the XProf trace attempted during the `vllm-torchtpu` `TP=8` run did not produce usable data.

### 4.7 Licenses for this section

- [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b): Apache-2.0. The model's usage policy adds: "By using OpenAI gpt-oss-120b, you agree to comply with all applicable law."
- [`z-lab/gpt-oss-120b-DFlash`](https://huggingface.co/z-lab/gpt-oss-120b-DFlash): MIT.
- Both were read before download. Neither has territorial, company-size or field-of-use restrictions. This repo does not redistribute weights.

## 5. TPU v5p vs. v7x, One Host Each (MXFP4)

These runs serve the native MXFP4 checkpoint (`openai/gpt-oss-120b`) on one TPU v5p host and one TPU v7x host, 4 chips each, with the same stacks, server flags and client settings. Raw results are in [`results/gptoss120b_v5p_vs_v7x/`](./results/gptoss120b_v5p_vs_v7x/). The v7x JAX runs with FP4 experts at 512 requests and above and with 22k-token prompts are the section 4 runs (same image and flags).

### 5.1 Hardware and setups

| | TPU v5p host | TPU v7x host |
| :--- | :--- | :--- |
| Chips | 4 | 4 |
| HBM per chip | 95 GiB HBM2e, 2,765 GB/s | 192 GB HBM3e, 7.4 TB/s |
| Peak BF16 compute per chip | 459 TFLOP/s | 2,307 TFLOP/s |
| Devices seen by the runtime | 4 (one per chip) | 8 (one per TensorCore) |

The chip figures are the published ones ([TPU v5p documentation](https://cloud.google.com/tpu/docs/v5p); v7x as in section 1 of the README).

- **Client.** `vllm bench serve` for JAX `vllm-tpu` and `sglang.bench_serving` for `sglang-torchtpu`, as in section 4.2: random prompts of exactly the stated length, end-of-sequence tokens ignored, and a fixed number of requests in flight. 2 prompts per concurrent request, with at least 16 prompts per point for JAX and 32 for sglang (12 for sglang with one request).
- **`sglang-torchtpu`.** The same flags on both chips except the tensor-parallel size, which covers all devices (4 on v5p, 8 on v7x): `torch.compile`, decode graphs up to batch 64, 256 running requests, 26,624-token context limit, memory fraction 0.74, radix cache off. The high-load setups use data-parallel attention over all devices (4 groups on v5p, 8 on v7x) with 128 or 256 running requests per group, decode graphs up to the same batch, and 16,384-token prefill chunks. Scripts: [`scripts/benchmark_sglang_v5p.py`](./scripts/benchmark_sglang_v5p.py) (v5p) and [`scripts/benchmark_sglang_mxfp4_single_host.py`](./scripts/benchmark_sglang_mxfp4_single_host.py) (v7x).
- **JAX `vllm-tpu`.** The same image on both chips (`vllm` 0.33.0rc1, `jax` 0.11.2, `libtpu` 0.0.48) and the section 4 server flags: 16,384 batched tokens per step, FP8 KV cache, async scheduling, chunked prefill, prefix cache off, 32,768-token context limit. The memory fraction was 0.90 on v5p and 0.86 on v7x (the section 4 value). On v5p the expert weights are converted to INT8 when the model loads (`MOE_REQUANTIZE_WEIGHT_DTYPE=int8`, block size 256 with `TP=4` and 32 with `TP=2`) and data parallelism runs in one process (`TPU_MULTIPROCESS_DP=0`). On v7x we ran the FP4 experts of section 4 and, for a same-format comparison, INT8 experts with block size 32. Script: [`scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh`](./scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh).
- **Output check.** Every server in this section answered the sanity prompt "Compute 37 * 43 step by step" with 1591.

### 5.2 Results

1k in / 1k out at 256 to 2,048 requests (output tok/s):

| Chip | Stack and setup | 256 | 512 | 1,024 | 2,048 | 4,096 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, FP4 experts, 512 per replica (section C) | – | 17,190.0 | 22,163.6 | 21,387.7 | – |
| v7x | same, 1,024 per replica (section C) | – | – | – | 22,952.2 | 22,046.9 |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, INT8 experts, 512 per replica | – | 12,733.4 | 17,861.2 | 21,992.2 | – |
| v5p | JAX `vllm-tpu` `TP=4`, INT8 experts, 1,024 running requests | – | 5,576.4 | 6,982.4 | – | – |
| v5p | same, 2,048 running requests | – | – | – | 8,061.2 | – |
| v5p | same, 4,096 running requests, memory fraction 0.85 | – | – | – | – | 6,956.6 |
| v7x | `vllm-torchtpu` + MXFP4 patch, `TP=2, DP=4`, best request limit at each load (section C) | 7,849.5 | 11,426.5 | 14,867.1 | 20,284.0 | 24,357.1 |
| v7x | `sglang-torchtpu`, 8 attention groups, decode batch 128, FP8 KV cache | 10,017.5 | 13,479.4 | 15,748.3 | – | – |
| v7x | same, BF16 KV cache | 9,793.2 | 13,430.8 | 15,280.7 | – | – |
| v7x | same, decode batch 256, FP8 KV cache | – | – | 16,246.4 | 16,046.6 | – |
| v7x | same, decode batch 256, BF16 KV cache | – | – | 15,913.2 | 13,842.7 | – |
| v5p | `sglang-torchtpu`, 4 attention groups, decode batch 128, FP8 KV cache | 4,184.4 | 4,040.7 | 4,433.5 | – | – |
| v5p | same, BF16 KV cache | 2,219.6 | 2,437.6 | 2,357.0 | – | – |
| v5p | same, decode batch 256, BF16 KV cache | – | 1,069.7 | 1,068.8 | 1,037.2 | – |

22k in / 4k out (output tok/s, with median time to first token / median time per output token):

| Chip | Stack and setup | 8 | 16 | 32 | 64 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, FP4 experts, 128 per replica (section C) | 1,079.4 (4.0 s / 6.4 ms) | 1,672.0 (7.0 s / 7.8 ms) | 2,295.6 (12.1 s / 10.9 ms) | 3,120.4 (17.8 s / 16.2 ms) |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, INT8 experts, 128 per replica | 939.8 (4.4 s / 7.4 ms) | 1,396.1 (7.2 s / 9.6 ms) | 1,955.0 (11.2 s / 13.5 ms) | 2,469.2 (16.6 s / 21.2 ms) |
| v5p | JAX `vllm-tpu` `TP=4`, INT8 experts, 256 running requests | 623.3 (6.8 s / 11.1 ms) | 855.3 (9.0 s / 16.6 ms) | 883.4 (6.1 s / 34.7 ms) | 1,066.7 (6.2 s / 58.6 ms) |

Every point with latency:

##### v5p: `sglang_mxfp4_tp4_ep4_compiled_fp8kv_v5p` (`TP=4`, FP8 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | – | 12 | 168.1 | 79 | 5.88 |
| 1k / 1k, 8 | – | 32 | 677.9 | 235 | 11.50 |
| 1k / 1k, 32 | – | 64 | 1,114.1 | 773 | 27.99 |
| 1k / 1k, 64 | – | 128 | 1,602.7 | 1,377 | 38.61 |
| 1k / 1k, 128 | – | 256 | 2,308.9 | 2,638 | 52.89 |
| 1k / 1k, 256 | – | 512 | 3,236.1 | 4,521 | 74.46 |

##### v5p: `sglang_mxfp4_tp4_ep4_compiled_v5p` (`TP=4`, KV cache type left at its default, which the server set to FP8)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | – | 12 | 168.8 | 78 | 5.84 |
| 1k / 1k, 8 | – | 32 | 679.9 | 236 | 11.50 |
| 1k / 1k, 32 | – | 64 | 1,082.5 | 766 | 28.86 |
| 1k / 1k, 64 | – | 128 | 1,594.4 | 1,418 | 38.79 |
| 1k / 1k, 128 | – | 256 | 2,298.5 | 2,635 | 53.08 |
| 1k / 1k, 256 | – | 512 | 3,227.1 | 4,539 | 74.72 |
| 22k / 4k, 1 request | – | 6 | 157.6 | 1,076 | 6.08 |

##### v5p: `sglang_mxfp4_tp4_ep4_compiled_bf16kv_v5p` (`TP=4`, BF16 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | – | 12 | 169.9 | 79 | 5.78 |
| 1k / 1k, 8 | – | 32 | 698.1 | 232 | 11.11 |
| 1k / 1k, 32 | – | 64 | 1,153.7 | 705 | 27.05 |
| 1k / 1k, 64 | – | 128 | 1,712.3 | 1,404 | 36.07 |
| 1k / 1k, 128 | – | 256 | 2,597.1 | 2,565 | 46.78 |
| 1k / 1k, 256 | – | 512 | 3,819.9 | 4,469 | 62.38 |

##### v5p: `sglang_mxfp4_dp4_tp4_ep4_compiled_bs128_chunk16k_v5p` (4 attention groups, decode batch 128, 512 running requests, 16,384-token chunks, FP8 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 256 | – | 512 | 4,184.4 | 3,616 | 57.22 |
| 1k / 1k, 512 | – | 1024 | 4,040.7 | 11,271 | 92.46 |
| 1k / 1k, 1,024 | – | 2048 | 4,433.5 | 100,340 | 86.32 |

##### v5p: `sglang_mxfp4_dp4_tp4_ep4_compiled_bs128_chunk16k_bf16kv_v5p` (same with BF16 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 256 | – | 512 | 2,219.6 | 8,388 | 64.04 |
| 1k / 1k, 512 | – | 1024 | 2,437.6 | 115,862 | 61.04 |
| 1k / 1k, 1,024 | – | 2048 | 2,357.0 | 316,641 | 58.95 |

##### v5p: `sglang_mxfp4_dp4_tp4_ep4_compiled_bs256_chunk16k_bf16kv_v5p` (decode batch 256, 1,024 running requests, BF16 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 512 | – | 1024 | 1,069.7 | 393,575 | 36.83 |
| 1k / 1k, 1,024 | – | 2048 | 1,068.8 | 840,145 | 36.46 |
| 1k / 1k, 2,048 | – | 4096 | 1,037.2 | 1,737,903 | 37.35 |

##### v7x: `sglang_mxfp4_tp8_ep8_compiled_bs64_r256` (`TP=8`, FP8 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | – | 12 | 186.2 | 50 | 5.26 |
| 1k / 1k, 8 | – | 32 | 1,125.7 | 142 | 6.96 |
| 1k / 1k, 32 | – | 64 | 2,557.9 | 433 | 12.12 |
| 1k / 1k, 64 | – | 128 | 3,683.9 | 739 | 16.62 |
| 1k / 1k, 128 | – | 256 | 5,046.1 | 1,373 | 24.02 |
| 1k / 1k, 256 | – | 512 | 6,657.7 | 2,472 | 36.03 |
| 22k / 4k, 1 request | – | 6 | 180.5 | 553 | 5.42 |

##### v7x: `sglang_mxfp4_tp8_ep8_compiled_bs64_r256_kv_auto` (`TP=8`, `--kv-cache-dtype auto`, which the server set to FP8)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | – | 12 | 182.2 | 50 | 5.43 |
| 1k / 1k, 8 | – | 32 | 1,128.6 | 143 | 6.93 |
| 1k / 1k, 32 | – | 64 | 2,564.7 | 434 | 12.08 |
| 1k / 1k, 64 | – | 128 | 3,668.2 | 742 | 16.72 |
| 1k / 1k, 128 | – | 256 | 5,057.2 | 1,385 | 23.95 |
| 1k / 1k, 256 | – | 512 | 6,665.7 | 2,386 | 36.00 |
| 22k / 4k, 1 request | – | 6 | 187.5 | 557 | 5.19 |

##### v7x: `sglang_mxfp4_tp8_ep8_compiled_bs64_r256_bf16kv` (`TP=8`, BF16 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1 request | – | 12 | 189.0 | 49 | 5.23 |
| 1k / 1k, 8 | – | 32 | 1,121.5 | 145 | 6.96 |
| 1k / 1k, 32 | – | 64 | 2,559.6 | 433 | 12.11 |
| 1k / 1k, 64 | – | 128 | 3,611.4 | 738 | 16.99 |
| 1k / 1k, 128 | – | 256 | 5,027.2 | 1,408 | 24.09 |
| 1k / 1k, 256 | – | 512 | 6,647.8 | 2,467 | 36.08 |

##### v7x: `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_c256` (8 attention groups, decode batch 128, 1,024 running requests, 16,384-token chunks, FP8 KV cache; the 512 and 1,024 points of this setup are in section 4.4)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 256 | – | 512 | 10,017.5 | 2,143 | 23.45 |

##### v7x: `sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_bf16kv` (same with BF16 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 256 | – | 512 | 9,793.2 | 2,110 | 24.07 |
| 1k / 1k, 512 | – | 1024 | 13,430.8 | 3,514 | 34.60 |
| 1k / 1k, 1,024 | – | 2048 | 15,280.7 | 7,340 | 59.33 |

##### v7x: `sglang_mxfp4_dp8_tp8_ep8_compiled_bs256_chunk16k_bf16kv` (decode batch 256, 2,048 running requests, BF16 KV cache)

| Input / output, concurrent requests | Prompts | Completed | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1k / 1k, 1,024 | – | 2048 | 15,913.2 | 5,830 | 58.45 |
| 1k / 1k, 2,048 | – | 4096 | 13,842.7 | 14,683 | 123.16 |

##### v5p: JAX `vllm-tpu` `TP=4`, 256 running requests, INT8 experts (block 256), FP8 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 141.2 | 75 | 7.01 |
| 8 | 16 | 828.4 | 353 | 9.29 |
| 32 | 64 | 1,370.7 | 936 | 22.50 |
| 64 | 128 | 2,000.2 | 1,438 | 30.74 |
| 128 | 256 | 2,846.5 | 2,329 | 42.78 |
| 256 | 512 | 4,088.6 | 1,799 | 60.22 |

##### v5p: same with BF16 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 142.1 | 74 | 7.00 |
| 8 | 16 | 812.8 | 353 | 9.53 |
| 32 | 64 | 1,396.7 | 1,003 | 22.14 |
| 64 | 128 | 1,975.9 | 1,435 | 30.86 |
| 128 | 256 | 2,803.5 | 1,766 | 43.94 |
| 256 | 512 | 4,000.2 | 2,391 | 61.96 |

##### v5p: JAX `vllm-tpu` `TP=2, DP=2`, 128 per replica, INT8 experts (block 32), FP8 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 66.8 | 124 | 14.22 |
| 8 | 16 | 356.5 | 414 | 15.02 |
| 32 | 64 | 849.3 | 1,446 | 28.33 |
| 64 | 128 | 1,315.6 | 2,076 | 42.72 |
| 128 | 256 | 1,857.7 | 2,665 | 63.39 |
| 256 | 512 | 2,909.2 | 3,237 | 80.85 |

##### v5p: JAX `vllm-tpu` `TP=4`, 1,024 running requests, INT8 experts (block 256), FP8 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1,024 | 5,576.4 | 2,444 | 88.69 |
| 1,024 | 2,048 | 6,982.4 | 3,175 | 141.40 |

##### v5p: JAX `vllm-tpu` `TP=4`, 2,048 running requests, INT8 experts (block 256), FP8 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4,096 | 8,061.2 | 7,428 | 239.78 |

##### v5p: JAX `vllm-tpu` `TP=4`, 4,096 running requests, INT8 experts (block 256), FP8 KV cache, memory fraction 0.85

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 8,192 | 6,956.6 | 302,388 | 245.59 |

##### v5p: JAX `vllm-tpu` `TP=4`, 256 running requests, INT8 experts, second run

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 144.1 | 75 | 6.88 |
| 8 | 16 | 832.0 | 451 | 9.25 |
| 32 | 64 | 1,343.4 | 1,039 | 22.81 |
| 64 | 128 | 2,050.9 | 1,747 | 29.91 |
| 128 | 256 | 2,859.2 | 1,762 | 43.76 |
| 256 | 512 | 4,076.0 | 1,843 | 60.38 |

##### v7x: JAX `vllm-tpu` `TP=2, DP=4`, 128 per replica, INT8 experts, second run

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 163.4 | 59 | 6.06 |
| 8 | 16 | 1,134.3 | 123 | 6.90 |
| 32 | 64 | 2,903.8 | 375 | 10.64 |
| 64 | 128 | 4,554.9 | 673 | 13.22 |
| 128 | 256 | 5,688.4 | 1,049 | 21.31 |
| 256 | 512 | 8,688.1 | 1,741 | 27.33 |

##### v5p: JAX `vllm-tpu` `TP=4`, 256 running requests, INT8 experts (block 256), FP8 KV cache, 22k-token prompts

| Concurrent requests (22k in / 4k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 623.3 | 6,776 | 11.14 |
| 16 | 32 | 855.3 | 8,980 | 16.59 |
| 32 | 64 | 883.4 | 6,113 | 34.70 |
| 64 | 128 | 1,066.7 | 6,226 | 58.62 |

##### v7x: JAX `vllm-tpu` `TP=2, DP=4`, 128 per replica, FP4 experts (as in section C), FP8 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 183.2 | 50 | 5.40 |
| 8 | 16 | 1,335.6 | 82 | 5.91 |
| 32 | 64 | 3,890.3 | 312 | 7.91 |
| 64 | 128 | 5,443.0 | 482 | 11.26 |
| 128 | 256 | 7,740.3 | 764 | 15.89 |
| 256 | 512 | 11,083.9 | 952 | 21.66 |

##### v7x: JAX `vllm-tpu` `TP=2, DP=4`, 128 per replica, INT8 experts (block 32), FP8 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 163.3 | 59 | 6.07 |
| 8 | 16 | 1,137.4 | 122 | 6.88 |
| 32 | 64 | 2,903.5 | 379 | 10.64 |
| 64 | 128 | 4,502.8 | 671 | 13.38 |
| 128 | 256 | 5,748.0 | 919 | 21.29 |
| 256 | 512 | 9,180.7 | 1,533 | 26.51 |

##### v7x: JAX `vllm-tpu` `TP=2, DP=4`, 512 per replica, INT8 experts (block 32), FP8 KV cache

| Concurrent requests (1k in / 1k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1,024 | 12,733.4 | 2,997 | 36.84 |
| 1,024 | 2,048 | 17,861.2 | 5,389 | 51.94 |
| 2,048 | 4,096 | 21,992.2 | 7,655 | 85.10 |

##### v7x: JAX `vllm-tpu` `TP=2, DP=4`, 128 per replica, INT8 experts (block 32), FP8 KV cache, 22k-token prompts

| Concurrent requests (22k in / 4k out) | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 939.8 | 4,395 | 7.40 |
| 16 | 32 | 1,396.1 | 7,242 | 9.63 |
| 32 | 64 | 1,955.0 | 11,236 | 13.51 |
| 64 | 128 | 2,469.2 | 16,635 | 21.21 |

### 5.3 Second runs

Two setups ran twice on each chip, each time with a fresh server: the FP8-KV `sglang-torchtpu` setup and JAX `vllm-tpu` with INT8 experts. For sglang, the second run left the KV cache type at its default (`auto` on v7x, unset on v5p), which the server resolved to FP8 e4m3 (server log), so both runs used the same configuration.

| Chip | Setup | Run | 1 request | 8 | 32 | 64 | 128 | 256 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| v7x | `sglang-torchtpu` `TP=8`, FP8 KV cache | first | 186.2 | 1,125.7 | 2,557.9 | 3,683.9 | 5,046.1 | 6,657.7 |
| v7x | `sglang-torchtpu` `TP=8`, FP8 KV cache | second | 182.2 | 1,128.6 | 2,564.7 | 3,668.2 | 5,057.2 | 6,665.7 |
| v5p | `sglang-torchtpu` `TP=4`, FP8 KV cache | first | 168.1 | 677.9 | 1,114.1 | 1,602.7 | 2,308.9 | 3,236.1 |
| v5p | `sglang-torchtpu` `TP=4`, FP8 KV cache | second | 168.8 | 679.9 | 1,082.5 | 1,594.4 | 2,298.5 | 3,227.1 |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, INT8 experts | first | 163.3 | 1,137.4 | 2,903.5 | 4,502.8 | 5,748.0 | 9,180.7 |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, INT8 experts | second | 163.4 | 1,134.3 | 2,903.8 | 4,554.9 | 5,688.4 | 8,688.1 |
| v5p | JAX `vllm-tpu` `TP=4`, INT8 experts | first | 141.2 | 828.4 | 1,370.7 | 2,000.2 | 2,846.5 | 4,088.6 |
| v5p | JAX `vllm-tpu` `TP=4`, INT8 experts | second | 144.1 | 832.0 | 1,343.4 | 2,050.9 | 2,859.2 | 4,076.0 |

The sglang runs differed by at most 2.2% on v7x and 2.8% on v5p. The JAX runs differed by at most 2.5% on v5p; on v7x they were within 1.2% up to 128 requests, and at 256 requests the second run was 5.4% lower (8,688.1 vs. 9,180.7 tok/s). Most other points in this section are single runs.

### 5.4 What worked and what did not

- **`sglang-torchtpu` ran on v5p with the v7x flags.** Only the tensor-parallel size changed (4 instead of 8). Startup took 2,173 s on v5p and 2,137 s on v7x for the same setup.
- **JAX `vllm-tpu` needed three changes on v5p.**
  1. The MoE kernel that multiplies BF16 activations by FP4 expert weights does not compile for v5p ("Bad lhs/rhs type: 'vector<32x128xbf16>' 'vector<128x128xf4E2M1FN>'"). Converting the experts to INT8 when the model loads (`MOE_REQUANTIZE_WEIGHT_DTYPE=int8`) fixed it, and the outputs were correct.
  2. With `TP=4`, the default conversion block of 512 values leaves 6 scale blocks along the 2,880-wide expert dimension (padded to 3,072), which cannot be split 4 ways ("IndivisibleError ... the dimension size is 6"). A block of 256 gives 12 blocks and works with `TP=4`; a block of 32 gives 90 and works with `TP=1` or `TP=2`.
  3. The multi-process data-parallel launcher in this build counts two cores per chip on v5p, so it gave each replica the wrong chips and the TPU runtime failed to start ("Failed to create the mesh (1, 2, 1)"). Single-process data parallelism (`TPU_MULTIPROCESS_DP=0`) works.
- **One replica per chip does not fit on v5p.** With FP4 experts, `TP=1, DP=4` failed with the kernel error above. With INT8 experts the expert weights alone take about 115 GB (128 experts × 36 layers × 3 × 2,880 × 2,880 bytes), more than a v5p chip's 95 GiB, so we did not try it.
- **On v5p, `TP=4` beat `TP=2, DP=2` at every load.** One request: 141.2 vs. 66.8 tok/s (7.01 vs. 14.22 ms per output token); 256 requests: 4,088.6 vs. 2,909.2. The two runs also differ in the INT8 block size (256 vs. 32).
- **KV cache type.** With `sglang-torchtpu` on v5p, a BF16 KV cache gave more throughput than FP8 as load grew: about the same for one request (169.9 vs. 168.1 tok/s), 3.6% more at 32 requests, 12.5% more at 128 and 18.0% more at 256 (3,819.9 vs. 3,236.1). On v7x the two were within 2.0% at every load from 1 to 256. With JAX on v5p the KV type made little difference (4,000.2 tok/s with BF16 vs. 4,088.6 with FP8 at 256 requests). With data-parallel attention, BF16 was much slower on v5p: 2,219.6 vs. 4,184.4 tok/s at 256 requests with decode batch 128, and 1,037.2 to 1,069.7 tok/s at 512 to 2,048 requests with decode batch 256. In the server logs, each of the 4 attention groups had room for 193,024 cached tokens with FP8 and decode batch 128, 67,840 with BF16 and decode batch 128, and 18,816 with BF16 and decode batch 256. On v7x, BF16 with data-parallel attention was within 3.0% of FP8 from 256 to 1,024 requests, and 13.7% slower at 2,048 requests with decode batch 256 (13,842.7 vs. 16,046.6 tok/s).
- **The default KV setting is FP8 here.** For this checkpoint `sglang-torchtpu` resolved `--kv-cache-dtype auto` (or no setting) to FP8 e4m3 on both chips (server log), so those runs repeat the FP8 setup; section 5.3 uses them as second runs.
- **The v7x advantage grew with load.** For one request the v7x host gave 1.11× the v5p host's throughput with sglang (5.26 vs. 5.88 ms per output token) and 1.16× with JAX and INT8 experts (6.07 vs. 7.01 ms). From 32 to 256 requests it gave 1.74× to 2.30× with the same setup on both, and from 512 to 4,096 requests the best setups on each host were 2.85× to 3.50× apart. One difference shows in the server logs: after loading the weights, the sglang server had room for 4,597,504 cached tokens on v7x (`TP=8`) and 1,239,104 on v5p (`TP=4`), and with data-parallel attention for about 8.2 million vs. 0.77 million. We did not profile these runs, so we cannot say which part of each chip limits a given point.
- **INT8 experts were slower than FP4 on v7x.** `TP=2, DP=4` with 128 requests per replica: 163.3 vs. 183.2 tok/s for one request (6.07 vs. 5.40 ms per output token) and 9,180.7 vs. 11,083.9 at 256 requests (17% less). With 512 requests per replica the gap narrowed as load grew: 12,733.4 vs. 17,190.0 tok/s at 512 requests (26% less), 17,861.2 vs. 22,163.6 at 1,024 (19% less) and 21,992.2 vs. 21,387.7 at 2,048 (3% more).
- **High load on v5p.** JAX with INT8 experts and `TP=4` gave 5,576.4 tok/s at 512 requests and 6,982.4 at 1,024 with a limit of 1,024 running requests, and 8,061.2 at 2,048 with a limit of 2,048. A limit of 4,096 failed to start at memory fraction 0.90: loading a compiled program needed 4.61 GB with 4.02 GB free (`RESOURCE_EXHAUSTED`). At 0.85 it started and gave 6,956.6 tok/s at 4,096 requests, with a median wait of 302 s for the first token. `sglang-torchtpu` with data-parallel attention over 4 groups (decode batch 128, FP8 KV cache) gave 4,184.4, 4,040.7 and 4,433.5 tok/s at 256, 512 and 1,024 requests. At 1,024 its median wait for the first token was 100 s, because this setup runs at most 512 requests at a time.
- **Long prompts.** One 22k-token prompt with `sglang-torchtpu`: 553 ms to the first token on v7x and 1,076 ms on v5p, and 5.42 vs. 6.08 ms per output token. With JAX at 8 to 64 requests (22k in / 4k out), v7x with INT8 experts gave 1.51× to 2.31× the v5p throughput, and 1.73× to 2.93× with FP4 experts. On v5p the time per output token rose from 11.1 ms at 8 requests to 58.6 ms at 64.
- **Startup.** JAX `vllm-tpu` took 311 to 640 s to start on v5p (INT8 conversion included) and 220 to 610 s on v7x. `sglang-torchtpu` took 1,961 to 2,237 s on v5p and 1,669 to 2,137 s on v7x. Startup is not part of any number.

### 5.5 License

Same checkpoint as sections B, C and 4: [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b), Apache-2.0 (see 4.7). It was read before download and has no territorial, company-size or field-of-use restrictions.

---

## 6. Additional `vllm-torchtpu` `TP=8` `22k/4k` Results (4-Host 16-Chip MXFP4, BF16 KV Cache, and BF16 Weights)

These runs use patched `vllm-torchtpu` (`TP=8` per 4-chip host, `--enable-expert-parallel`, `--no-enable-prefix-caching`) on `22,000`-token input / `4,000`-token output prompts (`vllm bench serve`, random tokens, `--ignore-eos`).

### 6.1 4-Host (16-Chip) `vllm-torchtpu` MXFP4 `TP=8` with FP8 KV Cache

Four independent 4-chip hosts behind [`scripts/lb_proxy.py`](./scripts/lb_proxy.py) (`least_conn`). Raw outputs: [`results/gptoss120b_4h_16c_torchtpu_mxfp4_fp8kv_tp8/`](./results/gptoss120b_4h_16c_torchtpu_mxfp4_fp8kv_tp8/).

| Workload (`22k/4k`) | Output tok/s | tok/s per chip | TTFT P50 / P95 (ms) | TPOT P50 / P95 (ms) | E2E P50 / P95 (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| CON 1 (10 prompts) | 201.7 | 12.6 | 721 / 734 | 4.77 / 4.80 | 19.8 / 19.9 |
| CON 6 (48 prompts) | 1,050.9 | 65.7 | 724 / 1,173 | 5.40 / 5.62 | 22.4 / 23.2 |
| CON 16 (64 prompts) | 2,182.0 | 136.4 | 1,760 / 2,542 | 6.88 / 7.18 | 29.3 / 29.5 |
| CON 32 (80 prompts) | 2,613.9 | 163.4 | 3,123 / 4,923 | 9.43 / 10.04 | 40.9 / 41.2 |
| RPS 1 (48 prompts) | 2,343.3 | 146.5 | 731 / 3,439 | 10.08 / 11.11 | 42.1 / 45.1 |
| RPS 2 (56 prompts) | 3,196.7 | 199.8 | 735 / 790 | 12.42 / 12.98 | 50.4 / 52.6 |
| RPS 6 (60 prompts) | 3,842.6 | 240.2 | 1,249 / 1,857 | 13.93 / 14.90 | 57.6 / 60.5 |

### 6.2 One Host (4 Chips): `vllm-torchtpu` `TP=8` on `22k/4k` — MXFP4 (FP8 KV vs. BF16 KV) vs. BF16 Weights + BF16 KV

Raw outputs: [`results/gptoss120b_1h_4c_mxfp4_followup/vllm_torchtpu_results.json`](./results/gptoss120b_1h_4c_mxfp4_followup/vllm_torchtpu_results.json) and [`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_lowconc/`](./results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_lowconc/).

| Concurrent reqs (1 host) | MXFP4 + FP8 KV (out tok/s, TTFT / TPOT P50) | MXFP4 + BF16 KV (out tok/s, TTFT / TPOT P50) | BF16 weights + BF16 KV (out tok/s, TTFT / TPOT P50) |
| ---: | :--- | :--- | :--- |
| 1 | 201.9 (708 ms / 4.77 ms) | 200.3 (702 ms / 4.82 ms) | 162.1 (874 ms / 5.95 ms) |
| 2 | 346.5 (1,015 ms / 5.52 ms) | 343.3 (1,012 ms / 5.57 ms) | 264.9 (1,265 ms / 7.22 ms) |
| 4 | 547.4 (1,742 ms / 6.87 ms) | 541.5 (1,743 ms / 6.94 ms) | 399.2 (2,173 ms / 9.47 ms) |
| 6 | 685.2 (2,435 ms / 8.13 ms) | 674.9 (2,431 ms / 8.29 ms) | 481.4 (3,030 ms / 11.72 ms) |
| 16 | 1,005.9 (3,707 ms / 14.80 ms) | 990.7 (3,651 ms / 15.14 ms) | 672.3 (3,644 ms / 22.76 ms) |

- **BF16 KV cache overhead:** switching KV cache from FP8 to BF16 with MXFP4 weights changed throughput by at most 1.5% and TPOT by at most 2.3% across `CON=1..16`.
- **MXFP4 vs. BF16 weights:** at `CON=4` per host (matching `CON=16` across 4 hosts), MXFP4 + FP8 KV reached `547.4 tok/s` (`6.87 ms` TPOT) vs. `399.2 tok/s` (`9.47 ms` TPOT) for BF16 weights + BF16 KV (+37.1% throughput). `lmsys/gpt-oss-120b-bf16` is `119 GB` on disk (experts stored in FP8 E5M2 and upcast to `~234 GB` BF16 in memory) and is derived from OpenAI's native MXFP4 release.
