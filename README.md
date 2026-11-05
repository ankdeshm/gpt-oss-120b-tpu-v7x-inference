# Serving `GPT-OSS 120B` on Google Cloud Ironwood TPU (v7x)

Production-ready inference recipes, **GKE + DWS Calendar Mode reservation** provisioning scripts and manifests, TorchTPU (`sglang-torchtpu` & `vllm-torchtpu`) and `vllm-tpu` (JAX) runtime configurations, and full-spectrum benchmark telemetry for serving **`GPT-OSS 120B`** across both **Native Unquantized BF16** (`lmsys/gpt-oss-120b-bf16`) and **Native MXFP4** (`openai/gpt-oss-120b`) checkpoints on **Google Cloud Ironwood TPU (`tpu7x`)**.

> **First-Time GCP User or Activating a DWS Calendar Reservation?**
> Start with **[`INFRA_SETUP.md`](INFRA_SETUP.md)** and **[`k8s/provision_gke_dws_calendar.sh`](k8s/provision_gke_dws_calendar.sh)** for a step-by-step guide to setting up your VPC, Cloud NAT, GCS bucket, Regional Workload Policy (`2x2x1`, `2x2x2`, or `2x2x4`), and GKE Node Pool bound to your **DWS Calendar Mode reservation (`--reservation-affinity=specific --reservation=<RESERVATION_NAME>`)**.

---

## Hardware & Architecture Overview

| Attribute | Ironwood TPU (`tpu7x`) Specification |
|---|---|
| **Single Host (`tpu7x-standard-4t`)** | `4` physical Ironwood chips (`2x2x1` ICI mesh) = `8` TensorCores + SparseCores |
| **HBM Capacity & Bandwidth** | `192 GB` HBM3e per chip (`768 GB` HBM3e per 4-chip host) @ `7.4 TB/s` per chip (`29.6 TB/s` per host) |
| **Compute per Chip (BF16 / FP8)** | `2,307 TFLOPs` BF16 / `4,614 TFLOPs` FP8 per chip (`9.23 PFLOPs` BF16 / `18.46 PFLOPs` FP8 per 4-chip host) |
| **2-Host Deployment (`8` Chips)** | `2 × tpu7x-standard-4t` (`2x2x2` WorkloadPolicy, `16` TensorCores, `1.54 TB` HBM3e) |
| **4-Host Deployment (`16` Chips)** | `4 × tpu7x-standard-4t` (`2x2x4` WorkloadPolicy, `32` TensorCores, `3.07 TB` HBM3e) |
| **Model Architecture (`GPT-OSS 120B`)** | Sparse MoE: `36` layers, `117B` total parameters, **`5.1B` active parameters per token** (`top-4` of `128` routed experts/layer), `hidden_size = 2880`, `intermediate_size = 2880` |
| **BF16 Checkpoint** | [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (`~233.7 GB`; BF16 conversion of the `openai/gpt-oss-120b` MXFP4 checkpoint with the MoE experts dequantized to BF16 — runs on **all 3 stacks**: `sglang-torchtpu`, `vllm-torchtpu`, `vllm-tpu`) |
| **MXFP4 Model Checkpoint** | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`~65.3 GB` native MXFP4 MoE + BF16 attention — runs on `sglang-torchtpu` & `vllm-tpu`) |
| **Draft Checkpoint** | [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) (`EAGLE3` speculative decoding) |

---

## Performance Highlights

All benchmarks use `vllm bench serve` with uncached random token IDs (`--ignore-eos`, zero prefix-cache reuse) and track strict per-request SLA Goodput (`TTFT <= 500 ms`, `TPOT <= 10 ms`, `E2EL <= 60 s`). BF16 weights with a BF16 KV cache are the primary configuration; rows that use an FP8 KV cache or MXFP4 experts are optional (non-BF16) and are labeled as such.

### 1. BF16 Weights (`lmsys/gpt-oss-120b-bf16`, `234 GB`) Across 1 & 2 Hosts

`lmsys/gpt-oss-120b-bf16` is a BF16 conversion of the `openai/gpt-oss-120b` MXFP4 checkpoint: the MoE expert weights are dequantized from MXFP4 to BF16, and the attention, router and embedding weights are BF16 in both checkpoints. Each peak cell shows the concurrency (`CON`) and the number of prompts (`n`) of the run that produced it.

| Scale | Serving Stack | Layout · KV Cache | `22k / 4k` `CON=1` TTFT P50 | `22k / 4k` `CON=1` TPOT P50 | `n` | `22k / 4k` Closed-Loop Peak Out tok/s | `1k / 1k` `CON=1` TPOT P50 | `1k / 1k` Peak Out tok/s | `8k / 1k` Peak Out tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1 Host (4 Chips)** | `sglang-torchtpu` (load-time BF16 patch) | `TP=8, EP=8` · BF16 KV | `739.5 ms` | `4.97 ms` | `12` | `1,150.1` (`CON=32`, `n=48`) | `4.85 ms` (`n=24`) | `2,530.8` (`CON=32`, `n=96`) | `1,641.9` (`CON=32`, `n=64`) |
| **1 Host (4 Chips)** | `sglang-torchtpu` + `EAGLE3` (load-time BF16 patch) | `TP=8, EP=8` · BF16 KV | `762.8 ms` | `5.48 ms` | `12` | `934.1` (`CON=32`, `n=48`) | `5.32 ms` (`n=24`) | `2,059.0` (`CON=32`, `n=96`) | `1,333.7` (`CON=32`, `n=64`) |
| **1 Host (4 Chips)** | `vllm-torchtpu` (load-time BF16 patch) | `TP=8, DP=1, EP=true` · BF16 KV | `878.7 ms` | `5.95 ms` | `12` | `733.6` (`CON=32`, `n=48`) | `5.30 ms` (`n=24`) | `2,243.0` (`CON=32`, `n=96`) | `1,223.2` (`CON=32`, `n=64`) |
| **1 Host (4 Chips)** | `vllm-tpu` (JAX, no patch) | `TP=8, DP=1, EP=true` · BF16 KV | `880.5 ms` | `8.54 ms` | `12` | `1,249.9` (`CON=32`, `n=48`) | `8.21 ms` (`n=24`) | `2,463.7` (`CON=32`, `n=96`) | `1,557.4` (`CON=32`, `n=64`) |
| **2 Hosts (8 Chips)** | `2x sglang-torchtpu` + `least_conn` LB | `2 x (TP=8, EP=8)` · BF16 KV | `771.4 ms` | `5.05 ms` | `10` | `2,433.8` (`CON=32`, `n=64`) | not run¹ | `5,084.5` (`CON=64`, `n=192`) | `3,281.0` (`CON=128`, `n=256`) |
| **2 Hosts (8 Chips)** | `2x vllm-tpu` (JAX) + `least_conn` LB | `2 x (TP=8, DP=1, EP=true)` · BF16 KV | `882.6 ms` | `5.99 ms` | `10` | `1,460.5` (`CON=64`, `n=96`) | not run¹ | `5,567.6` (`CON=128`, `n=256`) | `2,814.4` (`CON=128`, `n=256`) |
| **1 Host (4 Chips)** | `vllm-torchtpu` (load-time BF16 patch) | `TP=8, DP=1, EP=true` · FP8 KV — optional (non-BF16) | `877.1 ms` | `5.96 ms` | `12` | `739.8` (`CON=32`, `n=48`) | `5.30 ms` (`n=24`) | `2,284.8` (`CON=32`, `n=96`) | `1,243.1` (`CON=32`, `n=64`) |
| **1 Host (4 Chips)** | `vllm-tpu` (JAX, no patch) | `TP=8, DP=1, EP=true` · FP8 KV — optional (non-BF16) | `880.3 ms` | `8.48 ms` | `12` | `1,302.6` (`CON=32`, `n=48`) | `8.26 ms` (`n=24`) | `2,540.3` (`CON=32`, `n=96`) | `1,582.9` (`CON=32`, `n=64`) |

¹ The 2-host runs have no `1k / 1k` `CON=1` cell. Their lowest-concurrency `1k / 1k` cell is `CON=8` (`n=64` each): TPOT P50 `5.77 ms` (`sglang-torchtpu`) and `6.63 ms` (`vllm-tpu`).

**Sample size (`n`)**: the customer workload specifies `n=50` prompts per cell. The 1-host `22k / 4k` cells used `n=12 / 30 / 36 / 48` at `CON=1 / 6 / 16 / 32` and `n=28–40` for open-loop RPS; the 2-host `22k / 4k` cells used `n=10–96`.

**2-host vs 1-host at the same per-replica concurrency** (derived; 1 host at `CON=32` vs 2 hosts at `CON=64`, i.e. 32 in-flight requests per replica): `sglang-torchtpu` `2.01x` (`1k / 1k`), `1.97x` (`8k / 1k`), `1.94x` (`22k / 4k`); `vllm-tpu` `1.80x`, `1.57x`, `1.17x`. Source cells: [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) §4.3.

### 2. Optional (non-BF16): MXFP4 Experts (`openai/gpt-oss-120b`, `65.3 GB`) with FP8 KV, 1 / 2 / 4 Hosts

| Scale | Chips (Cores) | Serving Engine · Layout · KV Cache | `1k / 1k` Peak Out tok/s | `1k / 1k` Highest SLA Goodput | `8k / 1k` Peak Out tok/s (Total tok/s) | `22k / 4k` Closed-Loop Peak Out tok/s | `22k / 4k` @ `6 RPS` P50 TPOT / E2E |
|---|---:|---|---:|---:|---:|---:|---:|
| **1 Host** | `4` (`8`) | `vllm-tpu` (JAX) `TP=2, DP=4` · FP8 KV | `5,847.0` (`CON=64`, `n=128`) | `3.340 req/s` (`CON=32`, `n=128`) | `2,696.6` (`24,269.5`) (`CON=64`, `n=96`) | `1,789.4` (`CON=32`, `n=50`) | `14.76 ms` / `71.17 s` (`n=50`) |
| **1 Host** | `4` (`8`) | `sglang-torchtpu` `TP=8, EP=8` · FP8 KV | `2,587.5` (`CON=32`, `n=128`) | `1.104 req/s` (`CON=8`, `n=64`) | `1,630.1` (`14,670.9`) (`CON=32`, `n=64`) | `1,242.2` (`CON=16`, `n=48`) | not run (open-loop runs at `RPS=0.5–2.0` only) |
| **2 Hosts** | `8` (`16`) | `2x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB · FP8 KV | `14,277.1` (`CON=256`, `n=512`) | `5.054 req/s` (`CON=64`, `n=192`) | `6,533.7` (`58,803.6`) (`CON=128`, `n=256`) | `3,315.7` (`CON=64`, `n=96`) | `11.27 ms` / `50.51 s` (`n=60`) |
| **4 Hosts** | `16` (`32`) | `4x vllm-tpu` (`TP=2, DP=4`) + `least_conn` LB · FP8 KV | `20,366.9` (`CON=256`, `n=512`) | `8.584 req/s` (`CON=128`, `n=256`) | `9,820.8` (`88,387.3`) (`CON=128`, `n=256`) | `8,749.7` (`CON=128`, `n=128`) | `8.82 ms` (`P95 9.96 ms`) / `37.90 s` (`n=60`) |

**Sample size (`n`)**: `22k / 4k` cells used `n=16–50` at 1 host and `n=10–128` at 2 and 4 hosts (customer workload: `n=50`).

See **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** for complete 1-host, 2-host, and 4-host telemetry across all suites.

---

## Technical Deep Dive: How `GPT-OSS 120B` Works on `sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu` (MXFP4 vs. Native BF16 & Why Load-Time Patches Were Needed)

When evaluating **`GPT-OSS 120B`** on Ironwood TPU (`v7x`), customers typically ask three key architectural questions:
1. **Why does `openai/gpt-oss-120b` (Native MXFP4) work on `sglang-torchtpu` and `vllm-tpu`, but not on `vllm-torchtpu`?**
2. **Why does `lmsys/gpt-oss-120b-bf16` (`234 GB` Pure Unquantized BF16) perform nearly as fast as MXFP4 on a single Ironwood host?**
3. **Why did `lmsys/gpt-oss-120b-bf16` require a lightweight load-time patch on `vllm-torchtpu` and `sglang-torchtpu` (while `vllm-tpu` needed zero patches)?**

### Serving Stack & Checkpoint Compatibility Matrix on Ironwood TPU (`v7x`)

| Serving Stack | `openai/gpt-oss-120b` (`65.3 GB` Native MXFP4) | `lmsys/gpt-oss-120b-bf16` (`233.7 GB` Pure Unquantized BF16) | Best Use Case on Ironwood TPU (`v7x`) |
|---|---|---|---|
| **`sglang-torchtpu`** *(PyTorch + Pallas)* | **Supported Out-of-the-Box** (`TP=8, EP=8`; VPU unpacks 4-bit MXFP4 -> BF16/FP8 MXU)<br>**`737.0 ms` TTFT**, **`4.09 ms` TPOT** (`22k/4k`, `CON=1`; FP8 KV — optional, non-BF16) | **Supported via Load-Time Patch** ([`apply_sglang_bf16_patches.py`](scripts/patches/apply_sglang_bf16_patches.py))<br>**`739.5 ms` TTFT**, **`4.97 ms` TPOT** (`22k/4k`, `CON=1`, 1 host), **`5,084.5` tok/s** (`1k/1k`, 2 hosts, `CON=64`); BF16 KV | **#1 Lowest Latency (`TTFT` & `TPOT`)** and **#1 Long-Context (`22k/4k`) Throughput on BF16** within the PyTorch/TorchTPU ecosystem |
| **`vllm-torchtpu`** *(PyTorch / XLA + Pallas)* | **Not Supported Upstream Yet**<br>(`mxfp4.py` only binds GPU/ROCm Triton kernels; raises `ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu`) | **Supported via Load-Time Patch** ([`apply_vllm_torchtpu_bf16_patches.py`](scripts/patches/apply_vllm_torchtpu_bf16_patches.py))<br>**`878.7 ms` TTFT**, **`5.95 ms` TPOT** (`22k/4k`, `CON=1`), **`2,243.0` tok/s** (`1k/1k`, 1 host, `CON=32`); BF16 KV | **Standard vLLM PyTorch/TorchTPU Stack** for Pure BF16 serving (the 128-tile alignment patch is a correctness fix: without it the output is invalid, only `!` tokens) |
| **`vllm-tpu`** *(JAX `tpu-inference`)* | **Supported Out-of-the-Box** (`TP=2, DP=4` per host; 4 data-parallel replicas/host)<br>**`5,847.0` tok/s** (1h) -> **`20,366.9` tok/s** (4h) (`1k/1k`; FP8 KV — optional, non-BF16) | **Supported Out-of-the-Box (Zero Patches)** (`TP=8, --enable-expert-parallel`)<br>**`880.5 ms` TTFT**, **`8.54 ms` TPOT** (`22k/4k`, `CON=1`, 1 host; `5.99 ms` TPOT on the 2-host run), **`5,567.6` tok/s** (`1k/1k`, 2 hosts, `CON=128`); BF16 KV | **#1 Highest Multi-Host `1k/1k` Throughput** (`20,366.9` tok/s on MXFP4 + FP8 KV, 4 hosts; `5,567.6` tok/s on BF16, 2 hosts) |

---

### Question 1: Why does `openai/gpt-oss-120b` (Native MXFP4) work on `sglang-torchtpu` and `vllm-tpu`, but not on `vllm-torchtpu`?

1. **What `gpt_oss_mxfp4` Is**:
   - OpenAI released `openai/gpt-oss-120b` (`~65.3 GB`) with its `128` routed MoE experts per layer quantized in **`gpt_oss_mxfp4`** — a 4-bit microscaling format where every block of `32` weights is stored as packed 4-bit `E2M1` nibbles (`gate_up_proj_blocks`, `down_proj_blocks`) sharing an 8-bit `E8M0` power-of-two exponent scale (`gate_up_proj_scales`, `down_proj_scales`).
   - OpenAI designed this format around GPU Triton kernels (`triton_kernels.matmul_ogs`).
2. **How Ironwood TPU (`v7x`) Hardware Executes `gpt_oss_mxfp4`**:
   - Ironwood (`v7x`) Matrix Multiply Units (MXUs) execute matrix multiplications natively in **BF16 (`2,307 TFLOPs/chip`)** and **FP8 (`4,614 TFLOPs/chip`)** (native hardware FP4 MXU arrives in TPU v8).
   - Both **`sglang-torchtpu`** and **`vllm-tpu` (JAX)** implemented custom Pallas TPU kernels that keep the expert weights compressed as 4-bit `mxfp4` in HBM3e (`~65.3 GB` total footprint — reducing HBM bandwidth pressure during decode by `3.58x`) and **unpack the 4-bit weights + `E8M0` scales on the fly inside the TPU Vector Processing Unit (VPU)** into BF16/FP8 registers immediately before feeding the MXU.
   - That is why:
     - **`sglang-torchtpu`** runs `openai/gpt-oss-120b` natively (`TP=8, EP=8`) at **`737.0 ms` TTFT** (`22k` input) and **`4.09 ms` TPOT** (`244.5 tok/s` per stream).
     - **`vllm-tpu` (JAX)** fits a full replica on just `2` chips (`384 GB` HBM3e), running **`4` data-parallel replicas per 4-chip host (`TP=2, DP=4`)** and reaching **`5,847.0` out tok/s** on 1 host, **`14,277.1` out tok/s** on 2 hosts (`8` chips), and **`20,366.9` out tok/s** on 4 hosts (`16` chips).
3. **Why `vllm-torchtpu` Does Not Have MXFP4 Results**:
   - In upstream **`vllm-torchtpu`**, `vllm/model_executor/layers/quantization/mxfp4.py` only registers GPU/ROCm Triton kernels and has not wired up a TorchTPU Pallas VPU-unpacking kernel for `gpt_oss_mxfp4` yet.
   - Attempting to launch `openai/gpt-oss-120b` on `vllm-torchtpu` immediately fails during quantization config validation with:
     ```text
     ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu.
     ```

---

### Question 2: Why is `lmsys/gpt-oss-120b-bf16` (`234 GB` Pure Unquantized BF16) ideal for comparing all 3 stacks, and why is it nearly as fast as MXFP4 on Ironwood?

1. **Fits Easily in a Single 4-Chip Host's `768 GB` HBM3e**:
   - A single Ironwood host (`tpu7x-standard-4t`) provides **`768 GB` of HBM3e** (`192 GB/chip × 4`) and **`29.6 TB/s` of aggregate HBM3e bandwidth** (`7.4 TB/s × 4`).
   - Even completely unquantized in 16-bit `bfloat16`, [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) is **`233.7 GB`** — just **`30.4%` of a single host's HBM3e** (`~29.2 GB` per TensorCore when sharded across `TP=8, EP=8`), leaving **`> 400 GB` of HBM3e** available for KV cache!
2. **Only `5.1B` Active Parameters per Token**:
   - Although `GPT-OSS 120B` has `117B` total parameters (`128` experts per layer), it activates only **`top-4` experts (`5.1B` active parameters) per token**.
   - At `29.6 TB/s` host HBM3e bandwidth, reading `5.1B` BF16 parameters (`~10.2 GB`) takes `< 0.35 ms` across the 4 chips.
   - As a result, **`sglang-torchtpu` serves Pure Unquantized BF16 at `4.97 ms` TPOT** (BF16 KV; `201.2 tok/s` per stream — within `0.88 ms` of MXFP4's `4.09 ms` with FP8 KV) and **identical `22k` prefill latency (`739.5 ms` TTFT on BF16 vs. `737.0 ms` on MXFP4)** with **zero quantization accuracy loss**.

---

### Question 3: Why did `lmsys/gpt-oss-120b-bf16` need a lightweight load-time patch on `vllm-torchtpu` and `sglang-torchtpu` (while `vllm-tpu` needed zero patches)?

When running [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16):
- **`vllm-tpu` (JAX)** loads and serves the BF16 checkpoint out-of-the-box with **zero model-code patches** (`2,463.7` out tok/s on 1 host and **`5,567.6` out tok/s** across 2 hosts; `1k/1k`, BF16 KV).
- On **`vllm-torchtpu`** and **`sglang-torchtpu`**, upstream `gpt_oss.py` and the unquantized BF16 MoE layer had two small omissions because OpenAI originally released `gpt-oss-120b` only in MXFP4 and with a non-standard hidden dimension (`hidden_size = 2880`).
- We resolve both automatically at weight-loading time via **[`scripts/patches/apply_vllm_torchtpu_bf16_patches.py`](scripts/patches/apply_vllm_torchtpu_bf16_patches.py)** and **[`scripts/patches/apply_sglang_bf16_patches.py`](scripts/patches/apply_sglang_bf16_patches.py)**:

#### Detail A: Non-Standard `hidden_size = 2880` vs. TPU MXU `128 × 128` Hardware Tiles (Correctness Fix on `vllm-torchtpu`)
- **The Hardware Rule**: TPU Matrix Multiply Units (MXUs) and Pallas grouped-matmul (`gmm_v2`) MoE kernels operate on **`128 × 128` hardware tiles**, requiring both the input/output hidden dimension (`H`) and intermediate MoE dimension (`I`) of expert weight tensors to be **exact multiples of `128`**.
- **Why `GPT-OSS 120B` Triggers This**: Almost all standard LLMs use a `hidden_size` divisible by `128` (e.g., `2048 = 16×128`, `4096 = 32×128`, `5120 = 40×128`). However, **`GPT-OSS 120B` has `hidden_size = 2880`**, which is **`22.5 × 128`** — leaving an unaligned remainder of **`64` columns** (`2944 = 23 × 128` is the next multiple of `128`).
- **What Upstream `vllm-torchtpu` Missed**:
  - In `vllm_torchtpu/layers/adapter/quantization/unquantized.py` (`VllmUnquantizedFusedMoEMethod.process_weights_after_loading`), upstream added 128-tile padding for `intermediate_size` (`I`), **but forgot to pad `hidden_size` (`H = 2880 -> 2944`, `+64` zero columns)** on `w13_weight`, `w2_weight`, and `w2_bias`:
    ```python
    # Upstream padded intermediate_size (I), but left hidden_size (H = 2880) unpadded!
    # Our 6-line load-time patch in apply_vllm_torchtpu_bf16_patches.py pads H (2880 -> 2944):
    pad_H = (128 - (H % 128)) % 128  # (128 - (2880 % 128)) % 128 = 64
    if pad_H > 0:
        w13_weight = torch.nn.functional.pad(w13_weight, (0, pad_H))       # [16, 5760, 2880] -> [16, 5760, 2944]
        w2_weight  = torch.nn.functional.pad(w2_weight, (0, 0, 0, pad_H))  # [16, 2880, 2880] -> [16, 2944, 2880]
        if w2_bias is not None:
            w2_bias = torch.nn.functional.pad(w2_bias, (0, pad_H))         # [16, 2880]       -> [16, 2944]
    ```
  - **Without vs. With the Patch on `vllm-torchtpu`** (correctness):
    | Metric on `vllm-torchtpu` (`lmsys/gpt-oss-120b-bf16`) | Unpatched Upstream (`H = 2880` Unaligned) | With Load-Time Patch (`H = 2880 -> 2944`) | Effect |
    |---|---|---|---|
    | **Expert Weight Shapes (server log)** | `w13=[16, 2880, 5888]`, `w2=[16, 2944, 2880]` (`H` unpadded) | `w13=[16, 2944, 5888]`, `w2=[16, 2944, 2944]` (`H` padded) | `128`-aligned tiles |
    | **Sanity-Prompt Outputs** | Invalid: every response is a run of `!` tokens | Coherent answers | Valid output requires the patch |

    Raw records (BF16 KV, identical server arguments): without the patch [`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T074125Z`](results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T074125Z) (invalid output; not a performance baseline; includes `server_log_excerpt.txt`), with the patch [`results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T100207Z`](results/gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T100207Z). All `vllm-torchtpu` performance numbers in this repository come from the patched configuration.

#### Detail B: Standard Hugging Face BF16 Tensor Mapping & `swigluoai` Deinterleaving on `sglang-torchtpu`
- Because OpenAI originally released `gpt-oss-120b` only in MXFP4, upstream `sglang/srt/models/gpt_oss.py` only implemented weight loading for MXFP4 block/scale tensor names (`gate_up_proj_blocks`).
- Furthermore, OpenAI's custom `swigluoai` activation stores `gate` and `up` projection channels **interleaved** along the intermediate dimension (`even` indices `[0, 2, 4, ...]` = `gate`, `odd` indices `[1, 3, 5, ...]` = `up`), whereas SGLang's TPU `FusedMoE` kernel expects `[gate, up]` concatenated contiguously (`[0:2880]` = `gate`, `[2880:5760]` = `up`).
- **[`scripts/patches/apply_sglang_bf16_patches.py`](scripts/patches/apply_sglang_bf16_patches.py)** runs once at startup and:
  1. Maps standard Hugging Face BF16 expert tensor names (`experts.gate_up_proj`, `experts.down_proj`, `experts.gate_up_proj_bias`, `experts.down_proj_bias`) and slices the `128` experts across `EP=8` (`16` local experts per TensorCore).
  2. Deinterleaves `swigluoai` channels (`w13[:, ::2, :]` and `w13[:, 1::2, :]`) once during weight loading so no runtime permutation overhead occurs during inference.
  3. Pads `hidden_size` from `2880 -> 2944` (`+64` zero columns) to align with TPU Pallas `gmm` `128 × 128` tiles.
- With this startup script applied, **`sglang-torchtpu` achieves the #1 lowest latency across all stacks on Native BF16: `739.5 ms` TTFT on `22k` input, `4.97 ms` TPOT (`201.2 tok/s` per stream), `2,530.8` out tok/s on 1 host (`4` chips), and `5,084.5` out tok/s on 2 hosts (`8` chips) on `1k/1k` (`2.01x` at the same per-replica concurrency, derived)**!

---

## Repository Structure

```text
.
├── README.md                                     # Architecture, benchmark summary & TorchTPU deep dive
├── INFRA_SETUP.md                                # Step-by-step GKE + DWS Calendar Mode Reservation guide
├── BENCHMARK_RESULTS.md                          # Complete 1h, 2h, and 4h benchmark tables & measured SLA goodput
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
    │   ├── apply_vllm_torchtpu_bf16_patches.py   # Pallas hidden_size (2880->2944) padding (correctness fix)
    │   └── README.md                             # Detailed architectural & code-level patch documentation
    ├── stage_hf_to_gcs.py                        # Multi-threaded HF -> GCS weight streamer
    ├── run_v7x_bench.py                          # Unified server launcher, warmup verifier & bench runner
    ├── lb_proxy.py                               # Zero-buffering async least_conn HTTP load balancer
    ├── bench_agent.py                            # Sidecar HTTP agent for CPU-isolated benchmark execution
    ├── bench_client.py                           # Standalone async OpenAI streaming benchmark client
    ├── mock_metadata.py                          # Local GCE metadata proxy for single-host slice isolation
    ├── metadata_hook.c                           # LD_PRELOAD connect() redirector pairing with mock_metadata.py
    └── summarize_results.py                      # Renders Markdown SLA & throughput tables from summary.json
```
