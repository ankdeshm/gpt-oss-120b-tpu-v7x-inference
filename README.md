# GPT-OSS 120B on Google Cloud TPU v7x

Single-host (`4` chips) and multi-host (`8` and `16` chips) serving benchmarks for [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (unquantized BF16) and [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (native MXFP4) on Google Cloud Ironwood TPU (`v7x`) across `sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu` (JAX). Section C adds follow-up single-host runs with the MXFP4 checkpoint: higher loads, a load-time patch that lets `vllm-torchtpu` serve MXFP4, and DFlash speculative decoding.

**Summary:** On unquantized BF16 (`233.7 GB`, 1 host / 4 chips), `sglang-torchtpu` (`TP=8, EP=8` with load-time `swigluoai` deinterleaving and 128-tile alignment) achieves **4.97 ms decode TPOT and 739.5 ms TTFT on `22k/4k` (1.72× faster TPOT and 16.0% lower TTFT than the 8.54 ms / 880.5 ms `vllm-tpu` JAX BF16 baseline)** and scales linearly to **5,084.5 output tok/s on `1k/1k` across 2 hosts (`2.01×` vs. 1 host at 32 requests/replica)**. `vllm-tpu` (JAX) reaches **5,567.6 output tok/s on 2-host BF16 (`CON=128`)** and **20,366.9 output tok/s on 4-host (`16`-chip) MXFP4 + FP8 KV (`3.48×` vs. 1 host)**.

**Follow-up on one host (section C):** with higher request limits, JAX `vllm-tpu` gave the most throughput from 64 to 2,048 concurrent requests: 22,163.6 output tok/s at 1,024 (median 41.9 ms per output token) and **22,952.2 at 2,048**, vs. 5,847.0 at 64 in section B. `vllm-torchtpu` with a new MXFP4 load-time patch kept scaling to **26,738.2 tok/s at 8,192 requests** (26,626.0 in a second run), at a median 275.9 ms per output token and 15.1 s to the first token. With 22k-token prompts and 4k outputs, JAX `vllm-tpu` also gave the most throughput from 8 to 512 requests (5,648.1 tok/s at 512), while `vllm-torchtpu` returned the first token sooner under load (median 5.8 s vs. 9.8 s at 512 requests). For single requests, `sglang-torchtpu` with DFlash speculative decoding gave the lowest time per output token (3.22 ms, vs. 5.23 ms without DFlash).

**v5p vs. v7x (section D):** both stacks also ran on one TPU v5p host (4 chips); JAX `vllm-tpu` needed the expert weights converted to INT8 at load time. With the same stack and settings on both, the v7x host gave 1.11× to 1.16× the v5p host's throughput for one request and 1.74× to 2.30× from 32 to 256 concurrent requests. The most throughput measured at 1,024 requests was 22,163.6 tok/s on v7x and 6,982.4 on v5p (3.17×; FP4 experts on v7x, INT8 on v5p).


---

## 1. Model, Framework, and Hardware

- **Model:** GPT-OSS 120B sparse Mixture-of-Experts causal language model.
  - **Parameters & Architecture:** **117B total parameters** with **5.1B active parameters per token** across `36` transformer layers (`128` routed experts per layer, top-4 routing, `hidden_size = 2880`, `intermediate_size = 2880`, `64` query heads / `8` KV heads of dim `64`, custom interleaved `swigluoai` activation).
  - **Checkpoints & Precision:**
    1. **Unquantized `bfloat16` (primary):** [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (`~233.7 GB`, MoE experts dequantized from MXFP4 to BF16) with `bfloat16` KV cache (plus optional `fp8` KV cache).
    2. **Native `MXFP4` *(optional, non-BF16)*:** [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`~65.3 GB`, 4-bit `E2M1` + `E8M0` scale MoE experts and `bfloat16` attention) with `fp8` KV cache, plus optional [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) draft model.
    3. **DFlash draft model *(section C)*:** [`z-lab/gpt-oss-120b-DFlash`](https://huggingface.co/z-lab/gpt-oss-120b-DFlash), a small draft model that proposes 8 or 16 tokens per step for speculative decoding with the MXFP4 checkpoint.
  - **Licenses (read before download):** `openai/gpt-oss-120b` is Apache-2.0, and its usage policy adds "By using OpenAI gpt-oss-120b, you agree to comply with all applicable law." `z-lab/gpt-oss-120b-DFlash` is MIT. `lmsys/gpt-oss-120b-bf16` (a BF16 conversion of the Apache-2.0 weights) and `lmsys/EAGLE3-gpt-oss-120b-bf16` do not state a license on their model cards. The stated licenses have no territorial or company-size restrictions. This repo does not redistribute weights.
- **Frameworks:**
  1. **`sglang-torchtpu`:** PyTorch / TorchTPU + Pallas kernels (`TP=8, EP=8`; native VPU-unpacked MXFP4 support; BF16 via [`scripts/patches/apply_sglang_bf16_patches.py`](./scripts/patches/apply_sglang_bf16_patches.py)).
  2. **`vllm-torchtpu`:** PyTorch / TorchTPU + Pallas kernels (`TP=8, DP=1, EP=true`; BF16 via [`scripts/patches/apply_vllm_torchtpu_bf16_patches.py`](./scripts/patches/apply_vllm_torchtpu_bf16_patches.py), which pads `hidden_size = 2880 → 2944` to fix unaligned `128×128` Pallas `gmm_v2` reads; MXFP4 is not supported upstream, and the follow-up patch [`scripts/patches/apply_vllm_torchtpu_gptoss_mxfp4_patch.py`](./scripts/patches/apply_vllm_torchtpu_gptoss_mxfp4_patch.py) adds it, see section C).
  3. **`vllm-tpu` (JAX / `tpu-inference`):** Out-of-the-box support for both BF16 (`TP=8, EP=true`) and MXFP4 (`TP=2, DP=4`, 4 replicas per 4-chip host) with zero patches.
- **Hardware:** Google Cloud Ironwood TPU (`tpu7x-standard-4t`) on GKE (`192 GB` HBM3e, `7.4 TB/s` bandwidth, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip):
  - `1` Host (`4` chips / `8` TensorCores, `768 GB` HBM3e), `2` Hosts (`8` chips, `1.54 TB` HBM3e), and `4` Hosts (`16` chips, `3.07 TB` HBM3e) behind `least_conn` HTTP proxy [`scripts/lb_proxy.py`](./scripts/lb_proxy.py).
- **TPU v5p (section D only):** one host with `4` chips (`95 GiB` HBM2e, `2,765 GB/s`, `459` BF16 TFLOP/s per chip; one device per chip). JAX `vllm-tpu` on v5p converts the MXFP4 experts to INT8 at load time.

---

## 2. Main Results

All runs in sections A and B use `vllm bench serve` with `--no-enable-prefix-caching`, uncached random tokens (`0%` cache hit), and `--ignore-eos` (`0` failed requests; raw outputs in [`results/`](./results/)).

### A. Unquantized BF16 Weights (`lmsys/gpt-oss-120b-bf16`, `233.7 GB`) Across 1 & 2 Hosts

| Scale (`Chips`) | Serving Stack & Layout | KV Cache | `22k/4k` `CON=1` TTFT / TPOT P50 | `22k/4k` Closed-Loop Peak Out TPS | `1k/1k` `CON=1` TPOT P50 | `1k/1k` Peak Out TPS | `8k/1k` Peak Out TPS | Comparison vs. 1-Host `vllm-tpu` JAX BF16 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 Host (`4`) | `vllm-tpu` (JAX) `TP=8, EP` | BF16 | 880.5 ms / 8.54 ms | **1,249.9** (`CON=32`) | 8.21 ms | 2,463.7 (`CON=32`) | 1,557.4 (`CON=32`) | Baseline (`1.00×`, zero patches) |
| **1 Host (`4`)** | **`sglang-torchtpu` `TP=8, EP=8`** | **BF16** | **739.5 ms / 4.97 ms** | 1,150.1 (`CON=32`) | **4.85 ms** | **2,530.8** (`CON=32`) | **1,641.9** (`CON=32`) | **1.72× faster TPOT**; `-16.0%` TTFT; `+2.7%` `1k/1k` TPS |
| 1 Host (`4`) | `sglang-torchtpu + EAGLE3` `TP=8, EP=8` | BF16 | 762.8 ms / 5.48 ms | 934.1 (`CON=32`) | 5.32 ms | 2,059.0 (`CON=32`) | 1,333.7 (`CON=32`) | `1.56×` faster TPOT vs. JAX (`-18.6%` `1k/1k` vs. non-EAGLE3) |
| 1 Host (`4`) | `vllm-torchtpu` `TP=8, EP` | BF16 | 878.7 ms / 5.95 ms | 733.6 (`CON=32`) | 5.30 ms | 2,243.0 (`CON=32`) | 1,223.2 (`CON=32`) | `1.44×` faster TPOT vs. JAX; `-9.0%` `1k/1k` TPS |
| 1 Host (`4`) | `vllm-torchtpu` `TP=8, EP` | FP8 *(opt.)* | 877.1 ms / 5.96 ms | 739.8 (`CON=32`) | 5.30 ms | 2,284.8 (`CON=32`) | 1,243.1 (`CON=32`) | `+1.9%` `1k/1k` TPS over BF16-KV `vllm-torchtpu` |
| 1 Host (`4`) | `vllm-tpu` (JAX) `TP=8, EP` | FP8 *(opt.)* | 880.3 ms / 8.48 ms | **1,302.6** (`CON=32`) | 8.26 ms | **2,540.3** (`CON=32`) | 1,582.9 (`CON=32`) | `+3.1%` `1k/1k` & `+4.2%` `22k/4k` over BF16-KV JAX |
| **2 Hosts (`8`)** | **`2x sglang-torchtpu` `TP=8, EP=8`** | **BF16** | 771.4 ms / 5.05 ms | **2,433.8** (`CON=32`) | 5.77 ms (`CON=8`) | 5,084.5 (`CON=64`) | **3,281.0** (`CON=128`) | **2.01× `1k/1k` & 1.94× `22k/4k`** vs. 1-host `sglang` (32 req/rep) |
| **2 Hosts (`8`)** | **`2x vllm-tpu` (JAX) `TP=8, EP`** | **BF16** | 882.6 ms / 5.99 ms | 1,460.5 (`CON=64`) | 6.63 ms (`CON=8`) | **5,567.6** (`CON=128`) | 2,814.4 (`CON=128`) | **2.26× `1k/1k` peak** (`1.80×` at 32 req/rep) vs. 1-host JAX |

### B. Optional: Native MXFP4 Experts (`openai/gpt-oss-120b`, `65.3 GB`) with FP8 KV Across 1, 2 & 4 Hosts

| Scale (`Chips`) | Serving Stack & Layout | `22k/4k` `CON=1` TTFT / TPOT P50 | `22k/4k` Closed-Loop Peak Out TPS | `1k/1k` Peak Out TPS | `1k/1k` Max Goodput (`req/s`) | `8k/1k` Peak Out TPS | Comparison |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 Host (`4`) | `sglang-torchtpu` `TP=8, EP=8` | **737.0 ms / 4.09 ms** | 1,242.2 (`CON=16`) | 2,587.5 (`CON=32`) | 1.104 (`CON=8`) | 1,630.1 (`CON=32`) | **1.22× faster TPOT** vs. 1-host BF16 `sglang` (`4.97 ms`) |
| 1 Host (`4`) | `vllm-tpu` (JAX) `TP=2, DP=4` | 1,279.3 ms / 5.70 ms | 1,789.4 (`CON=32`) | 5,847.0 (`CON=64`) | 3.340 (`CON=32`) | 2,696.6 (`CON=64`) | **2.37× `1k/1k` TPS** vs. 1-host BF16 JAX (`4×` DP replicas/host) |
| 2 Hosts (`8`) | `2x vllm-tpu` (JAX) `TP=2, DP=4` | 1,389.1 ms / 5.68 ms | 3,315.7 (`CON=64`) | 14,277.1 (`CON=256`) | 5.054 (`CON=64`) | 6,533.7 (`CON=128`) | **2.44× `1k/1k` peak** vs. 1-host MXFP4 JAX |
| **4 Hosts (`16`)** | **`4x vllm-tpu` (JAX) `TP=2, DP=4`** | 1,381.2 ms / 5.70 ms | **8,749.7** (`CON=128`) | **20,366.9** (`CON=256`) | **8.584** (`CON=128`) | **9,820.8** (`CON=128`) | **3.48× `1k/1k` & 4.89× `22k/4k` peak** vs. 1-host MXFP4 JAX |

### C. Follow-Up: One Host at Higher Load (MXFP4 + FP8 KV)

These runs came after sections A and B, on one host (4 chips) with newer container images. `vllm-tpu` (JAX) and `vllm-torchtpu` were measured with `vllm bench serve`, and `sglang-torchtpu` with `sglang.bench_serving`. Both clients send random prompts of exactly the stated length and generate the full output length. Output tokens per second, 1k in / 1k out (raw results in [`results/gptoss120b_1h_4c_mxfp4_followup/`](./results/gptoss120b_1h_4c_mxfp4_followup/)):

| Stack and setup (requests per replica = running-request limit) | Client | 1 request | 64 | 128 | 256 | 512 | 1,024 | 2,048 | 4,096 | 8,192 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm-tpu` (JAX) `TP=2, DP=4`, 64 per replica, section B run | vllm | 180.7 | 5,847.0 | – | – | – | – | – | – | – |
| `vllm-tpu` (JAX) same settings, newer image | vllm | – | 5,788.5 | 7,589.8 | 9,434.6 | 10,134.9 | – | – | – | – |
| `vllm-tpu` (JAX) 128 per replica | vllm | – | – | – | 11,283.7 | 15,876.6 | 15,784.4 | – | – | – |
| `vllm-tpu` (JAX) 512 per replica | vllm | – | – | – | – | **17,190.0** | 22,163.6 | 21,387.7 | – | – |
| `vllm-tpu` (JAX) 1,024 per replica | vllm | – | – | – | – | – | – | 22,952.2 | 22,046.9 | – |
| `vllm-torchtpu` + MXFP4 patch, `TP=2, DP=4`, 64 per replica | vllm | 124.5 | 4,173.6 | 5,313.7 | – | – | – | – | – | – |
| same, 128 per replica | vllm | – | 3,337.2 | 5,806.9 | 7,849.5 | 11,426.5 | – | – | – | – |
| same, 256 per replica | vllm | – | – | – | 7,451.6 | 11,394.8 | 14,867.1 | – | – | – |
| same, 512 per replica | vllm | – | – | – | – | – | 13,774.0 | 20,284.0 | – | – |
| same, 1,024 per replica | vllm | – | – | – | – | – | – | 19,672.4 | 24,357.1 | – |
| same, 2,048 per replica | vllm | – | – | – | – | – | – | – | 21,451.4 | **26,738.2** |
| `sglang-torchtpu` `TP=8, EP=8`, compiled | sglang | 190.0 | 2,586.8 | – | – | – | – | – | – | – |
| `sglang-torchtpu`, 4 attention groups, decode batch 64 | sglang | 143.4 | 4,177.8 | 6,073.0 | 8,114.1 | – | – | – | – | – |
| same, decode batch 128 | sglang | – | – | 6,271.5 | 9,133.0 | 10,988.6 | – | – | – | – |
| same, decode batch 256 | sglang | – | – | – | 9,354.2 | 11,619.3 | 13,499.9 | – | – | – |
| `sglang-torchtpu`, 8 attention groups, decode batch 64 | sglang | – | – | 6,213.2 | 8,688.0 | 11,369.1 | – | – | – | – |
| same, decode batch 128 | sglang | – | – | – | – | 12,400.4 | 13,869.7 | – | – | – |
| same, decode batch 256 | sglang | – | – | – | – | – | 13,861.0 | 12,860.8 | – | – |
| same, decode batch 128, 16,384-token prefill chunks | sglang | – | – | – | – | 13,479.4 | 15,748.3 | – | – | – |
| same, decode batch 128, 32,768-token prefill chunks | sglang | – | – | – | – | 13,404.3 | 16,119.6 | – | – | – |
| same, decode batch 256, 16,384-token prefill chunks | sglang | – | – | – | – | – | 16,246.4 | 16,046.6 | – | – |
| `sglang-torchtpu`, 8 attention groups, decode batch 64, measured with `vllm bench serve` | vllm | – | – | 6,142.2 | 8,396.3 | 11,743.4 | – | – | – | – |

"Per replica" is the running-request limit of each of the 4 data-parallel replicas of the vLLM stacks. When the load is above a setup's total limit, the extra requests wait in a queue and the time to first token grows (33.4 s for the JAX 128-per-replica run at 1,024 requests). More concurrent requests also make each request slower: the median time per output token was 10.5 ms for JAX at 64 requests, 41.9 ms for JAX at 1,024, 160.1 ms for `vllm-torchtpu` at 4,096 and 275.9 ms at 8,192. The last row measures one sglang setup with `vllm bench serve`; its throughput agreed with sglang's own client within 3.4% (appendix section 4.5). Second runs of the main high-load setups, each with a fresh server, came within 4.2% of the first runs (appendix section 4.2). The 64-request point of the 128-per-replica `vllm-torchtpu` run was its first point after startup and came out lower than the 64-per-replica run (appendix section 4.6). Latency for every point is in section 4.4 of [`APPENDIX.md`](./APPENDIX.md).

Long prompts at higher load (8k in / 1k out), output tokens per second:

| Stack and setup | Client | 64 | 128 | 256 | 512 | 1,024 | 2,048 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm-tpu` (JAX) 128 per replica | vllm | 4,023.4 | 4,133.2 | 6,035.6 | 7,447.1 | – | – |
| `vllm-tpu` (JAX) 512 per replica | vllm | – | – | – | – | 8,798.0 | **9,743.8** |
| `vllm-torchtpu` + MXFP4 patch, `TP=2, DP=4`, 256 per replica | vllm | 3,117.3 | 3,829.3 | 5,061.6 | 5,971.3 | – | – |
| same, 512 per replica | vllm | – | – | – | – | 6,881.7 | – |
| same, 1,024 per replica | vllm | – | – | – | – | – | 7,632.6 |
| `sglang-torchtpu`, 8 attention groups, decode batch 128 | sglang | 2,482.9 | 3,165.4 | 3,783.8 | 3,823.3 | – | – |
| same, 16,384-token prefill chunks | sglang | 2,590.7 | 3,123.5 | 3,725.8 | 3,937.3 | – | – |

JAX `vllm-tpu` was again highest at every load where it was measured. With 8k prompts, `sglang-torchtpu` took a median 5.9 to 15.0 s to return the first token with its default prefill chunk and 4.6 to 10.1 s with 16,384-token chunks, vs. 1.5 to 1.9 s for `vllm-torchtpu` up to 512 requests. Section B measured 2,696.6 tok/s at 64 requests with 8k prompts (96 prompts, older image); these runs sent 2 prompts per concurrent request, so the two numbers are not directly comparable.

22k-token prompts with 4k outputs at higher load. Each cell is output tokens per second, with the median time to first token / median time per output token in brackets:

| Stack and setup | Client | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1,024 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm-tpu` (JAX) 128 per replica | vllm | 1,079.4 (4.0 s / 6.4 ms) | 1,672.0 (7.0 s / 7.8 ms) | 2,295.6 (12.1 s / 10.9 ms) | 3,120.4 (17.8 s / 16.2 ms) | 3,973.9 (13.7 s / 29.4 ms) | 4,885.5 (14.4 s / 48.7 ms) | **5,648.1** (9.8 s / 84.7 ms) | – |
| `vllm-torchtpu` + MXFP4 patch, `TP=2, DP=4`, 256 per replica | vllm | 833.2 (3.4 s / 8.8 ms) | 1,495.7 (4.0 s / 9.8 ms) | 2,193.1 (4.1 s / 13.6 ms) | 2,825.0 (4.3 s / 20.8 ms) | 3,479.5 (3.5 s / 34.8 ms) | 3,974.8 (5.5 s / 61.9 ms) | 4,488.9 (5.8 s / 108.0 ms) | 4,677.2 (62.0 s / 184.2 ms) |
| `sglang-torchtpu`, 8 attention groups, decode batch 128, 16,384-token prefill chunks | sglang | 903.0 (3.8 s / 7.9 ms) | 1,246.1 (6.6 s / 11.3 ms) | 1,758.1 (9.6 s / 15.9 ms) | 2,350.2 (15.8 s / 23.5 ms) | 2,875.3 (14.9 s / 39.3 ms) | 3,167.3 (27.5 s / 72.5 ms) | – | – |

JAX `vllm-tpu` gave the most throughput and the lowest time per output token at every load where it was measured (5,648.1 tok/s at 512 requests), but took longer to return the first token: a median 9.8 to 17.8 s from 32 to 512 requests, vs. 3.4 to 5.8 s for `vllm-torchtpu` up to 512 requests (same client). At 1,024 requests `vllm-torchtpu` gave 4,677.2 tok/s with a median 62.0 s to the first token. The JAX points at 128 and 256 requests come from a second server start with the same settings, and the 512-request point from a third; a fourth start repeated the 512-request point and gave 5,640.1 tok/s (0.1% apart). At 128 and 256 requests the sglang server answered 2 of 256 and 3 of 512 requests with HTTP 400 (Bad Request); those requests are not counted, and the log does not say why. sglang's first-token times come from its own client and should only be compared with other sglang numbers (appendix section 4.5). None of these runs went above its running-request limit.

One request at a time, and long prompts (median values):

| Setup | Client | 1k in / 1k out, 1 request: tok/s (time per output token) | 22k in / 4k out, 1 request: time to first token / per output token |
| :--- | :--- | ---: | ---: |
| `sglang-torchtpu` `TP=8, EP=8`, compiled | sglang | 190.0 (5.23 ms) | **542 ms** / 5.18 ms |
| `sglang-torchtpu` `TP=8` + DFlash-8 | sglang | 260.8 (3.47 ms) | 552 ms / 4.68 ms |
| `sglang-torchtpu` `TP=8` + DFlash-16 | sglang | **272.9 (3.22 ms)** | 549 ms / 4.21 ms |
| `vllm-torchtpu` + MXFP4 patch, `TP=8` | vllm | 226.2 (4.39 ms) | 707 ms / 4.76 ms |
| `vllm-torchtpu` + MXFP4 patch, `TP=8` + DFlash-8 | vllm | 233.6 (4.12 ms) | 832 ms / 5.41 ms |

DFlash speeds up single requests but lowers throughput under load (appendix section 4.6). None of these setups meets the `22k/4k` target of at most 500 ms to the first token from appendix section 2.C.

### D. TPU v5p vs. v7x, One Host Each (MXFP4)

A TPU v5p host and a TPU v7x host both have 4 chips. A v5p chip has 95 GiB of HBM and appears as one device; a v7x chip has 192 GB of HBM and two TensorCores, which appear as two devices. We served the MXFP4 checkpoint on one host of each with the same stacks, server flags and client settings, using all devices of the host (`TP=4` on v5p; `TP=8` or `TP=2, DP=4` on v7x). Both stacks ran on v5p. `sglang-torchtpu` needed no changes. JAX `vllm-tpu` needed two settings: the expert weights are converted from FP4 to INT8 when the model loads, because its MoE kernel for BF16 × FP4 does not compile for v5p, and data parallelism runs in a single process. Output tokens per second per host, 1k in / 1k out (raw results in [`results/gptoss120b_v5p_vs_v7x/`](./results/gptoss120b_v5p_vs_v7x/)):

| Chip | Stack and setup | 1 request | 8 | 32 | 64 | 128 | 256 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, FP4 experts (as in section C) | 183.2 | 1,335.6 | 3,890.3 | 5,443.0 | 7,740.3 | 11,083.9 |
| v7x | JAX `vllm-tpu` `TP=2, DP=4`, INT8 experts | 163.3 | 1,137.4 | 2,903.5 | 4,502.8 | 5,748.0 | 9,180.7 |
| v5p | JAX `vllm-tpu` `TP=4`, INT8 experts | 141.2 | 828.4 | 1,370.7 | 2,000.2 | 2,846.5 | 4,088.6 |
| v7x | `sglang-torchtpu` `TP=8`, FP8 KV cache | 186.2 | 1,125.7 | 2,557.9 | 3,683.9 | 5,046.1 | 6,657.7 |
| v5p | `sglang-torchtpu` `TP=4`, FP8 KV cache | 168.1 | 677.9 | 1,114.1 | 1,602.7 | 2,308.9 | 3,236.1 |
| v7x | `sglang-torchtpu` `TP=8`, BF16 KV cache | 189.0 | 1,121.5 | 2,559.6 | 3,611.4 | 5,027.2 | 6,647.8 |
| v5p | `sglang-torchtpu` `TP=4`, BF16 KV cache | 169.9 | 698.1 | 1,153.7 | 1,712.3 | 2,597.1 | 3,819.9 |

v7x host throughput divided by v5p host throughput at the same load (higher than 1.00× means the v7x host gave more):

| Comparison | 1 request | 8 | 32 | 64 | 128 | 256 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| JAX `vllm-tpu`, INT8 experts on both (v5p: better of `TP=4` and `TP=2, DP=2`) | 1.16× | 1.37× | 2.12× | 2.25× | 2.02× | 2.25× |
| `sglang-torchtpu`, FP8 KV cache on both | 1.11× | 1.66× | 2.30× | 2.30× | 2.19× | 2.06× |
| `sglang-torchtpu`, BF16 KV cache on both | 1.11× | 1.61× | 2.22× | 2.11× | 1.94× | 1.74× |
| Best measured setup on each chip (any stack in the table above) | 1.11× | 1.61× | 2.84× | 2.72× | 2.72× | 2.71× |

The running-request limits were 128 per replica (512 per host) for JAX on v7x, 256 for JAX on v5p and 256 for sglang on both, so none of these points waited in a queue. Under heavier load the gap was larger. Highest throughput measured on each host:

| Concurrent requests (1k in / 1k out) | v7x host: output tok/s (setup) | v5p host: output tok/s (setup) | v7x ÷ v5p |
| ---: | :--- | :--- | ---: |
| 512 | 17,190.0 (JAX, FP4 experts) | 5,576.4 (JAX, INT8 experts) | 3.08× |
| 1,024 | 22,163.6 (JAX, FP4 experts) | 6,982.4 (JAX, INT8 experts) | 3.17× |
| 2,048 | 22,952.2 (JAX, FP4 experts) | 8,061.2 (JAX, INT8 experts) | 2.85× |
| 4,096 | 24,357.1 (`vllm-torchtpu`, FP4 experts) | 6,956.6 (JAX, INT8 experts) | 3.50× |

The v7x results in this table use FP4 experts. JAX on v5p only runs with INT8 experts, so part of the gap comes from the expert format. With INT8 experts on both hosts, the gap was 2.28× at 512 requests, 2.56× at 1,024 and 2.73× at 2,048. The v5p host's highest result was at 2,048 requests: 8,061.2 tok/s, with a median of 7.4 s to the first token and 239.8 ms per output token. With a limit of 4,096 running requests it failed to start at memory fraction 0.90 and ran at 0.85, and at 4,096 requests it gave less (6,956.6 tok/s, with a median wait of 302 s for the first token). All v7x numbers in this table come from section C.

For one 22k-token prompt with `sglang-torchtpu`, the median time to the first token was 553 ms on v7x and 1,076 ms on v5p (5.42 ms and 6.08 ms per output token). With JAX at 8 to 64 requests of 22k in / 4k out, the v7x host gave 1.51× to 2.31× the v5p host's throughput with INT8 experts on both (939.8 vs. 623.3 tok/s at 8 requests, 2,469.2 vs. 1,066.7 at 64), and 1.73× to 2.93× with its FP4 experts. Every point with latency, second runs, and what v5p needed are in appendix section 5.

---

## 3. Reproducing the Main Results

For GKE Standard + WorkloadPolicy + DWS Calendar reservation setup, see [`INFRA_SETUP.md`](./INFRA_SETUP.md) or run [`k8s/provision_gke_dws_calendar.sh`](./k8s/provision_gke_dws_calendar.sh):

```bash
# 1. Stage BF16 (233.7 GB) and MXFP4 (65.3 GB) checkpoints to GCS
python3 scripts/stage_hf_to_gcs.py --repo-id lmsys/gpt-oss-120b-bf16 --gcs-bucket "${GCS_BUCKET}" --gcs-prefix models/lmsys/gpt-oss-120b-bf16
python3 scripts/stage_hf_to_gcs.py --repo-id openai/gpt-oss-120b --gcs-bucket "${GCS_BUCKET}" --gcs-prefix models/openai/gpt-oss-120b

# 2. Deploy BF16 on sglang-torchtpu (TP=8, EP=8; applies scripts/patches/apply_sglang_bf16_patches.py) or MXFP4 on vllm-tpu
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-bf16-sglang-v7x.yaml | kubectl apply -f -
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-vllm-tpu-v7x.yaml | kubectl apply -f -

# 3. Run reproducible benchmark suites via scripts/run_v7x_bench.py
python3 scripts/run_v7x_bench.py --config configs/bf16/gptoss120b_bf16_1h_4c_sglang_bf16kv_tp8ep8.json --bench-agent-url http://127.0.0.1:9100
python3 scripts/run_v7x_bench.py --config configs/gptoss120b_4h_16c_vllmtpu_fp8kv_tp2dp4.json --bench-only-url http://gpt-oss-120b-svc.default.svc.cluster.local:8000
```

Section C runs inside a pod on one v7x host with the matching container image (`sglang-torchtpu`, `vllm-torchtpu`, or `vllm/vllm-tpu`). Each script starts the server, runs the load points and writes one JSON file:

```bash
# sglang-torchtpu (setup names and exact flags are listed in the script)
python3 scripts/benchmark_sglang_mxfp4_single_host.py --setups sglang_mxfp4_dp8_tp8_ep8_compiled_bs256_chunk16k,sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_22k,sglang_mxfp4_dp8_tp8_ep8_compiled_bs128_chunk16k_22k_hi,sglang_mxfp4_tp8_ep8_dflash16

# vllm-torchtpu: applies scripts/patches/apply_vllm_torchtpu_gptoss_mxfp4_patch.py, then benchmarks
python3 scripts/benchmark_vllm_torchtpu_mxfp4.py --setups vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024,vllm_torchtpu_mxfp4_tp2_dp4_async_seqs2048,vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k,vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_hi,vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_c512,vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_22k_c1024

# vllm-tpu (JAX): section B settings with a higher running-request limit per replica
MAX_NUM_SEQS=512 LOADS="512 1024 2048" bash scripts/run_vllm_tpu_jax_high_load.sh
MAX_NUM_SEQS=128 INPUT_LEN=8192 LOADS="64 128 256 512" bash scripts/run_vllm_tpu_jax_high_load.sh
MAX_NUM_SEQS=128 INPUT_LEN=22000 OUTPUT_LEN=4000 LOADS="8 16 32 64 128 256 512" bash scripts/run_vllm_tpu_jax_high_load.sh
```

Section D (v5p vs. v7x) runs inside a pod on one host of each type, using the container image of the stack being measured:

```bash
# JAX vllm-tpu on v5p: INT8 experts (CHIP=v5p also selects single-process data parallelism)
CHIP=v5p TP=4 MAX_NUM_SEQS=256 MOE_REQUANTIZE_WEIGHT_DTYPE=int8 MOE_REQUANTIZE_BLOCK_SIZE=256 OUT=./results/v5p_tp4 bash scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh
CHIP=v5p TP=4 MAX_NUM_SEQS=1024 MOE_REQUANTIZE_WEIGHT_DTYPE=int8 MOE_REQUANTIZE_BLOCK_SIZE=256 LOADS="512 1024" OUT=./results/v5p_tp4_1024 bash scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh
CHIP=v5p TP=4 MAX_NUM_SEQS=2048 MOE_REQUANTIZE_WEIGHT_DTYPE=int8 MOE_REQUANTIZE_BLOCK_SIZE=256 LOADS="2048" OUT=./results/v5p_tp4_2048 bash scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh
CHIP=v5p TP=4 MAX_NUM_SEQS=256 MOE_REQUANTIZE_WEIGHT_DTYPE=int8 MOE_REQUANTIZE_BLOCK_SIZE=256 INPUT_LEN=22000 OUTPUT_LEN=4000 LOADS="8 16 32 64" OUT=./results/v5p_tp4_22k bash scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh

# JAX vllm-tpu on v7x: FP4 experts (section C layout), then INT8 experts as on v5p
CHIP=v7x TP=2 DP=4 MAX_NUM_SEQS=128 OUT=./results/v7x_tp2_dp4 bash scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh
CHIP=v7x TP=2 DP=4 MAX_NUM_SEQS=128 MOE_REQUANTIZE_WEIGHT_DTYPE=int8 MOE_REQUANTIZE_BLOCK_SIZE=32 OUT=./results/v7x_tp2_dp4_int8 bash scripts/run_vllm_tpu_jax_v5p_vs_v7x.sh

# sglang-torchtpu: the same flags on both chips (TP=4 on v5p, TP=8 on v7x)
python3 scripts/benchmark_sglang_v5p.py --setups sglang_mxfp4_tp4_ep4_compiled_fp8kv_v5p,sglang_mxfp4_tp4_ep4_compiled_bf16kv_v5p
python3 scripts/benchmark_sglang_mxfp4_single_host.py --setups sglang_mxfp4_tp8_ep8_compiled_bs64_r256,sglang_mxfp4_tp8_ep8_compiled_bs64_r256_bf16kv
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete per-suite concurrency and open-loop Poisson RPS tables across all 13 valid runs, MXFP4 vs. BF16 stack compatibility deep dive, `hidden_size = 2880 → 2944` Pallas tile-alignment correctness analysis (including the unpatched `!`-token run), `EAGLE3` ablation, and `22k/4k` goodput breakdown.
- **[`INFRA_SETUP.md`](./INFRA_SETUP.md)** & **[`scripts/patches/README.md`](./scripts/patches/README.md):** Step-by-step GKE + DWS Calendar reservation setup and code-level documentation of the TorchTPU load-time patches.
- **[`results/`](./results/):** Raw `vllm bench serve` JSONs and `summary.json` files for all runs (re-summarizable via `python3 scripts/summarize_results.py results`).
- **Section C (follow-up runs):** [`APPENDIX.md`](./APPENDIX.md) section 4 has every setup with its exact flags, full results with latency, the same-client check, and what did not work; raw outputs are in [`results/gptoss120b_1h_4c_mxfp4_followup/`](./results/gptoss120b_1h_4c_mxfp4_followup/).
- **Section D (v5p vs. v7x):** [`APPENDIX.md`](./APPENDIX.md) section 5 has the setups, every point with latency, second runs, and what v5p needed; raw outputs are in [`results/gptoss120b_v5p_vs_v7x/`](./results/gptoss120b_v5p_vs_v7x/).
