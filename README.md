# GPT-OSS 120B on Google Cloud TPU v7x

Single-host (`4` chips) and multi-host (`8` and `16` chips) serving benchmarks for [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (unquantized BF16) and [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (native MXFP4) on Google Cloud Ironwood TPU (`v7x`) across `sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu` (JAX).

**Summary:** On unquantized BF16 (`233.7 GB`, 1 host / 4 chips), `sglang-torchtpu` (`TP=8, EP=8` with load-time `swigluoai` deinterleaving and 128-tile alignment) achieves **4.97 ms decode TPOT and 739.5 ms TTFT on `22k/4k` (1.72× faster TPOT and 16.0% lower TTFT than the 8.54 ms / 880.5 ms `vllm-tpu` JAX BF16 baseline)** and scales linearly to **5,084.5 output tok/s on `1k/1k` across 2 hosts (`2.01×` vs. 1 host at 32 requests/replica)**. `vllm-tpu` (JAX) reaches **5,567.6 output tok/s on 2-host BF16 (`CON=128`)** and **20,366.9 output tok/s on 4-host (`16`-chip) MXFP4 + FP8 KV (`3.48×` vs. 1 host)**.

---

## 1. Model, Framework, and Hardware

- **Model:** GPT-OSS 120B sparse Mixture-of-Experts causal language model.
  - **Parameters & Architecture:** **117B total parameters** with **5.1B active parameters per token** across `36` transformer layers (`128` routed experts per layer, top-4 routing, `hidden_size = 2880`, `intermediate_size = 2880`, `64` query heads / `8` KV heads of dim `64`, custom interleaved `swigluoai` activation).
  - **Checkpoints & Precision:**
    1. **Unquantized `bfloat16` (primary):** [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (`~233.7 GB`, MoE experts dequantized from MXFP4 to BF16) with `bfloat16` KV cache (plus optional `fp8` KV cache).
    2. **Native `MXFP4` *(optional, non-BF16)*:** [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) (`~65.3 GB`, 4-bit `E2M1` + `E8M0` scale MoE experts and `bfloat16` attention) with `fp8` KV cache, plus optional [`lmsys/EAGLE3-gpt-oss-120b-bf16`](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16) draft model.
- **Frameworks:**
  1. **`sglang-torchtpu`:** PyTorch / TorchTPU + Pallas kernels (`TP=8, EP=8`; native VPU-unpacked MXFP4 support; BF16 via [`scripts/patches/apply_sglang_bf16_patches.py`](./scripts/patches/apply_sglang_bf16_patches.py)).
  2. **`vllm-torchtpu`:** PyTorch / TorchTPU + Pallas kernels (`TP=8, DP=1, EP=true`; BF16 via [`scripts/patches/apply_vllm_torchtpu_bf16_patches.py`](./scripts/patches/apply_vllm_torchtpu_bf16_patches.py), which pads `hidden_size = 2880 → 2944` to fix unaligned `128×128` Pallas `gmm_v2` reads; MXFP4 not yet supported upstream).
  3. **`vllm-tpu` (JAX / `tpu-inference`):** Out-of-the-box support for both BF16 (`TP=8, EP=true`) and MXFP4 (`TP=2, DP=4`, 4 replicas per 4-chip host) with zero patches.
- **Hardware:** Google Cloud Ironwood TPU (`tpu7x-standard-4t`) on GKE (`192 GB` HBM3e, `7.4 TB/s` bandwidth, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip):
  - `1` Host (`4` chips / `8` TensorCores, `768 GB` HBM3e), `2` Hosts (`8` chips, `1.54 TB` HBM3e), and `4` Hosts (`16` chips, `3.07 TB` HBM3e) behind `least_conn` HTTP proxy [`scripts/lb_proxy.py`](./scripts/lb_proxy.py).

---

## 2. Main Results

All runs use `vllm bench serve` with `--no-enable-prefix-caching`, uncached random tokens (`0%` cache hit), and `--ignore-eos` (`0` failed requests; raw outputs in [`results/`](./results/)).

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

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete per-suite concurrency and open-loop Poisson RPS tables across all 13 valid runs, MXFP4 vs. BF16 stack compatibility deep dive, `hidden_size = 2880 → 2944` Pallas tile-alignment correctness analysis (including the unpatched `!`-token run), `EAGLE3` ablation, and `22k/4k` goodput breakdown.
- **[`INFRA_SETUP.md`](./INFRA_SETUP.md)** & **[`scripts/patches/README.md`](./scripts/patches/README.md):** Step-by-step GKE + DWS Calendar reservation setup and code-level documentation of the TorchTPU load-time patches.
- **[`results/`](./results/):** Raw `vllm bench serve` JSONs and `summary.json` files for all runs (re-summarizable via `python3 scripts/summarize_results.py results`).
