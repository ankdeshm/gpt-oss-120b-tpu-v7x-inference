# Architectural & Code-Level Guide: Runtime Patches for `GPT-OSS 120B` on TorchTPU (`vllm-torchtpu` & `sglang-torchtpu`)

This directory contains the lightweight startup scripts (`apply_vllm_torchtpu_bf16_patches.py` and `apply_sglang_bf16_patches.py`) used when serving the **Pure Unquantized BF16** checkpoint **[`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16)** (`233.7 GB`) on **Google Cloud Ironwood TPU (`v7x` / `tpu7x`)** using **`vllm-torchtpu`** or **`sglang-torchtpu`**.

---

## 1. Compatibility Matrix: `openai/gpt-oss-120b` (MXFP4) vs. `lmsys/gpt-oss-120b-bf16` (Native BF16)

| Serving Stack | `openai/gpt-oss-120b` (`65.3 GB` Native MXFP4) | `lmsys/gpt-oss-120b-bf16` (`233.7 GB` Pure Unquantized BF16) | Runtime Patch Required? |
|---|---|---|---|
| **`sglang-torchtpu`** *(PyTorch + Pallas)* | **Works Out-of-the-Box** (`TP=8, EP=8`)<br>`737.0 ms` TTFT (`22k`), `4.09 ms` TPOT | **Works with Load-Time Patch** (`TP=8, EP=8`)<br>`739.5 ms` TTFT (`22k`), `4.97 ms` TPOT, `5,084.5` tok/s (`2` hosts) | **MXFP4**: None<br>**BF16**: `python3 scripts/patches/apply_sglang_bf16_patches.py` |
| **`vllm-torchtpu`** *(PyTorch / XLA + Pallas)* | **Not Supported Upstream Yet**<br>(`ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu`) | **Works with Load-Time Patch** (`TP=8, EP=true`)<br>`877.1 ms` TTFT (`22k`), `5.30–5.95 ms` TPOT, `2,284.8` tok/s (`1` host) | **BF16**: `python3 scripts/patches/apply_vllm_torchtpu_bf16_patches.py` |
| **`vllm-tpu`** *(JAX `tpu-inference`)* | **Works Out-of-the-Box** (`TP=2, DP=4` per host)<br>`5,847.0` tok/s (`1` host) -> `20,366.9` tok/s (`4` hosts) | **Works Out-of-the-Box** (`TP=8, --enable-expert-parallel`)<br>`880.3 ms` TTFT (`22k`), `8.21–8.48 ms` TPOT, `2,540.3` tok/s (`1` host) | **None** (Zero patches for both MXFP4 and BF16) |

---

## 2. Why Does `openai/gpt-oss-120b` (Native MXFP4) Work on `sglang-torchtpu` and `vllm-tpu`, But Not on `vllm-torchtpu`?

1. **What `gpt_oss_mxfp4` Is**:
   - OpenAI released `openai/gpt-oss-120b` with its `128` routed Mixture-of-Experts (MoE) weights quantized in **`gpt_oss_mxfp4`** — a 4-bit microscaling format where every block of `32` elements is stored as 4-bit `E2M1` values accompanied by a shared 8-bit `E8M0` exponent scale, designed around GPU Triton kernels.
2. **How Ironwood TPU (`v7x`) Executes It**:
   - Ironwood (`v7x`) Matrix Multiply Units (MXUs) execute natively in **BF16 (`2,307 TFLOPs/chip`)** and **FP8 (`4,614 TFLOPs/chip`)** (native hardware FP4 MXU arrives in TPU v8).
   - **`sglang-torchtpu`** and **`vllm-tpu` (JAX)** include custom Pallas TPU kernels that keep the weights compressed as 4-bit `mxfp4` in HBM3e (`65.3 GB` total) and **unpack the 4-bit blocks + `E8M0` scales on the fly inside the TPU Vector Processing Unit (VPU)** into BF16/FP8 registers right before feeding the MXU.
3. **Why Upstream `vllm-torchtpu` Raises an Error on MXFP4**:
   - In upstream `vllm-torchtpu`, `vllm/model_executor/layers/quantization/mxfp4.py` only registers GPU/ROCm Triton kernels and has not yet registered a TPU backend kernel for `gpt_oss_mxfp4`, raising:
     ```text
     ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu.
     ```

---

## 3. Why Did `lmsys/gpt-oss-120b-bf16` (Native BF16) Need a Lightweight Load-Time Patch on `vllm-torchtpu` and `sglang-torchtpu`?

### 3.1 `apply_vllm_torchtpu_bf16_patches.py` (`vllm-torchtpu`)

When loading [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) (`quantization_config=None`, `233.7 GB` unquantized BF16) on `vllm-torchtpu`, our script applies three targeted load-time fixes:

1. **Pallas `gmm_v2` `128 × 128` Tile Alignment for `hidden_size = 2880 -> 2944` (`vllm_torchtpu/layers/adapter/quantization/unquantized.py`)**:
   - **Root Cause**: TPU Matrix Multiply Units (MXUs) and Pallas grouped-matmul (`gmm_v2`) kernels require both matrix dimensions of MoE expert weights (`intermediate_size` and `hidden_size`) to be **exact multiples of `128`**.
   - While almost all LLMs have a `hidden_size` divisible by `128` (`4096`, `5120`, etc.), **`GPT-OSS 120B` has `hidden_size = 2880`**, which is **`22.5 × 128`** (leaving an unaligned remainder of `64` columns).
   - In `VllmUnquantizedFusedMoEMethod.process_weights_after_loading`, upstream `vllm-torchtpu` padded `intermediate_size` to a multiple of `128`, **but forgot to pad `hidden_size` (`2880 -> 2944`, `+64` zero columns)**.
   - Without padding `hidden_size`, Pallas `gmm_v2` reads `64` uninitialized HBM lanes past the end of every row, corrupting MoE layer outputs into `NaN`s and causing router logits to collapse all tokens onto Expert 0 (`18.80 ms` TPOT).
   - **The 6-Line Fix**:
     ```python
     pad_H = (128 - (H % 128)) % 128  # 64 columns (pads 2880 -> 2944 = 23 * 128)
     if pad_H > 0:
         w13_weight = torch.nn.functional.pad(w13_weight, (0, pad_H))       # [16, 5760, 2880] -> [16, 5760, 2944]
         w2_weight  = torch.nn.functional.pad(w2_weight, (0, 0, 0, pad_H))  # [16, 2880, 2880] -> [16, 2944, 2880]
         if w2_bias is not None:
             w2_bias = torch.nn.functional.pad(w2_bias, (0, pad_H))         # [16, 2880]       -> [16, 2944]
     ```
   - **Result**: Outputs are 100% coherent, expert routing is balanced across all `128` experts, and **single-stream decode latency drops `3.2x–3.5x` from `18.80 ms` down to `5.30–5.95 ms` TPOT (`2,284.8` out tok/s on `1k/1k`)**!

2. **Attention Head Dimension Alignment (`vllm_torchtpu/layers/adapter/attention.py`)**:
   - `GPT-OSS 120B` uses `head_size = 64` (`64` query heads, `8` KV heads). Preserves `pool_head_dim = 64` in `PallasAttentionBackendImpl` (`FLASH_ATTN` `hd64` layout).

3. **BF16 Checkpoint Tensor Dtype Casting (`vllm/model_executor/models/gpt_oss.py`)**:
   - Ensures expert weight slices loaded from `lmsys/gpt-oss-120b-bf16` are cast cleanly to `param.dtype` before `param.copy_`.

---

### 3.2 `apply_sglang_bf16_patches.py` (`sglang-torchtpu`)

When loading [`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) on `sglang-torchtpu`, our script applies two load-time fixes:

1. **Standard Hugging Face BF16 Expert Weight Loader (`sglang/srt/models/gpt_oss.py`)**:
   - Because OpenAI originally released `gpt-oss-120b` only in MXFP4, upstream `sglang/srt/models/gpt_oss.py` only handled MXFP4 block/scale tensor names.
   - `apply_sglang_bf16_patches.py` updates `_load_normal_weights` to map standard Hugging Face BF16 expert tensor names (`experts.gate_up_proj`, `experts.down_proj`, `experts.gate_up_proj_bias`, `experts.down_proj_bias`), slice the `128` experts across `EP=8` (`16` local experts per TensorCore), and transpose `[-2, -1]` to match SGLang's `FusedMoE` weight layout.

2. **`swigluoai` Channel Deinterleaving & `2880 -> 2944` Tile Padding (`sglang/srt/layers/quantization/unquant.py`)**:
   - OpenAI's custom `swigluoai` activation interleaves `gate` (`[::2]`) and `up` (`[1::2]`) channels along the `5760`-wide `gate_up_proj` dimension, whereas SGLang's TPU `FusedMoE` Pallas kernel expects `[gate, up]` concatenated contiguously (`[0:2880]` followed by `[2880:5760]`).
   - Our patch deinterleaves `gate` and `up` channels once during weight loading (`torch.cat([w13[:, ::2, :], w13[:, 1::2, :]], dim=1)`) and pads `hidden_size` from `2880 -> 2944` (`+64` zero columns) for Pallas `128 × 128` tile alignment.
   - **Result**: **`sglang-torchtpu` achieves `739.5 ms` TTFT on `22k` input, `4.97 ms` single-stream TPOT (`201.2 tok/s`), `2,530.8` out tok/s on 1 host (`4` chips), and `5,084.5` out tok/s (`2.01x` linear scaling) across 2 hosts (`8` chips)**!

---

## 4. Usage

Run the corresponding patcher once inside the container before starting the server (already automated in `k8s/*.yaml` and `configs/bf16/*.json`):

```bash
# For sglang-torchtpu:
python3 scripts/patches/apply_sglang_bf16_patches.py

# For vllm-torchtpu:
python3 scripts/patches/apply_vllm_torchtpu_bf16_patches.py
```
