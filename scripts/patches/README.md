# Runtime Patches for Native BF16 `lmsys/gpt-oss-120b-bf16` on Ironwood TPU (`tpu7x`)

[`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16) stores all `128` routed experts per layer in unquantized `bfloat16` (`quantization_config=None`, `~233.66 GB` in TPU HBM) with non-128-aligned hidden and intermediate dimensions (`hidden_size = 2880`, `intermediate_size = 2880`) and interleaved `swigluoai` gate/up channels.

## Summary by Serving Stack

1. **`vllm-tpu` (JAX / `tpu-inference`)**:
   - **Zero patches required** — loads and serves `lmsys/gpt-oss-120b-bf16` out-of-the-box at `TP=8, --enable-expert-parallel` with both `bfloat16` (`--kv-cache-dtype=auto`) and `fp8` (`--kv-cache-dtype=fp8`) KV caches.
2. **`sglang-torchtpu` (`apply_sglang_bf16_patches.py`)**:
   - **`sglang/srt/models/gpt_oss.py`**: Updates `_load_normal_weights` to map fused BF16 expert tensors (`gate_up_proj`, `down_proj`, `gate_up_proj_bias`, `down_proj_bias`), slice across `EP=8` (`16` local experts per TensorCore), and transpose `[-2, -1]` to match SGLang's `FusedMoE` weight layout.
   - **`sglang/srt/layers/quantization/unquant.py`**: Deinterleaves `swigluoai` gate/up channels (`::2`, `1::2`) before 128-boundary tile padding and pads `hidden_size` (`2880 -> 2944`) for Pallas `gmm` kernels.
3. **`vllm-torchtpu` (`apply_vllm_torchtpu_bf16_patches.py`)**:
   - **`vllm/model_executor/models/gpt_oss.py`**: Casts BF16 checkpoint expert slices to target parameter dtype prior to `param.copy_`.
   - **`vllm_torchtpu/layers/adapter/attention.py`**: Preserves `pool_head_dim = 64` for `head_size = 64` in `PallasAttentionBackendImpl` (`FLASH_ATTN` `hd64` layout).
   - **`vllm_torchtpu/layers/adapter/quantization/unquantized.py`**: Pads `hidden_size` (`2880 -> 2944`, multiple of `128`) on `w13_weight`, `w2_weight`, and `w2_bias` in `VllmUnquantizedFusedMoEMethod.process_weights_after_loading`. Without `hidden_size` padding, Pallas `gmm_v2` reads unaligned trailing lanes (`NaN` outputs + expert-0 routing skew at `18.8 ms` TPOT); with padding, outputs are 100% coherent and decode latency drops **`3.7x` to `5.15 ms` TPOT**.

## Usage

Run the corresponding patcher once inside the container prior to launching the server:

```bash
# For sglang-torchtpu:
python3 scripts/patches/apply_sglang_bf16_patches.py

# For vllm-torchtpu:
python3 scripts/patches/apply_vllm_torchtpu_bf16_patches.py
```
