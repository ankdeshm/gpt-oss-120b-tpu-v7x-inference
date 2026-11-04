#!/usr/bin/env python3
"""Apply runtime patches to vllm-torchtpu for Native BF16 lmsys/gpt-oss-120b-bf16 on Ironwood TPU (tpu7x).

Fixes three upstream issues when serving unquantized BF16 GPT-OSS 120B (`quantization_config=None`):
1. `vllm/model_executor/models/gpt_oss.py`:
   - Casts `narrow_weight` to `param.dtype` before `param.copy_(narrow_weight)` when loading BF16 expert weights.
2. `vllm_torchtpu/layers/adapter/attention.py` (`PallasAttentionBackendImpl`):
   - Preserves `pool_head_dim = 64` for `head_size=64` with `FLASH_ATTN` (`hd64` Pallas RPA layout) when `kv_cache.shape[-1] == 128`.
3. `vllm_torchtpu/layers/adapter/quantization/unquantized.py` (`VllmUnquantizedFusedMoEMethod.process_weights_after_loading`):
   - Pads `hidden = w13_weight.shape[1]` (`2880 -> 2944`, multiple of `128`) on `w13_weight`, `w2_weight`, and `w2_bias` to match `fused_moe_gmm.py` activation padding (`padded_hidden_size = w1.shape[1]`), eliminating out-of-bounds Pallas GMM lane reads and improving decode TPOT by `3.7x` (`18.8 ms -> 5.15 ms`).
"""
from __future__ import annotations

from pathlib import Path
import vllm.model_executor.models.gpt_oss as gpt_oss_mod
import vllm_torchtpu.layers.adapter.attention as attn_mod
import vllm_torchtpu.layers.adapter.quantization.unquantized as unq_mod


def patch_vllm_gpt_oss() -> None:
    path = Path(gpt_oss_mod.__file__)
    src = path.read_text()
    if "if narrow_weight.dtype != param.dtype:" in src:
        print(f"[skip] {path} already patched")
        return
    old = """                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)"""
    new = """                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]
                if narrow_weight.dtype != param.dtype:
                    narrow_weight = narrow_weight.to(param.dtype)

                param.copy_(narrow_weight)"""
    if src.count(old) < 2:
        raise RuntimeError(f"Expected 2 occurrences of narrow_weight copy anchor in {path}")
    src = src.replace(old, new)
    path.write_text(src)
    print(f"[ok] Patched {path}")


def patch_torchtpu_attention() -> None:
    path = Path(attn_mod.__file__)
    src = path.read_text()
    guard = 'elif self.head_size == 64 and kv_cache.shape[-1] == 128 and type(self).__name__ == "PallasAttentionBackendImpl":'
    if guard in src:
        print(f"[skip] {path} already patched")
        return
    old = """        elif self._pool_is_seq_along_lane:
            pool_head_dim = kv_cache.shape[-3] * kv_cache.shape[-2]
        else:
            pool_head_dim = kv_cache.shape[-1]"""
    new = f"""        elif self._pool_is_seq_along_lane:
            pool_head_dim = kv_cache.shape[-3] * kv_cache.shape[-2]
        {guard}
            pool_head_dim = 64
        else:
            pool_head_dim = kv_cache.shape[-1]"""
    if old not in src:
        raise RuntimeError(f"Expected pool_head_dim anchor not found in {path}")
    src = src.replace(old, new, 1)
    path.write_text(src)
    print(f"[ok] Patched {path}")


def patch_torchtpu_unquantized() -> None:
    path = Path(unq_mod.__file__)
    src = path.read_text()
    if "aligned_hidden = (hidden + 127) // 128 * 128" in src:
        print(f"[skip] {path} already patched")
        return
    old_weights = """            w2_weight = torch.cat([w2_weight, pad2], dim=1).contiguous()

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)"""
    new_weights = """            w2_weight = torch.cat([w2_weight, pad2], dim=1).contiguous()

        hidden = w13_weight.shape[1]
        aligned_hidden = (hidden + 127) // 128 * 128
        if aligned_hidden != hidden:
            hpad = aligned_hidden - hidden
            w13_weight = torch.nn.functional.pad(w13_weight, (0, 0, 0, hpad)).contiguous()
            w2_weight = torch.nn.functional.pad(w2_weight, (0, hpad)).contiguous()

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)"""

    old_bias = """            layer.w2_bias = torch.nn.Parameter(
                w2_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )"""
    new_bias = """            if aligned_hidden != hidden:
                w2_bias = torch.nn.functional.pad(w2_bias, (0, aligned_hidden - hidden)).contiguous()
            layer.w2_bias = torch.nn.Parameter(
                w2_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )"""

    if old_weights not in src or old_bias not in src:
        raise RuntimeError(f"Expected MoE alignment anchors not found in {path}")
    src = src.replace(old_weights, new_weights, 1).replace(old_bias, new_bias, 1)
    path.write_text(src)
    print(f"[ok] Patched {path}")


if __name__ == "__main__":
    patch_vllm_gpt_oss()
    patch_torchtpu_attention()
    patch_torchtpu_unquantized()
