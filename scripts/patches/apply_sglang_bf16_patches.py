#!/usr/bin/env python3
"""Apply runtime patches to sglang-torchtpu for Native BF16 lmsys/gpt-oss-120b-bf16 on Ironwood TPU (tpu7x).

Fixes two upstream issues when loading unquantized BF16 GPT-OSS 120B (`quantization_config=None`):
1. `sglang/srt/models/gpt_oss.py` (`_load_normal_weights`):
   - Maps fused expert checkpoint tensors (`gate_up_proj`, `down_proj`, `gate_up_proj_bias`, `down_proj_bias`),
   - Slices the expert axis `[E, ...]` across `moe_ep_size` ranks (`128 // 8 = 16` local experts per rank),
   - Transposes `[-2, -1]` from HF `[E, K, N]` layout to SGLang `[E, N, K]` layout and casts BF16 biases to parameter dtype.
2. `sglang/srt/layers/quantization/unquant.py` (`UnquantizedFusedMoEMethod.process_weights_after_loading`):
   - Deinterleaves `swigluoai` gate/up channels (`::2`, `1::2`) before padding halves to 128-element TPU tile boundaries,
   - Pads `hidden_size` (`2880 -> 2944`) on `w13_weight`, `w2_weight`, and `w2_weight_bias` for Pallas `gmm` alignment.
"""
from __future__ import annotations

from pathlib import Path
import sglang.srt.layers.quantization.unquant as unquant_mod
import sglang.srt.models.gpt_oss as gpt_oss_mod


def patch_gpt_oss() -> None:
    path = Path(gpt_oss_mod.__file__)
    src = path.read_text()
    if "_n_loc = self.config.num_local_experts // get_parallel().moe_ep_size" in src:
        print(f"[skip] {path} already patched")
        return

    old_map = """        expert_params_mapping = FusedMoE.make_expert_params_mapping_fused(
            ckpt_gate_up_proj_name="mlp1_weight",
            ckpt_down_proj_name="mlp2_weight",
            ckpt_gate_up_proj_bias_name="mlp1_bias",
            ckpt_down_proj_bias_name="mlp2_bias",
        )"""
    new_map = """        expert_params_mapping = FusedMoE.make_expert_params_mapping_fused(
            ckpt_gate_up_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_gate_up_proj_bias_name="gate_up_proj_bias",
            ckpt_down_proj_bias_name="down_proj_bias",
        )"""
    if old_map in src:
        src = src.replace(old_map, new_map, 1)

    old_load = """                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    if "w2_weight_bias" in name and get_parallel().moe_tp_rank != 0:"""
    new_load = """                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    if get_parallel().moe_ep_size > 1 and loaded_weight.shape[0] == self.config.num_local_experts:
                        _n_loc = self.config.num_local_experts // get_parallel().moe_ep_size
                        _ep_r = get_parallel().moe_ep_rank
                        loaded_weight = loaded_weight[_ep_r * _n_loc : (_ep_r + 1) * _n_loc]
                    if "bias" not in name:
                        loaded_weight = loaded_weight.transpose(-2, -1)
                    if loaded_weight.dtype != param.dtype:
                        loaded_weight = loaded_weight.to(param.dtype)
                    loaded_weight = loaded_weight.contiguous()
                    if "w2_weight_bias" in name and get_parallel().moe_tp_rank != 0:"""
    if old_load not in src:
        raise RuntimeError(f"Expected anchor not found in {path}")
    src = src.replace(old_load, new_load, 1)
    path.write_text(src)
    print(f"[ok] Patched {path}")


def patch_unquant() -> None:
    path = Path(unquant_mod.__file__)
    src = path.read_text()
    if "w13_interleaved = activation == \"swigluoai\"" in src:
        print(f"[skip] {path} already patched")
        return

    old_block = """            w13 = layer.w13_weight.data
            w2 = layer.w2_weight.data
            hidden = w13.shape[-1]
            inter = w2.shape[-1]
            activation = layer.moe_runner_config.activation
            if getattr(layer, "w13_weight_bias", None) is not None or hidden % 128 != 0:"""
    new_block = """            w13 = layer.w13_weight.data
            w2 = layer.w2_weight.data
            hidden = w13.shape[-1]
            inter = w2.shape[-1]
            activation = layer.moe_runner_config.activation
            w13_interleaved = activation == "swigluoai"
            if getattr(layer, "w13_weight_bias", None) is not None or hidden % 128 != 0 or w13_interleaved:"""

    if old_block in src:
        src = src.replace(old_block, new_block, 1)
        src = src.replace(
            "                w13 = _pad_gate_up_halves_axis1(w13, inter, pad_n)",
            "                if w13_interleaved:\n"
            "                    w13 = _deinterleave_axis1(w13)\n"
            "                w13 = _pad_gate_up_halves_axis1(w13, inter, pad_n)",
            1,
        )
        src = src.replace(
            "                    w13_b = layer.w13_weight_bias.data\n"
            "                    w13_b = _pad_gate_up_halves_axis1(w13_b.unsqueeze(-1), inter, pad_n).squeeze(-1)",
            "                    w13_b = layer.w13_weight_bias.data\n"
            "                    if w13_interleaved:\n"
            "                        w13_b = _deinterleave_axis1(w13_b.unsqueeze(-1)).squeeze(-1)\n"
            "                    w13_b = _pad_gate_up_halves_axis1(w13_b.unsqueeze(-1), inter, pad_n).squeeze(-1)",
            1,
        )
        path.write_text(src)
        print(f"[ok] Patched {path}")
    else:
        raise RuntimeError(f"Expected anchor not found in {path}")


if __name__ == "__main__":
    patch_gpt_oss()
    patch_unquant()
