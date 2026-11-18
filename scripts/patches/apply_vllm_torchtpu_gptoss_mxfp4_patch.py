#!/usr/bin/env python3
"""Load-time patch that lets `vllm-torchtpu` serve the native MXFP4 checkpoint `openai/gpt-oss-120b` on TPU v7x.

Run it once inside the vllm-torchtpu container before `vllm serve` (it edits the installed packages in place and is
safe to run twice). It also adds embedding sharing for the DFlash draft model. Fixes four upstream blockers that caused `openai/gpt-oss-120b` (`60.8 GiB` MXFP4) to fail at startup
with `NotImplementedError: gpt_oss_mxfp4 quantization method not supported on TPU`:
  1. `GptOssMxfp4Config.override_quantization_method` ignored `user_quant` (`"tpu-mxfp4"`), rewriting
     the TPU config name back to `"gpt_oss_mxfp4"`.
  2. `vllm_torchtpu.layers.adapter.quantization.get_tpu_quantization_config` only mapped `"mxfp4"`,
     missing `"gpt_oss_mxfp4"`.
  3. `VllmMxfp4MoEMethod` omitted `self.weight_dtype = "gpt_oss_mxfp4"`, causing `gpt_oss.py`'s
     `_get_moe_weight_dtype()` to return `None` and skip expert weight loading, and did not pad
     `hidden_size = 2880 -> 2944` (`128`-tile alignment for Pallas `gmm_v2`) or pack into native
     4-bit K-major FP4 (`quantize_native_fp4_kmajor`, shrinking weights ~4x vs. BF16 so `TP=2, DP=4`
     fits 4 replicas per 4-chip TPU v7x host).
  4. `PallasAttentionBackendImpl` `head_size=64` (`hd64`) KV pool dimension alignment when `kv_cache.shape[-1] == 128`.
"""

from __future__ import annotations

from pathlib import Path
import vllm.model_executor.layers.quantization.mxfp4 as vllm_mxfp4
import vllm_torchtpu.layers.adapter.attention as tpu_attn
import vllm_torchtpu.layers.adapter.quantization as tpu_quant_init
import vllm_torchtpu.layers.adapter.quantization.mxfp4 as tpu_mxfp4
import vllm_torchtpu.platforms.tpu_platform as tpu_platform_mod


def patch_tpu_platform_supported_quantization() -> None:
    path = Path(tpu_platform_mod.__file__)
    src = path.read_text()
    if '"gpt_oss_mxfp4",' in src:
        print(f"[skip] {path} already patched")
        return
    old = '        "mxfp4",'
    new = '        "mxfp4",\n        "gpt_oss_mxfp4",'
    if old not in src:
        raise RuntimeError(f"supported_quantization mxfp4 anchor not found in {path}")
    path.write_text(src.replace(old, new, 1))
    print(f"[ok] Patched {path}")


def patch_upstream_gpt_oss_mxfp4_override() -> None:
    path = Path(vllm_mxfp4.__file__)
    src = path.read_text()
    guard = 'if user_quant is not None and str(user_quant).startswith("tpu-"):'
    if guard in src:
        print(f"[skip] {path} already patched")
        return
    old = """    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:"""
    new = f"""    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        {guard}
            return None"""
    if old not in src:
        raise RuntimeError(f"override_quantization_method anchor not found in {path}")
    path.write_text(src.replace(old, new, 1))
    print(f"[ok] Patched {path}")


def patch_tpu_quant_registry() -> None:
    path = Path(tpu_quant_init.__file__)
    src = path.read_text()
    if '"gpt_oss_mxfp4": VllmMxfp4Config,' in src:
        print(f"[skip] {path} already patched")
        return
    old = "quant_methods.MXFP4: VllmMxfp4Config,"
    new = 'quant_methods.MXFP4: VllmMxfp4Config,\n        "gpt_oss_mxfp4": VllmMxfp4Config,'
    if old not in src:
        raise RuntimeError(f"MXFP4 registry anchor not found in {path}")
    path.write_text(src.replace(old, new, 1))
    print(f"[ok] Patched {path}")


def patch_tpu_mxfp4_moe_native_fp4_and_padding() -> None:
    path = Path(tpu_mxfp4.__file__)
    src = path.read_text()
    marker_v3 = "# [TPU patch v3] GPT-OSS MXFP4 Native FP4 + 512 SparseCore/Pallas Alignment"
    marker_v2 = "# [TPU patch v2] GPT-OSS MXFP4 Native FP4 + 512 SparseCore/Pallas Alignment"
    marker_v1 = "# [TPU patch v1] GPT-OSS MXFP4 Native FP4 + 128-Tile Alignment"
    if marker_v3 in src:
        print(f"[skip] {path} already patched (v3)")
        return

    old_fwd_kwargs = """        kwargs = {
            "hidden_states": x,
            "w1": layer.w13_weight,
            "w2": layer.w2_weight,
            "w1_scale": layer.w13_weight_scale,
            "w2_scale": layer.w2_weight_scale,
            "w1_bias": getattr(layer, "w13_bias", None),
            "w2_bias": getattr(layer, "w2_bias", None),
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "experts_start": layer._experts_start,
            "topk": layer.moe_config.experts_per_token,
            "activation": activation_str,
            "rhs_quant_dtype": self.rhs_quant_dtype,
        }
        if enable_pipelined_collective_and_compute():
            return pipelined_fused_moe_gmm(**kwargs)

        return fused_moe_gmm(**kwargs)"""

    new_fwd_kwargs = """        pad_h = getattr(self, "_pad_hidden", 0)
        x_gmm = torch.nn.functional.pad(x, (0, pad_h)) if pad_h > 0 else x
        kwargs = {
            "hidden_states": x_gmm,
            "w1": layer.w13_weight,
            "w2": layer.w2_weight,
            "w1_scale": layer.w13_weight_scale,
            "w2_scale": layer.w2_weight_scale,
            "w1_bias": getattr(layer, "w13_bias", None),
            "w2_bias": getattr(layer, "w2_bias", None),
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "experts_start": layer._experts_start,
            "topk": layer.moe_config.experts_per_token,
            "activation": activation_str,
            "rhs_quant_dtype": self.rhs_quant_dtype,
        }
        if enable_pipelined_collective_and_compute():
            out = pipelined_fused_moe_gmm(**kwargs)
        else:
            out = fused_moe_gmm(**kwargs)
        return out[..., : self._orig_hidden] if pad_h > 0 else out"""

    if marker_v2 in src or marker_v1 in src:
        src = src.replace(marker_v2, marker_v3, 1).replace(marker_v1, marker_v3, 1)
        src = src.replace(
            "        pad_hidden = (-orig_hidden) % block_sz\n        pad_inter = (-orig_inter) % block_sz",
            "        pad_hidden = (-orig_hidden) % 512\n        pad_inter = (-orig_inter) % block_sz\n        self._orig_hidden = orig_hidden\n        self._pad_hidden = pad_hidden",
        )
        if old_fwd_kwargs not in src:
            raise RuntimeError(f"old_fwd_kwargs anchor not found in {path}")
        src = src.replace(old_fwd_kwargs, new_fwd_kwargs, 1)
        path.write_text(src)
        print(f"[ok] Upgraded {path} to v3 (512-aligned + _forward_monolithic_tpu pad/slice)")
        return

    marker = marker_v3

    old_import = "from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod"
    new_import = (
        "from vllm.model_executor.layers.quantization.mxfp4 import (\n"
        "    GptOssMxfp4MoEMethod,\n"
        "    Mxfp4Config,\n"
        "    Mxfp4MoEMethod,\n"
        ")"
    )
    if old_import not in src:
        raise RuntimeError(f"import anchor not found in {path}")
    src = src.replace(old_import, new_import, 1)

    old_cls = """class VllmMxfp4MoEMethod(TpuMoEActivationMixin, Mxfp4MoEMethod):"""
    new_cls = """class VllmMxfp4MoEMethod(TpuMoEActivationMixin, GptOssMxfp4MoEMethod, Mxfp4MoEMethod):"""
    if old_cls not in src:
        raise RuntimeError(f"VllmMxfp4MoEMethod class anchor not found in {path}")
    src = src.replace(old_cls, new_cls, 1)

    old_init = """    def __init__(self, moe: FusedMoEConfig):
        # Skip Mxfp4MoEMethod.__init__ (GPU backend assertion)
        FusedMoEMethodBase.__init__(self, moe)
        self.mxfp4_backend = Mxfp4Backend.TRITON
        self.apply_monolithic = self._select_monolithic()"""
    new_init = f"""    {marker}
    def __init__(self, moe: FusedMoEConfig):
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        self.mxfp4_backend = Mxfp4Backend.TRITON
        self.rhs_quant_dtype = jnp.float4_e2m1fn
        self.apply_monolithic = self._select_monolithic()

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config,
    ) -> tuple[int, int]:
        return hidden_size, intermediate_size_per_partition"""
    if old_init not in src:
        raise RuntimeError(f"__init__ anchor not found in {path}")
    src = src.replace(old_init, new_init, 1)

    old_proc = """    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, RoutedExperts)
        layer = cast("TpuRoutedExperts", layer)
        assert layer.moe_config.has_bias, "MXFP4 quantization requires bias."

        activation_str = self._set_tpu_activation(layer)

        w13_weight = dequantize_mxfp4_packed(
            layer.w13_weight.data.to("cpu"),
            layer.w13_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w13_weight.device)
        w2_weight = dequantize_mxfp4_packed(
            layer.w2_weight.data.to("cpu"),
            layer.w2_weight_scale.data.to("cpu"),
            axis=-1,
            out_dtype=torch.bfloat16,
        ).to(layer.w2_weight.device)

        w13_interleave = activation_str == "swigluoai"
        if w13_interleave:
            w1_weight = w13_weight[:, ::2, :]
            w3_weight = w13_weight[:, 1::2, :]
            w13_weight = torch.cat([w1_weight, w3_weight], dim=1)

        w13_weight = w13_weight.transpose(1, 2).contiguous()
        w2_weight = w2_weight.transpose(1, 2).contiguous()

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        # Same as `= None`, which torch's stubs reject.
        layer.register_parameter("w13_weight_scale", None)
        layer.register_parameter("w2_weight_scale", None)

        if layer.w13_bias is not None:
            w13_bias = layer.w13_bias.data
            if w13_interleave:
                w1_bias = w13_bias[:, ::2]
                w3_bias = w13_bias[:, 1::2]
                w13_bias = torch.cat([w1_bias, w3_bias], dim=1)

            layer.w13_bias = torch.nn.Parameter(
                w13_bias.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )
        if layer.w2_bias is not None:
            layer.w2_bias = torch.nn.Parameter(
                layer.w2_bias.data.unsqueeze(1).to(torch.float32),
                requires_grad=False,
            )

        if layer.w13_weight.device.type == "tpu":
            to_sync = [layer.w13_weight, layer.w2_weight]
            if layer.w13_bias is not None:
                to_sync.append(layer.w13_bias)
            if layer.w2_bias is not None:
                to_sync.append(layer.w2_bias)
            synchronize_tensors(to_sync)

        logger.info_once(
            "MXFP4 weights dequantized to bfloat16 and transposed for GMM kernel."
        )
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(layer, device=layer.w13_weight.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )"""

    new_proc = """    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, RoutedExperts)
        layer = cast("TpuRoutedExperts", layer)
        device = layer.w13_weight.device
        activation_str = self._set_tpu_activation(layer)

        w13_deq = _dequantize_mxfp4_packed_on_device(
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
            axis=-1,
            out_dtype=torch.bfloat16,
        )
        w2_deq = _dequantize_mxfp4_packed_on_device(
            layer.w2_weight.data,
            layer.w2_weight_scale.data,
            axis=-1,
            out_dtype=torch.bfloat16,
        )

        w13_interleave = activation_str == "swigluoai"
        if w13_interleave:
            w1_part = w13_deq[:, ::2, :]
            w3_part = w13_deq[:, 1::2, :]
        else:
            half = w13_deq.shape[1] // 2
            w1_part = w13_deq[:, :half, :]
            w3_part = w13_deq[:, half:, :]

        # Transpose to [E, K, N] and pad both hidden (2880 -> 2944) and intermediate to 128-tile boundary
        w1_t = w1_part.transpose(1, 2).contiguous()
        w3_t = w3_part.transpose(1, 2).contiguous()
        w2_t = w2_deq.transpose(1, 2).contiguous()
        del w13_deq, w2_deq, w1_part, w3_part

        orig_hidden = w1_t.shape[1]
        orig_inter = w1_t.shape[2]
        block_sz = 128
        pad_hidden = (-orig_hidden) % 512
        pad_inter = (-orig_inter) % block_sz
        self._orig_hidden = orig_hidden
        self._pad_hidden = pad_hidden

        if pad_hidden > 0 or pad_inter > 0:
            w1_t = torch.nn.functional.pad(w1_t, (0, pad_inter, 0, pad_hidden))
            w3_t = torch.nn.functional.pad(w3_t, (0, pad_inter, 0, pad_hidden))
            w2_t = torch.nn.functional.pad(w2_t, (0, pad_hidden, 0, pad_inter))

        w13_padded = torch.cat([w1_t, w3_t], dim=2).contiguous()
        w2_padded = w2_t.contiguous()
        del w1_t, w3_t, w2_t

        # Quantize into native 4-bit K-major FP4 (`W4A16` Pallas GMM) so 120B fits in ~31 GiB/chip at TP=2!
        w13_fp4, w13_sc = quantize_native_fp4_kmajor(w13_padded.to(device), block=block_sz, pack=True)
        w2_fp4, w2_sc = quantize_native_fp4_kmajor(w2_padded.to(device), block=block_sz, pack=True)
        del w13_padded, w2_padded

        w13_b_orig = layer.w13_bias.data if getattr(layer, "w13_bias", None) is not None else None
        w2_b_orig = layer.w2_bias.data if getattr(layer, "w2_bias", None) is not None else None
        del layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale

        layer.w13_weight = torch.nn.Parameter(w13_fp4, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_fp4, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_sc, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_sc, requires_grad=False)

        if w13_b_orig is not None:
            if w13_interleave:
                b1 = w13_b_orig[:, ::2]
                b3 = w13_b_orig[:, 1::2]
            else:
                b1 = w13_b_orig[:, :orig_inter]
                b3 = w13_b_orig[:, orig_inter:]
            if pad_inter > 0:
                b1 = torch.nn.functional.pad(b1, (0, pad_inter))
                b3 = torch.nn.functional.pad(b3, (0, pad_inter))
            w13_b_pad = torch.cat([b1, b3], dim=1).unsqueeze(1).to(device=device, dtype=torch.float32)
            layer.w13_bias = torch.nn.Parameter(w13_b_pad, requires_grad=False)

        if w2_b_orig is not None:
            if pad_hidden > 0:
                w2_b_orig = torch.nn.functional.pad(w2_b_orig, (0, pad_hidden))
            layer.w2_bias = torch.nn.Parameter(
                w2_b_orig.unsqueeze(1).to(device=device, dtype=torch.float32),
                requires_grad=False,
            )

        if device.type == "tpu":
            to_sync = [layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale]
            if getattr(layer, "w13_bias", None) is not None:
                to_sync.append(layer.w13_bias)
            if getattr(layer, "w2_bias", None) is not None:
                to_sync.append(layer.w2_bias)
            synchronize_tensors(to_sync)

        logger.info_once(
            "GPT-OSS MXFP4 weights padded (H=%d->%d, I=%d->%d) and packed as native FP4 K-major (block=%d).",
            orig_hidden,
            orig_hidden + pad_hidden,
            orig_inter,
            orig_inter + pad_inter,
            block_sz,
        )
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(layer, device=device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )"""

    if old_proc not in src:
        raise RuntimeError(f"process_weights_after_loading anchor not found in {path}")
    src = src.replace(old_proc, new_proc, 1)
    if old_fwd_kwargs not in src:
        raise RuntimeError(f"old_fwd_kwargs anchor not found in {path}")
    src = src.replace(old_fwd_kwargs, new_fwd_kwargs, 1)
    path.write_text(src)
    print(f"[ok] Patched {path} (v3)")


def patch_tpu_hd64_attention() -> None:
    path = Path(tpu_attn.__file__)
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
    if old in src:
        path.write_text(src.replace(old, new, 1))
        print(f"[ok] Patched {path}")
    else:
        print(f"[info] hd64 guard anchor not needed or already handled in {path}")


def patch_tpu_moe_expert_staging_none_id() -> None:
    import vllm.model_executor.layers.fused_moe.routed_experts as re_mod
    import vllm_torchtpu.model_loader_patches as mlp

    path = Path(mlp.__file__)
    src = path.read_text()
    guard = "or expert_id is None or self._map_global_expert_id_to_local_expert_id(expert_id) == -1"
    if guard not in src:
        old = "or self._map_global_expert_id_to_local_expert_id(expert_id) == -1"
        if old not in src:
            raise RuntimeError(f"expert_id staging anchor not found in {path}")
        path.write_text(src.replace(old, guard, 1))
        print(f"[ok] Patched {path}")

    re_path = Path(re_mod.__file__)
    re_src = re_path.read_text()
    re_guard = 'if quant_config_name in ("gpt_oss_mxfp4", "mxfp4") and expert_id is None:'
    if re_guard not in re_src:
        re_old = 'if quant_config_name == "gpt_oss_mxfp4":'
        if re_old not in re_src:
            raise RuntimeError(f"gpt_oss_mxfp4 combined expert anchor not found in {re_path}")
        re_path.write_text(re_src.replace(re_old, re_guard, 1))
        print(f"[ok] Patched {re_path}")


def patch_tpu_dflash_embedding_sharing() -> None:
    import vllm_torchtpu.spec_decode.utils as sd_utils

    path = Path(sd_utils.__file__)
    src = path.read_text()
    guard = 'getattr(target_lm_model, "embedding", None)'
    if guard in src:
        print(f"[skip] {path} already patched")
        return
    old = """    target_embed = (
        getattr(target_lm_model, "embed_tokens", None)
        if target_lm_model is not None
        else None
    )"""
    new = """    target_embed = (
        (
            getattr(target_lm_model, "embed_tokens", None)
            or getattr(target_lm_model, "embedding", None)
        )
        if target_lm_model is not None
        else None
    )"""
    if old not in src:
        raise RuntimeError(f"maybe_share_embeddings anchor not found in {path}")
    path.write_text(src.replace(old, new, 1))
    print(f"[ok] Patched {path} (GptOssModel.embedding -> DFlash embed_tokens sharing)")


def apply_all() -> None:
    patch_tpu_platform_supported_quantization()
    patch_upstream_gpt_oss_mxfp4_override()
    patch_tpu_quant_registry()
    patch_tpu_mxfp4_moe_native_fp4_and_padding()
    patch_tpu_hd64_attention()
    patch_tpu_moe_expert_staging_none_id()
    patch_tpu_dflash_embedding_sharing()
    print("[DONE] All GPT-OSS MXFP4 + DFlash vllm-torchtpu patches applied.")


if __name__ == "__main__":
    apply_all()
