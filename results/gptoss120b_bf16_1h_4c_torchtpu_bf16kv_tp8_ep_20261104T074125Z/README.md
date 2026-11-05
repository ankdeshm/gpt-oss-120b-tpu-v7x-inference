# `gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T074125Z` — invalid output; not a performance baseline

`vllm-torchtpu` serving `lmsys/gpt-oss-120b-bf16` (BF16 KV, `TP=8, DP=1, EP=true`) **without** the load-time BF16 patch (`scripts/patches/apply_vllm_torchtpu_bf16_patches.py`), with the same server arguments as `gptoss120b_bf16_1h_4c_torchtpu_bf16kv_tp8_ep_20261104T100207Z`.

- The server leaves the MoE hidden dimension unpadded: `w13=[16, 2880, 5888]` (patched run: `[16, 2944, 5888]`); see `server_log_excerpt.txt`.
- Every sanity prompt returned only `!` tokens (`sanity.json`, `summary.json`).

This record documents the failure mode only. Its latency and throughput fields describe a configuration that produces invalid output and are not a performance baseline.
