# Working log: GPT-OSS 120B on Cloud TPU v7x (and v5p)

This is a plain-language record of what we tried, in order: what failed and why, and what worked in the end. The measured results are in [`README.md`](./README.md), and the full tables, exact flags and error details are in [`APPENDIX.md`](./APPENDIX.md).

## The goal

Serve GPT-OSS 120B (a Mixture-of-Experts model with 117B total and 5.1B active parameters) on Cloud TPU v7x with three serving stacks: `sglang-torchtpu`, `vllm-torchtpu` and JAX `vllm-tpu`. We wanted to find the fastest setup for:

- single requests with long prompts (22k tokens in, 4k out);
- many requests at once (1k in, 1k out);
- one, two and four hosts (4, 8 and 16 chips).

We used two versions of the weights: the original MXFP4 checkpoint (65.3 GB, experts in 4-bit) and an unquantized BF16 conversion (233.7 GB).

## Step by step

### Step 1: BF16 weights on one host, three stacks

- **`vllm-torchtpu` gave wrong output at first.** The model's hidden size (2880) is not a multiple of the 128-wide tiles the TPU kernels read, so the reads were misaligned and the model produced `!` tokens. A load-time patch pads the hidden size to 2944, after which the output was correct (appendix section 1.B).
- **`sglang-torchtpu` needed a loader fix.** GPT-OSS uses a custom gated activation (`swigluoai`) whose two halves are interleaved in the expert weights. The BF16 loader had to de-interleave them and align them to 128-wide tiles (appendix section 1.C).
- **Result:** `sglang-torchtpu` (`TP=8, EP=8`) gave the lowest latency on one long prompt: 4.97 ms per output token and 739.5 ms to the first token with 22k in / 4k out. That is 1.72× faster per token than the JAX `vllm-tpu` BF16 baseline (8.54 ms).
- **EAGLE3 speculative decoding did not pay off.** It cut 1k/1k throughput by 18.6% compared with the same sglang setup without it.

### Step 2: The MXFP4 checkpoint

- **It worked on `sglang-torchtpu` and JAX `vllm-tpu`, but not on `vllm-torchtpu`**, which had no TPU support for this weight format at the time (appendix section 1.A).
- **JAX `vllm-tpu` with 4 replicas per host (`TP=2, DP=4`)** reached 5,847.0 output tokens/s at 64 requests on one host, 2.37× the one-host BF16 JAX result.
- **More hosts:** 14,277.1 tokens/s on 2 hosts and 20,366.9 on 4 hosts (3.48× one host).
- **The long-prompt latency target was not met.** The target was at most 500 ms to the first token for a 22k-token prompt (appendix section 2.C). The lowest we measured was 542.5 ms.

### Step 3: Push one host much harder (section C of the README)

- **`vllm-torchtpu` learned to serve MXFP4.** A new load-time patch registers the MXFP4 format for TPU, packs the experts into native 4-bit, pads the hidden size and fixes the attention cache layout. All sanity prompts then returned the right answer (37 × 43 = 1591).
- **Higher request limits.** With more running requests allowed per replica:
  - JAX `vllm-tpu` gave 22,163.6 tokens/s at 1,024 requests and 22,952.2 at 2,048.
  - `vllm-torchtpu` kept scaling: 24,357.1 at 4,096 requests and 26,738.2 at 8,192. A second run gave 26,626.0.
- **`sglang-torchtpu` lessons:**
  - `torch.compile` is essential. Without it, one request ran at 8.4 tokens/s instead of 190.0.
  - Splitting attention into groups (data-parallel attention) raised throughput under load.
  - 16,384-token prefill chunks helped under heavy load: 16,246.4 tokens/s at 1,024 requests.
- **DFlash speculative decoding** made single requests faster (3.22 ms per token vs. 5.23 ms without it) but slowed things down under load. With 4 attention groups at 128 requests it gave 5,132.3 tokens/s vs. 6,103.0 without it.
- **Long prompts at higher load:** JAX `vllm-tpu` gave 5,648.1 tokens/s at 512 requests with 22k in / 4k out. A second run gave 5,640.1.
- **Things that broke along the way:**
  - One replica per TensorCore (`TP=1, DP=8`) is not supported by this `vllm-torchtpu` build ("No TPU topology found for count: 1").
  - DFlash with `TP=2, DP=4` on `vllm-torchtpu` failed at startup, and the saved error output did not show why.
  - The first try of one `vllm-torchtpu` run hit vLLM's default 600 s startup limit while it was still reading weights from Cloud Storage. Raising the limit to 3,600 s fixed it.
  - JAX on a node from a multi-host pool failed with "Error computing device indices for TPU_VISIBLE_CHIPS". Unsetting that variable fixed it.
  - Kubernetes evicted one sglang pod for low disk space: its compile cache had grown to 66 GiB. We now clear that cache between runs.

### Step 4: Compare TPU v5p with v7x, one host each

- **`sglang-torchtpu` ran on v5p unchanged**, apart from the tensor-parallel size (4 instead of 8).
- **JAX `vllm-tpu` needed three changes on v5p:**
  1. Its MoE kernel for BF16 × FP4 does not compile for v5p, so the experts are converted to INT8 when the model loads.
  2. The conversion block size had to be 256 to split evenly over 4 chips.
  3. Data parallelism has to run in a single process.
- **Results, per host (4 chips each):**
  - With the same setup on both, v7x gave 1.11× to 1.16× v5p's throughput for one request, and 1.74× to 2.30× at 32 to 256 requests.
  - With the best setup on each host, the gap was 2.85× to 3.50× at 512 to 4,096 requests.
  - The v5p host's highest result was 8,061.2 tokens/s at 2,048 requests.

## What worked and what didn't

| Didn't work | Worked |
| :--- | :--- |
| `vllm-torchtpu` BF16 without padding (`!` output) | Padding the hidden size from 2880 to 2944 |
| MXFP4 on `vllm-torchtpu` out of the box | A load-time MXFP4 patch |
| EAGLE3, and DFlash under load | DFlash for single requests |
| `sglang-torchtpu` without `torch.compile` | `torch.compile` with data-parallel attention and larger prefill chunks |
| One replica per TensorCore on `vllm-torchtpu` | 4 replicas of 2 TensorCores (`TP=2, DP=4`) |
| FP4 expert kernel on v5p | INT8 experts on v5p |

## Results worth knowing

- One v7x host (4 chips) served 26,738.2 output tokens/s at 8,192 requests with `vllm-torchtpu` and the MXFP4 patch (26,626.0 in a second run).
- Four v7x hosts served 20,366.9 tokens/s with JAX `vllm-tpu` (3.48× one host).
- For one long prompt, `sglang-torchtpu` with BF16 weights ran at 4.97 ms per output token, 1.72× faster than the JAX BF16 baseline.
- With the best setup on each, a v7x host gave 2.85× to 3.50× the throughput of a v5p host at 512 to 4,096 requests.

## Not tried or still open

- The 500 ms first-token target for a 22k-token prompt is still not met (best 542.5 ms).
- We have no usable profiler traces for these runs.
- Most points were measured once. Second runs of the main high-load setups came within 4.2% of the first runs, but two runs of one setup at 128 requests gave 5,313.7 and 5,806.9 tokens/s, so differences under about 10% should not be read as real.
