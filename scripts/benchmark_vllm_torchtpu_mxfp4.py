#!/usr/bin/env python3
"""Single-host GPT-OSS 120B (native MXFP4) serving benchmark on Cloud TPU v7x with vllm-torchtpu.

`vllm-torchtpu` cannot load the MXFP4 checkpoint out of the box, so this script first applies
`patches/apply_vllm_torchtpu_gptoss_mxfp4_patch.py` (load-time edits to the installed packages, safe to run twice).
For each named setup it then starts `vllm serve` on the local 4-chip host, waits until it is healthy, sends two sanity
prompts and runs `vllm bench serve` (random fixed-length prompts, EOS ignored) at several loads. Results go to one JSON
file (one entry per setup). The server flags below are exactly the ones used for the results in this repo.

Usage (inside a vllm-torchtpu pod on one v7x host):
    MODEL_PATH=/models/openai/gpt-oss-120b DFLASH_PATH=/models/z-lab/gpt-oss-120b-DFlash \\
        python3 benchmark_vllm_torchtpu_mxfp4.py --setups vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "patches"))
import apply_vllm_torchtpu_gptoss_mxfp4_patch as mxfp4_patch  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/openai/gpt-oss-120b")
DFLASH_PATH = os.environ.get("DFLASH_PATH", "/models/z-lab/gpt-oss-120b-DFlash")
MODEL_NAME = "openai/gpt-oss-120b"
PORT = 8000
MAX_MODEL_LEN = 26624
OUT_DIR = Path(os.environ.get("OUT_DIR", "./results"))

POINTS = [
    {"label": "1k_1k_c1", "isl": 1024, "osl": 1024, "conc": 1, "prompts": 12, "warmups": 2},
    {"label": "1k_1k_c8", "isl": 1024, "osl": 1024, "conc": 8, "prompts": 32, "warmups": 4},
    {"label": "1k_1k_c32", "isl": 1024, "osl": 1024, "conc": 32, "prompts": 96, "warmups": 8},
    {"label": "1k_1k_c64", "isl": 1024, "osl": 1024, "conc": 64, "prompts": 128, "warmups": 8},
    {"label": "1k_1k_c128", "isl": 1024, "osl": 1024, "conc": 128, "prompts": 256, "warmups": 12},
    {"label": "8k_1k_c1", "isl": 8192, "osl": 1024, "conc": 1, "prompts": 8, "warmups": 2},
    {"label": "8k_1k_c32", "isl": 8192, "osl": 1024, "conc": 32, "prompts": 64, "warmups": 6},
    {"label": "8k_1k_c64", "isl": 8192, "osl": 1024, "conc": 64, "prompts": 128, "warmups": 8},
    {"label": "22k_4k_c1", "isl": 22000, "osl": 4000, "conc": 1, "prompts": 6, "warmups": 1, "goodput": ["ttft:500", "tpot:10", "e2el:60000"]},
    {"label": "22k_4k_c8", "isl": 22000, "osl": 4000, "conc": 8, "prompts": 16, "warmups": 2, "goodput": ["ttft:500", "tpot:10", "e2el:60000"]},
    {"label": "22k_4k_c32", "isl": 22000, "osl": 4000, "conc": 32, "prompts": 48, "warmups": 4, "goodput": ["ttft:500", "tpot:10", "e2el:60000"]},
]

ARMS = [
    {
        "name": "vllm_torchtpu_mxfp4_tp8_dp1",
        "tp": 8,
        "dp": 1,
        "max_num_seqs": 128,
        "max_batched_tokens": 16384,
        "extra_flags": ["--enable-expert-parallel"],
    },
    {
        "name": "vllm_torchtpu_mxfp4_tp4_dp2_async",
        "tp": 4,
        "dp": 2,
        "max_num_seqs": 64,
        "max_batched_tokens": 16384,
        "extra_flags": ["--enable-expert-parallel", "--async-scheduling"],
    },
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 64,
        "max_batched_tokens": 8192,
        "extra_flags": ["--async-scheduling"],
    },
    {
        "name": "vllm_torchtpu_mxfp4_tp8_dflash8",
        "tp": 8,
        "dp": 1,
        "max_num_seqs": 32,
        "max_batched_tokens": 8192,
        "gpu_mem_util": "0.80",
        "extra_flags": [
            "--enable-expert-parallel",
            "--speculative-config",
            json.dumps({"method": "dflash", "model": DFLASH_PATH, "num_speculative_tokens": 8}, separators=(",", ":")),
        ],
    },
    {
        "name": "vllm_torchtpu_mxfp4_tp8_dflash16",
        "tp": 8,
        "dp": 1,
        "max_num_seqs": 32,
        "max_batched_tokens": 8192,
        "gpu_mem_util": "0.80",
        "extra_flags": [
            "--enable-expert-parallel",
            "--speculative-config",
            json.dumps({"method": "dflash", "model": DFLASH_PATH, "num_speculative_tokens": 16}, separators=(",", ":")),
        ],
    },
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_dflash8",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 32,
        "max_batched_tokens": 8192,
        "gpu_mem_util": "0.80",
        "extra_flags": [
            "--async-scheduling",
            "--speculative-config",
            json.dumps({"method": "dflash", "model": DFLASH_PATH, "num_speculative_tokens": 8}, separators=(",", ":")),
        ],
    },
    # The TP=2, DP=4 layout with 128 running requests per replica (512 in total), measured up to 512 requests.
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async_seqs128",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 128,
        "max_batched_tokens": 8192,
        "extra_flags": ["--async-scheduling"],
        "points_override": [
            {"label": "1k_1k_c64", "isl": 1024, "osl": 1024, "conc": 64, "prompts": 128, "warmups": 8},
            {"label": "1k_1k_c128", "isl": 1024, "osl": 1024, "conc": 128, "prompts": 256, "warmups": 12},
            {"label": "1k_1k_c256", "isl": 1024, "osl": 1024, "conc": 256, "prompts": 512, "warmups": 16},
            {"label": "1k_1k_c512", "isl": 1024, "osl": 1024, "conc": 512, "prompts": 1024, "warmups": 16},
        ],
    },
    # The same layout with 256 running requests per replica (1,024 in total), measured up to 1,024 requests.
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 256,
        "max_batched_tokens": 8192,
        "extra_flags": ["--async-scheduling"],
        "points_override": [
            {"label": "1k_1k_c256", "isl": 1024, "osl": 1024, "conc": 256, "prompts": 512, "warmups": 16},
            {"label": "1k_1k_c512", "isl": 1024, "osl": 1024, "conc": 512, "prompts": 1024, "warmups": 16},
            {"label": "1k_1k_c1024", "isl": 1024, "osl": 1024, "conc": 1024, "prompts": 2048, "warmups": 32},
        ],
    },
    # The same layout with 512 running requests per replica (2,048 in total), measured at 1,024 and 2,048 requests.
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 512,
        "max_batched_tokens": 8192,
        "extra_flags": ["--async-scheduling"],
        "points_override": [
            {"label": "1k_1k_c1024", "isl": 1024, "osl": 1024, "conc": 1024, "prompts": 2048, "warmups": 32},
            {"label": "1k_1k_c2048", "isl": 1024, "osl": 1024, "conc": 2048, "prompts": 4096, "warmups": 64},
        ],
    },
    # The same layout with 1,024 running requests per replica (4,096 in total), measured at 2,048 and 4,096 requests.
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async_seqs1024",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 1024,
        "max_batched_tokens": 8192,
        "extra_flags": ["--async-scheduling"],
        "points_override": [
            {"label": "1k_1k_c2048", "isl": 1024, "osl": 1024, "conc": 2048, "prompts": 4096, "warmups": 64},
            {"label": "1k_1k_c4096", "isl": 1024, "osl": 1024, "conc": 4096, "prompts": 8192, "warmups": 128},
        ],
    },
    # The 512-per-replica setup with 16,384 batched tokens per step (the JAX vllm-tpu setting) instead of 8,192.
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512_bt16384",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 512,
        "max_batched_tokens": 16384,
        "extra_flags": ["--async-scheduling"],
        "points_override": [
            {"label": "1k_1k_c1024", "isl": 1024, "osl": 1024, "conc": 1024, "prompts": 2048, "warmups": 32},
            {"label": "1k_1k_c2048", "isl": 1024, "osl": 1024, "conc": 2048, "prompts": 4096, "warmups": 64},
        ],
    },
    # Long prompts at higher load: 8k in / 1k out with 256 running requests per replica (1,024 in total).
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async_seqs256_8k",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 256,
        "max_batched_tokens": 8192,
        "extra_flags": ["--async-scheduling"],
        "points_override": [
            {"label": "8k_1k_c64", "isl": 8192, "osl": 1024, "conc": 64, "prompts": 128, "warmups": 8},
            {"label": "8k_1k_c128", "isl": 8192, "osl": 1024, "conc": 128, "prompts": 256, "warmups": 8},
            {"label": "8k_1k_c256", "isl": 8192, "osl": 1024, "conc": 256, "prompts": 512, "warmups": 16},
            {"label": "8k_1k_c512", "isl": 8192, "osl": 1024, "conc": 512, "prompts": 1024, "warmups": 16},
        ],
    },
    # 8k prompts were still rising at c=512 (5,971 tok/s): 512 running requests per replica, measured at c=1024.
    {
        "name": "vllm_torchtpu_mxfp4_tp2_dp4_async_seqs512_8k",
        "tp": 2,
        "dp": 4,
        "max_num_seqs": 512,
        "max_batched_tokens": 8192,
        "extra_flags": ["--async-scheduling"],
        "points_override": [
            {"label": "8k_1k_c1024", "isl": 8192, "osl": 1024, "conc": 1024, "prompts": 2048, "warmups": 32},
        ],
    },
]


def cleanup_vllm() -> None:
    me = os.getpid()
    pats = ("vllm serve", "vllm.entrypoints", "EngineCore", "Worker_TP", "DP_Coordinator", "VLLM::")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit() or int(pid_str) in (1, me):
                continue
            try:
                cmd = open(f"/proc/{pid_str}/cmdline", "rb").read().decode("utf-8", "ignore").replace("\x00", " ")
                if any(p in cmd for p in pats):
                    os.kill(int(pid_str), sig)
            except Exception:
                pass
        time.sleep(4)
    if os.path.exists("/tmp/libtpu_lockfile"):
        os.remove("/tmp/libtpu_lockfile")


def wait_health(port: int, proc: subprocess.Popen[Any], log_path: Path, timeout_s: int = 1800) -> float:
    t0 = time.perf_counter()
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else ""
            raise RuntimeError(f"vLLM exited early ({proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return round(time.perf_counter() - t0, 2)
        except Exception:
            pass
        time.sleep(4)
    tail = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else ""
    raise TimeoutError(f"Timed out after {timeout_s}s:\n{tail}")


def run_bench_point(pt: dict[str, Any], bench_dir: Path) -> dict[str, Any]:
    res_file = bench_dir / f"{pt['label']}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--model", MODEL_PATH,
        "--served-model-name", MODEL_NAME,
        "--dataset-name", "random",
        "--random-input-len", str(pt["isl"]),
        "--random-output-len", str(pt["osl"]),
        "--ignore-eos",
        "--max-concurrency", str(pt["conc"]),
        "--num-prompts", str(pt["prompts"]),
        "--num-warmups", str(pt["warmups"]),
        "--save-result",
        "--result-dir", str(bench_dir),
        "--result-filename", res_file.name,
    ]
    if pt.get("goodput"):
        cmd += ["--goodput"] + list(pt["goodput"])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not res_file.exists():
        return {"error": (proc.stderr or proc.stdout)[-1500:]}
    d = json.loads(res_file.read_text())
    return {
        "isl": pt["isl"],
        "osl": pt["osl"],
        "concurrency": pt["conc"],
        "completed": d.get("completed"),
        "failed": d.get("failed", 0),
        "request_throughput_req_s": round(float(d.get("request_throughput", 0.0)), 3),
        "request_goodput_req_s": round(float(d.get("request_goodput", 0.0) or 0.0), 3),
        "output_throughput_tok_s": round(float(d.get("output_throughput", 0.0)), 2),
        "total_token_throughput_tok_s": round(float(d.get("total_token_throughput", 0.0)), 2),
        "median_ttft_ms": round(float(d.get("median_ttft_ms", 0.0)), 2),
        "p95_ttft_ms": round(float(d.get("p95_ttft_ms", d.get("p99_ttft_ms", 0.0))), 2),
        "median_tpot_ms": round(float(d.get("median_tpot_ms", 0.0)), 2),
        "p95_tpot_ms": round(float(d.get("p95_tpot_ms", d.get("p99_tpot_ms", 0.0))), 2),
        "median_itl_ms": round(float(d.get("median_itl_ms", 0.0)), 2),
        "median_e2el_ms": round(float(d.get("median_e2el_ms", 0.0)), 2),
    }


def run_arm(arm: dict[str, Any]) -> dict[str, Any]:
    cleanup_vllm()
    work_dir = Path(f"/dev/shm/{arm['name']}")
    shutil.rmtree(work_dir, ignore_errors=True)
    bench_dir = work_dir / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    os.makedirs("/dev/shm/xla_gptoss", exist_ok=True)
    os.makedirs("/dev/shm/vllm_gptoss", exist_ok=True)
    shutil.rmtree("/dev/shm/vllm_gptoss/profile_cache", ignore_errors=True)
    env["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = "/dev/shm/xla_gptoss"
    env["VLLM_XLA_CACHE_PATH"] = "/dev/shm/xla_gptoss"
    env["VLLM_CACHE_ROOT"] = "/dev/shm/vllm_gptoss"
    env["VLLM_TPU_MXFP4_DEVICE_DEQUANT"] = "1"
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "3600")  # cold weight loads from GCS can take ~30 min

    argv = [
        "vllm", "serve", MODEL_PATH,
        "--served-model-name", MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--tensor-parallel-size", str(arm["tp"]),
        "--data-parallel-size", str(arm["dp"]),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(arm["max_num_seqs"]),
        "--max-num-batched-tokens", str(arm["max_batched_tokens"]),
        "--gpu-memory-utilization", str(arm.get("gpu_mem_util", "0.88")),
        "--kv-cache-dtype", "fp8",
        "--safetensors-load-strategy", "prefetch",
        "--no-enable-prefix-caching",
        "--no-enable-log-requests",
    ] + list(arm["extra_flags"])

    log_path = work_dir / "server.log"
    print(f"[{arm['name']}] Launching: {' '.join(argv)}", flush=True)
    with open(log_path, "w") as fh:
        srv = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

    try:
        startup_s = wait_health(PORT, srv, log_path)
        print(f"[{arm['name']}] Healthy in {startup_s}s", flush=True)

        # Sanity prompts: check that the output is real text and the arithmetic is right.
        sanity_outputs = {}
        for tag, prm in [
            ("explain_tpu", "Explain in two sentences why Cloud TPU v7x uses two TensorCores per chip:"),
            ("math_check", "Compute 37 * 43 step by step and state the final integer:"),
        ]:
            body = json.dumps({"model": MODEL_NAME, "prompt": prm, "max_tokens": 64, "temperature": 0.0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60.0) as r:
                sanity_outputs[tag] = json.loads(r.read().decode())["choices"][0]["text"]
        print(f"[{arm['name']}] Sanity outputs: {sanity_outputs}", flush=True)

        pts_out = {}
        for pt in arm.get("points_override") or POINTS:
            pts_out[pt["label"]] = run_bench_point(pt, bench_dir)
            m = pts_out[pt["label"]]
            print(
                f"  [{arm['name']}] {pt['label']}: out={m.get('output_throughput_tok_s')} tok/s | "
                f"goodput={m.get('request_goodput_req_s')} req/s | TTFT={m.get('median_ttft_ms')}ms | TPOT={m.get('median_tpot_ms')}ms",
                flush=True,
            )

        return {
            "arm": arm["name"],
            "model": MODEL_NAME,
            "license": "Apache-2.0",
            "quantization": "native_mxfp4_kmajor_fp4_padded_2944",
            "kv_cache_dtype": "fp8",
            "tp": arm["tp"],
            "dp": arm["dp"],
            "devices": arm["tp"] * arm["dp"],
            "startup_s": startup_s,
            "sanity_outputs": sanity_outputs,
            "points": pts_out,
        }
    finally:
        try:
            os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
        except Exception:
            pass
        cleanup_vllm()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", default="vllm_torchtpu_mxfp4_tp2_dp4_async", help="comma-separated setup names, or 'all'")
    ap.add_argument("--out", default=str(OUT_DIR / "vllm_torchtpu_mxfp4_results.json"))
    args = ap.parse_args()
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mxfp4_patch.apply_all()
    results: dict[str, Any] = json.loads(out_file.read_text()) if out_file.exists() else {}
    for arm in ARMS:
        if args.setups != "all" and arm["name"] not in args.setups.split(","):
            continue
        if "points" in results.get(arm["name"], {}):
            print(f"[skip] {arm['name']} already has results", flush=True)
            continue
        try:
            results[arm["name"]] = run_arm(arm)
        except Exception as exc:
            print(f"[error] {arm['name']}: {exc}", flush=True)
            results[arm["name"]] = {"error": str(exc)}
        out_file.write_text(json.dumps(results, indent=2))
    print(f"saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
