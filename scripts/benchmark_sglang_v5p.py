#!/usr/bin/env python3
"""Single-host GPT-OSS 120B (native MXFP4) serving benchmark on one Cloud TPU v5p host (4 chips) with sglang-torchtpu.

For each named setup this script starts an sglang server on the local 4-chip host, waits until it is healthy, sends
one short sanity prompt, then runs `sglang.bench_serving` with random fixed-length prompts at several loads. Results go
to one JSON file (one entry per setup); setups that already have results in it are skipped. The server flags below are
exactly the ones used for the results in this repo.

Usage (inside an sglang-torchtpu pod on one v5p host):
    MODEL_PATH=/models/openai/gpt-oss-120b \
        python3 benchmark_sglang_v5p.py --setups sglang_mxfp4_tp4_ep4_compiled_v5p

Each setup needs about 25 to 45 minutes to load and compile before it measures anything.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/openai/gpt-oss-120b")
DFLASH_PATH = os.environ.get("DFLASH_PATH", "/models/z-lab/gpt-oss-120b-DFlash")
MODEL_NAME = "openai/gpt-oss-120b"
PORT = 30000
OUT_DIR = Path(os.environ.get("OUT_DIR", "./results"))


POINTS = [
    {"label": "1k_1k_c1", "isl": 1024, "osl": 1024, "conc": 1, "prompts": 12},
    {"label": "1k_1k_c8", "isl": 1024, "osl": 1024, "conc": 8, "prompts": 32},
    {"label": "1k_1k_c32", "isl": 1024, "osl": 1024, "conc": 32, "prompts": 96},
    {"label": "1k_1k_c64", "isl": 1024, "osl": 1024, "conc": 64, "prompts": 128},
    {"label": "8k_1k_c1", "isl": 8192, "osl": 1024, "conc": 1, "prompts": 8},
    {"label": "8k_1k_c32", "isl": 8192, "osl": 1024, "conc": 32, "prompts": 64},
    {"label": "22k_4k_c1", "isl": 22000, "osl": 4000, "conc": 1, "prompts": 6},
    {"label": "22k_4k_c8", "isl": 22000, "osl": 4000, "conc": 8, "prompts": 16},
]
C128 = [{"label": "1k_1k_c128", "isl": 1024, "osl": 1024, "conc": 128, "prompts": 256}]
C256 = [{"label": "1k_1k_c256", "isl": 1024, "osl": 1024, "conc": 256, "prompts": 512}]
C512 = [{"label": "1k_1k_c512", "isl": 1024, "osl": 1024, "conc": 512, "prompts": 1024}]
C1024 = [{"label": "1k_1k_c1024", "isl": 1024, "osl": 1024, "conc": 1024, "prompts": 2048}]
C2048 = [{"label": "1k_1k_c2048", "isl": 1024, "osl": 1024, "conc": 2048, "prompts": 4096}]
HIGH_8K = [{"label": f"8k_1k_c{c}", "isl": 8192, "osl": 1024, "conc": c, "prompts": 2 * c} for c in (64, 128, 256, 512)]
LONG = [
    {"label": "8k_1k_c1", "isl": 8192, "osl": 1024, "conc": 1, "prompts": 8},
    {"label": "22k_4k_c1", "isl": 22000, "osl": 4000, "conc": 1, "prompts": 6},
    {"label": "22k_4k_c8", "isl": 22000, "osl": 4000, "conc": 8, "prompts": 16},
]

# `args` follow `--model-path ... --served-model-name ... --host 0.0.0.0 --port 30000` on the server command line.
# `exp_spec` turns on the experimental speculative-decoding worker that DFlash needs with data-parallel attention.
SETUPS: list[dict[str, Any]] = [
    # v5p host: 4 chips, one device per chip. TP=4, EP=4, torch.compile, decode graphs up to batch 64, 256 running
    # requests, default KV cache type (for this checkpoint the server picks FP8 e4m3; see the server log).
    {"name": "sglang_mxfp4_tp4_ep4_compiled_v5p", "exp_spec": False,
     "points": [{"label": "1k_1k_c1", "isl": 1024, "osl": 1024, "conc": 1, "prompts": 12}]
               + [{"label": f"1k_1k_c{c}", "isl": 1024, "osl": 1024, "conc": c, "prompts": max(2 * c, 32)} for c in (8, 32, 64, 128, 256)]
               + [{"label": "22k_4k_c1", "isl": 22000, "osl": 4000, "conc": 1, "prompts": 6}],
     "args": ["--tp-size", "4", "--ep-size", "4", "--context-length", "26624", "--mem-fraction-static", "0.74", "--disable-radix-cache", "--enable-torch-compile", "--cuda-graph-max-bs-decode", "64", "--max-running-requests", "256"]},
    # Same with the KV cache type set explicitly to FP8 e4m3, as in the v7x runs.
    {"name": "sglang_mxfp4_tp4_ep4_compiled_fp8kv_v5p", "exp_spec": False,
     "points": [{"label": "1k_1k_c1", "isl": 1024, "osl": 1024, "conc": 1, "prompts": 12}]
               + [{"label": f"1k_1k_c{c}", "isl": 1024, "osl": 1024, "conc": c, "prompts": max(2 * c, 32)} for c in (8, 32, 64, 128, 256)],
     "args": ["--tp-size", "4", "--ep-size", "4", "--kv-cache-dtype", "fp8_e4m3", "--context-length", "26624", "--mem-fraction-static", "0.74", "--disable-radix-cache", "--enable-torch-compile", "--cuda-graph-max-bs-decode", "64", "--max-running-requests", "256"]},
    # Same as the first setup with a BF16 KV cache (v5p has no FP8 hardware).
    {"name": "sglang_mxfp4_tp4_ep4_compiled_bf16kv_v5p", "exp_spec": False,
     "points": [{"label": "1k_1k_c1", "isl": 1024, "osl": 1024, "conc": 1, "prompts": 12}]
               + [{"label": f"1k_1k_c{c}", "isl": 1024, "osl": 1024, "conc": c, "prompts": max(2 * c, 32)} for c in (8, 32, 64, 128, 256)],
     "args": ["--tp-size", "4", "--ep-size", "4", "--kv-cache-dtype", "bfloat16", "--context-length", "26624", "--mem-fraction-static", "0.74", "--disable-radix-cache", "--enable-torch-compile", "--cuda-graph-max-bs-decode", "64", "--max-running-requests", "256"]},
    # Data-parallel attention over the 4 chips (the high-load layout of the v7x runs, which use 8 groups on 8 cores),
    # decode graphs up to batch 128, 512 running requests, FP8 KV cache, 16,384-token prefill chunks.
    {"name": "sglang_mxfp4_dp4_tp4_ep4_compiled_bs128_chunk16k_v5p", "exp_spec": False,
     "points": [{"label": f"1k_1k_c{c}", "isl": 1024, "osl": 1024, "conc": c, "prompts": 2 * c} for c in (256, 512, 1024)],
     "args": ["--tp-size", "4", "--ep-size", "4", "--kv-cache-dtype", "fp8_e4m3", "--context-length", "26624", "--mem-fraction-static", "0.74", "--disable-radix-cache", "--enable-torch-compile", "--cuda-graph-max-bs-decode", "128", "--enable-dp-attention", "--dp-size", "4", "--enable-dp-lm-head", "--max-running-requests", "512", "--chunked-prefill-size", "16384"]},
    # The data-parallel attention setup with a BF16 KV cache (BF16 KV gave more throughput than FP8 on v5p with TP=4).
    {"name": "sglang_mxfp4_dp4_tp4_ep4_compiled_bs128_chunk16k_bf16kv_v5p", "exp_spec": False,
     "points": [{"label": f"1k_1k_c{c}", "isl": 1024, "osl": 1024, "conc": c, "prompts": 2 * c} for c in (256, 512, 1024)],
     "args": ["--tp-size", "4", "--ep-size", "4", "--kv-cache-dtype", "bfloat16", "--context-length", "26624", "--mem-fraction-static", "0.74", "--disable-radix-cache", "--enable-torch-compile", "--cuda-graph-max-bs-decode", "128", "--enable-dp-attention", "--dp-size", "4", "--enable-dp-lm-head", "--max-running-requests", "512", "--chunked-prefill-size", "16384"]},
    # Same with decode graphs up to batch 256 and 1,024 running requests (512 running requests queued at 1,024).
    {"name": "sglang_mxfp4_dp4_tp4_ep4_compiled_bs256_chunk16k_bf16kv_v5p", "exp_spec": False,
     "points": [{"label": f"1k_1k_c{c}", "isl": 1024, "osl": 1024, "conc": c, "prompts": 2 * c} for c in (512, 1024, 2048)],
     "args": ["--tp-size", "4", "--ep-size", "4", "--kv-cache-dtype", "bfloat16", "--context-length", "26624", "--mem-fraction-static", "0.74", "--disable-radix-cache", "--enable-torch-compile", "--cuda-graph-max-bs-decode", "256", "--enable-dp-attention", "--dp-size", "4", "--enable-dp-lm-head", "--max-running-requests", "1024", "--chunked-prefill-size", "16384"]},
]


def stop_sglang() -> None:
    me = os.getpid()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit() or int(pid_str) in (1, me):
                continue
            try:
                cmd = open(f"/proc/{pid_str}/cmdline", "rb").read().decode("utf-8", "ignore")
                if "sglang.launch_server" in cmd or "sglang::" in cmd:
                    os.kill(int(pid_str), sig)
            except Exception:
                pass
        time.sleep(4)
    if os.path.exists("/tmp/libtpu_lockfile"):
        os.remove("/tmp/libtpu_lockfile")


def wait_healthy(proc: subprocess.Popen[Any], log_path: Path, timeout_s: int = 5400) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"sglang exited early ({proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2.0) as r:
                if r.status == 200:
                    return round(time.perf_counter() - t0, 2)
        except Exception:
            pass
        time.sleep(4)
    raise TimeoutError(f"server not healthy after {timeout_s}s")


def run_point(p: dict[str, Any], out_json: Path) -> dict[str, Any]:
    """One `sglang.bench_serving` run with prompts of exactly `isl` tokens and exactly `osl` output tokens."""
    # Newer sglang builds renamed --random-input/--random-output to --random-input-len/--random-output-len.
    for in_flag, out_flag in [("--random-input", "--random-output"), ("--random-input-len", "--random-output-len")]:
        cmd = [
            "python3", "-m", "sglang.bench_serving",
            "--backend", "sglang-oai",
            "--host", "127.0.0.1", "--port", str(PORT),
            "--model", MODEL_PATH,
            "--dataset-name", "random",
            in_flag, str(p["isl"]), out_flag, str(p["osl"]),
            "--random-range-ratio", "1.0",
            "--max-concurrency", str(p["conc"]),
            "--num-prompts", str(p["prompts"]),
            "--output-file", str(out_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_json.exists():
            break
    if not out_json.exists():
        return {"error": (proc.stderr or proc.stdout)[-1500:]}
    d = json.loads([l for l in out_json.read_text().splitlines() if l.strip()][-1])
    med_ttft = float(d.get("median_ttft_ms", 0.0))
    med_tpot = float(d.get("median_tpot_ms", d.get("median_itl_ms", 0.0)))
    med_e2e = float(d.get("median_e2e_latency_ms", 0.0))
    return {
        "isl": p["isl"], "osl": p["osl"], "concurrency": p["conc"],
        "completed": d.get("completed"),
        "request_throughput_req_s": round(float(d.get("request_throughput", 0.0)), 3),
        "output_throughput_tok_s": round(float(d.get("output_throughput", 0.0)), 2),
        "total_token_throughput_tok_s": round(float(d.get("total_throughput", d.get("total_token_throughput", 0.0))), 2),
        "median_ttft_ms": round(med_ttft, 2),
        "p99_ttft_ms": round(float(d.get("p99_ttft_ms", 0.0)), 2),
        "median_tpot_ms": round(med_tpot, 2),
        "p99_tpot_ms": round(float(d.get("p99_tpot_ms", d.get("p99_itl_ms", 0.0))), 2),
        "median_e2e_ms": round(med_e2e, 2),
        "accept_length": d.get("accept_length"),
        "passes_22k_median_slo": (med_ttft <= 500.0 and med_tpot <= 10.0 and med_e2e <= 60000.0) if p["isl"] >= 20000 else None,
    }


def run_setup(setup: dict[str, Any]) -> dict[str, Any]:
    stop_sglang()
    wdir = Path(f"/dev/shm/{setup['name']}")
    shutil.rmtree(wdir, ignore_errors=True)
    wdir.mkdir(parents=True)
    env = os.environ.copy()
    os.makedirs("/dev/shm/xla_sglang_gptoss", exist_ok=True)
    env["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = "/dev/shm/xla_sglang_gptoss"
    env["FORCE_SINGLE_HOST"] = "1"
    env["PJRT_DEVICE"] = "TPU"
    env["SGLANG_TPU_FUSED_MOE_SC_GATHER"] = "0"
    if setup["exp_spec"]:
        env["SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER"] = "1"
    argv = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
    ] + setup["args"]
    log_path = wdir / "server.log"
    print(f"[{setup['name']}] starting: {' '.join(argv)}", flush=True)
    with open(log_path, "w") as fh:
        srv = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        startup_s = wait_healthy(srv, log_path)
        print(f"[{setup['name']}] healthy after {startup_s}s", flush=True)
        body = json.dumps({"model": MODEL_NAME, "prompt": "Compute 37 * 43 step by step:", "max_tokens": 64,
                           "temperature": 0.0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60.0) as r:
            sanity = json.loads(r.read().decode())["choices"][0]["text"]
        points = {}
        for p in setup["points"]:
            points[p["label"]] = run_point(p, wdir / f"{p['label']}.jsonl")
            print(f"  [{setup['name']}] {p['label']}: {points[p['label']]}", flush=True)
        return {"arm": setup["name"], "startup_s": startup_s, "sanity_text": sanity, "points": points}
    finally:
        try:
            os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
        except Exception:
            pass
        stop_sglang()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", default="sglang_mxfp4_tp4_ep4_compiled_v5p",
                    help="comma-separated setup names, or 'all'")
    ap.add_argument("--out", default=str(OUT_DIR / "sglang_mxfp4_v5p_results.json"))
    args = ap.parse_args()
    wanted = [s for s in SETUPS if args.setups == "all" or s["name"] in args.setups.split(",")]
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = json.loads(out_file.read_text()) if out_file.exists() else {}
    for setup in wanted:
        if "points" in results.get(setup["name"], {}):
            print(f"[skip] {setup['name']} already has results", flush=True)
            continue
        try:
            results[setup["name"]] = run_setup(setup)
        except Exception as exc:
            print(f"[error] {setup['name']}: {exc}", flush=True)
            results[setup["name"]] = {"error": str(exc)}
        out_file.write_text(json.dumps(results, indent=2))
    print(f"saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
