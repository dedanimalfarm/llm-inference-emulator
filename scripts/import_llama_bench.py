#!/usr/bin/env python3
"""Import llama-bench CSV output into our internal leaderboard format.

llama-bench (https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
produces one row per (test, model, config). Tests can be:
  pp{N}    — prompt processing (prefill) of N tokens
  tg{N}    — text generation (decode) of N tokens
  pg{P,G}  — prompt+gen end-to-end

This script joins matching pp + tg rows for the same (model, config) and
emits a single row per pair in the same column shape we use for the
LLM-Perf Leaderboard CSVs (so calibration code can ingest both).

Run:
  python scripts/import_llama_bench.py path/to/llama_bench.csv \\
      --hw 1xT4 --backend "llama.cpp" --output data/llama_bench.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.calibrate import effective_bpw


# llama-bench backend string -> our backend label (rough mapping)
BACKEND_MAP = {
    "CUDA": "llama.cpp",
    "CPU": "llama.cpp",
    "Metal": "llama.cpp",
    "ROCm": "llama.cpp",
    "Vulkan": "llama.cpp",
    "SYCL": "llama.cpp",
}


def label_quantization(model_type: str, bpw: float) -> str:
    """Best-effort human label for the quant scheme."""
    t = (model_type or "").lower()
    for tag in ("q2_k", "q3_k", "q4_0", "q4_k", "q5_0", "q5_k", "q6_k",
                "q8_0", "iq2", "iq3", "iq4", "f16", "bf16", "f32"):
        if tag in t:
            return f"{tag.upper()} ({bpw:.2f}bpw)"
    return f"{bpw:.2f}bpw"


def join_pp_tg(df: pd.DataFrame) -> pd.DataFrame:
    """Pair prompt-processing (n_prompt>0, n_gen=0) with token-generation
    (n_prompt=0, n_gen>0) rows for the same model+config."""
    keys = [c for c in [
        "model_filename", "model_type", "model_size", "model_n_params",
        "backends", "n_batch", "n_ubatch", "n_threads",
        "n_gpu_layers", "type_k", "type_v", "flash_attn",
    ] if c in df.columns]

    pp = df[(df["n_prompt"] > 0) & (df["n_gen"] == 0)].copy()
    tg = df[(df["n_prompt"] == 0) & (df["n_gen"] > 0)].copy()
    pp = pp.rename(columns={"avg_ts": "pp_ts", "stddev_ts": "pp_std",
                            "n_prompt": "pp_n_prompt"})
    tg = tg.rename(columns={"avg_ts": "tg_ts", "stddev_ts": "tg_std",
                            "n_gen": "tg_n_gen"})
    merged = pd.merge(
        pp[keys + ["pp_n_prompt", "pp_ts", "pp_std", "n_depth"]],
        tg[keys + ["tg_n_gen", "tg_ts", "tg_std", "n_depth"]],
        on=keys + ["n_depth"], how="inner",
    )
    return merged


def to_internal_schema(merged: pd.DataFrame, hw: str) -> pd.DataFrame:
    out = []
    for _, r in merged.iterrows():
        n_params = float(r["model_n_params"])
        size = float(r["model_size"])
        bpw = effective_bpw(size, n_params)
        prefill_s = float(r["pp_n_prompt"]) / float(r["pp_ts"])
        memory_mb = (size + 2 * float(r.get("n_depth", 0)) * 1024) / (1024 ** 2)
        backend_lb = BACKEND_MAP.get(r["backends"], "llama.cpp")
        attn = "FAv2" if int(r.get("flash_attn", 0)) == 1 else "Eager"
        out.append({
            "Experiment 🧪":     f"{r.get('model_type','')}|t={r.get('n_threads','')}|ngl={r.get('n_gpu_layers','')}",
            "Model 🤗":          r.get("model_type", ""),
            "Prefill (s)":       round(prefill_s, 4),
            "Per Token (s)":     round(1.0 / float(r["tg_ts"]), 6),
            "Decode (tokens/s)": round(float(r["tg_ts"]), 3),
            "Energy (tokens/kWh)": float("nan"),
            "Backend 🏭":         backend_lb,
            "Precision 📥":       "fp16" if bpw >= 15 else "int4/8",
            "Quantization 🗜️":   label_quantization(r.get("model_type", ""), bpw),
            "Attention 👁️":       attn,
            "Kernel ⚛️":          "llama.cpp",
            "Architecture 🏛️":    r.get("model_type", ""),
            "End-to-End (s)":    round(prefill_s + float(r["tg_n_gen"]) / float(r["tg_ts"]), 4),
            "Open LLM Score (%)": float("nan"),
            "Params (B)":        round(n_params / 1e9, 3),
            "Memory (MB)":       round(memory_mb, 1),
            # extra columns we keep for llama.cpp-specific calibration
            "_effective_bpw":    round(bpw, 4),
            "_n_threads":        int(r.get("n_threads", 0) or 0),
            "_n_batch":          int(r.get("n_batch", 0) or 0),
            "_n_gpu_layers":     int(r.get("n_gpu_layers", 0) or 0),
            "_kv_type_k":        r.get("type_k", "f16"),
            "_kv_type_v":        r.get("type_v", "f16"),
            "_n_depth":          int(r.get("n_depth", 0) or 0),
            "_hw":               hw,
        })
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="llama-bench CSV file")
    p.add_argument("--hw", required=True,
                   help="hardware label (must match HARDWARE_SPECS), e.g. 1xT4 or 32vCPU-C7i")
    p.add_argument("--backend", default=None,
                   help="override Backend column (default: derived from llama-bench backends)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    print(f"loaded {len(df)} llama-bench rows")
    paired = join_pp_tg(df)
    print(f"paired {len(paired)} pp+tg combinations")
    out = to_internal_schema(paired, args.hw)
    if args.backend:
        out["Backend 🏭"] = args.backend
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(out)} rows")
    print("\nSample (first 3):")
    cols = ["Model 🤗", "Params (B)", "Quantization 🗜️", "Prefill (s)",
            "Decode (tokens/s)", "_effective_bpw", "_n_threads", "_n_gpu_layers"]
    print(out[cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
