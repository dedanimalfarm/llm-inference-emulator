#!/usr/bin/env python3
"""Import vLLM benchmark JSON results into our internal leaderboard format.

Supports vLLM 0.9+ 'bench throughput --output-json' format.

Usage:
  python scripts/import_vllm_bench.py out_7b.json out_32b.json \
      --hw 2xRTX-5090 --output data/leaderboard-2xRTX-5090-vllm.csv
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_vllm_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    # vLLM output can be a single object or a list of objects if multiple runs were captured
    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        rows = data
    else:
        print(f"Warning: unexpected JSON format in {path}")
        return []
    
    # Enrich data from filename if model/metadata is missing
    fname = os.path.basename(path).lower()
    for r in rows:
        if "model" not in r or r["model"] == "unknown":
            if "qwen7b" in fname: 
                r["model"] = "Qwen/Qwen2.5-7B-Instruct-AWQ"
                r["params_b"] = 7.0
            elif "qwen32b" in fname:
                r["model"] = "Qwen/Qwen2.5-32B-Instruct-AWQ"
                r["params_b"] = 32.0
            elif "llama70b" in fname:
                r["model"] = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
                r["params_b"] = 70.0
        
        if "tp" not in r:
            if "tp1" in fname: r["tp"] = 1
            elif "tp2" in fname: r["tp"] = 2
        
        # Estimate input/output len from total_num_tokens and num_requests if missing
        if "input_len" not in r and "total_num_tokens" in r and "num_requests" in r:
            # vLLM 0.20.1 defaults for random dataset
            r["input_len"] = 1024
            r["output_len"] = 128

    return rows

def parse_vllm_log(path):
    """Extract 'Avg prompt throughput' from vLLM log if available."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            for line in f:
                if "Avg prompt throughput:" in line:
                    # INFO 05-08 23:33:44 [loggers.py:271] Engine 000: Avg prompt throughput: 4971.3 tokens/s
                    parts = line.split("Avg prompt throughput:")
                    if len(parts) > 1:
                        val = parts[1].split("tokens/s")[0].strip()
                        return float(val)
    except:
        pass
    return None

def to_internal_schema(results, hw):
    out = []
    for r in results:
        model = r.get("model", "unknown")
        params_b = r.get("params_b", 7.0)
        n_batch = r.get("num_requests", 1)
        p_in = r.get("input_len", 1024)
        p_out = r.get("output_len", 128)

        # Aggregate prompt throughput from engine log (snapshot during prefill).
        # NaN if the run was too short for the engine to emit a logger line.
        # We DO NOT divide by batch — α calibration uses aggregate FLOPS
        # throughput directly (α = 2·N·prompt_tps / C), independent of batching.
        prompt_tps_aggregate = r.get("prompt_tps")
        if prompt_tps_aggregate is None or prompt_tps_aggregate <= 0:
            prompt_tps_aggregate = float("nan")

        # tokens_per_second from the JSON is end-to-end aggregate (prefill + decode
        # interleaved). Output_tps is the decode-phase aggregate; we recover it
        # by ratio because vLLM JSON doesn't break it out separately.
        total_tps = r.get("tokens_per_second", 0)
        decode_tps_aggregate = total_tps * (p_out / (p_in + p_out))

        # NOTE: per-request prefill time is unidentifiable from a single batch
        # run (need to know prefill concurrency, which depends on chunked-prefill
        # scheduler decisions). Leave Prefill (s) as NaN; α is calibrated from
        # _prompt_tps_agg via a vLLM-aware path in calibrate.py.
        prefill_s = float("nan")

        tp = r.get("tp", 1)
        quant = "AWQ.4bit"

        # Estimate memory (BUG-5)
        memory_mb = (params_b * 1000 * 4 / 8) + 1024

        out.append({
            "Experiment 🧪":     f"{model}|bs={n_batch}|tp={tp}",
            "Model 🤗":          model,
            "Prefill (s)":       float("nan"),
            "Per Token (s)":     round(1.0 / decode_tps_aggregate, 6) if decode_tps_aggregate > 0 else float("nan"),
            "Decode (tokens/s)": round(decode_tps_aggregate, 3),
            "_prompt_tps_agg":   round(prompt_tps_aggregate, 3) if not pd.isna(prompt_tps_aggregate) else float("nan"),
            "Energy (tokens/kWh)": float("nan"),
            "Backend 🏭":         "vllm",
            "Precision 📥":       "int4/8",
            "Quantization 🗜️":   quant,
            "Attention 👁️":       "PagedAttention",
            "Kernel ⚛️":          "vllm",
            "Architecture 🏛️":    model,
            "End-to-End (s)":    round(r.get("elapsed_time", 0), 4),
            "Open LLM Score (%)": float("nan"),
            "Params (B)":        params_b,
            "Memory (MB)":       memory_mb,
            "_n_prompt":         p_in,
            "_n_gen":            p_out,
            "_n_batch":          n_batch,
            "_tp":               tp,
            "_hw":               hw,
        })
    return pd.DataFrame(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="vLLM benchmark JSON files")
    p.add_argument("--hw", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    all_results = []
    for path in args.inputs:
        rows = parse_vllm_json(path)
        # Try to find log file
        log_path = os.path.join(os.path.dirname(path), "logs", os.path.basename(path).replace(".json", ".log"))
        prompt_tps = parse_vllm_log(log_path)
        for r in rows:
            r["prompt_tps"] = prompt_tps
        all_results.extend(rows)
    
    print(f"Loaded {len(all_results)} vLLM benchmark results")
    out = to_internal_schema(all_results, args.hw)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")

if __name__ == "__main__":
    main()
