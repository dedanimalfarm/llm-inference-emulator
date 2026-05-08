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
            # Assuming fixed split 256/64 as per our benchmark run
            r["input_len"] = 256
            r["output_len"] = 64

    return rows

def to_internal_schema(results, hw):
    out = []
    for r in results:
        model = r.get("model", "unknown")
        params_b = r.get("params_b", 7.0)
        n_batch = r.get("num_requests", 1)
        p_in = r.get("input_len", 256)
        p_out = r.get("output_len", 64)
        
        # TTFT not in summary, use End-to-End as upper bound or leave NaN
        ttft_avg = float("nan")
        
        # tokens_per_second is total throughput. 
        # Output throughput = total_throughput * (output_len / (input_len + output_len))
        total_tps = r.get("tokens_per_second", 0)
        decode_tps = total_tps * (p_out / (p_in + p_out))
        
        tp = r.get("tp", 1)
        quant = "AWQ.4bit" # all our vLLM tests used AWQ
        
        # Estimate memory (BUG-5)
        memory_mb = (params_b * 1000 * 4 / 8) + 1024 # weights + small overhead

        out.append({
            "Experiment 🧪":     f"{model}|bs={n_batch}|tp={tp}",
            "Model 🤗":          model,
            "Prefill (s)":       ttft_avg,
            "Per Token (s)":     round(1.0 / decode_tps, 6) if decode_tps > 0 else float("nan"),
            "Decode (tokens/s)": round(decode_tps, 3),
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
        all_results.extend(parse_vllm_json(path))
    
    print(f"Loaded {len(all_results)} vLLM benchmark results")
    out = to_internal_schema(all_results, args.hw)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")

if __name__ == "__main__":
    main()
