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
    
    return rows

def to_internal_schema(results, hw):
    out = []
    for r in results:
        # Map vLLM fields to our schema. 
        # Note: vLLM bench throughput summary might not split prefill/decode cleanly
        # if it's an end-to-end benchmark. We use available fields.
        
        model = r.get("model", "unknown")
        # Extract params from model name if not present (heuristic)
        params_b = r.get("params_b", 7.0) # fallback
        if "7b" in model.lower(): params_b = 7.0
        elif "8b" in model.lower(): params_b = 8.0
        elif "32b" in model.lower(): params_b = 32.0
        elif "70b" in model.lower(): params_b = 70.0

        n_batch = r.get("num_prompts", 1)
        p_in = r.get("input_len", 256)
        p_out = r.get("output_len", 64)
        
        # Prefill time is often not explicitly in the summary, 
        # but TTFT (Time To First Token) is a good proxy.
        # If not present, we might have to use NaN or estimate.
        ttft_avg = r.get("avg_ttft_ms", 0) / 1000.0
        
        # Decode throughput
        decode_tps = r.get("output_throughput", 0)
        
        # Quantization
        quant = "Unquantized"
        if "awq" in model.lower(): quant = "AWQ.4bit"
        elif "gptq" in model.lower(): quant = "GPTQ.4bit"
        elif "fp8" in model.lower(): quant = "FP8"

        out.append({
            "Experiment 🧪":     f"{model}|bs={n_batch}",
            "Model 🤗":          model,
            "Prefill (s)":       round(ttft_avg, 4) if ttft_avg > 0 else float("nan"),
            "Per Token (s)":     round(1.0 / decode_tps, 6) if decode_tps > 0 else float("nan"),
            "Decode (tokens/s)": round(decode_tps, 3),
            "Energy (tokens/kWh)": float("nan"),
            "Backend 🏭":         "vllm",
            "Precision 📥":       "int4/8" if "awq" in quant.lower() or "gptq" in quant.lower() else "fp16",
            "Quantization 🗜️":   quant,
            "Attention 👁️":       "PagedAttention",
            "Kernel ⚛️":          "vllm",
            "Architecture 🏛️":    model,
            "End-to-End (s)":    round(r.get("duration", 0), 4),
            "Open LLM Score (%)": float("nan"),
            "Params (B)":        params_b,
            "Memory (MB)":       float("nan"),
            "_n_prompt":         p_in,
            "_n_gen":            p_out,
            "_n_batch":          n_batch,
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
