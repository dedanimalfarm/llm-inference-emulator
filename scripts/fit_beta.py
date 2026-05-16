#!/usr/bin/env python3
"""Fit beta (KV-read efficiency) from long-context decode data.

Linear regression: t_dec_total = t_dec_base + (1/(MBW * beta)) * (KV_bytes_per_token * avg_context)
Where t_dec_total is the measured decode time per token.
"""
import os, sys, json, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.hardware import HARDWARE_SPECS
from emulator.formula import _arch_for

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Directory with JSONs", required=True)
    parser.add_argument("--hw", help="Hardware target", required=True)
    parser.add_argument("--model", default="7.0")
    args = parser.parse_args()

    if args.hw not in HARDWARE_SPECS:
        print(f"Error: unknown hardware {args.hw}")
        sys.exit(1)
    
    specs = HARDWARE_SPECS[args.hw]
    MBW = specs["memory_bandwidth_gbs"] * 1e9
    
    arch = _arch_for(float(args.model))
    L = arch["layers"]
    H_kv = arch["kv_heads"]
    d_model = arch["d_model"]
    H = arch.get("num_attention_heads", d_model // 128) # approximation
    head_dim = arch.get("head_dim", 128)
    
    # bytes per token in KV cache (FP16)
    # 2 (K+V) * L * H_kv * head_dim * 2 (bytes)
    kv_per_token_bytes = 2 * L * H_kv * head_dim * 2

    results = []
    import glob
    for path in glob.glob(f"{args.input}/qwen7b_ctx*.json"):
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, list): d = d[0]
        
        # Extract P_in and P_out from filename
        fname = os.path.basename(path)
        # qwen7b_ctx16k_o512.json
        parts = fname.replace(".json", "").split("_")
        p_in = int(parts[1].replace("ctx", "").replace("k", "000"))
        p_out = int(parts[2].replace("o", ""))
        
        # Measured decode time per token
        # Throughput reports total tokens per second (including prefill usually)
        # but in bench throughput with 1 request:
        # tokens_per_second = (p_in + p_out) / elapsed
        # total_latency = elapsed
        # t_prefill = p_in / (MFU * Peak) ... we need to isolate decode.
        
        # vLLM 0.9 bench throughput output-json doesn't give prefill separately.
        # But for p_out >> 1, we can estimate.
        # Better: use multiple p_out for same p_in.
        # elapsed = t_prefill + p_out * t_dec_per_token
        results.append({
            "p_in": p_in,
            "p_out": p_out,
            "elapsed": d["elapsed_time"],
            "tps": d["tokens_per_second"]
        })

    if len(results) < 2:
        print("Error: need at least 2 points with different p_out to isolate decode.")
        sys.exit(1)

    df = pd.DataFrame(results)
    # Regression: elapsed = a * p_out + b
    # slope 'a' is t_dec_per_token (at this p_in)
    
    # Group by p_in
    for p_in, group in df.groupby("p_in"):
        if len(group) < 2: continue
        x = group["p_out"].values
        y = group["elapsed"].values
        a, b = np.polyfit(x, y, 1)
        
        t_dec_per_token = a
        avg_ctx = p_in # approximation for slope
        
        # t_dec_per_token = W / (MBW * beta) + (kv_per_token_bytes * p_in) / (MBW * beta)
        # Actually, the slope 'a' ALREADY includes the KV-read cost for one token.
        # t_dec(ctx) = (W + kv_per_token_bytes * ctx) / (MBW * beta)
        # So 'a' = kv_per_token_bytes / (MBW * beta) ??? No.
        # 'a' is the time to generate ONE token when ctx is p_in.
        
        # beta = (W + kv_per_token_bytes * p_in) / (MBW * t_dec_per_token)
        w_bits = 4.25
        n_params = float(args.model) * 1e9
        W = n_params * w_bits / 8.0
        
        beta = (W + kv_per_token_bytes * p_in) / (MBW * t_dec_per_token)
        print(f"P_in: {p_in}")
        print(f"  Isolated t_dec: {t_dec_per_token*1000:.3f} ms/token")
        print(f"  Calculated beta: {beta:.3f}")

if __name__ == "__main__":
    main()
