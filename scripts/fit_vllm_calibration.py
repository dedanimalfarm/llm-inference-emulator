import os, sys, json, re, glob, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.hardware import HARDWARE_SPECS, get_peak_compute, get_memory_bandwidth

def fit_hill(batches, throughputs):
    """Brute-force grid search for asymptote A and half_point H minimising
    L2 error of   agg(batch) = A · batch / (batch + H)
    """
    best = (None, None, float("inf"))
    if not throughputs: return best
    A_max = max(throughputs) * 1.5
    for A_pct in range(80, 151, 1):  # asymptote 80%..150% of observed max
        A = max(throughputs) * A_pct / 100.0
        for H_idx in range(1, 200):
            H = H_idx * 0.5  # 0.5 .. 99.5
            err = 0.0
            for b, t in zip(batches, throughputs):
                pred = A * b / (b + H)
                err += (pred - t) ** 2
            if err < best[2]:
                best = (A, H, err)
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV or directory with JSONs", required=True)
    parser.add_argument("--hw", help="Hardware target (e.g. 1xA100-40)", required=True)
    parser.add_argument("--engine", default="vllm")
    args = parser.parse_args()

    if args.hw not in HARDWARE_SPECS:
        print(f"Error: unknown hardware {args.hw}")
        sys.exit(1)
    
    specs = HARDWARE_SPECS[args.hw]
    # For calibration we typically use FP16/BF16 peak
    CARD_PEAK_TFLOPS = specs["peak_tflops"][16]
    CARD_MBW_GBS = specs["memory_bandwidth_gbs"]

    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
        # Convert to the internal format used by the fitter
        runs = []
        for _, row in df.iterrows():
            runs.append({
                "model_tag": row["Model 🤗"],
                "params_b": row["Params (B)"],
                "tp": row["_tp"],
                "batch": row["_n_batch"],
                "tokens_per_second": row["Decode (tokens/s)"] * (row["_n_prompt"] + row["_n_gen"]) / row["_n_gen"]
            })
    else:
        # Assume directory with JSONs
        runs = []
        pattern = re.compile(r"^(.*)_b(\d+)\.json$")
        for path in glob.glob(f"{args.input}/*.json"):
            fname = os.path.basename(path).lower()
            m = pattern.match(fname)
            if not m: 
                # maybe no batch tag in filename?
                m_base = fname.replace(".json", "")
                batch = 1
            else:
                m_base, batch = m.groups()
                batch = int(batch)
            
            with open(path) as f:
                d = json.load(f)
            if isinstance(d, list): d = d[0]
            
            params_b = 7.0 # default
            if "70b" in m_base: params_b = 70.0
            elif "32b" in m_base: params_b = 32.0
            elif "8b" in m_base: params_b = 8.0
            elif "1.5b" in m_base: params_b = 1.5
            elif "7b" in m_base: params_b = 7.0

            runs.append({
                "model_tag": m_base,
                "params_b": params_b,
                "tp": 2 if "tp2" in m_base else 1,
                "batch": batch,
                "tokens_per_second": d["tokens_per_second"]
            })

    groups = {}
    for r in runs:
        key = (r["model_tag"], r["tp"])
        groups.setdefault(key, []).append(r)

    print(f"Hardware: {args.hw} (Peak: {CARD_PEAK_TFLOPS} TFLOPS, BW: {CARD_MBW_GBS} GB/s)")
    print(f"{'config':>22}  {'n':>2} {'asympt(t/s)':>11} {'half_pt':>7} {'alpha(eff)':>7} {'batch_max':>10}")
    print("-" * 85)

    for key in sorted(groups.keys()):
        group = groups[key]
        if len(group) < 2: continue
        
        batches = [r["batch"] for r in group]
        throughputs = [r["tokens_per_second"] for r in group]
        A, H, err = fit_hill(batches, throughputs)

        rep = group[0]
        N = rep["params_b"] * 1e9
        tp = rep["tp"]
        C_eff = CARD_PEAK_TFLOPS * 1e12 * tp
        MBW_eff = CARD_MBW_GBS * 1e9 * tp
        W = N * 4.25 / 8.0 # AWQ 4.25 bits/param
        
        alpha_eff = 2.0 * N * A / C_eff
        beta_assumed = 0.65
        t_dec_mem = W / (MBW_eff * beta_assumed)
        batch_max = A * t_dec_mem

        print(f"{key[0]:>22}  {len(group):>2} {A:>11.0f} {H:>7.2f} {alpha_eff:>7.3f} {batch_max:>10.1f}  sat=({batch_max:.1f}, {H:.1f})")

if __name__ == "__main__":
    main()
