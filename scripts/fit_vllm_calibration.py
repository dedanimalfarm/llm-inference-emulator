import os, sys, json, re, glob, argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.hardware import HARDWARE_SPECS, get_peak_compute, get_memory_bandwidth

def fit_hill_anchored(batches, throughputs):
    """Shifted Hill curve anchored at output_tps(b=1)=t1:
        out(batch) = t1 + (A - t1) · (batch - 1) / (batch - 1 + H)
    Returns (A, H, err). Uses t1 from data at batch=1 if present, else
    the smallest-batch observation.
    """
    best = (None, None, float("inf"))
    if not throughputs: return best
    # Anchor at the smallest-batch point (typically b=1).
    idx0 = batches.index(min(batches))
    b0, t0 = batches[idx0], throughputs[idx0]
    A_seed = max(throughputs) * 1.0
    for A_pct in range(80, 401, 2):  # asymptote 0.8x .. 4x of observed max
        A = A_seed * A_pct / 100.0
        if A <= t0: continue
        for H_idx in range(1, 400):
            H = H_idx * 0.25  # 0.25 .. 99.75
            err = 0.0
            for b, t in zip(batches, throughputs):
                if b <= b0:
                    pred = t0
                else:
                    pred = t0 + (A - t0) * (b - b0) / (b - b0 + H)
                err += (pred - t) ** 2
            if err < best[2]:
                best = (A, H, err)
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV or directory with JSONs", required=True)
    parser.add_argument("--hw", help="Hardware target (e.g. 1xA100-40)", required=True)
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--p-in", type=int, default=1024,
                        help="Prompt length used in the sweep (default 1024)")
    parser.add_argument("--p-out", type=int, default=128,
                        help="Output length used in the sweep (default 128)")
    parser.add_argument("--beta", type=float, default=0.65,
                        help="Assumed beta (MBU) for deriving batch_max from out_tps asymptote")
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
        runs = []
        for _, row in df.iterrows():
            P_in = row["_n_prompt"]
            P_out = row["_n_gen"]
            agg_tps = row["Decode (tokens/s)"] * (P_in + P_out) / P_out
            # Recover output-only tps from aggregate when only aggregate is available
            # (assumes prefill_share ≈ P_in/(P_in+P_out) of elapsed at compute-bound):
            out_tps = agg_tps * P_out / (P_in + P_out)
            runs.append({
                "model_tag": row["Model 🤗"],
                "params_b": row["Params (B)"],
                "active_b": row.get("Active (B)", row["Params (B)"]),
                "tp": row["_tp"],
                "batch": row["_n_batch"],
                "agg_tps": agg_tps,
                "out_tps": out_tps,
                "p_in": P_in, "p_out": P_out,
            })
    else:
        runs = []
        pattern = re.compile(r"^(.*)_b(\d+)\.json$")
        for path in glob.glob(f"{args.input}/*.json"):
            fname = os.path.basename(path).lower()
            m = pattern.match(fname)
            if not m:
                m_base = fname.replace(".json", "")
                batch = 1
            else:
                m_base, batch = m.groups()
                batch = int(batch)

            with open(path) as f:
                d = json.load(f)
            if isinstance(d, list): d = d[0]

            params_b = 7.0
            active_b = None
            if "mixtral" in m_base:
                params_b = 46.7
                active_b = 12.9
            elif "70b" in m_base: params_b = 70.0
            elif "32b" in m_base: params_b = 32.0
            elif "8b" in m_base: params_b = 8.0
            elif "1.5b" in m_base: params_b = 1.5
            elif "7b" in m_base: params_b = 7.0
            if active_b is None: active_b = params_b

            # vLLM bench's tokens_per_second is aggregate (P_in+P_out)/elapsed.
            # The emulator's saturation curve is on decode-only output throughput,
            # so split out the output share: out_tps = batch * P_out / elapsed.
            # P_in/P_out are not in the JSON; infer from total_num_tokens vs batch.
            total = d.get("total_num_tokens", 0)
            elapsed = d.get("elapsed_time", 1.0)
            agg = d.get("tokens_per_second", total/elapsed)
            # Heuristic split — caller can override via metadata when available.
            # For p1024/p128 sweep total ~= batch*(P_in+P_out) so P_out share = 128/1152.
            # Default to assuming the script-level P_out (set via --p-out arg).
            p_in = args.p_in
            p_out = args.p_out
            out_tps = batch * p_out / elapsed if elapsed > 0 else 0.0

            runs.append({
                "model_tag": m_base,
                "params_b": params_b,
                "active_b": active_b,
                "tp": 2 if "tp2" in m_base else 1,
                "batch": batch,
                "agg_tps": agg,
                "out_tps": out_tps,
                "p_in": p_in, "p_out": p_out,
            })

    groups = {}
    for r in runs:
        key = (r["model_tag"], r["tp"])
        groups.setdefault(key, []).append(r)

    print(f"Hardware: {args.hw} (Peak: {CARD_PEAK_TFLOPS} TFLOPS, BW: {CARD_MBW_GBS} GB/s)")
    print(f"Fit target: OUTPUT throughput (batch*P_out/elapsed). beta_assumed={args.beta}")
    print(f"{'config':>22}  {'n':>2} {'out_A':>7} {'half_pt':>7} {'alpha':>7} {'batch_max':>10}  saturation")
    print("-" * 90)

    for key in sorted(groups.keys()):
        group = groups[key]
        if len(group) < 2: continue

        batches = [r["batch"] for r in group]
        out_tps = [r["out_tps"] for r in group]
        agg_tps = [r["agg_tps"] for r in group]
        A_out, H, err = fit_hill_anchored(batches, out_tps)

        rep = group[0]
        N_total = rep["params_b"] * 1e9
        N_active = rep["active_b"] * 1e9
        tp = rep["tp"]
        C_eff = CARD_PEAK_TFLOPS * 1e12 * tp
        MBW_eff = CARD_MBW_GBS * 1e9 * tp
        W_active = N_active * 4.25 / 8.0  # AWQ 4.25 bits/param (memory read fraction)
        W_total = N_total * 4.25 / 8.0

        # alpha from prefill-bound regime at large batch:
        #   prefill_tps ≈ agg_tps · P_in/(P_in+P_out)  (compute-bound limit)
        #   alpha = (2 · N_active · prefill_tps) / C_eff
        agg_max = max(agg_tps)
        prefill_tps = agg_max * rep["p_in"] / (rep["p_in"] + rep["p_out"])
        alpha_eff = 2.0 * N_active * prefill_tps / C_eff

        # batch_max = output asymptote * decode-only t_dec.
        # MoE: weight read fraction = N_active / N_total.
        t_dec_mem = W_total * (N_active / N_total) / (MBW_eff * args.beta)
        batch_max = A_out * t_dec_mem

        print(f"{key[0]:>22}  {len(group):>2} {A_out:>7.0f} {H:>7.2f} {alpha_eff:>7.3f} {batch_max:>10.2f}  ({batch_max:.1f}, {H:.1f})")

if __name__ == "__main__":
    main()
