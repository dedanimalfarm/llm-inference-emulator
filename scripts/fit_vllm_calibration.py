#!/usr/bin/env python3
"""Fit α, β, batch_saturation for vLLM from multi-batch sweep data.

The fit uses `tokens_per_second` (total = prefill+decode combined over the
whole elapsed run) as the saturation target, because vLLM's chunked
prefill interleaves both phases and that aggregate is the GPU's actual
sustained token-processing rate. Pure-decode throughput would require
TTFT data we don't have.

Saturation model (Hill curve):
    agg_total_tps(batch) = asymptote · batch / (batch + half_point)

From the formula, asymptote ≈ bs_eff_max / t_dec_at_saturation, so once
we have asymptote (well-determined from data) and assume a value for β,
batch_max follows.
"""
import os, sys, json, re, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.formula import _arch_for

CARD_PEAK_TFLOPS_FP16 = 419.0
CARD_MBW_GBS         = 1792.0

DATA_DIR = "data/raw_vllm"


def parse_runs():
    pattern = re.compile(r"^(qwen7b|qwen32b|llama70b)_tp(\d+)(?:_b(\d+))?(.*)\.json$")
    runs = []
    for path in sorted(glob.glob(f"{DATA_DIR}/*.json")):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if not m: continue
        model_tag, tp_s, batch_s, suffix = m.groups()
        # exclude isolation/long-context variants
        if any(s in suffix for s in ["eager", "kvfp8", "apc_off", "apc_on", "_8k", "_4k"]):
            continue
        if model_tag == "qwen7b": params_b = 7.0
        elif model_tag == "qwen32b": params_b = 32.0
        elif model_tag == "llama70b": params_b = 70.0
        else: continue
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, list): d = d[0]
        batch = int(batch_s) if batch_s else d.get("num_requests", 1)
        runs.append({
            "model_tag": model_tag, "params_b": params_b, "tp": int(tp_s),
            "batch": batch,
            "tokens_per_second": d["tokens_per_second"],
            "elapsed_s": d["elapsed_time"],
            "total_tokens": d["total_num_tokens"],
        })
    return runs


def fit_hill(batches, throughputs):
    """Brute-force grid search for asymptote A and half_point H minimising
    L2 error of   agg(batch) = A · batch / (batch + H)
    """
    best = (None, None, float("inf"))
    A_max = max(throughputs) * 1.5
    for A_pct in range(80, 121, 1):  # asymptote 80%..120% of observed max
        A = max(throughputs) * A_pct / 100.0
        for H_idx in range(1, 100):
            H = H_idx * 0.5  # 0.5 .. 49.5
            err = 0.0
            for b, t in zip(batches, throughputs):
                pred = A * b / (b + H)
                err += (pred - t) ** 2
            if err < best[2]:
                best = (A, H, err)
    return best  # (asymptote, half_point, sum_sq_err)


def fit_config(group):
    """Returns dict with α, β, batch_max, batch_50pct, asymptote_obs."""
    rep = group[0]
    N = rep["params_b"] * 1e9
    bits = 4.25  # AWQ effective BPW
    W = N * bits / 8.0
    tp = rep["tp"]
    C_eff = CARD_PEAK_TFLOPS_FP16 * 1e12 * tp
    MBW_eff = CARD_MBW_GBS * 1e9 * tp

    batches = [r["batch"] for r in group]
    throughputs = [r["tokens_per_second"] for r in group]
    A, H, err = fit_hill(batches, throughputs)

    # α from asymptote (compute-bound regime: agg = C_eff·α/(2N))
    alpha_eff = 2.0 * N * A / C_eff

    # β: choose a literature value (0.65 for vLLM Marlin AWQ on Ampere/Ada,
    # similar on Blackwell because Marlin still bandwidth-bound at small batch).
    # We can't fit β independently from this single batch curve.
    beta_assumed = 0.65

    # batch_max: at the asymptote, formula predicts agg = bs_eff_max / t_dec
    # In compute-bound regime t_dec_compute = 2N·bs_eff_max / (C_eff · α_eff)
    #     = 2N / (C_eff · α_eff / bs_eff_max) — this is just bs_eff_max/asymptote
    # In memory-bound regime t_dec = W / (MBW·β), independent of bs_eff
    # Crossover batch: 2N·bs_eff_max/(C·α) == W/(MBW·β)
    #     bs_eff_crossover = (W·C·α) / (2N·MBW·β) = α/β · C/(MBW·) · bits/8 /2
    # Below crossover memory-bound, above compute-bound.
    # For the asymptote we ASSUME compute-bound saturation (which the data shows):
    #     batch_max ≈ asymptote · t_dec_at_max
    # but t_dec at max is ITSELF a function of batch_max (compute-bound):
    #     t_dec = 2N·batch_max/(C·α) → asymptote = C·α/(2N) (independent of batch_max!)
    # so batch_max isn't well-defined as a "concurrency cap" in this regime —
    # the saturation comes from compute, not from a scheduler limit.
    # We instead report H (the half-saturation batch) directly as the
    # batch_50pct, and fit batch_max so that bs_eff(big batch) ≈ asymptote·t_dec.
    #
    # Empirically, the observed half-point H is small (~3-6 for 7B), meaning
    # vLLM reaches near-asymptote already at small batch. The saturation curve
    # is tight.
    #
    # batch_max ≈ asymptote · t_dec_typical, where t_dec_typical is at memory-
    # bound (small batch):  t_dec = W/(MBW·β) → batch_max = A·W/(MBW·β)
    t_dec_mem = W / (MBW_eff * beta_assumed)
    batch_max_mem = A * t_dec_mem

    return {
        "model": rep["model_tag"], "params_b": rep["params_b"], "tp": tp,
        "C_eff_TFLOPS": C_eff/1e12, "MBW_eff_GBs": MBW_eff/1e9, "W_GB": W/1e9,
        "asymptote_obs": A, "half_point_obs": H, "fit_sse": err,
        "alpha_from_asymptote": alpha_eff,
        "beta_assumed": beta_assumed,
        "batch_max_at_beta_assumed": batch_max_mem,
        "batches": batches, "throughputs": throughputs,
    }


def main():
    runs = parse_runs()
    groups = {}
    for r in runs:
        key = (r["model_tag"], r["tp"])
        groups.setdefault(key, []).append(r)

    print(f"{'config':>22}  {'n':>2} {'asympt(t/s)':>11} {'half_pt':>7} {'α(eff)':>7} {'α(per-card)':>11} {'batch_max':>10}")
    print("-" * 95)
    for key in sorted(groups.keys()):
        group = groups[key]
        if len(group) < 2:
            continue
        cfg = fit_config(group)
        per_card_alpha = cfg["alpha_from_asymptote"] / cfg["tp"]
        cfg_str = f"{cfg['model']}-tp{cfg['tp']}"
        print(f"{cfg_str:>22}  {len(group):>2} {cfg['asymptote_obs']:>11.0f} "
              f"{cfg['half_point_obs']:>7.2f} "
              f"{cfg['alpha_from_asymptote']:>7.3f} {per_card_alpha:>11.3f} "
              f"{cfg['batch_max_at_beta_assumed']:>10.1f}")
        # Show table of fit quality
        print(f"{'':>22}  fit:")
        for b, t in zip(cfg['batches'], cfg['throughputs']):
            pred = cfg['asymptote_obs'] * b / (b + cfg['half_point_obs'])
            err_pct = 100 * (pred - t) / t
            print(f"{'':>22}    batch={b:>5}  obs={t:>8.0f}  pred={pred:>8.0f}  err={err_pct:+5.1f}%")
        print()

    # Also derive batch_50pct and tp_efficiency
    print("\n=== Multi-config interpretation ===\n")
    print("Aggregate token throughput SATURATION CURVE (Hill fit):")
    print("    agg(batch) = asymptote · batch / (batch + half_point)")
    print()
    print("This is the curve to encode in engines.py['vllm'].batch_saturation,")
    print("with batch_50pct = half_point and batch_max derived from asymptote.")
    print()

    # TP-efficiency check
    a7_tp1 = next((g for k, g in groups.items() if k == ("qwen7b", 1)), None)
    a7_tp2 = next((g for k, g in groups.items() if k == ("qwen7b", 2)), None)
    if a7_tp1 and a7_tp2:
        cfg1 = fit_config(a7_tp1)
        cfg2 = fit_config(a7_tp2)
        speedup = cfg2["asymptote_obs"] / cfg1["asymptote_obs"]
        tp_eff = speedup / 2.0
        print(f"7B TP=1 asymptote: {cfg1['asymptote_obs']:.0f} t/s")
        print(f"7B TP=2 asymptote: {cfg2['asymptote_obs']:.0f} t/s")
        print(f"  TP-2 speedup vs TP-1 = {speedup:.2f}x → tp_efficiency = {tp_eff:.2f}")


if __name__ == "__main__":
    main()
