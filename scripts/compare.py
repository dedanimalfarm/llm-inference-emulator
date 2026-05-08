#!/usr/bin/env python3
"""Compare emulator predictions vs observed leaderboard values.

Three modes:
  uncalibrated — single literature default (PyTorch)
  coarse       — calibrated by (hw, backend, precision_label)
  fine         — calibrated by (hw, backend, precision_label, attention)

Reports median and p90 of relative error per (hw, mode) and overall.
Also drops rows where the model didn't fit in memory — those reflect paging,
not the inference engine, and are noise here.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.formula import predict
from emulator.hardware import get_peak_compute, get_memory_bandwidth
from emulator.engines import ENGINE_DEFAULTS
from emulator.calibrate import PRECISION_BITS, P_IN, P_OUT, BATCH

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

HW_FILES = {
    "1xA10":      "leaderboard-1xA10.csv",
    "1xA100":     "leaderboard-1xA100.csv",
    "1xT4":       "leaderboard-1xT4.csv",
    "32vCPU-C7i": "leaderboard-32vCPU-C7i.csv",
    "RTX-3090":      "leaderboard-RTX-3090-llamacpp.csv",
}
HW_CAPACITY_MB = {"1xA10": 24576, "1xA100": 81920, "1xT4": 16384, "32vCPU-C7i": 65536, "RTX-3090": 24576}


def relative_error(pred, obs):
    return np.abs(pred - obs) / obs


def predict_row(row, hw, alpha, beta):
    n = float(row["Params (B)"])
    if "_effective_bpw" in row and not pd.isna(row["_effective_bpw"]):
        bits = float(row["_effective_bpw"])
    else:
        bits = PRECISION_BITS.get(row["Quantization 🗜️"], 16)
    return predict(
        n_params_b=n, bits=bits,
        p_in=P_IN, p_out=P_OUT, batch=BATCH,
        peak_flops=get_peak_compute(hw, bits),
        mem_bw=get_memory_bandwidth(hw),
        alpha=alpha, beta=beta,
    )


def evaluate(rows, alpha_lookup, beta_lookup, mode):
    out = []
    for hw, df in rows.items():
        for _, r in df.iterrows():
            quant = r["Quantization 🗜️"]
            backend = r["Backend 🏭"]
            attn = r["Attention 👁️"]
            a = alpha_lookup(hw, backend, quant, attn)
            b = beta_lookup(hw, backend, quant, attn)
            if a is None or b is None or pd.isna(a) or pd.isna(b):
                continue
            try:
                pred = predict_row(r, hw, a, b)
            except Exception:
                continue
            t_obs_prefill = float(r["Prefill (s)"])
            tps_obs = float(r["Decode (tokens/s)"])
            if t_obs_prefill <= 0 or tps_obs <= 0:
                continue
            tps_pred = 1.0 / pred.decode_per_token_s
            out.append({
                "mode": mode, "hw": hw, "model": r["Model 🤗"],
                "precision": quant, "attention": attn, "params_b": r["Params (B)"],
                "alpha_used": a, "beta_used": b,
                "prefill_obs": t_obs_prefill, "prefill_pred": pred.prefill_s,
                "prefill_rel_err": relative_error(pred.prefill_s, t_obs_prefill),
                "decode_tps_obs": tps_obs, "decode_tps_pred": tps_pred,
                "decode_rel_err": relative_error(tps_pred, tps_obs),
            })
    return pd.DataFrame(out)


def main():
    per_hw = {}
    for hw, fn in HW_FILES.items():
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[df["Memory (MB)"] <= 0.95 * HW_CAPACITY_MB[hw]]
            per_hw[hw] = df

    coarse = pd.read_csv(os.path.join(RESULTS_DIR, "calibrated_coefficients.csv"))
    coarse_lookup = {(r["hw"], r["backend"], r["precision_label"]):
                     (r["alpha_median"], r["beta_median"])
                     for _, r in coarse.iterrows()}

    fine = pd.read_csv(os.path.join(RESULTS_DIR, "calibrated_fine.csv"))
    fine_lookup = {(r["hw"], r["backend"], r["precision_label"], r["attention"]):
                   (r["alpha_median"], r["beta_median"])
                   for _, r in fine.iterrows()}

    # ---- A) uncalibrated default ----
    def_a = ENGINE_DEFAULTS["pytorch"]["alpha"]
    def_b = ENGINE_DEFAULTS["pytorch"]["beta"]
    df_def = evaluate(per_hw,
        alpha_lookup=lambda hw, e, p, a: def_a,
        beta_lookup=lambda hw, e, p, a: def_b,
        mode="uncalibrated")

    # ---- B) coarse calibration ----
    def cb_a(hw, e, p, a): v = coarse_lookup.get((hw, e, p)); return v[0] if v else None
    def cb_b(hw, e, p, a): v = coarse_lookup.get((hw, e, p)); return v[1] if v else None
    df_coarse = evaluate(per_hw, cb_a, cb_b, mode="coarse")

    # ---- C) fine calibration ----
    def fn_a(hw, e, p, a): v = fine_lookup.get((hw, e, p, a)); return v[0] if v else None
    def fn_b(hw, e, p, a): v = fine_lookup.get((hw, e, p, a)); return v[1] if v else None
    df_fine = evaluate(per_hw, fn_a, fn_b, mode="fine")

    combined = pd.concat([df_def, df_coarse, df_fine], ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "prediction_vs_actual.csv"), index=False)

    print("=" * 84)
    print("Median relative error  (1.0 = predicted is 100% off)")
    print("=" * 84)
    summary = (combined.groupby(["mode", "hw"])
               .agg(rows=("prefill_rel_err", "count"),
                    prefill_med=("prefill_rel_err", "median"),
                    prefill_p90=("prefill_rel_err", lambda s: s.quantile(0.9)),
                    decode_med=("decode_rel_err", "median"),
                    decode_p90=("decode_rel_err", lambda s: s.quantile(0.9)))
               .round(3))
    print(summary)
    print()
    overall = (combined.groupby("mode")
               .agg(prefill_med=("prefill_rel_err", "median"),
                    decode_med=("decode_rel_err", "median"),
                    n=("prefill_rel_err", "count"))
               .round(3))
    print("Overall:")
    print(overall)
    summary.to_csv(os.path.join(RESULTS_DIR, "error_summary_by_hw.csv"))
    overall.to_csv(os.path.join(RESULTS_DIR, "error_summary_overall.csv"))


if __name__ == "__main__":
    main()
