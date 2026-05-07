#!/usr/bin/env python3
"""CLI for the inference emulator.

Usage:
  python scripts/cli.py --model 7 --bits 16 --hw 1xA100 --engine vllm \\
                        --p-in 256 --p-out 64 --batch 8

If a calibrated_coefficients.csv exists in results/, it overrides the
literature default for matching (hw, engine, precision) buckets.
"""
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.formula import predict
from emulator.hardware import get_peak_compute, get_memory_bandwidth, HARDWARE_SPECS
from emulator.engines import ENGINE_DEFAULTS

CAL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "calibrated_coefficients.csv")


def get_calibrated(hw, engine, precision_label):
    if not os.path.exists(CAL_PATH):
        return None, None
    cal = pd.read_csv(CAL_PATH)
    sub = cal[(cal["hw"] == hw) & (cal["backend"] == engine)
              & (cal["precision_label"] == precision_label)]
    if sub.empty:
        return None, None
    r = sub.iloc[0]
    return float(r["alpha_median"]), float(r["beta_median"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=float, required=True, help="params in B (e.g. 7)")
    p.add_argument("--bits", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--hw", required=True, choices=list(HARDWARE_SPECS.keys()))
    p.add_argument("--engine", default="pytorch", choices=list(ENGINE_DEFAULTS.keys()))
    p.add_argument("--p-in", type=int, default=256)
    p.add_argument("--p-out", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--precision-label", default=None,
                   help="quantization label for calibration lookup, e.g. 'Unquantized', 'GPTQ.4bit'")
    args = p.parse_args()

    eng = ENGINE_DEFAULTS[args.engine]
    alpha, beta = eng["alpha"], eng["beta"]
    if args.precision_label:
        a_cal, b_cal = get_calibrated(args.hw, args.engine, args.precision_label)
        if a_cal is not None:
            alpha, beta = a_cal, b_cal
            print(f"# using calibrated coefs for ({args.hw}, {args.engine}, {args.precision_label})")
        else:
            print(f"# no calibration found, using engine defaults")

    res = predict(
        n_params_b=args.model,
        bits=args.bits,
        p_in=args.p_in, p_out=args.p_out, batch=args.batch,
        peak_flops=get_peak_compute(args.hw, args.bits),
        mem_bw=get_memory_bandwidth(args.hw),
        alpha=alpha, beta=beta,
        batch_mult=eng["batch_mult"],
    )

    print(f"\n{args.hw} | {args.engine} | {args.bits}-bit | {args.model}B params | bs={args.batch}")
    print(f"  alpha={alpha:.3f}, beta={beta:.3f}, batch_mult={eng['batch_mult']}")
    print(f"  prefill          : {res.prefill_s*1000:.1f} ms ({res.bottleneck_prefill}-bound)")
    print(f"  decode/token     : {res.decode_per_token_s*1000:.2f} ms ({res.bottleneck_decode}-bound)")
    print(f"  total latency    : {res.total_latency_s:.3f} s for {args.p_out} new tokens")
    print(f"  throughput       : {res.throughput_tok_s:.1f} tok/s (effective batch ×{eng['batch_mult']})")
    print(f"  approx. memory   : {res.memory_gb:.2f} GB (weights + KV)")
    cap = HARDWARE_SPECS[args.hw]["memory_capacity_gb"]
    if res.memory_gb > cap:
        print(f"  ! WARN: exceeds {args.hw} memory capacity ({cap:.1f} GB)")


if __name__ == "__main__":
    main()
