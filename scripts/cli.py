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
    p.add_argument("--kv-bits-k", type=float, default=16,
                   help="bits per element for K-cache (e.g. 8 for Q8_0, 4 for Q4_0)")
    p.add_argument("--kv-bits-v", type=float, default=16,
                   help="bits per element for V-cache")
    p.add_argument("--cost-per-hour", type=float, default=None,
                   help="GPU rental cost in $/hour (e.g. 1.50 for A100 on RunPod); "
                        "if set, prints $/M tokens alongside throughput")
    p.add_argument("--active", type=float, default=None,
                   help="MoE: active params per token in billions (overrides ARCH_DEFAULTS). "
                        "FLOPS use this value; memory still uses --model.")
    p.add_argument("--head-dim", type=int, default=None,
                   help="KV head dimension (default 128; use 512 for MLA models "
                        "like DeepSeek V3/V4).")
    p.add_argument("--sliding-window", type=int, default=None,
                   help="Max KV cache size in tokens. Caps both per-step KV "
                        "read cost and total KV memory. Default: unbounded "
                        "(full attention).")
    args = p.parse_args()

    eng = ENGINE_DEFAULTS[args.engine]
    alpha, beta = eng["alpha"], eng["beta"]
    sources = []
    if args.precision_label:
        a_cal, b_cal = get_calibrated(args.hw, args.engine, args.precision_label)
        if a_cal is not None and a_cal == a_cal:
            alpha = a_cal
            sources.append("α=calibrated")
        if b_cal is not None and b_cal == b_cal:
            beta = b_cal
            sources.append("β=calibrated")
        else:
            sources.append("β=engine-default (unidentifiable from current data)")
    if sources:
        print(f"# {', '.join(sources)} for ({args.hw}, {args.engine}, {args.precision_label})")
    else:
        print(f"# using engine defaults (no precision label given)")

    # peak_flops lookup must use the engine's compute_path (the precision
    # at which actual matmuls happen), not the weight bits. AWQ/GPTQ kernels
    # dequantize to FP16 before GEMM, so weight=4-bit uses the FP16 path.
    compute_bits = eng.get("compute_path", args.bits)
    res = predict(
        n_params_b=args.model,
        bits=args.bits,
        p_in=args.p_in, p_out=args.p_out, batch=args.batch,
        peak_flops=get_peak_compute(args.hw, compute_bits),
        mem_bw=get_memory_bandwidth(args.hw),
        alpha=alpha, beta=beta,
        batch_saturation=eng["batch_saturation"],
        kv_packing_eff=eng["kv_packing_eff"],
        compute_path=compute_bits,
        kv_bits_k=args.kv_bits_k, kv_bits_v=args.kv_bits_v,
        tp_size=HARDWARE_SPECS[args.hw].get("tp_size", 1),
        tp_efficiency=HARDWARE_SPECS[args.hw].get("tp_efficiency", 1.0),
        n_active_b=args.active,
        head_dim=args.head_dim,
        sliding_window=args.sliding_window,
    )

    # MoE: figure out the effective active count for the printout (CLI override > ARCH_DEFAULTS > dense).
    from emulator.formula import _arch_for
    arch_lookup = _arch_for(args.model)
    active_b = args.active if args.active is not None else arch_lookup.get("n_active_b", args.model)
    moe_tag = f" | active={active_b}B" if active_b != args.model else ""
    print(f"\n{args.hw} | {args.engine} | {args.bits}-bit | {args.model}B params{moe_tag} | bs={args.batch}")
    print(f"  alpha={alpha:.3f}, beta={beta:.3f}, batch_saturation={eng['batch_saturation']}, kv_packing_eff={eng['kv_packing_eff']}")
    print(f"  prefill          : {res.prefill_s*1000:.1f} ms ({res.bottleneck_prefill}-bound)")
    print(f"  decode/token     : {res.decode_per_token_s*1000:.2f} ms ({res.bottleneck_decode}-bound)")
    print(f"  total latency    : {res.total_latency_s:.3f} s for {args.p_out} new tokens")
    print(f"  throughput       : {res.throughput_tok_s:.1f} tok/s")
    if args.cost_per_hour is not None:
        cost_per_million = args.cost_per_hour * 1e6 / (res.throughput_tok_s * 3600)
        print(f"  cost @ ${args.cost_per_hour:g}/hr : ${cost_per_million:.3f} per million tokens")
    print(f"  approx. memory   : {res.memory_gb:.2f} GB (weights + KV)")
    cap = HARDWARE_SPECS[args.hw]["memory_capacity_gb"]
    if res.memory_gb > cap:
        print(f"  ! WARN: exceeds {args.hw} memory capacity ({cap:.1f} GB)")


if __name__ == "__main__":
    main()
