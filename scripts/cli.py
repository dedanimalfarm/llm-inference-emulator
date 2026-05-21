#!/usr/bin/env python3
"""CLI for the inference emulator.

Usage:
  python scripts/cli.py --model 7 --bits 16 --hw 1xA100 --engine vllm \\
                        --p-in 256 --p-out 64 --batch 8

If a calibrated_coefficients.csv exists in results/, it overrides the
literature default for matching (hw, engine, precision) buckets.
"""
import argparse
import os
import sys
import json
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.formula import predict
from emulator.hardware import get_peak_compute, get_memory_bandwidth, HARDWARE_SPECS
from emulator.engines import ENGINE_DEFAULTS

CAL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "calibrated_coefficients.csv")

# Standard hourly GPU rental rates in $/hour for sweeps/recommendations
DEFAULT_GPU_COSTS = {
    "8xH100": 16.00,
    "1xA100": 2.00,
    "1xT4": 0.35,
    "1xA10": 1.00,
    "RTX-3090": 0.50,
    "RTX-4090": 0.80,
    "32vCPU-C7i": 0.15,
}


def get_calibrated(hw, engine, precision_label):
    if not os.path.exists(CAL_PATH):
        return None, None, None
    cal = pd.read_csv(CAL_PATH)
    sub = cal[(cal["hw"] == hw) & (cal["backend"] == engine)
              & (cal["precision_label"] == precision_label)]
    if sub.empty:
        return None, None, None
    r = sub.iloc[0]
    alpha = float(r["alpha_median"])
    beta = float(r["beta_median"])
    
    batch_sat = None
    if "batch_max" in r and not pd.isna(r["batch_max"]):
        batch_sat = (float(r["batch_max"]), float(r["batch_50pct"]))
    
    return alpha, beta, batch_sat


def parse_nlp_query(query_str: str) -> dict:
    """Deterministic rule-based NLP parser to extract emulator options from a natural language query."""
    params = {}
    query_lower = query_str.lower()

    # 1. Model size matching: e.g. "70B", "70b", "8 billion", "1.5 B", "Llama-3-70b", "deepseek 671b"
    model_match = re.search(r'\b(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\b', query_str)
    if model_match:
        params["model"] = float(model_match.group(1))

    # 2. Hardware profile matching (RTX-3090, 8xH100, etc.)
    for hw_key in HARDWARE_SPECS.keys():
        hw_pattern = re.sub(r'[-\s]+', r'[-\\s]*', hw_key.lower())
        if re.search(r'\b' + hw_pattern + r'\b', query_lower):
            params["hw"] = hw_key
            break
        # Common short aliases (e.g. "a100" -> "1xA100")
        alias = hw_key.lower()
        if alias.startswith("1x"):
            short_alias = alias[2:]
            if re.search(r'\b' + re.escape(short_alias) + r'\b', query_lower) and "hw" not in params:
                params["hw"] = hw_key

    # 3. Engine / Backend matching
    for eng_key in ENGINE_DEFAULTS.keys():
        if re.search(r'\b' + re.escape(eng_key.lower()) + r'\b', query_lower):
            params["engine"] = eng_key
            break

    # 4. Precision bits matching: e.g. "4bit", "8-bit", "16 bits", "awq", "gptq"
    bits_match = re.search(r'\b(4|8|16)\s*[-_]?bits?\b', query_lower)
    if bits_match:
        params["bits"] = int(bits_match.group(1))
    elif "awq" in query_lower or "gptq" in query_lower:
        params["bits"] = 4
        params["precision_label"] = "AWQ.4bit" if "awq" in query_lower else "GPTQ.4bit"

    # 5. Batch size matching: e.g. "batch size 32", "batch of 8", "bs=16"
    batch_match = re.search(r'\b(?:batch|bs)\s*(?:size)?\s*(?:of|=)?\s*(\d+)\b', query_lower)
    if batch_match:
        params["batch"] = int(batch_match.group(1))

    # 6. Prompt length matching: e.g. "prompt of 256", "prompt len 512", "prompt=128", "input of 1024"
    prompt_match = re.search(r'\b(?:prompt|in|input)\s*(?:length|size|len)?\s*(?:of|=)?\s*(\d+)\b', query_lower)
    if prompt_match:
        params["p_in"] = int(prompt_match.group(1))

    # 7. Output/generation length matching: e.g. "gen 64", "output size of 128", "output=256"
    out_match = re.search(r'\b(?:generate|gen|out|output)\s*(?:length|size|len)?\s*(?:of|=)?\s*(\d+)\b', query_lower)
    if out_match:
        params["p_out"] = int(out_match.group(1))

    # 8. Check if recommendation is requested in the text query
    if any(keyword in query_lower for keyword in ["recommend", "sweep", "best", "cheapest", "fastest"]):
        params["recommend"] = True

    return params


def main():
    p = argparse.ArgumentParser(description="LLM Inference Emulator CLI - Agent-Friendly Interface")
    p.add_argument("--model", type=float, default=None, help="params in B (e.g. 7)")
    p.add_argument("--bits", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--hw", default=None, choices=list(HARDWARE_SPECS.keys()))
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
                   help="KV head dimension (default 128; use 512 for MLA models like DeepSeek V3/V4).")
    p.add_argument("--sliding-window", type=int, default=None,
                   help="Max KV cache size in tokens. Caps both per-step KV read cost and total KV memory.")
    p.add_argument("--speculative", action="store_true",
                   help="Enable speculative decoding model (Leviathan 2022 + draft/verify roofline).")
    p.add_argument("--spec-accept", type=float, default=0.7,
                   help="Per-position acceptance probability p (default 0.7).")
    p.add_argument("--spec-k", type=int, default=4,
                   help="Number K of tokens proposed by the draft per cycle (default 4).")
    p.add_argument("--spec-draft-b", type=float, default=None,
                   help="Draft model size in billions. When set, draft cost is computed from roofline.")
    p.add_argument("--spec-draft-active-b", type=float, default=None,
                   help="MoE draft: active params per token in billions.")
    p.add_argument("--spec-overhead", type=float, default=0.15,
                   help="Legacy heuristic: draft cost as a fraction of one target decode step.")
    p.add_argument("--spec-verify-scale", type=float, default=0.05,
                   help="Per-K cost penalty on target verify pass. Default 0.05.")
    p.add_argument("--chunked-prefill", action="store_true",
                   help="Model vLLM/SGLang chunked prefill (each chunk pays its own weight-load tax).")
    p.add_argument("--chunk-size", type=int, default=2048,
                   help="Tokens per prefill chunk (default 2048; vLLM uses 512).")
    
    # New agentic parameters
    p.add_argument("--json", action="store_true", help="Output results in clean, parseable JSON format")
    p.add_argument("--markdown", action="store_true", help="Output results in a GitHub-Flavored Markdown table")
    p.add_argument("--recommend", action="store_true", help="Sweep configurations and recommend optimal setups")
    p.add_argument("--query", type=str, default=None, help="Query emulator using natural language conversational prompts")
    p.add_argument("--max-latency-ms", type=float, default=None, help="Filter: max decode step latency in ms (used with --recommend)")
    p.add_argument("--max-cost-per-hour", type=float, default=None, help="Filter: max cost per hour in USD (used with --recommend)")

    args = p.parse_args()

    # Handle Natural Language Query if present
    if args.query:
        nlp_params = parse_nlp_query(args.query)
        if not args.json and not args.markdown:
            print(f"# Conversational parse: parsed parameters: {nlp_params}")
        for k, v in nlp_params.items():
            setattr(args, k, v)

    # Perform CLI arg validation
    if not args.recommend:
        if args.model is None:
            p.error("--model is required for single prediction runs (or provide via --query).")
        if args.hw is None:
            p.error("--hw is required for single prediction runs (or provide via --query).")

    # -------------------------------------------------------------------------
    # Recommendation Mode (Sweep across all hardware/engines)
    # -------------------------------------------------------------------------
    if args.recommend:
        if args.model is None:
            p.error("--model parameter size is required to run a recommendation sweep.")
        
        results_sweep = []
        for hw_key, hw_spec in HARDWARE_SPECS.items():
            cap = hw_spec["memory_capacity_gb"]
            for eng_key, eng_spec in ENGINE_DEFAULTS.items():
                alpha_val = eng_spec["alpha"]
                beta_val = eng_spec["beta"]
                batch_sat_val = eng_spec["batch_saturation"]
                
                # Check for calibration data
                prec_lbl = args.precision_label or (
                    "GPTQ.4bit" if args.bits == 4 else
                    "AWQ.4bit" if args.bits == 4 else
                    "Unquantized"
                )
                a_cal, b_cal, bs_cal = get_calibrated(hw_key, eng_key, prec_lbl)
                if a_cal is not None and a_cal == a_cal:
                    alpha_val = a_cal
                if b_cal is not None and b_cal == b_cal:
                    beta_val = b_cal
                if bs_cal is not None:
                    batch_sat_val = bs_cal

                compute_bits = eng_spec.get("compute_path", args.bits)
                
                try:
                    res_p = predict(
                        n_params_b=args.model,
                        bits=args.bits,
                        p_in=args.p_in, p_out=args.p_out, batch=args.batch,
                        peak_flops=get_peak_compute(hw_key, compute_bits),
                        mem_bw=get_memory_bandwidth(hw_key),
                        alpha=alpha_val, beta=beta_val,
                        batch_saturation=batch_sat_val,
                        kv_packing_eff=eng_spec["kv_packing_eff"],
                        compute_path=compute_bits,
                        kv_bits_k=args.kv_bits_k, kv_bits_v=args.kv_bits_v,
                        tp_size=hw_spec.get("tp_size", 1),
                        tp_efficiency=hw_spec.get("tp_efficiency", 1.0),
                        n_active_b=args.active,
                        head_dim=args.head_dim,
                        sliding_window=args.sliding_window,
                        speculative=args.speculative,
                        spec_accept_rate=args.spec_accept,
                        spec_k_proposed=args.spec_k,
                        spec_overhead=args.spec_overhead,
                        spec_draft_n_params_b=args.spec_draft_b,
                        spec_draft_active_b=args.spec_draft_active_b,
                        spec_verify_scale=args.spec_verify_scale,
                        chunked_prefill=args.chunked_prefill,
                        chunk_size=args.chunk_size,
                    )
                    
                    cost_rate = args.cost_per_hour or DEFAULT_GPU_COSTS.get(hw_key, 0.0)
                    cost_per_m = 0.0
                    if cost_rate > 0 and res_p.throughput_tok_s > 0:
                        cost_per_m = (cost_rate * 1e6) / (res_p.throughput_tok_s * 3600)

                    results_sweep.append({
                        "hw": hw_key,
                        "engine": eng_key,
                        "res": res_p,
                        "cost_rate": cost_rate,
                        "cost_per_m": cost_per_m,
                        "memory_gb": res_p.memory_gb,
                        "capacity_gb": cap,
                        "decode_ms": res_p.decode_per_token_s * 1000,
                        "throughput": res_p.throughput_tok_s,
                        "feasible": res_p.memory_gb <= cap
                    })
                except Exception:
                    # Skip configurations throwing mathematical boundaries exceptions
                    continue

        # Filter feasibility and optional parameters
        feasible_runs = [r for r in results_sweep if r["feasible"]]
        if args.max_latency_ms is not None:
            feasible_runs = [r for r in feasible_runs if r["decode_ms"] <= args.max_latency_ms]
        if args.max_cost_per_hour is not None:
            feasible_runs = [r for r in feasible_runs if r["cost_rate"] <= args.max_cost_per_hour]

        # Formatting Output
        if args.json:
            out_list = []
            for r in results_sweep:
                out_list.append({
                    "hardware": r["hw"],
                    "engine": r["engine"],
                    "throughput_tok_s": r["throughput"],
                    "decode_per_token_ms": r["decode_ms"],
                    "vram_gb": r["memory_gb"],
                    "capacity_gb": r["capacity_gb"],
                    "feasible": r["feasible"],
                    "cost_per_hour": r["cost_rate"],
                    "cost_per_m_tokens": r["cost_per_m"]
                })
            print(json.dumps({
                "model_size_b": args.model,
                "precision_bits": args.bits,
                "max_latency_ms_filter": args.max_latency_ms,
                "max_cost_per_hour_filter": args.max_cost_per_hour,
                "sweep_results": out_list,
                "has_feasible_runs": len(feasible_runs) > 0
            }, indent=2))
            return

        if args.markdown:
            print(f"### 📊 Сводная таблица сравнения конфигураций ({args.model}B, {args.bits}-бит)")
            print("| Видеокарта | Движок | Пропускная способность | Декод (мс/ток) | Память VRAM (GB) | Стоимость ($/час) | Feasible |")
            print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in results_sweep:
                f_lbl = "✓ Да" if r["feasible"] else "✗ OOM"
                print(f"| **{r['hw']}** | {r['engine']} | {r['throughput']:.1f} tok/s | {r['decode_ms']:.2f} ms | {r['memory_gb']:.1f} / {r['capacity_gb']:.1f} | ${r['cost_rate']:.2f} | {f_lbl} |")
            return

        # Human-Readable recommendations report
        print(f"\n# Рекомендации по оптимизации для модели {args.model}B ({args.bits}-бит)")
        if args.max_latency_ms:
            print(f"# Фильтр задержки: Макс. шаг декода <= {args.max_latency_ms} мс")
        if args.max_cost_per_hour:
            print(f"# Фильтр цены: Макс. стоимость в час <= ${args.max_cost_per_hour:.2f}")

        if not feasible_runs:
            print("\n❌ Ни одна из доступных конфигураций оборудования не удовлетворяет заданным ограничениям VRAM или задержки.")
            print("Попробуйте использовать меньшую разрядность квантования (например, `--bits 4`).")
            return

        # Sort recommendations
        fastest = max(feasible_runs, key=lambda x: x["throughput"])
        cheapest = min([r for r in feasible_runs if r["cost_per_m"] > 0] or feasible_runs, key=lambda x: x["cost_per_m"])
        compact = min(feasible_runs, key=lambda x: x["memory_gb"])

        print(f"\n🚀 1. САМАЯ БЫСТРАЯ КОНФИГУРАЦИЯ (FASTEST):")
        print(f"  - Видеокарта: **{fastest['hw']}** | Движок: **{fastest['engine']}**")
        print(f"  - Пропускная способность: {fastest['throughput']:.1f} tok/s | Шаг декода: {fastest['decode_ms']:.2f} мс")
        print(f"  - Потребление памяти VRAM: {fastest['memory_gb']:.1f} GB из {fastest['capacity_gb']:.1f} GB ({fastest['memory_gb']/fastest['capacity_gb']*100:.1f}%)")
        
        print(f"\n💰 2. САМАЯ ВЫГОДНАЯ КОНФИГУРАЦИЯ (CHEAPEST):")
        print(f"  - Видеокарта: **{cheapest['hw']}** | Движок: **{cheapest['engine']}**")
        print(f"  - Стоимость миллиона токенов: ${cheapest['cost_per_m']:.3f} | Аренда: ${cheapest['cost_rate']:.2f}/час")
        print(f"  - Пропускная способность: {cheapest['throughput']:.1f} tok/s")

        print(f"\n📉 3. САМАЯ КОМПАКТНАЯ КОНФИГУРАЦИЯ (COMPACT):")
        print(f"  - Видеокарта: **{compact['hw']}** | Движок: **{compact['engine']}**")
        print(f"  - Минимальное использование VRAM: {compact['memory_gb']:.1f} GB")
        print(f"  - Пропускная способность: {compact['throughput']:.1f} tok/s")
        return

    # -------------------------------------------------------------------------
    # Single Prediction Run
    # -------------------------------------------------------------------------
    eng = ENGINE_DEFAULTS[args.engine]
    alpha, beta = eng["alpha"], eng["beta"]
    batch_saturation = eng["batch_saturation"]
    sources = []
    
    if args.precision_label:
        a_cal, b_cal, bs_cal = get_calibrated(args.hw, args.engine, args.precision_label)
        if a_cal is not None and a_cal == a_cal:
            alpha = a_cal
            sources.append("alpha=calibrated")
        if b_cal is not None and b_cal == b_cal:
            beta = b_cal
            sources.append("beta=calibrated")
        if bs_cal is not None:
            batch_saturation = bs_cal
            sources.append("batch_saturation=calibrated")

    compute_bits = eng.get("compute_path", args.bits)
    res = predict(
        n_params_b=args.model,
        bits=args.bits,
        p_in=args.p_in, p_out=args.p_out, batch=args.batch,
        peak_flops=get_peak_compute(args.hw, compute_bits),
        mem_bw=get_memory_bandwidth(args.hw),
        alpha=alpha, beta=beta,
        batch_saturation=batch_saturation,
        kv_packing_eff=eng["kv_packing_eff"],
        compute_path=compute_bits,
        kv_bits_k=args.kv_bits_k, kv_bits_v=args.kv_bits_v,
        tp_size=HARDWARE_SPECS[args.hw].get("tp_size", 1),
        tp_efficiency=HARDWARE_SPECS[args.hw].get("tp_efficiency", 1.0),
        n_active_b=args.active,
        head_dim=args.head_dim,
        sliding_window=args.sliding_window,
        speculative=args.speculative,
        spec_accept_rate=args.spec_accept,
        spec_k_proposed=args.spec_k,
        spec_overhead=args.spec_overhead,
        spec_draft_n_params_b=args.spec_draft_b,
        spec_draft_active_b=args.spec_draft_active_b,
        spec_verify_scale=args.spec_verify_scale,
        chunked_prefill=args.chunked_prefill,
        chunk_size=args.chunk_size,
    )

    cap = HARDWARE_SPECS[args.hw]["memory_capacity_gb"]
    exceeds_cap = res.memory_gb > cap

    # 1. Output JSON Format
    if args.json:
        warnings = []
        if exceeds_cap:
            warnings.append(f"Exceeds memory capacity by {res.memory_gb - cap:.2f} GB! Run will OOM.")
            
        out_dict = {
            "hardware": args.hw,
            "engine": args.engine,
            "model_size_b": args.model,
            "active_params_b": args.active if args.active is not None else args.model,
            "precision_bits": args.bits,
            "batch_size": args.batch,
            "prompt_len": args.p_in,
            "generation_len": args.p_out,
            "prefill_ms": round(res.prefill_s * 1000, 2),
            "prefill_bottleneck": res.bottleneck_prefill,
            "decode_per_token_ms": round(res.decode_per_token_s * 1000, 2),
            "decode_bottleneck": res.bottleneck_decode,
            "total_latency_s": round(res.total_latency_s, 3),
            "throughput_tokens_per_s": round(res.throughput_tok_s, 1),
            "total_vram_gb": round(res.memory_gb, 2),
            "exceeds_memory_capacity": exceeds_cap,
            "capacity_limit_gb": cap,
            "warnings": warnings
        }
        print(json.dumps(out_dict, indent=2))
        return

    # 2. Output Markdown Table Format
    if args.markdown:
        limit_txt = f"**! EXCEEDS CAPACITY ({cap:.1f} GB)**" if exceeds_cap else f"Feasible ({cap:.1f} GB Capacity)"
        print("| Parameter / Metric | Value | Bottleneck / Warning |")
        print("| :--- | :--- | :--- |")
        print(f"| **Hardware** | {args.hw} | {limit_txt} |")
        print(f"| **Engine** | {args.engine} | Compute precision bits: {compute_bits} |")
        print(f"| **Workload** | {args.model}B params (dense) | batch={args.batch}, in={args.p_in}, out={args.p_out} |")
        print(f"| **Prefill Latency** | {res.prefill_s*1000:.1f} ms | {res.bottleneck_prefill.capitalize()}-bound |")
        print(f"| **Decode Latency** | {res.decode_per_token_s*1000:.2f} ms/token | {res.bottleneck_decode.capitalize()}-bound |")
        print(f"| **Total Latency** | {res.total_latency_s:.3f} s | For {args.p_out} output tokens |")
        print(f"| **Throughput** | {res.throughput_tok_s:.1f} tok/s | - |")
        print(f"| **VRAM Footprint** | {res.memory_gb:.2f} GB | Required memory size |")
        return

    # 3. Standard Human-Readable Text Format (Backward compatible)
    if sources:
        print(f"# {', '.join(sources)} for ({args.hw}, {args.engine}, {args.precision_label})")
    else:
        print(f"# using engine defaults (no precision label given)")

    from emulator.formula import _arch_for
    arch_lookup = _arch_for(args.model)
    active_b = args.active if args.active is not None else arch_lookup.get("n_active_b", args.model)
    moe_tag = f" | active={active_b}B" if active_b != args.model else ""
    print(f"\n{args.hw} | {args.engine} | {args.bits}-bit | {args.model}B params{moe_tag} | bs={args.batch}")
    print(f"  alpha={alpha:.3f}, beta={beta:.3f}, batch_saturation={batch_saturation}, kv_packing_eff={eng['kv_packing_eff']}")
    print(f"  prefill          : {res.prefill_s*1000:.1f} ms ({res.bottleneck_prefill}-bound)")
    print(f"  decode/token     : {res.decode_per_token_s*1000:.2f} ms ({res.bottleneck_decode}-bound)")
    print(f"  total latency    : {res.total_latency_s:.3f} s for {args.p_out} new tokens")
    print(f"  throughput       : {res.throughput_tok_s:.1f} tok/s")
    if args.speculative:
        draft_label = (f"draft={args.spec_draft_b}B (physics)"
                       if args.spec_draft_b is not None
                       else f"overhead={args.spec_overhead:.2f} (heuristic)")
        print(f"  speculative      : p={args.spec_accept:.2f}, K={args.spec_k}, "
              f"{draft_label}, verify_scale={args.spec_verify_scale:.2f}")
        print(f"    E[accepted]    : {res.spec_e_accept:.3f} tokens / cycle")
        print(f"    speedup        : {res.spec_speedup:.2f}× vs non-speculative")
    
    cost_rate = args.cost_per_hour or DEFAULT_GPU_COSTS.get(args.hw, None)
    if cost_rate is not None:
        cost_per_million = cost_rate * 1e6 / (res.throughput_tok_s * 3600)
        print(f"  cost @ ${cost_rate:g}/hr : ${cost_per_million:.3f} per million tokens")
    
    print(f"  approx. memory   : {res.memory_gb:.2f} GB (weights + KV)")
    if exceeds_cap:
        print(f"  ! WARN: exceeds {args.hw} memory capacity ({cap:.1f} GB)")


if __name__ == "__main__":
    main()
