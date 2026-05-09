"""Solve back the engine-specific MFU/MBU coefficients from observed data.

Given a row from the leaderboard with known (model size, precision, hardware)
and observed (prefill_s, decode_tps), invert the roofline equations:

    alpha = 2 * N * P_in * batch / (peak_flops * t_prefill)
    beta  = W_bytes / (mem_bw * (t_per_token - t_kv))

The benchmark scenario is fixed across all rows: P_in=256, P_out=64, bs=1.
"""
import pandas as pd
import numpy as np
from .hardware import get_peak_compute, get_memory_bandwidth, HARDWARE_SPECS
from .formula import _arch_for
from .engines import ENGINE_DEFAULTS


# benchmark scenario constants from the LLM-Perf Leaderboard
P_IN = 256
P_OUT = 64
BATCH = 1


PRECISION_BITS = {
    # nominal weight precision; for GPTQ/AWQ the on-disk size is ~4.25 bpw
    # because of scales/zeros, but the bandwidth-bound quantity is dominated
    # by the 4-bit weights themselves
    "Unquantized": 16,
    "BnB.4bit":    4,
    "BnB.8bit":    8,
    "GPTQ.4bit":   4,
    "AWQ.4bit":    4,
    "torchao.4bit": 4,
}


def effective_bpw(model_size_bytes: float, n_params: float) -> float:
    """Compute effective bits per weight from the actual on-disk size.

    Useful for GGUF / mixed-quantization formats (Q4_K_M ≈ 4.91, Q5_K_M ≈ 5.5,
    Q3_K_S ≈ 3.5, Q8_0 ≈ 8.5) where the nominal "4-bit" label hides scales,
    zeros and per-channel overhead.

    Example:
        >>> effective_bpw(4_677_120_000, 7_615_616_512)  # Qwen2.5-7B-Q4_K_M
        4.913...
    """
    return 8.0 * model_size_bytes / n_params


def calibrate_row(row, hw):
    """Return (alpha_prefill, beta_decode, t_kv_estimated) for one row."""
    n = float(row["Params (B)"])
    quant = row["Quantization 🗜️"]
    backend = row.get("Backend 🏭", "pytorch")
    
    if "_effective_bpw" in row and not pd.isna(row["_effective_bpw"]):
        bits = float(row["_effective_bpw"])
    else:
        bits = PRECISION_BITS.get(quant, 16)

    t_prefill = float(row["Prefill (s)"])
    decode_tps = float(row["Decode (tokens/s)"])
    if pd.isna(decode_tps) or decode_tps <= 0:
        return None

    t_per_token = 1.0 / decode_tps

    # Determine peak flops based on engine's compute path (BUG-1)
    engine_conf = ENGINE_DEFAULTS.get(backend, ENGINE_DEFAULTS["pytorch"])
    compute_bits = engine_conf.get("compute_path", 16)

    # Handle TP scaling (Step 4.7)
    tp_size = HARDWARE_SPECS[hw].get("tp_size", 1)
    tp_eff = HARDWARE_SPECS[hw].get("tp_efficiency", 1.0)

    C = get_peak_compute(hw, compute_bits) * tp_size * tp_eff
    MBW = get_memory_bandwidth(hw) * tp_size * tp_eff

    N = n * 1e9
    W = N * bits / 8

    p_in = float(row["_n_prompt"]) if "_n_prompt" in row and not pd.isna(row["_n_prompt"]) else P_IN

    # ---- vLLM aggregate-throughput path ----
    # vLLM bench reports aggregate prompt/decode throughput across all concurrent
    # requests. The single-request inversion below would conflate batch concurrency
    # with MFU/MBU. For vLLM rows, calibrate α directly from aggregate prompt
    # throughput (independent of batching in compute-bound regime) and leave β as
    # NaN until a multi-batch sweep makes (β, batch_saturation) jointly fittable.
    is_vllm = (backend == "vllm")
    prompt_tps_agg = float(row.get("_prompt_tps_agg", float("nan"))) if "_prompt_tps_agg" in row else float("nan")

    # KV cache part (estimated from arch) (BUG-2)
    arch = _arch_for(n)
    layers = arch["layers"]
    d_model = arch["d_model"]
    kv_heads = arch.get("kv_heads")

    n_depth = float(row.get("_n_depth", 0)) if not pd.isna(row.get("_n_depth")) else 0
    n_gen = float(row.get("_n_gen", P_OUT)) if not pd.isna(row.get("_n_gen")) else P_OUT
    avg_ctx = n_depth + n_gen / 2

    # head_dim is usually 128
    if kv_heads is not None:
        kv_per_token_bytes = layers * kv_heads * 128 * 2 * 2 # K+V, FP16
    else:
        kv_per_token_bytes = layers * d_model * 2 * 2

    if is_vllm:
        # α from aggregate prompt throughput in compute-bound regime:
        #     α = 2·N·prompt_throughput_aggregate / C_eff
        # No batch dependence: each prefill token costs 2N FLOPS regardless
        # of how many concurrent requests share the GPU.
        if not pd.isna(prompt_tps_agg) and prompt_tps_agg > 0:
            alpha = (2.0 * N * prompt_tps_agg) / C
        else:
            alpha = float("nan")
        # β unidentifiable from single-batch vLLM data — see header comment.
        beta = float("nan")
    else:
        # Legacy LLM-Perf path (single-request, batch=1): existing inversion.
        alpha = float("nan")
        if not pd.isna(t_prefill) and t_prefill > 0:
            t_pre_mem_floor = W / MBW
            if t_prefill <= t_pre_mem_floor * 1.05:
                alpha = float("nan")
            else:
                alpha = (2 * N * p_in * BATCH) / (C * t_prefill)
        # for beta we factor out the KV term; assume KV reads use the same MBU
        # so the equation t_token = (W / (MBW*beta)) + (kv*ctx / (MBW*beta))
        # => beta = (W + kv*ctx) / (MBW * t_per_token)
        beta = (W + kv_per_token_bytes * avg_ctx) / (MBW * t_per_token)

    return {
        "alpha_prefill": alpha,
        "beta_decode":   beta,
        "t_per_token_s": t_per_token,
        "weights_gb":    W / 1e9,
    }


def calibrate(per_hw_df: dict) -> pd.DataFrame:
    rows = []
    for hw, df in per_hw_df.items():
        for _, r in df.iterrows():
            res = calibrate_row(r, hw)
            if res is None:
                continue
            rows.append({
                "hw": hw,
                "model": r.get("Model 🤗"),
                "params_b": r.get("Params (B)"),
                "precision_label": r.get("Quantization 🗜️"),
                "attention": r.get("Attention 👁️"),
                "kernel": r.get("Kernel ⚛️"),
                "backend": r.get("Backend 🏭"),
                "prefill_s_obs": r.get("Prefill (s)"),
                "decode_tps_obs": r.get("Decode (tokens/s)"),
                "_n_batch": r.get("_n_batch"),
                "_n_depth": r.get("_n_depth"),
                "_n_prompt": r.get("_n_prompt"),
                "_n_threads": r.get("_n_threads"),
                **res,
            })
    return pd.DataFrame(rows)


def filter_outliers(calib_df: pd.DataFrame,
                    alpha_range=(0.005, 1.0),
                    beta_range=(0.005, 1.0)) -> pd.DataFrame:
    """Drop rows whose alpha/beta are physically implausible.

    Values >1.0 mean our peak-FLOPS / MBW assumption is wrong for that case
    (e.g. INT4 kernels on H100 use 1248 TFLOPS, not 312, and the row's α
    breaks the limit; or the row's decode_tps was an aggregate across many
    concurrent requests — a calibration math bug, not a physical signal).
    Values <0.005 mean the model didn't actually fit — observed time was
    dominated by paging.

    NaN α or β passes through (used for vLLM rows where one of the two is
    deliberately not solvable from single-batch data — downstream aggregation
    just skips NaN cells in the median).
    """
    alpha_ok = calib_df["alpha_prefill"].isna() | calib_df["alpha_prefill"].between(*alpha_range)
    beta_ok  = calib_df["beta_decode"].isna()  | calib_df["beta_decode"].between(*beta_range)
    return calib_df[alpha_ok & beta_ok].copy()


def aggregate(calib_df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Aggregate calibrated coefficients by chosen group columns."""
    if by is None:
        by = ["hw", "backend", "precision_label"]
    grouped = calib_df.groupby(by).agg(
        n_rows=("alpha_prefill", "count"),
        alpha_median=("alpha_prefill", "median"),
        alpha_p25=("alpha_prefill", lambda s: s.quantile(0.25)),
        alpha_p75=("alpha_prefill", lambda s: s.quantile(0.75)),
        beta_median=("beta_decode", "median"),
        beta_p25=("beta_decode", lambda s: s.quantile(0.25)),
        beta_p75=("beta_decode", lambda s: s.quantile(0.75)),
    ).reset_index()
    return grouped.round(4)
