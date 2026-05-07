"""Roofline-based inference latency / throughput estimator.

Two phases of LLM inference are modelled separately:
  * prefill — usually compute-bound, processes all P_in tokens in parallel
  * decode  — usually memory-bound, generates tokens one by one

For each phase we take the slower of compute-time and memory-time.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceResult:
    prefill_s: float
    decode_per_token_s: float
    total_latency_s: float
    throughput_tok_s: float
    memory_gb: float
    bottleneck_prefill: str
    bottleneck_decode: str


# Rough heuristic shapes for typical decoder-only architectures.
# For finer accuracy, pass `layers` and `d_model` explicitly.
ARCH_DEFAULTS = {
    1.0:  {"layers": 22, "d_model": 2048},
    3.0:  {"layers": 26, "d_model": 3072},
    7.0:  {"layers": 32, "d_model": 4096},
    13.0: {"layers": 40, "d_model": 5120},
    34.0: {"layers": 48, "d_model": 7168},
    70.0: {"layers": 80, "d_model": 8192},
    110.0:{"layers": 80, "d_model": 8192},
}


def _arch_for(n_params_b: float):
    keys = sorted(ARCH_DEFAULTS.keys())
    chosen = keys[0]
    for k in keys:
        if n_params_b >= k:
            chosen = k
    return ARCH_DEFAULTS[chosen]


def predict(
    n_params_b: float,
    bits: float,
    p_in: int,
    p_out: int,
    batch: int,
    peak_flops: float,
    mem_bw: float,
    alpha: float = 0.25,
    beta: float = 0.60,
    batch_mult: float = 1.0,
    layers: Optional[int] = None,
    d_model: Optional[int] = None,
    kv_bits_k: float = 16.0,
    kv_bits_v: float = 16.0,
) -> InferenceResult:
    if layers is None or d_model is None:
        arch = _arch_for(n_params_b)
        layers = arch["layers"] if layers is None else layers
        d_model = arch["d_model"] if d_model is None else d_model

    N = n_params_b * 1e9
    W = N * bits / 8.0  # weights in bytes

    # ---- prefill ----
    t_pre_compute = 2.0 * N * p_in * batch / (peak_flops * alpha)
    t_pre_mem     = W / mem_bw
    if t_pre_compute >= t_pre_mem:
        t_pre, b_pre = t_pre_compute, "compute"
    else:
        t_pre, b_pre = t_pre_mem, "memory"

    # ---- decode (single token, static base) ----
    t_dec_mem     = W / (mem_bw * beta)
    t_dec_compute = 2.0 * N * batch / (peak_flops * alpha)
    if t_dec_mem >= t_dec_compute:
        t_dec_base, b_dec = t_dec_mem, "memory"
    else:
        t_dec_base, b_dec = t_dec_compute, "compute"

    # ---- KV-cache cost averaged over the response ----
    # K and V can be stored at different precisions (e.g. llama.cpp -ctk Q8_0 -ctv Q4_0)
    avg_ctx = p_in + p_out / 2.0
    kv_per_token_bytes = layers * d_model * (kv_bits_k + kv_bits_v) / 8.0
    t_kv = kv_per_token_bytes * avg_ctx / (mem_bw * beta)
    t_dec = t_dec_base + t_kv

    bs_eff = batch * batch_mult
    total_latency = t_pre + p_out * t_dec
    kv_total = kv_per_token_bytes * (p_in + p_out) * batch
    return InferenceResult(
        prefill_s=t_pre,
        decode_per_token_s=t_dec,
        total_latency_s=total_latency,
        throughput_tok_s=bs_eff / t_dec,
        memory_gb=(W + kv_total) / 1e9,
        bottleneck_prefill=b_pre,
        bottleneck_decode=b_dec,
    )
