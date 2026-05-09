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
    # size_b: {layers, d_model, kv_heads}
    # kv_heads defaults to None (means same as heads, which is d_model/128)
    1.0:  {"layers": 22, "d_model": 2048, "kv_heads": 32}, # generic
    1.5:  {"layers": 28, "d_model": 1536, "kv_heads": 2},  # Qwen2.5-1.5B
    3.0:  {"layers": 36, "d_model": 2048, "kv_heads": 2},  # Qwen2.5-3B
    7.0:  {"layers": 28, "d_model": 3584, "kv_heads": 4},  # Qwen2.5-7B
    8.0:  {"layers": 32, "d_model": 4096, "kv_heads": 8},  # Llama-3.1-8B
    13.0: {"layers": 40, "d_model": 5120, "kv_heads": 40}, # Llama-2-13B (Full)
    14.0: {"layers": 48, "d_model": 5120, "kv_heads": 8},  # Qwen2.5-14B
    34.0: {"layers": 48, "d_model": 7168, "kv_heads": 8},  # Yi-34B
    70.0: {"layers": 80, "d_model": 8192, "kv_heads": 8},  # Llama-3-70B
    110.0:{"layers": 80, "d_model": 8192, "kv_heads": 8},
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
    layers: Optional[int] = None,
    d_model: Optional[int] = None,
    kv_heads: Optional[int] = None,
    kv_bits_k: float = 16.0,
    kv_bits_v: float = 16.0,
    tp_size: int = 1,
    tp_efficiency: float = 1.0,
    prefix_cache_hit: float = 0.0,
    batch_saturation: Optional[tuple] = None,  # (batch_max, batch_50pct)
    kv_packing_eff: float = 1.0,
    speculative: bool = False,
    spec_accept_rate: float = 0.7,
    spec_k_proposed: int = 4,
    spec_overhead: float = 0.15,
    compute_path: int = 16,
) -> InferenceResult:
    """Predict inference performance.
    
    prefix_cache_hit: ratio of prompt tokens served from cache
    batch_saturation: (batch_max, batch_50pct) for MFU scaling
    kv_packing_eff: efficiency of PagedAttention allocation
    speculative: enable speculative decoding mode
    spec_accept_rate: ratio of accepted draft tokens
    spec_k_proposed: tokens proposed by draft model
    spec_overhead: relative cost of draft model pass
    compute_path: bits for peak_flops lookup
    """
    if layers is None or d_model is None:
        arch = _arch_for(n_params_b)
        layers = arch["layers"] if layers is None else layers
        d_model = arch["d_model"] if d_model is None else d_model
        kv_heads = arch.get("kv_heads") if kv_heads is None else kv_heads

    if not 0.0 <= prefix_cache_hit <= 1.0:
        raise ValueError(f"prefix_cache_hit must be in [0, 1], got {prefix_cache_hit}")
    if not 0.0 < kv_packing_eff <= 1.0:
        raise ValueError(f"kv_packing_eff must be in (0, 1], got {kv_packing_eff}")

    N = n_params_b * 1e9
    W = N * bits / 8.0  # weights in bytes

    # effective resources for multi-GPU TP
    eff_flops = peak_flops * tp_size * tp_efficiency
    eff_mbw = mem_bw * tp_size * tp_efficiency

    # ---- Effective concurrent batch (continuous-batching saturation) ----
    # Scheduler can keep `batch_max` slots filled; saturation curve is a
    # Hill-style approximation: at batch == batch_50pct the effective
    # concurrency is half of batch_max; at batch >> batch_50pct it asymptotes.
    # Computed BEFORE the timing terms because both prefill and decode compute
    # scale with the *active* batch, not the queued batch.
    if batch_saturation is not None:
        batch_max, batch_50pct = batch_saturation
        bs_eff = batch_max * batch / (batch + batch_50pct)
    else:
        bs_eff = float(batch)

    # ---- prefill ----
    # Prefix caching: cached tokens skip the matmul entirely. Memory term
    # (weight load) is unaffected — weights still have to be read once.
    # Compute term scales with bs_eff (active concurrency), not raw batch.
    p_in_eff = p_in * (1.0 - prefix_cache_hit)
    t_pre_compute = 2.0 * N * p_in_eff * bs_eff / (eff_flops * alpha)
    t_pre_mem     = W / eff_mbw
    if t_pre_compute >= t_pre_mem:
        t_pre, b_pre = t_pre_compute, "compute"
    else:
        t_pre, b_pre = t_pre_mem, "memory"

    # ---- decode (single token, static base) ----
    # Memory term is independent of batch (weights read once per step).
    # Compute term scales with bs_eff (active concurrent requests per step).
    t_dec_mem     = W / (eff_mbw * beta)
    t_dec_compute = 2.0 * N * bs_eff / (eff_flops * alpha)
    if t_dec_mem >= t_dec_compute:
        t_dec_base, b_dec = t_dec_mem, "memory"
    else:
        t_dec_base, b_dec = t_dec_compute, "compute"

    # ---- KV-cache cost averaged over the response ----
    # K and V can be stored at different precisions (e.g. llama.cpp -ctk Q8_0 -ctv Q4_0)
    avg_ctx = p_in + p_out / 2.0

    # head_dim is usually 128 or d_model/heads.
    # For simplicity in this roofline, we use d_model and kv_heads/total_heads ratio.
    # But llama.cpp/GGUF uses: layers * kv_heads * head_dim * 2 (for K and V) * bytes_per_element
    # We'll assume head_dim = 128 as it's nearly universal for these models.
    if kv_heads is not None:
        kv_per_token_bytes = layers * kv_heads * 128 * (kv_bits_k + kv_bits_v) / 8.0
    else:
        # Fallback to full attention if kv_heads not specified
        kv_per_token_bytes = layers * d_model * (kv_bits_k + kv_bits_v) / 8.0

    t_kv = kv_per_token_bytes * avg_ctx / (eff_mbw * beta)
    t_dec = t_dec_base + t_kv

    # ---- Speculative decoding ----
    # Each main-model step costs (1 + overhead) extra (the draft pass), but
    # in expectation produces (accept_rate * k) accepted tokens per step.
    if speculative:
        accepted_per_step = spec_accept_rate * spec_k_proposed
        if accepted_per_step <= 0:
            raise ValueError("speculative: spec_accept_rate * spec_k_proposed must be > 0")
        t_dec_final = t_dec * (1.0 + spec_overhead) / accepted_per_step
    else:
        t_dec_final = t_dec

    # ---- Memory budget ----
    # Naive allocators reserve a worst-case KV buffer per request; PagedAttention
    # packs blocks tightly. kv_packing_eff = (used / allocated), so allocated =
    # actual_kv / kv_packing_eff. Default 1.0 keeps existing callers stable.
    kv_total = kv_per_token_bytes * (p_in + p_out) * batch / kv_packing_eff

    total_latency = t_pre + p_out * t_dec_final
    return InferenceResult(
        prefill_s=t_pre,
        decode_per_token_s=t_dec_final,
        total_latency_s=total_latency,
        throughput_tok_s=bs_eff / t_dec_final,
        memory_gb=(W + kv_total) / 1e9,
        bottleneck_prefill=b_pre,
        bottleneck_decode=b_dec,
    )
