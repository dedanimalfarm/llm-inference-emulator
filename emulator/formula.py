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
    # Keyed by total parameter count in billions. Each entry has:
    #   layers, d_model, kv_heads — required, dense baseline (head_dim=128).
    # Optional MoE / architectural extras:
    #   n_active_b — active params per token. If absent, equals key (dense).
    #   head_dim   — KV head dimension (default 128). MLA models override (e.g. 512).
    #   sliding_window — KV cache cap (default None = unbounded; applied in
    #                    sliding-window support, currently informational only).
    # Dense models
    1.0:  {"layers": 22, "d_model": 2048, "kv_heads": 32}, # generic
    1.5:  {"layers": 28, "d_model": 1536, "kv_heads": 2},  # Qwen2.5-1.5B
    3.0:  {"layers": 36, "d_model": 2048, "kv_heads": 2},  # Qwen2.5-3B
    7.0:  {"layers": 28, "d_model": 3584, "kv_heads": 4},  # Qwen2.5-7B
    8.0:  {"layers": 32, "d_model": 4096, "kv_heads": 8},  # Llama-3.1-8B
    13.0: {"layers": 40, "d_model": 5120, "kv_heads": 40}, # Llama-2-13B (Full)
    14.0: {"layers": 48, "d_model": 5120, "kv_heads": 8},  # Qwen2.5-14B
    32.0: {"layers": 64, "d_model": 5120, "kv_heads": 8},  # Qwen3-32B (dense)
    34.0: {"layers": 48, "d_model": 7168, "kv_heads": 8},  # Yi-34B
    70.0: {"layers": 80, "d_model": 8192, "kv_heads": 8},  # Llama-3-70B
    110.0:{"layers": 80, "d_model": 8192, "kv_heads": 8},
    # MoE models — n_active_b set means FLOPS use the active count while
    # memory keeps using the total. Defaults match published configs.
    46.7: {"layers": 32, "d_model": 4096, "kv_heads": 8,
           "n_active_b": 12.9},                                # Mixtral 8x7B (8 exp, top-2)
    109.0:{"layers": 48, "d_model": 5120, "kv_heads": 8,
           "sliding_window": 8192, "n_active_b": 17.0},        # Llama 4 Scout (16 exp top-1, iRoPE+SW8k)
    141.0:{"layers": 56, "d_model": 6144, "kv_heads": 8,
           "n_active_b": 39.0},                                # Mixtral 8x22B (8 exp, top-2)
    235.0:{"layers": 94, "d_model": 4096, "kv_heads": 4,
           "n_active_b": 22.0},                                # Qwen3-235B-A22B (128 exp, top-8)
    284.0:{"layers": 43, "d_model": 4096, "kv_heads": 1,
           "head_dim": 512, "sliding_window": 128,
           "n_active_b": 13.0},                                # DeepSeek-V4-Flash (MLA, 256 exp, top-6)
    671.0:{"layers": 61, "d_model": 7168, "kv_heads": 1,
           "head_dim": 128,
           "n_active_b": 37.0},                                # DeepSeek-V3 (MLA, 256 exp, top-8)
    1600.0:{"layers": 61, "d_model": 7168, "kv_heads": 1,
            "head_dim": 512, "sliding_window": 128,
            "n_active_b": 49.0},                               # DeepSeek-V4-Pro (MLA, 384 exp, top-6)
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
    n_active_b: Optional[float] = None,
    head_dim: Optional[int] = None,
    sliding_window: Optional[int] = None,
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
    n_active_b: for MoE models, active params per token (billions). Defaults
                to n_params_b (dense). Memory still uses n_params_b (all
                experts in VRAM); FLOPS use n_active_b (only the routed
                experts execute per token).
    head_dim: KV head dimension. Defaults to 128 (universal for current
              dense LLMs). MLA-style models (DeepSeek V3/V4) override
              with larger values (typically 512) reflecting compressed
              KV latents.
    sliding_window: max number of tokens kept in the KV cache. If set,
              both per-step KV read cost and total KV memory are capped
              at this value. Defaults to None (unbounded) for classic
              full attention; Mistral/Gemma/DeepSeek-V4 set 128–4096.
    """
    # Pull architectural defaults if not explicitly provided.
    arch_lookup_needed = (layers is None or d_model is None or
                         n_active_b is None or head_dim is None or
                         sliding_window is None)
    if arch_lookup_needed:
        arch = _arch_for(n_params_b)
        layers = arch["layers"] if layers is None else layers
        d_model = arch["d_model"] if d_model is None else d_model
        kv_heads = arch.get("kv_heads") if kv_heads is None else kv_heads
        if n_active_b is None:
            n_active_b = arch.get("n_active_b", n_params_b)
        if head_dim is None:
            head_dim = arch.get("head_dim", 128)
        if sliding_window is None:
            sliding_window = arch.get("sliding_window")  # may stay None
    if n_active_b is None:
        n_active_b = n_params_b
    if head_dim is None:
        head_dim = 128

    if not 0.0 <= prefix_cache_hit <= 1.0:
        raise ValueError(f"prefix_cache_hit must be in [0, 1], got {prefix_cache_hit}")
    if not 0.0 < kv_packing_eff <= 1.0:
        raise ValueError(f"kv_packing_eff must be in (0, 1], got {kv_packing_eff}")
    if n_active_b > n_params_b:
        raise ValueError(f"n_active_b ({n_active_b}B) must be <= n_params_b ({n_params_b}B)")

    N = n_params_b * 1e9
    N_active = n_active_b * 1e9
    W = N * bits / 8.0  # weights in bytes

    # effective resources for multi-GPU TP
    eff_flops = peak_flops * tp_size * tp_efficiency
    eff_mbw = mem_bw * tp_size * tp_efficiency

    # ---- Effective concurrent batch (continuous-batching saturation) ----
    # Hill curve bounded by the physical floor and ceiling:
    #   - cannot exceed the queue depth `batch` (no parallelism out of thin air)
    #   - cannot fall below min(1, batch) (a queued request always counts as 1)
    # At batch >> batch_50pct the Hill term dominates and asymptotes to
    # `batch_max` (the scheduler cap).
    if batch_saturation is not None:
        batch_max, batch_50pct = batch_saturation
        hill = batch_max * batch / (batch + batch_50pct)
        floor = min(1.0, float(batch))
        bs_eff = min(float(batch), max(floor, hill))
    else:
        bs_eff = float(batch)

    # ---- prefill ----
    # Prefix caching: cached tokens skip the matmul entirely. Memory term
    # (weight load) is unaffected — weights still have to be read once.
    # Compute term scales with bs_eff (active concurrency), not raw batch.
    # For MoE: FLOPS use N_active (only routed experts execute per token);
    # memory still uses W (all experts live in VRAM).
    p_in_eff = p_in * (1.0 - prefix_cache_hit)
    t_pre_compute = 2.0 * N_active * p_in_eff * bs_eff / (eff_flops * alpha)
    t_pre_mem     = W / eff_mbw
    if t_pre_compute >= t_pre_mem:
        t_pre, b_pre = t_pre_compute, "compute"
    else:
        t_pre, b_pre = t_pre_mem, "memory"

    # ---- decode (single token, static base) ----
    # Memory term is independent of batch (weights read once per step).
    # Compute term scales with bs_eff (active concurrent requests per step).
    # MoE: per-step weight read scales with active fraction. At decode the
    # router picks top-k experts per token, and only those expert weights
    # cross the HBM boundary. Dense models have N_active==N → factor 1.
    # Note: prefill is treated separately above; with P_in tokens routing
    # is likely to touch every expert, so t_pre_mem keeps using full W.
    weight_read_fraction = N_active / N
    t_dec_mem     = W * weight_read_fraction / (eff_mbw * beta)
    t_dec_compute = 2.0 * N_active * bs_eff / (eff_flops * alpha)
    if t_dec_mem >= t_dec_compute:
        t_dec_base, b_dec = t_dec_mem, "memory"
    else:
        t_dec_base, b_dec = t_dec_compute, "compute"

    # ---- KV-cache cost averaged over the response ----
    # K and V can be stored at different precisions (e.g. llama.cpp -ctk Q8_0 -ctv Q4_0)
    # Sliding-window attention (Mistral, Gemma, DeepSeek-V4) caps the KV
    # cache at `sliding_window` tokens — past tokens are evicted and don't
    # contribute to per-step read cost or to memory.
    avg_ctx = p_in + p_out / 2.0
    if sliding_window is not None:
        avg_ctx = min(avg_ctx, sliding_window)

    # KV bytes per token = layers * kv_heads * head_dim * 2 * bytes_per_element
    # (factor 2 for K and V, captured by (kv_bits_k + kv_bits_v)/8).
    # head_dim defaults to 128 (universal for dense LLMs); MLA models like
    # DeepSeek V3/V4 use larger values (typically 512) to represent the
    # compressed latent KV.
    if kv_heads is not None:
        kv_per_token_bytes = layers * kv_heads * head_dim * (kv_bits_k + kv_bits_v) / 8.0
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
    # Sliding window caps the per-request KV footprint at `sliding_window`.
    ctx_kept = p_in + p_out
    if sliding_window is not None:
        ctx_kept = min(ctx_kept, sliding_window)
    kv_total = kv_per_token_bytes * ctx_kept * batch / kv_packing_eff

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
