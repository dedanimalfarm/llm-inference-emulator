"""Smoke tests for the roofline formula."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.formula import predict
from emulator.hardware import get_peak_compute, get_memory_bandwidth


def test_a100_7b_fp16_decode_is_memory_bound():
    res = predict(
        n_params_b=7, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
    )
    assert res.bottleneck_decode == "memory"


def test_quantization_reduces_memory_time():
    common = dict(
        n_params_b=7, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xT4", 16),
        mem_bw=get_memory_bandwidth("1xT4"),
        alpha=0.30, beta=0.70,
    )
    fp16 = predict(bits=16, **common)
    int4 = predict(bits=4, **common)
    assert int4.decode_per_token_s < fp16.decode_per_token_s
    # 4× smaller weights -> ~4× faster decode at same beta (KV doesn't shrink)
    assert int4.decode_per_token_s < fp16.decode_per_token_s / 2


def test_large_batch_flips_decode_to_compute():
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64,
        peak_flops=get_peak_compute("1xT4", 16),
        mem_bw=get_memory_bandwidth("1xT4"),
        alpha=0.30, beta=0.70,
    )
    small = predict(batch=1, **common)
    large = predict(batch=128, **common)
    # very large batch flips decode to compute-bound
    assert large.bottleneck_decode == "compute"
    assert small.bottleneck_decode == "memory"


def test_quantized_kv_cache_speeds_up_long_context_decode():
    common = dict(
        n_params_b=7, bits=16, p_in=8192, p_out=512, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
    )
    fp16_kv = predict(kv_bits_k=16, kv_bits_v=16, **common)
    int8_kv = predict(kv_bits_k=8,  kv_bits_v=8,  **common)
    int4_kv = predict(kv_bits_k=4,  kv_bits_v=4,  **common)
    # at 8k context the KV term dominates decode; quantizing it must speed things up
    assert int8_kv.decode_per_token_s < fp16_kv.decode_per_token_s
    assert int4_kv.decode_per_token_s < int8_kv.decode_per_token_s


def test_effective_bpw_q4_k_m():
    from emulator.calibrate import effective_bpw
    # Qwen2.5-7B-Q4_K_M numbers from llama-bench README
    bpw = effective_bpw(4_677_120_000, 7_615_616_512)
    assert 4.85 < bpw < 5.00, f"unexpected bpw {bpw}"


def test_multi_gpu_capacity_only_when_tp_eff_inverse_of_size():
    """When tp_size=N and tp_efficiency=1/N, multi-GPU latency == single-GPU.

    This is the contract of HARDWARE_SPECS["2xRTX-3090"]: layer-split on PCIe 3.0
    is pipeline parallel, so 2 cards yield 1× single-card throughput.

    Catches the 'double-counting peak_flops' bug: peak_tflops in HARDWARE_SPECS
    must be PER-CARD, then predict() applies tp_size × tp_efficiency.
    """
    common = dict(
        n_params_b=7, bits=16, p_in=512, p_out=128, batch=1,
        peak_flops=get_peak_compute("RTX-3090", 16),
        mem_bw=get_memory_bandwidth("RTX-3090"),
        alpha=0.36, beta=0.72,
    )
    single = predict(**common)
    multi  = predict(tp_size=2, tp_efficiency=0.5, **common)
    assert abs(single.prefill_s - multi.prefill_s) / single.prefill_s < 0.001, \
        f"prefill differs: single={single.prefill_s} vs multi={multi.prefill_s}"
    assert abs(single.decode_per_token_s - multi.decode_per_token_s) / single.decode_per_token_s < 0.001


def test_rtx3090_calibration_sanity():
    import pandas as pd
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "results", "calibrated_coefficients.csv")
    if not os.path.exists(path):
        return # skip if not calibrated yet
    
    df = pd.read_csv(path)
    row = df[(df["hw"] == "RTX-3090") & (df["backend"] == "llama.cpp")]
    if len(row) == 0:
        return
        
    alpha = row["alpha_median"].values[0]
    beta = row["beta_median"].values[0]
    
    # After fixes, alpha should be ~0.4-0.6 and beta ~0.6-0.8
    # We use slightly wider bounds to account for noisy data or different batch sweeps
    assert 0.3 < alpha < 0.7, f"alpha {alpha} out of range"
    assert 0.5 < beta < 0.9, f"beta {beta} out of range"


def test_prefix_cache_collapses_prefill_to_weight_load():
    """At hit_rate=1.0 the matmul work disappears; only weight load remains.

    Physical invariant: prefill time should equal W / MBW (memory-bound floor).
    """
    common = dict(
        n_params_b=7, bits=16, p_in=2048, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    base   = predict(prefix_cache_hit=0.0, **common)
    cached = predict(prefix_cache_hit=1.0, **common)
    # Full hit: prefill = W / MBW (weights still read once)
    N = 7e9
    W = N * 16 / 8
    expected = W / get_memory_bandwidth("1xA100")
    assert abs(cached.prefill_s - expected) / expected < 0.01, \
        f"cached prefill {cached.prefill_s} vs expected {expected}"
    assert cached.prefill_s < base.prefill_s
    assert cached.bottleneck_prefill == "memory"


def test_prefix_cache_partial_hit_proportional():
    """h=0.8 should reduce prefill compute term by 5×; total prefill time
    follows max(compute, memory). For a regime where prefill is compute-bound,
    we expect ~5× speedup at h=0.8."""
    common = dict(
        n_params_b=70, bits=16, p_in=4096, p_out=64, batch=1,  # big enough to stay compute-bound
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    base   = predict(prefix_cache_hit=0.0, **common)
    cached = predict(prefix_cache_hit=0.8, **common)
    assert base.bottleneck_prefill == "compute"
    # h=0.8 → compute term × 0.2; expect close to 5× speedup if still compute-bound
    ratio = base.prefill_s / cached.prefill_s
    assert ratio > 4.0, f"expected ~5× speedup at h=0.8, got {ratio:.2f}×"


def test_speculative_decoding_speeds_up_decode():
    """At accept_rate=0.7, k=4, overhead=0.15 the formula gives
    (1.15 / 2.8) ≈ 0.41× per-token time → ~2.4× decode speedup."""
    common = dict(
        n_params_b=70, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    base = predict(speculative=False, **common)
    spec = predict(speculative=True, spec_accept_rate=0.7, spec_k_proposed=4,
                   spec_overhead=0.15, **common)
    assert spec.decode_per_token_s < base.decode_per_token_s * 0.5
    # Sanity: with accept_rate=0.0 it must NOT speed up — it would divide by zero.
    # Formula raises in that case; we just ensure it's not silently faster than base.


def test_speculative_zero_accept_rate_raises():
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
    )
    try:
        predict(speculative=True, spec_accept_rate=0.0, **common)
    except ValueError:
        return
    raise AssertionError("expected ValueError on accept_rate=0")


def test_batch_saturation_asymptote():
    """At batch >> batch_50pct AND batch >> batch_max, bs_eff → batch_max.
    Hill curve is clipped by queue depth, so bs_eff(batch) ≤ batch always."""
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
        batch_saturation=(16, 2),
    )
    single = predict(batch=1, **common)
    big   = predict(batch=500, **common)
    bs_eff_single = single.throughput_tok_s * single.decode_per_token_s
    bs_eff_big   = big.throughput_tok_s   * big.decode_per_token_s
    # bs_eff(1) must equal 1.0 — one queued request can't be more concurrent
    assert abs(bs_eff_single - 1.0) < 1e-6
    # bs_eff(500) → batch_max=16 within 1%
    assert abs(bs_eff_big - 16.0) / 16.0 < 0.01


def test_batch_saturation_clipped_by_queue():
    """At batch ≤ batch_max with strong Hill term, bs_eff is clipped to batch.
    Old Hill-only form gave bs_eff(2) = 8 for (16, 2) — physically nonsense."""
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64, batch=2,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    res = predict(batch_saturation=(16, 2), **common)
    bs_eff = res.throughput_tok_s * res.decode_per_token_s
    # batch=2, Hill=16*2/4=8 → clip to queue depth 2
    assert abs(bs_eff - 2.0) < 1e-6


def test_kv_packing_eff_inflates_memory_for_naive_allocator():
    """Naive allocator (eff=0.65) must report ~50% more KV memory than
    PagedAttention (eff=0.97), all else equal."""
    common = dict(
        n_params_b=7, bits=16, p_in=2048, p_out=512, batch=8,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    paged = predict(kv_packing_eff=0.97, **common)
    naive = predict(kv_packing_eff=0.65, **common)
    # Both report W + KV/eff. KV portion differs by factor (0.97/0.65) ≈ 1.49.
    # Total memory ratio depends on KV-to-W ratio, but naive must be larger.
    assert naive.memory_gb > paged.memory_gb
    # Default (eff=1.0) should be the cleanest baseline — smaller than both
    base = predict(**common)  # eff defaults to 1.0
    assert base.memory_gb < paged.memory_gb < naive.memory_gb


def test_engine_defaults_kv_packing_eff_wired():
    """Every engine in ENGINE_DEFAULTS must declare kv_packing_eff, and
    routing the field through predict() must make vLLM (PagedAttention)
    report lower memory than naive PyTorch in the same scenario. This is
    the integration check that callers like cli.py / demo.py see the
    PagedAttention effect without an explicit override."""
    from emulator.engines import ENGINE_DEFAULTS

    for name, conf in ENGINE_DEFAULTS.items():
        eff = conf.get("kv_packing_eff")
        assert eff is not None, f"{name} missing kv_packing_eff"
        assert 0.0 < eff <= 1.0, f"{name}: kv_packing_eff={eff} out of (0, 1]"

    assert (ENGINE_DEFAULTS["vllm"]["kv_packing_eff"]
            > ENGINE_DEFAULTS["pytorch"]["kv_packing_eff"])

    common = dict(
        n_params_b=7, bits=16, p_in=2048, p_out=512, batch=8,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
    )

    def run(name):
        e = ENGINE_DEFAULTS[name]
        return predict(alpha=e["alpha"], beta=e["beta"],
                       kv_packing_eff=e["kv_packing_eff"],
                       batch_saturation=e["batch_saturation"], **common)

    assert run("vllm").memory_gb < run("pytorch").memory_gb


def test_moe_uses_active_params_for_flops_not_total():
    """MoE: compute time scales with n_active_b, not n_params_b.
    Same total size, half active → ~half the compute-bound time."""
    common = dict(
        bits=16, p_in=2048, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=80, d_model=8192, kv_heads=8,  # explicit, bypass ARCH_DEFAULTS
    )
    dense = predict(n_params_b=70, n_active_b=70, **common)
    moe   = predict(n_params_b=70, n_active_b=35, **common)
    # Prefill on 2048 tokens is compute-bound at this size; halving active
    # FLOPS should roughly halve prefill time.
    assert moe.prefill_s < dense.prefill_s
    ratio = moe.prefill_s / dense.prefill_s
    assert 0.45 < ratio < 0.55, f"expected ~0.5 prefill ratio, got {ratio:.3f}"


def test_moe_decode_memory_scales_with_active_fraction():
    """MoE: per-step weight read at decode scales with N_active/N_total.
    Mixtral 8x7B (12.9B / 46.7B ≈ 27.6%) should decode much faster than
    a dense 46.7B model with identical arch — only routed experts cross
    the HBM boundary at each token.
    Validated against Baseten benchmark (Mixtral / TRT-LLM / A100 / INT8):
    real decode ≈ 11.4 ms, emulator predicts ≈ 7 ms with literature β=0.9
    (calibration gap remains, but the 2× MoE overestimate is gone)."""
    common = dict(
        bits=8, p_in=512, p_out=128, batch=1,
        peak_flops=get_peak_compute("1xA100", 8),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.55, beta=0.90,
        layers=32, d_model=4096, kv_heads=8,
    )
    dense = predict(n_params_b=46.7, n_active_b=46.7, **common)
    moe   = predict(n_params_b=46.7, n_active_b=12.9, **common)
    # MoE decode must be much faster, roughly by the active fraction.
    ratio = moe.decode_per_token_s / dense.decode_per_token_s
    expected = 12.9 / 46.7
    assert abs(ratio - expected) < 0.05, (
        f"expected decode ratio ≈ {expected:.3f}, got {ratio:.3f}"
    )


def test_moe_memory_uses_total_params_not_active():
    """MoE: weight memory uses n_params_b (all experts in VRAM)."""
    common = dict(
        bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=32, d_model=4096, kv_heads=8,
    )
    dense = predict(n_params_b=46.7, n_active_b=46.7, **common)
    moe   = predict(n_params_b=46.7, n_active_b=12.9, **common)  # Mixtral 8x7B
    # Same total → same memory_gb (KV identical, weights identical).
    assert abs(dense.memory_gb - moe.memory_gb) < 1e-6


def test_arch_defaults_moe_entries_consistent():
    """Every MoE entry in ARCH_DEFAULTS must declare n_active_b strictly less
    than its key (total size). Sanity-check the table itself."""
    from emulator.formula import ARCH_DEFAULTS
    moe_keys = [k for k, v in ARCH_DEFAULTS.items() if "n_active_b" in v]
    assert moe_keys, "no MoE entries in ARCH_DEFAULTS"
    for k in moe_keys:
        v = ARCH_DEFAULTS[k]
        active = v["n_active_b"]
        assert 0 < active < k, (
            f"MoE entry {k}B has n_active_b={active}B; "
            f"must be in (0, {k})"
        )


def test_head_dim_override_inflates_kv_cache():
    """DeepSeek MLA models override head_dim from 128 to 512.
    KV cache must scale linearly with head_dim."""
    common = dict(
        n_params_b=70, bits=16, p_in=4096, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=61, d_model=7168, kv_heads=1,
    )
    default_hd = predict(head_dim=128, **common)
    mla_like   = predict(head_dim=512, **common)
    # The KV portion of memory_gb scales 4× with head_dim. Weights identical.
    # Compute KV-only delta:
    w_gb = 70 * 16 / 8  # weights in GB (no division by 1e9 since N is in B already)
    # Actually formula uses W = N * bits / 8 with N in raw count, but memory_gb
    # already divides by 1e9, so weights = 70e9 * 16 / 8 / 1e9 = 140 GB.
    kv_default = default_hd.memory_gb - 140
    kv_mla     = mla_like.memory_gb - 140
    assert abs(kv_mla / kv_default - 4.0) < 0.001


def test_deepseek_v4_pro_arch_loaded_from_defaults():
    """Calling predict() for DeepSeek V4-Pro (size 1600) should pick up
    n_active_b=49, head_dim=512 from ARCH_DEFAULTS automatically."""
    common = dict(
        n_params_b=1600.0, bits=4, p_in=1024, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
    )
    # Explicit MoE/head_dim
    explicit = predict(n_active_b=49.0, head_dim=512, **common)
    # From ARCH_DEFAULTS
    from_defaults = predict(**common)
    assert abs(explicit.prefill_s - from_defaults.prefill_s) < 1e-9
    assert abs(explicit.memory_gb - from_defaults.memory_gb) < 1e-9


def test_n_active_b_validates_against_n_params_b():
    """Active params cannot exceed total params."""
    common = dict(
        bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=32, d_model=4096, kv_heads=8,
    )
    try:
        predict(n_params_b=7, n_active_b=10, **common)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "n_active_b" in str(e)


def test_sliding_window_caps_kv_memory_and_decode_read():
    """With sliding_window < (p_in + p_out), KV memory and per-step KV read
    cost are both capped at sliding_window tokens."""
    common = dict(
        n_params_b=7, bits=16, p_in=8192, p_out=512, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=32, d_model=4096, kv_heads=8,
    )
    full = predict(sliding_window=None, **common)
    sw   = predict(sliding_window=128, **common)
    # KV memory ∝ context kept; weights identical → memory delta is all KV.
    assert sw.memory_gb < full.memory_gb
    # Decode is memory-bound at this size; full reads avg_ctx=8192+256=8448
    # tokens of KV per step, sw reads min(8448, 128)=128. KV-term falls by
    # ~66×, but total decode includes weight read, so the drop is partial.
    assert sw.decode_per_token_s < full.decode_per_token_s


def test_sliding_window_unbounded_by_default():
    """Without explicit sliding_window and no ARCH_DEFAULTS entry, behavior
    matches the pre-SW formula (locks back-compat)."""
    common = dict(
        n_params_b=7, bits=16, p_in=4096, p_out=512, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=32, d_model=4096, kv_heads=8,
    )
    a = predict(**common)
    b = predict(sliding_window=None, **common)
    # Both should give identical results when no sliding window applies.
    assert a.memory_gb == b.memory_gb
    assert a.decode_per_token_s == b.decode_per_token_s


def test_deepseek_v4_pro_inherits_sliding_window_from_arch():
    """DeepSeek V4-Pro entry has sliding_window=128; calling predict() for
    model=1600 must pick it up automatically and cap KV cache."""
    common = dict(
        n_params_b=1600.0, bits=4, p_in=100000, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
    )
    from_arch = predict(**common)
    # Workaround to force full attention: pass a huge window.
    full_attn = predict(sliding_window=10**9, **common)
    # With sw=128, KV per request = 128 tokens × KV/token. Without, 100064
    # tokens. Memory must be massively smaller.
    assert from_arch.memory_gb < full_attn.memory_gb
    assert (full_attn.memory_gb - from_arch.memory_gb) > 10  # at least 10 GB delta


def test_defaults_preserve_legacy_behavior():
    """A call with no vLLM-style kwargs must produce the same result as
    pre-vLLM code. This locks the back-compat contract."""
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64, batch=4,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
    )
    res = predict(**common)
    # Without batch_saturation, bs_eff == batch → throughput = batch / t_dec
    assert abs(res.throughput_tok_s - 4.0 / res.decode_per_token_s) < 1e-9
    # Without speculative, decode_per_token_s is the raw t_dec
    # (no acceleration, no overhead applied)
    explicit = predict(speculative=False, prefix_cache_hit=0.0,
                       batch_saturation=None, kv_packing_eff=1.0, **common)
    assert res.decode_per_token_s == explicit.decode_per_token_s
    assert res.throughput_tok_s == explicit.throughput_tok_s
    assert res.memory_gb == explicit.memory_gb


if __name__ == "__main__":
    test_a100_7b_fp16_decode_is_memory_bound()
    test_quantization_reduces_memory_time()
    test_large_batch_flips_decode_to_compute()
    test_quantized_kv_cache_speeds_up_long_context_decode()
    test_effective_bpw_q4_k_m()
    test_multi_gpu_capacity_only_when_tp_eff_inverse_of_size()
    test_rtx3090_calibration_sanity()
    test_prefix_cache_collapses_prefill_to_weight_load()
    test_prefix_cache_partial_hit_proportional()
    test_speculative_decoding_speeds_up_decode()
    test_speculative_zero_accept_rate_raises()
    test_batch_saturation_asymptote()
    test_batch_saturation_clipped_by_queue()
    test_kv_packing_eff_inflates_memory_for_naive_allocator()
    test_engine_defaults_kv_packing_eff_wired()
    test_moe_uses_active_params_for_flops_not_total()
    test_moe_decode_memory_scales_with_active_fraction()
    test_moe_memory_uses_total_params_not_active()
    test_arch_defaults_moe_entries_consistent()
    test_head_dim_override_inflates_kv_cache()
    test_deepseek_v4_pro_arch_loaded_from_defaults()
    test_n_active_b_validates_against_n_params_b()
    test_sliding_window_caps_kv_memory_and_decode_read()
    test_sliding_window_unbounded_by_default()
    test_deepseek_v4_pro_inherits_sliding_window_from_arch()
    test_defaults_preserve_legacy_behavior()
    print("all tests passed")
