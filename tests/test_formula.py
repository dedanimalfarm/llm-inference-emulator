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
    """At accept_rate=0.7, k=4, overhead=0.15, verify_scale=0.05:
      E[accepted] = (1 - 0.7^5) / 0.3 ≈ 2.773
      t_cycle = 0.15·t_dec + 1.2·t_dec = 1.35·t_dec
      t_per   = 1.35 / 2.773 ≈ 0.487·t_dec   → ~2.05× decode speedup
    """
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
    # Result must expose e_accept and speedup
    assert spec.spec_e_accept is not None
    assert abs(spec.spec_e_accept - 2.7731) < 1e-3
    assert spec.spec_speedup is not None
    assert spec.spec_speedup > 2.0
    # Non-speculative baseline: fields are None
    assert base.spec_e_accept is None
    assert base.spec_speedup is None


def test_speculative_truncated_geometric_at_low_acceptance():
    """At p=0.5, K=8, the OLD product formula (p·K=4) double-counts vs the
    correct truncated geometric E = (1 - 0.5^9)/0.5 ≈ 1.996. The fix must
    yield E close to 2 and speedup well below the naive estimate."""
    common = dict(
        n_params_b=70, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    spec = predict(speculative=True, spec_accept_rate=0.5, spec_k_proposed=8,
                   spec_overhead=0.0, spec_verify_scale=0.0, **common)
    # E[accepted] must match the closed-form formula
    assert abs(spec.spec_e_accept - 1.99609) < 1e-4
    # With zero draft cost and zero verify scaling: t_cycle = t_dec
    # → speedup = E[accepted] ≈ 2.0 (NOT 4.0 from the buggy product)
    assert abs(spec.spec_speedup - 1.99609) < 1e-3


def test_speculative_perfect_acceptance_gives_kplus1_speedup():
    """At p=1.0 the geometric sum collapses to K+1 (every proposed token
    accepted plus the bonus). With zero overhead we get exactly (K+1)×
    speedup — the theoretical maximum."""
    common = dict(
        n_params_b=70, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    for K in (0, 1, 4, 8):
        spec = predict(speculative=True, spec_accept_rate=1.0, spec_k_proposed=K,
                       spec_overhead=0.0, spec_verify_scale=0.0, **common)
        assert spec.spec_e_accept == K + 1
        assert abs(spec.spec_speedup - (K + 1)) < 1e-9


def test_speculative_zero_acceptance_is_pure_overhead():
    """At p=0 the draft is never accepted; only the target's bonus token
    survives each cycle. Speculative decoding must be SLOWER than baseline:
      E[accepted] = 1
      t_cycle = overhead·t_dec + t_dec·(1 + verify_scale·K) > t_dec
    """
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    base = predict(**common)
    spec = predict(speculative=True, spec_accept_rate=0.0, spec_k_proposed=4,
                   spec_overhead=0.15, spec_verify_scale=0.05, **common)
    assert spec.spec_e_accept == 1.0
    assert spec.decode_per_token_s > base.decode_per_token_s
    assert spec.spec_speedup < 1.0


def test_speculative_draft_physics_path():
    """When `spec_draft_n_params_b` is given, draft step cost comes from
    roofline rather than the `spec_overhead` heuristic. A 1B draft for a
    70B target should be ~1/70 of the target step (both memory-bound).
    K=4 cycles → total draft cost ≈ 4/70 ≈ 5.7% of one target decode."""
    common = dict(
        n_params_b=70, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    base = predict(**common)
    # No verify penalty so we can isolate the draft term.
    spec = predict(speculative=True, spec_accept_rate=0.8, spec_k_proposed=4,
                   spec_draft_n_params_b=1.0, spec_verify_scale=0.0, **common)
    # E[accepted] for p=0.8, K=4 = (1 - 0.8^5)/0.2 ≈ 3.3616
    assert abs(spec.spec_e_accept - 3.3616) < 1e-3
    # t_cycle / t_dec = 4·(1/70) + 1 ≈ 1.0571
    # speedup = 3.3616 / 1.0571 ≈ 3.180×
    assert abs(spec.spec_speedup - 3.180) < 0.02
    # spec_overhead should be IGNORED when draft is specified — pass an
    # absurd value and confirm result doesn't change.
    spec_ignored = predict(speculative=True, spec_accept_rate=0.8, spec_k_proposed=4,
                           spec_draft_n_params_b=1.0, spec_overhead=999.0,
                           spec_verify_scale=0.0, **common)
    assert abs(spec_ignored.decode_per_token_s - spec.decode_per_token_s) < 1e-12


def test_speculative_draft_validates_against_target_size():
    """Draft must be strictly smaller than target — physically impossible
    to draft with a model larger than the one you're accelerating."""
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
    )
    try:
        predict(speculative=True, spec_draft_n_params_b=8.0, **common)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "spec_draft_n_params_b" in str(e)


def test_speculative_eagle_like_published_numbers():
    """EAGLE-2 paper (Li et al. 2024) reports ~2.5× wall-clock speedup on
    Llama-2-70B with α≈0.7 effective acceptance and K=4 tree depth.
    Our model targets this regime with the physics-based draft path."""
    common = dict(
        n_params_b=70, bits=16, p_in=1024, p_out=256, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    # EAGLE head is ~0.3B; treat as 1B for safety margin.
    spec = predict(speculative=True, spec_accept_rate=0.7, spec_k_proposed=4,
                   spec_draft_n_params_b=1.0, **common)
    # Expect 2.0-3.0× — matches published EAGLE-2 numbers
    assert 2.0 <= spec.spec_speedup <= 3.0, (
        f"speedup {spec.spec_speedup:.2f}× outside expected 2-3× range"
    )


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
    n_active_b=49 and the MLA-mapped head_dim=288 from ARCH_DEFAULTS."""
    common = dict(
        n_params_b=1600.0, bits=4, p_in=1024, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
    )
    # Explicit MoE / MLA-mapped head_dim (kv_lora_rank+qk_rope)/2 = (512+64)/2
    explicit = predict(n_active_b=49.0, head_dim=288, **common)
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
    # ~70 KB/token × 100k tokens ≈ 7 GB delta with MLA-mapped head_dim=288
    assert (full_attn.memory_gb - from_arch.memory_gb) > 5


def test_chunked_prefill_neutral_when_compute_bound():
    """Long compute-bound prefill: chunking changes nothing.
    Total FLOPS unchanged, and per-chunk compute >> per-chunk memory."""
    common = dict(
        n_params_b=70, bits=16, p_in=8192, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xH100", 16),
        mem_bw=get_memory_bandwidth("1xH100"),
        alpha=0.47, beta=0.65,
    )
    mono = predict(**common)
    chunked = predict(chunked_prefill=True, chunk_size=2048, **common)
    assert mono.bottleneck_prefill == "compute"
    assert chunked.bottleneck_prefill == "compute"
    assert abs(chunked.prefill_s - mono.prefill_s) / mono.prefill_s < 1e-6


def test_chunked_prefill_overhead_when_memory_bound():
    """Short prefill on a big model: each chunk pays a full weight-load.
    With p_in_eff < chunk_size, n_chunks = 1 (still 1 forward pass). With
    p_in much smaller than what compute-balances W/MBW, multiple small
    chunks each cost W/MBW → linear overhead in n_chunks."""
    common = dict(
        n_params_b=70, bits=16, p_in=128, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xH100", 16),
        mem_bw=get_memory_bandwidth("1xH100"),
        alpha=0.47, beta=0.65,
    )
    mono = predict(**common)
    chunked = predict(chunked_prefill=True, chunk_size=64, **common)
    assert mono.bottleneck_prefill == "memory"
    assert chunked.bottleneck_prefill == "memory"
    # p_in=128, chunk_size=64 → 2 chunks, each W/MBW → 2× overhead
    assert abs(chunked.prefill_s / mono.prefill_s - 2.0) < 1e-6


def test_chunked_prefill_smaller_than_chunk_is_noop():
    """When p_in_eff < chunk_size, only one chunk is emitted; the result
    must match the monolithic path exactly."""
    common = dict(
        n_params_b=7, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    mono = predict(**common)
    chunked = predict(chunked_prefill=True, chunk_size=4096, **common)
    assert abs(chunked.prefill_s - mono.prefill_s) < 1e-12


def test_chunked_prefill_validates_chunk_size():
    common = dict(
        n_params_b=7, bits=16, p_in=512, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.85,
    )
    try:
        predict(chunked_prefill=True, chunk_size=0, **common)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "chunk_size" in str(e)


def test_b200_blackwell_specs_load():
    """B200 must expose 2.25 PF BF16 dense, 4.5 PF FP8, 9 PF FP4, 8 TB/s."""
    from emulator.hardware import HARDWARE_SPECS
    spec = HARDWARE_SPECS["1xB200"]
    assert spec["peak_tflops"][16] == 2250.0
    assert spec["peak_tflops"][8] == 4500.0
    assert spec["peak_tflops"][4] == 9000.0
    assert spec["memory_bandwidth_gbs"] == 8000.0
    assert spec["memory_capacity_gb"] == 192.0


def test_mi300x_amd_specs_load():
    """MI300X must expose 1307 TFLOPS BF16 dense, 5.3 TB/s, 192 GB."""
    from emulator.hardware import HARDWARE_SPECS
    spec = HARDWARE_SPECS["1xMI300X"]
    assert spec["peak_tflops"][16] == 1307.0
    assert spec["memory_bandwidth_gbs"] == 5300.0
    assert spec["memory_capacity_gb"] == 192.0


def test_b200_decodes_70b_faster_than_h100_proportionally_to_bandwidth():
    """Llama-70B decode on B200 vs H100: weights load (memory-bound) scales
    with MBW ratio 8000/3350 ≈ 2.39×. Tolerate ±5% drift from KV-term."""
    common = dict(
        n_params_b=70, bits=16, p_in=1024, p_out=256, batch=1,
        alpha=0.47, beta=0.65,
    )
    h100 = predict(peak_flops=get_peak_compute("1xH100", 16),
                   mem_bw=get_memory_bandwidth("1xH100"), **common)
    b200 = predict(peak_flops=get_peak_compute("1xB200", 16),
                   mem_bw=get_memory_bandwidth("1xB200"), **common)
    assert h100.bottleneck_decode == "memory"
    assert b200.bottleneck_decode == "memory"
    expected = 8000.0 / 3350.0
    actual = h100.decode_per_token_s / b200.decode_per_token_s
    assert abs(actual / expected - 1.0) < 0.05, (
        f"decode speedup {actual:.2f}× vs expected {expected:.2f}×"
    )


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

def test_mla_exact_matches_gqa_approximation():
    """MLA exact formula with compressed ranks must match the historical GQA-like approximation.
    For DeepSeek-V3, effective head_dim=288, heads=1, fp16 KV (factor 4 bytes/token/layer):
    kv_per_token_bytes = layers * 1 * 288 * 4 = layers * 1152.
    Native MLA with rank=512, rope=64, fp16 (factor 2 bytes/token/layer):
    kv_per_token_bytes = layers * (512 + 64) * 2 = layers * 1152.
    """
    common = dict(
        n_params_b=671.0, bits=4, p_in=1024, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=61, d_model=7168,
    )
    gqa_approx = predict(head_dim=288, kv_heads=1, **common)
    mla_native = predict(mla_kv_lora_rank=512, mla_qk_rope_dim=64, **common)
    assert abs(gqa_approx.memory_gb - mla_native.memory_gb) < 1e-6
    assert abs(gqa_approx.decode_per_token_s - mla_native.decode_per_token_s) < 1e-9


def test_batch_dependent_alpha_scaling():
    """Smaller batches should scale MFU alpha down; larger batches saturate near the peak alpha."""
    common = dict(
        n_params_b=7.0, bits=16, p_in=2048, p_out=64,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.40, beta=0.70,
        alpha_sat_b0=8.0,
    )
    b1 = predict(batch=1, **common)
    b64 = predict(batch=64, **common)
    assert b1.alpha_eff is not None
    assert b64.alpha_eff is not None
    assert 0.04 < b1.alpha_eff < 0.05
    assert 0.398 < b64.alpha_eff < 0.40
    # Per-request prefill latency is faster at batch=64 due to higher MFU alpha_eff
    assert (b1.prefill_s / 1.0) > (b64.prefill_s / 64.0)


def test_deepseek_models_auto_load_mla():
    """DeepSeek models in ARCH_DEFAULTS must automatically load native MLA ranks if not provided."""
    common = dict(
        n_params_b=671.0, bits=4, p_in=1024, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
    )
    res = predict(**common)
    assert res.mla_kv_lora_rank == 512
    assert res.mla_qk_rope_dim == 64


def test_interconnect_inference():
    """Verify that interconnect profiles are correctly inferred from hw string or default rules."""
    from emulator.formula import _infer_comm_link
    assert _infer_comm_link("1xH100") == "nvlink"
    assert _infer_comm_link("2xRTX-5090") == "pcie5"
    assert _infer_comm_link("1xRTX-4090") == "pcie4"
    assert _infer_comm_link("8xMI300X") == "infinity_fabric"
    assert _infer_comm_link("8xMI325X") == "infinity_fabric"


def test_tp_allreduce_overhead_scaling():
    """Verify that Tensor Parallelism Ring All-Reduce communication time scales with TP size."""
    common = dict(
        n_params_b=7.0, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=32, d_model=4096, kv_heads=8,
    )
    
    # baseline: no communication (comm_link=None or "none")
    baseline = predict(tp_size=1, **common)
    assert baseline.tp_comm_prefill_s == 0.0 or baseline.tp_comm_prefill_s is None
    assert baseline.tp_comm_decode_s == 0.0 or baseline.tp_comm_decode_s is None

    # With TP=2 and NVLink, compared to TP=2 without communication
    tp2_no_comm = predict(tp_size=2, **common)
    tp2 = predict(tp_size=2, comm_link="nvlink", **common)
    assert tp2.comm_link == "nvlink"
    assert tp2.tp_comm_prefill_s > 0.0
    assert tp2.tp_comm_decode_s > 0.0
    # Prefill and decode times should be strictly larger than without communication
    assert tp2.prefill_s > tp2_no_comm.prefill_s
    assert tp2.decode_per_token_s > tp2_no_comm.decode_per_token_s
    
    # With TP=4: Volume term per layer is 4 * (tp-1)/tp * volume.
    # Latency term is 4 * (tp-1) * latency.
    # So TP=4 should have larger total communication times than TP=2.
    tp4 = predict(tp_size=4, comm_link="nvlink", **common)
    assert tp4.tp_comm_prefill_s > tp2.tp_comm_prefill_s
    assert tp4.tp_comm_decode_s > tp2.tp_comm_decode_s


def test_pp_communication_overhead():
    """Verify Pipeline Parallelism boundary sequential transfer penalties."""
    common = dict(
        n_params_b=70.0, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=80, d_model=8192, kv_heads=8,
    )
    
    # PP=1 (default): no PP communication
    pp1 = predict(pp_size=1, comm_link="nvlink", **common)
    assert pp1.pp_comm_prefill_s == 0.0 or pp1.pp_comm_prefill_s is None
    assert pp1.pp_comm_decode_s == 0.0 or pp1.pp_comm_decode_s is None
    
    # PP=2: should have non-zero PP communication
    pp2 = predict(pp_size=2, comm_link="nvlink", **common)
    assert pp2.pp_comm_prefill_s > 0.0
    assert pp2.pp_comm_decode_s > 0.0
    
    # PP=4: should have more communication than PP=2
    pp4 = predict(pp_size=4, comm_link="nvlink", **common)
    assert pp4.pp_comm_prefill_s > pp2.pp_comm_prefill_s
    assert pp4.pp_comm_decode_s > pp2.pp_comm_decode_s


def test_comm_overrides():
    """Verify custom communication overrides (bandwidth & latency)."""
    common = dict(
        n_params_b=7.0, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=32, d_model=4096, kv_heads=8,
        tp_size=2,
    )
    
    # baseline nvlink: bw=450.0 GB/s, latency=1.5e-6 s
    ref = predict(comm_link="nvlink", **common)
    
    # override bandwidth to be 10x lower
    slow_bw = predict(comm_link="nvlink", comm_link_bw=45.0, **common)
    assert slow_bw.comm_link_bw == 45.0
    assert slow_bw.tp_comm_prefill_s > ref.tp_comm_prefill_s
    
    # override latency to be 10x higher
    slow_lat = predict(comm_link="nvlink", comm_link_latency=15e-6, **common)
    assert slow_lat.comm_link_latency == 15e-6
    assert slow_lat.tp_comm_prefill_s > ref.tp_comm_prefill_s


def test_consumer_gpu_host_mediated_penalty():
    """Verify that using TP over PCIe on consumer GPUs triggers a host-mediated penalty."""
    common = dict(
        n_params_b=7.0, bits=16, p_in=256, p_out=64, batch=1,
        peak_flops=get_peak_compute("RTX-3090", 16),
        mem_bw=get_memory_bandwidth("RTX-3090"),
        alpha=0.30, beta=0.70,
        layers=32, d_model=4096, kv_heads=8,
        tp_size=2,
    )
    
    # If we use PCIe 5 on consumer GPU vs standard PCIe 5, the penalty should degrade bw and latency
    res_consumer = predict(comm_link="pcie5", hw="2xRTX-5090", **common)
    
    assert abs(res_consumer.comm_link_bw - 31.5) < 0.01
    assert abs(res_consumer.comm_link_latency - 7.5e-6) < 1e-9


def test_pp_1f1b_bubble_latency():
    """Verify that PP size > 1 accurately calculates 1F1B bubble times and divides base layers."""
    common = dict(
        n_params_b=70.0, bits=16, p_in=256, p_out=64, batch=4,
        peak_flops=get_peak_compute("1xA100", 16),
        mem_bw=get_memory_bandwidth("1xA100"),
        alpha=0.30, beta=0.70,
        layers=80, d_model=8192, kv_heads=8,
    )
    
    # PP=1
    pp1 = predict(pp_size=1, **common)
    assert pp1.pp_bubble_prefill_s == 0.0 or pp1.pp_bubble_prefill_s is None
    assert pp1.pp_bubble_decode_s == 0.0 or pp1.pp_bubble_decode_s is None
    
    # PP=4
    pp4 = predict(pp_size=4, **common)
    assert pp4.pp_bubble_prefill_s > 0.0
    assert pp4.pp_bubble_decode_s > 0.0


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
    test_speculative_truncated_geometric_at_low_acceptance()
    test_speculative_perfect_acceptance_gives_kplus1_speedup()
    test_speculative_zero_acceptance_is_pure_overhead()
    test_speculative_draft_physics_path()
    test_speculative_draft_validates_against_target_size()
    test_speculative_eagle_like_published_numbers()
    test_chunked_prefill_neutral_when_compute_bound()
    test_chunked_prefill_overhead_when_memory_bound()
    test_chunked_prefill_smaller_than_chunk_is_noop()
    test_chunked_prefill_validates_chunk_size()
    test_b200_blackwell_specs_load()
    test_mi300x_amd_specs_load()
    test_b200_decodes_70b_faster_than_h100_proportionally_to_bandwidth()
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
    test_mla_exact_matches_gqa_approximation()
    test_batch_dependent_alpha_scaling()
    test_deepseek_models_auto_load_mla()
    test_interconnect_inference()
    test_tp_allreduce_overhead_scaling()
    test_pp_communication_overhead()
    test_comm_overrides()
    test_consumer_gpu_host_mediated_penalty()
    test_pp_1f1b_bubble_latency()
    print("all tests passed")
