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


if __name__ == "__main__":
    test_a100_7b_fp16_decode_is_memory_bound()
    test_quantization_reduces_memory_time()
    test_large_batch_flips_decode_to_compute()
    test_quantized_kv_cache_speeds_up_long_context_decode()
    test_effective_bpw_q4_k_m()
    print("all tests passed")
