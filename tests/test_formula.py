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


if __name__ == "__main__":
    test_a100_7b_fp16_decode_is_memory_bound()
    test_quantization_reduces_memory_time()
    test_large_batch_flips_decode_to_compute()
    print("all tests passed")
