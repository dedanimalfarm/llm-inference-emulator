#!/usr/bin/env python3
"""Print a small demo table comparing engines on a few realistic scenarios."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.formula import predict
from emulator.hardware import get_peak_compute, get_memory_bandwidth
from emulator.engines import ENGINE_DEFAULTS

SCENARIOS = [
    ("7B  FP16",  7,  16),
    ("13B FP16", 13,  16),
    ("13B 4bit", 13,   4),
    ("70B 4bit", 70,   4),
]

HW = ["1xA10", "1xA100", "1xT4", "32vCPU-C7i"]
ENGINES = ["pytorch", "vllm", "tensorrt", "llama.cpp"]


def main():
    print(f"{'scenario':<11} {'hw':<11} {'engine':<11} "
          f"{'prefill_ms':>11} {'tok/s':>8} {'mem_gb':>7} {'fits':>6}")
    print("-" * 78)
    for label, n, bits in SCENARIOS:
        for hw in HW:
            for eng_name in ENGINES:
                if eng_name == "llama.cpp" and hw != "32vCPU-C7i":
                    continue
                if eng_name == "tensorrt" and hw == "32vCPU-C7i":
                    continue
                eng = ENGINE_DEFAULTS[eng_name]
                res = predict(
                    n_params_b=n, bits=bits,
                    p_in=256, p_out=64, batch=1,
                    peak_flops=get_peak_compute(hw, bits),
                    mem_bw=get_memory_bandwidth(hw),
                    alpha=eng["alpha"], beta=eng["beta"],
                    batch_mult=eng["batch_mult"],
                )
                mem_cap = {"1xA10": 24, "1xA100": 80, "1xT4": 16, "32vCPU-C7i": 64}[hw]
                fits = "yes" if res.memory_gb <= mem_cap else "NO"
                print(f"{label:<11} {hw:<11} {eng_name:<11} "
                      f"{res.prefill_s*1000:>11.1f} {res.throughput_tok_s:>8.1f} "
                      f"{res.memory_gb:>7.2f} {fits:>6}")
        print()


if __name__ == "__main__":
    main()
