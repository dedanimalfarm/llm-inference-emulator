"""Peak compute and memory bandwidth per hardware target.

Numbers are vendor-specified peaks for tensor-core / AMX paths, since real
LLM inference uses those. Memory bandwidth is the aggregate HBM/GDDR/DDR
bandwidth visible to a single accelerator/instance.
"""

HARDWARE_SPECS = {
    "RTX-3090": {
        "peak_tflops": {16: 142.0, 8: 284.0, 4: 284.0},
        "memory_bandwidth_gbs": 936.0,
        "memory_capacity_gb": 24.0,
        "tdp_w": 350,
    },

    "1xA100": {
        # NVIDIA A100 80GB PCIe — Ampere
        "peak_tflops": {16: 312.0, 8: 624.0, 4: 624.0},  # INT4 not natively accelerated
        "memory_bandwidth_gbs": 2039.0,
        "memory_capacity_gb": 80.0,
        "tdp_w": 275,
    },
    "1xA10": {
        # NVIDIA A10 — Ampere (workstation)
        "peak_tflops": {16: 125.0, 8: 250.0, 4: 250.0},
        "memory_bandwidth_gbs": 600.0,
        "memory_capacity_gb": 24.0,
        "tdp_w": 150,
    },
    "1xT4": {
        # NVIDIA T4 — Turing
        "peak_tflops": {16: 65.0, 8: 130.0, 4: 130.0},
        "memory_bandwidth_gbs": 300.0,
        "memory_capacity_gb": 16.0,
        "tdp_w": 70,
    },
    "32vCPU-C7i": {
        # AWS c7i (Intel Xeon Sapphire Rapids), 32 vCPU = 16 physical cores
        # AMX BF16: ~0.5 TFLOPS/core practical -> ~8 TFLOPS for 16 cores
        # DDR5 8-channel ~200 GB/s effective
        "peak_tflops": {16: 8.0, 8: 16.0, 4: 16.0},
        "memory_bandwidth_gbs": 200.0,
        "memory_capacity_gb": 64.0,
        "tdp_w": 385,
    },
    "2xRTX-3090": {
        # Dual RTX 3090 setup
        # TP efficiency is significantly lower on PCIe 3.0 (observed ~0.50 for layer split)
        "peak_tflops": {16: 284.0, 8: 568.0, 4: 568.0},
        "memory_bandwidth_gbs": 1872.0,
        "memory_capacity_gb": 48.0,
        "tdp_w": 700,
        "tp_size": 2,
        "tp_efficiency": 0.50,
    },
}


def get_peak_compute(hw: str, bits: int) -> float:
    """Return peak compute in FLOPS for (hardware, weight-precision).

    Note: even with INT4 weights, dequantize-then-FP16-matmul is the common
    path on Ampere/Turing, so we cap at the 8-bit tensor throughput where
    relevant. The calibration step absorbs the residual mismatch into alpha.
    """
    specs = HARDWARE_SPECS[hw]["peak_tflops"]
    for b in sorted(specs.keys()):
        if b >= bits:
            return specs[b] * 1e12
    return specs[max(specs.keys())] * 1e12


def get_memory_bandwidth(hw: str) -> float:
    return HARDWARE_SPECS[hw]["memory_bandwidth_gbs"] * 1e9
