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
        # Dual RTX 3090 over PCIe 3.0 (no NVLink).
        #
        # IMPORTANT — this profile gives CAPACITY benefit, not throughput:
        #   With tp_size=2 and tp_efficiency=0.50, predict() returns the same
        #   latency as a single RTX-3090 for any model that fits on one card.
        #   That is mathematically correct: 2 × 0.5 = 1.0× scaling.
        #
        # Why: llama.cpp split-mode `layer` is pipeline-parallel — tokens
        # traverse GPUs sequentially, throughput equals single-card. The 0.50
        # efficiency is empirically measured (not literature default), see
        # results/REPORT.md "Multi-GPU calibration" section.
        #
        # Use this profile only when the model exceeds 24 GB VRAM (e.g.
        # Llama-70B Q4_K_M ≈ 40 GB). For smaller models, use "RTX-3090".
        # For real tensor-parallel speedup, hardware needs NVLink (A100 SXM,
        # H100) and a different engine (vLLM with NCCL).
        #
        # peak_tflops and memory_bandwidth are PER-CARD. The actual scaling
        # is applied by predict() as `peak × tp_size × tp_efficiency`.
        # memory_capacity_gb is aggregate — that's the real combined VRAM
        # available to host weights when split-mode is on.
        "peak_tflops": {16: 142.0, 8: 284.0, 4: 284.0},  # per-card, same as RTX-3090
        "memory_bandwidth_gbs": 936.0,                    # per-card
        "memory_capacity_gb": 48.0,                       # aggregate (for capacity check)
        "tdp_w": 700,
        "tp_size": 2,
        "tp_efficiency": 0.50,    # split-mode=layer (pipeline parallel)
    },
    "1xRTX-5090": {
        # NVIDIA RTX 5090 — Blackwell consumer
        "peak_tflops": {16: 419.0, 8: 838.0, 4: 1676.0},
        "memory_bandwidth_gbs": 1792.0,
        "memory_capacity_gb": 32.0,
        "tdp_w": 575,
    },
    "2xRTX-5090": {
        # Dual RTX 5090 — Blackwell consumer, PCIe 5.0 x16, no NVLink.
        #
        # tp_efficiency = 0.74 calibrated empirically from vLLM AWQ
        # multi-batch sweep (Qwen-7B): TP=2 asymptote 20796 t/s vs TP=1
        # asymptote 14093 t/s → speedup 1.48× → tp_eff = 0.74.
        # Larger models (32B, 70B) hide PCIe latency better; for a
        # workload-specific number, recalibrate against the deployed model.
        "peak_tflops": {16: 419.0, 8: 838.0, 4: 1676.0},
        "memory_bandwidth_gbs": 1792.0,
        "memory_capacity_gb": 64.0,
        "tdp_w": 1150,
        "tp_size": 2,
        "tp_efficiency": 0.74,
        "notes": "PCIe 5.0 x16, no NVLink. tp_efficiency calibrated on Qwen-7B AWQ.",
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
