"""Engine-specific MFU/MBU and batching multipliers.

`alpha` — Model FLOPs Utilization in the prefill (compute) phase.
`beta`  — Memory Bandwidth Utilization in the decode (memory) phase.
`batch_mult` — effective batch multiplier vs. naive batching, capturing
gains from continuous batching / PagedAttention / dynamic batching.

Numbers below are starting points from public benchmarks. For NVIDIA
hardware they get overridden by `calibrated_coefficients.csv` produced
from the leaderboard.
"""

ENGINE_DEFAULTS = {
    "pytorch":      {"alpha": 0.20, "beta": 0.55, "batch_mult": 1.0,
                     "notes": "eager mode reference"},
    "pytorch+sdpa": {"alpha": 0.30, "beta": 0.65, "batch_mult": 1.0,
                     "notes": "scaled-dot-product-attention fused kernel"},
    "vllm":         {"alpha": 0.40, "beta": 0.75, "batch_mult": 5.0,
                     "notes": "PagedAttention + continuous batching"},
    "tensorrt":     {"alpha": 0.55, "beta": 0.90, "batch_mult": 2.0,
                     "notes": "kernel fusion + INT8/FP8 on Hopper"},
    "llama.cpp":    {"alpha": 0.14, "beta": 0.75, "batch_mult": 1.0,
                     "notes": "GGUF/CUDA calibrated on RTX-3090. WARNING: mixed-precision KV (e.g. -ctk f16 -ctv q8_0) on llama.cpp falls back to slow path — symmetric KV quant required"},
    "openvino":     {"alpha": 0.35, "beta": 0.70, "batch_mult": 1.0,
                     "notes": "Intel CPU / iGPU"},
    "onnxruntime":  {"alpha": 0.30, "beta": 0.65, "batch_mult": 1.0,
                     "notes": "graph-level optimization"},
    "triton":       {"alpha": 0.25, "beta": 0.60, "batch_mult": 3.0,
                     "notes": "serving framework; alpha/beta from the underlying engine"},
}
