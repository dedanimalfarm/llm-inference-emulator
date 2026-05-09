"""Engine-specific MFU/MBU and batching scaling.

`alpha` — Model FLOPs Utilization in the prefill (compute) phase.
`beta`  — Memory Bandwidth Utilization in the decode (memory) phase.
`batch_saturation` — (batch_max, batch_50pct) tuple for throughput scaling,
capturing gains from continuous batching / PagedAttention / dynamic batching.

Numbers below are starting points from public benchmarks. For NVIDIA
hardware they get overridden by `calibrated_coefficients.csv` produced
from the leaderboard.
"""

ENGINE_DEFAULTS = {
    "pytorch":      {"alpha": 0.20, "beta": 0.55, "batch_saturation": None, "compute_path": 16,
                     "notes": "eager mode reference"},
    "pytorch+sdpa": {"alpha": 0.30, "beta": 0.65, "batch_saturation": None, "compute_path": 16,
                     "notes": "scaled-dot-product-attention fused kernel"},
    "vllm":         {"alpha": 0.47, "beta": 0.65, "batch_saturation": (45, 7), "compute_path": 16,
                     "notes": "PagedAttention + continuous batching + APC + Marlin AWQ. "
                              "α=0.47 calibrated on RTX 5090 / Qwen-7B AWQ. "
                              "batch_saturation=(45, 7) calibrated from multi-batch sweep "
                              "(half-saturation at batch≈7 for 7B; varies by model size — "
                              "70B saturates at batch≈0.5, 7B-TP2 at batch≈20)."},
    "tensorrt":     {"alpha": 0.55, "beta": 0.90, "batch_saturation": (8, 2), "compute_path": 8,
                     "notes": "kernel fusion + INT8/FP8 on Hopper"},
    "llama.cpp":    {"alpha": 0.36, "beta": 0.72, "batch_saturation": None, "compute_path": 16,
                     "notes": "GGUF/CUDA calibrated on RTX-3090. WARNING: mixed-precision KV (e.g. -ctk f16 -ctv q8_0) on llama.cpp falls back to slow path — symmetric KV quant required"},
    "openvino":     {"alpha": 0.35, "beta": 0.70, "batch_saturation": None, "compute_path": 16,
                     "notes": "Intel CPU / iGPU"},
    "onnxruntime":  {"alpha": 0.30, "beta": 0.65, "batch_saturation": None, "compute_path": 16,
                     "notes": "graph-level optimization"},
    "triton":       {"alpha": 0.25, "beta": 0.60, "batch_saturation": (8, 2), "compute_path": 16,
                     "notes": "serving framework; alpha/beta from the underlying engine"},
}
