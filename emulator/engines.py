"""Engine-specific MFU/MBU and batching scaling.

`alpha` — Model FLOPs Utilization in the prefill (compute) phase.
`beta`  — Memory Bandwidth Utilization in the decode (memory) phase.
`batch_saturation` — (batch_max, batch_50pct) tuple for throughput scaling,
capturing gains from continuous batching / PagedAttention / dynamic batching.
`kv_packing_eff` — share of allocated KV memory that is actually used.
PagedAttention-style allocators reach 0.95-0.99; naive contiguous reserves
worst-case per request and lands around 0.60-0.70.

Numbers below are starting points from public benchmarks. For NVIDIA
hardware they get overridden by `calibrated_coefficients.csv` produced
from the leaderboard.
"""

ENGINE_DEFAULTS = {
    "pytorch":      {"alpha": 0.20, "beta": 0.55, "batch_saturation": None, "compute_path": 16,
                     "kv_packing_eff": 0.65,
                     "notes": "eager mode reference; naive contiguous KV allocator"},
    "pytorch+sdpa": {"alpha": 0.30, "beta": 0.65, "batch_saturation": None, "compute_path": 16,
                     "kv_packing_eff": 0.65,
                     "notes": "scaled-dot-product-attention fused kernel; same naive KV allocator as eager"},
    "vllm":         {"alpha": 0.47, "beta": 0.65, "batch_saturation": (45, 7), "compute_path": 16,
                     "kv_packing_eff": 0.97,
                     "notes": "PagedAttention + continuous batching + APC + Marlin AWQ. "
                              "α=0.47 calibrated on RTX 5090 / Qwen-7B AWQ. "
                              "batch_saturation=(45, 7) — RTX-5090-specific calibration. "
                              "For A100/H100 always read calibrated_coefficients.csv: "
                              "A100-40 ranges (3.8..7.8, 1.8..12.2) by model size. "
                              "kv_packing_eff=0.97 from PagedAttention paging (literature default). "
                              "WARNING: KV-FP8 on Ampere (A100) falls back from FlashAttention "
                              "to XFormers — observed −50% throughput on A100-40GB Qwen-7B. "
                              "Use FP16 KV unless VRAM-bound. Hopper (H100) supports FP8 KV natively."},
    "tensorrt":     {"alpha": 0.55, "beta": 0.90, "batch_saturation": (8, 2), "compute_path": 8,
                     "kv_packing_eff": 0.92,
                     "notes": "kernel fusion + INT8/FP8 on Hopper; paged KV allocator"},
    "llama.cpp":    {"alpha": 0.36, "beta": 0.72, "batch_saturation": None, "compute_path": 16,
                     "kv_packing_eff": 0.85,
                     "notes": "GGUF/CUDA calibrated on RTX-3090. Ring-buffer KV (no worst-case "
                              "per-request reservation but no paging either). "
                              "WARNING: mixed-precision KV (e.g. -ctk f16 -ctv q8_0) on llama.cpp "
                              "falls back to slow path — symmetric KV quant required"},
    "openvino":     {"alpha": 0.35, "beta": 0.70, "batch_saturation": None, "compute_path": 16,
                     "kv_packing_eff": 0.65,
                     "notes": "Intel CPU / iGPU; naive contiguous KV"},
    "onnxruntime":  {"alpha": 0.30, "beta": 0.65, "batch_saturation": None, "compute_path": 16,
                     "kv_packing_eff": 0.65,
                     "notes": "graph-level optimization; naive contiguous KV"},
    "triton":       {"alpha": 0.25, "beta": 0.60, "batch_saturation": (8, 2), "compute_path": 16,
                     "kv_packing_eff": 0.85,
                     "notes": "serving framework; alpha/beta/kv_packing from the underlying engine"},
}
