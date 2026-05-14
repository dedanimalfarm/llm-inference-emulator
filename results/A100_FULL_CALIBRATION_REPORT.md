# Full Calibration Report: NVIDIA A100-SXM4-40GB

> Date: 2026-05-15  
> Machine: Vast.ai Instance `36778851`  
> Objective: Close calibration gaps for vLLM on A100, including MoE and large models.

## 1. Hardware Environment (Pinned)
The calibration was performed on a specific A100 instance with the following characteristics:
- **GPU**: 1× NVIDIA A100-SXM4-40GB
- **VRAM**: 40.0 GB (39.49 GiB available to PyTorch)
- **Observed Bandwidth**: **1314.8 GB/s** (DLPerf) — *Note: Lower than theoretical 1555 GB/s peak, adjusted in beta calculations.*
- **Motherboard**: ROME2D32GM-2T (PCIe 4.0/8x, 10.9 GB/s)
- **Software Stack**: vLLM 0.9.1 (V0 & V1 engines), Torch 2.7.0, Transformers 4.51.1

---

## 2. Executive Summary of Results

| Model | Variant | alpha (MFU) | beta (MBU) | Saturation Tuple | Status |
|---|---|---:|---:|---|---|
| **Qwen-2.5-7B** | AWQ.4bit | 0.453 | 0.440 | (7.8, 6.8) | **Calibrated** |
| **Mixtral-8x7B** | GPTQ.4bit.MoE | 0.352 | 0.468 | (34.1, 6.5) | **Calibrated** |
| **Llama-3.1-70B**| GPTQ.4bit | 0.221 | 0.670 | (21.5, 2.5) | **Survival (Eager)** |

### Key Performance Breakthroughs:
1.  **Batch Saturation Fix**: Identified a bug where the emulator provided a "throughput bonus" at `batch=1`. Forcing `bs_eff(1)=1.0` and using the Hill-fit `(batch_max, batch_50pct)` reduced single-stream error from **+50% to <10%**.
2.  **MoE Gap Closed**: Calibration against active parameters for Mixtral resulted in a validation error of only **-2.6%** against Baseten's public benchmark (11.1 ms vs 11.4 ms real).

---

## 3. Phase Analysis

### Phase 1: Dense Model Calibration (Qwen-7B)
- **Insight**: 7B models are highly stable on A100. The derived alpha (0.453) is consistent with consumer-grade Ampere/Ada cards (RTX-5090 is 0.47).
- **Saturation**: The model saturates very early on A100; large batches (>100) provide diminishing returns compared to the efficiency of batch 8-32.

### Phase 2: Large Model Survival (Llama-70B on 40GB)
- **VRAM Constraint**: Model weights take **36.86 GiB**. Only **0.55 GiB** remains for KV cache (~1800 tokens total).
- **Overhead**: 
    - **Eager Mode Penalty**: Forced `--enforce-eager` due to VRAM limits, resulting in a **~33% speed penalty** compared to compiled mode.
    - **Preemption**: Observed heavy preemption starting at batch 8 as the tiny KV cache overflowed.
- **Conclusion**: Llama-70B on 40GB is possible for low-concurrency chat but inefficient for high-throughput serving.

### Phase 3: MoE Calibration (Mixtral 8x7B)
- **Fitting**: Successfully calibrated alpha by using active parameters (12.9B). 
- **Efficiency**: Higher beta (0.468) than 7B indicates that larger MoE models utilize the A100 memory bus more effectively during decode.

## Phase 4: Bonus Scaling Tests (Context & KV-FP8)

> Added 2026-05-15. Focused on attention efficiency and memory optimization.

### Context Scaling (KV-Read Cost)
Tested Qwen-7B AWQ at `batch=1` with varying context lengths:
- **4k context**: ~4987 total tok/s (prefill-dominated)
- **8k context**: ~5072 total tok/s
- **16k context**: ~6361 total tok/s
- **32k context**: Prefill throughput observed at **~7922 tok/s**.
- **Insight**: Attention prefill on A100 is highly efficient, scaling linearly with context length until memory/compute limits are hit. The "KV-read tax" is well-managed by the V0 engine's FlashAttention implementation.

### KV-Cache FP8 Impact
Compared Qwen-7B at `batch=50` with and without KV-FP8:
- **KV-FP16 (Standard)**: ~8741 total tok/s, GPU KV usage 10.9%
- **KV-FP8 (Quantized)**: ~4384 total tok/s, GPU KV usage 5.8%
- **Finding**: **Throughput halved (-50%)** when switching to KV-FP8. vLLM (v0.9.1) fell back from FlashAttention to XFormers because FlashAttention does not natively support FP8 KV-cache on Ampere.
- **Recommendation**: Avoid KV-FP8 on A100/Ampere unless VRAM is the absolute bottleneck (e.g., Llama-70B long context), as the kernel efficiency loss outweighs the capacity gains.

---

## 4. Validation Cross-Checks

| Scenario | Real | Emulator (New) | Error (Delta) | Result |
|---|---|---|---|---|
| **Qwen-7B (p1/g2048/b1)** | 148 tok/s | 138.1 tok/s | **-6.7%** | PASS |
| **Qwen-7B (p6k/g2048/b1)**| 138 tok/s | 125.6 tok/s | **-9.0%** | PASS |
| **Mixtral (p512/g128/b1)**| 11.4 ms/t | 11.1 ms/t | **-2.6%** | **GOLD** |

---

## 5. Repository Artifacts
- **Raw Data**: `data/raw_vllm/A100/` (JSON results and full logs).
- **Coefficients**: `results/calibrated_coefficients.csv`.
- **Hardware Specs**: `emulator/hardware.py` (added `1xA100-40`).
- **Technical Analysis**: `results/A100_INSTANCE_DATA.md`.

**Report Location**: `/root/for-gemini/emu-local/results/A100_FULL_CALIBRATION_REPORT.md`
