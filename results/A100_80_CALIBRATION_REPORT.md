# A100-80GB Calibration Study (2026 Models)

> Date: 2026-05-15  
> Machine: A100-SXM4-80GB (Vast.ai)  
> Observed Bandwidth: **1314.8 GB/s** (DLPerf)

## 1. Executive Summary
Calibration on the 80GB instance successfully revealed the true potential of the SXM4 architecture without the memory-pressure artifacts of Step 5. We achieved a first contact validation with **Llama 4 Scout** and **Gemma 4 31B**, identifying their unique architectural bottlenecks.

## 2. Calibration Results (vLLM 0.9.1 / llama.cpp)

| Model | Variant | alpha (MFU) | beta (MBU) | Saturation | Notes |
|---|---|---:|---:|---|---|
| **Qwen-2.5-7B** | AWQ.4bit | 0.453 | 0.560 | (7.8, 6.8) | beta isolated from 16k context |
| **Qwen3-32B** | AWQ.4bit | 0.487 | 0.560 | (7.0, 4.2) | 2026 Dense Model |
| **Qwen-2.5-72B**| AWQ.4bit | 0.537 | 0.560 | (8.2, 5.5) | High MFU on large cards |
| **Llama 4 Scout**| IQ2_M | 0.087* | 0.560 | (3.8, 1.8) | *llama.cpp prefill efficiency |
| **Gemma 4 31B** | IQ2_M | 0.105* | 0.560 | (7.0, 4.2) | *llama.cpp prefill efficiency |

## 3. Findings
- **2026 Model Bottlenecks**:
  - **Llama 4 Scout (109B MoE)**: Achieved **68.1 tok/s** (decode) and **798.6 tok/s** (prefill) in IQ2_M. Despite the large total size, active parameter scaling holds well.
  - **Gemma 4 31B**: Achieved **44.7 tok/s** (decode) and **956.6 tok/s** (prefill) in IQ2_M. Shows high prefill efficiency due to head_dim=256 and global attention layers.
- **Multimodal Overhead**: vLLM 0.21.0 falling back to Transformers for Gemma 4 caused OOM during initialization even on 80GB, suggesting multimodal LLMs require significant "memory padding" beyond the base weights.
- **Speculative Decoding Alignment**: Confirmed that vocab mismatch is a primary friction point for spec-decode (7B vs 1.5B Qwen).

## 4. Validation Cross-Check
| Scenario | Real | Emulator (Calibrated) | Delta |
|---|---|---|---|
| Qwen-72B b=32 | 138.0 tok/s | 134.5 tok/s | **-2.5%** |
| Scout IQ2_M b=1 | 68.1 tok/s | 66.4 tok/s | **-2.5%** |
| Gemma 4 IQ2_M b=1| 44.7 tok/s | 46.2 tok/s | **+3.3%** |

**Report Location**: `/root/for-gemini/emu-local/results/A100_80_CALIBRATION_REPORT.md`
