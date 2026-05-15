# A100-80GB Calibration Study

> Date: 2026-05-15  
> Machine: A100-SXM4-80GB (Vast.ai)  
> Observed Bandwidth: **1314.8 GB/s** (DLPerf)

## 1. Executive Summary
Calibration on the 80GB instance successfully removed the memory-pressure artifacts observed on the 40GB machine (Step 5). Moving to 80GB allowed disabling `--enforce-eager` and removing KV-preemption, revealing the true potential of the SXM4 architecture.

## 2. Calibration Results (vLLM 0.9.1)

| Model | Variant | alpha (MFU) | beta (MBU) | Saturation | Notes |
|---|---|---:|---:|---|---|
| **Qwen-2.5-7B** | AWQ.4bit | 0.453 | 0.560 | (7.8, 6.8) | beta isolated from 16k context |
| **Qwen3-32B** | AWQ.4bit | 0.487 | 0.560 | (7.0, 4.2) | 2026 Dense Model |
| **Qwen-2.5-72B**| AWQ.4bit | 0.537 | 0.560 | (8.2, 5.5) | Replaces Llama-70B 40GB baseline |

### MoE First Contact (llama.cpp)
Due to weight-name conflicts in vLLM 0.9.1 for the Llama-4-Scout MoE model, initial validation was performed via `llama.cpp` (GGUF IQ2_M):
- **Llama 4 Scout (109B MoE, 17B active)**:
  - Prefill (p1024): **798.6 tok/s**
  - Decode (b1): **68.1 tok/s**
  - Implied $\alpha$ (compute-bound): 0.087 (prefill efficiency in llama.cpp)
  - Implied $\beta$ (bandwidth-bound): 0.56 (consistent with dense models)

## 3. Findings
- **MFU Scaling**: Larger models (72B) achieve significantly higher MFU (**0.537**) than smaller models (7B, 0.453) on A100-80GB, indicating better hardware utilization by larger tensor-core kernels.
- **Eager Penalty Removal**: The 72B model on 80GB (without eager) is **~2.3x faster** than the 70B model on 40GB (with eager/preemption), confirming that VRAM overhead is the primary bottleneck for large-scale serving.
- **2026 Model Support**:
  - **Llama 4 Scout**: Verified native architecture support in vLLM 0.21.0, though weight loading required patches to `transformers` for `rope_theta` and `attn_temperature_tuning`.
  - **Gemma 4 31B**: Successfully parsed config after patching `transformers` with a `__getattr__` proxy. Still prone to OOM in multimodal components when falling back to Transformers implementation.

## 4. Validation Cross-Check
| Scenario | Real | Emulator (Calibrated) | Delta |
|---|---|---|---|
| Qwen-72B b=32 | 138.0 tok/s | 134.5 tok/s | **-2.5%** |
| Qwen-7B ctx=16k| 6.31 ms/t | 6.45 ms/t | **+2.2%** |

**Report Location**: `/root/for-gemini/emu-local/results/A100_80_CALIBRATION_REPORT.md`
