# A100 Instance Details (Pinned)

> Data captured 2026-05-15 from Vast.ai instance `36778851`.

## Hardware Specifications
- **GPU**: 1× NVIDIA A100-SXM4-40GB
- **VRAM**: 40.0 GB (39.49 GiB available to PyTorch)
- **Memory Bandwidth**: 1314.8 GB/s (Observed DLPerf)
- **Compute**: 15.6 TFLOPS (Spec reported, FP16 peak is 312 TFLOPS)
- **CPU**: AMD EPYC 7713 64-Core (32 vCPUs assigned)
- **RAM**: 129.0 GB
- **Disk**: Micron_7450 150.1 GB (2363.0 MB/s)
- **Motherboard**: ROME2D32GM-2T (PCIe 4.0/8x, 10.9 GB/s)

## Calibration Findings (2026-05-14/15)

### Qwen-7B AWQ (1xA100-40)
- **alpha**: 0.453
- **beta**: 0.44 (fitted for 1314.8 GB/s)
- **batch_saturation**: (7.8, 6.8)
- **Notes**: Very stable, fits well in 40GB.

### Llama-3.1-70B GPTQ (1xA100-40)
- **alpha (eff)**: 0.221 (low due to preemption and eager mode)
- **beta (fitted)**: 0.67 (fits single-stream b1 baseline)
- **batch_saturation**: (21.5, 2.5)
- **Notes**: Extremely tight VRAM. Model weights take 36.86 GiB. vLLM leaves only 0.55 GiB for KV cache. Preemption observed starting from Batch 8. Eager mode required to load.

## Model Versions
- **vLLM**: 0.9.1 (V0 engine used for 70B stability)
- **Transformers**: 4.51.1
- **Torch**: 2.7.0
