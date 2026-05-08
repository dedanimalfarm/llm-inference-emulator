# Calibration report

> Generated 2026-05-08 from `scripts/run_calibration.py` and
> `scripts/compare.py` on the `data/leaderboard-*.csv` snapshot.

## Summary

| Metric | Uncalibrated | Coarse calibration | Fine calibration |
|---|---:|---:|---:|
| Prefill median rel-err | 0.790 | 0.652 | 0.644 |
| Decode  median rel-err | **5.353** | **0.750** | 0.734 |
| Rows evaluated | 3290 | 3290 | 3260 |

**Decode prediction error drops 7×** once we replace literature defaults with
medians solved back from observed (prefill, decode/throughput). Prefill
improves only modestly because the underlying benchmark mixes Eager, SDPA
and FlashAttention-2 in the same row population, and the
`(hw, backend, precision)` bucket is too coarse to separate them. Splitting
by `attention` ("fine" mode) brings barely any extra signal — meaning the
remaining variance comes from per-model architecture, not from the engine
itself.

## Per-hardware breakdown

```
                         rows  prefill_med  prefill_p90  decode_med  decode_p90
mode         hw                                                                
uncalibrated 1xA10       1736        0.785       1.000      6.626      117.904
             1xA100       286        0.874       1.000     44.769      450.408
             1xT4         971        0.819       1.000      2.692       66.944
             32vCPU-C7i   297        0.259       1.000      0.829       26.151
coarse       1xA10       1736        0.747       1.000      0.855       36.030
             1xA100       286        0.395       1.000      0.594       29.075
             1xT4         971        0.498       1.000      0.949       36.090
             32vCPU-C7i   297        0.264       1.000      0.300       13.933
fine         1xA10       1736        0.727       1.000      0.870       37.207
             1xA100       286        0.384       1.000      0.583       29.575
             1xT4         971        0.520       1.000      0.839       36.385
             32vCPU-C7i   267        0.258       1.000      0.286       14.005
```

Read this as: *the median predicted decode/throughput on A100 is 59% off after
coarse calibration vs. 4500%(!) off with literature defaults*. The huge
uncalibrated A100 error is because PyTorch on A100 reaches MFU/MBU much higher
than the 0.20/0.55 default, and the unscaled formula is therefore an
order of magnitude too pessimistic.

## Calibrated medians per bucket

```
        hw     backend precision_label  n  alpha_med  beta_med
     1xA10     pytorch     Unquantized 270    0.2614    0.4554
     1xA10     pytorch        AWQ.4bit 457    0.0731    0.1648
     1xA10     pytorch        BnB.4bit 143    0.0427    0.0942
     1xA10     pytorch        BnB.8bit 128    0.1127    0.0979
     1xA10     pytorch       GPTQ.4bit 296    0.0476    0.0355
    1xA100     pytorch     Unquantized  79    0.2525    0.1974
    1xA100     pytorch        BnB.4bit  44    0.0202    0.0288
    1xA100     pytorch        BnB.8bit  41    0.0509    0.0305
    1xA100     pytorch       GPTQ.4bit  71    0.0340    0.0113
      1xT4     pytorch     Unquantized 142    0.0430    0.4795
      1xT4     pytorch        AWQ.4bit 210    0.0646    0.3033
      1xT4     pytorch        BnB.4bit  88    0.0268    0.1485
      1xT4     pytorch        BnB.8bit  75    0.2003    0.1766
      1xT4     pytorch       GPTQ.4bit 168    0.1245    0.3272
      1xT4     pytorch    torchao.4bit   4    0.0216    0.0707
32vCPU-C7i     pytorch     Unquantized 200    0.1861    0.3773
32vCPU-C7i onnxruntime     Unquantized  12    0.1671    0.2795
```

### Read the table

- **Unquantized rows are higher on both axes** — this is the reference
  case where the kernel does plain FP16 matmul. A10 reaches ~26% MFU
  prefill and 46% MBU decode; A100 ~25% / 20%. T4 PyTorch unquantized
  has very low alpha because most rows in that bucket are small models
  where prefill is dominated by overhead.
- **GPTQ/AWQ/BnB drag both numbers down sharply.** The dequantization-on-the-fly
  kernels do significantly more work per matmul than peak FLOPS would suggest,
  which the formula attributes to a low effective `α`. Likewise, `β` falls
  below the unquantized case because INT4 weights are read at peak speed but
  the kernel additionally reads scales / zeros, etc. These calibrated values
  should be read as "what PyTorch eager achieves" — not what TensorRT-LLM
  can achieve (often 2-3× higher).
- **A100 BnB.4bit `α=0.020`** stands out as very low. Cross-checking against
  the leaderboard: 7B BnB.4bit prefill on A100 is ~2.1 s — vs ~0.07 s for
  GPTQ.4bit on the same hardware. BnB has a known slow CUDA kernel; this
  is a real engine artefact, captured correctly by the calibration.

## Where the prediction is wrong

The 90th-percentile decode error stays large (15-30×) even after calibration
because some rows reflect:

- thrashing into CPU-pinned memory at the edge of the GPU's capacity,
- experiments with very short generation that include a disproportionate
  warm-up cost,
- attention implementations whose memory traffic doesn't match the simple
  KV-cache assumption (e.g. multi-query attention reads a lot less per token).

For better tails, one would need either (a) a compiled benchmark per
(model architecture × engine × precision) cell, or (b) a learned residual
on top of the roofline prediction.

## Cross-check: an example prediction

```
$ python scripts/cli.py --model 7 --bits 4 --hw 1xT4 \
                        --engine pytorch --precision-label GPTQ.4bit

# using calibrated coefs for (1xT4, pytorch, GPTQ.4bit)
1xT4 | pytorch | 4-bit | 7.0B params | bs=1
  alpha=0.124, beta=0.327, batch_mult=1.0
  prefill          : 221.4 ms (compute-bound)
  decode/token     : 37.19 ms (memory-bound)
  total latency    : 2.602 s for 64 new tokens
  throughput       : 26.9 tok/s (effective batch ×1.0)
```

Compare to the leaderboard for 7B GPTQ on T4 (median 27 tok/s decode,
prefill ranges 0.20–0.30 s depending on attention) — the calibrated
emulator lands inside the observed envelope.

## RTX 3090 / llama.cpp Calibration Study

> Added 2026-05-08 based on local runs of `llama-bench`.

### Setup
- **Hardware**: 2× RTX 3090 on Vast.ai (benchmarked on 1 unit via `CUDA_VISIBLE_DEVICES=0`).
- **Engine**: `llama.cpp` with CUDA backend.
- **Model**: Qwen-2.5-7B Q4_K_M (4.91 bpw effective).
- **Scenarios**: `pp512` / `tg128`, context depth up to 24,576 tokens, batch sizes 128 to 4096, symmetric and mixed KV quantization, Flash-Attention on/off.

### Calibrated Coefficients
The following coefficients were derived from 11 paired benchmark observations (after outlier filtering and deduplication of replicates):

```
RTX-3090, llama.cpp, Q4_K (4.91bpw):
  α (prefill MFU) = 0.36 (median, range 0.17-0.62)
  β (decode MBU)  = 0.72 (median, range 0.25-0.79)
  n_rows = 11
```

### Sanity Check
Manual baseline computation for the cleanest scenario (pp512, d=0, FA=1, batch=2048):
- **Observed**: avg_ts = 5746.9 t/s
- **Calculated**: $\alpha$ = 0.616
- **Result**: Matches the upper end (p75) of our calibrated range, confirming the formula is physically grounded.

### Findings

1.  **Mixed-precision KV — slow path** (50× regression):
    We confirmed a massive performance drop when using asymmetric KV quantization (e.g., `-ctk f16 -ctv q8_0`).
    - `f16/f16` prefill: 5710 t/s
    - `f16/q8_0` prefill: 110 t/s (**×52 slower!**)
    - **Mitigation**: Always use symmetric KV quantization in `llama.cpp` (e.g., `-ctk q8_0 -ctv q8_0`).

2.  **GQA Support in formula**:
    Accurate `kv_per_token_bytes` calculation is critical for models using Grouped Query Attention (GQA). For Qwen-2.5 (kv_heads=4) and Llama-3 (kv_heads=8), neglecting GQA overestimates KV-cache size by 8-32×, leading to significantly lower calibrated $\beta$.

3.  **Batch Saturation Curve**:
    Prefill performance (`pp1024`) on RTX 3090 saturates early:
    - `n_batch=128`: 2557 t/s
    - `n_batch=256`: 5124 t/s
    - `n_batch=512`: 5237 t/s
    - `n_batch=1024`: 5158 t/s
    Saturation is reached at approximately **256 tokens**.

4.  **Cross-Model Invariance**:
    Validation run on **Llama-3.1-8B** yielded $\alpha$ and $\beta$ within **3.1%** and **1.7%** of the Qwen-7B baseline respectively (at depth 0). This confirms that calibrated coefficients for the engine/hardware pair are model-invariant when architecture parameters (GQA heads, layers) are correctly specified.

### Tier A Bug-Fix Summary
The initial calibration showed significantly lower values ($\alpha \approx 0.09, \beta \approx 0.39$). Tier A fixes addressed:
- **Compute Path**: Corrected peak FLOPS calculation for GGUF (must use FP16 path, 142 TFLOPS).
- **KV Context**: Used real `n_depth` instead of fixed 288 tokens.
- **Result**: Calibrated medians improved to **$\alpha=0.36, \beta=0.72$**, aligning with hardware limits.

## Multi-GPU calibration (2× RTX 3090)

> Added 2026-05-08.

### Setup
We tested large models that either fit on one card (Qwen-32B) or require two (Llama-70B) to measure scaling efficiency. 
- **Hardware**: 2× RTX 3090 connected via PCIe 3.0 (No NVLink).
- **Models**: Qwen-2.5-32B (Q4_K_M), Llama-3.1-70B (Q4_K_M).

### Two split modes — two different parallelism strategies

llama.cpp offers two `--split-mode` strategies on multi-GPU, and they
behave very differently:

| Split mode | What it does | What it parallelises |
|---|---|---|
| `layer` | partitions decoder layers across GPUs | **pipeline parallel** (sequential) |
| `row`   | partitions weight tensors across GPUs (per-layer) | **tensor parallel** (concurrent) |

Both add multi-GPU **capacity**, but only `row` adds compute concurrency.

### Observed numbers (Qwen-2.5-32B Q4_K_M, pp512 baseline)

| Configuration | pp512 (t/s) | TP-efficiency vs theoretical 2× |
|---|---:|---:|
| Single GPU (1× 3090) | 1367.77 | (baseline) |
| Multi-GPU `layer` split | **1367.39** | **0.500** |
| Multi-GPU `row` split   | 614.72   | 0.225 |

### Interpretation

**Layer split = pipeline parallel.** Each token traverses both GPUs
sequentially — GPU 0 runs layers 1-32, then hands the activations to
GPU 1 for layers 33-64. The two cards do **not** work concurrently on
the same token; they alternate. So aggregate throughput is bounded by
single-card throughput, not 2×. The "0.50 TP-efficiency" by our metric
(`multi / (2 × single)`) is the inevitable consequence — we expected
2× and got 1×.

**Row split = real tensor parallel.** Each layer's weights are sliced
between cards and computed concurrently; an all-reduce after the
matmul reconciles partial outputs. This **does** require fast
inter-GPU bandwidth, which PCIe 3.0 (16 GB/s) lacks compared to
NVLink (300+ GB/s on H100). Result: row-split is **slower than single
card** (0.225 efficiency) because inter-GPU communication eats more
than the parallelism saves.

### So what does multi-GPU on 2× 3090 actually buy?

**Capacity, not throughput.** It lets you run models that don't fit on a
single 24 GB card (like Llama-70B Q4_K_M ≈ 40 GB). For models that
fit on one card, a second 3090 is useless for performance — you get
the same prefill and same decode as with one card.

Practical implications for emulator users:
- `predict(model=7B, hw="2xRTX-3090")` returns **the same latency** as
  `predict(model=7B, hw="RTX-3090")` — that's mathematically correct
  (`tp_size × tp_efficiency = 2 × 0.5 = 1.0`).
- Only consider 2× 3090 when the model exceeds 24 GB.
- For higher throughput on small models, prefer **one bigger GPU** (A100,
  H100, etc.) over **two smaller ones** without NVLink.

### Large model that needs both cards (Llama-3.1-70B Q4_K_M)

40 GB weights — fits only when split across two cards.

| Metric | Value |
|---|---:|
| pp512 | 638.6 t/s |
| tg64  | **19.33 t/s** |

Decode 19 t/s on a $1500 used-3090-pair is competitive with cloud
A100 deploy of the same model. The bottleneck is memory bandwidth on
the local cards (936 GB/s × 2 effective ≈ 1.87 TB/s combined for
weights, but each card serves only its own slice).

### Emulator updates from this study

- `predict()` accepts `tp_size: int = 1` and `tp_efficiency: float = 1.0`.
  Default behaviour unchanged.
- `2xRTX-3090` entry added to `HARDWARE_SPECS` with `tp_size=2,
  tp_efficiency=0.50`.
- `engines.py['llama.cpp']` notes already warn about mixed-quant KV
  (50× regression). No change there.

The 0.50 efficiency is **not** a literature-default — it is empirically
measured for `split-mode layer` (the most common llama.cpp config). For
other engines (vLLM with NCCL all-reduce on NVLink hosts), `tp_efficiency`
should be re-calibrated separately.
