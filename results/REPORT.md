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
