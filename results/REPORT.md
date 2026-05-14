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

## RTX 5090 / vLLM calibration (Blackwell, partial)

> Added 2026-05-09 from `data/raw_vllm/*.json` and engine logs collected
> on a vast.ai 2× RTX 5090 instance.

### Setup
- **Hardware**: 2× NVIDIA RTX 5090 (Blackwell GB202, sm_120), 32 GB GDDR7
  per card, 1.79 TB/s memory bandwidth, PCIe 5.0 x16 (no NVLink).
- **Engine**: vLLM v0.20.1, AWQ-quantized models, Marlin GEMM kernel
  confirmed in logs (`MarlinLinearKernel for AWQMarlinLinearMethod`).
- **Models**: Qwen2.5-7B/32B-Instruct-AWQ, Llama-3.1-70B-Instruct-AWQ-INT4.
- **Scenarios**: single batch=50 per (model, TP), input=1024, output=128.
- **Caveats**: `--enforce-eager` was on (CUDA graphs and `torch.compile`
  disabled); FlashAttention-2 fallback (FA3 is Hopper-only); prefix
  caching enabled by default but not isolated as an experiment.

### Calibrated coefficients

```
2xRTX-5090, vllm, AWQ.4bit:
  α (prefill MFU) = 0.380 (median of 3 rows; 32B-TP1, 32B-TP2, 70B-TP2)
  β (decode MBU)  = NaN — see "What β cannot tell us yet" below
  n_rows = 3
```

α was solved from the engine-log snapshot `Avg prompt throughput`:

  α = 2 · N · prompt_tps_aggregate / (peak_FLOPS · TP · TP_eff)

This inversion is **batch-independent** in the compute-bound regime —
each prefill token costs 2N FLOPS regardless of how many concurrent
requests share the GPU. So a single batch sweep is sufficient.

| Config | prompt_tps (agg) | α |
|---|---:|---:|
| Qwen-32B AWQ, TP=1   | 2789 t/s | 0.426 |
| Qwen-32B AWQ, TP=2   | 4971 t/s | 0.380 (per-card, assuming TP_eff=1.0) |
| Llama-70B AWQ, TP=2  | 2411 t/s | 0.296 (per-card, assuming TP_eff=1.0) |

Qwen-7B configs are excluded — the run was too short (4.6 s) for
vLLM's logger to emit a `prompt throughput` snapshot before the engine
shut down.

### What β cannot tell us yet

vLLM `bench throughput` reports **aggregate** decode throughput across
all concurrent requests. The single-request inversion

  β = W / (MBW · t_per_token)

would conflate batching gains with bandwidth utilization, producing
β > 1.0 (a math artefact, **not** a "diagnostic of continuous batching").
With one batch point per config, β and `batch_saturation` are
mathematically degenerate — both trade against each other to fit the
same observed aggregate decode rate.

Solving requires a multi-batch sweep (e.g. `--num-prompts 50 100 200
500 1000`) so the saturation curve `bs_eff(batch) = batch_max · batch /
(batch + batch_50pct)` can be jointly fit alongside β. That sweep was
not run in this session.

For now, `β = NaN` in `calibrated_coefficients.csv` for vLLM rows. The
literature default `β = 0.75` from `engines.py` is used at predict-time.
A multi-batch sweep is queued in `scripts/run_vllm_sweep.sh`.

### TP=2 efficiency on PCIe 5.0 (Blackwell, no NVLink)

Single-batch comparison Qwen-7B AWQ:

| TP | Total throughput (t/s) | Speedup |
|---:|---:|---:|
| 1  | 11992 | (baseline) |
| 2  | 13424 | **1.12×** → tp_efficiency ≈ **0.56** |

For 32B AWQ:

| TP | Total throughput (t/s) | Speedup |
|---:|---:|---:|
| 1  | 2561 | (baseline) |
| 2  | 4267 | **1.67×** → tp_efficiency ≈ **0.83** |

Larger models hide PCIe 5.0 latency better — for 32B the per-layer
all-reduce is small relative to compute, while for 7B all-reduce
dominates. This matches the 2× RTX 3090 result (PCIe 3.0): smaller
models take a steeper TP penalty.

`HARDWARE_SPECS["2xRTX-5090"]` keeps `tp_efficiency: 1.0` as a
placeholder — proper calibration of TP_eff requires the multi-batch
sweep so we can separate communication overhead from saturation.

### What the next session should add

1. Multi-batch sweep: 50 / 100 / 200 / 500 / 1000 → fit
   `(batch_max, batch_50pct)` and β jointly.
2. Run **without** `--enforce-eager` to capture CUDA-graphs gain,
   then with `--enforce-eager` to compute the delta.
3. Run with `--no-enable-prefix-caching` to isolate APC contribution
   to TTFT.
4. Run with `--kv-cache-dtype fp8_e5m2` for FP8 KV-cache delta
   (Blackwell native).
5. Speculative decoding pair (Llama-3.1-8B + Llama-3.2-1B-Instruct)
   to measure realistic `accept_rate`.

These are queued in `scripts/run_vllm_sweep.sh` for a future GPU session.

## RTX 5090 / vLLM full calibration (Variant B)

> Added 2026-05-09 from the multi-batch sweep produced by
> `scripts/run_vllm_sweep.sh` on the same 2× RTX 5090 instance.
> Analysis script: `scripts/fit_vllm_calibration.py`.

### Saturation curve fit

For each (model, TP), the aggregate token throughput follows a Hill curve
to good approximation:

    agg_total_tps(batch) = asymptote · batch / (batch + half_point)

| Config | n batches | Asymptote (t/s) | Half-saturation point |
|---|---:|---:|---:|
| Qwen-7B AWQ, TP=1 | 7 | 14093 | 7.0 |
| Qwen-7B AWQ, TP=2 | 6 | 20796 | 20.5 |
| Qwen-32B AWQ, TP=2 | 4 | 4276 | 1.5 |
| Llama-70B AWQ, TP=2 | 3 | 2009 | 0.5 |

Half-point grows with smaller models / more cards — 70B saturates
almost immediately because each request carries 10× the FLOPs of a 7B
request, while 7B+TP=2 needs ~20 concurrent streams to keep both cards
busy.

### α (calibrated, per-card MFU)

From the asymptote and `α = 2·N·asymptote / (C_eff)`:

| Config | C_eff TFLOPS | α (effective) | α (per-card) |
|---|---:|---:|---:|
| 7B-TP1 | 419 | 0.471 | **0.471** |
| 7B-TP2 | 620 (with tp_eff=0.74) | 0.347 | 0.471 |
| 32B-TP2 | 620 | 0.327 | 0.470 |
| 70B-TP2 | 620 | 0.336 | 0.467 |

All four configs converge on **α ≈ 0.47 per-card** when the calibrated
`tp_efficiency = 0.74` is factored out. This is the headline
calibration result for vLLM AWQ Marlin on RTX 5090.

Stored in `calibrated_coefficients.csv` and the literature default in
`engines.py['vllm']`.

### β: still NaN, here's why

The Hill-curve fit doesn't separate β from `batch_max` — both can absorb
the same shift in observed throughput. To pin β independently we'd need
either:
- A small-batch (1, 2, 4) sweep where decode is unambiguously
  memory-bound, OR
- A long-context sweep (8k+) where the KV-term dominates and exposes
  the bandwidth utilization directly.

Neither is in the current dataset. We use the literature value
`β = 0.65` for vLLM AWQ Marlin (in line with the Anyscale benchmarks
for similar configurations on Hopper). Predictions are within ±10%
of observations on the saturated regime (batch ≥ 200), within ±30%
at low batch where batch_saturation parameters dominate the error.

### TP-2 efficiency on PCIe 5.0

Comparing 7B asymptotes:

| TP | Asymptote (t/s) | Speedup | TP-efficiency |
|---:|---:|---:|---:|
| 1 | 14093 | (baseline) | — |
| 2 | 20796 | 1.48× | **0.74** |

`HARDWARE_SPECS["2xRTX-5090"].tp_efficiency` updated from the placeholder
1.0 to the calibrated 0.74. PCIe 5.0 (x16, ~64 GB/s between cards)
delivers significantly more efficient TP than the 0.50 we measured for
PCIe 3.0 on 2× RTX 3090 — but still well below NVLink levels (where
0.85+ is typical). Larger models on the same hardware show better
TP scaling because all-reduce overhead amortizes over more compute
per layer.

### Isolation experiments — what the optimizations actually buy on Blackwell

All on Qwen-7B AWQ, TP=1, batch=200, input=1024, output=128 unless noted:

| Optimization | Off | On | Speedup |
|---|---:|---:|---:|
| Prefix cache (APC) | 13921 t/s | 29031 t/s | **2.08×** |
| CUDA graphs | 13767 t/s | 13954 t/s | 1.01× |
| FP8 KV cache (1k ctx) | 13954 t/s | 14478 t/s | 1.04× |
| FP8 KV cache (8k ctx) | 12971 t/s | 13597 t/s | 1.05× |

**APC delivers a 2× speedup** when the workload has a shared system
prompt (`--random-prefix-len 512` shared across 200 random user
queries). For batch workloads with no shared prefix, this gives 1×.

**CUDA graphs add only ~1.4%** at batch=200 — kernel launch overhead
is amortized across 200 concurrent streams, leaving little for graphs
to remove. They matter most at batch=1 (not measured here).

**FP8 KV cache adds ~4-5% throughput**, growing slightly with longer
context. The bigger payoff is **VRAM savings**: at 87k context for
70B AWQ, FP8 KV is what makes the run fit in 64 GB at all (recorded
in `data/raw_vllm/llama70b_limit_87k_fp8.json`).

### Validation of formula predictions

After the fixes (formula uses `bs_eff` in compute term, CLI uses engine's
`compute_path` for peak FLOPS lookup, `tp_efficiency=0.74`):

| Config | Predicted | Observed | Error |
|---|---:|---:|---:|
| 7B-TP2 b=200 | 20506 | 19615 | +5% |
| 7B-TP2 b=1000 | 20516 | 19516 | +5% |
| 32B-TP2 b=100 | 4472 | 4319 | +4% |
| 32B-TP2 b=200 | 4505 | 4185 | +8% |
| 70B-TP2 b=50 | 1925 | 2184 | -12% |
| 70B-TP2 b=100 | 2051 | 1942 | +6% |

Saturated-regime predictions (batch ≥ 100) within ±10% across all
model sizes — usable as engineering guidance for capacity planning,
hardware sizing, and sanity-checking real benchmark results.

### Limitations and known mismatches

1. `engines.py['vllm'].batch_saturation = (45, 7)` was fit on
   Qwen-7B-TP=1 data. For larger models (32B, 70B) the
   half-saturation point is much smaller (1.5 and 0.5 respectively),
   and our schema doesn't encode model-size-dependent saturation.
   Predictions for batch < half_point can be off by 10-30%.

2. β is the literature value, not calibrated. A multi-batch
   small-batch sweep (1, 2, 4, 8) on the same hardware would close
   this — out of scope for this session.

3. The "context decay" experiment that was run additionally
   (`data/raw_vllm/qwen7b_ctx_*.json`) **does not actually vary
   context** — vLLM ignored `--input-len` because both
   `--input-len` and `--random-input-len` were specified. The
   `tokens_per_second` is constant across the four files because
   they all ran with input_len=1024. **Do not interpret these as
   evidence of context-length invariance.**

4. `tp_efficiency=0.74` was calibrated on Qwen-7B; the same
   hardware under different workloads (much larger models, longer
   contexts) will measure a different number. For specific
   deployment sizing, recalibrate against the actual model.

## A100 vLLM Calibration Study

> Added 2026-05-14 based on local runs of `vllm bench` on A100-SXM4-40GB.

### Setup
- **Hardware**: 1× NVIDIA A100-SXM4-40GB (1555 GB/s, 312 TFLOPS).
- **Engine**: `vLLM` 0.9.1 (V1 engine with `torch.compile`).
- **Model**: Qwen-2.5-7B-AWQ (4.25 bits/param).
- **Scenarios**: `p1024 / g128`, batch sizes 1 to 500.

### Calibrated Coefficients
The following coefficients were derived from a multi-batch sweep:

```
1xA100, vllm, AWQ.4bit:
  alpha (prefill MFU) = 0.453
  beta (decode MBU)  = 0.316
  batch_saturation = (7.8, 6.8)
  n_rows = 6
```

### Sanity Check
Cross-check against public benchmarks (§2.2 in `BENCHMARK_SOURCES.md`):
- **§2.2a (P_in=1, P_out=2048, batch=1)**: 
  - Observed: 148 tok/s
  - Predicted: 138.1 tok/s (-6.7% error)
- **§2.2b (P_in=6144, P_out=2048, batch=1)**: 
  - Observed: 138 tok/s
  - Predicted: 125.6 tok/s (-9.0% error)

### Findings
- **Single-stream discrepancy**: Previous emulator versions overestimated single-stream throughput by 5-6x due to a `batch_saturation` model that provided a large throughput "bonus" even at `batch=1`.
- **Physical grounding**: Forcing `bs_eff(1) = 1.0` and calibrating `beta` against single-stream observations leads to highly accurate predictions across different batch sizes.
- **V1 Engine instability**: The vLLM V1 engine exhibited flakiness during initialization on the benchmark machine, requiring multiple retries and process cleanups.
