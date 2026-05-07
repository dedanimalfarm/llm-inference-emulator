# LLM Inference Emulator

A roofline-based estimator for LLM inference latency / throughput, with
**engine-aware** coefficients (PyTorch, vLLM, TensorRT, llama.cpp, ...) and
calibration against a public benchmark
([LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)).

The goal is **back-of-the-envelope answers** to questions like
*"How many tokens/s do I get from a 13B model on A10 with vLLM at INT4?"* —
without renting the GPU.

## The formula

LLM autoregressive decoding has two phases with different bottlenecks:

| Phase | Per request | Bottleneck |
|---|---|---|
| **Prefill** | encodes `P_in` prompt tokens in parallel | compute |
| **Decode** | generates each of `P_out` tokens sequentially | memory bandwidth |

Both are bounded by the **roofline model** — the slower of (compute-time, memory-time):

```
t_prefill = max(  2·N·P_in·bs / (C · α),     W / MBW              )
                  └ compute path ─┘           └ at least one weight pass ┘

t_decode  = max(  W / (MBW · β),              2·N·bs / (C · α)    ) + KV cache term
                  └ memory path ─┘             └ compute path ┘

throughput = bs · γ_batch / t_decode
```

where:
- `N` — model parameters (e.g. 7e9)
- `b` — bits per weight (16/8/4); `W = N·b/8` is model size in bytes
- `C` — peak FLOPS for the precision (HW spec)
- `MBW` — memory bandwidth in B/s (HW spec)
- `α` — Model FLOPs Utilization in prefill (engine-specific, 0.15–0.65)
- `β` — Memory Bandwidth Utilization in decode (engine-specific, 0.55–0.95)
- `γ_batch` — effective batching multiplier (vLLM PagedAttention etc.)

**Engines** don't change `C`/`MBW` (those are physics). They change `α`, `β`,
`γ_batch` — see [`emulator/engines.py`](emulator/engines.py).

## Quick start

```bash
pip install -r requirements.txt

# emulate a single configuration
python scripts/cli.py --model 7 --bits 4 --hw 1xA100 --engine vllm \
                     --batch 8 --precision-label GPTQ.4bit

# show a comparison grid across engines / hardware
python scripts/demo.py
```

## Calibration on real data

```bash
# read leaderboard CSVs, solve for (alpha, beta) per row,
# aggregate by (hw, backend, precision_label)
python scripts/run_calibration.py

# evaluate predicted vs actual on every leaderboard row
python scripts/compare.py
```

The output of compare.py on the snapshot included here:

```
Median relative error  (1.0 = predicted is 100% off)
                         rows  prefill_med  decode_med
mode         hw                                      
uncalibrated 1xA10       1736        0.785       6.626
             1xA100       286        0.874      44.769
             1xT4         971        0.819       2.692
             32vCPU-C7i   297        0.259       0.829
calibrated   1xA10       1736        0.747       0.855
             1xA100       286        0.395       0.594
             1xT4         971        0.498       0.949
             32vCPU-C7i   297        0.264       0.300
```

Calibration improves the **decode prediction by ~7×** (median error 5.4 → 0.75).
Prefill improves more modestly (0.79 → 0.65) because the leaderboard mixes
several attention implementations (Eager / SDPA / FAv2) into one bucket and
the per-row spread within a bucket is large.

See [`results/REPORT.md`](results/REPORT.md) for analysis, scatter plots
in CSV form, and the full `prediction_vs_actual.csv`.

## Layout

```
emulator/
  hardware.py         # peak FLOPS / memory bandwidth per HW
  engines.py          # default α, β, batch multiplier per engine
  formula.py          # roofline calculation
  calibrate.py        # invert the formula on observed (prefill, decode)
scripts/
  run_calibration.py  # produces calibrated_coefficients.csv
  compare.py          # predicted vs actual on the leaderboard
  cli.py              # one-shot CLI
  demo.py             # comparison grid
data/                 # leaderboard snapshot used for calibration
  leaderboard-1xA10.csv
  leaderboard-1xA100.csv
  leaderboard-1xT4.csv
  leaderboard-32vCPU-C7i.csv
results/
  raw_coefficients.csv          # one row per observation
  calibrated_coefficients.csv   # by (hw, backend, precision)
  calibrated_fine.csv           # by (hw, backend, precision, attention)
  prediction_vs_actual.csv
  error_summary_*.csv
  REPORT.md                     # analysis
```

## Limitations

- A roofline model is **an upper bound**. The formula assumes the engine can
  saturate either compute or bandwidth, then a single coefficient absorbs all
  reality (kernel launch overhead, scheduler stalls, padding, etc.).
- **Single accelerator only.** No model parallelism, no tensor / pipeline
  parallel splits, no PCIe-bound multi-GPU traffic.
- **Decode is modelled at fixed avg context.** For very long prompts the KV
  cache term dominates and the residual error grows.
- The benchmark scenario is fixed: **`bs=1, P_in=256, P_out=64`** (LLM-Perf
  Leaderboard standard). Calibration outside that regime is interpolation.
- vLLM `γ_batch` and TensorRT `α` come from the literature, not from this
  dataset (the leaderboard is PyTorch-only). Treat them as starting points,
  override from your own measurements.

## How to extend

- New hardware → add to [`HARDWARE_SPECS`](emulator/hardware.py).
- New engine → add to [`ENGINE_DEFAULTS`](emulator/engines.py) (or measure on
  a representative model and call `calibrate_row` directly).
- Long-context: tweak the KV cache term in `formula.py` to account for
  multi-query / grouped-query / sliding-window attention.

## Source data attribution

The CSVs in `data/` are derived from
[`optimum-benchmark/llm-perf-leaderboard`](https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard)
via the [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)
processing pipeline. Snapshot reproduced in
[`dedanimalfarm/llm-perf-leaderboard-snapshot`](https://github.com/dedanimalfarm/llm-perf-leaderboard-snapshot).
