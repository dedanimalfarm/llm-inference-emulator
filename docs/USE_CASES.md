# Use cases — что эмулятор отвечает

Roofline-эмулятор приблизителен (см. *Limitations* в конце), но закрывает
~80% deployment-вопросов **без аренды GPU**. Ниже — 7 конкретных решений,
которые он разблокирует, с числами, посчитанными прямо в этом репо.

## What this gives you

A roofline emulator like this is approximate (see [Limitations](#limitations)),
but it answers ~80% of inference-deployment questions **without spinning up
the actual hardware**. Here are the concrete decisions it unblocks, with
numbers run on this very repo.

### 1. Pick hardware before paying for it

You're shipping a 7B chatbot, vLLM, batch=8. Question: T4, A10 or A100?

```bash
$ python scripts/cli.py --model 7 --bits 16 --engine vllm --batch 8 --hw 1xT4
  prefill : 1102 ms     throughput : 636 tok/s    memory : 15.3 GB
$ python scripts/cli.py --model 7 --bits 16 --engine vllm --batch 8 --hw 1xA10
  prefill :  573 ms     throughput : 1272 tok/s   memory : 15.3 GB
$ python scripts/cli.py --model 7 --bits 16 --engine vllm --batch 8 --hw 1xA100
  prefill :  230 ms     throughput : 4322 tok/s   memory : 15.3 GB
```

Rough AWS pricing puts the per-request cost roughly equal across these
three (the A10 wins by a hair on $/M tokens at this load). The decision
becomes **latency-driven**: if your SLO is *first-token < 300 ms*, only
A100 qualifies; if you have ~1 s budget, A10 saves money. T4 is rarely
the right answer for 7B+ today.

### 2. Quantization isn't a free lunch — model the trade-off

13B model on A10. Compare FP16 vs GPTQ.4bit:

```bash
$ python scripts/cli.py --model 13 --bits 16 --hw 1xA10 --engine pytorch \
                       --precision-label Unquantized
  prefill : 203 ms      throughput : 10.4 tok/s   memory : 26.3 GB  ← exceeds 24 GB!
$ python scripts/cli.py --model 13 --bits 4 --hw 1xA10 --engine pytorch \
                       --precision-label GPTQ.4bit
  prefill : 559 ms      throughput :  3.2 tok/s   memory :  6.8 GB  ← fits
```

The first surprise: **GPTQ.4bit on PyTorch is *slower* than FP16 here**, even
though the weights are 4× smaller. The reason is in the calibrated `α`/`β`:
GPTQ's dequantize-on-the-fly kernel achieves only ~12% of peak bandwidth
versus 46% for unquantized. *To actually realise the 4× theoretical decode
speedup you need a fused INT4 GEMM* — i.e. **vLLM, TensorRT-LLM,
ExLlamaV2** — none of which PyTorch eager has.

The second insight: FP16 *doesn't fit anyway*, so the real choice is
"GPTQ on a worse engine" vs "fewer params" vs "bigger GPU". This is exactly
the trade-off the emulator surfaces in seconds.

### 3. "Will it fit?" — the very first question

70B FP16 on A100-80GB at batch 1, vLLM:

```bash
$ python scripts/cli.py --model 70 --bits 16 --hw 1xA100 --engine vllm --batch 1
  memory : 140.8 GB  !! exceeds 80 GB
```

Not even close. Decision tree the emulator collapses for you:

| Path | Memory at bs=1 | Speedup vs A100 fp16 |
|---|---:|---|
| 2× A100-80GB tensor-parallel | ~70 GB / GPU | works, ~half t/s |
| 1× A100 + GPTQ.4bit | 35.8 GB | works, ~3× decode |
| 1× H100-141GB | 140.8 GB | works, ~2× decode |

You can plug each branch into `cli.py` and decide before requesting quota.

### 4. Capacity planning: tokens/sec → replicas

Production SLO: **1000 RPS, average 100 tokens response, p50 latency < 5 s**.
Throughput required = 1000 × 100 = **100,000 tok/s**.

```bash
$ python scripts/cli.py --model 7 --bits 16 --hw 1xA100 --engine vllm --batch 8
  throughput : 4322 tok/s
```

So you need ~ ⌈100 000 / 4322⌉ = **24 A100 replicas** for the steady-state
throughput, plus headroom for traffic spikes. At ~$3/hr on AWS, that's
~$1700/day for inference compute — the kind of number that should be
known *before* the launch meeting, not after.

### 5. Cost per million tokens

Pass your hourly GPU price via `--cost-per-hour` and the CLI prints
`$ per million tokens` directly:

```bash
$ python scripts/cli.py --model 7 --bits 16 --hw 1xA100 --engine vllm \
        --batch 8 --cost-per-hour 4.10
  throughput       : 4322 tok/s
  cost @ $4.1/hr   : $0.264 per million tokens
```

Comparing across hardware:

```
# decode-only, single-stream, 7B FP16 vLLM
                    tok/s     $/hr      $/M tokens
1xT4   (g4dn.xl)    636      0.526         0.23
1xA10  (g5.xl)     1272      1.006         0.22
1xA100 (1/8 p4d)   4322      4.10          0.26
32vCPU-C7i (8xl)     30      1.428        13.22  ← 50× more expensive
```

Useful when fixing budget: if you charge $2/M output tokens, your inference
margin per token on GPU is ~85%, on CPU you'd be losing money.

### 6. CI/CD pre-deployment gate

The emulator is a Python function — it slots straight into a pipeline:

```python
# scripts/preflight_check.py (sketch — adapt to your model registry)
from emulator import predict
from emulator.hardware import get_peak_compute, get_memory_bandwidth

SLO_FIRST_TOKEN_MS = 500
SLO_TOK_PER_S      = 30
TARGET_HW          = "1xA10"

def check(n_params_b, bits, engine_alpha, engine_beta):
    res = predict(
        n_params_b=n_params_b, bits=bits,
        p_in=512, p_out=128, batch=4,
        peak_flops=get_peak_compute(TARGET_HW, bits),
        mem_bw=get_memory_bandwidth(TARGET_HW),
        alpha=engine_alpha, beta=engine_beta,
    )
    assert res.prefill_s * 1000 < SLO_FIRST_TOKEN_MS, "prefill too slow"
    assert res.throughput_tok_s > SLO_TOK_PER_S, "throughput below SLO"
    assert res.memory_gb < 24, "won't fit on A10"
```

Run this in a GitHub Action on every PR that touches the model spec. PRs
that violate the SLO never reach staging.

### 7. Build intuition

Even if you never ship a single prediction, working with the formula
crystallises three things that aren't obvious:

- **Decode is memory-bound**, not compute-bound. That's why bigger
  *batches* (not bigger GPUs) buy throughput on H100. It's also why a
  cheap A10 can match an A100 for single-stream decode.
- **Quantization speeds up decode but not prefill** (and only when the
  kernel actually exploits the smaller weights — see §2).
- **Long contexts kill you through KV cache**, not through prefill
  compute. A 32k-context request runs the same FLOPs as 32× of 1k-context
  requests for prefill, but the KV term in decode keeps growing.

### What this is *not*

- **Not a substitute for actual benchmarking** when you're inside the last
  20% of optimisation.
- **Not architecture-aware**: MoE, multi-query / grouped-query attention,
  sliding-window — all need explicit terms the formula currently doesn't
  carry.
- **Not multi-GPU**: tensor-parallel and pipeline-parallel splits introduce
  PCIe / NVLink traffic that's outside the model.
- **Not a regression suite**: 50–80% median error on individual rows.
  Use it for *order-of-magnitude* answers and *relative comparisons*, not
  absolute SLA compliance.

## Importing llama-bench results

[llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
is the official benchmarking tool for `llama.cpp`. It outputs CSV with one
row per *(model, config, test)* — where `test` is either prompt-processing
(`pp{N}`), token-generation (`tg{N}`), or end-to-end (`pg{P,G}`).

```bash
# 1) on a real machine
llama-bench -m model.gguf -p 512 -n 128 -ctk q8_0 -ctv q8_0 -o csv > runs.csv

# 2) ingest into our schema (joins matching pp+tg rows by model+config)
python scripts/import_llama_bench.py runs.csv \
       --hw 1xT4 --backend llama.cpp --output data/llama_bench.csv

# 3) re-run calibration so llama.cpp gets calibrated alongside PyTorch
python scripts/run_calibration.py
```

The importer pulls **effective bits-per-weight from the actual on-disk size**
(`8 · model_size / model_n_params`) instead of guessing from a label —
Q4_K_M is 4.91 bpw, Q5_K_M is ~5.5 bpw, Q3_K_S is ~3.5 bpw, etc. This
matters: a "4-bit" GGUF can vary by 25% depending on which K-quant was
used, and the calibration only works if we feed the right `W` into the
roofline.

A small sample drawn from llama-bench's own README is included at
`data/llama_bench_sample.csv` — it covers Qwen-2.5-7B-Q4_K_M and Llama-7B/13B
on a CUDA RTX 4080, plus CPU runs at 8 / 32 threads. Run the importer
against it to see the expected output shape.

## Quantized KV cache

`predict()` now accepts `kv_bits_k` and `kv_bits_v` (each defaulting to 16).
Drop them to model `--cache-type-k q8_0 --cache-type-v q8_0` (halving KV
bandwidth) or `q4_0` (quartering). For long-context decode the KV term
dominates, so this can be a 2× speedup on its own:

```bash
python scripts/cli.py --model 7 --bits 16 --hw 1xA100 --engine vllm \
                      --p-in 8192 --p-out 512 --kv-bits-k 8 --kv-bits-v 8
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

Полный каталог всех использованных источников — данные для калибровки,
внешние бенчмарки для валидации, источники архитектур в `ARCH_DEFAULTS`,
методологические ссылки — собран в [`docs/BENCHMARK_SOURCES.md`](docs/BENCHMARK_SOURCES.md).

Краткая сводка:

- **Calibration data**: [`optimum-benchmark/llm-perf-leaderboard`](https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard)
