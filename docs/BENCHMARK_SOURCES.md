# Каталог источников данных и бенчмарков

Перечень всех внешних источников, на которые опирается этот эмулятор:
данные для калибровки коэффициентов, бенчмарки для валидации прогнозов,
архитектурные характеристики моделей в `ARCH_DEFAULTS`, и методологические
ссылки на статьи / блоги. Цель — single source of truth по тому, *откуда*
взято каждое число в коде.

## 1. Данные для калибровки (`α`, `β`, `batch_saturation`)

Калиброванные значения в `results/calibrated_coefficients.csv` получены
фитом против реальных замеров из четырёх независимых источников.

### 1.1. LLM-Perf Leaderboard (Hugging Face)

**Источник:** [`optimum-benchmark/llm-perf-leaderboard`](https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard)
([Spaces UI](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)).

**Что:** систематические замеры PyTorch (eager / SDPA), AWQ-4bit,
BnB-4bit/8bit, GPTQ-4bit на нескольких NVIDIA-GPU и одной CPU-машине.
Сценарий зафиксирован: `bs=1, P_in=256, P_out=64`.

**Покрытие в наших данных:**
- `data/leaderboard-1xA10.csv` — 1× A10 (g5.xl)
- `data/leaderboard-1xA100.csv` — 1× A100 (1/8 p4d)
- `data/leaderboard-1xT4.csv` — 1× T4 (g4dn.xl)
- `data/leaderboard-32vCPU-C7i.csv` — 32-vCPU C7i

**Снимок (для воспроизводимости):** [`dedanimalfarm/llm-perf-leaderboard-snapshot`](https://github.com/dedanimalfarm/llm-perf-leaderboard-snapshot).

**Что калибровано:** строки PyTorch / PyTorch+SDPA для A10/A100/T4/C7i в
`results/calibrated_coefficients.csv` (≈40-450 строк на (hw, quant) bucket).

### 1.2. llama.cpp llama-bench (собственные замеры)

**Источник:** [`llama.cpp`'s `llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench),
официальный benchmarking tool ggerganov'а. Выдаёт CSV с `tg128 / pp512`
(throughput decode / prefill для фиксированных длин).

**Покрытие в наших данных:**
- `data/llama_bench_rtx3090.csv` — RTX-3090, Qwen-2.5-7B Q4_K_M / Q5_K_M /
  Q6_K + симметричные KV-quants (Q8_0, Q4_0, F16).
- `data/llama_bench_llama8b.csv` — RTX-3090, Llama-3.1-8B Q4_K_M.
- `data/llama_bench_qwen32b_multi.csv` — 2× RTX-3090, Qwen-2.5-32B Q4_K_M
  с разбивкой по `row-split` и `layer-split`.
- `data/llama_bench_llama70b_multi.csv` — 2× RTX-3090, Llama-3.1-70B Q4_K_M.

**Setup:** см. `docs/MANUAL_BENCHMARK.md` для точных команд воспроизведения.

**Что калибровано:** `engines.py["llama.cpp"]: alpha=0.36, beta=0.72`,
плюс multi-GPU инвариант `tp_size × tp_efficiency = 1.0` для PCIe без NVLink.

### 1.3. vLLM bench (собственные замеры на 2× RTX-5090)

**Источник:** официальный [`vllm bench throughput`](https://docs.vllm.ai/en/latest/getting_started/benchmark.html)
из vLLM CLI; logs в `data/raw_vllm/logs/`.

**Покрытие:**
- Qwen-7B AWQ.4bit на 1× и 2× RTX-5090, multi-batch sweep
  `(batch=50, 100, 200, 500, 1000)`, контексты от 4K до 65K.
- Qwen-32B AWQ.4bit на 2× RTX-5090, batch sweep.
- Llama-3.1-70B AWQ.4bit на 2× RTX-5090, batch sweep (с фиксированной
  и нефиксированной длиной).
- Дополнительные конфигурации: KV-FP8, APC on/off, eager mode.

**Что калибровано:**
`engines.py["vllm"]: alpha=0.47, batch_saturation=(45, 7)`.
`beta=0.65` оставлен как literature default — из текущих vLLM-замеров
не идентифицируется (см. commit `82341a6`, Variant A).

### 1.4. vLLM bench (собственные замеры на 1× A100-SXM4)

**Источник:** `vllm bench throughput` на Vast.ai (A100-SXM4-40GB);
logs в `data/raw_vllm/A100/`.

**Покрытие:**
- Qwen-7B AWQ.4bit, batch sweep `(1, 8, 32, 100, 200, 500)`,
  `P_in=1024, P_out=128`.

**Что калибровано:**
- `alpha = 0.453` (близко к RTX-5090);
- `beta = 0.316` (значительно ниже literature 0.65, откалибровано по
  single-stream throughput);
- `batch_saturation = (7.8, 6.8)` (переход к Hill-кривой, стартующей с
  `bs_eff(1)=1.0`).

### 1.5. Сводка калиброванных коэффициентов

| Hardware × Engine × Precision | α | β | n | Источник |
|---|---:|---:|---:|---|
| 1xA10 / pytorch / Unquantized | 0.26 | 0.45 | 270 | §1.1 |
| 1xA100 / pytorch / Unquantized | 0.25 | 0.20 | 79 | §1.1 |
| 1xA100 / vllm / AWQ.4bit | 0.45 | 0.32 | 6 | §1.4 (own) |
| 1xT4 / pytorch / Unquantized | 0.04 | 0.47 | 142 | §1.1 |
| 32vCPU-C7i / pytorch / Unquantized | 0.19 | 0.37 | 200 | §1.1 |
| 32vCPU-C7i / onnxruntime / Unquantized | 0.17 | 0.28 | 12 | §1.1 |
| RTX-3090 / llama.cpp / Q4_K (4.91bpw) | 0.36 | 0.72 | 11 | §1.2 |
| 2xRTX-3090 / llama.cpp / Q4_K (4.85bpw) | 0.58 | 0.83 | 2 | §1.2 |
| 2xRTX-5090 / vllm / AWQ.4bit | 0.47 | 0.65 | 15 | §1.3 (own) |

Полная таблица с p25/p75 — `results/calibrated_coefficients.csv`. Анализ
расхождений predicted vs observed — `results/REPORT.md`.

## 2. Cross-Check: Эмулятор vs Публичные бенчмарки

В отличие от данных §1, эти числа **не использовались** для калибровки;
вместо этого мы сверяем прогнозы эмулятора с ними, чтобы померить точность.

**Сводка валидационных прогонов:**

| # | HW | Engine | Сценарий | Real | Emulator | Δ |
|---|---|---|---|---:|---:|---:|
| 2.1 | 1× A100 | TRT-LLM INT8 | Mixtral 8x7B, P_in=512, P_out=128, b=1 | decode 11.4 ms/тkn | 7.07 ms (после фикса) | −38% ✓ |
| 2.2a | 1× A100 | vLLM 0.9.1 AWQ | Qwen2.5-7B, P_in=1, P_out=2048, b=1 | 148 tok/s output | 138 tok/s | −6.7% ✓ |
| 2.2b | 1× A100 | vLLM 0.9.1 AWQ | Qwen2.5-7B, P_in=6144, P_out=2048, b=1 | 138 tok/s output | 126 tok/s | −9.0% ✓ |
| 2.3 | 1× RTX-5090 | vLLM AWQ | Qwen2.5-7B, P_in=1024, P_out=128, b=200 | 13 954 tok/s total | 13 827 tok/s total | **−0.9% ✓✓** |
| 2.4a | 1× H100 | vLLM BF16 | Llama-3.1-8B, P_in=256, P_out=256, b=1 | TTFT 72 ms (mean) | TTFT 49.6 ms | −31% ✓ |
| 2.4b | 1× H100 | vLLM BF16 (high concurrency) | Llama-3.1-8B, server stress | 12 500 tok/s total | ≤5 503 tok/s (b=64) | −56% ✗ |

**Self-consistency** (§2.3) показывает ~1% точности **внутри откалиброванной
области**. После калибровки A100 (§2.2), точность на этом железе также
поднялась до <10% ошибки. Основной calibration gap остаётся на H100 (§2.4),
где `batch_saturation` существенно выше.

### 2.1. Baseten — Mixtral 8x7B / TensorRT-LLM / A100

**Источник:** [Baseten blog, «Faster Mixtral inference with TensorRT-LLM
and quantization»](https://www.baseten.co/blog/faster-mixtral-inference-with-tensorrt-llm-and-quantization/)
(2024).

**Сценарий:** Mixtral 8x7B, TensorRT-LLM, 1× A100, INT8 (weights-only),
`P_in=512, P_out=128, batch=1`.

**Заявленные числа:** TTFT 127 ms, perceived throughput 81 tok/s
(= decode ≈ 11.4 ms/токен).

**Результат сверки** (2026-05-14):
- Эмулятор TTFT: 103 ms (Δ −19% — в духе roofline-«верхней границы»).
- Эмулятор decode/token: 7.07 ms (Δ −38% после фикса MoE memory bug,
  commit `4fdaa87`; до фикса было +124%, переоценка в 2×).

**Что выявил прогон:** bug в `t_dec_mem` для MoE — формула читала весь
`W` вместо только активных экспертов. Фикс: `t_dec_mem = W · (N_active / N_total) / (eff_mbw · β)`.
Тест-инвариант: `tests/test_formula.py::test_moe_decode_memory_scales_with_active_fraction`.

Остающийся гэп −38% объясняется literature β=0.90 для TensorRT в
`ENGINE_DEFAULTS`. Реальное β для MoE-serving ближе к 0.55-0.60; это
calibration concern, не формула.

### 2.2. Qwen Speed Benchmark — A100 / vLLM

**Источник:** [Qwen2.5 Speed Benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html).

**Сценарий:** Qwen2.5-7B-Instruct-AWQ на 1× A100-80GB, vLLM 0.6.3 (valid for 0.9.1),
FlashAttention 2.6.3, single-stream (batch=1), output фиксирован 2048
токенов, input варьируется.

**Заявленные числа (output tokens / total time, включая prefill):**

| $P_{\text{in}}$ | Real (tok/s) | Эмулятор (калиброванный, tok/s) | Δ |
|---:|---:|---:|---:|
| 1 | 148.10 | 138.1 | −6.7% |
| 6144 | 137.64 | 125.6 | −9.0% |

**Результаты сверки** (2026-05-14):

После калибровки по собственным замерам (§1.4), точность предсказаний для A100
существенно выросла (с +50% ошибки до <10%). Ключевые изменения:
- **β = 0.316** (вместо literature 0.65) — vLLM на Ampere SXM показывает меньший
  MBU при типичных нагрузках.
- **`batch_saturation = (7.8, 6.8)`** — переход к модели, где `bs_eff(1) = 1.0`,
  устранил ложное ускорение на малых батчах.

### 2.3. Self-consistency — наш собственный замер vLLM на 2× RTX-5090

**Источник:** `data/raw_vllm/logs/qwen7b_tp1_b200.log` (наш собственный
прогон 2026-05-09).

**Сценарий:** Qwen2.5-7B-Instruct-AWQ на 1× RTX-5090 (TP=1), vLLM
v0.20.1 (V1 engine), FlashAttention 2, AWQ Marlin kernel, chunked
prefill, APC, P_in=1024, P_out=128, batch=200 параллельных запросов.

**Заявленные числа (наш бенчмарк):**

| Метрика | Значение |
|---|---:|
| Total throughput | 13 953.87 tokens/s |
| Output throughput | 1 550.43 tokens/s |
| Request throughput | 12.11 req/s |
| Время прогона 200 req | ~16.5 с |

**Результаты сверки** (2026-05-14):

| Метрика | Real | Эмулятор | Δ |
|---|---:|---:|---:|
| Total throughput | 13 953.87 tok/s | 13 826.8 tok/s | **−0.9%** ✓✓ |
| Время прогона 200 req | 16.5 с | 16.67 с | +1.0% ✓ |

Это **внутренняя точка** — α=0.47 и `batch_saturation=(45, 7)` именно
на этом логе (и подобных) откалиброваны через
`scripts/fit_vllm_calibration.py`. Эмулятор воспроизводит свой
обучающий датасет с точностью ~1%, что и должен делать корректно
сфиченный фит.

Команда воспроизведения:
```bash
python3 scripts/cli.py --model 7 --bits 4 --hw 1xRTX-5090 --engine vllm \
    --p-in 1024 --p-out 128 --batch 200 --precision-label "AWQ.4bit"
```

### 2.4. Morphllm — Llama-3.1-8B / vLLM / 1× H100

**Источник:** [Morphllm vLLM benchmarks page](https://www.morphllm.com/vllm-benchmarks).

**Заявленные числа:**
- Llama-3.1-8B BF16, 1× H100 80GB, vLLM + FlashInfer, gpu_memory_utilization=0.8.
- TTFT mean: **72 ms** (low concurrency), P99: **79 ms**.
- Total throughput: **~12 500 tok/s** (server stress, high concurrency).

**Результаты сверки** (2026-05-14):

Два аспекта проверены отдельно — TTFT при batch=1 и общий throughput
при server-стрессе.

| Метрика | Real | Emulator | Δ | Комментарий |
|---|---:|---:|---:|---|
| TTFT (batch=1) | 72 ms | 49.6 ms | −31% | roofline-undershoot, ожидаемо |
| Throughput @ b=8 | — | 3 256 tok/s | — | для chat-low concurrency |
| Throughput @ b=64 | — | 5 503 tok/s | — | плато `batch_saturation=(45,7)` |
| **Throughput high-c** | **12 500 tok/s** | **≤ 5 503** | **−56%** | calibration gap |

**Диагноз gap'а:** `ENGINE_DEFAULTS["vllm"].batch_saturation = (45, 7)`
калиброван на 2× RTX-5090 / Qwen-7B AWQ. На H100 с FlashInfer и более
эффективным scheduler'ом реальный `batch_max` существенно выше — судя
по Morphllm 12.5K tok/s, ~90+. Чтобы подогнать прогноз:

- Текущее: `batch_saturation=(45, 7)` → max throughput 45 / t_dec
- Нужно: `(92, 5)` → ~12 500 tok/s при той же t_dec ≈ 7.4 ms

Эта точка валидирует **необходимость per-hardware калибровки
`batch_saturation`**, а не только α/β. Похоже что для каждой пары
`(hw, engine)` плато continuous batching различается заметно.

Команды воспроизведения:
```bash
# TTFT comparison
python3 scripts/cli.py --model 8 --bits 16 --hw 1xH100 --engine vllm \
    --p-in 256 --p-out 256 --batch 1

# Throughput plateau
python3 scripts/cli.py --model 8 --bits 16 --hw 1xH100 --engine vllm \
    --p-in 256 --p-out 256 --batch 64
```

## 3. Источники архитектур (`ARCH_DEFAULTS`)

Все записи в `emulator/formula.py::ARCH_DEFAULTS` взяты из публичных
конфигов соответствующих моделей.

### 3.1. Dense LLMs

| Модель | `L` | `d_model` | `H_kv` | Источник |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B/3B/7B/14B | 28/36/28/48 | 1536/2048/3584/5120 | 2/2/4/8 | [Qwen2.5 model cards](https://huggingface.co/Qwen) |
| Llama-3.1-8B | 32 | 4096 | 8 | [Meta Llama-3.1 paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) |
| Llama-2-13B | 40 | 5120 | 40 (MHA) | [Llama-2 paper](https://arxiv.org/abs/2307.09288) |
| Yi-34B | 48 | 7168 | 8 | [01-ai/Yi-34B card](https://huggingface.co/01-ai/Yi-34B) |
| Llama-3-70B | 80 | 8192 | 8 | Llama-3 paper |

### 3.2. MoE LLMs

| Модель | `N_total` / `N_active` | `L` | `d_model` | `H_kv` | `head_dim` | `sliding_window` | Источник |
|---|---|---:|---:|---:|---:|---:|---|
| Mixtral 8x7B | 46.7B / 12.9B | 32 | 4096 | 8 | 128 | — | [`mistralai/Mixtral-8x7B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) config |
| Mixtral 8x22B | 141B / 39B | 56 | 6144 | 8 | 128 | — | [`mistralai/Mixtral-8x22B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) config |
| Qwen3-235B-A22B | 235B / 22B | 94 | 4096 | 4 | 128 | — | [`Qwen/Qwen3-235B-A22B`](https://huggingface.co/Qwen/Qwen3-235B-A22B) config |
| DeepSeek-V3 | 671B / 37B | 61 | 7168 | 1 (MLA) | 128 | — | [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437) + config.json |
| DeepSeek-V4-Flash | 284B / 13B | 43 | 4096 | 1 (MLA) | 512 | 128 | [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json) (verbatim) |
| DeepSeek-V4-Pro | 1.6T / 49B | 61 | 7168 | 1 (MLA) | 512 | 128 | [`deepseek-ai/DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json) (verbatim) |

DeepSeek-V4 архитектура verified pulling `config.json` напрямую с
Hugging Face — числа в `ARCH_DEFAULTS` не приближены, а взяты буквально.

## 4. Hardware-спецификации (`HARDWARE_SPECS`)

`emulator/hardware.py::HARDWARE_SPECS` опирается на официальные
production-datasheets:

| GPU | Источник peak FLOPS / bandwidth |
|---|---|
| T4 | [NVIDIA T4 product brief](https://www.nvidia.com/en-us/data-center/tesla-t4/) |
| RTX-3090 | [NVIDIA RTX 3090 specs](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) |
| RTX-5090 | [NVIDIA RTX 5090 product page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) (Blackwell) |
| A10 | [NVIDIA A10 datasheet](https://www.nvidia.com/en-us/data-center/a10-gpu/) |
| A100 | [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) — SXM4-40GB (1.55 TB/s) и PCIe-80GB (2.04 TB/s) |
| H100 | [NVIDIA H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/) — SXM5 dense peaks (989 TFLOPS BF16, 1979 TFLOPS FP8, 3.35 TB/s) |

`tp_efficiency` для multi-GPU конфигураций — эмпирически из §1.2 (для
2× RTX-3090 PCIe без NVLink: `0.50`).

## 5. Методологические источники

Научные статьи и блоги, на которые опираются формулы и калибровочные
методы эмулятора.

### Roofline-модель и физика inference

- **Williams, Waterman, Patterson.** «Roofline: An Insightful Visual
  Performance Model». CACM 2009 ([DOI](https://dl.acm.org/doi/10.1145/1498765.1498785)).
  — каноническая формулировка roofline.
- **Rush, Sasha.** «The LLM Scaling Book». 2024 ([online](https://github.com/srush/scaling-book)).
  — современный учебник по физике LLM-инференса.

### Архитектура трансформера

- **Vaswani et al.** «Attention is All You Need». NeurIPS 2017 ([arXiv](https://arxiv.org/abs/1706.03762)).
- **Ainslie et al.** «GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints». EMNLP 2023 ([arXiv](https://arxiv.org/abs/2305.13245)).
- **Shazeer, Noam.** «GLU Variants Improve Transformer». 2020 ([arXiv](https://arxiv.org/abs/2002.05202))
  — SwiGLU activation, используется в Llama / Mistral / Qwen / DeepSeek.
- **Su, Lu, Pan, Murtadha, Wen, Liu.** «RoFormer: Enhanced Transformer
  with Rotary Position Embedding». 2021 ([arXiv](https://arxiv.org/abs/2104.09864))
  — RoPE.
- **Ba, Kiros, Hinton.** «Layer Normalization». 2016 ([arXiv](https://arxiv.org/abs/1607.06450)).
- **He, Zhang, Ren, Sun.** «Deep Residual Learning for Image Recognition».
  CVPR 2016 ([arXiv](https://arxiv.org/abs/1512.03385)) — residual connections.

### Сервинг и оптимизации

- **Kwon et al.** «Efficient Memory Management for Large Language Model
  Serving with PagedAttention». SOSP 2023 ([arXiv](https://arxiv.org/abs/2309.06180))
  — vLLM paper.
- **Leviathan, Kalman, Matias.** «Fast Inference from Transformers via
  Speculative Decoding». ICML 2023 ([arXiv](https://arxiv.org/abs/2211.17192)).
- **Dao, Fu, Ermon, Rudra, Ré.** «FlashAttention: Fast and Memory-Efficient
  Exact Attention with IO-Awareness». NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2205.14135)).
- **Shah, Bikshandi, Zhang, Thakkar, Tri Dao.** «FlashAttention-3: Fast
  and Accurate Attention with Asynchrony and Low-precision». 2024
  ([arXiv](https://arxiv.org/abs/2407.08608)).
- **Anyscale blog.** «How Continuous Batching Enables 23× Throughput in
  LLM Inference While Reducing p50 Latency». 2023
  ([link](https://www.anyscale.com/blog/continuous-batching-llm-inference)).
- **vLLM blog.** «Introducing Automatic Prefix Caching». 2024
  ([link](https://blog.vllm.ai/2024/01/27/apc.html)).

### Квантизация

- **Frantar, Ashkboos, Hoefler, Alistarh.** «GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers». ICLR 2023
  ([arXiv](https://arxiv.org/abs/2210.17323)).
- **Lin, Tang, Tang, Yang, Wang, Han.** «AWQ: Activation-aware Weight
  Quantization for LLM Compression and Acceleration». MLSys 2024
  ([arXiv](https://arxiv.org/abs/2306.00978)).
- **Frantar, Castro, Chen, Hoefler, Alistarh.** «Marlin: Mixed-precision
  Auto-Regressive Parallel Inference on Large Language Models». 2024
  ([repo](https://github.com/IST-DASLab/marlin)) — INT4 GEMM kernel.

### Токенизация

- **Sennrich, Haddow, Birch.** «Neural Machine Translation of Rare Words
  with Subword Units». ACL 2016 ([arXiv](https://arxiv.org/abs/1508.07909))
  — BPE.

## 6. Внутренние инструменты

Скрипты для сбора и обработки бенчмарков (см. `docs/MANUAL_BENCHMARK.md`
для пошаговых инструкций):

| Скрипт | Назначение |
|---|---|
| `scripts/import_llama_bench.py` | конвертация llama-bench CSV в схему эмулятора |
| `scripts/import_vllm_bench.py` | конвертация `vllm bench` JSON в схему эмулятора |
| `scripts/run_calibration.py` | фит α/β из данных для (hw, engine, precision) buckets |
| `scripts/fit_vllm_calibration.py` | специализированный фит α / `batch_saturation` для vLLM multi-batch sweep |
| `scripts/compare.py` | сравнение predicted vs observed; вывод relative-error агрегатов |

## Как добавить новый источник

1. Положить сырые данные в `data/` (CSV) или `data/raw_*/logs/` (JSON/text).
2. Адаптировать `import_*.py` или написать новый импортёр, выводящий
   нашу схему `(hw, engine, precision, ...)`.
3. Запустить `python3 scripts/run_calibration.py` (для классических
   бенчмарков) или `fit_vllm_calibration.py` (для multi-batch).
4. Обновить `results/calibrated_coefficients.csv` коммитом.
5. Добавить запись в §1 этого документа (источник, URL, что калибровано).
