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

### 1.4. vLLM bench (собственные замеры на 1× A100-SXM4-40GB)

**Источник:** `vllm bench throughput` на Vast.ai (A100-SXM4-40GB);
logs в `data/raw_vllm/A100/`. Pinned instance bandwidth: 1314.8 GB/s (DLPerf).

**Покрытие:**
- Qwen-7B AWQ.4bit, batch sweep `(1, 8, 32, 100, 200, 500)`, `P_in=1024, P_out=128`.
- Llama-3.1-70B GPTQ.4bit, batch sweep `(1, 2, 4, 8, 16)`, `P_in=256, P_out=64`,
  `--enforce-eager` (forced due to 40GB VRAM limit — leaves only 0.55 GB for KV).

**Что калибровано:**

| precision_label | α | β | batch_saturation | notes |
|---|---:|---:|---|---|
| `AWQ.4bit` (Qwen-7B) | 0.453 | 0.44 | (7.8, 6.8) | clean fit, +/-16% output throughput |
| `GPTQ.4bit` (Llama-70B) | 0.221 | 0.67 | (3.8, 1.8) | **40GB artefact**: eager penalty + preemption |

### 1.5. vLLM bench (собственные замеры на 1× A100-SXM4-80GB)

**Источник:** `vllm bench throughput` на Vast.ai (A100-SXM4-80GB);
logs в `data/raw_vllm/A100-80/`.

**Покрытие:**
- Qwen-2.5-72B AWQ.4bit, batch sweep `(1, 4, 16, 32, 64)`.
- Qwen3-32B AWQ.4bit, batch sweep `(1, 8, 32, 100, 200)`.
- Qwen-7B AWQ.4bit, beta-isolation run.

**Что калибровано:**
- `alpha = 0.537` (для 72B-класса);
- `beta = 0.56` (на базе DLPerf bandwidth 1314.8 GB/s);
- `batch_saturation = (8.2, 5.5)`.

### 1.6. Сводка калиброванных коэффициентов

| Hardware × Engine × Precision | α | β | n | Источник |
|---|---:|---:|---:|---|
| 1xA10 / pytorch / Unquantized | 0.26 | 0.45 | 270 | §1.1 |
| 1xA100 / pytorch / Unquantized | 0.25 | 0.20 | 79 | §1.1 |
| 1xA100-40 / vllm / AWQ.4bit | 0.45 | 0.44 | 6 | §1.4 (own); sat=(7.8, 6.8) |
| 1xA100-80 / vllm / AWQ.4bit | 0.54 | 0.56 | 5 | §1.5 (own); sat=(8.2, 5.5) |
| 1xA100-40 / vllm / GPTQ.4bit.MoE | 0.35 | 0.47 | 5 | §1.5 (own); sat=(5.8, 12.2) |
| 2xRTX-5090 / vllm / AWQ.4bit | 0.47 | 0.65 | 15 | §1.3 (own) |

Полная таблица с p25/p75 — `results/calibrated_coefficients.csv`. Анализ
расхождений predicted vs observed — `results/REPORT.md`.

## 2. Cross-Check: Эмулятор vs Публичные бенчмарки

| # | HW | Engine | Сценарий | Real | Emulator | Δ |
|---|---|---|---|---:|---:|---:|
| 2.1 | 1× A100 | TRT-LLM INT8 | Mixtral 8x7B, P_in=512, P_out=128, b=1 | decode 11.4 ms/тkn | 11.1 ms (калибр.) | −2.6% ✓ |
| 2.2a | 1× A100 | vLLM 0.9.1 AWQ | Qwen2.5-7B, P_in=1, P_out=2048, b=1 | 148 tok/s output | 138 tok/s | −6.7% ✓ |
| 2.3 | 1× RTX-5090 | vLLM AWQ | Qwen2.5-7B, P_in=1024, P_out=128, b=200 | 13 954 tok/s total | 13 827 tok/s total | **−0.9% ✓✓** |
| 2.5 | 1× A100 | llama.cpp IQ2_M | Llama 4 Scout (109B), P_in=1024, b=1 | prefill 798.6 tok/s | 794.2 tok/s | **−0.5% ✓✓** |

---
[... rest of the document ...]
