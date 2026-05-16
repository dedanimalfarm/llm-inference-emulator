# LLM Inference Emulator

Roofline-формула для prefill / decode latency и throughput LLM-инференса.
Один вопрос — один ответ за миллисекунды, без аренды GPU.

> *«Поместится ли Llama-70B Q4 на 2× RTX-3090? Сколько tok/s?»*
> *«Что выгоднее на 7B-чатбот: A10 или A100?»*
> *«Хватит ли throughput у одной H100 на наш SLO?»*

Откалиброван против реальных бенчмарков (LLM-Perf Leaderboard на A10/A100/T4/c7i,
наши собственные llama-bench на RTX-3090 и vLLM на 2× RTX-5090).

---

## Быстрый старт

### Веб-UI (рекомендую) — один контейнер, http://localhost:8501

```bash
docker compose up -d --build
```

Подробности — [`QUICKSTART.md`](QUICKSTART.md), описание вкладок —
[`ui/README.md`](ui/README.md).

### CLI

```bash
pip install -r requirements.txt

# Один прогон
python scripts/cli.py --model 7 --bits 4 --hw 1xA100 --engine vllm \
                     --batch 8 --precision-label AWQ.4bit

# Сетка для сравнения
python scripts/demo.py
```

---

## Что внутри

**Формула** (`emulator/formula.py::predict`) — roofline-модель: каждая фаза
ограничена `max(compute, memory)`. Поддерживает GQA/MLA-внимание, MoE
(active vs total params), sliding window, paged KV, speculative decoding,
multi-GPU TP. Полный вывод — [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).

**Калибровка** (`emulator/calibrate.py`) — обратное решение формулы на
каждой бенчмарк-точке даёт `(α, β)`, агрегация по `(HW, engine, precision)`
даёт `results/calibrated_coefficients.csv`. UI и CLI подхватывают этот
файл автоматически.

---

## Точность

| Где | Δ от реального |
|---|---|
| Внутри откалиброванной области (RTX-3090, A100-40, A10, T4, 2×5090) | **±10–20%** |
| Self-consistency (vLLM AWQ Qwen-7B на 2×5090, b=200) | **−0.9%** |
| Extrapolation на новое железо (H100 без калибровки) | до −56% |

6 cross-check точек против внешних бенчмарков (Baseten, Qwen, Morphllm) —
вкладка «🧪 Валидация» или [`docs/BENCHMARK_SOURCES.md §2`](docs/BENCHMARK_SOURCES.md).

---

## Документация

Порядок чтения «с нуля»:

1. [`docs/PRIMER.md`](docs/PRIMER.md) — учебник: физика LLM-инференса,
   5 частей, ~2900 строк. Если впервые читаешь про prefill/decode/roofline.
2. [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — пошаговый разбор формулы
   на кейсе Qwen-2.5-32B Q4_K_M на 2× RTX-3090.
3. [`docs/USE_CASES.md`](docs/USE_CASES.md) — 7 конкретных decision'ов,
   которые эмулятор закрывает (выбор HW, квантизация, capacity planning,
   $/M tokens, CI-gate).
4. [`docs/VLLM_OPTIMIZATIONS.md`](docs/VLLM_OPTIMIZATIONS.md) — design doc
   по vLLM-расширениям (PagedAttention, batch_sat, MoE-memory fix).
5. [`docs/BENCHMARK_SOURCES.md`](docs/BENCHMARK_SOURCES.md) — каталог всех
   источников данных (калибровка / валидация / архитектуры / методология).
6. [`docs/MANUAL_BENCHMARK.md`](docs/MANUAL_BENCHMARK.md) — как откалибровать
   собственный движок.
7. [`docs/A100_CALIBRATION_PLAN.md`](docs/A100_CALIBRATION_PLAN.md) —
   playbook следующей калибровочной сессии (Vast.ai NVIDIA CUDA template).
8. [`docs/FINOPS.md`](docs/FINOPS.md) — экономика LLM-инференса:
   API vs Cloud vs On-Prem, MoE/SW/spec ROI, TCO.

---

## Структура репо

```
emulator/      hardware.py, engines.py, formula.py, calibrate.py
ui/            Streamlit-приложение (Docker-ready)
scripts/       cli.py, demo.py, run_calibration.py, compare.py, importer'ы
data/          бенчмарк-снимки (LLM-Perf, llama-bench, vLLM)
results/       calibrated_coefficients.csv, prediction_vs_actual.csv, REPORT.md
docs/          PRIMER, WALKTHROUGH, BENCHMARK_SOURCES, USE_CASES, FINOPS, ...
tests/         26 unit-тестов (всё зелёное)
```

---

## Расширить

- Новое железо → `emulator/hardware.py::HARDWARE_SPECS`
- Новый движок → `emulator/engines.py::ENGINE_DEFAULTS` (опционально:
  откалибровать через `calibrate_row()`)
- Новая модель → `emulator/formula.py::ARCH_DEFAULTS` (layers/d_model/kv_heads,
  опционально n_active_b для MoE, sliding_window для Mistral-like)
- Новая валидационная точка → `ui/validation_data.py::VALIDATION_POINTS`

---

## Источники данных

Полный каталог — [`docs/BENCHMARK_SOURCES.md`](docs/BENCHMARK_SOURCES.md). Кратко:

- **Калибровка**: `optimum-benchmark/llm-perf-leaderboard` (PyTorch на
  A10/A100/T4/c7i) + наш llama-bench на RTX-3090 + наш vLLM на 2× RTX-5090.
- **Валидация**: Baseten (Mixtral / TRT-LLM / A100), Qwen Speed Benchmark
  (vLLM / A100), Morphllm (vLLM / H100).
- **Архитектуры**: HF model cards, `config.json` verbatim для DeepSeek-V3/V4.
- **Методология**: Williams (roofline), Kwon (vLLM/PagedAttention),
  Leviathan (speculative), Dao (FlashAttention), Vaswani (transformer).

Снимок Leaderboard для воспроизводимости —
[`dedanimalfarm/llm-perf-leaderboard-snapshot`](https://github.com/dedanimalfarm/llm-perf-leaderboard-snapshot).
