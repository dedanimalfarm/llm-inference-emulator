# План калибровки эмулятора на NVIDIA A100

Развёрнутый playbook для следующей сессии бенчмарков. Аналог
`docs/MANUAL_BENCHMARK.md` (RTX-5090, уже выполнен), но для A100 —
самой востребованной GPU в production-сервинге.

## Цели

Закрыть три **calibration gap'а**, выявленных в
[`docs/BENCHMARK_SOURCES.md`](BENCHMARK_SOURCES.md) §2:

| Gap | Проявление | Что закроет калибровка |
|---|---|---|
| **vLLM на A100 не калиброван** | §2.2: переоценка single-stream throughput на +50% (β=0.65 literature, реально ≈ 0.26) | прямой фит α, β из multi-batch sweep |
| **`batch_saturation` зависит от железа** | §2.4: на H100 plateau (~92, 5) вместо нашего (45, 7) | фит `batch_saturation` для A100 |
| **MoE на A100 не калиброван** | §2.1: после MoE memory fix всё ещё −38% от Baseten | калибровка Mixtral / 8x22B на A100 |

После калибровки прогнозы эмулятора для A100/vLLM должны попасть в
±15% от реальных бенчмарков (вместо текущих ±50%).

## Выбор шаблонов на cloud

**Важно**: на Vast.ai шаблон `vLLM` доступен только для серверного B200
(Blackwell). Для A100 vLLM устанавливается вручную из шаблона
`NVIDIA CUDA` — это стандартный workflow и занимает 5 минут
(`pip install vllm` + кэш моделей).

| Шаблон | Phase | Зачем |
|---|---|---|
| **NVIDIA CUDA** | **1, 2, (3, 4)** | bare Ubuntu + CUDA toolkit; ставим vLLM / TGI / SGLang / TRT-LLM вручную через pip / Docker |
| **HuggingFace TGI API** | 3 (опц.) | если предпочесть preinstalled TGI, есть готовый шаблон |
| **SGLang** | 3 (опц.) | если предпочесть preinstalled SGLang |
| **Llama.cpp** | опц. | cross-check нашей RTX-3090 калибровки на A100 |

Шаблоны, **не нужные** для калибровки inference:
- Fine-tuning (Axolotl, Unsloth Studio, Kohya's GUI, Flux Gym) — обучение, не inference
- Image / video / audio (ComfyUI, InvokeAI, SD WebUI, Wan2GP, Open-Sora,
  Voicebox, Whisper, ACE Step) — не LLM-задачи
- LLM UI-обёртки (Open WebUI, Oobabooga, Langflow, Ollama) — для интерактивного
  использования, не для bench
- Прочее (RAPIDS, TensorFlow CUDA, Hashcat, Pixel Streaming, Pinokio,
  Linux Desktop, Ubuntu VM, PyTorch NGC) — не наш сценарий

### Setup vLLM на bare CUDA (один раз в начале сессии)

```bash
# Шаблон NVIDIA CUDA приходит с CUDA toolkit, но без Python ML stack.
# Ставим:
pip install --upgrade pip
pip install "vllm>=0.9.0"

# Проверка:
python3 -c "import vllm; print(vllm.__version__)"
# Должно напечатать что-то вроде 0.9.x

# Зафиксировать версию для воспроизводимости:
pip show vllm | grep Version > ~/vllm_version.txt

# Кэш моделей на быстрый диск (workspace обычно ~200 GB SSD):
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME

# A100 — Ampere; vLLM V1 scheduler полностью стабилен.
# На Blackwell нужен был VLLM_USE_V1=0, здесь НЕ нужен.
```

Полный setup ~5 минут (vLLM 1.5 GB, скачивание моделей в отдельной
строчке ниже).

## Phase 1 — vLLM single-GPU calibration (priority #1)

**Цель:** закрыть main calibration gap «vLLM на A100». Получить
`α(A100, vllm, AWQ.4bit)`, `β(A100, vllm, AWQ.4bit)`,
`batch_saturation(A100)` для двух моделей разных размеров.

**Hardware:** 1× A100 80GB (PCIe или SXM — обе пойдут; SXM
предпочтительнее для согласованности с literature).

**Шаблон:** `NVIDIA CUDA` (bare). vLLM ставится вручную — см. раздел
«Setup vLLM на bare CUDA» выше.

```bash
# Дополнительно (помимо общего setup из раздела выше):
export OMP_NUM_THREADS=8   # предсказуемость на CPU-стороне
```

**Bench-команды** (батч-свип для каждой модели):

```bash
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct-AWQ"
    "meta-llama/Llama-3.1-8B-Instruct"   # BF16 baseline
)
BATCHES=(1 8 32 100 200 500)

mkdir -p ~/vllm_results/logs ~/vllm_results/json

for MODEL in "${MODELS[@]}"; do
    NAME=$(echo $MODEL | tr '/' '_')
    for B in "${BATCHES[@]}"; do
        TAG="${NAME}_b${B}"
        echo "=== $TAG ==="
        taskset -c 0-31 vllm bench throughput \
            --model "$MODEL" \
            --tensor-parallel-size 1 \
            --num-prompts $B \
            --input-len 1024 --output-len 128 \
            --dataset-name random \
            --output-json ~/vllm_results/json/${TAG}.json \
            2>&1 | tee ~/vllm_results/logs/${TAG}.log
    done
done
```

**Дополнительно — context sweep** (для KV-term / sliding window):

```bash
# Qwen-7B-AWQ, batch=50, P_in варьируется
for P_IN in 1024 4096 8192 16384 32768; do
    TAG="qwen7b_ctx_${P_IN}"
    taskset -c 0-31 vllm bench throughput \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --num-prompts 50 \
        --input-len $P_IN --output-len 128 \
        --dataset-name random \
        --output-json ~/vllm_results/json/${TAG}.json \
        2>&1 | tee ~/vllm_results/logs/${TAG}.log
done
```

**Стоимость:** ~$1.50/hr × 4-6 часов = **$6-9** (включая warm-up,
скачивание моделей).

**Output:** ~18 JSON-логов (2 модели × 6 batch × 1.5 запасной).
Качаем локально в `data/raw_vllm/A100/`.

## Phase 2 — vLLM multi-GPU + MoE (priority #2)

**Цель:** закрыть `tp_efficiency(A100)` и валидировать MoE memory fix
(commit `4fdaa87`) на independent железе.

**Hardware:** 2× A100 80GB (NVLink желательно — для real TP).

**Шаблон:** `NVIDIA CUDA` (тот же, что Phase 1; vLLM уже установлен).

**Bench-команды:**

```bash
# Mixtral 8x7B AWQ (MoE) — single A100 ещё держит (46.7 GB AWQ).
# На 1× A100 если влезает: TP=1; иначе TP=2.
for B in 1 8 32 100 200; do
    taskset -c 0-31 vllm bench throughput \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --quantization awq_marlin \
        --tensor-parallel-size 1 \
        --num-prompts $B \
        --input-len 1024 --output-len 128 \
        --dataset-name random \
        --output-json ~/vllm_results/json/mixtral8x7b_tp1_b${B}.json \
        2>&1 | tee ~/vllm_results/logs/mixtral8x7b_tp1_b${B}.log
done

# Llama-3.1-70B AWQ — точно требует TP=2 (40 GB AWQ + KV).
for B in 1 8 32 100; do
    taskset -c 0-31 vllm bench throughput \
        --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
        --tensor-parallel-size 2 \
        --num-prompts $B \
        --input-len 1024 --output-len 128 \
        --dataset-name random \
        --output-json ~/vllm_results/json/llama70b_tp2_b${B}.json \
        2>&1 | tee ~/vllm_results/logs/llama70b_tp2_b${B}.log
done
```

**Стоимость:** ~$3/hr (2× A100) × 4-6 часов = **$12-18**.

## Phase 3 — Cross-engine validation (опционально, для diff'а vLLM vs TGI/SGLang)

**Цель:** проверить, что наши калибровочные коэффициенты движков
(`α=0.47` для vLLM vs `α≈0.55` для TGI с FlashInfer) отражают реальность,
а не просто эффект железа.

**Шаблон:** `HuggingFace TGI API` ИЛИ `SGLang`.

**Bench-команды (TGI):**

```bash
# TGI запущен через docker, отдаёт OpenAI-compatible API на порту 8080.
# Используем тот же vllm bench, но через --backend openai-chat.
for B in 1 8 32 100; do
    vllm bench serve \
        --backend openai-chat \
        --base-url http://localhost:8080 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts $B \
        --random-input-len 1024 --random-output-len 128 \
        2>&1 | tee ~/tgi_results/llama8b_b${B}.log
done
```

**Output:** `data/raw_tgi/A100/`, отдельный bucket в калибровочной CSV.

**Стоимость:** ~$1.50/hr × 2-3 часа = **$3-5**.

## Phase 4 — TensorRT-LLM (опционально, для FP8 path)

**Цель:** калибровать `α/β` для TRT-LLM на A100/H100, заменив literature
`(0.55, 0.90)` — наш Baseten сверка (§2.1) показала overestimate на
−38% даже после MoE fix.

**Hardware:** A100 или H100. Для FP8 path обязательно **H100/H200**
(A100 имеет только INT8, не FP8).

**Шаблон:** `NVIDIA CUDA` (bare Ubuntu + CUDA toolkit). TRT-LLM нет
в стандартных шаблонах — нужен ручной install.

**Setup:**

```bash
# Установить TensorRT-LLM (из NGC container — самый надёжный путь)
docker pull nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3
# (или из исходников, см. https://github.com/NVIDIA/TensorRT-LLM)

# Build engine для модели (AOT compile — 30-60 минут per model)
trtllm-build --checkpoint_dir /path/to/llama-8b-fp8 \
             --output_dir ./engine_llama8b_fp8 \
             --gemm_plugin fp8 \
             --max_batch_size 256 \
             --max_input_len 4096 --max_output_len 1024
```

**Bench:**

```bash
# TRT-LLM имеет свой benchmark runner (gptManagerBenchmark)
for B in 1 8 32 100; do
    ./gptManagerBenchmark \
        --engine_dir ./engine_llama8b_fp8 \
        --request_rate -1 \
        --dataset workload_${B}.json \
        > ~/trtllm_results/llama8b_b${B}.log
done
```

**Стоимость:** ~$3-4/hr × 8-12 часов (с учётом compile time) = **$25-50**.

## Импорт и фит результатов

Локально, после возврата с cloud:

```bash
# 1. Скопировать сырые данные
scp -r vast:vllm_results/* data/raw_vllm/A100/

# 2. Импорт JSON → CSV в нашей схеме
python3 scripts/import_vllm_bench.py \
    data/raw_vllm/A100/json/*.json \
    --hw 1xA100 \
    --output data/leaderboard-1xA100-vllm.csv

# 3. Фит α/β/batch_saturation
python3 scripts/fit_vllm_calibration.py \
    --input data/leaderboard-1xA100-vllm.csv \
    --hw 1xA100 \
    --engine vllm
# Output: новые α, β, (batch_max, batch_50pct) для (1xA100, vllm, AWQ.4bit)

# 4. Обновить calibrated_coefficients.csv
python3 scripts/run_calibration.py  # пересоберёт все buckets

# 5. Прогнать тесты — не должны сломаться
python3 tests/test_formula.py
```

## Что обновить в репо после калибровки

Если new α/β сильно отличаются (>10%) от literature defaults в
`emulator/engines.py`:

1. **Не трогаем `ENGINE_DEFAULTS`** — он содержит fallback для случаев,
   когда калибровки нет. CLI автоматически использует калиброванные
   значения из CSV для (hw, engine, precision) bucket.

2. **`results/REPORT.md`** — добавить раздел «A100 vLLM calibration
   study» с найденными α, β, batch_saturation, графиками
   predicted-vs-observed (как для RTX-3090 / llama.cpp).

3. **`docs/BENCHMARK_SOURCES.md` §1** — добавить §1.4 «Собственный
   замер vLLM на A100», с указанием:
   - сколько строк добавлено в calibrated_coefficients.csv,
   - какие модели покрыты,
   - cross-check против Phase 1 Baseten / Phase 2 Qwen Speed Bench.

4. **`docs/BENCHMARK_SOURCES.md` §2** — повторить prediction sweep
   против тех же §2.2 / §2.4 точек после калибровки и записать **δ
   до vs после**. Цель — продемонстрировать что калибровка работает.

## Бюджет суммарно

| Phase | Cost | Hours | Что закрывает |
|---|---|---|---|
| 1: vLLM single-GPU | $6-9 | 4-6 | главный gap (vLLM/A100) |
| 2: vLLM multi-GPU + MoE | $12-18 | 4-6 | tp_efficiency, MoE на A100 |
| **MUST-HAVE сумма** | **$18-27** | **8-12** | основные gap'ы |
| 3: TGI/SGLang cross-engine | $3-5 | 2-3 | engine cross-validation |
| 4: TensorRT-LLM | $25-50 | 8-12 | FP8/TRT calibration |
| **Полный комплект** | **$46-82** | **18-27** | всё закрыто |

Один full-day budget Phase 1+2+3 ≈ **$25** — достаточно для
закрытия 80% gap'ов.

## Success criteria

После калибровки прогон CLI должен совпадать с реальностью:

| Сценарий | Текущая ошибка | Целевая ошибка после калибровки |
|---|---:|---:|
| §2.2a: A100 vLLM Qwen-7B single | +48% | < ±15% |
| §2.2b: A100 vLLM Qwen-7B P_in=6144 | +59% | < ±15% |
| §2.4a: H100 vLLM Llama-8B TTFT | −31% | без изменения (это другое железо) |
| §2.4b: H100 vLLM Llama-8B throughput | −56% | требует отдельной H100 калибровки |

Если эмулятор после калибровки выдаёт прогнозы для (A100, vLLM, *) с
ошибкой < 15% относительно публичных бенчмарков — Phase 1+2 успешны.

## Открытые вопросы для решения перед стартом

- **Какие модели приоритетнее в Phase 1**: Qwen-7B (есть на RTX-5090, прямой compare) или Llama-8B (другая семья, расширяет coverage)?
- **TP-efficiency на A100 SXM vs PCIe**: SXM имеет NVLink (быстро), PCIe — нет. Если в Vast только PCIe — `tp_eff` для 2× A100 будет ниже типичных «full» значений.
- **Сохранять ли raw чекпоинты на репо?** Логи в `data/raw_vllm/logs/`
  занимают ~150KB на штуку — не проблема. JSON-выходы тоже маленькие.
- **Включать ли APC в Phase 1 sweep?** Прямо сейчас vLLM по умолчанию
  включает APC; для чистой калибровки α/β нужен прогон с
  `--no-enable-prefix-caching` хотя бы один раз для контроля.
