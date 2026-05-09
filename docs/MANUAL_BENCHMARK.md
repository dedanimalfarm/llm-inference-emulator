# Инструкция по ручному бенчмаркингу (vLLM / Blackwell)

Этот документ описывает, как воспроизвести калибровку или добавить новые точки данных вручную.

## 1. Спецификация тестовой машины (Текущая)

Для точности эмулятора важно фиксировать параметры "железа", на котором снимались коэффициенты:

*   **Инстанс ID**: `36374596` (Vast.ai)
*   **GPU**: 2 × NVIDIA GeForce RTX 5090 (Blackwell)
    *   VRAM: 32 GB на карту (64 GB суммарно)
    *   Peak FP16: 419 TFLOPS (на карту)
    *   Memory BW: 1792 GB/s (на карту)
*   **CPU**: AMD EPYC 9654 (96 ядер) — критично для быстрой подготовки промптов.
*   **Disk**: 200 GB SSD (NVMe) — необходимо для кэша компиляции vLLM (~2-5 GB на модель).
*   **RAM**: 193 GB.

## 2. Подготовка окружения

```bash
# 1. Установка vLLM
pip install "vllm>=0.9.0"

# 2. Настройка переменных (важно для Blackwell)
export VLLM_USE_V1=0             # Стабильная версия планировщика
export OPENBLAS_NUM_THREADS=1    # Предотвращение Resource temporarily unavailable
export HF_HOME=/workspace/hf_cache # Хранение моделей на большом диске
```

## 3. Пример запуска одного теста

Разберем команду на примере **Qwen-7B (Batch 100)**:

```bash
taskset -c 0-31 vllm bench throughput \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --tensor-parallel-size 1 \
    --num-prompts 100 \
    --input-len 1024 \
    --output-len 128 \
    --dataset-name random \
    --output-json ~/vllm_results/qwen7b_b100.json
```

**Зачем эти параметры:**
*   `taskset -c 0-31`: Ограничиваем vLLM 32-мя ядрами. На 96-ядерных машинах vLLM может плодить слишком много потоков и вылетать по лимиту PID.
*   `--tensor-parallel-size`: 1 для моделей до 30B, 2 для 70B+ (или для замера TP-efficiency).
*   `--num-prompts`: Размер батча. Мы варьируем его от 1 до 1000, чтобы найти точку "сатурации" (когда скорость перестает расти).
*   `--input-len 1024 / --output-len 128`: Стандарт "длинного промпта" для замера чистой ПСП памяти.
*   `--dataset-name random`: Исключает влияние конкретных текстов на результат.

## 4. Сбор логов для калибровки

Для калибровки $\alpha$ (Prefill) нам недостаточно JSON, нам нужен **stdout**, чтобы достать строку:
`Avg prompt throughput: XXXX tokens/s`

Поэтому всегда запускайте с перенаправлением:
`vllm bench ... 2>&1 | tee ~/vllm_results/logs/test_name.log`

## 5. Интеграция в эмулятор

После получения JSON и лога на сервере:
1. Скопируйте их в `data/raw_vllm/` на хост.
2. Запустите импорт:
   `python scripts/import_vllm_bench.py data/raw_vllm/*.json --hw 2xRTX-5090 --output data/leaderboard-custom.csv`
3. Запустите калибровку:
   `python scripts/run_calibration.py`
