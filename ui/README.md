# LLM Inference Emulator — Streamlit UI

Локальный веб-дашборд поверх `emulator.formula.predict()`. Шесть вкладок:

| Tab | Что показывает |
|---|---|
| 🎛 **Predictor** | Форма ввода (HW / engine / model / batch / контекст) → метрики (prefill, decode, throughput, memory, $/M tokens) + breakdown по членам формулы с подсветкой bottleneck |
| 📐 **Formula** | Roofline-формулы в LaTeX, расшифровка переменных, ссылки на код (`formula.py`, `hardware.py`, `engines.py`) |
| 📦 **Presets** | Содержимое `HARDWARE_SPECS`, `ENGINE_DEFAULTS`, `ARCH_DEFAULTS` в виде интерактивных таблиц |
| 🎯 **Calibration** | `results/calibrated_coefficients.csv` с фильтрами, scatter α по HW, медианная ошибка по железу |
| 🧪 **Validation** | Cross-check vs внешние бенчмарки (Baseten Mixtral, Qwen Speed Bench, self-vLLM на RTX-5090, Morphllm H100), таблица с Δ% и bar-chart отклонений |
| 📊 **Predicted vs Actual** | 9.9k бенчмарк-точек из `prediction_vs_actual.csv` — log-log scatter с фильтрами по mode (calibrated/uncalibrated/fine) и HW |

## Запуск

### Через Docker (рекомендуется)

```bash
# из корня репо emu-local/
docker compose up --build       # http://localhost:8501

# или вручную:
docker build -t llm-inference-emulator .
docker run --rm -p 8501:8501 llm-inference-emulator
```

UI откроется на `http://localhost:8501`.

### Локально без Docker

```bash
pip install -r requirements.txt -r requirements-ui.txt
streamlit run ui/app.py
```

## Как добавить валидационную точку

Отредактируй `ui/validation_data.py` — модуль `VALIDATION_POINTS` это просто
list dict'ов. Добавь запись с полями `id, hw, engine, scenario, metric, real,
pred, unit, delta_pct, verdict, source, url, comment` — UI подхватит при
следующем рендере (Streamlit hot-reloads).

## Как добавить hardware / engine / model preset

Изменения вносятся в источник правды:
- HW → `emulator/hardware.py::HARDWARE_SPECS`
- Engine → `emulator/engines.py::ENGINE_DEFAULTS`
- Model → `emulator/formula.py::ARCH_DEFAULTS`

UI читает их напрямую через import — никакой синхронизации не нужно.
