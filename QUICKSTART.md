# Быстрый старт — UI эмулятора

Один контейнер, один порт, одна команда.

## Требования

- Docker 20+ с плагином `compose` (`docker compose version` → `v2.x`)
- Свободный порт `8501` на localhost
- Для CLI без Docker — Python 3.11 / 3.12 / 3.13 (тестируем в CI на всех трёх)

Проверка:
```bash
docker compose version
ss -ltn | grep -q ':8501 ' && echo "порт 8501 занят" || echo "порт 8501 свободен"
```

## Запуск

Из корня репо (`/root/for-gemini/emu-local`):

```bash
docker compose up -d --build
```

Что произойдёт:
1. Соберётся образ `llm-inference-emulator:latest` (~30 сек, только в первый раз).
2. Поднимется контейнер `llm-emulator-ui`, проброс порта `8501 → 8501`.
3. Streamlit стартует на `http://localhost:8501`.

Открыть в браузере: **http://localhost:8501**

## Проверить, что живо

```bash
docker ps --filter name=llm-emulator-ui
# колонка STATUS должна быть "Up X seconds (healthy)"

curl -sf http://localhost:8501/_stcore/health
# должно ответить: ok
```

## Что внутри

Шесть вкладок сверху страницы:

| Вкладка | Что там |
|---|---|
| 🎛 **Predictor** | Форма ввода → результат + breakdown по членам формулы |
| 📐 **Formula** | LaTeX-формулы и расшифровка |
| 📦 **Presets** | Все hardware / engines / архитектуры моделей |
| 🎯 **Calibration** | Наши α/β из реальных замеров |
| 🧪 **Validation** | Сравнение формула vs внешние бенчмарки (с Δ%) |
| 📊 **Predicted vs Actual** | Scatter 9.9k точек |

## Логи

```bash
docker compose logs -f                # стрим логов
docker compose logs --tail=50         # последние 50 строк
```

## Перезапуск после правок

UI и `emulator/` смонтированы как volume → правки в коде подхватываются автоматически (Streamlit перерендерит страницу при сохранении файла). Контейнер перезапускать **не нужно**.

Если поменял `Dockerfile` или `requirements*.txt`:
```bash
docker compose up -d --build
```

## Остановить

```bash
docker compose down                   # остановить и удалить контейнер
docker compose stop                   # просто остановить, оставить контейнер
```

## Если порт 8501 занят

Меняем проброс в `docker-compose.yml`:
```yaml
ports:
  - "8600:8501"        # снаружи 8600, внутри тот же 8501
```
Затем `docker compose up -d` — UI откроется на `http://localhost:8600`.

## Если что-то сломалось

```bash
docker compose logs --tail=100        # смотрим Python traceback
docker compose down && docker compose up -d --build    # чистый перезапуск
```

---

## Известные UX-долги (для следующей итерации)

- Predictor: форма перегружена — стоит спрятать advanced-блок ещё глубже, добавить пресет-кнопки («T4 chatbot», «A100 70B», «H100 long-context»)
- Validation: таблицу превратить в карточки с цветовой индикацией Δ
- Calibration: scatter α/β зажат — переделать в facet по engine
- Predicted-vs-Actual: 9.9k точек тяжелы для глаза — добавить агрегацию (heatmap/bins), оставить scatter опциональным
- Общее: нет boot-state «выберите HW/engine/модель → жмите Predict». Сейчас Streamlit пересчитывает при каждом изменении любого input, что путает
- Нет тёмной темы / кастомизации `.streamlit/config.toml`
