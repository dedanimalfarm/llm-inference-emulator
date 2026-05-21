# Задачи для Gemini

Hand-off файл для случаев когда твоя работа уходит на ревью, и нужно
вернуть тебе на доработку с конкретными критериями приёмки.

**Правила**:
- Не делать новых features в рамках задач этого файла — только починка.
- После каждой задачи — `python3 tests/test_formula.py` должен печатать
  `all tests passed`. Pytest-harness не используем.
- Streamlit UI: после правок прокликать **каждый** `ui_mode` в AppTest
  (см. T4 ниже). Это обязательный гейт перед push.
- Перед коммитом — `gh run list --workflow=test.yml --limit 1` →
  `completed success`. Не push'ить если красный.
- Не править то, что не указано в задаче.

---

## Round 1 — регрессии после `c6b1979` (закрыт)

- **T1 [x]** CI красный из-за `import pytest` → закрыт коммитом `bff270f`
- **T2 [x]** Ложный claim про обновлённые артефакты → закрыт `f2e9ddb`
  (реально обновил `docs/WALKTHROUGH.md`)
- **T3 [x]** Не все строки восстановлены в `prediction_vs_actual.csv` →
  закрыт `cc92c40` с root-cause анализом (33 = 11×3 modes корректно,
  60 был stale-data от обрезанного в `86f93c7` CSV).

---

## Round 2 — Pedagogical mode upcaught NameError (закрыт мной, починка верификации)

### T4 [ ] Добавить AppTest smoke-проверку всех `ui_mode` в pre-commit / CI

**Контекст**: при ревью feature/intuitive-ui (коммиты `cbba87e`/`b6f84da`)
обнаружено **3 разных NameError** в Pedagogical-режиме, ни один из них
не воспроизводился в Simple или Advanced. Все три легко ловятся через
`streamlit.testing.v1.AppTest` за 5 секунд. Ни один не был пойман до push.

Конкретно (для истории и тестового набора — fix уже в `e9794ff`):

| Стр. (на момент `b6f84da`) | Симптом | Исходная причина |
|---|---|---|
| 806, 1079 | `NameError: name 'kv_total' is not defined` | Переменная использовалась раньше своего определения; правильно — посчитать `kv_total = kv_tok_b * ctx_used * batch / kv_eff_pct` сразу после `kv_tok_b`. |
| 918, 1008 | `NameError: name 'p_in_eff' is not defined` | `p_in_eff` — внутренняя переменная `formula.py::predict()` (учитывает prefix caching). В UI её нет. Образовательный расчёт должен использовать просто `p_in`. |
| 1009, 1012 | `NameError: 't_pre'` / `'t_dec_base'` | Тоже внутренние имена `predict()`. В UI есть публичные `res.prefill_s` и `res.decode_per_token_s` — они и есть фактические значения, которые надо использовать как знаменатель для «achieved TFLOPS». |

**Симптом — почему не поймали**: при разработке Pedagogical-режима ни
разу не запустили его через AppTest или браузер с реальным кликом
«▶ Predict». Только UI-load (без клика) запускает script, но не доходит
до `else:`-ветки snap-рендера, где живёт baggy блок.

**Что сделать**:

1. **Создать `tests/test_ui_smoke.py`** — plain-runner script (не pytest),
   который циклит по `ui_mode` ∈ {Simple, Pedagogical, Advanced} и в
   каждом режиме:
   - запускает AppTest
   - находит кнопку «▶ Predict» и кликает
   - проверяет `at.exception == ElementList()` (нет Python-исключений)
   - игнорирует `st.error()` / `st.warning()` которые UI рендерит как
     часть бейджа точности или bottleneck-объяснения (это не errors,
     это форматированные предупреждения для пользователя)
   - печатает `all ui modes ok` или fail с конкретным трейсом

   Пример скелета:
   ```python
   from streamlit.testing.v1 import AppTest
   MODES = ['Базовый (Simple)', '🎓 Учебный (Pedagogical)', '⚙️ Продвинутый (Advanced)']
   for mode in MODES:
       at = AppTest.from_file('ui/app.py', default_timeout=30)
       at.session_state['ui_mode'] = mode
       at.run()
       btn = next((b for b in at.button if b.label == '▶ Predict'), None)
       assert btn is not None, f'Predict button missing in mode {mode}'
       btn.click(); at.run()
       assert at.exception == [] or at.exception is None or len(list(at.exception)) == 0, \
           f'mode {mode} threw: {at.exception}'
   print('all ui modes ok')
   ```

2. **Добавить новый job в `.github/workflows/test.yml`**: `ui-smoke`,
   который ставит `streamlit` + `altair` + `pandas` и запускает
   `python3 tests/test_ui_smoke.py`. Не должен ломаться matrix; можно
   отдельный job или extra step в существующем.

3. **Документировать в `CLAUDE.md` §4 (Continuous Verification Loop)**:
   «Streamlit UI: перед коммитом запустить `python3 tests/test_ui_smoke.py`,
   убедиться что печатает `all ui modes ok`».

**Критерий приёмки**:
- `python3 tests/test_ui_smoke.py` локально печатает `all ui modes ok`
- CI workflow `tests` зелёный на новой job-е
- `CLAUDE.md` обновлён, упоминание новой команды в §4

**Что НЕ трогать**:
- `ui/app.py` — fix уже в `e9794ff` (мой коммит на feature/intuitive-ui,
  смёржен в main через `2f34057`)
- Существующие unit-тесты `tests/test_formula.py` — они зелёные

---

## Что обычно остаётся не отмеченным

После закрытия задачи — пометить `[ ]` → `[x]` и/или удалить раздел
если он больше не актуален. Очень старые закрытые секции архивировать
в commit message и удалять из файла — этот файл должен быть «текущий
back-pressure», а не лог.
