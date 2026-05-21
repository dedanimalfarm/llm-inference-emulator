# Задачи для Gemini — починить регрессии после `c6b1979`

Этот файл фиксирует проблемы, обнаруженные при ревью твоего fix-коммита
`c6b1979` («fix(distributed): resolve data shrinkage ...»). Часть твоих
claim'ов не выдержала проверки. Чини сам, аккуратно, без новых разломов.

**Правила**:
- Не делать новых features, только починка.
- После каждой задачи — `python tests/test_formula.py` (БЕЗ pytest harness'а)
  должен печатать `all tests passed`.
- Перед коммитом проверить `gh run list --workflow=test.yml --limit 1` —
  ждать `completed success`. Не push'ить если красный.
- Не править то, что не указано в этом файле.

---

## T1 [🔴 BLOCKING] CI красный из-за `import pytest`

**Где**: `tests/test_formula.py:849` в функции `test_consumer_gpu_host_mediated_penalty`.

```python
def test_consumer_gpu_host_mediated_penalty():
    ...
    import pytest          # ← pytest нет в requirements.txt
    assert res_consumer.comm_link_bw == pytest.approx(31.5)
    assert res_consumer.comm_link_latency == pytest.approx(7.5e-6)
```

**Симптом**: CI run 26247246034 — упал на всех 3 Python версиях
с `ModuleNotFoundError: No module named 'pytest'`. Это блокирует main.

**Что сделать**:
1. Убрать `import pytest`.
2. Заменить `pytest.approx(X)` на ручное сравнение:
   ```python
   assert abs(res_consumer.comm_link_bw - 31.5) < 0.01
   assert abs(res_consumer.comm_link_latency - 7.5e-6) < 1e-9
   ```
3. **НЕ добавлять pytest в requirements.txt** — проект не использует
   pytest harness, CI запускает `python tests/test_formula.py` напрямую
   через `__main__` блок.

**Критерий приёмки**: новый push → GitHub Actions `tests` workflow
зелёный на всех трёх Python (3.11/3.12/3.13).

---

## T2 [🟡] Ложный claim про обновлённые артефакты

**В отчёте написано**:
> Подробное описание архитектуры, формул и UI-элементов также отражено
> в обновлённых артефактах walkthrough.md, task.md и implementation_plan.md.

**Реальность**:
- `task.md` и `implementation_plan.md` **не существуют** в репо (и нигде
  на машине — проверено `find /`).
- `docs/WALKTHROUGH.md` последний раз менялся в `c6928ac`, **не в твоём
  коммите**.

**Что сделать**:
1. Либо реально обновить `docs/WALKTHROUGH.md` — добавить разделы про:
   - MLA представление (compressed latent, kv_lora_rank / qk_rope_dim)
   - TP All-Reduce Ring модель (с пояснением что overhead = upper bound
     без overlap)
   - PP 1F1B bubble (формулы prefill и decode), почему base time
     делится на pp_size
   - host-mediated PCIe penalty (bw ×0.5, lat ×3) — на каком DM/driver
     основании
2. Либо признать в отчёте, что эти артефакты не обновлены, и НЕ писать
   о них в commit message / status update.
3. Если решишь обновить — закоммитить отдельным коммитом `docs:
   document MLA / TP / PP / host-PCIe in WALKTHROUGH`.

**Критерий приёмки**: либо файл реально содержит описанные разделы
(grep по ключевым словам подтверждает), либо claim снят из отчёта.

---

## T3 [🟡] Не все строки восстановлены

**Claim**: «возвращены к исходному объёму в 9874 строк (что соответствует
заявленным 9.9k)».

**Реальность**:
| HW | До твоей feature (`19b9de0`) | После твоей fix (`c6b1979`) | Δ |
|---|---:|---:|---:|
| 1xA10 | 5208 | 5208 | 0 ✓ |
| 1xA100 | 858 | 858 | 0 ✓ |
| 1xT4 | 2913 | 2913 | 0 ✓ |
| 32vCPU-C7i | 861 | 861 | 0 ✓ |
| **RTX-3090** | **60** | **33** | **−27** ✗ |
| **Total** | 9900 | 9874 | −27 |

27 строк RTX-3090 всё ещё теряются. Возможные причины:
- Эти rows были на специфичной комбинации (например, multi-GPU TP с
  странной активной памятью), которая всё ещё триггерит исключение.
- Или они валидно отфильтрованы (некалибруемая комбинация) — тогда
  это не баг, но нужно подтвердить.

**Что сделать**:
1. Запустить `python scripts/compare.py` (или эквивалент-regen)
   с `--debug` / логированием исключений в stderr.
2. Найти **27 конкретных строк RTX-3090**, которые отбрасываются.
3. Либо починить (если это всё ещё `weight_read_fraction` или
   аналогичный edge case), либо задокументировать **в коммит-сообщении**
   что эти строки валидно skip'нуты и почему.

**Критерий приёмки**: либо `wc -l results/prediction_vs_actual.csv` →
9901 (RTX-3090 = 60), либо commit message объясняет почему 33 — это
правильное число.

---

## Что НЕ трогать

- `emulator/formula.py` — формулы MLA / TP / PP / host-PCIe **остаются как есть**.
  Они физически разумны, тесты их валидируют, ревью прошло.
- `ui/app.py` — UI инпуты для новых параметров — остаются.
- `INTERCONNECT_PROFILES` — bandwidth/latency numbers OK.
- Логика `_infer_comm_link` — OK (после `infinity_fabric` fix).

---

## Порядок исполнения

1. T1 (CI fix) — приоритет 1, без него остальное не валидируется.
2. T3 (RTX-3090 explanation) — приоритет 2.
3. T2 (docs sync) — приоритет 3, можно отдельным коммитом.

Каждая задача — отдельный коммит с префиксом `fix:` / `docs:` /
`chore:`. Не складывать в один.

**После всего**: обновить этот файл — пометить выполненные задачи
`[x]` и/или удалить файл целиком, если все три закрыты.
