# Speculative Decoding в эмуляторе

Модель основана на Leviathan et al. 2022 *"Fast Inference from Transformers
via Speculative Decoding"* и расширена для современных вариантов (EAGLE,
Medusa, DeepSeek MTP) через roofline-вычисление стоимости draft-модели.

## Теория

За один **цикл** speculative-декода:

1. **Draft model** делает K последовательных forward-pass'ов и предлагает
   K новых токенов (стоимость: `K · t_draft_step`).
2. **Target model** в одном forward-pass проверяет все K позиций сразу
   плюс генерирует один «бонусный» токен из последней позиции
   (стоимость: `t_verify ≈ t_dec · (1 + ε·K)`).
3. Принятие — **последовательное**: токен i принимается с вероятностью
   `p` *только если* i−1 уже принят. Первый отвергнутый токен прерывает
   цепочку, дальнейшие отбрасываются. Из target всегда остаётся бонусный
   токен (либо resample, если был reject).

**Ожидаемое число принятых токенов** за цикл:

```
              ⎧ K + 1                       , если p = 1
E[N_accept] = ⎨
              ⎩ (1 − p^(K+1)) / (1 − p)     , иначе
```

Это **усечённая геометрическая прогрессия**. Старая формула
`accepted = p · K` неверна: она недооценивает позиции с высоким p
(не учитывает бонусный токен) и сильно переоценивает при низком p
(не учитывает каскадную потерю после первого reject).

**Время на принятый токен:**

```
t_per_accepted = (K · t_draft_step + t_verify) / E[N_accept]
speedup        = t_dec / t_per_accepted        # vs non-speculative baseline
```

## Что считает эмулятор

В `predict()` параметры:

| Параметр | Смысл | Default |
|---|---|---|
| `speculative` | вкл/выкл | False |
| `spec_accept_rate` | p, вероятность принятия | 0.7 |
| `spec_k_proposed` | K, длина драфта | 4 |
| `spec_overhead` | стоимость draft в долях `t_dec` (legacy) | 0.15 |
| `spec_draft_n_params_b` | размер draft-модели в B (физика, override) | None |
| `spec_draft_active_b` | active params для MoE-draft | = draft_n_params_b |
| `spec_verify_scale` | ε, штраф verify за лишние позиции | 0.05 |

**Возвращаемые поля** (в `InferenceResult`):

- `decode_per_token_s` — время на **принятый** (выходной) токен,
- `throughput_tok_s` — соответствующий throughput,
- `spec_e_accept` — E[N_accept] за цикл,
- `spec_speedup` — отношение non-speculative `t_dec` / speculative `t_per`.

### Два пути расчёта draft-стоимости

**(a) Legacy heuristic — `spec_overhead`.**
Если `spec_draft_n_params_b=None`, общая стоимость драфта за цикл
равна `spec_overhead · t_dec`. Удобно, когда конкретная draft-модель
не названа.

**(b) Физика — `spec_draft_n_params_b`.**
Draft step моделируется как memory-bound roofline:

```
t_draft_step = W_draft · (active/total) / (eff_mbw · β)
W_draft      = spec_draft_n_params_b · 1e9 · bits / 8
```

Используется `β` target-модели (упрощение: считаем, что эффективность
HBM одинакова). При MoE-draft `spec_draft_active_b` уменьшает чтение
весов. Этот путь предпочтительнее, когда draft назван
(Llama-3.2-1B → Llama-3-70B, EAGLE head, DeepSeek MTP).

## Sanity-таблица для 70B на 1xA100, vLLM, FP16, P_in=1024, P_out=256

Baseline non-speculative: `t/tok ≈ 81 ms`, `throughput ≈ 12.3 tok/s`.

| p | K | E[accept] | speedup | Комментарий |
|---:|---:|---:|---:|---|
| 0.50 | 2 | 1.75 | 1.55× | низкое p, малое K |
| 0.50 | 4 | 1.94 | 1.54× | плато |
| 0.50 | 8 | 2.00 | 1.32× | переоценка K вредна |
| **0.70** | **4** | **2.77** | **2.21×** | **EAGLE-2 zone** |
| 0.85 | 4 | 3.71 | 2.95× | Medusa, tree-attn |
| 0.85 | 8 | 5.12 | 3.38× | глубокие деревья |
| 0.95 | 4 | 4.52 | 3.60× | MTP / tight draft |
| 0.95 | 8 | 7.40 | 4.88× | теор. потолок K+1=9 |

(Получено через `predict()` с `spec_draft_n_params_b=1.0`.)

## Внешние ориентиры (для валидации формулы)

| Источник | Setup | Реальный speedup | Эмулятор |
|---|---|---:|---:|
| Leviathan 2022 (XSum + T5-XXL+T5-small) | p≈0.7, K=4 | 2.6× | 2.2× |
| EAGLE-2 (Li et al. 2024, Llama-2-70B) | p≈0.7, K=4 (tree) | 2.5–2.8× | 2.2× |
| DeepSeek-V3 MTP (technical report) | p≈0.85, K=1 | 1.8× | 1.8× (см. ниже) |
| vLLM v0.6 (Llama-3.1-70B draft 8B) | p≈0.6, K=5 | 1.5–1.7× | 1.7× |

Эмулятор систематически слегка занижает по сравнению с EAGLE/Medusa —
они используют tree attention (несколько кандидатов на позицию), что
эффективно увеличивает p без увеличения K. Для tree-вариантов
подставляйте «эффективный» p, который выше per-token p из бумаги.

## Примеры CLI

**EAGLE-style: 1B draft для 70B target на A100, P=1024, K=4, p=0.7:**

```bash
python scripts/cli.py --model 70 --bits 16 --hw 1xA100 --engine vllm \
  --p-in 1024 --p-out 256 --batch 1 \
  --speculative --spec-accept 0.7 --spec-k 4 --spec-draft-b 1.0
```

**DeepSeek MTP: K=1, p=0.85, draft = 1 MTP-блок (~2B).**
DeepSeek-V3 active=37B при decode (MoE top-8 из 256 экспертов), MTP-голова
— один transformer-блок поверх, на порядок меньше:

```bash
python scripts/cli.py --model 671 --bits 4 --hw 1xA100 --engine vllm \
  --p-in 1024 --p-out 256 --batch 1 \
  --speculative --spec-accept 0.85 --spec-k 1 --spec-draft-b 2.0
```

Если задать `--spec-draft-b 30`, эмулятор покажет ~1× — это корректная
физика: 30B draft для 37B-active таргета почти равны по стоимости, и
весь выигрыш acceptance съедается draft-проходом. Полезный sanity-check.

**Legacy heuristic (без конкретной draft-модели):**

```bash
python scripts/cli.py --model 70 --bits 16 --hw 1xA100 --engine vllm \
  --p-in 1024 --p-out 256 --batch 1 \
  --speculative --spec-accept 0.6 --spec-k 5 --spec-overhead 0.20
```

## Ограничения модели

1. **Acceptance rate p — внешний вход.** Эмулятор не предсказывает p
   из draft/target пары; нужно брать из бумаг или замерять.
2. **Tree attention** (EAGLE, Medusa) моделируется через подъём
   эффективного p, без явного перебора кандидатов.
3. **β draft = β target** — упрощение. На практике маленькие draft часто
   менее эффективны по HBM.
4. **Batch>1** — формула считает корректно (draft и verify скейлятся с
   bs_eff), но при continuous batching реальное взаимодействие
   draft/verify со scheduler'ом сложнее (вытеснение, длина запросов).
5. **KV-draft не моделируется** — у draft своя KV-cache, обычно
   маленькая; игнорируем для простоты.

Для калибровки против конкретного benchmark — задайте `p` и `K` из
отчёта и сравните `spec_speedup` с измеренным wall-clock speedup'ом.
