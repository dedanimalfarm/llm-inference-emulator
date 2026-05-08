# Оптимизации vLLM и корректировки формулы

> Статус: **проектный документ**, имплементация в коде ещё не сделана.
> См. секцию «Что добавить в код» в конце — это roadmap для следующего PR.

vLLM — самый продвинутый из открытых LLM-серверов на 2026 год. Он быстрее
наивного PyTorch в 3-10× благодаря набору оптимизаций, каждая из которых
требует **корректировки нашей формулы**. Этот документ их перечисляет и
показывает что именно меняется в `predict()`.

## TL;DR

| Что в vLLM | Где попадает в формулу | Тип корректировки |
|---|---|---|
| **PagedAttention** | `memory_gb` | новый параметр `kv_packing_efficiency` |
| **Continuous batching** | `throughput_tok_s` | замена `batch_mult` на saturation-функцию |
| **Prefix caching (APC)** | `t_pre` через `P_in_eff` | новый параметр `prefix_cache_hit_rate` |
| **Speculative decoding** | `t_dec_final` | новые параметры `accept_rate`, `k_proposed` |
| **CUDA graphs** | `α_decode_compute` | повышение калиброванного α |
| **Chunked prefill** | mixed prefill+decode batch | повышение α/β через калибровку |
| **Marlin/Machete kernels** (INT4/8) | `α` для квантованных | per-quant калибровка |
| **FlashAttention-3** | `t_kv` | косвенно через β |

Из них **4 требуют новых параметров** (PagedAttention, continuous batching,
prefix caching, speculative decoding). Остальные сводятся к **сдвигу α/β**
и закрываются обычной калибровкой engine-specific коэффициентов.

---

## 1. PagedAttention

### Что делает

KV-кэш разбивается на фиксированные блоки (типично 16 токенов на блок).
Память аллоцируется через таблицу страниц как в виртуальной памяти ОС:
блоки разных запросов могут лежать рядом, фрагментация около нуля.

### Эффект на физику

В наивной реализации для KV выделяется максимально возможный буфер
заранее → ~50-65% реальной утилизации памяти из-за того что большинство
запросов короче максимума. PagedAttention даёт **95-99% утилизации**.

### Корректировка формулы

Сейчас наш расчёт памяти:
```python
memory_gb = (W + KV_per_tok · ctx_total) / 1e9
```

С PagedAttention:
```python
memory_gb = (W + KV_per_tok · ctx_total / kv_packing_efficiency) / 1e9
```

### Параметр

```python
"vllm":     {"kv_packing_eff": 0.97}
"tensorrt": {"kv_packing_eff": 0.92}
"pytorch":  {"kv_packing_eff": 0.65}    # naive contiguous
"llama.cpp":{"kv_packing_eff": 0.85}    # ring buffer
```

### Эффект

Не меняет throughput напрямую. Меняет **сколько concurrent requests
влезает в VRAM** → косвенно поднимает суммарную пропускную способность
сервиса, но не на одного пользователя.

---

## 2. Continuous batching

### Что делает

Стандартный «static batching»: собрали N запросов, прогнали их вместе,
ждём пока последний закончит. Самый медленный держит остальные.

Continuous batching: на каждом decode-шаге планировщик может **взять
новый запрос из очереди** и поставить в свободный slot. Когда запрос
заканчивается, его слот сразу занимает следующий.

### Эффект на физику

GPU **никогда не простаивает** между requests. На рабочей нагрузке с
переменной длиной ответов это даёт ×3-10 throughput vs static batching.

### Корректировка формулы

Сейчас у нас:
```python
throughput = batch · batch_mult / t_dec
```
где `batch_mult = 5.0` для vLLM — грубый множитель.

Реально нужна **saturation-функция**:
```python
def effective_batch(batch, batch_max, batch_50pct):
    """Сколько concurrent requests реально работает.
    При batch=batch_50pct получаем 50% от максимума.
    При batch >> batch_max — выходим на batch_max."""
    return batch_max * batch / (batch + batch_50pct)

bs_eff = effective_batch(batch, batch_max=16, batch_50pct=2)
throughput = bs_eff / t_dec
```

### Параметры в `engines.py`

```python
"vllm": {
    "batch_max": 16,        # пиковая concurrency на 1× A100
    "batch_50pct": 2,       # batch для 50% saturation
}
"pytorch": {
    "batch_max": None,      # без saturation: bs_eff = batch
}
```

### Калибровка

`batch_max` зависит от модели и железа. Для 7B Q4 на A100 ≈ 32, на T4 ≈ 8.
Нужны замеры через `vllm bench throughput --num-prompts 100,200,500,1000`.

---

## 3. Prefix caching (APC, Automatic Prefix Caching)

### Что делает

Системные промпты, few-shot примеры, документы для RAG — это часто **одно
и то же** для разных запросов. vLLM хеширует префиксы (по блокам
PagedAttention) и переиспользует уже посчитанный KV-кэш.

### Эффект на физику

Если у вас 500-токенный системный промпт + 100-токенный пользовательский
вопрос, и кэш горячий — обрабатывается только 100 токенов. **TTFT падает
в 5-10×** для повторяющихся паттернов. Decode не меняется (всё равно
читает полный KV).

### Корректировка формулы

Введём `h ∈ [0, 1]` — доля токенов промпта, попавших в кэш:

```python
P_in_effective = P_in · (1 - h)
t_pre_compute  = 2 · N · P_in_effective · batch / (C · α)
t_pre_mem      = W / MBW          # не меняется (веса всё равно читаем раз)
t_pre          = max(t_pre_compute, t_pre_mem)
```

### Параметр

```python
predict(..., prefix_cache_hit_rate=0.0)  # default: нет кэша
```

Типичные значения для real-world сценариев:
- chat с системным промптом 200 токенов и user 50 → `h ≈ 0.8` для 2-го+ запроса
- RAG с 1000-токенным контекстом + 100 user → `h ≈ 0.91`
- batch обработка независимых документов → `h = 0.0`

### Эффект

При `h = 0.8` и P_in=500: prefill ускоряется в 5×, что для коротких
ответов почти полностью устраняет TTFT.

---

## 4. Speculative decoding

### Что делает

Маленькая «draft» модель (например Llama-3.2-1B) предлагает `k`
кандидатных токенов за дешёвый forward pass. Большая «main» модель
(Llama-3.1-70B) проверяет все `k` за **один** forward pass через
параллельное вычисление.

Если все `k` верны — генерируем `k` токенов за 1 шаг main модели.
Если первый отвергнут — берём только тот, что сама main выдала бы.

### Эффект на физику

Acceptance rate `r ∈ [0.5, 0.9]` для хороших draft+main пар. С `k = 4`
и `r = 0.7`: средне принятых = 2.8 токена за шаг. **Ускорение decode
в 2-3×** для batch=1.

При больших batch исчезает (draft становится бутылочным горлышком).

### Корректировка формулы

```python
if speculative:
    draft_overhead = 0.10..0.20    # время draft относительно main step
    t_step = t_dec_base · (1 + draft_overhead)
    accepted_per_step = accept_rate · k_proposed
    t_dec_final = t_step / accepted_per_step
else:
    t_dec_final = t_dec_base
```

### Параметры

```python
predict(...,
    speculative=False,
    spec_accept_rate=0.7,        # типично для размер-парных моделей
    spec_k_proposed=4,           # стандарт vLLM
    spec_draft_overhead=0.15,    # draft в 5-10× меньше main
)
```

### Эффект

Llama-3.1-70B + Llama-3.2-1B draft, batch=1: с `r=0.7, k=4` decode
ускоряется в `(0.7 × 4) / 1.15 ≈ 2.4×`.

---

## 5. CUDA graphs

### Что делает

Захват цепочки kernel-вызовов и их ре-проигрывание одной командой.
Убирает overhead от kernel-launch (5-50 мкс на kernel × 100+ kernels
на decode step = 0.5-5 ms потерь, что **много** для batch=1 где decode
≈ 10-50 ms).

### Эффект на формулу

Не требует новых параметров — сводится к **повышению α на decode** для
маленьких батчей. Захватывается калибровкой:

```python
# было
"vllm": {"alpha": 0.40, ...}

# с CUDA graphs
"vllm": {"alpha": 0.50, ...}
```

Точные числа — из калибровочных замеров.

---

## 6. Chunked prefill

### Что делает

Длинный prefill режется на чанки (например по 512 токенов), и эти чанки
**смешиваются с decode-токенами других запросов** в одном forward pass.
GPU всегда загружен либо чистым prefill либо смесью prefill+decode.

### Эффект на формулу

Не требует новых параметров. Эффект — более **стабильная latency** и
**выше суммарный throughput** на сервере. Захватывается через
повышенный `α` для `vllm` при чалибровке.

---

## 7. Marlin / Machete / другие optimised quant kernels

### Что делает

Стандартный путь для Q4 на CUDA: dequant fp16 → fp16 GEMM. Marlin (для
INT4) или Machete (для INT4/INT8) делают **прямой GEMM на упакованных
данных**, без промежуточного dequant. На H100 это даёт +30-50% к prefill.

### Эффект на формулу

Через `compute_path` параметр (уже есть в нашем `engines.py`):

```python
"vllm": {
    "compute_path": 16,    # сейчас — assume fp16 path
    # с Marlin для AWQ:
    "compute_path_quantized": 8,    # ← новое поле
}
```

Или просто per-quant калибровка `α`:
```
vllm + Q4_K_M (Marlin):  α = 0.55
vllm + Q4_K_M (naive):   α = 0.40
```

Та же машинерия что у нас уже есть для llama.cpp Q4_K. Просто нужны
данные.

---

## 8. FlashAttention-3 (FA3)

### Что делает

Версия 3 для Hopper с поддержкой FP8 и асинхронных warp-specialization.
На длинных контекстах (32k+) даёт +50-100% к attention computation vs FA2.

### Эффект на формулу

KV-term:
```python
t_kv = KV_per_tok · ctx_avg / (MBW · β)
```

С FA3 эффективное `β` для KV выше → `t_kv` уменьшается. Можно
сэмулировать отдельным `β_kv` параметром или включить в общий
калиброванный `β`.

---

## Полная скорректированная формула для vLLM

```
ВХОД (новые параметры выделены):
  N, bits, P_in, P_out, batch, C, MBW, α, β,
  L, H_kv, d_head, b_kv_k, b_kv_v, TP, E_TP,
  ★ h_prefix          — prefix cache hit rate
  ★ batch_max         — peak concurrency (e.g. 16 for vLLM 7B/A100)
  ★ batch_50pct       — saturation midpoint
  ★ kv_pack_eff       — PagedAttention efficiency (0.97)
  ★ spec_decode       — speculative on/off
  ★ spec_accept_rate  — token acceptance rate
  ★ spec_k_proposed   — draft tokens per step
  ★ spec_overhead     — draft model relative cost


ШАГ 1. Производные:
  W           = N · bits / 8
  C_eff       = C · TP · E_TP
  MBW_eff     = MBW · TP · E_TP
  P_in_eff    = P_in · (1 - h_prefix)              ← prefix caching
  bs_eff      = batch_max · batch / (batch + batch_50pct)   ← continuous batching


ШАГ 2. Prefill:
  t_pre = max(
      2·N·P_in_eff·batch / (C_eff · α),
      W / MBW_eff
  )


ШАГ 3. Decode базовый:
  t_dec_base = max(W / (MBW_eff · β),  2·N·batch / (C_eff · α))


ШАГ 4. KV-cache:
  KV_per_tok = L · H_kv · d_head · (b_kv_k + b_kv_v) / 8
  t_kv       = KV_per_tok · (P_in + P_out/2) / (MBW_eff · β)
  t_dec      = t_dec_base + t_kv


ШАГ 5. Speculative decoding:
  if spec_decode:
      t_step  = t_dec · (1 + spec_overhead)
      t_dec_final = t_step / (spec_accept_rate · spec_k_proposed)
  else:
      t_dec_final = t_dec


ШАГ 6. Memory budget (с PagedAttention):
  memory_GB = (W + KV_per_tok · (P_in + P_out) · batch / kv_pack_eff) / 1e9


ШАГ 7. Метрики:
  latency    = t_pre + P_out · t_dec_final
  throughput = bs_eff / t_dec_final
```

---

## Численный пример: 7B Q4 на 1× A100

Сценарий: chat с system prompt, prefill 1024, decode 256, batch=8.

| Конфиг | TTFT | Throughput | Memory |
|---|---:|---:|---:|
| **PyTorch baseline** (α=0.20, β=0.55, batch_mult=1.0) | 350 ms | 30 t/s | 8 GB |
| **vLLM, без spec, h=0** (α=0.50, β=0.85, batch_max=16, kv_pack=0.97) | 175 ms | 110 t/s | 6 GB |
| **+ prefix caching h=0.4** | **105 ms** | 110 t/s | 6 GB |
| **+ speculative (r=0.7, k=4)** | 105 ms | **220 t/s** | 6 GB |
| **+ всё** | 105 ms | 220 t/s | 6 GB |

Видно изолированные эффекты каждой оптимизации:
- PagedAttention → −25% memory
- prefix caching → −40% TTFT
- continuous batching уже зашит в `α/β/batch_max`
- speculative decoding → ×2 throughput (только при batch=1, не на всех нагрузках)

---

## Что добавить в код

### Шаг A — новые параметры в `predict()` без поломки совместимости

```python
def predict(
    n_params_b, bits, p_in, p_out, batch,
    peak_flops, mem_bw,
    alpha=0.25, beta=0.60, batch_mult=1.0,
    layers=None, d_model=None, kv_heads=None,
    kv_bits_k=16.0, kv_bits_v=16.0,
    tp_size=1, tp_efficiency=1.0,
    # === новые vLLM-параметры ===
    prefix_cache_hit=0.0,
    batch_saturation=None,        # tuple (batch_max, batch_50pct) или None
    kv_packing_eff=1.0,
    speculative=False,
    spec_accept_rate=0.7,
    spec_k_proposed=4,
    spec_overhead=0.15,
):
    ...
```

Все дефолты «как было» → существующие тесты не ломаются.

### Шаг B — миграция `engines.py` с `batch_mult` на `batch_saturation`

```python
"vllm": {
    "alpha": 0.50, "beta": 0.85,
    "batch_saturation": (16, 2),       # (max, 50pct)
    "kv_packing_eff": 0.97,
    "compute_path": 16,
    "notes": "PagedAttention + continuous batching + APC; "
             "use prefix_cache_hit/speculative for fine-tuning",
},
"pytorch": {
    "alpha": 0.20, "beta": 0.55,
    "batch_saturation": None,           # = batch_mult=1
    "kv_packing_eff": 0.65,
    ...
},
```

### Шаг C — реальная vLLM-калибровка

Сейчас все vLLM-числа literature defaults. Чтобы получить честные:

1. Поднять GPU-инстанс (A10 / A100 / 3090).
2. Установить vLLM, запустить `vllm bench throughput --model <m> --num-prompts 100 200 500 1000`.
3. Импортёр `scripts/import_vllm_bench.py` (нужно написать) преобразует CSV в нашу схему.
4. `python scripts/run_calibration.py` даёт `α/β` для (`hw, vllm, precision`).

Ожидаемые числа после калибровки:
- A100 vLLM AWQ.4bit: α ≈ 0.55, β ≈ 0.85, batch_max ≈ 32
- T4 vLLM Q4: α ≈ 0.50, β ≈ 0.78, batch_max ≈ 8

### Шаг D — расширить `WALKTHROUGH.md`

Добавить раздел «vLLM-стиль оптимизации в формуле» — пример с
prefix-кэшем и спекулятивным декодингом, как меняются числа.

### Шаг E — тесты

```python
def test_prefix_cache_reduces_ttft():
    base   = predict(..., prefix_cache_hit=0.0)
    cached = predict(..., prefix_cache_hit=0.8)
    assert cached.prefill_s < base.prefill_s * 0.3   # 5× speedup at h=0.8

def test_speculative_decoding_speeds_up_decode():
    base = predict(..., speculative=False)
    spec = predict(..., speculative=True, spec_accept_rate=0.7, spec_k_proposed=4)
    assert spec.decode_per_token_s < base.decode_per_token_s * 0.5   # ~2× speedup
```

---

## Ссылки на источники

- vLLM paper: https://arxiv.org/abs/2309.06180 (PagedAttention)
- vLLM blog about APC: https://blog.vllm.ai/2024/01/27/apc.html
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Marlin kernel: https://github.com/IST-DASLab/marlin
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- Continuous batching benchmark: https://www.anyscale.com/blog/continuous-batching-llm-inference

---

## Roadmap

- [ ] Шаг A — новые параметры в `predict()` (~1 час)
- [ ] Шаг B — миграция `batch_mult` → `batch_saturation` (~30 мин)
- [ ] Шаг C — vLLM-калибровка (требует GPU-инстанс с vLLM, ~3 часа)
- [ ] Шаг D — обновить `WALKTHROUGH.md` (~30 мин)
- [ ] Шаг E — тесты (~30 мин)

Итого: ~5-6 часов работы (или один долгий сеанс) для полной поддержки
vLLM-стиля оптимизаций в эмуляторе.
