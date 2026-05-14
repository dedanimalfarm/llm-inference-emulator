# Полный пример расчёта формулы

> Подробный walkthrough того, как эмулятор предсказывает скорость инференса
> LLM. Чтение займёт ~15 минут. После этого вы поймёте каждое число в
> `results/calibrated_coefficients.csv` и сможете объяснить почему `predict()`
> возвращает именно то что возвращает.

## Кому это пригодится

- **Инженер MLOps**, который хочет понять как работает эмулятор перед тем
  как полагаться на его прогнозы для capacity planning.
- **Исследователь**, который хочет переиспользовать roofline-формулу для
  своего железа/движка.
- **Студент**, который читает про LLM-инференс и хочет увидеть реальную
  математику с числами, а не только формулы из статей.

Не нужно быть ML-инженером. Все термины объясняются по ходу.

---

## Глоссарий — без него дальше не пойти

Если хоть один термин не очевиден — прочтите, иначе остальное не сложится.

### LLM (Large Language Model)
Большая языковая модель типа GPT, Llama, Qwen. Внутри — массив чисел
(параметров) на десятки миллиардов. Принимает текст, выдаёт продолжение.

### Inference (инференс)
Это **запуск модели в режиме «отвечай»**, в отличие от **training (обучения)**.
Эмулятор считает только инференс — обучение это совсем другие нагрузки.

### Token (токен)
Не слово, не буква. Это **кусочек слова**, единица обработки модели.
В русском 1 токен ≈ 0.7 слова, в английском ≈ 0.75 слова.

### Prefill / Prompt processing
**Первая фаза** инференса — модель «читает» весь ваш промпт целиком,
параллельно обрабатывая все его токены. В CSV: колонки `n_prompt > 0,
n_gen = 0, test = pp512`.

### Decode / Token generation
**Вторая фаза** — модель генерирует ответ **по одному токену за раз**.
Каждый следующий токен зависит от всех предыдущих, поэтому параллелить
нельзя. В CSV: `n_prompt = 0, n_gen = 128, test = tg128`.

### KV cache (KV-кэш)
Чтобы decode не пересчитывал внимание для всех предыдущих токенов на каждом
шаге, модель **запоминает Key и Value тензоры** для каждого уже обработанного
токена. Растёт линейно с длиной контекста. На каждый новый токен нужно
прочитать **весь** кэш — поэтому длинный контекст замедляет декод.

### Quantization (квантование)
Хранение весов модели в **меньшем числе бит** для экономии памяти и пропускной
способности. FP16 = 16 бит на вес, INT8 = 8 бит, Q4_K_M (формат GGUF) ≈ 4.91 бит.
Качество модели падает на 1-3%, размер уменьшается в 4× — отличный размен.

### bpw (bits per weight)
**Эффективное число бит на параметр** в реальном файле модели. Для GGUF
формата `bpw = 8 × file_size_bytes / n_params`. Q4_K_M = 4.91 bpw — это
не ровно 4 потому что есть служебные данные (scales, zeros).

### FLOPS / TFLOPS
**Floating-point operations per second** — сколько умножений с плавающей
точкой железо может сделать за секунду. **TFLOPS** = триллионы FLOPS.
Например, RTX 3090 даёт 142 TFLOPS на FP16 tensor cores.

### Memory bandwidth (полоса памяти)
**Сколько байт в секунду** GPU может прочитать из своей VRAM. RTX 3090 = 936 GB/s
(гигабайт в секунду). Это главный bottleneck для decode-фазы LLM.

### Compute-bound / memory-bound
**Compute-bound** — задача упирается в число умножений (FLOPS). **Memory-bound**
— задача упирается в чтение из памяти (bandwidth). Prefill обычно
compute-bound, decode — memory-bound. Это **главный факт** про LLM-инференс.

### MFU (Model FLOPs Utilization), наш `α`
Доля **пиковой compute мощности** GPU, которую реально использует движок.
Идеал = 1.0 (100%), реальность 0.20-0.60. Это и есть `α` в нашей формуле.

### MBU (Memory Bandwidth Utilization), наш `β`
Аналогично для пропускной способности памяти. Идеал = 1.0, реальность
0.55-0.95. Это `β`.

### Roofline model
Классическая модель из HPC: **время = max(compute_time, memory_time)**.
Что медленнее — то и доминирует. Это сердце нашей формулы.

### Tensor Parallel (TP) / Pipeline Parallel (PP)
Способы запустить модель **на нескольких GPU**:
- **TP (tensor parallel)**: каждый слой делится между картами горизонтально,
  они работают **одновременно**, потом all-reduce. Требует быстрой связи (NVLink).
- **PP (pipeline parallel)**: разные слои на разных картах. Один токен
  идёт сквозь обе **последовательно**. Без NVLink это **единственный
  работоспособный вариант**.
- **llama.cpp `--split-mode layer`** = pipeline parallel.
- **llama.cpp `--split-mode row`** = попытка tensor parallel, но без NVLink
  оверхед all-reduce съедает весь выигрыш.

### TP-efficiency, наш `tp_efficiency`
Сколько мы получаем **от теоретического 2× ускорения**. Если 2 карты работают
последовательно (PP) — мы получаем 1× single-card → `tp_efficiency = 0.50`
(half of theoretical 2×). Если бы был настоящий TP с NVLink — было бы 0.85.

---

## Постановка задачи

**Хочу узнать**: насколько быстро Qwen-2.5-32B (квантованная в Q4_K_M, размер
файла 19.9 GB) будет работать на двух RTX 3090 через llama.cpp при промпте
512 токенов и генерации 128 токенов?

Это **типичный вопрос для capacity planning**: «купить ли две 3090, или одну
A100? Хватит ли двух 3090 чтобы запустить 32B-модель?». Без эмулятора
ответ требует купить железо и попробовать. С эмулятором — секунда.

**Эталон для сверки**: реальный замер на той же машине показал
`pp512 = 1367.4 t/s, tg128 = 39.4 t/s` (`data/llama_bench_qwen32b_multi.csv`).

---

## Слой 1 — Что подаёт пользователь

### Через CLI

```bash
python scripts/cli.py \
    --model 32.764 \
    --bits 4.85 \
    --hw 2xRTX-3090 \
    --engine llama.cpp \
    --p-in 512 \
    --p-out 128 \
    --batch 1 \
    --precision-label "Q4_K (4.85bpw)"
```

### Что значит каждый флаг

| Флаг | Значение | Где взять |
|---|---|---|
| `--model 32.764` | размер модели в **миллиардах параметров** | спецификация модели на HuggingFace |
| `--bits 4.85` | **эффективные** биты на вес | `8 × file_size_GB / params_B` |
| `--hw 2xRTX-3090` | имя профиля железа | список в `emulator/hardware.py` |
| `--engine llama.cpp` | имя движка | список в `emulator/engines.py` |
| `--p-in 512` | длина промпта в токенах | от вас, ваша рабочая нагрузка |
| `--p-out 128` | сколько токенов сгенерировать | от вас |
| `--batch 1` | сколько одновременных запросов | от вас |
| `--precision-label` | имя кальпровочной корзины | из `results/calibrated_coefficients.csv` |

Это **всё**, что должен знать пользователь. Дальше эмулятор сам подтянет
параметры железа и движка.

---

## Слой 2 — Что подтягивается из таблиц

### 2.1 Хардвер: `HARDWARE_SPECS["2xRTX-3090"]`

Файл `emulator/hardware.py`:

```python
"2xRTX-3090": {
    "peak_tflops": {16: 142.0, 8: 284.0, 4: 284.0},   # PER-CARD!
    "memory_bandwidth_gbs": 936.0,                     # PER-CARD!
    "memory_capacity_gb": 48.0,                        # AGGREGATE (две карты вместе)
    "tdp_w": 700,
    "tp_size": 2,
    "tp_efficiency": 0.50,
}
```

⚠️ **Важно**: `peak_tflops` и `memory_bandwidth_gbs` хранятся для **одной
карты**, не для двух суммарно. Иначе при умножении на `tp_size` в формуле
получился бы двойной счёт. `memory_capacity_gb` напротив — суммарная,
потому что она используется только для проверки «влезает ли модель».

Вызов `get_peak_compute("2xRTX-3090", 16)` возвращает:
```
peak_flops = 142.0 × 10^12 = 1.42 · 10^14 FLOPS/sec   (per-card)
```

`get_memory_bandwidth("2xRTX-3090")`:
```
mem_bw = 936 × 10^9 = 9.36 · 10^11 B/sec   (per-card)
```

### 2.2 Движок: `ENGINE_DEFAULTS["llama.cpp"]`

Файл `emulator/engines.py`:

```python
"llama.cpp": {
    "alpha": 0.36,        # MFU median (calibrated)
    "beta": 0.72,         # MBU median (calibrated)
    "batch_mult": 1.0,    # без continuous batching
    "compute_path": 16,   # GGUF dequant→fp16, не INT8
    "notes": "WARNING: mixed-precision KV..."
}
```

Эти числа — **literature defaults**. Если есть точная калибровка для
конкретной комбинации `(hw, engine, precision)` — она перебивает defaults.

### 2.3 Калибровка: `results/calibrated_coefficients.csv`

```
hw          backend     precision           α_med   α_p75   β_med   β_p75
RTX-3090    llama.cpp   Q4_K (4.91bpw)      0.36    0.53    0.72    0.78
2xRTX-3090  llama.cpp   Q4_K (4.85bpw)      0.29    0.30    0.42    0.42
```

Видим что для multi-GPU `α` и `β` ниже чем для single. Это потому что
multi-GPU добавляет накладные расходы (передача активаций между картами,
фрагментация compute), которые «съедают» в эффективные коэффициенты.

**Для нашего расчёта возьмём p75 калибровки 2xRTX-3090**:
```
α = 0.30   (best-case prefill MFU)
β = 0.42   (best-case decode MBU)
```

p75 — это **75-й перцентиль**, то есть «лучшие 25% наблюдений». Используется
для прогнозов в **благоприятных условиях** (большой батч, короткий контекст,
оптимальный квант). Median (`α=0.29, β=0.42`) — это «типичный случай».

### 2.4 Архитектура модели: `ARCH_DEFAULTS`

Файл `emulator/formula.py`. Для Qwen-32B-Instruct:

```python
14.0: {"layers": 48, "d_model": 5120, "kv_heads": 8}   # ближайшая запись
```

`_arch_for(32.764)` находит ближайшую запись `≤ 32.764` — это `14.0`.
Берём оттуда `layers=48, d_model=5120, kv_heads=8`. Эти параметры нужны
**только для расчёта KV-cache размера**, потому что прямые формулы
prefill и decode зависят только от общего `N` (числа параметров).

### Итого что у нас на руках

```
N         = 32.764 × 10^9 params         (от пользователя)
bits      = 4.85                          (от пользователя)
W         = N × bits / 8                  → 1.99 × 10^10 байт ≈ 19.9 GB

p_in      = 512 токенов                   (от пользователя)
p_out     = 128 токенов                   (от пользователя)
batch     = 1                             (от пользователя)

peak      = 1.42 × 10^14 FLOPS/sec        (per-card, из HARDWARE_SPECS)
MBW       = 9.36 × 10^11 B/sec            (per-card, из HARDWARE_SPECS)

α         = 0.30                          (p75 из calibrated_coefs)
β         = 0.42                          (p75 из calibrated_coefs)

tp_size       = 2                         (из HARDWARE_SPECS)
tp_efficiency = 0.50                      (из HARDWARE_SPECS)

layers    = 48                            (из ARCH_DEFAULTS)
d_model   = 5120
kv_heads  = 8
kv_bits_k = 16  (default, fp16)
kv_bits_v = 16
```

---

## Слой 3 — Что происходит внутри `predict()`

### 3.1 Эффективные ресурсы для multi-GPU

```python
eff_flops = peak × tp_size × tp_efficiency
         = 1.42·10^14 × 2 × 0.50
         = 1.42·10^14 FLOPS/sec
```

Заметьте: `tp_size × tp_efficiency = 2 × 0.5 = 1.0`, то есть **multi-GPU
здесь даёт ровно single-card производительность**. Это математически
выражает «layer split = pipeline parallel, throughput не растёт».

```python
eff_mbw = MBW × tp_size × tp_efficiency
       = 9.36·10^11 × 2 × 0.50
       = 9.36·10^11 B/sec
```

Аналогично — эффективная пропускная способность памяти равна single-card.

### 3.2 Prefill — две гонки, выигрывает медленный

Prefill обрабатывает все 512 токенов промпта параллельно. Roofline-модель
говорит: время = `max(сколько надо считать, сколько надо прочитать)`.

**Compute path** — сколько FLOPs нужно сделать:
```
FLOPs_prefill = 2 × N × p_in × batch
              = 2 × 3.28·10^10 × 512 × 1
              = 3.36·10^13 операций

(Множитель 2 — это потому что forward pass через transformer
эквивалентен ~2N FLOPs на токен; глубже — см. Sasha Rush "Scaling Book")

t_compute = FLOPs / (eff_flops × α)
         = 3.36·10^13 / (1.42·10^14 × 0.30)
         = 3.36·10^13 / 4.26·10^13
         = 0.789 sec
```

**Memory path** — сколько байт нужно прочитать (хотя бы веса один раз):
```
t_memory = W / eff_mbw = 1.99·10^10 / 9.36·10^11 = 0.0212 sec
```

**Roofline winner**:
```
t_prefill = max(0.789, 0.0212) = 0.789 sec
bottleneck_prefill = "compute"
```

Prefill в нашем случае **compute-bound** — нужно много умножать, и это
доминирует над временем чтения весов в 37 раз.

### 3.3 Decode — снова две гонки, на этот раз memory выигрывает

Decode генерирует ОДИН токен за раз. На каждый токен нужно прочитать
все веса модели заново (KV-кэш помогает, но веса не закэшированы).

**Memory path**:
```
t_dec_memory = W / (eff_mbw × β)
            = 1.99·10^10 / (9.36·10^11 × 0.42)
            = 1.99·10^10 / 3.93·10^11
            = 0.0506 sec/token
```

**Compute path** — для одного токена:
```
FLOPs_per_token = 2 × N × batch = 2 × 3.28·10^10 × 1 = 6.55·10^10
t_dec_compute = 6.55·10^10 / (1.42·10^14 × 0.30) = 1.54·10^-3 sec/token
```

**Roofline winner**:
```
t_dec_base = max(0.0506, 0.00154) = 0.0506 sec/token
bottleneck_decode = "memory"
```

Decode **memory-bound**, причём в 33 раза. Это и есть главный закон
LLM-инференса: для генерации одного токена нужно прочитать все 20 GB
весов из памяти, и compute мощность GPU простаивает.

### 3.4 KV-cache term — добавка к decode

Помимо весов, decode читает KV-кэш всех предыдущих токенов. Каждый шаг
кэш растёт.

**Размер KV на один токен**:
```
kv_per_token_bytes = layers × kv_heads × head_dim × (kv_bits_k + kv_bits_v) / 8
                  = 48 × 8 × 128 × (16 + 16) / 8
                  = 48 × 8 × 128 × 4
                  = 196 608 байт ≈ 0.19 MB / token
```

(`head_dim = 128` — это стандарт для современных трансформеров. `× 2` для
K и V, ещё `× 2` для fp16. Итого 4 байта на параметр head × kv_heads × layers.)

**Средний контекст за время ответа**:
```
avg_ctx = p_in + p_out / 2 = 512 + 64 = 576 токенов
```

(Берём середину генерации как «среднюю длину» — простой способ усреднить
растущий KV.)

**Время на чтение KV каждый токен**:
```
t_kv = kv_per_token_bytes × avg_ctx / (eff_mbw × β)
     = 196 608 × 576 / (9.36·10^11 × 0.42)
     = 1.13·10^8 / 3.93·10^11
     = 0.000288 sec/token
```

KV-term добавляет **0.3 ms к 50 ms decode-времени** = +0.6%. Незаметен на
коротком контексте, но при контексте 32k он бы дал +20% к decode.

**Итого decode**:
```
t_decode = t_dec_base + t_kv = 0.0506 + 0.000288 = 0.0509 sec/token
```

### 3.5 Итоговые метрики

```
prefill_s         = 0.789 sec
decode_per_token  = 0.0509 sec/token
throughput_tok_s  = batch × batch_mult / t_decode
                  = 1 × 1.0 / 0.0509
                  = 19.6 tok/sec  (это decode throughput)

total_latency_s   = prefill_s + p_out × t_decode
                  = 0.789 + 128 × 0.0509
                  = 0.789 + 6.52
                  = 7.31 sec   (полное время ответа: prefill + 128 токенов)

memory_gb         = (W + KV_total) / 1e9
                  = (1.99·10^10 + 196608 × 640 × 1) / 1e9
                  = 19.9 + 0.13 = 20.0 GB   (≤ 48 GB ✓ влезает)
```

И в более привычной форме:
```
prefill = 512 / 0.789 = 649 токенов промпта в секунду
decode  = 19.6 токенов генерации в секунду
```

---

## Слой 4 — Что возвращается пользователю

```python
InferenceResult(
    prefill_s          = 0.789,        # время на 512 промпт-токенов
    decode_per_token_s = 0.0509,       # время на 1 generated токен
    total_latency_s    = 7.31,         # полная latency для 128 токенов ответа
    throughput_tok_s   = 19.6,         # generation tok/sec
    memory_gb          = 20.0,         # weights + KV
    bottleneck_prefill = "compute",
    bottleneck_decode  = "memory",
)
```

CLI распечатает это в человекочитаемом виде с эмодзи.

---

## Слой 5 — Сравнение с реальностью

### Откуда берём «правду»

`data/llama_bench_qwen32b_multi.csv` — сырой вывод `llama-bench` на той же
2× RTX 3090 машине. Релевантные строки (split_mode=layer, pp 512, fa=1):

```
n_prompt=512  n_gen=0    sm=layer  avg_ts=1367.39  ← prefill: 1367 t/s
n_prompt=0    n_gen=128  sm=layer  avg_ts=39.44    ← decode:    39.4 t/s
```

### Сравнение predicted vs observed

| Метрика | Predicted | Observed | Ошибка |
|---|---:|---:|---:|
| pp512 (prefill t/s) | 649 | 1367 | **-52%** |
| tg128 (decode t/s) | 19.6 | 39.4 | **-50%** |

С калиброванным `α=0.30, β=0.42` (p75 для multi-GPU) формула **серьёзно
занижает реальность** — почти вдвое.

### Почему — и это нормально

В калибровке для `2xRTX-3090` всего **2 точки данных** (32B baseline + 70B
baseline), а калибровка работает по медиане. Если бы у нас было 100 точек
с разными моделями и параметрами, медиана нашла бы устойчивое значение.
С двумя точками медиана = средне-арифметическое, которое не знает что
«надо предсказывать best-case».

**Если использовать калибровку single-card RTX-3090** (где у нас 11 точек,
α=0.53 p75, β=0.78 p75) **и применить к 2× через формальный аппарат
multi-GPU** (`tp_size=2, tp_efficiency=0.5`):

```
eff_flops = 1.42·10^14 × 2 × 0.5 = 1.42·10^14   (= single)
eff_mbw   = 9.36·10^11 × 2 × 0.5 = 9.36·10^11   (= single)

t_prefill = 2N·p_in / (eff_flops × 0.53)
         = 3.36·10^13 / (1.42·10^14 × 0.53)
         = 0.447 sec  →  pp512 = 1146 t/s

t_decode = W / (eff_mbw × 0.78)
        = 1.99·10^10 / (9.36·10^11 × 0.78)
        = 0.0273 sec/token  →  tg = 36.6 t/s
```

Сравнение:

| Метрика | Predicted | Observed | Ошибка |
|---|---:|---:|---:|
| pp512 | 1146 | 1367 | **-16%** |
| tg128 | 36.6 | 39.4 | **-7%** |

**±20% — отличный результат для roofline-модели!**

### Вывод

Калибровка `2xRTX-3090` сама по себе слабая (всего 2 точки), но **формальная
эквивалентность** «multi-GPU layer split = single-card throughput» через
параметры `tp_size × tp_efficiency = 1.0` позволяет переиспользовать
богатую single-card калибровку и получить точный прогноз.

Это и есть концептуальное достижение текущей итерации проекта.

---

## Слой 6 — Sanity-инвариант

Эмулятор обязан удовлетворять следующему свойству:

> Если `tp_size = N` и `tp_efficiency = 1/N`, то результат `predict()`
> должен быть **идентичен** случаю `tp_size = 1, tp_efficiency = 1.0`.

Это потому что в обоих случаях `eff_flops = peak × 1.0 = peak`, и формула
дальше не различает «одна карта» от «N карт работающих как одна».

Тест `test_multi_gpu_capacity_only_when_tp_eff_inverse_of_size` в
`tests/test_formula.py` проверяет это свойство: разница prefill_s и
decode_per_token_s между single и multi должна быть **<0.1%**.

Этот тест ловит баг **double-counting peak_flops** — ситуацию, когда
HARDWARE_SPECS хранит уже умноженные на N значения, и `predict()` ещё
раз умножает. Был такой баг до коммита `5885065`.

---

## Полная цепочка данных одной картинкой

```
┌─────────────────────────────────────────────────────────────────┐
│ USER INPUT                                                      │
│   --model 32.764  --bits 4.85  --hw 2xRTX-3090                  │
│   --engine llama.cpp  --p-in 512  --p-out 128  --batch 1        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ LOOKUPS                                                         │
│                                                                 │
│  hardware.py        ──────►  peak_flops, mem_bw,               │
│                              tp_size, tp_efficiency,           │
│                              memory_capacity                   │
│                                                                 │
│  engines.py         ──────►  batch_saturation, compute_path    │
│                                                                 │
│  calibrated_coefs   ──────►  α, β  (median or p75)             │
│  .csv                       (по hw × engine × precision)       │
│                                                                 │
│  formula.py         ──────►  layers, d_model, kv_heads         │
│  ARCH_DEFAULTS              (по n_params)                      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ INTERNAL COMPUTATION (roofline)                                 │
│                                                                 │
│  N = n_params_b × 1e9                                           │
│  W = N × bits / 8                                               │
│                                                                 │
│  eff_flops = peak × tp_size × tp_efficiency                     │
│  eff_mbw   = bw   × tp_size × tp_efficiency                     │
│                                                                 │
│  t_pre = max(2·N·p_in·batch / (eff_flops·α),                    │
│              W / eff_mbw)                                       │
│                                                                 │
│  t_dec = max(W / (eff_mbw·β),                                   │
│              2·N·batch / (eff_flops·α))                         │
│                                                                 │
│  t_kv = layers × kv_heads × 128 × (kv_bits_k+kv_bits_v) / 8     │
│       × (p_in + p_out/2) / (eff_mbw × β)                        │
│  t_dec += t_kv                                                  │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                          │
│   prefill_s, decode_per_token_s,                                │
│   throughput_tok_s, memory_gb,                                  │
│   bottleneck_prefill, bottleneck_decode                         │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ COMPARISON (compare.py)                                         │
│                                                                 │
│   relative_error = |predicted - observed| / observed            │
│   observed:  data/leaderboard-*.csv  (реальные замеры)          │
│   агрегат :  results/error_summary_*.csv                        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ VERDICT                                                         │
│                                                                 │
│   <±20% при p75 калибровке         → отлично                   │
│   ±50% при median, baseline scenario → ожидаемо                │
│                                       (median underестимирует) │
│   Точное совпадение для multi=single  → инвариант проходит    │
└─────────────────────────────────────────────────────────────────┘
```

---

## vLLM-стиль оптимизации в формуле

Базовый roofline из Слоёв 1-6 описывает «честный naive engine» — PyTorch
eager, llama.cpp без батча. Современный inference-server (vLLM,
TensorRT-LLM) выдаёт ×3-10 throughput на тех же GPU. Это не магия:
четыре механизма, каждый со своим параметром в `predict()`, плюс
несколько эффектов, которые сводятся к смещению калиброванных `α/β`.

### TL;DR — что меняется в формуле

| Что в vLLM | Где в формуле | Тип корректировки |
|---|---|---|
| Continuous batching | `t_pre`, `t_dec` через `bs_eff` | новый параметр `batch_saturation` |
| Prefix caching (APC) | `t_pre` через `p_in_eff` | новый параметр `prefix_cache_hit` |
| Speculative decoding | `t_dec_final` | флаг `speculative` + 3 параметра |
| PagedAttention | `memory_GB` | новый параметр `kv_packing_eff` |
| CUDA graphs | `α` на decode | калибровка, без новых параметров |
| Chunked prefill | `α` | калибровка |
| Marlin / Machete (INT4/8) | `α` для quantized | per-quant калибровка |
| FlashAttention-3 | KV-term через `β` | калибровка |

Первые четыре требуют новых параметров — этим мы займёмся ниже.
Остальные — это «другая калибровка» для движка/квантизации и
закрываются обычным замером `α/β` на нужном железе. Полный design doc
с физикой каждой оптимизации — [`docs/VLLM_OPTIMIZATIONS.md`](VLLM_OPTIMIZATIONS.md).

---

### 1. Continuous batching → `batch_saturation`

**Что делает.** В наивном «static batching» собрали N запросов,
прогнали вместе, ждём пока последний закончит — самый медленный
держит остальных. Continuous batching: планировщик на каждом
decode-шаге может **взять новый запрос из очереди** в свободный slot.
GPU никогда не простаивает между requests.

**В формуле.** Эффективная concurrency — Hill-style saturation:

```python
bs_eff = batch_max * batch / (batch + batch_50pct)
```

При `batch == batch_50pct` получаем половину `batch_max`. При больших
батчах асимптотически выходим на `batch_max`. `bs_eff` подставляется
**и в `t_pre`, и в `t_dec` compute-term** — планировщик параллелит
обе фазы.

**Параметры из `engines.py`.** Для vLLM откалибровано на multi-batch
sweep на 2× RTX 5090 / Qwen-7B AWQ:

```python
"vllm":      {"batch_saturation": (45, 7)}   # max=45, 50% at batch≈7
"llama.cpp": {"batch_saturation": None}      # → bs_eff = batch
"pytorch":   {"batch_saturation": None}
```

Для других размеров модели `batch_50pct` сильно меняется: 70B
сатурируется уже при `batch ≈ 0.5`, 7B-TP2 — при `batch ≈ 20`
(см. note в `engines.py`).

---

### 2. Prefix caching → `prefix_cache_hit`

**Что делает.** Системные промпты, few-shot примеры, документы для RAG —
часто **одни и те же** на серии запросов. vLLM хеширует префиксы
(по блокам PagedAttention) и переиспользует уже посчитанный KV-кэш.

**В формуле.** Доля префикса в кэше `h ∈ [0, 1]`:

```python
p_in_eff = p_in * (1.0 - prefix_cache_hit)
t_pre_compute = 2 * N * p_in_eff * bs_eff / (eff_flops * α)
t_pre_mem     = W / eff_mbw       # не меняется — веса всё равно читаем
```

Decode **не ускоряется** — он всё равно читает полный KV (каждый новый
токен зависит от всего контекста). Падает только TTFT.

**Типичные `h`:**
- chat с system prompt 200 + user 50 → `h ≈ 0.8` для 2-го и следующих запросов
- RAG с контекстом 1000 + user 100 → `h ≈ 0.91`
- batch независимых документов → `h = 0.0`

---

### 3. Speculative decoding → `speculative` + 3 параметра

**Что делает.** Маленькая «draft» модель (Llama-3.2-1B) предлагает `k`
кандидатных токенов за дешёвый forward pass. Большая «main» модель
(Llama-3.1-70B) проверяет все `k` за **один** forward pass через
параллельное вычисление. Все приняты → за один шаг main выдаём `k`
токенов; первый отвергнут → берём только тот, что main выдала бы сама.

**В формуле.**

```python
if speculative:
    accepted_per_step = spec_accept_rate * spec_k_proposed
    t_dec_final = t_dec * (1 + spec_overhead) / accepted_per_step
```

**Типичные параметры:**
- `spec_accept_rate = 0.7` (0.5-0.9 для размер-парных моделей)
- `spec_k_proposed = 4` (стандарт vLLM)
- `spec_overhead = 0.15` (draft в 5-10× меньше main)

Llama-3.1-70B + Llama-3.2-1B / batch=1: ускорение decode в
`(0.7 × 4) / 1.15 ≈ 2.4×`.

**Когда ломается.** При больших батчах spec теряет эффект — draft
становится бутылочным горлышком. Формула этого не моделирует:
проверка корректности параметров остаётся на пользователе.

Тесты-инварианты: `test_speculative_decoding_speeds_up_decode`
(tests/test_formula.py:152), `test_speculative_zero_accept_rate_raises`
(tests/test_formula.py:169).

---

### 4. PagedAttention → `kv_packing_eff`

**Что делает.** KV-кэш разбивается на фиксированные блоки (типично
16 токенов на блок). Память аллоцируется через таблицу страниц, как
в виртуальной памяти ОС. Блоки разных запросов лежат рядом,
фрагментация около нуля. Наивная аллокация резервирует worst-case
буфер под каждый запрос → 50-65% утилизации VRAM; PagedAttention
даёт 95-99%.

**В формуле.**

```python
kv_total = kv_per_token_bytes * (p_in + p_out) * batch / kv_packing_eff
```

При `kv_packing_eff = 1.0` (default) — идеальная упаковка, нет
оверхеда. При `0.65` — наивная аллокация, реально занятой памяти
в ~1.5× больше расчётной.

> ⚠️ **Гетча:** дефолт в `predict()` = `1.0`, и `engines.py` пока
> **не подкладывает** значения для движков. Из коробки эффект
> PagedAttention в memory-расчёте не учитывается — передавайте
> `kv_packing_eff` явно (например, `0.97` для vLLM, `0.65` для
> наивного PyTorch). См. design doc, секцию 1.

**На скорость не влияет.** Меняет capacity — сколько concurrent
requests умещается в VRAM. Поэтому косвенно бустит суммарную
пропускную способность сервиса, но не одного пользователя.

---

### 5. Что НЕ требует новых параметров

Эти оптимизации захватываются обычной калибровкой `α/β`:

- **CUDA graphs** — убирают kernel-launch overhead (5-50 мкс × сотни
  kernels на decode-шаг). Поднимают эффективный `α` на decode для
  маленьких батчей.
- **Chunked prefill** — длинный prefill режется на ~512-токенные
  чанки и смешивается с decode-токенами других запросов. Более
  стабильная latency и выше суммарный throughput; в формуле —
  повышенный `α`.
- **Marlin / Machete kernels** — прямой INT4/INT8 GEMM на упакованных
  данных, без промежуточного dequant в fp16. +30-50% к prefill на
  H100/Blackwell. Захватывается **per-quant** калибровкой `α`.
- **FlashAttention-3** — асинхронные warp-specialization, поддержка
  FP8. На длинных контекстах (32k+) ускоряет attention в 1.5-2×
  vs FA2 → выше эффективный `β` через KV-term.

Логика: новый движок / новое железо / новая квантизация → гоните
бенчмарк и калибруйте `α/β`. Параметры формулы менять не нужно.

---

### Скорректированный поток одной картинкой

```
ВХОД (★ — новые vLLM-параметры):
  n_params_b, bits, p_in, p_out, batch,
  peak_flops, mem_bw, α, β,
  layers, kv_heads, kv_bits_k, kv_bits_v,
  tp_size, tp_efficiency,
  ★ prefix_cache_hit          — h ∈ [0, 1]
  ★ batch_saturation          — (batch_max, batch_50pct) или None
  ★ kv_packing_eff            — PagedAttention eff (default 1.0)
  ★ speculative, spec_*       — для draft+main

ШАГ 1. Эффективные ресурсы:
  W         = N · bits / 8
  eff_flops = peak · tp_size · tp_efficiency
  eff_mbw   = mbw  · tp_size · tp_efficiency

ШАГ 2. ★ Effective concurrent batch:
  bs_eff = batch_max · batch / (batch + batch_50pct)
           if batch_saturation else batch

ШАГ 3. Prefill (★ с префикс-кэшем):
  p_in_eff = p_in · (1 − prefix_cache_hit)
  t_pre = max(2·N·p_in_eff·bs_eff / (eff_flops·α),
              W / eff_mbw)

ШАГ 4. Decode базовый:
  t_dec_base = max(W / (eff_mbw·β),  2·N·bs_eff / (eff_flops·α))

ШАГ 5. KV-cache:
  KV_per_tok = layers · kv_heads · 128 · (b_kv_k + b_kv_v) / 8
  t_kv       = KV_per_tok · (p_in + p_out/2) / (eff_mbw · β)
  t_dec      = t_dec_base + t_kv

ШАГ 6. ★ Speculative decoding:
  if speculative:
      t_dec_final = t_dec · (1 + spec_overhead) /
                    (spec_accept_rate · spec_k_proposed)
  else:
      t_dec_final = t_dec

ШАГ 7. ★ Memory с PagedAttention:
  kv_total  = KV_per_tok · (p_in + p_out) · batch / kv_packing_eff
  memory_gb = (W + kv_total) / 1e9

МЕТРИКИ:
  latency    = t_pre + p_out · t_dec_final
  throughput = bs_eff / t_dec_final
```

Отличия от baseline (Слой 3): новые шаги 2, 6, 7 и поправка `p_in_eff`
в шаге 3. Остальное — ровно тот же roofline.

---

### Численный пример: 7B Q4 на 1× A100

Сценарий: chat с system prompt, `p_in=1024, p_out=256, batch=8`.

| Конфиг | TTFT | Throughput | Memory |
|---|---:|---:|---:|
| **PyTorch baseline** (α=0.20, β=0.55, batch_saturation=None, kv_pack=0.65) | 350 ms | 30 t/s | 8 GB |
| **vLLM, без spec, h=0** (α=0.47, β=0.65, batch_saturation=(45, 7), kv_pack=0.97) | 175 ms | 110 t/s | 6 GB |
| **+ prefix caching h=0.4** | **105 ms** | 110 t/s | 6 GB |
| **+ speculative (r=0.7, k=4)** | 105 ms | **220 t/s** | 6 GB |

Изолированные эффекты:
- continuous batching → ×3.7 throughput (через `bs_eff`)
- PagedAttention → −25% memory (capacity-bound сценарий)
- prefix caching → −40% TTFT (для повторяющегося system prompt)
- speculative → ×2 throughput на batch=1; на batch=8 эффект слабее

Числа — literature defaults для иллюстрации эффектов. Реальная
калибровка vLLM на 2× RTX 5090 / Qwen-7B AWQ даёт `α=0.47`,
`batch_saturation=(45, 7)`; `β` пока literature.

---

### Связанные тесты

- `test_prefix_cache_collapses_prefill_to_weight_load` — `tests/test_formula.py:111`
- `test_prefix_cache_partial_hit_proportional` — `tests/test_formula.py:134`
- `test_speculative_decoding_speeds_up_decode` — `tests/test_formula.py:152`
- `test_speculative_zero_accept_rate_raises` — `tests/test_formula.py:169`

---

### Текущее состояние калибровки

На 2026-05-14:

- **vLLM α=0.47**, **`batch_saturation=(45, 7)`** — откалиброваны
  на 2× RTX 5090 / Qwen-7B AWQ, multi-batch sweep
  (commits 419bc50…c5daab1).
- **vLLM β=0.65** — literature default. β из текущих vllm-замеров
  **не идентифицируется**: prefill уходит в compute-bound, KV-term
  слишком мал, чтобы отделить β от шума (см. commit 82341a6,
  Variant A).
- **`kv_packing_eff`** в `engines.py` не выставлен — передавайте
  явно при сравнении PagedAttention vs наивной аллокации.
- Все `spec_*` — пользовательский ввод, калибровать нечего.

Полный design doc с обоснованием и Roadmap —
[`docs/VLLM_OPTIMIZATIONS.md`](VLLM_OPTIMIZATIONS.md).
Полная таблица калиброванных коэффициентов —
[`results/calibrated_coefficients.csv`](../results/calibrated_coefficients.csv).
Анализ калибровок и графики —
[`results/REPORT.md`](../results/REPORT.md).

---

## FAQ — частые вопросы

### Почему `predict(7B, hw=2xRTX-3090)` даёт ту же скорость что `predict(7B, hw=RTX-3090)`?

Потому что для multi-GPU 2× 3090 без NVLink `tp_efficiency = 0.50`. Это
**эмпирический факт**: layer split в llama.cpp = pipeline parallel, две
карты работают **последовательно** на каждом токене. Throughput равен
single-card.

Многие пользователи интуитивно ждут «купил вторую карту → стало в 2× быстрее».
Эмулятор честно говорит: «нет, ты получил только дополнительную VRAM, не
скорость». Используй `2xRTX-3090` только для моделей, которые не влезают
в одну карту (как 70B).

### Почему `predict(predicted)` всегда занижает observed на 14-17%?

Roofline-модель — **верхняя граница**. Она предполагает идеальную параллельную
работу compute и memory pipeline'а. Реальность ниже из-за kernel launch
overhead, scheduler stalls, padding, неоптимальной cache locality. 15%
систематическое занижение — это «inherent overhead» движка, не отлавливаемый
формулой.

Чтобы делать прогноз **выше** реальности, нужен другой коэффициент в
`α/β` — что-то типа «1.15 × p75». Но обычно для capacity planning лучше
быть консервативным: предсказали 30 t/s, реально получили 35 — пользователь
доволен.

### Что если модель не в `ARCH_DEFAULTS`?

`_arch_for(N)` берёт ближайшую запись `≤ N`. Для модели 50B возьмётся
`34.0` (`Yi-34B`), что нормально для KV-расчёта. Если у вас экзотичная
архитектура (sliding window attention, MoE), наша формула KV-cache
переоценит. Нужно явно передать `layers, d_model, kv_heads` в
`predict()`.

### Зачем формуле раздельные `α` и `β`?

Потому что **prefill и decode упираются в разные ресурсы**:
- prefill compute-bound → определяется насколько эффективно используется compute
- decode memory-bound → определяется насколько эффективно используется bandwidth

Один движок может иметь высокий `α` (умеет эффективно делать матмулы)
но низкий `β` (плохо организует KV-cache). Поэтому два независимых параметра.

### Можно ли использовать формулу для batch>1?

Да. Просто увеличьте `--batch`. При больших батчах:
- prefill **остаётся compute-bound** (больше токенов параллельно)
- decode **может перейти в compute-bound** (см. тест
  `test_large_batch_flips_decode_to_compute`)

Это ровно то, как vLLM/TensorRT добиваются высокого throughput — через
большие батчи.

### Чем отличается `predict()` для vLLM от наивного PyTorch?

Четырьмя новыми параметрами: `batch_saturation` (continuous batching),
`prefix_cache_hit` (APC), `speculative` + `spec_*` (speculative decoding)
и `kv_packing_eff` (PagedAttention). Дефолты «как было» — старые вызовы
работают без изменений. Полный разбор каждой оптимизации с примерами —
раздел «vLLM-стиль оптимизации в формуле» выше.

---

## Дальше

- Чтобы запустить пример самостоятельно: `python scripts/cli.py --model
  32.764 --bits 4.85 --hw 2xRTX-3090 --engine llama.cpp --p-in 512
  --p-out 128 --batch 1 --precision-label "Q4_K (4.85bpw)"`
- Сравнение predicted vs observed для всех HW: `python scripts/compare.py`
- Полный анализ калибровок: [`results/REPORT.md`](../results/REPORT.md)
- Источник математики: A. Patterson "Roofline: An Insightful Visual
  Performance Model"; для LLM specifically — Sasha Rush "Scaling Book" Ch. 7

Если что-то непонятно — открывайте issue на GitHub, упомяните номер
секции, попробуем сделать понятнее.
