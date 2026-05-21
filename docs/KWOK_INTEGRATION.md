# Интеграция с kwok — K8s sandbox для GPU-нод

> Design doc для проекта, который превращает roofline-эмулятор в источник
> реалистичных performance-чисел внутри симулированного Kubernetes-кластера.
> Цель: изучать поведение autoscaling / scheduling / multi-tenant GPU
> sharing без аренды реального железа.

**Статус**: draft (2026-05-21). Код не написан, обсуждаем архитектуру.

---

## Зачем

Эмулятор отвечает на вопрос «сколько tok/s даст модель X на железе Y».
Но реальный LLM-инференс деплоится **внутри Kubernetes**: Deployment'ы,
HPA, KEDA, Karpenter, GPU Operator, KServe, KubeRay. Эти слои привносят
своё поведение поверх голого GPU — и **их тестирование без реальных
H100/A100 сейчас невозможно**.

`kwok` ([Kubernetes WithOut Kubelet](https://kwok.sigs.k8s.io/)) — это
симулятор нод/подов: регистрирует fake-nodes в kube-api-server, обрабатывает
поды через configurable lifecycle stages (CRD `Stage`). Реальных
контейнеров нет, реального компьюта нет, но **K8s control plane не
отличает** фейковую ноду от настоящей.

Совмещение даёт:

| Эмулятор сам по себе | kwok сам по себе | Эмулятор + kwok |
|---|---|---|
| «Q4 7B на A100: 4322 tok/s» | «1000 fake-нод, scheduler не задыхается» | «HPA на vLLM-Deployment с 50 репликами на 5 H100-нодах правильно scale'нет вверх когда burst пришёл?» |
| CLI / Streamlit / Python lib | kubectl / манифесты | Helm chart → kubectl apply → реальная картина |

**Главный use case (выбран как первый)**: autoscaling.
- HPA на `vllm:request_success_rate`
- KEDA scaler на queue depth / custom metric
- Karpenter / Cluster-Autoscaler по node-pool с GPU-типами
- Combined: scale-up / scale-down под bursty inference traffic

Остальные use cases — после autoscaling:
- Multi-tenant GPU sharing (time-slicing, MPS, MIG)
- GPU Operator failure modes (DCGM gap, device-plugin restart)
- Scheduler / placement (binpacking H100 vs A100 vs B200 nodes)

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                  Kubernetes API server (real)                   │
│         (стандартный kube-apiserver + etcd, без kubelet'ов)     │
└─────────────────────────────────────────────────────────────────┘
        ▲                ▲                ▲                  ▲
        │                │                │                  │
   ┌────┴───┐      ┌─────┴────┐    ┌──────┴──────┐    ┌──────┴────┐
   │  kwok  │      │ emulator │    │ DCGM-fake   │    │  vLLM-    │
   │ nodes  │      │   pod    │    │  exporter   │    │  metrics  │
   │ (fake  │      │controller│    │             │    │  exporter │
   │ H100/  │      │  (NEW)   │    │   (NEW)     │    │   (NEW)   │
   │ A100/  │      │          │    │             │    │           │
   │ ...)   │      │ formula. │    │ "GPU util", │    │  tok/s,   │
   └────────┘      │ predict()│    │ "FB used"   │    │ latency   │
                   └──────────┘    └─────────────┘    └───────────┘
                          │                ▲                  ▲
                          │                │                  │
                          └──────► aggregated per-node metrics ─┘
                                          ▼
                              ┌────────────────────────┐
                              │ Prometheus + Grafana   │
                              │ (kube-prometheus stack)│
                              └────────────────────────┘
                                          ▼
                              ┌────────────────────────┐
                              │ HPA / KEDA / Karpenter │
                              │  (стандартные, без     │
                              │   модификаций)         │
                              └────────────────────────┘
```

### Компоненты (что нужно построить)

**1. Fake GPU nodes (kwok-side)**

YAML-манифесты для kwok, регистрирующие ноды с реалистичными labels
и capacity:

```yaml
apiVersion: v1
kind: Node
metadata:
  name: gpu-h100-01
  labels:
    type: kwok
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
    nvidia.com/gpu.memory: "81920"
    nvidia.com/gpu.count: "8"
    node.kubernetes.io/instance-type: "p5.48xlarge"  # AWS-equivalent
spec:
  taints:
    - key: nvidia.com/gpu
      effect: NoSchedule
status:
  capacity:
    cpu: "192"
    memory: 2048Gi
    nvidia.com/gpu: "8"
  allocatable:
    nvidia.com/gpu: "8"
```

Шаблоны на: 1×H100, 8×H100, 1×A100-80, 4×A100-40, 2×RTX-5090, 1×B200,
8×MI300X. **Берём из `HARDWARE_SPECS`** — single source of truth.

**2. Emulator pod controller (NEW)**

Python service (kopf-style operator или client-go), который:
- Watch'ит pods с label `llm-emulator/workload=true`
- На pod CREATE парсит annotations:
  ```yaml
  llm-emulator/model-b: "7"
  llm-emulator/engine: "vllm"
  llm-emulator/bits: "4"
  llm-emulator/p-in: "1024"
  llm-emulator/p-out: "128"
  llm-emulator/batch: "8"
  llm-emulator/precision: "AWQ.4bit"
  ```
- Извлекает HW из node labels (`nvidia.com/gpu.product`)
- Зовёт `emulator.formula.predict(...)`
- Патчит pod annotations с результатами:
  ```yaml
  llm-emulator/predicted-throughput-tok-s: "4322"
  llm-emulator/predicted-prefill-ms: "230"
  llm-emulator/predicted-memory-gb: "15.3"
  llm-emulator/predicted-decode-ms-per-tok: "1.85"
  ```
- Обновляет shared state (per-node aggregate) для exporter'ов

**3. DCGM-compatible Prometheus exporter (NEW)**

HTTP endpoint `:9400/metrics` с метриками которые `kube-prometheus` уже
умеет парсить:
```
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="...",Hostname="gpu-h100-01"} 87.3
DCGM_FI_DEV_FB_USED{gpu="0",...} 15300
DCGM_FI_DEV_FB_FREE{gpu="0",...} 66620
DCGM_FI_DEV_POWER_USAGE{gpu="0",...} 542.1
```

Значения **вычисляются из running pods на ноде**:
- `FB_USED` = Σ predicted-memory по подам
- `GPU_UTIL` = `min(100, Σ predicted-tok-s / peak-tok-s × 100)`
- `POWER_USAGE` = `tdp_w × util_fraction`

**4. vLLM-compatible metrics exporter (NEW)**

Аналогично, но для pod-уровневых метрик которые HPA/KEDA читают:
```
vllm:request_success_total{model="...",pod="..."} <counter>
vllm:e2e_request_latency_seconds{...} <gauge>
vllm:gpu_cache_usage_perc{...} <gauge>
vllm:num_requests_running{...} <gauge>
vllm:num_requests_waiting{...} <gauge>
```

Это позволяет писать **стандартные** HPA-манифесты:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef: { name: my-vllm }
  metrics:
    - type: Pods
      pods:
        metric: { name: vllm_request_success_rate }
        target: { type: AverageValue, averageValue: 100 }
```

без знания что под капотом — fake.

### Что НЕ строим (на этой фазе)

- Свой scheduler / scheduler-plugin. Default scheduler + extended resources
  делают всё что нужно.
- Custom CRD `LLMWorkload`. Стандартный Deployment + аннотации
  достаточно для autoscaling-экспериментов.
- Реальные контейнеры. kwok stages эмулируют lifecycle без процессов.
- Network simulation между подами. Эмулятор уже моделирует TP/PP
  comm на уровне формулы — на K8s-уровне это не нужно.

---

## Workflow эксперимента (пример autoscaling-сценария)

**Сценарий**: «У меня 5 H100-нод (40 GPU суммарно), Deployment с vLLM
Qwen-7B-AWQ, batch=50. Нагрузка варьируется от 100 до 10000 запросов/мин.
Хочу understand: KEDA с правильным scaling правильно scale'нет реплики?
Не перебежит ли cluster-autoscaler?»

**Шаги**:

1. Поднять kwok cluster:
   ```bash
   kwokctl create cluster --name=gpu-sim --runtime=binary
   kubectl apply -f deploy/kwok/nodes/h100-pool-5x.yaml   # 5 H100 нод
   ```

2. Развернуть emulator-pod-controller + exporters:
   ```bash
   kubectl apply -f deploy/emulator-controller.yaml
   kubectl apply -f deploy/dcgm-fake-exporter.yaml
   kubectl apply -f deploy/vllm-metrics-exporter.yaml
   ```

3. Развернуть kube-prometheus + KEDA:
   ```bash
   helm install prom prometheus-community/kube-prometheus-stack
   helm install keda kedacore/keda
   ```

4. Запустить vLLM-Deployment + ScaledObject:
   ```bash
   kubectl apply -f examples/qwen7b-vllm/deployment.yaml
   kubectl apply -f examples/qwen7b-vllm/keda-scaledobject.yaml
   ```

5. Запустить traffic-генератор (тоже kwok-pod):
   ```bash
   kubectl apply -f examples/traffic-burst.yaml   # 100→10000 req/min
   ```

6. Наблюдать в Grafana:
   - Сколько реплик KEDA создал на пике
   - Где они сели (binpack по нодам или spread?)
   - Какие predicted-throughput / predicted-memory выдал эмулятор для
     каждой реплики
   - Когда нагрузка спала — scale-down policy сработала?

**Цель — найти**: edge cases где K8s primitives «врут» о готовности
кластера: HPA scaled up но новый pod упёрся в VRAM existing pod'ов;
Karpenter выбрал не тот instance-type; KEDA cooldown слишком длинный
для bursty traffic.

---

## Фазы реализации

### Фаза 0 — Bootstrap (1 день)

- `kwokctl create cluster`, проверить `kubectl get nodes` показывает 0 нод.
- Применить 1 fake-node manifest вручную, убедиться что `kubectl describe
  node` показывает правильные GPU-labels.
- Развернуть тестовый Deployment без эмулятора, убедиться что pods
  переходят в `Running` через kwok stages.

**Deliverable**: `deploy/kwok/00-cluster-up.sh` + `deploy/kwok/nodes/`.

### Фаза 1 — Emulator controller skeleton (2-3 дня)

- Python kopf-controller, который watch'ит pods, парсит аннотации,
  зовёт `emulator.formula.predict()`, патчит pod annotations.
- Не нужны exporters пока — просто proof что моделирование работает
  внутри K8s контекста.
- Тесты: `pytest tests/test_controller.py` с fake-API.

**Deliverable**: `controller/emulator_controller.py` + Dockerfile + Deployment.

### Фаза 2 — Metrics exporters (3-4 дня)

- DCGM-fake exporter (FastAPI, port 9400)
- vLLM-metrics exporter (port 8000, vLLM-compatible endpoint)
- Аггрегация per-node / per-pod из shared state контроллера
- ServiceMonitor манифесты для kube-prometheus

**Deliverable**: `exporters/` + дашборды Grafana.

### Фаза 3 — KEDA autoscaling experiment (2-3 дня)

- Helm chart `llm-emulator/vllm-mock` — Deployment + Service +
  ScaledObject + traffic generator
- Сценарий: burst traffic → KEDA scale up → cooldown → scale down
- Отчёт: что увидели, какие edge cases поймали

**Deliverable**: `examples/autoscaling-keda/` + `docs/KWOK_KEDA_REPORT.md`.

### Фаза 4+ — Multi-tenant / MIG / failures (TBD)

После Фазы 3 переоценим что важнее.

---

## Где живёт код

**Решено (2026-05-22)**: отдельный репозиторий
[`dedanimalfarm/llm-cluster-sandbox`](https://github.com/dedanimalfarm/llm-cluster-sandbox)
(приватный). emu-local остаётся CLI/lib, импортируется как dependency.

Структура:
```
llm-cluster-sandbox/
├── cluster/         # Фаза 0 — kwokctl bootstrap + 5 fake-node манифестов + smoke
├── controller/      # Фаза 1 — Python pod controller, зовёт emulator.predict()
├── exporters/       # Фаза 2 — DCGM-fake + vLLM-metrics Prometheus exporters
└── examples/        # Фаза 3 — KEDA autoscaling эксперимент
```

Связь: `pip install -e ../emu-local` (dev) или
`pip install emulator @ git+https://...@v0.X.Y` (pinned, когда эмулятор
будет тегирован).

Причина выделения сразу (а не позже): разные деплой-юниты (Python lib vs
K8s controllers + manifests + Helm charts), разные CI workflow'ы
(emu-local гоняет `pytest`-стиль unit-tests, sandbox понадобится
kwokctl-setup в Actions), разные релизные циклы.

---

## Открытые вопросы

1. **Python kopf vs Go client-go для controller?** Kopf проще
   написать (Python + наш эмулятор уже Python), но медленнее на 1000+
   подах. Для PoC — kopf. Для production-grade — переписать.

2. **Где параметры workload?** Сейчас предложил pod annotations.
   Альтернативы: container env vars, ConfigMap. Annotations — самое
   K8s-native, видны в `kubectl describe`, не требуют restart pod'a
   при изменении.

3. **Engine variety**: эмулятор поддерживает vllm / llama.cpp / pytorch /
   TGI / SGLang. Какие engines симулируем в первую очередь?
   Probably **vllm only для Фазы 3** — потому что vLLM metrics
   format — реальный стандарт, и autoscaling-эксперименты крутятся
   вокруг него.

4. **Cluster fan-out — реалистично?** Real H100-кластеры — 8-128
   нод. Наш kwok должен спокойно держать 100+ fake-нод и 1000+ pods
   без тормозов. **Стресс-тест нужен в Фазе 0**.

5. **GPU sharing (MIG / time-slicing)**: моделируем как «один pod на
   один GPU» в Фазе 1-3? Multi-tenancy откладываем на Фазу 4? Или
   сразу делаем MIG-aware controller? **Откладываем на Фазу 4.**

6. **vLLM-metrics: какой conformance level?** Минимум — 4-5 ключевых
   метрик которые HPA/KEDA реально читают. Полный набор vLLM экспортирует
   ~50 метрик, не все нужны.

7. **Замеры производительности**: насколько kwok + наш контроллер
   масштабируется при 1000 pods / 100 nodes? Контроллер на kopf
   делает sync per-pod-event — нужна batching/queue, или нет?

---

## Связь с эмулятором

Эмулятор остаётся **источником истины** по GPU-performance:
- Изменения формулы → автоматически попадают в K8s-симуляцию
  (controller re-imports modul)
- Новое железо в `HARDWARE_SPECS` → автоматически доступно как
  fake-node template
- Калибровочные коэффициенты `results/calibrated_coefficients.csv` →
  используются controller'ом без перекомпиляции

Что меняем в эмуляторе под эту интеграцию: ничего на старте.
Если controller сильно бьёт по `predict()` performance — добавим
`predict_batch(list_of_configs)` для амортизации Python overhead'а.
Это уже Фаза 2+.

---

## Что дальше

После apply этого design doc и согласия — переходим к **Фазе 0**:
`kwokctl create cluster`, 5 fake-node манифестов, smoke-test что
`kubectl get nodes -L nvidia.com/gpu.product` показывает правильное.

Если архитектура устраивает — ставлю task'и:
- `docs/KWOK_INTEGRATION.md` — этот документ (merged)
- `deploy/kwok/` — kwok setup scripts
- `controller/` — emulator pod controller
- `exporters/` — DCGM + vLLM metrics
- `examples/autoscaling-keda/` — первый эксперимент
