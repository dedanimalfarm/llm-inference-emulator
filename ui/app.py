"""LLM Inference Emulator — interactive Streamlit dashboard.

Wraps `emulator.formula.predict()` with a UI organised around a single
question: "Will my model run on this hardware, and how fast?"

Layout:
- Sidebar: persistent configuration (HW × engine × model × bits) — affects
  every tab. Preset buttons jump-start common scenarios.
- Tabs: Start (что это), Эмуляция (predictor), Формула, Каталог (presets),
  Калибровка, Валидация, Predicted-vs-Actual.

Run:
    streamlit run ui/app.py             # bare metal
    docker compose up                   # containerised, http://localhost:8501
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emulator.engines import ENGINE_DEFAULTS
from emulator.formula import ARCH_DEFAULTS, _arch_for, predict
from emulator.hardware import HARDWARE_SPECS, get_memory_bandwidth, get_peak_compute
from ui.validation_data import VALIDATION_POINTS

CAL_PATH = REPO_ROOT / "results" / "calibrated_coefficients.csv"
PVA_PATH = REPO_ROOT / "results" / "prediction_vs_actual.csv"
ERR_PATH = REPO_ROOT / "results" / "error_summary_by_hw.csv"

st.set_page_config(
    page_title="LLM Inference Emulator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_calibration() -> pd.DataFrame:
    return pd.read_csv(CAL_PATH) if CAL_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_pred_vs_actual() -> pd.DataFrame:
    return pd.read_csv(PVA_PATH) if PVA_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_error_summary() -> pd.DataFrame:
    return pd.read_csv(ERR_PATH) if ERR_PATH.exists() else pd.DataFrame()


def lookup_calibration(hw: str, engine: str, precision: str):
    """Return (alpha, beta, batch_sat, n_rows) for a (HW, engine, precision)
    bucket, or None if the bucket has no calibration."""
    df = load_calibration()
    if df.empty:
        return None
    sub = df[(df["hw"] == hw) & (df["backend"] == engine) & (df["precision_label"] == precision)]
    if sub.empty:
        return None
    r = sub.iloc[0]
    alpha = float(r["alpha_median"]) if pd.notna(r["alpha_median"]) else None
    beta = float(r["beta_median"]) if pd.notna(r["beta_median"]) else None
    bs = None
    if "batch_max" in r and pd.notna(r["batch_max"]):
        bs = (float(r["batch_max"]), float(r["batch_50pct"]))
    return {"alpha": alpha, "beta": beta, "batch_saturation": bs,
            "n_rows": int(r["n_rows"])}


# ---------------------------------------------------------------------------
# Presets — one click to fill the whole sidebar
# ---------------------------------------------------------------------------
PRESETS = {
    "🤖 7B чатбот / A100":      dict(hw="1xA100",     engine="vllm",      model_b=7.0,   bits=4,  precision="AWQ.4bit"),
    "🐘 70B Q4 / 2×3090":       dict(hw="2xRTX-3090", engine="llama.cpp", model_b=70.0,  bits=4,  precision="Q4_K (4.85bpw)"),
    "🧠 Mixtral MoE / 1×A100":  dict(hw="1xA100",     engine="vllm",      model_b=46.7, bits=4,  precision="AWQ.4bit"),
    "⚡ Self-hosted / 2×5090":  dict(hw="2xRTX-5090", engine="vllm",      model_b=7.0,   bits=4,  precision="AWQ.4bit"),
    "🔬 DeepSeek-V3 / H100":    dict(hw="1xH100",     engine="vllm",      model_b=671.0, bits=4,  precision="AWQ.4bit"),
    "💾 Llama-8B BF16 / H100":  dict(hw="1xH100",     engine="vllm",      model_b=8.0,   bits=16, precision="Unquantized"),
}


def apply_preset(name: str) -> None:
    p = PRESETS[name]
    st.session_state["sb_hw"] = p["hw"]
    st.session_state["sb_engine"] = p["engine"]
    st.session_state["sb_model"] = p["model_b"]
    st.session_state["sb_bits"] = p["bits"]
    st.session_state["sb_precision"] = p["precision"]
    st.session_state["sb_preset_active"] = name


# ---------------------------------------------------------------------------
# Session-state defaults (must run before widgets read these keys)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "ui_mode": "Базовый (Simple)",
    "sb_hw": "1xA100",
    "sb_engine": "vllm",
    "sb_model": 7.0,
    "sb_bits": 16,
    "sb_precision": "Unquantized",
    "sb_preset_active": None,
    "sb_custom_model_arch": False,
    "sb_custom_b": 0.0,
    "sb_custom_active": 0.0,
    "sb_preset_selector": "Индивидуальная настройка (Manual)",
    # workload (Эмуляция tab)
    "wk_p_in": 256,
    "wk_p_out": 64,
    "wk_batch": 8,
    # advanced (Эмуляция tab — expander inside form)
    "adv_kv_k": 16.0,
    "adv_kv_v": 16.0,
    "adv_sw": 0,
    "adv_hd": 128,
    "adv_cost": 0.0,
    "adv_oa": False,
    "adv_alpha": 0.2,
    "adv_ob": False,
    "adv_beta": 0.4,
    "adv_alpha_sat": 0.0,
    "adv_mla_rank": 0,
    "adv_mla_rope": 0,
    "adv_pp": 1,
    "adv_comm_link": "Auto",
    "adv_comm_bw": 0.0,
    "adv_comm_lat": 0.0,
    # snapshot of all inputs captured on last "Predict" press; None = boot state
    "pred_snapshot": None,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# Sidebar — persistent configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎯 Конфигурация")
    
    st.selectbox(
        "🎓 Режим интерфейса",
        ["Базовый (Simple)", "🎓 Учебный (Pedagogical)", "⚙️ Продвинутый (Advanced)"],
        key="ui_mode",
        help="Базовый — скрывает сложные настройки и формулы. Учебный — показывает пошаговый разбор формул с вашими числами, график Roofline и жизненный цикл токена. Продвинутый — полный доступ к тонким параметрам и экспертной аналитике."
    )
    
    st.divider()

    st.markdown("**📌 Готовые сценарии**")
    preset_names = ["Индивидуальная настройка (Manual)"] + list(PRESETS.keys())
    active_preset = st.session_state.get("sb_preset_active")
    default_idx = preset_names.index(active_preset) if active_preset in PRESETS else 0
    
    def on_preset_change():
        chosen = st.session_state["sb_preset_selector"]
        if chosen == "Индивидуальная настройка (Manual)":
            st.session_state["sb_preset_active"] = None
        else:
            apply_preset(chosen)
            
    st.selectbox(
        "Выберите готовый сценарий",
        preset_names,
        index=default_idx,
        key="sb_preset_selector",
        on_change=on_preset_change,
        help="Выбор готового сценария автоматически заполнит поля железа, движка и модели."
    )

    st.divider()

    st.markdown("**🛠 Ручная настройка:**")
    st.selectbox("Железо", list(HARDWARE_SPECS.keys()), key="sb_hw")
    st.selectbox("Движок", list(ENGINE_DEFAULTS.keys()), key="sb_engine")

    arch_keys = sorted(ARCH_DEFAULTS.keys())

    def _fmt_model(k: float) -> str:
        a = ARCH_DEFAULTS[k]
        moe = f"  • MoE active={a['n_active_b']}B" if "n_active_b" in a else ""
        return f"{k:>6.1f}B{moe}"

    st.selectbox("Модель (preset, B)", arch_keys, key="sb_model",
                 format_func=_fmt_model)
    st.radio("Bits весов", [4, 8, 16], key="sb_bits", horizontal=True)

    with st.expander("⚙️ Дополнительно"):
        st.text_input("Precision label (для калибровки)", key="sb_precision",
                      help="Используется при поиске α/β в results/calibrated_coefficients.csv.")
        st.checkbox("Своя архитектура модели", key="sb_custom_model_arch",
                    help="Позволяет вручную задать размер параметров и число активных параметров MoE.")
        if st.session_state.get("sb_custom_model_arch", False):
            st.number_input("Custom size (B), 0 = use preset", min_value=0.0,
                            max_value=2000.0, value=0.0, step=0.5, key="sb_custom_b",
                            help="Перебивает preset — полезно для нестандартных моделей.")
            st.number_input("Active params (B), MoE override (0 = preset)",
                            min_value=0.0, max_value=2000.0, value=0.0, step=0.5,
                            key="sb_custom_active",
                            help="Для MoE моделей — количество активных параметров на токен.")

    st.divider()
    hw_spec = HARDWARE_SPECS[st.session_state["sb_hw"]]
    eng_spec = ENGINE_DEFAULTS[st.session_state["sb_engine"]]
    st.caption(
        f"💻 **{st.session_state['sb_hw']}** — "
        f"{hw_spec['peak_tflops'][16]:.0f} TFLOPS BF16, "
        f"{hw_spec['memory_bandwidth_gbs']:.0f} GB/s, "
        f"{hw_spec['memory_capacity_gb']:.0f} GB VRAM"
    )
    st.caption(
        f"🔧 **{st.session_state['sb_engine']}** — "
        f"α={eng_spec['alpha']:.2f} (default), β={eng_spec['beta']:.2f} (default)"
    )


# ---------------------------------------------------------------------------
# Helper: compute current effective config (resolves Custom overrides)
# ---------------------------------------------------------------------------
def current_config() -> dict:
    s = st.session_state
    n_b = s["sb_custom_b"] if s["sb_custom_b"] > 0 else s["sb_model"]
    active = s["sb_custom_active"] if s["sb_custom_active"] > 0 else None
    return {
        "hw": s["sb_hw"],
        "engine": s["sb_engine"],
        "n_params_b": float(n_b),
        "bits": int(s["sb_bits"]),
        "active_b": active,
        "precision": s["sb_precision"],
    }


def build_pred_inputs() -> dict:
    """Snapshot every input that feeds predict() — sidebar + form fields.

    Used both for capturing on Predict press and for staleness detection
    (compare to pred_snapshot on subsequent reruns).
    """
    s = st.session_state
    cfg = current_config()
    return {
        "hw": cfg["hw"],
        "engine": cfg["engine"],
        "n_params_b": cfg["n_params_b"],
        "bits": cfg["bits"],
        "active_b": cfg["active_b"],
        "precision": cfg["precision"],
        "p_in": int(s.get("wk_p_in", DEFAULTS["wk_p_in"])),
        "p_out": int(s.get("wk_p_out", DEFAULTS["wk_p_out"])),
        "batch": int(s.get("wk_batch", DEFAULTS["wk_batch"])),
        "kv_k": float(s.get("adv_kv_k", DEFAULTS["adv_kv_k"])),
        "kv_v": float(s.get("adv_kv_v", DEFAULTS["adv_kv_v"])),
        "sw": int(s.get("adv_sw", DEFAULTS["adv_sw"])),
        "head_dim": int(s.get("adv_hd", DEFAULTS["adv_hd"])),
        "cost_per_hour": float(s.get("adv_cost", DEFAULTS["adv_cost"])),
        "override_alpha": bool(s.get("adv_oa", DEFAULTS["adv_oa"])),
        "alpha_in": float(s.get("adv_alpha", DEFAULTS["adv_alpha"])),
        "override_beta": bool(s.get("adv_ob", DEFAULTS["adv_ob"])),
        "beta_in": float(s.get("adv_beta", DEFAULTS["adv_beta"])),
        "alpha_sat_b0": float(s.get("adv_alpha_sat", DEFAULTS["adv_alpha_sat"])) if s.get("adv_alpha_sat", DEFAULTS["adv_alpha_sat"]) > 0 else None,
        "mla_kv_lora_rank": int(s.get("adv_mla_rank", DEFAULTS["adv_mla_rank"])) if s.get("adv_mla_rank", DEFAULTS["adv_mla_rank"]) > 0 else None,
        "mla_qk_rope_dim": int(s.get("adv_mla_rope", DEFAULTS["adv_mla_rope"])) if s.get("adv_mla_rope", DEFAULTS["adv_mla_rope"]) > 0 else None,
        "pp_size": int(s.get("adv_pp", DEFAULTS["adv_pp"])),
        "comm_link": s.get("adv_comm_link", DEFAULTS["adv_comm_link"]),
        "comm_link_bw": float(s.get("adv_comm_bw", DEFAULTS["adv_comm_bw"])) if s.get("adv_comm_bw", DEFAULTS["adv_comm_bw"]) > 0 else None,
        "comm_link_latency": float(s.get("adv_comm_lat", DEFAULTS["adv_comm_lat"])) / 1e3 if s.get("adv_comm_lat", DEFAULTS["adv_comm_lat"]) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Header + tabs
# ---------------------------------------------------------------------------
st.title("🧮 LLM Inference Emulator")

ui_mode = st.session_state.get("ui_mode", "Базовый (Simple)")
is_edu = ui_mode == "🎓 Учебный (Pedagogical)"
is_adv = ui_mode == "⚙️ Продвинутый (Advanced)"

tabs_list = ["🏠 Эмулятор", "📦 Справочник"]
if is_adv:
    tabs_list.append("🔬 Аналитика")

tabs = st.tabs(tabs_list)
tab_pred = tabs[0]
tab_pre = tabs[1]
if is_adv:
    tab_sca = tabs[2]
else:
    tab_sca = None


# =============================================================================
# 🏠 Эмулятор
# =============================================================================


with tab_pred:
    # Sleek collapsible introduction card to declutter the dashboard
    with st.expander("ℹ️ Что такое эмулятор инференса и как им пользоваться? (Нажмите для справки)"):
        st.markdown(
            """
### Один вопрос — один ответ

> *«Сколько tok/s выдаст моя модель X на железе Y, и хватит ли VRAM?»*

**Эмулятор отвечает за миллисекунды** — без аренды GPU, без скачивания
весов, без бенчмарка. Внутри — roofline-формула, откалиброванная против
реальных замеров (LLM-Perf Leaderboard PyTorch + наши собственные
llama-bench/vLLM прогоны на RTX-3090, A100, RTX-5090).

### Как пользоваться

1. **Слева в сайдбаре** выберите железо + движок + модель (или готовый пресет).
2. **Ниже на этой странице** задайте нагрузку (входные и выходные токены, размер батча).
3. Нажмите кнопку **🔮 Рассчитать инференс**! Вы получите prefill, decode, throughput, распределение памяти и разбор всех бутылочных горлышек.
            """
        )
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1:
            st.success("**±10-20%** внутри калиброванной области\n\n"
                       "RTX-3090 / A100-40 / A10 / T4 / 2×5090 — real-world.")
        with c_i2:
            st.warning("**±30-60%** на extrapolation\n\n"
                       "Новое железо без калибровки (H100, B200) — формула "
                       "может недооценивать throughput.")
        with c_i3:
            st.info("**Не SLA-инструмент**\n\n"
                    "Roofline даёт верхнюю границу. "
                    "Финальный замер — на реальном железе.")

    cfg = current_config()
    arch = _arch_for(cfg["n_params_b"])

    # Top context strip — reflects CURRENT sidebar choice (always live)
    moe_str = ""
    eff_active = cfg["active_b"] if cfg["active_b"] else arch.get("n_active_b", cfg["n_params_b"])
    if eff_active != cfg["n_params_b"]:
        moe_str = f" • MoE: active={eff_active}B"
    st.info(
        f"**Текущий выбор (sidebar):** "
        f"{cfg['hw']} • {cfg['engine']} • {cfg['n_params_b']}B {cfg['bits']}-bit{moe_str}"
        f" — *поменять можно в сайдбаре слева*"
    )

    # ---- INPUT FORM — gated by Predict button (no auto-rerun) ----
    with st.form("pred_form", clear_on_submit=False):
        st.subheader("1️⃣ Нагрузка (Workload)")
        st.markdown(
            "*(Определяет количество обрабатываемых токенов и параллельных запросов. Напрямую влияет на фазы Prefill/Decode и размер KV-кэша)*"
        )
        c1, c2, c3 = st.columns(3)
        c1.number_input("📥 Размер промпта (P_in, входных токенов)", min_value=1, max_value=131072,
                        key="wk_p_in", step=128,
                        help="Длина входящего текста. Фаза Prefill обрабатывает все эти токены одновременно. При этом вычисляется и сохраняется KV-кэш для каждого токена.")
        c2.number_input("📤 Длина генерации (P_out, выходных токенов)", min_value=1, max_value=8192,
                        key="wk_p_out", step=32,
                        help="Количество токенов, которое модель сгенерирует в ответ. Фаза Decode генерирует токены последовательно, один за другим (авторегрессионно).")
        c3.number_input("📦 Размер батча (Batch size, запросов)", min_value=1, max_value=2048,
                        key="wk_batch", step=1,
                        help="Количество параллельно обрабатываемых запросов/пользователей. Увеличение батча повышает загрузку GPU (throughput), но пропорционально увеличивает расход VRAM под KV-кэш.")

        with st.expander("⚙️ Дополнительные параметры и оптимизации"):
            adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
                "💾 Оптимизации KV-Кэша",
                "🧠 Архитектура & MLA",
                "🌐 Распределенный инференс (TP/PP)",
                "📈 Коэффициенты & Экономика"
            ])

            with adv_tab1:
                st.markdown("**Настройки структуры и сжатия стандартного KV-кэша:**")
                a1, a2 = st.columns(2)
                with a1:
                    st.number_input("Квантование Key (KV-K bits)", step=4.0,
                                    min_value=2.0, max_value=16.0, key="adv_kv_k",
                                    help="Точность хранения ключей в KV-кэше. Понижение разрядности (например, до 4 бит) значительно экономит видеопамять (VRAM), позволяя обрабатывать длинные контексты или использовать более крупный батч.")
                    st.number_input("Квантование Value (KV-V bits)", step=4.0,
                                    min_value=2.0, max_value=16.0, key="adv_kv_v",
                                    help="Точность хранения значений в KV-кэше. Аналогично снижает требования к видеопамяти при квантовании.")
                with a2:
                    st.number_input("Скользящее окно (Sliding Window, 0 = без ограничений)",
                                    min_value=0, max_value=131072, key="adv_sw",
                                    help="Ограничивает максимальный размер сохраняемых токенов контекста внимания. Помогает жестко контролировать размер KV-кэша и предотвращает переполнение VRAM на длинных диалогах.")
                    st.number_input("Размерность головы (head_dim, для GQA)",
                                    min_value=32, max_value=1024,
                                    step=32, key="adv_hd",
                                    help="Размерность проекции одной головы внимания (обычно 128). Влияет на форму и объем тензоров KV-кэша.")

            with adv_tab2:
                st.markdown("**Сжатие внимания Multi-Head Latent Attention (MLA) — как в DeepSeek V3/V4:**")
                st.info("💡 MLA сжимает ключи и значения в латентное пространство низкого ранга, снижая требования к памяти на больших батчах в разы. При значении 0 используется стандартный GQA/MHA.")
                ma1, ma2 = st.columns(2)
                ma1.number_input("Ранг сжатия KV (MLA KV Lora Rank, 0 = выключено)", min_value=0, max_value=2048, step=64, key="adv_mla_rank",
                                 help="Ранг латентного сжатия KV (в DeepSeek-V3 = 512).")
                ma2.number_input("Размерность QK RoPE (MLA QK RoPE Head Dim, 0 = выключено)", min_value=0, max_value=512, step=16, key="adv_mla_rope",
                                 help="Размерность отдельной Key RoPE проекции для позиционного кодирования (в DeepSeek-V3 = 64).")

            with adv_tab3:
                st.markdown("**Распределенное моделирование коммуникаций (TP/PP) & Сеть:**")
                st.info("💡 Разделение слоев (PP) или тензоров (TP) снижает требования к VRAM на одну карту, но вводит сетевые накладные расходы на стыках.")
                net1, net2 = st.columns(2)
                with net1:
                    st.number_input("Размер Pipeline Parallelism (PP) Size", min_value=1, max_value=64, step=1, key="adv_pp",
                                      help="Количество последовательных стадий конвейера. Модель разбивается на группы слоев, которые выполняются на разных GPU друг за другом. Это порождает задержки конвейера (пузырь PP / 1F1B bubble).")
                    st.selectbox("Интерконнект (Comm Link)", ["Auto", "none", "nvlink", "infinity_fabric", "pcie5", "pcie4", "pcie3", "ethernet"], key="adv_comm_link",
                                   help="Тип сетевого соединения для моделирования задержек TP и PP. 'Auto' автоматически выбирает NVLink для серверных GPU и PCIe для десктопных.")
                with net2:
                    st.number_input("Полоса линка (Link BW, GB/s, 0 = по умолчанию)", min_value=0.0, step=10.0, key="adv_comm_bw",
                                      help="Перекрывает стандартную пропускную способность выбранной шины (например, PCIe Gen4 x16 = ~31.5 GB/s).")
                    st.number_input("Задержка линка (Link Latency, ms, 0 = по умолчанию)", min_value=0.0, step=0.001, format="%.4f", key="adv_comm_lat",
                                      help="Перекрывает стандартный пинг сетевого линка.")

            with adv_tab4:
                st.markdown("**Ручные настройки формулы и экономика:**")
                e1, e2 = st.columns(2)
                with e1:
                    st.number_input("Стоимость инстанса ($/час, 0 = выключено)",
                                    min_value=0.0, step=0.5, key="adv_cost",
                                    help="Позволяет рассчитать экономическую эффективность эмуляции: стоимость генерации 1 миллиона токенов.")
                    st.number_input("Насыщение MFU батча b0 (Alpha Saturation b0, 0 = авто)",
                                    min_value=0.0, step=1.0, key="adv_alpha_sat",
                                    help="Задаёт параметр насыщения MFU батча b0. Влияет на рост вычислительной эффективности при увеличении батча.")
                with e2:
                    st.checkbox("Ручной перенос α", key="adv_oa")
                    st.number_input("Коэффициент эффективности вычислений (α)", step=0.05, key="adv_alpha",
                                    disabled=not st.session_state.get("adv_oa", False),
                                    help="Переопределяет коэффициент вычислений альфа.")
                    st.checkbox("Ручной перенос β", key="adv_ob")
                    st.number_input("Коэффициент эффективности памяти (β)", step=0.05, key="adv_beta",
                                    disabled=not st.session_state.get("adv_ob", False),
                                    help="Переопределяет коэффициент пропускной способности памяти бета.")

        st.divider()
        submitted = st.form_submit_button(
            "▶ Predict", type="primary", width="stretch",
            help="Снимок текущих параметров (sidebar + форма) → прогон формулы.",
        )

    # Snapshot all inputs on submit
    if submitted:
        st.session_state["pred_snapshot"] = build_pred_inputs()

    snap = st.session_state.get("pred_snapshot")

    # Boot state — no prediction yet
    if snap is None:
        st.divider()
        st.info(
            "👆 **Выбери параметры выше и сайдбар слева, затем нажми ▶ Predict.**\n\n"
            "Результат появится здесь: prefill / decode / throughput / memory, "
            "$/M tokens, breakdown по членам формулы, bottleneck-объяснение."
        )
    else:
        # Staleness banner — current inputs differ from last submitted snapshot
        if build_pred_inputs() != snap:
            st.warning(
                "⚠️ Параметры изменились с момента последнего **▶ Predict** — "
                "текущий результат отражает предыдущий снимок. Нажми **▶ Predict** ещё раз для пересчёта."
            )

        # All downstream rendering reads from `snap` — never from session_state directly
        hw_spec = HARDWARE_SPECS[snap["hw"]]
        eng_spec = ENGINE_DEFAULTS[snap["engine"]]
        p_in = snap["p_in"]
        p_out = snap["p_out"]
        batch = snap["batch"]
        kv_k = snap["kv_k"]
        kv_v = snap["kv_v"]
        sliding_window = int(snap["sw"]) if snap["sw"] > 0 else None
        head_dim = snap["head_dim"]
        cost_per_hour = snap["cost_per_hour"]
        override_alpha = snap["override_alpha"]
        alpha_in = snap["alpha_in"]
        override_beta = snap["override_beta"]
        beta_in = snap["beta_in"]

        # Resolve coefficients
        alpha, beta = eng_spec["alpha"], eng_spec["beta"]
        batch_saturation = eng_spec["batch_saturation"]
        cal = lookup_calibration(snap["hw"], snap["engine"], snap["precision"])

        badge_text, badge_kind = "🔴 Литературный default движка", "error"
        if cal:
            if cal["alpha"] is not None:
                alpha = cal["alpha"]
            if cal["beta"] is not None:
                beta = cal["beta"]
            if cal["batch_saturation"] is not None:
                batch_saturation = cal["batch_saturation"]
            if cal["alpha"] is not None and cal["beta"] is not None:
                badge_text = f"🟢 Калиброванo на реальных данных (n={cal['n_rows']} замеров)"
                badge_kind = "success"
            else:
                badge_text = (f"🟡 Частичная калибровка (n={cal['n_rows']}): "
                              f"{'α' if cal['alpha'] else ''} "
                              f"{'β' if cal['beta'] else ''} — остальное literature")
                badge_kind = "warning"
        if override_alpha:
            alpha = alpha_in
        if override_beta:
            beta = beta_in

        # ---- Run prediction (using snapshot values) ----
        compute_bits = eng_spec.get("compute_path", snap["bits"])
        res = predict(
            n_params_b=snap["n_params_b"], bits=snap["bits"],
            p_in=p_in, p_out=p_out, batch=batch,
            peak_flops=get_peak_compute(snap["hw"], compute_bits),
            mem_bw=get_memory_bandwidth(snap["hw"]),
            alpha=alpha, beta=beta,
            batch_saturation=batch_saturation,
            kv_packing_eff=eng_spec["kv_packing_eff"],
            compute_path=compute_bits,
            kv_bits_k=kv_k, kv_bits_v=kv_v,
            tp_size=hw_spec.get("tp_size", 1),
            tp_efficiency=hw_spec.get("tp_efficiency", 1.0),
            n_active_b=snap["active_b"],
            head_dim=head_dim,
            sliding_window=sliding_window,
            alpha_sat_b0=snap.get("alpha_sat_b0"),
            mla_kv_lora_rank=snap.get("mla_kv_lora_rank"),
            mla_qk_rope_dim=snap.get("mla_qk_rope_dim"),
            pp_size=snap["pp_size"],
            comm_link=snap["comm_link"],
            comm_link_bw=snap["comm_link_bw"],
            comm_link_latency=snap["comm_link_latency"],
            hw=snap["hw"],
        )

        # ---- Results ----
        st.subheader("2️⃣ Результат")

        # Calibration confidence badge
        {"success": st.success, "warning": st.warning, "error": st.error}[badge_kind](
            f"**Точность предсказания:** {badge_text} • "
            f"α={alpha:.3f}, β={beta:.3f}"
            + (f", batch_sat={batch_saturation}" if batch_saturation else "")
        )

        # Big metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⚡ Prefill", f"{res.prefill_s * 1000:.0f} ms",
                  help=f"Время на обработку всех {p_in} prompt-токенов "
                       f"(bottleneck: **{res.bottleneck_prefill}**)")
        m2.metric("🔄 Decode", f"{res.decode_per_token_s * 1000:.1f} ms/tok",
                  help=f"Время на генерацию одного токена "
                       f"(bottleneck: **{res.bottleneck_decode}**)")
        m3.metric("🚀 Throughput", f"{res.throughput_tok_s:.0f} tok/s",
                  help=f"Total throughput для batch={batch}")
        m4.metric("⏱ Total latency", f"{res.total_latency_s:.2f} s",
                  help=f"Prefill + {p_out}× decode")

        # Memory progress
        st.markdown("**💾 Распределение VRAM (памяти видеокарты):**")
        cap = hw_spec["memory_capacity_gb"]
        used = res.memory_gb
        
        weights_gb = snap["n_params_b"] * snap["bits"] / 8.0
        kv_gb = max(0.0, used - weights_gb)

        c_mem1, c_mem2, c_mem3 = st.columns(3)
        c_mem1.metric("📦 Веса модели (Weights)", f"{weights_gb:.1f} GB",
                      help="Память, занимаемая статическими весами модели в VRAM. Зависит от количества параметров модели (B) и разрядности квантования (bits).")
        c_mem2.metric("💾 KV-Кэш (Context)", f"{kv_gb:.1f} GB",
                      help="Динамическая память для хранения истории контекста (ключей и значений). Растет линейно с ростом батча и суммарной длины контекста (P_in + P_out).")
        c_mem3.metric("📊 Всего требуется VRAM", f"{used:.1f} GB / {cap:.0f} GB",
                      help="Суммарный объем VRAM, необходимый для запуска. Должен быть меньше физического объема VRAM видеокарты.")

        pct = min(used / cap, 1.5)
        if used > cap:
            st.error(f"⛔ **ОШИБКА: Превышен лимит VRAM!** Требуется **{used:.1f} GB**, но на видеокарте доступно только {cap:.0f} GB.\n\n"
                     f"**Решения для оптимизации VRAM:**\n"
                     f"- ✂️ **Использовать квантование**: Снизьте разрядность весов (с 16 до 4-bit) или KV-кэша.\n"
                     f"- 📦 **Уменьшить батч**: Снизьте размер батча (Batch size) или длину контекста.\n"
                     f"- 🌐 **Распределенный инференс**: Разделите модель на несколько карт с помощью Pipeline Parallelism (PP) или Tensor Parallelism (TP).")
        elif used > 0.85 * cap:
            st.warning(f"⚠️ **ВНИМАНИЕ: Память загружена почти полностью!** Занято **{used:.1f} GB** из {cap:.0f} GB ({100 * used / cap:.0f}%).\n\n"
                       f"В реальном инференсе (например, vLLM) необходим запас в 15-20% под накладные расходы CUDA-драйвера и пиковые всплески длинных диалогов.")
        else:
            st.success(f"✅ **Достаточно памяти!** Занято **{used:.1f} GB** из {cap:.0f} GB ({100 * used / cap:.0f}%). Модель свободно помещается в VRAM.")
        st.progress(min(used / cap, 1.0))

        # Cost
        if cost_per_hour > 0:
            per_million = cost_per_hour * 1e6 / (res.throughput_tok_s * 3600)
            st.info(f"💰 При **${cost_per_hour:.2f}/час** → "
                    f"**${per_million:.3f}** за миллион токенов "
                    f"(throughput {res.throughput_tok_s:.0f} tok/s).")

        # ---- Bottleneck explainer ----
        st.subheader("3️⃣ Разбор узких мест (Bottleneck Explainer)")
        st.caption("Анализ того, что сдерживает производительность вашей системы. Roofline-модель берёт максимальное время из всех фаз (выигрывает самое медленное звено).")

        eff_flops = get_peak_compute(snap["hw"], compute_bits) * \
                    hw_spec.get("tp_size", 1) * hw_spec.get("tp_efficiency", 1.0)
        eff_mbw = get_memory_bandwidth(snap["hw"]) * \
                  hw_spec.get("tp_size", 1) * hw_spec.get("tp_efficiency", 1.0)
        snap_arch = _arch_for(snap["n_params_b"])
        snap_eff_active = snap["active_b"] if snap["active_b"] else \
            snap_arch.get("n_active_b", snap["n_params_b"])
        N = snap["n_params_b"] * 1e9
        N_active = snap_eff_active * 1e9
        W = N * snap["bits"] / 8.0
        bs_eff = float(batch)
        if batch_saturation:
            bm, b50 = batch_saturation
            hill = bm * batch / (batch + b50)
            bs_eff = min(float(batch), max(min(1.0, float(batch)), hill))

        t_pre_compute = 2.0 * N_active * p_in * bs_eff / (eff_flops * alpha)
        t_pre_mem = W / eff_mbw
        t_dec_mem = W * (N_active / N) / (eff_mbw * beta)
        t_dec_compute = 2.0 * N_active * bs_eff / (eff_flops * alpha)

        pp_size = snap.get("pp_size", 1)
        if pp_size > 1:
            t_pre_compute /= pp_size
            t_pre_mem /= pp_size
            t_dec_mem /= pp_size
            t_dec_compute /= pp_size

        t_tp_comm_prefill = res.tp_comm_prefill_s or 0.0
        t_pp_comm_prefill = res.pp_comm_prefill_s or 0.0
        t_pp_bubble_prefill = res.pp_bubble_prefill_s or 0.0
        t_tp_comm_decode = res.tp_comm_decode_s or 0.0
        t_pp_comm_decode = res.pp_comm_decode_s or 0.0
        t_pp_bubble_decode = res.pp_bubble_decode_s or 0.0

        breakdown_rows = [
            {"phase": "Prefill", "term": "compute  2·N_act·P_in·bs / (C·α)",
             "ms": t_pre_compute * 1000,
             "winner": res.bottleneck_prefill == "compute"},
            {"phase": "Prefill", "term": "memory   W / MBW (один проход весов)",
             "ms": t_pre_mem * 1000,
             "winner": res.bottleneck_prefill == "memory"},
        ]
        if t_tp_comm_prefill > 0:
            breakdown_rows.append({
                "phase": "Prefill", "term": "TP All-Reduce network overhead",
                "ms": t_tp_comm_prefill * 1000,
                "winner": True
            })
        if t_pp_comm_prefill > 0:
            breakdown_rows.append({
                "phase": "Prefill", "term": "PP boundary transfer overhead",
                "ms": t_pp_comm_prefill * 1000,
                "winner": True
            })
        if t_pp_bubble_prefill > 0:
            breakdown_rows.append({
                "phase": "Prefill", "term": "PP 1F1B scheduling bubble overhead",
                "ms": t_pp_bubble_prefill * 1000,
                "winner": True
            })

        breakdown_rows.extend([
            {"phase": "Decode", "term": "memory   W·(N_act/N) / (MBW·β)",
             "ms": t_dec_mem * 1000,
             "winner": res.bottleneck_decode == "memory"},
            {"phase": "Decode", "term": "compute  2·N_act·bs / (C·α)",
             "ms": t_dec_compute * 1000,
             "winner": res.bottleneck_decode == "compute"},
        ])
        if t_tp_comm_decode > 0:
            breakdown_rows.append({
                "phase": "Decode", "term": "TP All-Reduce network overhead",
                "ms": t_tp_comm_decode * 1000,
                "winner": True
            })
        if t_pp_comm_decode > 0:
            breakdown_rows.append({
                "phase": "Decode", "term": "PP boundary transfer overhead",
                "ms": t_pp_comm_decode * 1000,
                "winner": True
            })
        if t_pp_bubble_decode > 0:
            breakdown_rows.append({
                "phase": "Decode", "term": "PP 1F1B scheduling bubble overhead",
                "ms": t_pp_bubble_decode * 1000,
                "winner": True
            })

        breakdown = pd.DataFrame(breakdown_rows)
        chart = alt.Chart(breakdown).mark_bar().encode(
            x=alt.X("ms:Q", title="миллисекунды"),
            y=alt.Y("term:N", sort=None, title=""),
            color=alt.Color("winner:N",
                            scale=alt.Scale(domain=[True, False],
                                            range=["#e45756", "#bbbbbb"]),
                            legend=alt.Legend(title="bottleneck", labelExpr="datum.label == 'true' ? 'выигрывает (медленнее)' : 'не bottleneck'")),
            row=alt.Row("phase:N", header=alt.Header(title=None, labelFontSize=14)),
            tooltip=["phase", "term", alt.Tooltip("ms:Q", format=".3f")],
        ).properties(height=80)
        st.altair_chart(chart, width="stretch")

        # Visual columns for Prefill and Decode phase bottlenecks
        col_pre, col_dec = st.columns(2)
        
        with col_pre:
            st.markdown("### 📥 Фаза Prefill (Анализ входящего текста)")
            if res.bottleneck_prefill == "compute":
                st.error("🚨 **Ограничение: Вычисления (Compute-bound)**")
                st.markdown(
                    "**Что это значит?** Графический чип загружен математическими операциями на 100%. "
                    "Входящий промпт большой, и CUDA-ядра заняты перемножением матриц.\n\n"
                    "**Как ускорить:**\n"
                    "- ⚡ Выберите более мощный GPU с более высоким показателем TFLOPS.\n"
                    "- 💾 Настройте **Prefix Caching** (кэширование часто повторяющихся промптов), чтобы избежать повторных вычислений."
                )
            else:
                st.warning("🔵 **Ограничение: Пропускная способность памяти (Memory-bound)**")
                st.markdown(
                    "**Что это значит?** Математические ядра простаивают, потому что GPU тратит больше времени на чтение весов модели из VRAM, чем на сами вычисления. "
                    "Это типично для коротких промптов или маленького батча.\n\n"
                    "**Как ускорить:**\n"
                    "- 📦 Увеличьте размер батча (Batch size) — это позволит амортизировать время чтения весов на большее число запросов, повышая эффективность ядер."
                )

        with col_dec:
            st.markdown("### 🔄 Фаза Decode (Генерация ответа)")
            if res.bottleneck_decode == "memory":
                st.warning("🔵 **Ограничение: Пропускная способность памяти (Memory-bound)**")
                st.markdown(
                    f"**Что это значит?** Модель генерирует токены по одному. На каждом шаге GPU приходится полностью считывать все веса модели (**{W / 1e9:.1f} GB**) из памяти VRAM. "
                    f"Это классическое «бутылочное горлышко» для LLM.\n\n"
                    f"**Как ускорить:**\n"
                    f"- ✂️ **Примените квантование (4/8-bit)**: Это уменьшит размер весов в VRAM и ускорит их чтение.\n"
                    f"- 📦 Увеличьте размер батча — это повысит общую пропускную способность (throughput), распределяя одно чтение весов на много параллельных запросов.\n"
                    f"- 💻 Выберите GPU с большей пропускной способностью памяти (HBM3 вместо GDDR6)."
                )
            else:
                st.error("🚨 **Ограничение: Вычисления (Compute-bound)**")
                st.markdown(
                    "**Что это значит?** При декодировании GPU ограничен вычислительной мощностью. Такое случается при очень больших батчах, когда накладные расходы ядер (MFU) становятся доминирующими.\n\n"
                    "**Как ускорить:**\n"
                    "- ⚙️ Уменьшите размер батча или примените **Speculative Decoding** (когда легкая модель-черновик генерирует токены вперед, а целевая модель верифицирует их за один compute-шаг)."
                )

        # Satellite warnings (Saturation, Parallelism communication)
        if batch_saturation and bs_eff < batch:
            st.info(
                f"⚠️ **Эффективный батч: {bs_eff:.1f}** (запрошено {batch}).\n\n"
                f"Кривая насыщения MFU непрерывного батчинга (Continuous Batching) приближается к физическому лимиту "
                f"движка (`batch_max = {batch_saturation[0]}`). Дальнейшее увеличение батча почти не увеличит общую скорость (throughput)."
            )

        if (res.tp_comm_prefill_s and res.tp_comm_prefill_s > 0) or (res.pp_comm_prefill_s and res.pp_comm_prefill_s > 0) or (res.pp_bubble_prefill_s and res.pp_bubble_prefill_s > 0):
            link_info = f"{res.comm_link.upper()}"
            if res.comm_link_bw is not None:
                link_info += f" ({res.comm_link_bw:.1f} GB/s"
            if res.comm_link_latency is not None:
                link_info += f", {res.comm_link_latency * 1e6:.1f} µs latency)"
            else:
                link_info += ")"
            
            p2p_note = ""
            if snap.get("tp_size", 1) > 1 and res.comm_link.startswith("pcie"):
                p2p_note = "\n\n⚠️ **Host-Mediated TP PCIe:** На потребительских видеокартах (RTX 3090/4090/5090) драйвер NVIDIA блокирует прямой P2P DMA через PCIe. Обмен идет транзитом через оперативную память хоста, что снижает скорость шины в ~2 раза и втрое увеличивает задержки (учтено в расчёте)."

            st.warning(
                f"🌐 **Сетевые накладные расходы распределенного выполнения ({link_info}):**\n\n"
                f"При разбиении модели на {snap.get('tp_size', 1)} GPU (TP) и {snap['pp_size']} конвейеров (PP) возникают следующие задержки:\n"
                f"- **TP All-Reduce (Ring):** Prefill = {t_tp_comm_prefill * 1000:.2f} ms, Decode = {t_tp_comm_decode * 1000:.3f} ms/token\n"
                f"- **PP stage-to-stage boundary:** Prefill = {t_pp_comm_prefill * 1000:.2f} ms, Decode = {t_pp_comm_decode * 1000:.3f} ms/token\n"
                f"- **PP 1F1B scheduling bubble (простои стадий конвейера):** Prefill = {t_pp_bubble_prefill * 1000:.2f} ms, Decode = {t_pp_bubble_decode * 1000:.3f} ms/token\n\n"
                "Эти задержки добавлены поверх времени вычислений и памяти в соответствии с топологией Megatron-LM.\n\n"
                "💡 *Совет:* В реальных движках (vLLM/SGLang) за счет асинхронного перекрытия (overlap) коммуникации и вычислений фактическое замедление может быть в 2-3 раза меньше."
                f"{p2p_note}"
            )

        if is_edu:
            st.divider()
            st.subheader("🎓 Учебный разбор: Физика и математика инференса")
            
            # Common parameters for calculations
            arch_temp = _arch_for(snap["n_params_b"])
            L_layers = snap.get("layers") if snap.get("layers") is not None else arch_temp.get("layers", 32)
            H_kv = snap.get("kv_heads") if snap.get("kv_heads") is not None else arch_temp.get("kv_heads", 8)
            hd_dim = head_dim
            kv_eff_pct = eng_spec["kv_packing_eff"]
            is_mla = snap.get("mla_kv_lora_rank") is not None and snap.get("mla_qk_rope_dim") is not None
            
            if is_mla:
                r_lora = snap["mla_kv_lora_rank"]
                r_rope = snap["mla_qk_rope_dim"]
                kv_tok_b = L_layers * (r_lora + r_rope) * kv_k / 8.0
            else:
                kv_tok_b = L_layers * H_kv * hd_dim * (kv_k + kv_v) / 8.0

            # Sliding-window-aware total KV bytes — must be computed before
            # the VRAM breakdown bar and the "Шаг 3" caption both reference
            # `kv_total`. Same math reused below in edu_tab2 and matches
            # the t_kv term in formula.py (kv_per_token_bytes * ctx_kept *
            # batch / kv_packing_eff).
            ctx_used = p_in + p_out
            if sliding_window is not None:
                ctx_used = min(ctx_used, sliding_window)
            kv_total = kv_tok_b * ctx_used * batch / kv_eff_pct

            # 📊 Visual VRAM Allocation Breakdown
            capacity_gb = hw_spec["memory_capacity_gb"]
            w_gb = W / 1e9
            kv_gb = kv_total / 1e9
            # CUDA context & Workspace allocation estimate
            ov_gb = min(2.0, max(0.5, capacity_gb * 0.05))
            total_used_gb = w_gb + kv_gb + ov_gb
            free_gb = max(0.0, capacity_gb - total_used_gb)

            # Convert to percentages for styling
            w_pct = w_gb / capacity_gb * 100
            kv_pct = kv_gb / capacity_gb * 100
            ov_pct = ov_gb / capacity_gb * 100
            free_pct = free_gb / capacity_gb * 100

            # Normalize percentages if over capacity (to fit 100%)
            if total_used_gb > capacity_gb:
                scale_factor = 100.0 / (w_pct + kv_pct + ov_pct)
                w_pct *= scale_factor
                kv_pct *= scale_factor
                ov_pct *= scale_factor
                free_pct = 0.0
                bar_border = "2px solid #e74c3c"
            else:
                bar_border = "1px solid #ddd"

            st.markdown(f"### 📊 Интерактивная карта распределения памяти VRAM ({capacity_gb:.0f} GB)")
            st.caption("Цветная шкала показывает, какую долю физической памяти GPU занимают различные компоненты инференса.")
            
            # Labels
            w_label = f"Веса ({w_gb:.1f} GB)" if w_pct > 12 else "Веса" if w_pct > 6 else ""
            kv_label = f"KV Cache ({kv_gb:.1f} GB)" if kv_pct > 12 else "KV" if kv_pct > 6 else ""
            ov_label = f"Ов. ({ov_gb:.1f} GB)" if ov_pct > 12 else "Ов." if ov_pct > 6 else ""
            free_label = f"Свободно ({free_gb:.1f} GB)" if free_pct > 12 else "Своб." if free_pct > 6 else ""

            bar_html = f"""
            <div style="width: 100%; font-family: sans-serif; margin-bottom: 20px;">
                <div style="display: flex; width: 100%; height: 32px; border-radius: 8px; overflow: hidden; border: {bar_border}; background-color: #f0f2f6; box-shadow: inset 0 1px 3px rgba(0,0,0,0.12);">
                    {f'<div style="width: {w_pct}%; background-color: #2ca02c; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; border-right: 1px solid rgba(255,255,255,0.2);" title="Статические веса модели: {w_gb:.2f} GB">{w_label}</div>' if w_pct > 0 else ""}
                    {f'<div style="width: {kv_pct}%; background-color: #1f77b4; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; border-right: 1px solid rgba(255,255,255,0.2);" title="Кэш Key-Value контекстов: {kv_gb:.2f} GB">{kv_label}</div>' if kv_pct > 0 else ""}
                    {f'<div style="width: {ov_pct}%; background-color: #ff7f0e; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; border-right: 1px solid rgba(255,255,255,0.2);" title="CUDA Context / Драйверные накладные расходы: {ov_gb:.2f} GB">{ov_label}</div>' if ov_pct > 0 else ""}
                    {f'<div style="width: {free_pct}%; background-color: #bbbbbb; color: #333333; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold;" title="Свободная память VRAM: {free_gb:.2f} GB">{free_label}</div>' if free_pct > 0 else ""}
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 12px; color: #555;">
                    <div>🟢 <b>Веса модели:</b> {w_gb:.2f} GB</div>
                    <div>🔵 <b>KV Cache:</b> {kv_gb:.2f} GB</div>
                    <div>🟠 <b>Оверхед CUDA/Движка:</b> {ov_gb:.2f} GB</div>
                    <div>⚫ <b>Всего занято:</b> {total_used_gb:.2f} / {capacity_gb:.0f} GB</div>
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
            
            # Stepper Tabs
            edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs([
                "🧮 Пошаговый расчет памяти и времени",
                "📉 Интерактивный Roofline-график",
                "🔄 Жизненный цикл запроса к LLM",
                "🧠 Проверь свои знания (Викторина)"
            ])
            
            with edu_tab1:
                st.markdown("### 1. Веса модели в памяти")
                w_params = snap["n_params_b"]
                w_bits = snap["bits"]
                w_gb_calc = w_params * w_bits / 8.0
                st.markdown(
                    f"Размер статических весов модели в VRAM рассчитывается по формуле:\n"
                    f"$$W = N_{{params}} \\times \\frac{{bits}}{{8}}$$\n\n"
                    f"Для вашей модели **{w_params:.1f}B** параметров и разрядности весов **{w_bits} бит**:\n"
                    f"$$W = {w_params:.2f} \\times 10^9 \\times \\frac{{{w_bits}}}{{8}} = {w_gb_calc:.2f}\\text{{ GB}}$$"
                )
                
                st.markdown("### 2. Динамический KV-кэш")
                ctx_total = p_in + p_out
                sw_cap_str = ""
                if sliding_window is not None:
                    ctx_total = min(ctx_total, sliding_window)
                    sw_cap_str = f" (ограничено скользящим окном в {sliding_window} токенов)"
                    
                if is_mla:
                    r_lora = snap["mla_kv_lora_rank"]
                    r_rope = snap["mla_qk_rope_dim"]
                    kv_tok_formula = f"L \\times (r_{{lora}} + r_{{rope}}) \\times \\frac{{b_{{kv}}}}{{8}}"
                    kv_tok_calc = f"{L_layers} \\times ({r_lora} + {r_rope}) \\times \\frac{{{kv_k}}}{{8}} = {kv_tok_b:.0f}\\text{{ байт/токен}}"
                else:
                    kv_tok_formula = f"L \\times H_{{kv}} \\times d_{{head}} \\times \\frac{{b_K + b_V}}{{8}}"
                    kv_tok_calc = f"{L_layers} \\times {H_kv} \\times {hd_dim} \\times \\frac{{{kv_k} + {kv_v}}}{{8}} = {kv_tok_b:.0f}\\text{{ байт/токен}}"
                    
                kv_total_calc_bytes = kv_tok_b * ctx_total * batch / kv_eff_pct
                kv_total_gb = kv_total_calc_bytes / 1e9
                
                st.markdown(
                    f"Размер динамического кэша Key-Value рассчитывается следующим образом:\n"
                    f"1. **Объем кэша на один токен (в байтах):**\n"
                    f"$$KV_{{token}} = {kv_tok_formula}$$\n"
                    f"С архитектурными параметрами выбранной модели:\n"
                    f"$$KV_{{token}} = {kv_tok_calc}$$\n\n"
                    f"2. **Суммарный объем кэша для всей нагрузки с учетом батча:**\n"
                    f"$$KV_{{total}} = \\frac{{KV_{{token}} \\times (P_{{in}} + P_{{out}}) \\times Batch}}{{\\eta_{{packing}}}}$$\n\n"
                    f"В вашем случае с батчем **{batch}**, общей длиной контекста **{p_in} + {p_out} = {p_in+p_out} токенов**{sw_cap_str} и эффективностью PagedAttention **{kv_eff_pct:.0%}**:\n"
                    f"$$KV_{{total}} = \\frac{{{kv_tok_b:.0f} \\times {ctx_total} \\times {batch}}}{{{kv_eff_pct}}} = {kv_total_gb * 1024:.1f}\\text{{ MB}} \\approx {kv_total_gb:.3f}\\text{{ GB}}$$"
                )
                
                st.markdown("### 3. Производительность и фазы инференса (TFLOPS vs. Memory)")
                intensity_prefill = (2.0 * N_active * bs_eff * p_in) / W if W > 0 else 1.0
                intensity_decode = (2.0 * N_active * bs_eff) / W if W > 0 else 1.0
                ridge_point = eff_flops / eff_mbw
                
                pre_verdict = "**Compute-bound** (ограничено ядрами вычислений)" if res.bottleneck_prefill == "compute" else "**Memory-bound** (ограничено скоростью чтения VRAM)"
                dec_verdict = "**Compute-bound** (ограничено ядрами вычислений)" if res.bottleneck_decode == "compute" else "**Memory-bound** (ограничено скоростью чтения VRAM)"
                
                st.markdown(
                    f"**Арифметическая интенсивность (Arithmetic Intensity)** — это отношение математических операций (FLOPs) к объёму считываемой памяти (Bytes).\n"
                    f"У вашего графического процессора точка перелома (**Ridge Point**) равна: TFLOPS / MBW = **{ridge_point:.1f} FLOP/byte**.\n"
                    f"- Если интенсивность нагрузки **выше** Ridge Point, система упирается в ядра GPU (**Compute-bound**).\n"
                    f"- Если интенсивность нагрузки **ниже** Ridge Point, система упирается в память VRAM (**Memory-bound**).\n\n"
                    f"**1️⃣ Фаза Prefill (Анализ prompt):**\n"
                    f"- Математические вычисления: $2 \\times N_{{active}} \\times P_{{in}} \\times bs = 2 \\times {snap_eff_active:.1f}\\text{{B}} \\times {p_in} \\times {bs_eff:.1f} = {2.0 * snap_eff_active * p_in * bs_eff:.1f}\\text{{ TFLOPs}}$\n"
                    f"- Арифметическая интенсивность: $2 \\times N_{{active}} \\times P_{{in}} \\times bs / W \\approx$ **{intensity_prefill:.1f} FLOP/byte**.\n"
                    f"- Вердикт: {pre_verdict}.\n\n"
                    f"**2️⃣ Фаза Decode (Генерация ответа):**\n"
                    f"- Математические вычисления на шаг: $2 \\times N_{{active}} \\times bs = 2 \\times {snap_eff_active:.1f}\\text{{B}} \\times {bs_eff:.1f} = {2.0 * snap_eff_active * bs_eff / 1e3:.3f}\\text{{ TFLOPs}}$\n"
                    f"- Арифметическая интенсивность: $2 \\times N_{{active}} \\times bs / W \\approx$ **{intensity_decode:.2f} FLOP/byte**.\n"
                    f"- Вердикт: {dec_verdict}. Поскольку декод генерирует по одному токену, интенсивность падает в $P_{{in}}$ раз! Поэтому генерация почти всегда упирается в пропускную способность памяти."
                )

                st.divider()
                st.markdown("### 🎛️ Песочница «Что, если?»: Эксперименты со масштабированием")
                st.caption(
                    "Поиграйте с ползунками ниже, чтобы мгновенно увидеть перерасчет памяти и арифметической интенсивности "
                    "в реальном времени. Это безопасная песочница, которая не сбивает ваши основные настройки в сайдбаре."
                )

                c_wi1, c_wi2, c_wi3 = st.columns(3)
                with c_wi1:
                    wi_batch = st.slider("Экспериментальный батч (bs)", 1, 128, int(batch), key="wi_batch_slider")
                with c_wi2:
                    wi_pin = st.slider("Длина промпта (P_in)", 32, 8192, int(p_in), step=32, key="wi_pin_slider")
                with c_wi3:
                    wi_pout = st.slider("Длина генерации (P_out)", 16, 2048, int(p_out), step=16, key="wi_pout_slider")

                # Recalculate everything for What-If
                wi_ctx_total = wi_pin + wi_pout
                if sliding_window is not None:
                    wi_ctx_total = min(wi_ctx_total, sliding_window)
                
                # KV Cache on single token in what-if
                if is_mla:
                    wi_kv_tok_b = L_layers * (snap["mla_kv_lora_rank"] + snap["mla_qk_rope_dim"]) * kv_k / 8.0
                else:
                    wi_kv_tok_b = L_layers * H_kv * hd_dim * (kv_k + kv_v) / 8.0
                
                wi_kv_bytes = wi_kv_tok_b * wi_ctx_total * wi_batch / kv_eff_pct
                wi_kv_gb = wi_kv_bytes / 1e9
                wi_total_vram = w_gb + wi_kv_gb + ov_gb

                # Intensities for what-if
                wi_intensity_prefill = (2.0 * N_active * wi_batch * wi_pin) / W if W > 0 else 1.0
                wi_intensity_decode = (2.0 * N_active * wi_batch) / W if W > 0 else 1.0

                st.markdown("#### 📊 Мгновенный результат эксперимента:")
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Размер KV-кэша (Что-если)", f"{wi_kv_gb * 1024:.1f} MB", 
                              delta=f"{(wi_kv_gb - kv_gb) * 1024:+.1f} MB", delta_color="inverse")
                with col_res2:
                    st.metric("Арифм. инт. Prefill", f"{wi_intensity_prefill:.1f} FLOP/byte",
                              delta=f"{wi_intensity_prefill - intensity_prefill:+.1f} FLOP/B")
                with col_res3:
                    st.metric("Арифм. инт. Decode", f"{wi_intensity_decode:.2f} FLOP/byte",
                              delta=f"{wi_intensity_decode - intensity_decode:+.2f} FLOP/B")

                # Show visual math formula with live values for What-If
                st.latex(rf"""
                KV_{{\text{{total}}}} \;=\; \frac{{{wi_kv_tok_b:.0f}\text{{ B/tok}} \times {wi_ctx_total}\text{{ ток}} \times {wi_batch}}}{{{kv_eff_pct}}} \;=\; {wi_kv_gb:.3f}\text{{ GB}}
                """)
                
                if wi_total_vram > capacity_gb:
                    st.error(f"❌ **OOM! При этих параметрах вы выйдете за лимит памяти GPU:** требуется **{wi_total_vram:.1f} GB** при ёмкости **{capacity_gb:.0f} GB**.")
                else:
                    st.success(f"✅ **Модель поместится в VRAM:** требуется **{wi_total_vram:.1f} GB** (свободно еще **{capacity_gb - wi_total_vram:.1f} GB**).")
            
            with edu_tab2:
                st.markdown("### Визуализация Roofline-модели")
                st.caption(
                    "На этом интерактивном графике показана физическая граница возможностей выбранного GPU. "
                    "Зеленая линия — пиковый теоретический лимит. "
                    "Звезды показывают, где именно находятся фазы Prefill и Decode для вашей нагрузки."
                )
                import numpy as np
                xs = np.logspace(-1, 4, 100)
                ys = np.minimum(eff_flops, eff_mbw * xs) / 1e12
                roof_df = pd.DataFrame({"Intensity": xs, "Performance": ys, "Type": "GPU Roofline Ceiling"})
                
                prefill_flops = 2.0 * N_active * p_in * bs_eff
                prefill_tflops = (prefill_flops / res.prefill_s) / 1e12 if res.prefill_s > 0 else 0.0
                
                decode_flops = 2.0 * N_active * bs_eff
                decode_tflops = (decode_flops / res.decode_per_token_s) / 1e12 if res.decode_per_token_s > 0 else 0.0
                
                pts_df = pd.DataFrame([
                    {"Intensity": intensity_prefill, "Performance": prefill_tflops, "Phase": "⭐ Prefill Phase (Prompt)"},
                    {"Intensity": intensity_decode, "Performance": decode_tflops, "Phase": "⭐ Decode Phase (Generation)"}
                ])
                
                line_chart = alt.Chart(roof_df).mark_line(color="#2ca02c", strokeWidth=3).encode(
                    x=alt.X("Intensity:Q", scale=alt.Scale(type="log"), title="Арифметическая интенсивность (FLOP / Byte)"),
                    y=alt.Y("Performance:Q", scale=alt.Scale(type="log"), title="Производительность (TFLOPS)"),
                    tooltip=["Intensity", "Performance"]
                )
                
                pts_chart = alt.Chart(pts_df).mark_point(size=220, filled=True, color="#d62728").encode(
                    x="Intensity:Q",
                    y="Performance:Q",
                    color=alt.Color("Phase:N", legend=alt.Legend(title="Фазы инференса")),
                    tooltip=["Phase", alt.Tooltip("Intensity:Q", format=".2f"), alt.Tooltip("Performance:Q", format=".2f")]
                )
                
                st.altair_chart((line_chart + pts_chart).properties(height=350), width="stretch")
                st.caption("Если точка лежит на горизонтальном участке зеленой линии — она ограничена Compute (ядрами). Если на наклонном — Memory (памятью).")
                
                st.divider()
                st.markdown("### 📉 Точка перелома (Ridge Point) на разном железе")
                st.caption(
                    "Точка перелома рассчитывается как отношение пиковой производительности к пропускной способности памяти: "
                    "**Ridge Point = Peak TFLOPS / Memory Bandwidth**."
                )

                ridge_data = [
                    {"GPU": "RTX 4090 (Desktop)", "TFLOPS (BF16)": 83.0, "Bandwidth (GB/s)": 1008.0, "Ridge Point (FLOP/byte)": 82.3, "Класс": "Потребительский"},
                    {"GPU": "NVIDIA L4", "TFLOPS (BF16)": 30.0, "Bandwidth (GB/s)": 300.0, "Ridge Point (FLOP/byte)": 100.0, "Класс": "Энергоэффективный"},
                    {"GPU": "NVIDIA A100 (PCIe)", "TFLOPS (BF16)": 312.0, "Bandwidth (GB/s)": 1935.0, "Ridge Point (FLOP/byte)": 161.2, "Класс": "Серверный (предыдущий)"},
                    {"GPU": "NVIDIA H100 (SXM5)", "TFLOPS (BF16)": 989.0, "Bandwidth (GB/s)": 3350.0, "Ridge Point (FLOP/byte)": 295.2, "Класс": "Серверный (флагман)"},
                    {"GPU": "NVIDIA H200 (SXM)", "TFLOPS (BF16)": 989.0, "Bandwidth (GB/s)": 4800.0, "Ridge Point (FLOP/byte)": 206.0, "Класс": "HBM3e (сверхбыстрый)"},
                ]
                df_ridge = pd.DataFrame(ridge_data)
                
                st.dataframe(df_ridge, width="stretch", hide_index=True)
                
                st.info(
                    f"💡 **Обратите внимание:** Для выбранного вами GPU **{snap['hw']}** Ridge Point равен **{ridge_point:.1f} FLOP/byte**.\n\n"
                    "**Физический вывод:** Если арифметическая интенсивность вашей операции (см. вкладку 1) меньше этой величины, "
                    "ядра видеокарты будут простаивать, ожидая данные из памяти. "
                    "На флагманских чипах (например, H100 SXM5 с Ridge Point 295.2) получить максимальную отдачу намного сложнее, чем на RTX 4090, "
                    "поэтому серверные архитектуры критически зависят от технологий сжатия памяти (GQA, MLA) и больших батчей."
                )
            
            with edu_tab3:
                st.markdown("### 🔄 Жизненный цикл обработки вашего запроса в GPU:")
                st.caption("Нажмите на стрелочки ниже, чтобы рассмотреть каждый шаг процесса инференса в деталях.")
                
                st.info(
                    "**📥 Шаг 1: Токенизация (Tokenization)**\n\n"
                    "Ваш входной текст разбивается на токены (словоформы или части слов). "
                    f"Ваш промпт преобразован в **{p_in} токенов**."
                )
                st.markdown("⬇️")
                
                prefill_flops_total = 2.0 * N_active * p_in * bs_eff / 1e12
                st.success(
                    f"**⚡ Шаг 2: Фаза Prefill (Насыщение)**\n\n"
                    f"GPU считывает веса модели и обрабатывает все **{p_in} токенов** промпта параллельно в один проход. "
                    f"Это требует огромных параллельных вычислений на CUDA-ядрах.\n\n"
                    f"- 🧮 **Объем вычислений:** {prefill_flops_total:.3f} TFLOPs операций\n"
                    f"- 🕒 **Время выполнения:** {res.prefill_s * 1000:.0f} мс\n"
                    f"- 📈 **Арифметическая интенсивность:** {intensity_prefill:.1f} FLOP/byte\n"
                    f"- 🚨 **Физический предел:** упирается в **{res.bottleneck_prefill.upper()}**."
                )
                st.markdown("⬇️")
                
                st.info(
                    f"**💾 Шаг 3: Запись в KV-кэш (KV Cache Storage)**\n\n"
                    f"Для каждого из {p_in} токенов промпта вычисляются векторы ключей (Key) и значений (Value). "
                    "Они сохраняются во VRAM, чтобы избежать квадратичного пересчета внимания на последующих шагах.\n\n"
                    f"- 📐 **Размер KV на токен:** {kv_tok_b:.0f} байт\n"
                    f"- 💾 **Всего выделено во VRAM:** {kv_total / 1e6:.1f} MB (для всего батча из {batch} запросов)\n"
                    f"- 🌐 **Скорость записи:** ограничена пропускной способностью HBM/VRAM ({eff_mbw / 1e9:.0f} GB/s)."
                )
                st.markdown("⬇️")
                
                st.warning(
                    f"**🔄 Шаг 4: Цикл Decode (Авторегрессия)**\n\n"
                    f"Модель начинает генерировать ответ последовательно, токен за токеном. На генерацию каждого из **{p_out} токенов** ответа "
                    f"GPU вынужден считывать из памяти VRAM абсолютно все веса модели (**{W / 1e9:.1f} GB**).\n\n"
                    f"- 🕒 **Время на один токен:** {res.decode_per_token_s * 1000:.1f} мс/токен\n"
                    f"- 📈 **Арифметическая интенсивность:** {intensity_decode:.2f} FLOP/byte\n"
                    f"- 🚨 **Физический лимит:** упирается в **{res.bottleneck_decode.upper()}** (низкий батч не позволяет загрузить ядра GPU)."
                )
                st.markdown("⬇️")
                
                st.success(
                    f"**📤 Шаг 5: Вывод и Детокенизация (Detokenization)**\n\n"
                    f"Сгенерированные токены переводятся обратно в человеческий текст и выводятся пользователю. "
                    f"Всего сгенерировано **{p_out} токенов** со средней скоростью **{res.throughput_tok_s:.0f} токенов/сек**."
                )

            with edu_tab4:
                st.markdown("### 🧠 Обучающая мини-викторина")
                st.caption("Проверьте свои знания о физических ограничениях и архитектуре LLM-инференса.")

                # Question 1
                st.markdown("---")
                st.markdown("**Вопрос 1:** Какая фаза инференса в подавляющем большинстве случаев упирается в пропускную способность памяти (Memory-bound) при реальном деплое?")
                q1 = st.radio(
                    "Выберите правильный вариант:",
                    [
                        "1. Фаза Prefill (обработка входного промпта)",
                        "2. Фаза Decode (авторегрессионная генерация токенов)",
                        "3. Обе фазы всегда одинаково Compute-bound"
                    ],
                    key="quiz_q1"
                )
                if st.button("Проверить Вопрос 1"):
                    if "2." in q1:
                        st.success(
                            "🎉 **Правильно!**\n\n"
                            "При генерации каждого последующего токена GPU вынужден считывать абсолютно все гигабайты весов модели из VRAM "
                            "ради выполнения всего пары FLOP вычислений над одним новым токеном. "
                            "Поэтому Decode почти всегда ограничен пропускной способностью памяти (Memory-bound)."
                        )
                    else:
                        st.error(
                            "❌ **Неверно.**\n\n"
                            "Попробуйте еще раз! Подсказка: на фазе Prefill все токены промпта обрабатываются параллельно (что дает высокую арифметическую интенсивность), "
                            "а на фазе Decode токены генерируются по одному, заставляя GPU бесконечно перечитывать свои веса из памяти."
                        )

                # Question 2
                st.markdown("---")
                st.markdown("**Вопрос 2:** Каким образом технология Grouped-Query Attention (GQA) ускоряет инференс больших моделей?")
                q2 = st.radio(
                    "Выберите правильный вариант:",
                    [
                        "1. Уменьшает количество параметров в полносвязных слоях (MLP)",
                        "2. Сжимает размер KV-кэша в памяти, уменьшая требования к пропускной способности VRAM",
                        "3. Позволяет обрабатывать промпт параллельно на нескольких GPU без сетевых задержек"
                    ],
                    key="quiz_q2"
                )
                if st.button("Проверить Вопрос 2"):
                    if "2." in q2:
                        st.success(
                            "🎉 **Правильно!**\n\n"
                            "GQA группирует несколько голов Key и Value, привязывая их к большему числу Query голов. "
                            "Это снижает объем памяти для хранения KV-кэша (обычно в 4-8 раз) и радикально сокращает трафик данных "
                            "из VRAM во время фазы Decode."
                        )
                    else:
                        st.error(
                            "❌ **Неверно.**\n\n"
                            "Подсказка: GQA влияет не на веса модели (MLP слои не меняются), а на динамические вектора Key и Value, "
                            "сохраняемые на каждом слое внимания."
                        )

                # Question 3
                st.markdown("---")
                st.markdown("**Вопрос 3:** Если мы увеличиваем Batch Size с 1 до 64 при постоянной длине контекста, как это влияет на арифметическую интенсивность фазы Decode?")
                q3 = st.radio(
                    "Выберите правильный вариант:",
                    [
                        "1. Арифметическая интенсивность возрастает линейно, так как веса модели переиспользуются для нескольких запросов одновременно",
                        "2. Арифметическая интенсивность падает, так как возрастают накладные расходы KV-кэша",
                        "3. Арифметическая интенсивность не меняется"
                    ],
                    key="quiz_q3"
                )
                if st.button("Проверить Вопрос 3"):
                    if "1." in q3:
                        st.success(
                            "🎉 **Правильно!**\n\n"
                            "При увеличении батча веса модели считываются из VRAM один раз для всего пакета из 64 запросов. "
                            "Объем вычислений (FLOPs) возрастает в 64 раза при том же трафике весов. "
                            "Это резко повышает арифметическую интенсивность, приближая операцию к Compute-bound зоне и повышая MFU GPU!"
                        )
                    else:
                        st.error(
                            "❌ **Неверно.**\n\n"
                            "Подсказка: подумайте, считывается ли вес модели из VRAM для каждого запроса отдельно, или один раз для всего батча."
                        )
            st.divider()

        with st.expander("🔢 Сырые числа (W, C, MBW, batch_eff)"):
            alpha_eff_str = f"alpha_effective (MFU)   = {res.alpha_eff:.4f}\n" if res.alpha_eff is not None else ""
            mla_str = ""
            if res.mla_kv_lora_rank is not None and res.mla_qk_rope_dim is not None:
                mla_str = (f"mla_kv_lora_rank        = {res.mla_kv_lora_rank}\n"
                           f"mla_qk_rope_dim         = {res.mla_qk_rope_dim}\n")
            
            comm_str = ""
            if res.comm_link is not None:
                comm_str = (
                    f"comm_link               = {res.comm_link}\n"
                    f"comm_link_bw            = {res.comm_link_bw:.1f} GB/s\n"
                    f"comm_link_latency_us    = {res.comm_link_latency * 1e6:.2f} µs\n"
                    f"pp_size                 = {snap['pp_size']}\n"
                    f"tp_comm_prefill_ms      = {t_tp_comm_prefill * 1000:.3f} ms\n"
                    f"tp_comm_decode_ms       = {t_tp_comm_decode * 1000:.3f} ms\n"
                    f"pp_comm_prefill_ms      = {t_pp_comm_prefill * 1000:.3f} ms\n"
                    f"pp_comm_decode_ms       = {t_pp_comm_decode * 1000:.3f} ms\n"
                    f"pp_bubble_prefill_ms    = {t_pp_bubble_prefill * 1000:.3f} ms\n"
                    f"pp_bubble_decode_ms     = {t_pp_bubble_decode * 1000:.3f} ms\n"
                )

            st.code(
                f"W (weight bytes)        = {W / 1e9:.3f} GB\n"
                f"C (peak compute)        = {eff_flops / 1e12:.1f} TFLOPS  "
                f"@ {compute_bits}-bit\n"
                f"MBW (memory bandwidth)  = {eff_mbw / 1e9:.0f} GB/s\n"
                f"batch_requested         = {batch}\n"
                f"batch_effective         = {bs_eff:.2f}  "
                + (f"(Hill: max={batch_saturation[0]}, 50%={batch_saturation[1]})"
                   if batch_saturation else "(no saturation)") + "\n"
                f"alpha (MFU preset)      = {alpha:.4f}\n"
                + alpha_eff_str
                + mla_str
                + f"beta (MBU)              = {beta:.4f}\n"
                f"kv_packing_eff          = {eng_spec['kv_packing_eff']}\n"
                + comm_str,
                language="text",
            )


# =============================================================================
# 📦 Справочник
# =============================================================================
with tab_pre:
    st.markdown("## 📖 Обучающий Справочник параметров и формул")

    with st.expander("📐 Теория: Математические формулы Roofline-модели"):
        st.markdown("### Roofline-модель LLM-инференса\n"
                    "Две независимые фазы, каждая ограничена `max(compute, memory)`.")
        st.latex(r"""
        t_{\text{prefill}} \;=\; \max\!\left(
          \underbrace{\tfrac{2\,N_{\text{active}}\,P_{\text{in}}\,\text{bs}}{C\cdot\alpha}}_{\text{compute path}},\;\;
          \underbrace{\tfrac{W}{\text{MBW}}}_{\text{at least one weight pass}}
        \right)
        """)
        st.latex(r"""
        t_{\text{decode}} \;=\; \max\!\left(
          \underbrace{\tfrac{W\cdot N_{\text{active}}/N}{\text{MBW}\cdot\beta}}_{\text{memory path (MoE-aware)}},\;\;
          \underbrace{\tfrac{2\,N_{\text{active}}\,\text{bs}}{C\cdot\alpha}}_{\text{compute path}}
        \right)
        \;+\; t_{\text{KV}}
        """)
        st.latex(r"""
        t_{\text{KV}} \;=\; \frac{L\cdot H_{kv}\cdot d_{\text{head}}\cdot (b_K + b_V)/8 \;\cdot\; \text{avg\_ctx}}{\text{MBW}\cdot\beta}
        """)
        st.latex(r"""
        \text{throughput} \;=\; \frac{\text{bs}_{\text{eff}}}{t_{\text{decode}}},
        \qquad
        \text{bs}_{\text{eff}} \;=\; \min\!\left(\text{bs},\;\max\!\big(1,\;\tfrac{\text{bs}_{\max}\cdot\text{bs}}{\text{bs}+\text{bs}_{50\%}}\big)\right)
        """)

        st.markdown("### Расшифровка переменных")
        df_vars = pd.DataFrame([
            ("N", "model parameters (e.g. 7e9 for 7B)", "штук"),
            ("N_active", "active parameters per token (MoE: routed experts only)", "штук"),
            ("W = N · bits / 8", "weight bytes", "B"),
            ("C", "peak FLOPS for the precision (HW spec)", "FLOPS"),
            ("MBW", "memory bandwidth (HW spec)", "B/s"),
            ("α", "Model FLOPs Utilization in prefill (engine-specific, 0.04–0.65)", "—"),
            ("β", "Memory Bandwidth Utilization in decode (engine-specific, 0.20–0.95)", "—"),
            ("bs_max, bs_50%", "Hill-curve batch saturation (continuous batching)", "—"),
            ("L", "transformer layers", "—"),
            ("H_kv, d_head", "KV heads × per-head dimension (GQA/MLA aware)", "—"),
            ("b_K, b_V", "bits per element for K/V cache (16/8/4)", "bits"),
            ("sliding_window", "max KV tokens kept (Mistral/Gemma/V4); else unbounded", "tokens"),
            ("tp_size, tp_eff", "tensor-parallel size & efficiency (multi-GPU)", "—"),
        ], columns=["символ", "что значит", "единица"])
        st.dataframe(df_vars, width="stretch", hide_index=True)

        st.info(
            "**Где в коде:** `emulator/formula.py::predict()` — single source of truth. "
            "Спецификации железа — `emulator/hardware.py`, дефолты движков — `emulator/engines.py`. "
            "Калиброванные перекрытия читаются из `results/calibrated_coefficients.csv`."
        )

    cur_hw = st.session_state["sb_hw"]
    cur_eng = st.session_state["sb_engine"]
    cur_model = st.session_state["sb_model"]

    st.subheader("💻 Hardware (HARDWARE_SPECS)")
    st.caption(f"Текущий выбор подсвечен: **{cur_hw}**")
    hw_rows = []
    for name, spec in HARDWARE_SPECS.items():
        hw_rows.append({
            "▶": "✅" if name == cur_hw else "",
            "name": name,
            "FP16 TFLOPS": spec["peak_tflops"][16],
            "FP8 TFLOPS": spec["peak_tflops"][8],
            "INT4 TFLOPS": spec["peak_tflops"][4],
            "MBW GB/s": spec["memory_bandwidth_gbs"],
            "VRAM GB": spec["memory_capacity_gb"],
            "TDP W": spec.get("tdp_w"),
            "tp_size": spec.get("tp_size", 1),
            "tp_eff": spec.get("tp_efficiency", 1.0),
        })
    st.dataframe(pd.DataFrame(hw_rows), width="stretch", hide_index=True)

    st.subheader("🔧 Движки (ENGINE_DEFAULTS)")
    st.caption(f"Текущий выбор подсвечен: **{cur_eng}**")
    eng_rows = []
    for name, e in ENGINE_DEFAULTS.items():
        bs = e.get("batch_saturation")
        eng_rows.append({
            "▶": "✅" if name == cur_eng else "",
            "engine": name,
            "α default": e["alpha"],
            "β default": e["beta"],
            "kv_packing_eff": e["kv_packing_eff"],
            "compute_path bits": e.get("compute_path", 16),
            "batch_saturation": f"({bs[0]}, {bs[1]})" if bs else "—",
            "notes": e["notes"],
        })
    st.dataframe(pd.DataFrame(eng_rows), width="stretch", hide_index=True)

    st.subheader("🧠 Архитектуры моделей (ARCH_DEFAULTS)")
    st.caption(f"Текущий выбор подсвечен: **{cur_model}B**")
    arch_rows = []
    for size, a in sorted(ARCH_DEFAULTS.items()):
        active = a.get("n_active_b")
        sw = a.get("sliding_window")
        arch_rows.append({
            "▶": "✅" if size == cur_model else "",
            "size B": size,
            "L (layers)": a["layers"],
            "d_model": a["d_model"],
            "kv_heads": a.get("kv_heads"),
            "head_dim": a.get("head_dim", 128),
            "n_active B (MoE)": f"{active}" if active is not None else "dense",
            "sliding_window": str(sw) if sw is not None else "—",
        })
    st.dataframe(pd.DataFrame(arch_rows), width="stretch", hide_index=True)

# =============================================================================
# 🔬 Аналитика
# =============================================================================
if is_adv and tab_sca is not None:
    with tab_sca:
        st.markdown("## 🔬 Раздел глубокой аналитики и калибровки")
        st.caption("Этот раздел доступен только в Продвинутом режиме. Здесь сгруппированы данные калибровки, результаты валидации и графики сходимости.")

        cal_sub, val_sub, pva_sub = st.tabs([
            "🎯 Калибровочные коэффициенты",
            "🧪 Валидация против бенчмарков",
            "📊 Predicted vs Actual (9.9k точек)"
        ])

        with cal_sub:
            cur_hw = st.session_state["sb_hw"]
            cur_eng = st.session_state["sb_engine"]
            cur_prec = st.session_state["sb_precision"]

            st.subheader("Откуда взялись α/β для текущего сочетания?")
            cal = lookup_calibration(cur_hw, cur_eng, cur_prec)
            if cal:
                col1, col2, col3 = st.columns(3)
                col1.metric("α median",
                            f"{cal['alpha']:.3f}" if cal["alpha"] is not None else "—")
                col2.metric("β median",
                            f"{cal['beta']:.3f}" if cal["beta"] is not None else "—")
                col3.metric("n измерений", cal["n_rows"])
                st.success(f"✅ Сочетание **{cur_hw} × {cur_eng} × {cur_prec}** откалибровано "
                           f"на {cal['n_rows']} реальных замерах.")
            else:
                eng_def = ENGINE_DEFAULTS[cur_eng]
                st.warning(
                    f"⚠️ Сочетание **{cur_hw} × {cur_eng} × {cur_prec}** "
                    f"не имеет калибровки. Используется literature default движка: "
                    f"α={eng_def['alpha']}, β={eng_def['beta']}. "
                    f"Точность предсказания может пострадать."
                )

            st.divider()

            st.subheader("Все откалиброванные сочетания")
            df_cal = load_calibration()
            if df_cal.empty:
                st.warning("`results/calibrated_coefficients.csv` не найден.")
            else:
                cols = st.columns(2)
                with cols[0]:
                    hw_filter = st.multiselect("Фильтр по железу",
                                               sorted(df_cal["hw"].unique()),
                                               default=list(df_cal["hw"].unique()))
                with cols[1]:
                    eng_filter = st.multiselect("Фильтр по движку",
                                                sorted(df_cal["backend"].unique()),
                                                default=list(df_cal["backend"].unique()))
                view = df_cal[df_cal["hw"].isin(hw_filter) & df_cal["backend"].isin(eng_filter)]
                st.dataframe(view, width="stretch", hide_index=True)
                st.caption(
                    f"{len(view)} строк / всего {len(df_cal)}. "
                    "p25/p75 — интерквартильный диапазон по сырым (α, β) с каждой "
                    "бенчмарк-точки."
                )

                st.markdown("### α по железу")
                chart = alt.Chart(view).mark_circle(size=120, opacity=0.7).encode(
                    x=alt.X("alpha_median:Q", title="α median"),
                    y=alt.Y("hw:N", sort=None),
                    color=alt.Color("backend:N", legend=alt.Legend(title="движок")),
                    size=alt.Size("n_rows:Q", title="n rows"),
                    tooltip=["hw", "backend", "precision_label", "alpha_median",
                             "beta_median", "n_rows"],
                ).properties(height=300)
                st.altair_chart(chart, width="stretch")

            st.subheader("Медианная ошибка по железу (calibrated vs uncalibrated)")
            df_err = load_error_summary()
            if not df_err.empty:
                st.dataframe(df_err, width="stretch", hide_index=True)
                st.caption(
                    "`prefill_med` / `decode_med` — медианная относительная ошибка "
                    "`|pred − obs| / obs`. Ближе к нулю → лучше. "
                    "`coarse` / `fine` — гранулярность калибровки (по precision vs +attention)."
                )

        with val_sub:
            st.subheader("Cross-check — формула vs внешние бенчмарки")
            st.caption(
                "Каждая точка — реальный опубликованный замер (или наш собственный) "
                "и предсказание формулы для того же сценария. "
                "См. `docs/BENCHMARK_SOURCES.md §2`."
            )

            df_val = pd.DataFrame(VALIDATION_POINTS)
            df_show = df_val.assign(
                Δ=df_val["delta_pct"].map(lambda x: f"{x:+.1f}%"),
            )[["id", "hw", "engine", "scenario", "metric", "real", "pred", "unit",
               "Δ", "verdict", "comment"]]
            st.dataframe(df_show, width="stretch", hide_index=True)

            st.markdown("### Δ (отклонение, %)")
            st.caption(
                "Отрицательный Δ = эмулятор недооценивает реальную метрику. "
                "Серые пунктиры — ±20% (рабочий коридор для roofline)."
            )
            chart = alt.Chart(df_val).mark_bar().encode(
                x=alt.X("delta_pct:Q", title="Δ %"),
                y=alt.Y("id:N", sort=None, title="benchmark id"),
                color=alt.condition(
                    alt.datum.delta_pct < 0,
                    alt.value("#f58518"),
                    alt.value("#54a24b"),
                ),
                tooltip=["id", "hw", "engine", "metric", "real", "pred",
                         alt.Tooltip("delta_pct:Q", format="+.1f")],
            ).properties(height=240)
            rule = alt.Chart(pd.DataFrame({"x": [-20, 20]})).mark_rule(
                strokeDash=[4, 4], color="#999"
            ).encode(x="x:Q")
            st.altair_chart(chart + rule, width="stretch")

            st.markdown("### Источники")
            for v in VALIDATION_POINTS:
                st.markdown(f"- **{v['id']}** — [{v['source']}]({v['url']})")

        with pva_sub:
            st.subheader("Predicted vs Actual — все 9.9k бенчмарк-точек")
            st.caption(
                "Каждая точка из LLM-Perf Leaderboard / собственных замеров. "
                "Идеальное предсказание лежит на диагонали."
            )

            df_pva = load_pred_vs_actual()
            if df_pva.empty:
                st.warning("`results/prediction_vs_actual.csv` не найден.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    modes = sorted(df_pva["mode"].unique())
                    preferred = next((m for m in ("fine", "coarse") if m in modes), modes[0])
                    mode = st.selectbox("Mode", modes, index=modes.index(preferred),
                                        help="`fine` — калибровка по (HW, engine, precision, attention). "
                                             "`coarse` — без attention. "
                                             "`uncalibrated` — literature default.")
                with c2:
                    hws = sorted(df_pva["hw"].unique())
                    hw_filter = st.multiselect("Hardware", hws, default=hws)
                with c3:
                    metric = st.radio("Метрика", ["prefill", "decode"], horizontal=True, key="pva_metric")
                with c4:
                    plot_type = st.radio("Тип графика", ["Scatter Plot", "Density Heatmap"], horizontal=True,
                                         index=1,
                                         help="Scatter Plot показывает отдельные точки (лимит 5k). Density Heatmap строит красивую 2D-гистограмму плотности для всех 9.9k точек без лагов.")

                sub = df_pva[(df_pva["mode"] == mode) & (df_pva["hw"].isin(hw_filter))].copy()
                if metric == "prefill":
                    sub = sub[(sub["prefill_obs"] > 0) & (sub["prefill_pred"] > 0)]
                    sub["obs"] = sub["prefill_obs"]
                    sub["pred"] = sub["prefill_pred"]
                    sub["rel_err"] = sub["prefill_rel_err"]
                    unit = "s"
                else:
                    sub = sub[(sub["decode_tps_obs"] > 0) & (sub["decode_tps_pred"] > 0)]
                    sub["obs"] = sub["decode_tps_obs"]
                    sub["pred"] = sub["decode_tps_pred"]
                    sub["rel_err"] = sub["decode_rel_err"]
                    unit = "tok/s"

                if sub.empty:
                    st.info("Нет данных под выбранные фильтры.")
                else:
                    import numpy as np
                    if plot_type == "Density Heatmap":
                        sub["log_obs"] = np.log10(sub["obs"])
                        sub["log_pred"] = np.log10(sub["pred"])
                        
                        heatmap = alt.Chart(sub).mark_rect().encode(
                            x=alt.X("log_obs:Q", bin=alt.Bin(maxbins=35), title=f"observed ({unit}, log10)"),
                            y=alt.Y("log_pred:Q", bin=alt.Bin(maxbins=35), title=f"predicted ({unit}, log10)"),
                            color=alt.Color("count():Q", scale=alt.Scale(scheme="viridis"), title="Точек в бине"),
                            tooltip=[
                                alt.Tooltip("count():Q", title="Точек в бине"),
                            ]
                        )
                        
                        lo = max(min(sub["log_obs"].min(), sub["log_pred"].min()), -4.0)
                        hi = max(sub["log_obs"].max(), sub["log_pred"].max())
                        diag = alt.Chart(pd.DataFrame({"x": [lo, hi]})).mark_line(
                            color="#f58518", strokeDash=[4, 4], strokeWidth=2
                        ).encode(x="x:Q", y="x:Q")
                        
                        st.altair_chart((heatmap + diag).properties(height=440), width="stretch")
                        st.caption(
                            f"Показано тепловое распределение всех {len(sub):,} бенчмарк-точек. "
                            "Оранжевый пунктир — идеальная диагональ."
                        )
                    else:
                        sub_capped = sub.head(5000)
                        scatter = alt.Chart(sub_capped).mark_circle(size=40, opacity=0.45).encode(
                            x=alt.X("obs:Q", scale=alt.Scale(type="log"),
                                    title=f"observed ({unit}, log)"),
                            y=alt.Y("pred:Q", scale=alt.Scale(type="log"),
                                    title=f"predicted ({unit}, log)"),
                            color=alt.Color("hw:N"),
                            tooltip=["hw", "model", "precision", "attention",
                                     alt.Tooltip("obs:Q", format=".3g"),
                                     alt.Tooltip("pred:Q", format=".3g"),
                                     alt.Tooltip("rel_err:Q", format=".2f")],
                        )
                        lo = max(min(sub_capped["obs"].min(), sub_capped["pred"].min()), 1e-4)
                        hi = max(sub_capped["obs"].max(), sub_capped["pred"].max())
                        diag = alt.Chart(pd.DataFrame({"x": [lo, hi]})).mark_line(
                            color="#888", strokeDash=[4, 4]
                        ).encode(x="x:Q", y="x:Q")
                        st.altair_chart((scatter + diag).properties(height=440),
                                        width="stretch")
                        st.caption(
                            f"Показано {len(sub_capped):,} из {len(sub):,} точек "
                            "(жёсткий cap 5k чтобы Altair не тормозил). "
                            "Диагональ — идеальное предсказание."
                        )

                    st.markdown("### Медианная ошибка по железу")
                    agg = sub.groupby("hw")["rel_err"].agg(["median", "count"]).reset_index()
                    agg.columns = ["hw", "median rel.err", "n"]
                    st.dataframe(agg, width="stretch", hide_index=True)
