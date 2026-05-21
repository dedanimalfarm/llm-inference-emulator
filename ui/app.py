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
    "sb_hw": "1xA100",
    "sb_engine": "vllm",
    "sb_model": 7.0,
    "sb_bits": 16,
    "sb_precision": "Unquantized",
    "sb_preset_active": None,
    "sb_custom_b": 0.0,
    "sb_custom_active": 0.0,
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
    st.caption("Эти настройки видят все вкладки.")

    st.markdown("**📌 Готовые сценарии**")
    st.caption("Один клик — заполнит всё ниже.")
    cols = st.columns(2)
    for i, name in enumerate(PRESETS):
        cols[i % 2].button(
            name, on_click=apply_preset, args=[name],
            width="stretch",
            type="primary" if st.session_state["sb_preset_active"] == name else "secondary",
        )

    st.divider()

    st.markdown("**🛠 Или вручную:**")
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
                      help="Используется при поиске α/β в "
                           "results/calibrated_coefficients.csv. "
                           "Типичные значения: Unquantized, AWQ.4bit, "
                           "GPTQ.4bit, BnB.4bit, Q4_K (4.91bpw).")
        st.number_input("Custom size (B), 0 = use preset", min_value=0.0,
                        max_value=2000.0, value=0.0, step=0.5, key="sb_custom_b",
                        help="Перебивает preset — полезно для нестандартных моделей.")
        st.number_input("Active params (B), MoE override (0 = preset)",
                        min_value=0.0, max_value=2000.0, value=0.0, step=0.5,
                        key="sb_custom_active")

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
        "p_in": int(s["wk_p_in"]),
        "p_out": int(s["wk_p_out"]),
        "batch": int(s["wk_batch"]),
        "kv_k": float(s["adv_kv_k"]),
        "kv_v": float(s["adv_kv_v"]),
        "sw": int(s["adv_sw"]),
        "head_dim": int(s["adv_hd"]),
        "cost_per_hour": float(s["adv_cost"]),
        "override_alpha": bool(s["adv_oa"]),
        "alpha_in": float(s["adv_alpha"]),
        "override_beta": bool(s["adv_ob"]),
        "beta_in": float(s["adv_beta"]),
        "alpha_sat_b0": float(s["adv_alpha_sat"]) if s["adv_alpha_sat"] > 0 else None,
        "mla_kv_lora_rank": int(s["adv_mla_rank"]) if s["adv_mla_rank"] > 0 else None,
        "mla_qk_rope_dim": int(s["adv_mla_rope"]) if s["adv_mla_rope"] > 0 else None,
        "pp_size": int(s["adv_pp"]),
        "comm_link": s["adv_comm_link"],
        "comm_link_bw": float(s["adv_comm_bw"]) if s["adv_comm_bw"] > 0 else None,
        "comm_link_latency": float(s["adv_comm_lat"]) / 1e3 if s["adv_comm_lat"] > 0 else None,
    }


# ---------------------------------------------------------------------------
# Header + tabs
# ---------------------------------------------------------------------------
st.title("🧮 LLM Inference Emulator")

tab_home, tab_pred, tab_form, tab_pre, tab_cal, tab_val, tab_sca = st.tabs([
    "🏠 Старт",
    "🎛 Эмуляция",
    "📐 Формула",
    "📦 Каталог",
    "🎯 Калибровка",
    "🧪 Валидация",
    "📊 Pred vs Actual",
])


# =============================================================================
# 🏠 Старт
# =============================================================================
with tab_home:
    st.markdown(
        """
### Один вопрос — один ответ

> *«Сколько tok/s выдаст моя модель X на железе Y, и хватит ли VRAM?»*

**Эмулятор отвечает за миллисекунды** — без аренды GPU, без скачивания
весов, без бенчмарка. Внутри — roofline-формула, откалиброванная против
реальных замеров (LLM-Perf Leaderboard PyTorch + наши собственные
llama-bench/vLLM прогоны на RTX-3090, A100, RTX-5090).

### Как пользоваться

1. **Слева в сайдбаре** выбери железо + движок + модель — или нажми
   готовый сценарий.
2. **Вкладка «🎛 Эмуляция»** — задай нагрузку (P_in / P_out / batch),
   получи prefill, decode, throughput, memory + почему именно это число.
3. **Вкладка «🎯 Калибровка»** — посмотри откуда взялись коэффициенты для
   твоего сочетания (HW × engine × precision).
4. **Вкладка «🧪 Валидация»** — насколько эмулятор сходится с реальными
   опубликованными бенчмарками.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("**±10-20%** внутри откалиброванной области\n\n"
                   "RTX-3090 / A100-40 / A10 / T4 / 2×5090 — real-world.")
    with c2:
        st.warning("**±30-60%** на extrapolation\n\n"
                   "Новое железо без калибровки (H100, B200) — формула "
                   "может недооценивать throughput.")
    with c3:
        st.info("**Не SLA-инструмент**\n\n"
                "Roofline даёт верхнюю границу. "
                "Финальный замер — на реальном железе.")

    st.divider()

    st.markdown("### 🚀 Быстрые сценарии")
    st.caption("Клик — заполнит сайдбар, переходи на «🎛 Эмуляция».")

    grid_cols = st.columns(3)
    descriptions = {
        "🤖 7B чатбот / A100": "Базовый сценарий: Llama-7B / Qwen-7B AWQ на одном A100. "
                              "Внутри откалиброванной области.",
        "🐘 70B Q4 / 2×3090": "Llama-70B Q4_K_M, требует 2 карты для capacity. "
                             "Pipeline-parallel (НЕ tensor-parallel).",
        "🧠 Mixtral MoE / 1×A100": "Mixtral 8x7B (46.7B total / 12.9B active). "
                                  "Декод читает только активные эксперты.",
        "⚡ Self-hosted / 2×5090": "Self-validation point (§2.3). "
                                  "Качество предсказания: −0.9% от реального.",
        "🔬 DeepSeek-V3 / H100": "Большой MoE (671B / 37B active) с MLA. "
                                "На 1 H100 не помещается, увидите warning.",
        "💾 Llama-8B BF16 / H100": "Validation point (§2.4). "
                                  "Известный gap: батч-сатурация не калиброванa для H100.",
    }
    for i, (name, desc) in enumerate(descriptions.items()):
        with grid_cols[i % 3]:
            st.button(name, key=f"home_{i}", on_click=apply_preset, args=[name],
                      width="stretch")
            st.caption(desc)


# =============================================================================
# 🎛 Эмуляция
# =============================================================================
with tab_pred:
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
        st.subheader("1️⃣ Нагрузка")
        c1, c2, c3 = st.columns(3)
        c1.number_input("📥 P_in (prompt tokens)", min_value=1, max_value=131072,
                        key="wk_p_in", step=128)
        c2.number_input("📤 P_out (output tokens)", min_value=1, max_value=8192,
                        key="wk_p_out", step=32)
        c3.number_input("📦 Batch size", min_value=1, max_value=2048,
                        key="wk_batch", step=1)

        with st.expander("⚙️ Дополнительно — KV bits, MLA, sliding window, $/час, ручной α/β"):
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.number_input("KV-K bits", step=4.0,
                                min_value=2.0, max_value=16.0, key="adv_kv_k")
                st.number_input("KV-V bits", step=4.0,
                                min_value=2.0, max_value=16.0, key="adv_kv_v")
            with a2:
                st.number_input("Sliding window (0 = unbounded)",
                                min_value=0, max_value=131072, key="adv_sw")
                st.number_input("head_dim (GQA)",
                                min_value=32, max_value=1024,
                                step=32, key="adv_hd")
            with a3:
                st.number_input("Cost $/hour (0 = off)",
                                min_value=0.0, step=0.5, key="adv_cost")
                st.number_input("Alpha Saturation b0 (0=off)",
                                min_value=0.0, step=1.0, key="adv_alpha_sat",
                                help="Задаёт параметр насыщения MFU батча b0.")
            with a4:
                st.checkbox("Override α", key="adv_oa")
                st.number_input("α", step=0.05, key="adv_alpha",
                                disabled=not st.session_state.get("adv_oa", False))
                st.checkbox("Override β", key="adv_ob")
                st.number_input("β", step=0.05, key="adv_beta",
                                disabled=not st.session_state.get("adv_ob", False))
            
            st.markdown("**Native Multi-Head Latent Attention (MLA) overrides (DeepSeek V3/V4):**")
            ma1, ma2 = st.columns(2)
            ma1.number_input("MLA KV Lora Rank (0 = GQA)", min_value=0, max_value=2048, step=64, key="adv_mla_rank",
                             help="Ранг латентного сжатия KV (DeepSeek-V3 = 512). При 0 используется стандартный GQA.")
            ma2.number_input("MLA QK RoPE Head Dim (0 = GQA)", min_value=0, max_value=512, step=16, key="adv_mla_rope",
                             help="Размерность decoupled Key RoPE проекции (DeepSeek-V3 = 64). При 0 используется стандартный GQA.")

            st.markdown("**Распределенное моделирование коммуникаций (TP/PP) & Сеть:**")
            net1, net2, net3, net4 = st.columns(4)
            net1.number_input("Pipeline Parallelism (PP) Size", min_value=1, max_value=64, step=1, key="adv_pp",
                              help="Количество стадий пайплайна (PP). Распределяет слои последовательно.")
            net2.selectbox("Интерконнект (Comm Link)", ["Auto", "none", "nvlink", "infinity_fabric", "pcie5", "pcie4", "pcie3", "ethernet"], key="adv_comm_link",
                           help="Тип сетевого соединения для моделирования задержек TP и PP. 'Auto' выбирает NVLink для серверных GPU, PCIe для десктопных.")
            net3.number_input("Полоса линка (Link BW, GB/s)", min_value=0.0, step=10.0, key="adv_comm_bw",
                              help="Перекрывает стандартную пропускную способность профиля (0.0 = по умолчанию для выбранного линка).")
            net4.number_input("Задержка линка (Link Latency, ms)", min_value=0.0, step=0.001, format="%.4f", key="adv_comm_lat",
                              help="Перекрывает стандартный пинг сетевого линка (0.0 = по умолчанию для выбранного линка).")

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
        st.markdown("**💾 Память:**")
        cap = hw_spec["memory_capacity_gb"]
        used = res.memory_gb
        pct = min(used / cap, 1.5)
        if used > cap:
            st.error(f"⛔ **{used:.1f} GB** требуется — НЕ влезает в {cap:.0f} GB. "
                     f"Перегрузка: {used / cap:.1f}× capacity. "
                     "Решения: меньшая модель, более глубокая квантизация (4-bit), "
                     "tensor-parallel на нескольких GPU, GPU с большей VRAM.")
        elif used > 0.85 * cap:
            st.warning(f"⚠️ **{used:.1f} GB** из {cap:.0f} GB ({100 * used / cap:.0f}%) "
                       "— впритык; для production нужен запас 15-20% на KV-всплески.")
        else:
            st.success(f"✅ **{used:.1f} GB** из {cap:.0f} GB ({100 * used / cap:.0f}%) — есть запас.")
        st.progress(min(used / cap, 1.0))

        # Cost
        if cost_per_hour > 0:
            per_million = cost_per_hour * 1e6 / (res.throughput_tok_s * 3600)
            st.info(f"💰 При **${cost_per_hour:.2f}/час** → "
                    f"**${per_million:.3f}** за миллион токенов "
                    f"(throughput {res.throughput_tok_s:.0f} tok/s).")

        # ---- Bottleneck explainer ----
        st.subheader("3️⃣ Что сейчас bottleneck?")
        st.caption("Roofline берёт **max(compute, memory)** — выигрывает медленный.")

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

        # Plain language interpretation
        interp_lines = []
        if res.bottleneck_decode == "memory":
            interp_lines.append(
                f"🟦 **Decode упирается в память** — что нормально для LLM. "
                f"Веса (**{W / 1e9:.1f} GB**) надо прокачать через {eff_mbw / 1e9:.0f} GB/s "
                f"на каждый сгенерированный токен. Ускорить: квантизация (меньший W), "
                f"GPU с большей MBW, или прирост batch (амортизация одного weight-pass на больше токенов)."
            )
        else:
            interp_lines.append(
                f"🟥 **Decode упирается в compute** — необычно. "
                f"Скорее всего batch={batch} большой и MFU доминирует. "
                f"Ускорить: меньший batch, более быстрое железо, "
                f"speculative decoding."
            )
        if res.bottleneck_prefill == "compute":
            interp_lines.append(
                f"🟧 **Prefill compute-bound** — ожидаемо для длинных промптов "
                f"(P_in={p_in}). Ускорить: prefix caching (prefix_cache_hit), "
                f"chunked prefill, более быстрый GPU."
            )
        else:
            interp_lines.append(
                f"🟦 **Prefill memory-bound** — короткий промпт (P_in={p_in}), "
                f"compute не успевает использоваться. Это типично при batch=1 "
                f"и коротких запросах."
            )
        if batch_saturation and bs_eff < batch:
            interp_lines.append(
                f"⚠️ **Эффективный batch = {bs_eff:.1f}** (запросили {batch}). "
                f"Hill-кривая континуального батчинга насыщается на bs_max={batch_saturation[0]}. "
                f"Дальнейший рост batch почти не даст throughput."
            )
        for line in interp_lines:
            st.markdown(line)

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
                p2p_note = "\n\n⚠️ **Host-Mediated TP PCIe:** На потребительских видеокартах (RTX 3090/4090/5090) драйвер NVIDIA блокирует прямой P2P DMA через PCIe. Обмен идет транзитом через RAM хоста, что срезает шину в ~2 раза и втрое завышает latency (учтено в расчёте)."

            st.warning(
                f"🌐 **Коммуникационные и распределенные задержки ({link_info}):**\n\n"
                f"- **TP All-Reduce (Ring):** Prefill = {t_tp_comm_prefill * 1000:.2f} ms, Decode = {t_tp_comm_decode * 1000:.3f} ms/token\n"
                f"- **PP stage-to-stage boundary:** Prefill = {t_pp_comm_prefill * 1000:.2f} ms, Decode = {t_pp_comm_decode * 1000:.3f} ms/token\n"
                f"- **PP 1F1B scheduling bubble:** Prefill = {t_pp_bubble_prefill * 1000:.2f} ms, Decode = {t_pp_bubble_decode * 1000:.3f} ms/token (простои стадий)\n\n"
                "Эти задержки добавлены поверх времени вычислений и памяти в соответствии с топологией Megatron-LM.\n\n"
                "💡 *Примечание:* Ring All-Reduce смоделирован как неперекрывающийся барьер синхронизации (upper-bound roofline limit). Реальные runtime-движки (vLLM/Megatron) за счет асинхронного pipelining (overlap) сокращают фактическую пенализацию в 2-3 раза."
                f"{p2p_note}"
            )

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
# 📐 Формула
# =============================================================================
with tab_form:
    st.markdown("## Roofline-модель LLM-инференса\n"
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

    st.markdown("### Расшифровка")
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


# =============================================================================
# 📦 Каталог
# =============================================================================
with tab_pre:
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
# 🎯 Калибровка
# =============================================================================
with tab_cal:
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


# =============================================================================
# 🧪 Валидация
# =============================================================================
with tab_val:
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


# =============================================================================
# 📊 Predicted vs Actual
# =============================================================================
with tab_sca:
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
            metric = st.radio("Метрика", ["prefill", "decode"], horizontal=True)
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
                # Compute log10 values for beautiful grid alignment
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
