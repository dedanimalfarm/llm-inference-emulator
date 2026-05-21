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

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* Global Styles and Background overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
        background: #080a0f !important;
        color: #e2e8f0 !important;
    }
    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        background: #0c0f16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    /* Elegant Title and Header styling */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #ffffff !important;
    }
    /* Metrics Enhancements with hover animations */
    div[data-testid="metric-container"] {
        background: rgba(18, 22, 32, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 15px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        background: rgba(18, 22, 32, 0.7) !important;
        border-color: rgba(0, 198, 255, 0.4);
        box-shadow: 0 10px 30px rgba(0, 198, 255, 0.15);
    }
    /* Glassmorphic boxes for alerts/success/info */
    div.element-container:has(div.stAlert) {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    /* Premium Streamlit expanders */
    div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(18, 22, 32, 0.4) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(0, 198, 255, 0.3) !important;
        box-shadow: 0 8px 30px rgba(0, 198, 255, 0.08) !important;
    }
    /* Premium Streamlit tabs active state indicators */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #00c6ff !important;
    }
    button[aria-selected="true"] {
        color: #00c6ff !important;
        border-bottom-color: #00c6ff !important;
    }
    /* Active step glowing card */
    .active-step-card {
        border: 1px solid rgba(0, 198, 255, 0.25) !important;
        box-shadow: 0 0 25px rgba(0, 198, 255, 0.08) !important;
        background: rgba(18, 22, 32, 0.6) !important;
        border-radius: 14px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-top: 15px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    /* Form styling and buttons styling */
    div[data-testid="stForm"] {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(18, 22, 32, 0.3);
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    /* Smooth button transitions */
    button[kind="primary"] {
        background: linear-gradient(135deg, #00c6ff, #0072ff) !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
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


def render_vram_bar(used_gb: float, capacity_gb: float, weights_gb: float, kv_gb: float, title_text: str = ""):
    """Render a premium gradient HTML VRAM breakdown bar."""
    ov_gb = min(2.0, max(0.5, capacity_gb * 0.05))
    total_used_gb = weights_gb + kv_gb + ov_gb
    free_gb = max(0.0, capacity_gb - total_used_gb)

    # Convert to percentages for styling
    w_pct = weights_gb / capacity_gb * 100
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
        bar_border = "2px solid #ff3b30"
    else:
        bar_border = "1px solid rgba(255,255,255,0.1)"

    # Labels
    w_label = f"Веса ({weights_gb:.1f} GB)" if w_pct > 12 else "Веса" if w_pct > 6 else ""
    kv_label = f"KV Cache ({kv_gb:.1f} GB)" if kv_pct > 12 else "KV" if kv_pct > 6 else ""
    ov_label = f"CUDA ({ov_gb:.1f} GB)" if ov_pct > 12 else "CUDA" if ov_pct > 6 else ""
    free_label = f"Свободно ({free_gb:.1f} GB)" if free_pct > 12 else "Своб." if free_pct > 6 else ""

    bar_html = f"""
    <div style="width: 100%; font-family: 'Outfit', 'Segoe UI', sans-serif; margin-bottom: 25px;">
        {f'<div style="font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #a0aec0;">{title_text}</div>' if title_text else ""}
        <div style="display: flex; width: 100%; height: 36px; border-radius: 10px; overflow: hidden; border: {bar_border}; background: #1e222b; box-shadow: 0 8px 24px rgba(0,0,0,0.15); transition: all 0.5s ease;">
            {f'<div style="width: {w_pct}%; background: linear-gradient(135deg, #11998e, #38ef7d); color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; border-right: 1px solid rgba(0,0,0,0.2); text-shadow: 0 1px 2px rgba(0,0,0,0.3);" title="Веса модели: {weights_gb:.2f} GB">{w_label}</div>' if w_pct > 0 else ""}
            {f'<div style="width: {kv_pct}%; background: linear-gradient(135deg, #00c6ff, #0072ff); color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; border-right: 1px solid rgba(0,0,0,0.2); text-shadow: 0 1px 2px rgba(0,0,0,0.3);" title="Кэш Key-Value контекстов: {kv_gb:.2f} GB">{kv_label}</div>' if kv_pct > 0 else ""}
            {f'<div style="width: {ov_pct}%; background: linear-gradient(135deg, #f12711, #f5af19); color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; border-right: 1px solid rgba(0,0,0,0.2); text-shadow: 0 1px 2px rgba(0,0,0,0.3);" title="Накладные расходы CUDA/Движка: {ov_gb:.2f} GB">{ov_label}</div>' if ov_pct > 0 else ""}
            {f'<div style="width: {free_pct}%; background: #2c303b; color: #8892b0; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600;" title="Свободная память VRAM: {free_gb:.2f} GB">{free_label}</div>' if free_pct > 0 else ""}
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 15px; justify-content: space-between; margin-top: 10px; font-size: 13px; color: #a0aec0; background: rgba(255,255,255,0.02); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
            <div><span style="color: #38ef7d; font-size: 14px; vertical-align: middle;">●</span> <b>Веса модели:</b> <span style="color: #fff; font-weight: 600;">{weights_gb:.2f} GB</span></div>
            <div><span style="color: #00c6ff; font-size: 14px; vertical-align: middle;">●</span> <b>KV Cache:</b> <span style="color: #fff; font-weight: 600;">{kv_gb:.2f} GB</span></div>
            <div><span style="color: #f5af19; font-size: 14px; vertical-align: middle;">●</span> <b>CUDA Оверхед:</b> <span style="color: #fff; font-weight: 600;">{ov_gb:.2f} GB</span></div>
            <div><span style="color: #8892b0; font-size: 14px; vertical-align: middle;">●</span> <b>Всего занято:</b> <span style="color: #fff; font-weight: 700;">{total_used_gb:.2f}</span> / <span style="color: #fff; font-weight: 700;">{capacity_gb:.0f} GB</span></div>
        </div>
    </div>
    """


def generate_attention_svg(mode: str) -> str:
    """Generate high-fidelity vector graphic (SVG) diagram dynamically for attention modes."""
    svg_filter = """<defs>
    <filter id="glow-cyan" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
        </feMerge>
    </filter>
    <filter id="glow-orange" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
        </feMerge>
    </filter>
    <filter id="glow-purple" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="5" result="blur" />
        <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
        </feMerge>
    </filter>
</defs>"""

    width = 720
    height = 240
    
    q_y = 50
    kv_y = 180
    q_x = [60 + i * 85 for i in range(8)]
    
    content = []
    
    # Draw Background
    content.append(f'<rect width="{width}" height="{height}" fill="rgba(10, 14, 23, 0.9)" rx="16" stroke="rgba(255,255,255,0.06)" stroke-width="1.5"/>')
    
    # Title overlay
    content.append(f'<text x="20" y="30" fill="#a0aec0" font-size="12" font-family="system-ui, sans-serif" font-weight="700" letter-spacing="1">{mode} SCHEMA</text>')

    if mode == "MHA":
        # Connections
        for i in range(8):
            content.append(f'<line x1="{q_x[i]}" y1="{q_y}" x2="{q_x[i]}" y2="{kv_y}" stroke="rgba(0, 198, 255, 0.4)" stroke-width="2"/>')
        # Q Heads
        for i in range(8):
            content.append(f'<circle cx="{q_x[i]}" cy="{q_y}" r="10" fill="#00c6ff" filter="url(#glow-cyan)"/>')
            content.append(f'<text x="{q_x[i]}" y="{q_y - 18}" fill="#a0aec0" font-size="10" text-anchor="middle" font-family="system-ui, sans-serif">Q{i+1}</text>')
        # KV Heads
        for i in range(8):
            content.append(f'<circle cx="{q_x[i]}" cy="{kv_y}" r="10" fill="#ff9f43" filter="url(#glow-orange)"/>')
            content.append(f'<text x="{q_x[i]}" y="{kv_y + 22}" fill="#a0aec0" font-size="10" text-anchor="middle" font-family="system-ui, sans-serif">KV{i+1}</text>')
            
    elif mode == "GQA":
        # Group 1: Q1..4 -> KV1, Group 2: Q5..8 -> KV2
        kv_x = [187, 527]
        
        # Connections group 1
        for i in range(4):
            content.append(f'<line x1="{q_x[i]}" y1="{q_y}" x2="{kv_x[0]}" y2="{kv_y}" stroke="rgba(255, 159, 67, 0.4)" stroke-width="2"/>')
        # Connections group 2
        for i in range(4, 8):
            content.append(f'<line x1="{q_x[i]}" y1="{q_y}" x2="{kv_x[1]}" y2="{kv_y}" stroke="rgba(255, 159, 67, 0.4)" stroke-width="2"/>')
            
        # Q Heads
        for i in range(8):
            content.append(f'<circle cx="{q_x[i]}" cy="{q_y}" r="10" fill="#00c6ff" filter="url(#glow-cyan)"/>')
            content.append(f'<text x="{q_x[i]}" y="{q_y - 18}" fill="#a0aec0" font-size="10" text-anchor="middle" font-family="system-ui, sans-serif">Q{i+1}</text>')
        # KV Heads
        for i in range(2):
            content.append(f'<circle cx="{kv_x[i]}" cy="{kv_y}" r="12" fill="#ff9f43" filter="url(#glow-orange)"/>')
            content.append(f'<text x="{kv_x[i]}" y="{kv_y + 24}" fill="#a0aec0" font-size="10" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="bold">Group {i+1} KV</text>')
            
    elif mode == "MQA":
        # All Q -> 1 KV at center
        kv_center_x = 357
        for i in range(8):
            content.append(f'<line x1="{q_x[i]}" y1="{q_y}" x2="{kv_center_x}" y2="{kv_y}" stroke="rgba(255, 56, 56, 0.35)" stroke-width="1.8"/>')
            
        # Q Heads
        for i in range(8):
            content.append(f'<circle cx="{q_x[i]}" cy="{q_y}" r="10" fill="#00c6ff" filter="url(#glow-cyan)"/>')
            content.append(f'<text x="{q_x[i]}" y="{q_y - 18}" fill="#a0aec0" font-size="10" text-anchor="middle" font-family="system-ui, sans-serif">Q{i+1}</text>')
        # Single KV Head
        content.append(f'<circle cx="{kv_center_x}" cy="{kv_y}" r="14" fill="#ff3838" filter="url(#glow-orange)"/>')
        content.append(f'<text x="{kv_center_x}" y="{kv_y + 26}" fill="#a0aec0" font-size="10" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="bold">Shared Single KV</text>')
        
    elif mode == "MLA":
        # DeepSeek Latent Bottleneck at center
        latent_x = 357
        latent_y = 115
        
        # Connect all Q to Latent Bottleneck
        for i in range(8):
            content.append(f'<line x1="{q_x[i]}" y1="{q_y}" x2="{latent_x}" y2="{latent_y}" stroke="rgba(224, 86, 253, 0.45)" stroke-width="2"/>')
            
        # Connect Latent Bottleneck to all reconstructed KV heads
        for i in range(8):
            content.append(f'<line x1="{latent_x}" y1="{latent_y}" x2="{q_x[i]}" y2="{kv_y}" stroke="rgba(255, 159, 67, 0.4)" stroke-width="1.5" stroke-dasharray="3,3"/>')
            
        # Q Heads
        for i in range(8):
            content.append(f'<circle cx="{q_x[i]}" cy="{q_y}" r="10" fill="#00c6ff" filter="url(#glow-cyan)"/>')
            content.append(f'<text x="{q_x[i]}" y="{q_y - 18}" fill="#a0aec0" font-size="10" text-anchor="middle" font-family="system-ui, sans-serif">Q{i+1}</text>')
            
        # Latent Bottleneck circle
        content.append(f'<circle cx="{latent_x}" cy="{latent_y}" r="18" fill="#e056fd" filter="url(#glow-purple)"/>')
        content.append(f'<text x="{latent_x}" y="{latent_y + 4}" fill="#fff" font-size="9" text-anchor="middle" font-family="system-ui, sans-serif" font-weight="bold">c_KV</text>')
        content.append(f'<text x="{latent_x + 105}" y="{latent_y + 4}" fill="#e056fd" font-size="9" font-family="system-ui, sans-serif" font-weight="bold">Latent Bottleneck (Compressed VRAM)</text>')
        
        # Reconstructed KV Heads
        for i in range(8):
            content.append(f'<circle cx="{q_x[i]}" cy="{kv_y}" r="7" fill="#ff9f43" filter="url(#glow-orange)"/>')
            content.append(f'<text x="{q_x[i]}" y="{kv_y + 18}" fill="#a0aec0" font-size="9" text-anchor="middle" font-family="system-ui, sans-serif">Proj KV{i+1}</text>')

    svg_str = f'<svg width="100%" height="100%" viewBox="0 0 {width} {height}" fill="none" xmlns="http://www.w3.org/2000/svg">{svg_filter}{"".join(content)}</svg>'
    return svg_str


# ---------------------------------------------------------------------------
# Presets — one click to fill the whole sidebar
# ---------------------------------------------------------------------------
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

tabs_list = ["🏠 Эмулятор", "🔌 Анатомия GPU", "📦 Справочник"]
if is_adv:
    tabs_list.append("🔬 Аналитика")

tabs = st.tabs(tabs_list)
tab_pred = tabs[0]
tab_anat = tabs[1]
tab_pre = tabs[2]
if is_adv:
    tab_sca = tabs[3]
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
        cap = hw_spec["memory_capacity_gb"]
        used = res.memory_gb
        
        weights_gb = snap["n_params_b"] * snap["bits"] / 8.0
        kv_gb = max(0.0, used - weights_gb)

        # Draw beautiful gradient VRAM bar instead of plain progress
        render_vram_bar(used, cap, weights_gb, kv_gb, title_text="💾 Интерактивная карта распределения памяти VRAM")

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

        # Smart Optimization Assistant
        st.markdown("### 📈 Советник по оптимизации инференса")
        
        # Analyze performance bottleneck
        is_oom = used > cap
        
        if is_oom:
            st.markdown(
                """
                <div style="background: rgba(255, 59, 48, 0.08); border: 1px solid rgba(255, 59, 48, 0.2); border-radius: 12px; padding: 18px 22px; margin-bottom: 25px;">
                    <h5 style="color: #ff453a; margin-top: 0; margin-bottom: 8px; font-weight: 700; font-size: 16px;">🚨 Критический дефицит памяти (Out of Memory)</h5>
                    <p style="color: #ff9f0a; font-size: 13.5px; margin: 0 0 10px 0; line-height: 1.5;">Суммарный объем весов и KV-кэша превышает физический VRAM вашего GPU. Выберите один из следующих путей для решения:</p>
                    <ul style="color: #f5f5f7; font-size: 13px; margin: 0; padding-left: 20px; line-height: 1.6;">
                        <li><b>Квантование весов:</b> Снизьте разрядность весов модели в сайдбаре слева (например, с 16 до 4 или 8 бит). Это сократит размер модели во VRAM в 2-4 раза.</li>
                        <li><b>Снижение Batch Size:</b> Уменьшите размер батча в форме нагрузки. KV-кэш масштабируется линейно с ростом батча.</li>
                        <li><b>Параллельный инференс (TP/PP):</b> Включите Tensor Parallelism (TP) или Pipeline Parallelism (PP) в расширенных параметрах, чтобы распределить веса и KV-кэш по нескольким видеокартам.</li>
                        <li><b>Квантование KV-кэша:</b> В расширенных параметрах в разделе "Оптимизации KV-Кэша" понизьте разрядность хранения Key/Value (например, до 4 или 8 бит).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            prefill_bot = res.bottleneck_prefill
            decode_bot = res.bottleneck_decode
            
            prefill_text = ""
            decode_text = ""
            
            if prefill_bot == "memory":
                prefill_text = "<li><b>Prefill (Анализ промпта) ограничен памятью (Memory-bound):</b> Ваш промпт слишком короткий или батч мал, чтобы загрузить ядра GPU. Вы можете увеличить батч для более эффективной утилизации чипа или применить квантование весов, чтобы повысить скорость чтения.</li>"
            else:
                prefill_text = "<li><b>Prefill ограничен вычислениями (Compute-bound):</b> Ядра GPU загружены на максимум. Для ускорения используйте более мощный GPU, включите <b>FlashAttention</b> для снижения накладных расходов внимания или примените тензорный параллелизм (TP).</li>"
                
            if decode_bot == "memory":
                decode_text = "<li><b>Decode (Генерация ответа) ограничен памятью (Memory-bound):</b> Это стандартное поведение авторегрессионного инференса, так как веса перечитываются из VRAM для каждого токена. Для кратного роста скорости генерации используйте:<ul><li><b>Квантование весов (AWQ, GPTQ, GGUF):</b> Уменьшит размер весов, считываемых из VRAM на каждом шаге.</li><li><b>Сжатие KV-кэша (GQA, MLA):</b> Значительно разгружает шину памяти от передачи контекста на больших батчах.</li><li><b>Увеличение Batch Size:</b> Переиспользует считанные из VRAM веса модели для многих пользователей одновременно, драматически повышая совокупную пропускную способность (Throughput).</li></ul></li>"
            else:
                decode_text = "<li><b>Decode ограничен вычислениями (Compute-bound):</b> Ваш батч настолько велик (или модель огромна), что ядра GPU перегружены вычислениями, а не чтением памяти. Вы упёрлись в пиковую производительность. Для ускорения используйте более мощный GPU или распределите вычисления (TP/PP).</li>"
                
            st.markdown(
                f"""
                <div style="background: rgba(0, 198, 255, 0.05); border: 1px solid rgba(0, 198, 255, 0.15); border-radius: 12px; padding: 18px 22px; margin-bottom: 25px;">
                    <h5 style="color: #00c6ff; margin-top: 0; margin-bottom: 10px; font-weight: 700; font-size: 16px;">💡 Анализ производительности и рекомендации</h5>
                    <ul style="color: #f5f5f7; font-size: 13px; margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 10px; line-height: 1.6;">
                        {prefill_text}
                        {decode_text}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )


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
            render_vram_bar(total_used_gb, capacity_gb, w_gb, kv_gb)
            
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
                
                # --- 🧠 Interactive Attention Visualizer ---
                st.markdown("---")
                st.markdown("#### 🧠 Интерактивное сравнение механизмов внимания (MHA vs GQA vs MQA vs MLA)")
                st.caption(
                    "Современные LLM оптимизируют хранение KV-кэша в памяти VRAM с помощью различных архитектур группировки и сжатия голов внимания. "
                    "Переключайте режимы ниже, чтобы увидеть их физическое устройство в виде интерактивной схемы и узнать размер занимаемой памяти для вашей текущей нагрузки."
                )
                
                att_mode = st.radio(
                    "Выберите механизм внимания для визуализации:",
                    ["Multi-Head Attention (MHA)", "Grouped-Query Attention (GQA)", "Multi-Query Attention (MQA)", "Multi-Head Latent Attention (MLA)"],
                    horizontal=True,
                    key="att_visualizer_mode"
                )
                
                selected_mode_key = "MHA"
                if "GQA" in att_mode:
                    selected_mode_key = "GQA"
                elif "MQA" in att_mode:
                    selected_mode_key = "MQA"
                elif "MLA" in att_mode:
                    selected_mode_key = "MLA"
                    
                st.markdown(generate_attention_svg(selected_mode_key), unsafe_allow_html=True)
                
                # Math footprints calculation side-by-side
                # Let's get active query heads count:
                d_model = arch_temp.get("d_model", 4096)
                n_heads_act = d_model // hd_dim if hd_dim > 0 else 32
                if n_heads_act <= 0:
                    n_heads_act = 32
                
                # Calculate KV bytes per token for all modes
                mha_tok_b = L_layers * n_heads_act * hd_dim * 2 * (16 / 8.0) # assuming fp16
                gqa_tok_b = L_layers * max(1, n_heads_act // 8) * hd_dim * 2 * (16 / 8.0)
                mqa_tok_b = L_layers * 1 * hd_dim * 2 * (16 / 8.0)
                mla_tok_b = L_layers * (512 + 64) * (16 / 8.0) # latent rank 512 + 64 rope dim
                
                # Calculate total GB for each mode using current batch, context, and packing efficiency
                mha_total_gb = mha_tok_b * ctx_total * batch / kv_eff_pct / 1e9
                gqa_total_gb = gqa_tok_b * ctx_total * batch / kv_eff_pct / 1e9
                mqa_total_gb = mqa_tok_b * ctx_total * batch / kv_eff_pct / 1e9
                mla_total_gb = mla_tok_b * ctx_total * batch / kv_eff_pct / 1e9
                
                # Render comparison scorecard metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("MHA (Базовый)", f"{mha_total_gb * 1024:.1f} MB", "0% сжатие", delta_color="off")
                with col_m2:
                    savings_gqa = (1.0 - gqa_total_gb / mha_total_gb) * 100 if mha_total_gb > 0 else 0
                    st.metric("GQA (Llama-3)", f"{gqa_total_gb * 1024:.1f} MB", f"-{savings_gqa:.0f}% памяти", delta_color="inverse")
                with col_m3:
                    savings_mqa = (1.0 - mqa_total_gb / mha_total_gb) * 100 if mha_total_gb > 0 else 0
                    st.metric("MQA (Falcon)", f"{mqa_total_gb * 1024:.1f} MB", f"-{savings_mqa:.0f}% памяти", delta_color="inverse")
                with col_m4:
                    savings_mla = (1.0 - mla_total_gb / mha_total_gb) * 100 if mha_total_gb > 0 else 0
                    st.metric("MLA (DeepSeek)", f"{mla_total_gb * 1024:.1f} MB", f"-{savings_mla:.0f}% памяти", delta_color="inverse")
                
                st.info(
                    "💡 **Физический вывод:** Внедрение GQA (в Llama-3, Mistral) дает кратное сокращение кэша KV. "
                    "Внедрение MLA (DeepSeek-V3) сжимает KV-кэш почти в 15 раз относительно MHA, "
                    "что позволяет обрабатывать гигантские контексты и обслуживать миллионы пользователей без перегрузки VRAM!"
                )
                st.markdown("---")
                
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

                # Custom model architecture override in the sandbox
                with st.expander("🔧 Настроить архитектуру экспериментальной модели (Advanced)"):
                    wi_custom_arch = st.checkbox("Переопределить архитектуру модели в песочнице", value=False, key="wi_custom_arch")
                    if wi_custom_arch:
                        wi_layers = st.slider("Количество слоев модели (L)", 8, 128, int(L_layers), key="wi_layers_val")
                        wi_h_kv = st.slider("Количество KV-голов модели (H_kv)", 1, 64, int(H_kv), key="wi_h_kv_val")
                        wi_hd_dim = st.slider("Размерность головы (head_dim)", 32, 256, int(hd_dim), step=16, key="wi_hd_dim_val")
                        wi_is_mla = st.checkbox("Использовать MLA сжатие KV", value=is_mla, key="wi_is_mla_val")
                        if wi_is_mla:
                            wi_mla_rank = st.slider("Ранг сжатия KV (MLA Lora Rank)", 64, 1024, int(snap.get("mla_kv_lora_rank") or 512), step=64, key="wi_mla_rank_val")
                            wi_mla_rope = st.slider("Размерность QK RoPE", 16, 256, int(snap.get("mla_qk_rope_dim") or 64), step=16, key="wi_mla_rope_val")
                        else:
                            wi_mla_rank = 0
                            wi_mla_rope = 0
                    else:
                        wi_layers = L_layers
                        wi_h_kv = H_kv
                        wi_hd_dim = hd_dim
                        wi_is_mla = is_mla
                        wi_mla_rank = snap.get("mla_kv_lora_rank")
                        wi_mla_rope = snap.get("mla_qk_rope_dim")

                # Recalculate everything for What-If
                wi_ctx_total = wi_pin + wi_pout
                if sliding_window is not None:
                    wi_ctx_total = min(wi_ctx_total, sliding_window)
                
                # Calculate weights size if custom architecture is overridden
                if wi_custom_arch:
                    scale_w = wi_layers / L_layers
                    wi_W = W * scale_w
                    wi_w_gb = wi_W / 1e9
                else:
                    wi_W = W
                    wi_w_gb = w_gb

                # KV Cache on single token in what-if
                if wi_is_mla and wi_mla_rank and wi_mla_rope:
                    wi_kv_tok_b = wi_layers * (wi_mla_rank + wi_mla_rope) * kv_k / 8.0
                    wi_tok_formula_str = f"{wi_layers} \\times ({wi_mla_rank} + {wi_mla_rope}) \\times \\frac{{{kv_k}}}{{8}}"
                else:
                    wi_kv_tok_b = wi_layers * wi_h_kv * wi_hd_dim * (kv_k + kv_v) / 8.0
                    wi_tok_formula_str = f"{wi_layers} \\times {wi_h_kv} \\times {wi_hd_dim} \\times \\frac{{{kv_k} + {kv_v}}}{{8}}"
                
                wi_kv_bytes = wi_kv_tok_b * wi_ctx_total * wi_batch / kv_eff_pct
                wi_kv_gb = wi_kv_bytes / 1e9
                wi_total_vram = wi_w_gb + wi_kv_gb + ov_gb

                # Intensities for what-if
                wi_intensity_prefill = (2.0 * N_active * wi_batch * wi_pin) / wi_W if wi_W > 0 else 1.0
                wi_intensity_decode = (2.0 * N_active * wi_batch) / wi_W if wi_W > 0 else 1.0

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
                KV_{{\text{{total}}}} \;=\; \frac{{{wi_tok_formula_str} \times {wi_ctx_total}\text{{ ток}} \times {wi_batch}}}{{{kv_eff_pct}}} \;=\; {wi_kv_gb:.3f}\text{{ GB}}
                """)
                
                if wi_total_vram > capacity_gb:
                    st.error(f"❌ **OOM! При этих параметрах вы выйдете за лимит памяти GPU:** требуется **{wi_total_vram:.1f} GB** (веса: {wi_w_gb:.1f} GB, KV-кэш: {wi_kv_gb:.1f} GB) при ёмкости **{capacity_gb:.0f} GB**.")
                else:
                    st.success(f"✅ **Модель поместится в VRAM:** требуется **{wi_total_vram:.1f} GB** (веса: {wi_w_gb:.1f} GB, KV-кэш: {wi_kv_gb:.1f} GB, свободно еще **{capacity_gb - wi_total_vram:.1f} GB**).")
            
            with edu_tab2:
                st.markdown("### Визуализация Roofline-модели")
                st.caption(
                    "На этом интерактивном графике показана физическая граница возможностей выбранного GPU. "
                    "Зеленая линия — пиковый теоретический лимит. "
                    "Звезды показывают, где именно находятся фазы Prefill и Decode для вашей нагрузки."
                )
                import numpy as np
                # Interactive multiselect to compare with other GPUs on the Roofline chart
                gpus_to_compare = st.multiselect(
                    "🔍 Сравнить с другими графическими процессорами на Roofline-графике:",
                    list(HARDWARE_SPECS.keys()),
                    default=[snap["hw"]],
                    help="Вы можете выбрать несколько видеокарт из списка, чтобы наложить их теоретические границы (Ceilings) друг на друга для наглядного сравнения."
                )
                
                # Make sure the currently active GPU is always included
                if snap["hw"] not in gpus_to_compare:
                    gpus_to_compare = [snap["hw"]] + gpus_to_compare

                roof_rows = []
                xs = np.logspace(-1, 4, 100)
                for hw_name in gpus_to_compare:
                    hw_flops = get_peak_compute(hw_name, compute_bits)
                    hw_bw = get_memory_bandwidth(hw_name)
                    for x in xs:
                        y = min(hw_flops, hw_bw * x) / 1e12
                        roof_rows.append({
                            "Intensity": x,
                            "Performance": y,
                            "GPU": hw_name
                        })
                roof_df = pd.DataFrame(roof_rows)
                
                prefill_flops = 2.0 * N_active * p_in * bs_eff
                prefill_tflops = (prefill_flops / res.prefill_s) / 1e12 if res.prefill_s > 0 else 0.0
                
                decode_flops = 2.0 * N_active * bs_eff
                decode_tflops = (decode_flops / res.decode_per_token_s) / 1e12 if res.decode_per_token_s > 0 else 0.0
                
                pts_df = pd.DataFrame([
                    {"Intensity": intensity_prefill, "Performance": prefill_tflops, "Phase": "⭐ Prefill Phase (Prompt)"},
                    {"Intensity": intensity_decode, "Performance": decode_tflops, "Phase": "⭐ Decode Phase (Generation)"}
                ])
                
                line_chart = alt.Chart(roof_df).mark_line(strokeWidth=3).encode(
                    x=alt.X("Intensity:Q", scale=alt.Scale(type="log"), title="Арифметическая интенсивность (FLOP / Byte)"),
                    y=alt.Y("Performance:Q", scale=alt.Scale(type="log"), title="Производительность (TFLOPS)"),
                    color=alt.Color("GPU:N", title="Модели GPU", scale=alt.Scale(scheme="category10")),
                    tooltip=["GPU", alt.Tooltip("Intensity:Q", format=".1f"), alt.Tooltip("Performance:Q", format=".2f")]
                )
                
                pts_chart = alt.Chart(pts_df).mark_point(size=220, filled=True, color="#d62728").encode(
                    x="Intensity:Q",
                    y="Performance:Q",
                    color=alt.Color("Phase:N", legend=alt.Legend(title="Фазы вашей нагрузки")),
                    tooltip=["Phase", alt.Tooltip("Intensity:Q", format=".2f"), alt.Tooltip("Performance:Q", format=".2f")]
                )
                
                st.altair_chart((line_chart + pts_chart).properties(height=380), width="stretch")
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
                st.markdown("### 🔄 Жизненный цикл обработки вашего запроса в GPU")
                st.caption("Нажимайте на вкладки шагов ниже, чтобы изучить подробную физику и математику инференса на каждом этапе прохождения запроса.")

                # Initialize stepper step
                if "lifecycle_step" not in st.session_state:
                    st.session_state["lifecycle_step"] = 0

                # Render horizontal 5-column navigation stepper
                cols_step = st.columns(5)
                steps_meta = [
                    ("📥 1. Токенизация", 0),
                    ("⚡ 2. Prefill", 1),
                    ("💾 3. KV Cache", 2),
                    ("🔄 4. Decode", 3),
                    ("📤 5. Вывод", 4)
                ]
                
                for label, idx in steps_meta:
                    with cols_step[idx]:
                        btn_type = "primary" if st.session_state["lifecycle_step"] == idx else "secondary"
                        if st.button(label, key=f"step_btn_{idx}", use_container_width=True, type=btn_type):
                            st.session_state["lifecycle_step"] = idx
                            st.rerun()

                # Get active step
                active_step = st.session_state["lifecycle_step"]

                # Resolve hidden parameters
                d_model = arch_temp.get("d_model", 4096)
                prefill_flops_total = 2.0 * N_active * p_in * bs_eff / 1e12

                # Render active step card
                if active_step == 0:
                    st.markdown(
                        f"""
                        <div class="active-step-card">
                            <h3 style="margin-top:0; color:#00c6ff; border:none !important;">📥 Шаг 1: Токенизация (Tokenization)</h3>
                            <p style="color:#e2e8f0; font-size:14.5px; line-height:1.6;">
                                Входной текстовый промпт разбивается на элементарные токены (словоформы, слоги или отдельные символы) 
                                с помощью словаря токенизатора. Графический чип (GPU) не может работать напрямую со строками — 
                                он оперирует исключительно числовыми идентификаторами токенов.
                            </p>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; margin-top:15px;">
                                <ul style="margin:0; padding-left:20px; color:#a0aec0; font-size:14px; display:flex; flex-direction:column; gap:8px;">
                                    <li>📝 <b>Длина вашего промпта (P_in):</b> <span style="color:#fff; font-weight:600;">{p_in} токенов</span></li>
                                    <li>🧩 <b>Параллельный батч (Batch):</b> <span style="color:#fff; font-weight:600;">{batch} параллельных запросов</span></li>
                                    <li>🧠 <b>Размер скрытого слоя модели (d_model):</b> <span style="color:#fff; font-weight:600;">{d_model} параметров</span></li>
                                    <li>⚙️ <b>Аппаратный процесс:</b> Числа-токены переводятся в плотные векторы (Embeddings) размерности <code>{d_model}</code> путем аппаратного поиска в таблице весов эмбеддингов. Этот процесс занимает ничтожно мало времени во VRAM.</li>
                                </ul>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif active_step == 1:
                    st.markdown(
                        f"""
                        <div class="active-step-card">
                            <h3 style="margin-top:0; color:#00c6ff; border:none !important;">⚡ Шаг 2: Фаза Prefill (Насыщение / Обработка промпта)</h3>
                            <p style="color:#e2e8f0; font-size:14.5px; line-height:1.6;">
                                Графический процессор считывает веса модели и обрабатывает абсолютно все <b>{p_in} токенов</b> входного текста параллельно. 
                                Это высокоэффективный процесс с точки зрения утилизации вычислительных мощностей GPU (высокая параллельная утилизация ядер).
                            </p>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; margin-top:15px;">
                                <ul style="margin:0; padding-left:20px; color:#a0aec0; font-size:14px; display:flex; flex-direction:column; gap:8px;">
                                    <li>🧮 <b>Объем вычислений:</b> <span style="color:#fff; font-weight:600;">{prefill_flops_total:.3f} TFLOPs</span> операций (параллельное умножение матриц по всей длине промпта)</li>
                                    <li>🕒 <b>Задержка выполнения Prefill:</b> <span style="color:#fff; font-weight:600;">{res.prefill_s * 1000:.1f} мс</span></li>
                                    <li>📈 <b>Арифметическая интенсивность:</b> <span style="color:#00c6ff; font-weight:700;">{intensity_prefill:.1f} FLOP / byte</span> (высокая, ядра полностью загружены)</li>
                                    <li>🚨 <b>Физический лимит:</b> фаза упирается в <span style="color:#e74c3c; font-weight:700;">{res.bottleneck_prefill.upper()}</span></li>
                                </ul>
                            </div>
                            <p style="color:#a0aec0; font-size:13px; margin-top:15px; font-style:italic;">
                                💡 Примечание: Поскольку мы перемножаем матрицы активаций промпта на веса модели одновременно для всех токенов, интенсивность превосходит «точку перелома» (Ridge Point) на многих GPU, поэтому фаза Prefill часто утилизирует чистые TFLOPS ядер GPU.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif active_step == 2:
                    st.markdown(
                        f"""
                        <div class="active-step-card">
                            <h3 style="margin-top:0; color:#00c6ff; border:none !important;">💾 Шаг 3: Запись Key-Value кэша (KV Cache Storage)</h3>
                            <p style="color:#e2e8f0; font-size:14.5px; line-height:1.6;">
                                Чтобы не вычислять квадратичное самовнимание заново на каждом шаге генерации, векторы Ключей (Key) и Значений (Value) 
                                для всех слоев модели сохраняются во VRAM GPU. Это критическая точка масштабирования по памяти.
                            </p>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; margin-top:15px;">
                                <ul style="margin:0; padding-left:20px; color:#a0aec0; font-size:14px; display:flex; flex-direction:column; gap:8px;">
                                    <li>📐 <b>Вес KV-кэша на токен:</b> <span style="color:#fff; font-weight:600;">{kv_tok_b:.0f} байт / токен</span></li>
                                    <li>💾 <b>Всего выделено во VRAM:</b> <span style="color:#00c6ff; font-weight:700;">{kv_total / 1e6:.1f} MB</span> (для всего батча из {batch} запросов)</li>
                                    <li>🧩 <b>Утилизация кэша (PagedAttention):</b> <span style="color:#fff; font-weight:600;">{kv_eff_pct:.0%} эффективного заполнения</span></li>
                                    <li>🌐 <b>Скорость шины VRAM:</b> <span style="color:#fff; font-weight:600;">{eff_mbw / 1e9:.0f} GB/s</span> (лимитирует скорость записи и чтения кэша)</li>
                                </ul>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif active_step == 3:
                    st.markdown(
                        f"""
                        <div class="active-step-card">
                            <h3 style="margin-top:0; color:#00c6ff; border:none !important;">🔄 Шаг 4: Цикл Decode (Авторегрессионная генерация токенов)</h3>
                            <p style="color:#e2e8f0; font-size:14.5px; line-height:1.6;">
                                Начинается последовательный процесс генерации ответа. Для генерации каждого последующего токена GPU вынужден считать 
                                абсолютно все веса модели из памяти VRAM (<span style="color:#fff; font-weight:600;">{W / 1e9:.2f} GB</span>) ради выполнения ничтожного объема операций.
                            </p>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; margin-top:15px;">
                                <ul style="margin:0; padding-left:20px; color:#a0aec0; font-size:14px; display:flex; flex-direction:column; gap:8px;">
                                    <li>🕒 <b>Скорость генерации токена:</b> <span style="color:#fff; font-weight:600;">{res.decode_per_token_s * 1000:.1f} мс / токен</span></li>
                                    <li>📈 <b>Арифметическая интенсивность Decode:</b> <span style="color:#ff9f43; font-weight:700;">{intensity_decode:.2f} FLOP / byte</span> (крайне низкая, в {p_in} раз ниже префилла!)</li>
                                    <li>🚨 <b>Физический лимит:</b> фаза упирается в <span style="color:#e74c3c; font-weight:700;">{res.bottleneck_decode.upper()}</span> (Memory-bound)</li>
                                </ul>
                            </div>
                            <p style="color:#a0aec0; font-size:13px; margin-top:15px; font-style:italic;">
                                💡 Это классическое «бутылочное горлышко памяти». Вычислительные ядра GPU 95% времени простаивают, просто ожидая, пока веса перекачаются из VRAM в кэш процессора. Решается переходом на HBM3/HBM3e память с широкой шиной, квантованием (до 4/8 бит) или увеличением размера батча.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif active_step == 4:
                    st.markdown(
                        f"""
                        <div class="active-step-card">
                            <h3 style="margin-top:0; color:#00c6ff; border:none !important;">📤 Шаг 5: Вывод и Детокенизация (Detokenization)</h3>
                            <p style="color:#e2e8f0; font-size:14.5px; line-height:1.6;">
                                Сгенерированная последовательность числовых токенов переводится обратно в понятные слова текста с использованием словаря токенизатора 
                                и порционно отправляется пользователю (Streaming).
                            </p>
                            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:15px; border-radius:10px; margin-top:15px;">
                                <ul style="margin:0; padding-left:20px; color:#a0aec0; font-size:14px; display:flex; flex-direction:column; gap:8px;">
                                    <li>📝 <b>Сгенерировано ответов (P_out):</b> <span style="color:#fff; font-weight:600;">{p_out} токенов</span></li>
                                    <li>🚀 <b>Суммарная скорость вывода системы (Throughput):</b> <span style="color:#38ef7d; font-weight:700;">{res.throughput_tok_s:.1f} токенов / сек</span></li>
                                    <li>⏱️ <b>Полное время обработки запроса (Total Latency):</b> <span style="color:#fff; font-weight:600;">{res.total_latency_s:.2f} сек</span></li>
                                </ul>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
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
# 🔌 Анатомия GPU
# =============================================================================
with tab_anat:
    st.markdown("## 🔌 Анатомия GPU: Промышленные AI-ускорители vs Бытовые видеокарты")
    st.caption(
        "Понимание физических различий между профессиональными HBM-видеокартами и игровыми "
        "GDDR-чипами критически важно при масштабировании систем инференса LLM."
    )

    # 1. Interactive specifications scorecard comparison
    st.markdown("### 📊 Интерактивное сравнение характеристик")
    st.caption("Выберите два любых графических процессора для детального сопоставления их физических параметров.")

    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        gpu_a = st.selectbox("Сравниваемая видеокарта А:", list(HARDWARE_SPECS.keys()), index=0, key="anat_gpu_a")
    with c_sel2:
        gpu_b = st.selectbox("Сравниваемая видеокарта Б:", list(HARDWARE_SPECS.keys()), index=5 if len(HARDWARE_SPECS) > 5 else 1, key="anat_gpu_b")

    spec_a = HARDWARE_SPECS[gpu_a]
    spec_b = HARDWARE_SPECS[gpu_b]

    flops_a = spec_a["peak_tflops"][16]
    flops_b = spec_b["peak_tflops"][16]
    bw_a = spec_a["memory_bandwidth_gbs"]
    bw_b = spec_b["memory_bandwidth_gbs"]
    cap_a = spec_a["memory_capacity_gb"]
    cap_b = spec_b["memory_capacity_gb"]
    tdp_a = spec_a.get("tdp_w", 0)
    tdp_b = spec_b.get("tdp_w", 0)

    # Render scorecard
    st.markdown(
        f"""
        <div style="background: rgba(18, 22, 32, 0.4); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 16px; padding: 25px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.2); backdrop-filter: blur(10px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div style="width: 42%; text-align: right;">
                    <h3 style="margin: 0; color: #00c6ff; font-size: 22px; font-weight: 800; border: none !important;">{gpu_a}</h3>
                    <span style="font-size: 12px; color: #a0aec0;">Карта А</span>
                </div>
                <div style="width: 16%; text-align: center;">
                    <span style="background: linear-gradient(135deg, #00c6ff, #0072ff); color: white; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 800; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">VS</span>
                </div>
                <div style="width: 42%; text-align: left;">
                    <h3 style="margin: 0; color: #38ef7d; font-size: 22px; font-weight: 800; border: none !important;">{gpu_b}</h3>
                    <span style="font-size: 12px; color: #a0aec0;">Карта Б</span>
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 15px;">
                <!-- VRAM -->
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 4px; color: #a0aec0;">
                        <span style="color: #00c6ff; font-weight: 700;">{cap_a:.1f} GB VRAM</span>
                        <b style="color: #fff; font-family: 'Outfit', sans-serif;">Объем памяти VRAM</b>
                        <span style="color: #38ef7d; font-weight: 700;">{cap_b:.1f} GB VRAM</span>
                    </div>
                    <div style="display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #1a1e29;">
                        <div style="width: {cap_a / (cap_a + cap_b) * 100 if cap_a + cap_b > 0 else 50}%; background: #00c6ff; border-right: 1px solid #000;"></div>
                        <div style="width: {cap_b / (cap_a + cap_b) * 100 if cap_a + cap_b > 0 else 50}%; background: #38ef7d;"></div>
                    </div>
                </div>
                <!-- Bandwidth -->
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 4px; color: #a0aec0;">
                        <span style="color: #00c6ff; font-weight: 700;">{bw_a:.0f} GB/s</span>
                        <b style="color: #fff; font-family: 'Outfit', sans-serif;">Пропускная способность шины</b>
                        <span style="color: #38ef7d; font-weight: 700;">{bw_b:.0f} GB/s</span>
                    </div>
                    <div style="display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #1a1e29;">
                        <div style="width: {bw_a / (bw_a + bw_b) * 100 if bw_a + bw_b > 0 else 50}%; background: #00c6ff; border-right: 1px solid #000;"></div>
                        <div style="width: {bw_b / (bw_a + bw_b) * 100 if bw_a + bw_b > 0 else 50}%; background: #38ef7d;"></div>
                    </div>
                </div>
                <!-- FLOPS -->
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 4px; color: #a0aec0;">
                        <span style="color: #00c6ff; font-weight: 700;">{flops_a:.0f} TFLOPS</span>
                        <b style="color: #fff; font-family: 'Outfit', sans-serif;">FP16 Compute (Tensor Cores)</b>
                        <span style="color: #38ef7d; font-weight: 700;">{flops_b:.0f} TFLOPS</span>
                    </div>
                    <div style="display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #1a1e29;">
                        <div style="width: {flops_a / (flops_a + flops_b) * 100 if flops_a + flops_b > 0 else 50}%; background: #00c6ff; border-right: 1px solid #000;"></div>
                        <div style="width: {flops_b / (flops_a + flops_b) * 100 if flops_a + flops_b > 0 else 50}%; background: #38ef7d;"></div>
                    </div>
                </div>
                <!-- TDP -->
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 4px; color: #a0aec0;">
                        <span style="color: #00c6ff; font-weight: 700;">{tdp_a if tdp_a > 0 else "—"} W</span>
                        <b style="color: #fff; font-family: 'Outfit', sans-serif;">Потребление энергии (TDP)</b>
                        <span style="color: #38ef7d; font-weight: 700;">{tdp_b if tdp_b > 0 else "—"} W</span>
                    </div>
                    <div style="display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: #1a1e29;">
                        <div style="width: {tdp_a / (tdp_a + tdp_b) * 100 if tdp_a + tdp_b > 0 else 50}%; background: #00c6ff; border-right: 1px solid #000;"></div>
                        <div style="width: {tdp_b / (tdp_a + tdp_b) * 100 if tdp_a + tdp_b > 0 else 50}%; background: #38ef7d;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Side-by-side graphic panels and deep comparative details
    st.markdown("### 🔍 Сравнение физической архитектуры и внутреннего устройства")
    
    col_anat1, col_anat2 = st.columns(2)
    
    with col_anat1:
        st.subheader("🏢 Промышленные AI-ускорители (Industrial Server Components)")
        st.caption("Пример: NVIDIA H100 SXM5 / A100 / AMD Instinct MI300X")
        st.image("ui/assets/enterprise_gpu.png", width="stretch", caption="Промышленный AI-ускоритель в форм-факторе SXM5")
        
        st.markdown(
            """
            * **Широкополосная память HBM (High Bandwidth Memory)**:
              - Располагается вертикальными 3D-стеками непосредственно на одной кремниевой подложке с кристаллом GPU.
              - Обладает колоссальной разрядностью шины (до **4096-8192 бит**), что позволяет достигать скоростей передачи данных от **2.0 до 8.0 ТБ/с** при низком энергопотреблении и компактности.
            * **Высокоскоростная шина NVLink / NVSwitch**:
              - Позволяет объединять видеокарты в единый виртуальный суперчип с пропускной способностью до **900 ГБ/с в обе стороны**.
              - Полностью устраняет задержки сетевого обмена при Tensor Parallelism и Pipeline Parallelism.
            * **Мезонинный форм-фактор SXM / OAM**:
              - Устанавливаются напрямую на материнскую плату сервера в специальные разъемы высокой плотности. Исключает механические провисания кабелей и питается напрямую от шины 54V.
            * **Профессиональная виртуализация и надежность**:
              - Поддерживают технологии **MIG (Multi-Instance GPU)** и vGPU на аппаратном уровне. Оснащены ECC-памятью для защиты от битовых сбоев при длительном обучении.
            """
        )
        
    with col_anat2:
        st.subheader("🎮 Бытовые (Игровые) Видеокарты (Consumer Gaming GPUs)")
        st.caption("Пример: NVIDIA RTX 4090 / RTX 3090 / AMD RX 7900 XTX")
        st.image("ui/assets/consumer_gpu.png", width="stretch", caption="Потребительская игровая видеокарта с активным вентиляторным охлаждением")
        
        st.markdown(
            """
            * **Классическая память GDDR6 / GDDR7**:
              - Распаяна на печатной плате дискретными микросхемами вокруг GPU.
              - Подключается по узкой шине (**192-384 бит**). Для компенсации пропускной способности (макс. **1.0-1.5 ТБ/с**) работает на экстремально высоких тактовых частотах, что вызывает сильный нагрев.
            * **Стандартный интерфейс PCIe Gen4 / Gen5**:
              - Видеокарты общаются друг с другом через системную шину материнской платы PCIe (максимум **64 ГБ/с** для Gen5 x16) с проходом через ОЗУ хост-процессора.
              - Внутренние задержки и ограничения вызывают огромный пенальти при попытке распределить инференс (TP/PP).
            * **Потребительский форм-фактор PCIe со слотовым охлаждением**:
              - Подключаются в стандартный слот расширения, требуют громоздкого воздушного охлаждения (2.5-4 слота) и питаются от внешних кабелей 12VHPWR.
            * **Программные ограничения драйвера**:
              - Искусственно заблокированы функции vGPU, MIG, ограничен объем VRAM (не более 24 ГБ на RTX 4090) для разделения потребительского и корпоративного секторов.
            """
        )

    st.markdown("---")


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
