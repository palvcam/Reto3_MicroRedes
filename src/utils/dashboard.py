import streamlit as st
import numpy as np

st.set_page_config(page_title="Microgrid RL Step Dashboard", layout="wide")

# =========================================================
# PALETA
# =========================================================
BG = "#F7F1E8"
CARD = "#EFE6D8"
CARD_2 = "#F4ECE1"
TEXT = "#4E5A4F"
SUBTEXT = "#7A7A6D"
GREEN = "#9CAF88"
GREEN_DARK = "#6F8A68"
ORANGE = "#E8B07A"
LINE = "#D8CCBC"
WHITE_SOFT = "#FAF7F2"

# =========================================================
# STEP REALISTA Y FIEL AL ENTORNO
# =========================================================
current_step = 158
hour = current_step % 24                  # 14
day_of_year = (current_step // 24) % 365 # 6

current_load = 41.8
current_pv = 16.3
net_load_kw = current_load - current_pv   # 25.5

soc = 0.68
current_import_price = 0.146
current_export_price = 0.073

net_load_min = -40.64
net_load_max = 62.45
price_min = 0.0206
price_max = 0.42315

net_load_norm = np.clip(
    2 * ((net_load_kw - net_load_min) / (net_load_max - net_load_min)) - 1,
    -1.0,
    1.0,
)

import_price_norm = np.clip(
    (current_import_price - price_min) / (price_max - price_min),
    0.0,
    1.0,
)

angle_hour = 2.0 * np.pi * hour / 24.0
hour_sin = np.sin(angle_hour)
hour_cos = np.cos(angle_hour)

angle_day = 2.0 * np.pi * day_of_year / 365.0
day_sin = np.sin(angle_day)
day_cos = np.cos(angle_day)

action = 0.36
battery_kw = action * 50.0                # 18.0 kW
grid_kw = net_load_kw - battery_kw        # 7.5 kW

grid_import_kw = max(0.0, grid_kw)
grid_export_kw = max(0.0, -grid_kw)

reward_scale_C = 91.88
mg_reward = -1.10
reward_normalized = mg_reward / reward_scale_C

low_soc_threshold = 0.2
low_soc_penalty_applied = 0.0

# =========================================================
# CSS
# =========================================================
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
    }}

    .block-container {{
        max-width: 1120px;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        padding-left: 1.0rem;
        padding-right: 1.0rem;
    }}

    .title {{
        font-size: 1.75rem;
        color: {TEXT};
        font-weight: 700;
        margin-bottom: 0.08rem;
        letter-spacing: 0.15px;
    }}

    .subtitle {{
        font-size: 0.92rem;
        color: {SUBTEXT};
        margin-bottom: 0.65rem;
    }}

    .flow {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-bottom: 0.8rem;
    }}

    .flow-pill {{
        background: {WHITE_SOFT};
        border: 1px solid {LINE};
        border-radius: 999px;
        color: {TEXT};
        padding: 0.28rem 0.78rem;
        font-size: 0.82rem;
        font-weight: 600;
    }}

    .flow-arrow {{
        color: {GREEN_DARK};
        font-size: 1rem;
        font-weight: 700;
    }}

    .section {{
        background: {CARD};
        border: 1px solid {LINE};
        border-radius: 20px;
        padding: 13px 14px;
        box-shadow: 0 2px 8px rgba(80, 70, 50, 0.04);
        margin-bottom: 10px;
    }}

    .section-title {{
        color: {TEXT};
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 0.12rem;
    }}

    .section-note {{
        color: {SUBTEXT};
        font-size: 0.83rem;
        margin-bottom: 0.65rem;
    }}

    .mini {{
        background: {CARD_2};
        border: 1px solid {LINE};
        border-radius: 15px;
        padding: 10px 11px;
        min-height: 94px;
    }}

    .name {{
        color: {TEXT};
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}

    .value {{
        color: {TEXT};
        font-size: 1.12rem;
        font-weight: 700;
        line-height: 1.05;
    }}

    .extra {{
        color: {SUBTEXT};
        font-size: 0.76rem;
        margin-top: 0.34rem;
        line-height: 1.2;
    }}

    .action-box {{
        background: linear-gradient(180deg, {CARD_2} 0%, {CARD} 100%);
        border: 1px solid {LINE};
        border-radius: 16px;
        padding: 12px 13px;
    }}

    .big {{
        color: {TEXT};
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.02;
    }}

    .mid {{
        color: {TEXT};
        font-size: 0.96rem;
        font-weight: 600;
        margin-top: 0.32rem;
    }}

    .small {{
        color: {SUBTEXT};
        font-size: 0.8rem;
        margin-top: 0.45rem;
        line-height: 1.25;
    }}

    .formula {{
        background: {WHITE_SOFT};
        border: 1px solid {LINE};
        border-radius: 14px;
        padding: 10px 12px;
        color: {TEXT};
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 0.75rem;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def progress_html(value, fill_color, bg_color="#F7F1E8", h=10):
    value = max(0.0, min(1.0, value))
    return f"""
    <div style="
        width:100%;
        height:{h}px;
        background:{bg_color};
        border-radius:999px;
        overflow:hidden;
        border:1px solid {LINE};
        margin-top:7px;
    ">
        <div style="
            width:{value*100:.1f}%;
            height:100%;
            background:{fill_color};
            border-radius:999px;
        "></div>
    </div>
    """

def signed_bar_html(value, vmin=-1.0, vmax=1.0, h=11):
    value = max(vmin, min(vmax, value))
    pos = (0 - vmin) / (vmax - vmin) * 100
    width = abs(value) / (vmax - vmin) * 100
    left = pos if value >= 0 else pos - width
    color = ORANGE if value >= 0 else GREEN

    return f"""
    <div style="
        position:relative;
        width:100%;
        height:{h}px;
        background:{WHITE_SOFT};
        border:1px solid {LINE};
        border-radius:999px;
        margin-top:7px;
        overflow:hidden;
    ">
        <div style="
            position:absolute;
            left:{pos:.1f}%;
            top:0;
            width:1.4px;
            height:100%;
            background:{TEXT};
            opacity:0.85;
        "></div>
        <div style="
            position:absolute;
            left:{left:.1f}%;
            top:0;
            width:{width:.1f}%;
            height:100%;
            background:{color};
        "></div>
    </div>
    """

def mini_card(name, value, extra="", bar=None):
    st.markdown('<div class="mini">', unsafe_allow_html=True)
    st.markdown(f'<div class="name">{name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="value">{value}</div>', unsafe_allow_html=True)
    if bar is not None:
        st.markdown(bar, unsafe_allow_html=True)
    if extra:
        st.markdown(f'<div class="extra">{extra}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">Microgrid RL — One Step Snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Vista compacta y lista para captura: estado observado, acción continua y resultados del step.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="flow">
    <div class="flow-pill">Observation state</div>
    <div class="flow-arrow">→</div>
    <div class="flow-pill">Action</div>
    <div class="flow-arrow">→</div>
    <div class="flow-pill">Results after step</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# OBSERVATION STATE
# =========================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Observation state</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Vector de observación continuo de dimensión 7 que recibe el agente.</div>',
    unsafe_allow_html=True
)

r1 = st.columns(4, gap="small")
with r1[0]:
    mini_card(
        "net_load_norm",
        f"{net_load_norm:.3f}",
        extra=f"net load = {net_load_kw:.1f} kW",
        bar=signed_bar_html(net_load_norm),
    )
with r1[1]:
    mini_card(
        "soc",
        f"{soc:.2f}",
        extra="Estado de carga",
        bar=progress_html(soc, GREEN),
    )
with r1[2]:
    mini_card(
        "import_price_norm",
        f"{import_price_norm:.3f}",
        extra=f"price = {current_import_price:.3f} €/kWh",
        bar=progress_html(import_price_norm, ORANGE),
    )
with r1[3]:
    mini_card(
        "hour_sin",
        f"{hour_sin:.3f}",
        extra=f"Hora = {hour:02d}:00",
        bar=signed_bar_html(hour_sin),
    )

r2 = st.columns(3, gap="small")
with r2[0]:
    mini_card(
        "hour_cos",
        f"{hour_cos:.3f}",
        extra=f"Hora = {hour:02d}:00",
        bar=signed_bar_html(hour_cos),
    )
with r2[1]:
    mini_card(
        "day_sin",
        f"{day_sin:.3f}",
        extra=f"Día = {day_of_year}",
        bar=signed_bar_html(day_sin),
    )
with r2[2]:
    mini_card(
        "day_cos",
        f"{day_cos:.3f}",
        extra=f"Día = {day_of_year}",
        bar=signed_bar_html(day_cos),
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ACTION + RESULTS
# =========================================================
left, right = st.columns([0.82, 1.18], gap="medium")

with left:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Action</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Acción continua unidimensional elegida por el agente.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="action-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="big">action = {action:.2f}</div>', unsafe_allow_html=True)
    st.markdown(signed_bar_html(action, -1, 1, h=12), unsafe_allow_html=True)
    st.markdown(f'<div class="mid">battery_kw = {battery_kw:.1f} kW</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small">Transformación del entorno: <b>battery_kw = action × 50</b>.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results after step</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Consecuencias físicas y económicas tras aplicar la acción.</div>',
        unsafe_allow_html=True
    )

    rr1 = st.columns(3, gap="small")
    with rr1[0]:
        mini_card("current_load", f"{current_load:.1f} kW", extra="Demanda instantánea")
    with rr1[1]:
        mini_card("current_pv", f"{current_pv:.1f} kW", extra="Generación PV")
    with rr1[2]:
        mini_card("grid_import_kw", f"{grid_import_kw:.1f} kW", extra="Importación")

    rr2 = st.columns(4, gap="small")
    with rr2[0]:
        mini_card("grid_export_kw", f"{grid_export_kw:.1f} kW", extra="Exportación")
    with rr2[1]:
        mini_card("mg_reward", f"{mg_reward:.2f}", extra="Reward bruto")
    with rr2[2]:
        mini_card("reward_normalized", f"{reward_normalized:.4f}", extra=f"C = {reward_scale_C:.2f}")
    with rr2[3]:
        mini_card("low_soc_penalty_applied", f"{low_soc_penalty_applied:.2f}", extra="Penalización SoC")

    st.markdown(
        f'<div class="formula">grid = net load − battery = {net_load_kw:.1f} − {battery_kw:.1f} = {grid_kw:.1f} kW</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small">La red no la decide el agente directamente: el entorno la calcula para cerrar el balance energético.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)