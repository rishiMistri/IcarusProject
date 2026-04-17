"""
Icarus Protocol — Streamlit UI Components
==========================================
Panel renderers, charts, HUD elements.
All functions receive data; none mutate session state directly.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from models import VAD_COLORS
from utils import grade_color, zone_color


# ─── CSS Injection ────────────────────────────────────────────────────────────

ICARUS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --bg:        #050810;
    --surface:   #0a0f1e;
    --border:    #1a2540;
    --accent:    #00d4ff;
    --accent2:   #ff6b35;
    --green:     #00ff9f;
    --yellow:    #ffc300;
    --red:       #ff3333;
    --text:      #c8d8e8;
    --muted:     #4a6080;
    --font-mono: 'Share Tech Mono', monospace;
    --font-head: 'Orbitron', sans-serif;
    --font-body: 'Rajdhani', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: var(--font-body);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Remove default streamlit padding */
.block-container { padding: 1rem 1.5rem !important; }

/* HUD Card */
.hud-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    position: relative;
    overflow: hidden;
}
.hud-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}

/* Metric label */
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.2rem;
}

/* Metric value */
.metric-value {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
    color: var(--accent);
}

.metric-value.grade { font-size: 2.5rem; }
.metric-value.danger { color: var(--red); }
.metric-value.warn { color: var(--yellow); }
.metric-value.ok { color: var(--green); }

/* Section header */
.panel-header {
    font-family: var(--font-head);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}

/* Heat gauge */
.heat-bar-container {
    background: var(--border);
    border-radius: 2px;
    height: 8px;
    width: 100%;
    margin: 0.4rem 0;
    overflow: hidden;
}
.heat-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
}

/* VAD badge */
.vad-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    border: 1px solid;
}

/* Lockout overlay */
.lockout-banner {
    background: rgba(255,51,51,0.15);
    border: 2px solid var(--red);
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
    font-family: var(--font-head);
    color: var(--red);
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    margin: 1rem 0;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Lucky flag */
.lucky-tag {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--yellow);
    border: 1px solid var(--yellow);
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    vertical-align: middle;
    margin-left: 0.4rem;
}

/* Strealined streamlit inputs */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

.stButton button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 2px;
    transition: all 0.2s;
}
.stButton button:hover {
    background: var(--accent);
    color: var(--bg);
}

/* Titles */
h1 { font-family: var(--font-head) !important; color: var(--accent) !important; }
h2, h3 { font-family: var(--font-head) !important; color: var(--text) !important; }
</style>
"""


def inject_css():
    st.markdown(ICARUS_CSS, unsafe_allow_html=True)


# ─── Plotly Theme ─────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,30,0.8)",
    font=dict(family="Share Tech Mono, monospace", color="#c8d8e8", size=10),
    xaxis=dict(gridcolor="#1a2540", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1a2540", showgrid=True, zeroline=False),
    margin=dict(l=40, r=20, t=30, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2540", borderwidth=1),
)


# ─── Top Banner / Header ──────────────────────────────────────────────────────

def render_header(metrics: dict):
    grade = metrics["grade"]
    projected = metrics["projected_iat"]
    theta = metrics["theta"]
    admit = metrics["admit_prob"]
    heat_zone_val = metrics["heat_zone"]
    gcolor = grade_color(grade)
    zcolor = zone_color(heat_zone_val)

    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 1.5, 1.5, 2])

    with col1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">GLOBAL GRADE</div>
            <div class="metric-value grade" style="color:{gcolor}">{grade}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">θ ABILITY</div>
            <div class="metric-value" style="color:{'#00ff9f' if theta > 0 else '#ff6b35'}">
                {theta:+.3f}
            </div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">PROJ. IAT SCORE</div>
            <div class="metric-value" style="color:{gcolor}">{projected:.0f}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">ADMIT PROB</div>
            <div class="metric-value" style="color:{gcolor}">{admit}%</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="hud-card">
            <div class="metric-label">SYSTEM STATUS</div>
            <div style="display:flex; gap:1rem; align-items:center; margin-top:0.3rem">
                <span style="font-family:'Share Tech Mono';font-size:0.75rem;color:{zcolor}">
                    ■ HEAT: {heat_zone_val}
                </span>
                <span style="font-family:'Share Tech Mono';font-size:0.75rem;color:{'#00ff9f' if metrics['yoga_active'] else '#4a6080'}">
                    {'▲ YOGA ACTIVE' if metrics['yoga_active'] else '○ NO BOOST'}
                </span>
                <span style="font-family:'Share Tech Mono';font-size:0.75rem;color:#4a6080">
                    {metrics['total_questions']}Q
                </span>
            </div>
        </div>""", unsafe_allow_html=True)


# ─── Lockout Banner ───────────────────────────────────────────────────────────

def render_lockout(active: bool, shutdown_threshold: float):
    if active:
        st.markdown(f"""
        <div class="lockout-banner">
            ⛔ SYSTEM LOCKOUT — ACCURACY BELOW {shutdown_threshold*100:.0f}% THRESHOLD<br>
            <span style="font-size:0.75rem;font-family:'Share Tech Mono'">
                MANDATORY RECOVERY PROTOCOL ENGAGED. STEP AWAY FROM DESK.
            </span>
        </div>""", unsafe_allow_html=True)


# ─── MCQ Input Panel (Left) ───────────────────────────────────────────────────

def render_input_panel(locked: bool):
    """Returns dict of form data or None if not submitted."""
    st.markdown('<div class="panel-header">MCQ INPUT</div>', unsafe_allow_html=True)

    if locked:
        st.markdown("""
        <div style="font-family:'Share Tech Mono';font-size:0.75rem;color:#ff3333;
             border:1px solid #ff3333;padding:0.5rem;border-radius:2px;text-align:center;">
            INPUT DISABLED<br>ACCURACY CRITICAL
        </div>""", unsafe_allow_html=True)
        return None

    with st.form("mcq_form", clear_on_submit=True):
        subject = st.selectbox("SUBJECT", ["Physics", "Chemistry", "Math"])
        topic = st.text_input("TOPIC", placeholder="e.g. Kinematics")
        difficulty = st.slider("DIFFICULTY (b)", min_value=-3.0, max_value=3.0,
                               value=0.5, step=0.1)
        time_taken = st.number_input("TIME TAKEN (sec)", min_value=1.0,
                                     max_value=300.0, value=45.0, step=1.0)
        correct = st.radio("RESULT", ["✓ CORRECT", "✗ INCORRECT"],
                           horizontal=True)
        discrimination = st.slider("DISCRIMINATION (a)", 0.5, 2.0, 1.0, 0.05)

        submitted = st.form_submit_button("SUBMIT ATTEMPT")

    if submitted:
        return {
            "subject": subject,
            "topic": topic if topic else subject,
            "difficulty_b": difficulty,
            "time_taken": float(time_taken),
            "correct": 1 if "CORRECT" in correct else 0,
            "discrimination_a": discrimination,
        }
    return None


# ─── Right Panel — VAD, Heat, Circadian ───────────────────────────────────────

def render_right_panel(metrics: dict):
    st.markdown('<div class="panel-header">SYSTEM VITALS</div>', unsafe_allow_html=True)

    # VAD State
    vad = metrics["vad_state"]
    vcolor = VAD_COLORS.get(vad, "#888")
    st.markdown(f"""
    <div class="hud-card">
        <div class="metric-label">VAD STATE</div>
        <div class="vad-badge" style="color:{vcolor};border-color:{vcolor};margin-top:0.3rem">
            {vad}
        </div>
        <div style="margin-top:0.5rem;font-family:'Share Tech Mono';font-size:0.7rem;color:#4a6080">
            VEL: {metrics['vad_velocity']:.2f} q/min &nbsp;|&nbsp; ACC: {metrics['vad_accuracy']*100:.1f}%
        </div>
    </div>""", unsafe_allow_html=True)

    # Heat Gauge
    heat = metrics["heat"]
    heat_zone_val = metrics["heat_zone"]
    hcolor = zone_color(heat_zone_val)
    heat_pct = int(heat * 100)
    st.markdown(f"""
    <div class="hud-card">
        <div class="metric-label">HEAT GAUGE — {heat_zone_val}</div>
        <div class="heat-bar-container" style="margin-top:0.5rem">
            <div class="heat-bar-fill" style="width:{heat_pct}%;background:{hcolor}"></div>
        </div>
        <div style="font-family:'Share Tech Mono';font-size:0.75rem;color:{hcolor}">
            {heat_pct}% THERMAL LOAD
        </div>
    </div>""", unsafe_allow_html=True)

    if heat_zone_val == "RED":
        st.markdown("""
        <div style="background:rgba(255,51,51,0.1);border:1px solid #ff3333;
             border-radius:4px;padding:0.6rem;text-align:center;
             font-family:'Share Tech Mono';font-size:0.7rem;color:#ff3333;
             margin-bottom:0.75rem;">
            ⚠ FORCED BREAK REQUIRED<br>STEP AWAY — 20 MIN
        </div>""", unsafe_allow_html=True)

    # Circadian Penalty
    mu = metrics["circadian_penalty"]
    mu_pct = mu * 100
    mu_color = "#00ff9f" if mu == 0 else ("#ffcc00" if mu < 0.2 else "#ff3333")
    st.markdown(f"""
    <div class="hud-card">
        <div class="metric-label">CIRCADIAN PENALTY</div>
        <div class="metric-value" style="font-size:1.4rem;color:{mu_color}">μ = {mu:.3f}</div>
        <div style="font-family:'Share Tech Mono';font-size:0.65rem;color:#4a6080;margin-top:0.2rem">
            SCORE SUPPRESSION: {mu_pct:.1f}%
        </div>
    </div>""", unsafe_allow_html=True)

    # Entropy
    entropy = metrics["entropy"]
    e_color = "#00ff9f" if entropy < 2 else ("#ffcc00" if entropy < 3.5 else "#ff3333")
    st.markdown(f"""
    <div class="hud-card">
        <div class="metric-label">COGNITIVE ENTROPY S(t)</div>
        <div class="metric-value" style="font-size:1.4rem;color:{e_color}">{entropy:.3f}</div>
        <div style="font-family:'Share Tech Mono';font-size:0.65rem;color:#4a6080;margin-top:0.2rem">
            FATIGUE: {metrics['fatigue']*100:.1f}% &nbsp;|&nbsp; AWAKE: {metrics['hours_awake']:.1f}h
        </div>
    </div>""", unsafe_allow_html=True)

    # Opportunity Cost
    opp = metrics["opportunity_cost"]
    st.markdown(f"""
    <div class="hud-card">
        <div class="metric-label">OPPORTUNITY COST</div>
        <div class="metric-value" style="font-size:1.4rem;color:#ff6b35">
            {opp:.1f} pts
        </div>
        <div style="font-family:'Share Tech Mono';font-size:0.65rem;color:#4a6080;margin-top:0.2rem">
            IDLE TIME PENALTY
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Chart: Theta Over Time ───────────────────────────────────────────────────

def render_theta_chart(df: pd.DataFrame):
    if df.empty or "theta" not in df.columns:
        st.info("No data yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["idx"], y=df["theta"],
        mode="lines",
        line=dict(color="#00d4ff", width=2, shape="spline", smoothing=0.8),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.07)",
        name="θ Ability",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#4a6080", line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="LATENT ABILITY θ", font=dict(size=11), x=0.01),
        yaxis_title="θ",
        xaxis_title="Question #",
        height=200,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_entropy_chart(df: pd.DataFrame):
    if df.empty or "entropy" not in df.columns:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["idx"], y=df["entropy"],
        mode="lines",
        line=dict(color="#ff6b35", width=2, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.07)",
        name="Entropy",
    ))
    fig.add_trace(go.Scatter(
        x=df["idx"], y=df["fatigue"],
        mode="lines",
        line=dict(color="#ff3333", width=1.5, dash="dot"),
        name="Fatigue",
        yaxis="y2",
    ))

        # Apply base layout (which already includes yaxis, xaxis)
    layout = {**PLOTLY_LAYOUT}
    layout["title"] = dict(
    text="ENTROPY S(t) + FATIGUE",
    font=dict(size=11),
    x=0.01
    )

    layout["xaxis_title"] = "Question #"

    layout["yaxis"] = dict(
        title="S(t)",
        gridcolor="#1a2540"
    )

    layout["yaxis2"] = dict(
        title="Fatigue",
        overlaying="y",
        side="right",
        range=[0, 1],
        gridcolor="rgba(0,0,0,0)"
    )

    layout["height"] = 200
    fig.update_layout(**layout)
    # Set primary y-axis title and grid color without overriding whole yaxis dict
    fig.update_yaxes(title_text="S(t)", gridcolor="#1a2540")

    st.plotly_chart(fig, use_container_width=True)


def render_accuracy_velocity_chart(df: pd.DataFrame):
    if df.empty:
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["idx"], y=df["accuracy"],
        mode="lines",
        line=dict(color="#00ff9f", width=2, shape="spline", smoothing=0.8),
        name="Accuracy",
    ), secondary_y=False)

    # Velocity computed from timestamps (correct method)
    if len(df) >= 3:
        timestamps = df["timestamp"].tolist()
        vel = []
        window = 5  # same as before
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            t0 = timestamps[start_idx]
            t1 = timestamps[i]
            elapsed = max((t1 - t0).total_seconds() / 60.0, 0.01)
            count = i - start_idx + 1
            vel.append(count / elapsed)
        fig.add_trace(go.Scatter(
            x=df["idx"], y=vel,
            mode="lines",
            line=dict(color="#ffc300", width=1.5, dash="dot"),
            name="Velocity (q/min)",
        ), secondary_y=True)

    fig.add_hline(y=0.45, line_dash="dash", line_color="#ff3333",
                  line_width=1, annotation_text="LOCKOUT THRESHOLD",
                  annotation_font_color="#ff3333",
                  annotation_font_size=9)

    layout = {
    **PLOTLY_LAYOUT,
    "title": dict(text="ACCURACY & VELOCITY", font=dict(size=11), x=0.01),
    "xaxis_title": "Question #",
    "height": 200,
    "legend": dict(x=0.01, y=0.01, bgcolor="rgba(0,0,0,0)"),
}

    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="q/min", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ─── Forecast Panel ───────────────────────────────────────────────────────────

def render_forecast_panel(metrics: dict):
    projected = metrics["projected_iat"]
    admit = metrics["admit_prob"]
    grade = metrics["grade"]
    gcolor = grade_color(grade)

    # Score gauge
    pct = projected / 180.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=projected,
        delta={"reference": 150, "relative": False,
               "font": {"size": 12, "color": "#4a6080"}},
        number={"font": {"size": 32, "family": "Orbitron", "color": gcolor}},
        gauge={
            "axis": {
                "range": [0, 180],
                "tickfont": {"size": 9, "color": "#4a6080"},
                "tickwidth": 1,
                "tickcolor": "#1a2540",
            },
            "bar": {"color": gcolor, "thickness": 0.6},
            "bgcolor": "#0a0f1e",
            "bordercolor": "#1a2540",
            "steps": [
                {"range": [0, 90],   "color": "rgba(255,51,51,0.15)"},
                {"range": [90, 140], "color": "rgba(255,195,0,0.10)"},
                {"range": [140, 180],"color": "rgba(0,255,159,0.10)"},
            ],
            "threshold": {
                "line": {"color": "#ff3333", "width": 2},
                "thickness": 0.8,
                "value": 150,
            },
        },
        title={"text": "PROJECTED IAT SCORE / 180",
               "font": {"size": 10, "color": "#4a6080", "family": "Share Tech Mono"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d8e8"),
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="display:flex;gap:1rem;justify-content:center;margin-top:0.5rem">
        <div class="hud-card" style="flex:1;text-align:center">
            <div class="metric-label">ADMISSION PROBABILITY</div>
            <div class="metric-value" style="color:{gcolor}">{admit}%</div>
        </div>
        <div class="hud-card" style="flex:1;text-align:center">
            <div class="metric-label">GRADE</div>
            <div class="metric-value grade" style="color:{gcolor}">{grade}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Recent Questions Log ─────────────────────────────────────────────────────

def render_question_log(questions: list, n: int = 10):
    st.markdown('<div class="panel-header">RECENT ATTEMPTS</div>', unsafe_allow_html=True)
    if not questions:
        st.markdown('<span style="font-family:\'Share Tech Mono\';font-size:0.7rem;color:#4a6080">NO DATA</span>',
                    unsafe_allow_html=True)
        return

    recent = questions[-n:][::-1]
    rows = []
    for q in recent:
        lucky_html = '<span class="lucky-tag">LUCKY</span>' if q.lucky_flag else ""
        correct_icon = "✓" if q.correct else "✗"
        correct_color = "#00ff9f" if q.correct else "#ff3333"
        rows.append(
            f'<tr style="border-bottom:1px solid #1a2540">'
            f'<td style="color:{correct_color};font-weight:700;padding:3px 6px">{correct_icon}</td>'
            f'<td style="color:#c8d8e8;padding:3px 6px">{q.subject[:4].upper()}</td>'
            f'<td style="color:#4a6080;padding:3px 6px">b={q.difficulty_b:+.1f}</td>'
            f'<td style="color:#4a6080;padding:3px 6px">{q.time_taken:.0f}s</td>'
            f'<td style="padding:3px 6px">{lucky_html}</td>'
            f'</tr>'
        )

    table_html = f"""
    <table style="width:100%;font-family:'Share Tech Mono';font-size:0.7rem;border-collapse:collapse">
        <thead>
            <tr style="color:#4a6080;border-bottom:1px solid #1a2540">
                <th style="padding:3px 6px;text-align:left">RES</th>
                <th style="padding:3px 6px;text-align:left">SUBJ</th>
                <th style="padding:3px 6px;text-align:left">DIFF</th>
                <th style="padding:3px 6px;text-align:left">TIME</th>
                <th style="padding:3px 6px;text-align:left">FLAG</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>"""
    st.markdown(table_html, unsafe_allow_html=True)