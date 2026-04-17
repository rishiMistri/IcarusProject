"""
Icarus Protocol — Main Entry Point
====================================
Run:   streamlit run main.py

The system integrates:
  - 3PL IRT cognitive modelling
  - Entropy / fatigue physics
  - Circadian penalty scheduling
  - VAD (Velocity-Accuracy Divergence) state machine
  - Heat gauge + lockout controller
  - Real-time Streamlit HUD
"""
import pandas as pd
from utils import process_question_batch   # add this to existing utils import
import streamlit as st
from datetime import datetime, timedelta
import uuid

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ICARUS PROTOCOL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports (consolidated at top) ───────────────────────────────────────
from models import (
    Question, YogaSession, SessionState,
    update_theta, is_statistically_lucky, p_correct_3pl,
    compute_circadian_penalty,
)
from utils import (
    compute_all_metrics, build_time_series, questions_to_df,
    simulate_full_session, run_simulation_batch,
    rolling_accuracy,
    load_session, save_session, delete_save,
    delete_question_by_index, recompute_theta_from_questions,
    delete_questions_by_indices,   # <-- add this
)
from ui import (
    inject_css,
    render_header, render_lockout,
    render_input_panel, render_right_panel,
    render_theta_chart, render_entropy_chart, render_accuracy_velocity_chart,
    render_forecast_panel, render_question_log,
)


# ─── Session State Bootstrap ──────────────────────────────────────────────────

def init_session():
    if "initialized" not in st.session_state:
        # Try to load from file
        loaded_session, loaded_config = load_session()
        if loaded_session is not None:
            st.session_state.session = loaded_session
            st.session_state.config = loaded_config
        else:
            # Fresh session
            now = datetime.now()
            wake = now.replace(hour=7, minute=30, second=0, microsecond=0)
            if wake > now:
                wake = now - timedelta(hours=2)
            st.session_state.session = SessionState(wake_time=wake)
            st.session_state.config = {
                "shutdown_threshold": 0.45,
                "penalty_per_30min": 0.05,
                "yoga_min_duration": 30,
                "vad_window": 10,
                "idle_minutes": 0.0,
            }
        st.session_state.initialized = True
init_session()
inject_css()

# ─── Sidebar: Configuration Panel ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',sans-serif;font-size:1rem;font-weight:900;
         color:#00d4ff;letter-spacing:0.2em;margin-bottom:0.3rem">
         ⚡ ICARUS
    </div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;
         color:#4a6080;letter-spacing:0.15em;margin-bottom:1.5rem">
         PROTOCOL v1.0 — BEHAVIORAL OPTIMIZER
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Share Tech Mono\';font-size:0.65rem;color:#4a6080;'
                'letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1a2540;'
                'padding-bottom:0.4rem;margin-bottom:0.8rem">Configuration</div>',
                unsafe_allow_html=True)

    cfg = st.session_state.config

    cfg["shutdown_threshold"] = st.slider(
        "Shutdown Accuracy %", 20, 70,
        int(cfg["shutdown_threshold"] * 100), 5
    ) / 100.0

    cfg["penalty_per_30min"] = st.slider(
        "Circadian Penalty / 30min", 1, 15,
        int(cfg["penalty_per_30min"] * 100), 1
    ) / 100.0

    cfg["vad_window"] = st.slider("VAD Window (questions)", 5, 30, cfg["vad_window"], 1)
    cfg["idle_minutes"] = st.number_input("Idle Time (minutes)", 0.0, 480.0,
                                          float(cfg["idle_minutes"]), 15.0)

    # Apply idle to session
    st.session_state.session.total_idle_minutes = cfg["idle_minutes"]

    st.markdown("---")
    st.markdown('<div style="font-family:\'Share Tech Mono\';font-size:0.65rem;color:#4a6080;'
                'letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1a2540;'
                'padding-bottom:0.4rem;margin-bottom:0.8rem">Yoga Session</div>',
                unsafe_allow_html=True)

    yoga_duration = st.number_input("Duration (min)", 0, 120, 40, 5)

    if st.button("LOG YOGA SESSION"):
        if yoga_duration >= cfg["yoga_min_duration"]:
            ys = YogaSession(
                timestamp=datetime.now(),
                duration_minutes=yoga_duration,
            )
            st.session_state.session.yoga_sessions.append(ys)
            st.session_state.session.yoga_boost_until = (
                datetime.now() + timedelta(hours=3)
            )
            st.success(f"Yoga logged. Entropy boost active for 3h.")
        else:
            st.error(f"Minimum {cfg['yoga_min_duration']} min required.")

    st.markdown("---")
    st.markdown('<div style="font-family:\'Share Tech Mono\';font-size:0.65rem;color:#4a6080;'
                'letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1a2540;'
                'padding-bottom:0.4rem;margin-bottom:0.8rem">Wake Time</div>',
                unsafe_allow_html=True)

    wake_h = st.slider("Wake Hour", 4, 12, st.session_state.session.wake_time.hour)
    wake_m = st.slider("Wake Minute", 0, 59, st.session_state.session.wake_time.minute, 5)
    if st.button("SET WAKE TIME"):
        new_wake = datetime.now().replace(hour=wake_h, minute=wake_m, second=0)
        st.session_state.session.wake_time = new_wake
        st.session_state.session.circadian_penalty = compute_circadian_penalty(
            new_wake, cfg["penalty_per_30min"]
        )
        st.rerun()  # Added: update display
    st.markdown("---")
    st.markdown('<div style="font-family:\'Share Tech Mono\';font-size:0.65rem;color:#4a6080;'
                'letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1a2540;'
                'padding-bottom:0.4rem;margin-bottom:0.8rem">💪 STRENGTH ACTIVATION</div>',
                unsafe_allow_html=True)

    pushups = st.number_input("Pushups (reps)", 0, 50, 0, step=5)
    pullups = st.number_input("Pullups (reps)", 0, 20, 0, step=1)

    if st.button("LOG STRENGTH SET"):
        if pushups > 0 or pullups > 0:
            now = datetime.now()
            boost_until = now + timedelta(minutes=30)
            ss = StrengthSession(
                timestamp=now,
                pushups=pushups,
                pullups=pullups,
                boost_until=boost_until
            )
            st.session_state.session.strength_sessions.append(ss)
            st.success(f"Logged {pushups} pushups / {pullups} pullups. +0.1 θ boost for 30 min.")
            st.rerun()
        else:
            st.warning("Enter at least one rep.")


    st.markdown("---")
    st.markdown('<div style="font-family:\'Share Tech Mono\';font-size:0.65rem;color:#4a6080;'
                'letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1a2540;'
                'padding-bottom:0.4rem;margin-bottom:0.8rem">Simulation</div>',
                unsafe_allow_html=True)


    sim_n = st.slider("Questions to simulate", 5, 100, 20, 5)
    sim_subject = st.selectbox("Subject", ["Random", "Physics", "Chemistry", "Math"])

    if st.button("▶ RUN SIMULATION"):
        subj = None if sim_subject == "Random" else sim_subject
        run_simulation_batch(
            st.session_state.session,
            n_questions=sim_n,
            subject=subj,
        )
        st.success(f"Simulated {sim_n} questions.")
        st.rerun()  # Added: update UI immediately

    if st.button("⟳ FULL RESET"):
        delete_save()  # Remove saved file
        for key in ["session", "config", "initialized"]:
            if key in st.session_state:
                del st.session_state[key]
        init_session()
        st.rerun()
            

    if st.button("💾 SAVE SESSION"):
        save_session(st.session_state.session, st.session_state.config)
        st.success("Session saved successfully.")


# ─── Compute Metrics ──────────────────────────────────────────────────────────

session: SessionState = st.session_state.session
cfg = st.session_state.config

# Now compute_all_metrics uses config for lockout threshold
metrics = compute_all_metrics(session, config=cfg)

ts_df = build_time_series(session)


# ─── Layout ───────────────────────────────────────────────────────────────────

# TOP: Header HUD
render_header(metrics)

# Lockout banner (full-width)
render_lockout(metrics["lockout"], cfg["shutdown_threshold"])

# Heat RED forced break banner
if metrics["heat_zone"] == "RED" and not metrics["lockout"]:
    st.markdown("""
    <div style="background:rgba(255,51,51,0.08);border:1px solid #ff3333;
         border-radius:4px;padding:0.6rem;text-align:center;
         font-family:'Share Tech Mono';font-size:0.75rem;color:#ff3333;margin-bottom:0.5rem">
        ⚠ THERMAL OVERLOAD — FORCED BREAK RECOMMENDED. HEAT ZONE: RED.
    </div>""", unsafe_allow_html=True)

# MAIN 3-column layout
left_col, center_col, right_col = st.columns([1, 2.2, 1], gap="medium")
# ── LEFT: Batch Question Upload ────────────────────────────────────────────
with left_col:
    st.markdown('<div class="panel-header">📂 BATCH UPLOAD (CSV)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono';font-size:0.7rem;color:#4a6080;margin-bottom:1rem">
        Upload a CSV with 10–100 questions. Required columns:<br>
        <code>subject, difficulty_b, time_taken, correct</code><br>
        Optional: <code>topic, discrimination_a, guessing_c</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            required_cols = {"subject", "difficulty_b", "time_taken", "correct"}
            if not required_cols.issubset(df_batch.columns):
                st.error(f"Missing columns. Required: {required_cols}")
            else:
                n_questions = len(df_batch)
                if n_questions < 10 or n_questions > 100:
                    st.error("Batch must contain between 10 and 100 questions.")
                else:
                    st.success(f"✅ Found {n_questions} questions. Ready to process.")
                    if st.button("🚀 PROCESS BATCH", key="process_batch"):
                        batch_data = df_batch.to_dict(orient="records")
                        count = process_question_batch(session, batch_data)
                        st.success(f"Processed {count} questions.")
                        st.rerun()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
        # --- Delete Specific Question ---
    st.markdown("---")
    st.markdown('<div class="panel-header">🗑 DELETE LOG ENTRY</div>', unsafe_allow_html=True)

    if session.questions:
        # Create a dropdown with readable labels
        options = []
        for i, q in enumerate(session.questions):
            label = f"#{i+1} | {q.subject} | b={q.difficulty_b:+.1f} | {'✓' if q.correct else '✗'} | {q.timestamp.strftime('%H:%M:%S')}"
            options.append((i, label))
        
        selected_idx = st.selectbox(
            "Select question to delete",
            options,
            format_func=lambda x: x[1],
            key="delete_select"
        )
        
        if st.button("DELETE SELECTED", key="delete_btn"):
            idx = selected_idx[0]
            if delete_question_by_index(session, idx):
                st.success(f"Deleted question #{idx+1} and recomputed ability.")
                st.rerun()
            else:
                st.error("Deletion failed.")
    else:
        st.markdown('<span style="font-family:\'Share Tech Mono\';font-size:0.7rem;color:#4a6080">No questions to delete.</span>',
                    unsafe_allow_html=True)

        st.markdown("---")
        # Keep the question log to show recent attempts
        render_question_log(session.questions, n=12)

        # --- Batch Delete Multiple Questions ---
    st.markdown('<div class="panel-header">🗑 BATCH DELETE</div>', unsafe_allow_html=True)

    if session.questions:
        # Create options for multi-select: (index, label)
        multi_options = []
        for i, q in enumerate(session.questions):
            label = f"#{i+1} | {q.subject} | b={q.difficulty_b:+.1f} | {'✓' if q.correct else '✗'} | {q.timestamp.strftime('%H:%M:%S')}"
            multi_options.append((i, label))

        selected_items = st.multiselect(
            "Select questions to delete",
            multi_options,
            format_func=lambda x: x[1],
            key="batch_delete_select"
        )

        if st.button("DELETE SELECTED BATCH", key="batch_delete_btn"):
            indices_to_delete = [idx for idx, _ in selected_items]
            if indices_to_delete:
                deleted_count = delete_questions_by_indices(session, indices_to_delete)
                st.success(f"Deleted {deleted_count} question(s) and recomputed ability.")
                st.rerun()
            else:
                st.warning("No questions selected.")
    else:
        st.markdown('<span style="font-family:\'Share Tech Mono\';font-size:0.7rem;color:#4a6080">No questions to delete.</span>',
                    unsafe_allow_html=True)
# ── CENTER: Live Graphs ────────────────────────────────────────────────────
with center_col:
    st.markdown('<div class="panel-header">LIVE ANALYTICS</div>', unsafe_allow_html=True)

    if ts_df.empty:
        st.markdown("""
        <div style="font-family:'Share Tech Mono';font-size:0.75rem;color:#4a6080;
             text-align:center;padding:2rem;border:1px dashed #1a2540;border-radius:4px">
            NO SESSION DATA<br>
            <span style="font-size:0.65rem">Submit MCQ attempts or run simulation</span>
        </div>""", unsafe_allow_html=True)
    else:
        render_theta_chart(ts_df)
        render_entropy_chart(ts_df)
        render_accuracy_velocity_chart(ts_df)


# ── RIGHT: Vitals + Forecast ──────────────────────────────────────────────
with right_col:
    render_right_panel(metrics)
    st.markdown("<br>", unsafe_allow_html=True)
    render_forecast_panel(metrics)


# ─── Bottom: Subject Breakdown ─────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="panel-header">SUBJECT PERFORMANCE MATRIX</div>',
            unsafe_allow_html=True)

if session.questions:
    subj_stats = {}
    for q in session.questions:
        s = q.subject
        if s not in subj_stats:
            subj_stats[s] = {"correct": 0, "total": 0, "time": [], "theta_sum": 0}
        subj_stats[s]["total"] += 1
        subj_stats[s]["correct"] += q.correct
        subj_stats[s]["time"].append(q.time_taken)
        subj_stats[s]["theta_sum"] += q.theta_at_attempt

    cols = st.columns(len(subj_stats))
    colors = {"Physics": "#00d4ff", "Chemistry": "#ff6b35", "Math": "#00ff9f"}

    for idx, (subj, data) in enumerate(subj_stats.items()):
        acc = data["correct"] / data["total"] * 100
        avg_time = sum(data["time"]) / len(data["time"])
        avg_theta = data["theta_sum"] / data["total"]
        c = colors.get(subj, "#888888")

        with cols[idx]:
            st.markdown(f"""
            <div class="hud-card" style="border-left-color:{c}">
                <div class="metric-label">{subj.upper()}</div>
                <div style="font-family:'Orbitron';font-size:1.5rem;color:{c};font-weight:700">
                    {acc:.0f}%
                </div>
                <div style="font-family:'Share Tech Mono';font-size:0.65rem;color:#4a6080;margin-top:0.3rem">
                    {data['total']} attempts &nbsp;|&nbsp; {avg_time:.0f}s avg<br>
                    θ mean: {avg_theta:+.3f}
                </div>
            </div>""", unsafe_allow_html=True)
else:
    st.markdown('<span style="font-family:\'Share Tech Mono\';font-size:0.7rem;color:#4a6080">'
                'No attempts yet.</span>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;font-family:'Share Tech Mono';font-size:0.6rem;
     color:#1a2540;margin-top:2rem;letter-spacing:0.15em">
    ICARUS PROTOCOL — NOT A PRODUCTIVITY TOOL. A CONSTRAINT SYSTEM. A FEEDBACK CONTROLLER.
</div>""", unsafe_allow_html=True)