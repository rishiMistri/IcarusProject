"""
Icarus Protocol — Utilities & Simulation Engine
================================================
Data helpers, simulation, serialisation, formatting.
"""
from models import Question, is_statistically_lucky, p_correct_3pl, update_theta
import random
import uuid
import math
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import numpy as np
import json
import os
from dataclasses import asdict, is_dataclass


from models import (
    Question, YogaSession, SessionState,
    p_correct_3pl, update_theta, is_statistically_lucky,
    compute_entropy, compute_fatigue, classify_vad,
    compute_heat, heat_zone, compute_circadian_penalty,
    compute_opportunity_cost, project_iat_score,
    score_to_grade, admission_probability,
    rolling_accuracy, rolling_velocity,
    ENTROPY_K,
    compute_activation_boost 
)


# ─── Simulation Engine ────────────────────────────────────────────────────────

SUBJECTS = ["Physics", "Chemistry", "Math"]
TOPICS = {
    "Physics": ["Mechanics", "Optics", "Thermodynamics", "Electrostatics", "Waves"],
    "Chemistry": ["Organic", "Inorganic", "Physical", "Equilibrium", "Kinetics"],
    "Math": ["Calculus", "Algebra", "Probability", "Vectors", "Coordinate Geometry"],
}


def simulate_question(
    theta: float,
    subject: Optional[str] = None,
    difficulty_range: tuple = (-2.0, 2.5),
) -> Question:
    """Generate a simulated MCQ attempt based on current theta."""
    if subject is None:
        subject = random.choice(SUBJECTS)

    topic = random.choice(TOPICS[subject])
    b = round(random.uniform(*difficulty_range), 2)
    a = round(random.uniform(0.8, 1.5), 2)
    c = 0.25

    p = p_correct_3pl(theta, a, b, c)
    correct = int(random.random() < p)

    # Time taken: harder questions take longer, jitter included
    base_time = 30 + 20 * abs(b)  # seconds
    time_taken = max(5.0, base_time + random.gauss(0, 10))

    q = Question(
        question_id=str(uuid.uuid4())[:8],
        subject=subject,
        topic=topic,
        difficulty_b=b,
        discrimination_a=a,
        guessing_c=c,
        time_taken=round(time_taken, 1),
        correct=correct,
        timestamp=datetime.now(),
        theta_at_attempt=theta,
        p_correct=round(p, 3),
    )
    q.lucky_flag = is_statistically_lucky(q)
    return q


def run_simulation_batch(
    session: SessionState,
    n_questions: int = 20,
    subject: Optional[str] = None,
) -> SessionState:
    """Run a batch of simulated questions, updating session state."""
    for _ in range(n_questions):
        q = simulate_question(session.theta, subject)
        q.lucky_flag = is_statistically_lucky(q)
        session.questions.append(q)
        if not q.lucky_flag:
            session.theta = update_theta(session.theta, q)
        session.theta_history.append(
            {"timestamp": q.timestamp, "theta": round(session.theta, 4)}
        )
    return session


def simulate_full_session(
    wake_hour: int = 7,
    wake_minute: int = 30,
    n_questions: int = 80,
    include_yoga: bool = True,
) -> SessionState:
    """Build a complete simulated day session for demo purposes."""
    now = datetime.now()
    wake_time = now.replace(hour=wake_hour, minute=wake_minute, second=0, microsecond=0)

    session = SessionState(wake_time=wake_time, theta=random.gauss(0.3, 0.5))
    session.circadian_penalty = compute_circadian_penalty(wake_time)

    # Simulate yoga session mid-morning
    if include_yoga:
        yoga_time = wake_time + timedelta(hours=2)
        yoga = YogaSession(timestamp=yoga_time, duration_minutes=40, heart_rate=72)
        session.yoga_sessions.append(yoga)
        session.yoga_boost_until = yoga_time + timedelta(hours=3)

    # Simulate idle time
    session.total_idle_minutes = random.uniform(30, 90)

    # Simulate questions across the day
    run_simulation_batch(session, n_questions=n_questions)

    return session


# ─── Session → Pandas DataFrames ──────────────────────────────────────────────

def questions_to_df(questions: List[Question]) -> pd.DataFrame:
    if not questions:
        return pd.DataFrame()
    rows = []
    for q in questions:
        rows.append({
            "question_id": q.question_id,
            "subject": q.subject,
            "topic": q.topic,
            "difficulty_b": q.difficulty_b,
            "discrimination_a": q.discrimination_a,
            "guessing_c": q.guessing_c,
            "time_taken": q.time_taken,
            "correct": q.correct,
            "timestamp": q.timestamp,
            "theta_at_attempt": q.theta_at_attempt,
            "lucky_flag": q.lucky_flag,
            "p_correct": q.p_correct,
        })
    return pd.DataFrame(rows)


def theta_history_to_df(history: list) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["timestamp", "theta"])
    return pd.DataFrame(history)


# ─── Derived Metrics Bundle ───────────────────────────────────────────────────

def compute_all_metrics(session: SessionState, config: dict = None) -> dict:
    """
    Single call to get all dashboard metrics.
    If config provided, uses its shutdown_threshold for lockout check.
    """
    now = datetime.now()
    hours_awake = max((now - session.wake_time).total_seconds() / 3600.0, 0.01)

    # Determine active subject (most recent or default)
    subject = "Physics"
    if session.questions:
        subject = session.questions[-1].subject

    yoga_active = (
        session.yoga_boost_until is not None
        and now < session.yoga_boost_until
    )

    # Entropy
    entropy = compute_entropy(hours_awake, subject, yoga_active=yoga_active)
    fatigue = compute_fatigue(entropy)

    # VAD
    vad = classify_vad(session.questions)

    # Heat
    heat = compute_heat(entropy, fatigue, vad["state"])
    zone = heat_zone(heat)

    # Opportunity cost
    opp_cost = compute_opportunity_cost(session.total_idle_minutes)

    # Projected score
    # After computing opp_cost, add:
    activation_boost = compute_activation_boost(session, datetime.now())
    projected = project_iat_score(
        session.theta,
        session.circadian_penalty,
        opp_cost,
        activation_boost
    )
    grade = score_to_grade(projected)
    admit_prob = admission_probability(projected)

    # Rolling stats
    acc = rolling_accuracy(session.questions, window=20)
    vel = rolling_velocity(session.questions, window=10)

    # Shutdown check (using config if provided)
    threshold = 0.45
    if config and "shutdown_threshold" in config:
        threshold = config["shutdown_threshold"]
    lockout = False
    if len(session.questions) >= 10:
        lockout = rolling_accuracy(session.questions, window=20) < threshold

    return {
        "hours_awake": round(hours_awake, 2),
        "theta": round(session.theta, 4),
        "entropy": round(entropy, 4),
        "fatigue": round(fatigue, 4),
        "vad_state": vad["state"],
        "vad_velocity": vad["velocity"],
        "vad_accuracy": vad["accuracy"],
        "heat": round(heat, 4),
        "heat_zone": zone,
        "circadian_penalty": round(session.circadian_penalty, 4),
        "opportunity_cost": round(opp_cost, 2),
        "projected_iat": round(projected, 1),
        "grade": grade,
        "admit_prob": round(admit_prob * 100, 1),
        "rolling_accuracy": round(acc, 3),
        "rolling_velocity": round(vel, 3),
        "total_questions": len(session.questions),
        "lockout": lockout,
        "yoga_active": yoga_active,
    }


# ─── Build Time-Series for Graphs ─────────────────────────────────────────────

def build_time_series(session: SessionState) -> pd.DataFrame:
    """
    Build a row-per-question DataFrame with running metrics for plotting.
    """
    rows = []
    running_theta = 0.0
    for i, q in enumerate(session.questions):
        hours_awake = max((q.timestamp - session.wake_time).total_seconds() / 3600.0, 0.01)
        yoga_active = (
            session.yoga_boost_until is not None
            and q.timestamp < session.yoga_boost_until
        )
        entropy = compute_entropy(hours_awake, q.subject, yoga_active)
        fatigue = compute_fatigue(entropy)
        running_theta = q.theta_at_attempt

        # Rolling accuracy up to this point
        past = session.questions[: i + 1]
        acc = rolling_accuracy(past, window=20)

        rows.append({
            "idx": i + 1,
            "timestamp": q.timestamp,
            "theta": running_theta,
            "entropy": entropy,
            "fatigue": fatigue,
            "accuracy": acc,
            "correct": q.correct,
            "time_taken": q.time_taken,
            "subject": q.subject,
            "lucky": q.lucky_flag,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── Formatting Helpers ───────────────────────────────────────────────────────

def format_timedelta(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def grade_color(grade: str) -> str:
    return {
        "A+": "#00ff9f",
        "A":  "#44ffbb",
        "B+": "#88ff44",
        "B":  "#ccff00",
        "C":  "#ffcc00",
        "D":  "#ff8800",
        "F":  "#ff2200",
    }.get(grade, "#ffffff")


def zone_color(zone: str) -> str:
    return {"GREEN": "#00ff9f", "YELLOW": "#ffcc00", "RED": "#ff3333"}.get(zone, "#888")

def process_question_batch(session: SessionState, questions_data: list) -> int:
    """
    Process a batch of question attempts and update session state.
    Each dict in questions_data must contain keys:
    subject, topic, difficulty_b, discrimination_a, time_taken, correct
    Returns number of questions processed.
    """
    count = 0
    for data in questions_data:
        q = Question(
            question_id=str(uuid.uuid4())[:8],
            subject=data["subject"],
            topic=data.get("topic", data["subject"]),
            difficulty_b=data["difficulty_b"],
            discrimination_a=data.get("discrimination_a", 1.0),
            guessing_c=data.get("guessing_c", 0.25),
            time_taken=data["time_taken"],
            correct=data["correct"],
            timestamp=datetime.now(),
            theta_at_attempt=session.theta,
        )
        q.lucky_flag = is_statistically_lucky(q)
        q.p_correct = p_correct_3pl(q.theta_at_attempt, q.discrimination_a,
                                     q.difficulty_b, q.guessing_c)
        session.questions.append(q)
        if not q.lucky_flag:
            session.theta = update_theta(session.theta, q)
        session.theta_history.append({
            "timestamp": q.timestamp,
            "theta": round(session.theta, 4),
        })
        count += 1
    return count


# ----------------------------------------------------------------------
# Persistence (Save / Load)
# ----------------------------------------------------------------------

SAVE_FILE = "icarus_save.json"

def _datetime_to_str(obj):
    """Convert datetime objects to ISO string for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def _datetime_from_str(dct):
    """Convert ISO strings back to datetime objects."""
    for key, value in dct.items():
        if isinstance(value, str) and key.endswith("_time") or key == "timestamp":
            try:
                dct[key] = datetime.fromisoformat(value)
            except (ValueError, TypeError):
                pass
    return dct

def save_session(session: SessionState, config: dict, filepath: str = SAVE_FILE) -> None:
    """Save the current session state and config to a JSON file."""
    # Convert session to dict, handling dataclasses and datetime
    session_dict = asdict(session)
    # Convert any remaining datetime objects (nested in lists)
    session_dict = json.loads(json.dumps(session_dict, default=_datetime_to_str))
    config_copy = config.copy()
    data = {
        "session": session_dict,
        "config": config_copy,
        "version": 1
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_session(filepath: str = SAVE_FILE):
    """Load session and config from a JSON file. Returns (session, config) or (None, None)."""
    if not os.path.exists(filepath):
        return None, None
    with open(filepath, "r") as f:
        data = json.load(f, object_hook=_datetime_from_str)
    # Rebuild SessionState from dict
    session_dict = data["session"]
    # Recreate Question and YogaSession lists (they are dataclasses)
    from models import Question, YogaSession, SessionState
    questions = []
    for q_dict in session_dict.get("questions", []):
        questions.append(Question(**q_dict))
    yoga_sessions = []
    for y_dict in session_dict.get("yoga_sessions", []):
        yoga_sessions.append(YogaSession(**y_dict))
    # Recreate theta_history (list of dicts with timestamp)
    theta_history = session_dict.get("theta_history", [])
    # Build SessionState
    session = SessionState(
        wake_time=session_dict["wake_time"],
        theta=session_dict["theta"],
        theta_history=theta_history,
        questions=questions,
        yoga_sessions=yoga_sessions,
        idle_blocks=session_dict.get("idle_blocks", 0),
        total_idle_minutes=session_dict.get("total_idle_minutes", 0.0),
        yoga_boost_until=session_dict.get("yoga_boost_until"),
        circadian_penalty=session_dict.get("circadian_penalty", 0.0),
        entropy_k_override=session_dict.get("entropy_k_override"),
    )
    config = data.get("config", {})
    return session, config

def delete_save(filepath: str = SAVE_FILE) -> None:
    """Remove the save file if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)


def recompute_theta_from_questions(questions: List[Question], initial_theta: float = 0.0) -> float:
    """
    Recompute latent ability theta by sequentially processing all questions
    (ignoring lucky flags). Returns the final theta.
    """
    theta = initial_theta
    for q in questions:
        if not q.lucky_flag:
            theta = update_theta(theta, q)
        # Update theta_at_attempt for each question to reflect recalculated value
        q.theta_at_attempt = theta
    return theta

def delete_question_by_index(session: SessionState, index: int) -> bool:
    """
    Delete a question from session.questions at the given index (0-based).
    Also removes corresponding entry from theta_history and recomputes theta
    from the remaining questions.
    Returns True if successful.
    """
    if index < 0 or index >= len(session.questions):
        return False

    # Remove the question
    del session.questions[index]

    # Remove corresponding theta_history entry (if aligned by index)
    if index < len(session.theta_history):
        del session.theta_history[index]

    # Recompute theta from scratch using remaining questions
    # Start from initial theta = 0.0 (or could keep previous? safer to reset)
    new_theta = recompute_theta_from_questions(session.questions, initial_theta=0.0)
    session.theta = new_theta

    # Also update theta_at_attempt in each question (already done inside recompute)
    return True

def delete_questions_by_indices(session: SessionState, indices: List[int]) -> int:
    """
    Delete multiple questions by their indices (0-based).
    Indices are processed in descending order to avoid shifting issues.
    Returns the number of deleted questions.
    """
    if not indices:
        return 0

    # Sort indices descending so deletion doesn't affect earlier positions
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(session.questions):
            del session.questions[idx]
            if idx < len(session.theta_history):
                del session.theta_history[idx]

    # Recompute theta from the remaining questions
    new_theta = recompute_theta_from_questions(session.questions, initial_theta=0.0)
    session.theta = new_theta


    return len(indices)