"""
Icarus Protocol — Core Mathematical Engine
==========================================
3PL IRT model, VAD classifier, entropy system, fatigue/circadian logic.
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import math


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Question:
    question_id: str
    subject: str          # Physics / Chemistry / Math
    topic: str
    difficulty_b: float   # IRT difficulty
    discrimination_a: float = 1.0
    guessing_c: float = 0.25
    time_taken: float = 0.0   # seconds
    correct: int = 0          # 0 or 1
    timestamp: datetime = field(default_factory=datetime.now)
    theta_at_attempt: float = 0.0
    lucky_flag: bool = False
    p_correct: float = 0.0


@dataclass
class YogaSession:
    timestamp: datetime
    duration_minutes: float
    heart_rate: Optional[float] = None


@dataclass
class SessionState:
    wake_time: datetime
    theta: float = 0.0          # Latent ability estimate
    theta_history: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    yoga_sessions: list = field(default_factory=list)
    idle_blocks: int = 0        # 15-min idle blocks
    total_idle_minutes: float = 0.0
    yoga_boost_until: Optional[datetime] = None
    circadian_penalty: float = 0.0
    entropy_k_override: Optional[float] = None   # post-yoga reduced k
    strength_sessions: list = field(default_factory=list)

# ─── 3PL IRT Model ────────────────────────────────────────────────────────────

def p_correct_3pl(theta: float, a: float, b: float, c: float) -> float:
    """3-Parameter Logistic IRT probability of correct response."""
    return c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))


def is_statistically_lucky(q: Question) -> bool:
    """Flag responses that are statistically suspicious."""
    return (
        q.time_taken < 15.0
        and q.difficulty_b > 1.0
        and q.correct == 1
    )


def update_theta(theta: float, question: Question, learning_rate: float = 0.3) -> float:
    """
    Simplified Bayesian-gradient update of latent ability theta.
    Uses first derivative of log-likelihood of the 3PL model.
    Skips update for statistically lucky responses.
    """
    if question.lucky_flag:
        return theta  # No credit for lucky guesses

    a = question.discrimination_a
    b = question.difficulty_b
    c = question.guessing_c
    u = question.correct

    p = p_correct_3pl(theta, a, b, c)

    # Guard against numerical instability
    p = np.clip(p, 1e-6, 1.0 - 1e-6)

    # Score function (gradient of log-likelihood)
    numerator = a * (1.0 - c) * math.exp(-a * (theta - b))
    denominator = (c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))) * \
                  (1.0 + math.exp(-a * (theta - b))) ** 2

    # Avoid division by zero
    if denominator < 1e-10:
        return theta

    score = numerator / denominator
    delta = learning_rate * (u - p) * score

    # Clamp theta to realistic range [-4, 4]
    return float(np.clip(theta + delta, -4.0, 4.0))


# ─── Entropy Model ────────────────────────────────────────────────────────────

ENTROPY_K = {
    "Math": 0.08,
    "Physics": 0.06,
    "Chemistry": 0.05,
    "Revision": 0.02,
}

BASE_ENTROPY = 1.0  # S0


def compute_entropy(
    hours_awake: float,
    subject: str,
    yoga_active: bool = False,
    k_override: Optional[float] = None
) -> float:
    """
    S(t) = S0 * exp(k * t)
    Yoga reduces k by 15% for 3 hours post-session.
    """
    k_base = ENTROPY_K.get(subject, 0.06)
    if k_override is not None:
        k = k_override
    elif yoga_active:
        k = k_base * 0.85  # 15% reduction
    else:
        k = k_base
    return BASE_ENTROPY * math.exp(k * hours_awake)


def compute_fatigue(entropy: float) -> float:
    """Fatigue is a normalised function of entropy. Returns 0.0–1.0."""
    # Saturates at entropy ~ 5
    return float(np.clip((entropy - 1.0) / 4.0, 0.0, 1.0))


# ─── Circadian Penalty ────────────────────────────────────────────────────────

def compute_circadian_penalty(wake_time: datetime, penalty_per_30min: float = 0.05) -> float:
    """
    Penalty μ applied per 30-minute delay past 07:30.
    effective_score = raw_score * (1 - μ)
    Returns μ (0–1).
    """
    target = wake_time.replace(hour=7, minute=30, second=0, microsecond=0)
    if wake_time <= target:
        return 0.0
    delay_minutes = (wake_time - target).total_seconds() / 60.0
    blocks = delay_minutes / 30.0
    mu = blocks * penalty_per_30min
    return float(np.clip(mu, 0.0, 0.95))


def apply_circadian_penalty(raw_score: float, mu: float) -> float:
    return raw_score * (1.0 - mu)


# ─── VAD (Velocity-Accuracy Divergence) ───────────────────────────────────────

VAD_STATES = ["OPTIMAL", "PANIC", "EXHAUSTION", "GRINDING", "UNKNOWN"]
VAD_COLORS = {
    "OPTIMAL":    "#00ff9f",
    "PANIC":      "#ff4444",
    "EXHAUSTION": "#ff9900",
    "GRINDING":   "#4488ff",
    "UNKNOWN":    "#888888",
}


def classify_vad(
    questions: list,
    window: int = 10,
    vel_threshold: float = 1.0,   # questions/min
    acc_threshold: float = 0.60
) -> dict:
    """
    Classify the last `window` questions into a VAD state.
    Returns state label and computed metrics.
    """
    if len(questions) < 2:
        return {"state": "UNKNOWN", "velocity": 0.0, "accuracy": 0.0}

    recent = questions[-window:]

    # Velocity: questions per minute over the window
    t0 = recent[0].timestamp
    t1 = recent[-1].timestamp
    elapsed_min = max((t1 - t0).total_seconds() / 60.0, 0.01)
    velocity = len(recent) / elapsed_min

    # Accuracy
    accuracy = sum(q.correct for q in recent) / len(recent)

    if velocity >= vel_threshold and accuracy >= acc_threshold:
        state = "OPTIMAL"
    elif velocity >= vel_threshold and accuracy < acc_threshold:
        state = "PANIC"
    elif velocity < vel_threshold and accuracy < acc_threshold:
        state = "EXHAUSTION"
    elif velocity < vel_threshold and accuracy >= acc_threshold:
        state = "GRINDING"
    else:
        state = "UNKNOWN"

    return {"state": state, "velocity": round(velocity, 3), "accuracy": round(accuracy, 3)}


# ─── Heat Gauge ───────────────────────────────────────────────────────────────

def compute_heat(entropy: float, fatigue: float, vad_state: str) -> float:
    """
    Composite heat score 0–1.
    PANIC adds 0.25, EXHAUSTION adds 0.15.
    """
    vad_bonus = {"PANIC": 0.25, "EXHAUSTION": 0.15}.get(vad_state, 0.0)
    # Normalise entropy to 0-1 range (max sane value ~5)
    e_norm = np.clip((entropy - 1.0) / 4.0, 0.0, 1.0)
    heat = (0.4 * e_norm + 0.35 * fatigue + 0.25 * vad_bonus)
    return float(np.clip(heat, 0.0, 1.0))


def heat_zone(heat: float) -> str:
    if heat < 0.40:
        return "GREEN"
    elif heat < 0.70:
        return "YELLOW"
    else:
        return "RED"


# ─── Opportunity Cost ─────────────────────────────────────────────────────────

def compute_opportunity_cost(idle_minutes: float, score_per_block: float = 1.5) -> float:
    """
    Each 15-min idle block → −score_per_block projected IAT points.
    """
    blocks = idle_minutes / 15.0
    return -blocks * score_per_block


# ─── Score Projection ─────────────────────────────────────────────────────────

def project_iat_score(
    theta: float,
    circadian_penalty: float,
    opportunity_cost_loss: float,
    activation_boost: float = 0.0   # new parameter
) -> float:
    raw = 90.0 + 45.0 * math.tanh((theta + activation_boost) * 0.6)
    penalised = apply_circadian_penalty(raw, circadian_penalty)
    final = penalised + opportunity_cost_loss
    return float(np.clip(final, 0.0, 180.0))


def score_to_grade(projected: float) -> str:
    if projected >= 155: return "A+"
    if projected >= 140: return "A"
    if projected >= 125: return "B+"
    if projected >= 110: return "B"
    if projected >= 90:  return "C"
    if projected >= 70:  return "D"
    return "F"


def admission_probability(projected_score: float) -> float:
    """Sigmoid admission probability. Threshold ~150."""
    return float(np.clip(1.0 / (1.0 + math.exp(-(projected_score - 140) / 10.0)), 0.0, 1.0))


# ─── Rolling Statistics ───────────────────────────────────────────────────────

def rolling_accuracy(questions: list, window: int = 20) -> float:
    if not questions:
        return 0.0
    recent = questions[-window:]
    return sum(q.correct for q in recent) / len(recent)


def rolling_velocity(questions: list, window: int = 10) -> float:
    if len(questions) < 2:
        return 0.0
    recent = questions[-window:]
    t0, t1 = recent[0].timestamp, recent[-1].timestamp
    elapsed = max((t1 - t0).total_seconds() / 60.0, 0.01)
    return len(recent) / elapsed

@dataclass
class StrengthSession:
    timestamp: datetime
    pushups: int = 0
    pullups: int = 0
    boost_until: Optional[datetime] = None   # activation boost expiry

def compute_activation_boost(session: SessionState, now: datetime) -> float:
    """
    Returns a temporary θ boost (0 to +0.2) based on recent strength exercises.
    Each pushup/pullup set adds +0.1, decays to zero after 30 minutes.
    """
    total_boost = 0.0
    for s in session.strength_sessions:
        if s.boost_until and now < s.boost_until:
            # Boost decays linearly from start to end
            duration = (s.boost_until - s.timestamp).total_seconds()
            elapsed = (now - s.timestamp).total_seconds()
            if duration > 0:
                remaining = 1.0 - (elapsed / duration)
                boost = 0.1 * remaining   # max 0.1 per session
                total_boost += boost
    return min(total_boost, 0.3)   # cap at +0.3 θ