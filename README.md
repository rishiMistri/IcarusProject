⚡ Icarus Protocol

Real-time cognitive-physiological optimizer for high-stakes test preparation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

![License](https://img.shields.io/badge/License-MIT-green)

Icarus Protocol is a behavioural feedback dashboard that models latent ability using 3PL Item Response Theory (IRT), tracks cognitive entropy, applies circadian penalties, and projects an IAT (Ivy-equivalent) score. It functions as a constraint system – not a simple tracker – by forcing breaks, flagging lucky guesses, and visualising the real cost of fatigue and idleness.

---
✨ Features

IRT 3PL ability estimation (θ) with statistically-lucky guess detection


Subject-specific entropy growth (Math 0.08, Physics 0.06, Chemistry 0.05)

Circadian penalty for waking after 07:30 (configurable per 30min)

Opportunity cost from idle minutes (–1.5 points per 15min block)

Velocity-Accuracy Divergence (VAD) state machine: `OPTIMAL` / `PANIC` / `EXHAUSTION` / `GRINDING`

Heat gauge (GREEN → YELLOW → RED) with lockout when accuracy < threshold

Yoga session logging → reduces entropy growth (k × 0.85) for 3 hours

Strength activation (push-ups / pull-ups) → temporary θ boost (+0.1 decaying over 30 min)

Batch CSV upload – 10-100 questions with columns: `subject, difficulty_b, time_taken, correct`

Single / batch deletion of logs with automatic θ recomputation

Full session persistence – save/load/reset via JSON

Simulation engine – generate random or subject-specific MCQ attempts

Interactive Plotly charts – θ over time, entropy+fatigue, accuracy+velocity

Cyber-HUD UI – custom CSS (Orbitron, Share Tech Mono)

---

🛠️ Tech Stack

Layer	Technology

Frontend	Streamlit, Plotly, Pandas

Backend	Python 3.10+

State	Dataclasses + JSON persistence

Maths	NumPy, math (IRT, entropy, fatigue)

Styling	Custom CSS (cyber theme)

---

🚀 Installation

Clone the repository
```bash
   git clone https://github.com/yourusername/icarus-protocol.git
   cd icarus-protocol
   ```
Create a virtual environment (optional but recommended)
```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```
Install dependencies
```bash
   pip install -r requirements.txt
   ```
If `requirements.txt` is not present, install manually:
```bash
   pip install streamlit pandas plotly numpy
   ```
Run the application
```bash
   streamlit run main.py
   ```
---
📖 Usage

Sidebar Configuration

Control	Description

Shutdown Accuracy %	Lockout threshold (e.g., 45%) – blocks input when accuracy falls below

Circadian Penalty / 30min	Penalty rate for waking after 07:30

VAD Window (questions)	Number of recent questions for velocity/accuracy classification

Idle Time (minutes)	Manual entry of break minutes → opportunity cost penalty

Log Yoga Session	Duration in minutes – reduces entropy growth for 3h

Log Strength Set	Push-ups / pull-ups reps → temporary θ boost

Batch CSV Upload	Upload 10-100 questions for bulk processing

Delete Logs	Single or batch deletion with automatic θ recomputation

Simulation	Generate random or subject-specific MCQ attempts

Save / Load / Reset	Persistent state across app restarts

Dashboard Panels

Header – Global grade, θ ability, projected IAT, admit probability, system status

Left Column – Batch upload, delete controls, recent question log (with "LUCKY" flag)

Centre Column – Three interactive charts (θ over time, entropy+fatigue, accuracy+velocity)

Right Column – VAD state, heat gauge, circadian penalty, entropy & fatigue %, opportunity cost

Forecast Panel – Projected IAT gauge (0-180), admission probability, letter grade

Footer – Subject performance matrix (accuracy, avg time, mean θ per subject)

---

🧠 Mathematical Core
Component	Formula	Effect on IAT Score

Ability (θ)	`P(correct) = c + (1-c)/(1+exp(-a(θ-b)))`	

Mapped to raw IAT via `90 + 45·tanh(0.6·θ)`

Cognitive Entropy	`S(t) = S₀·exp(k·t)`	Indirect – increases fatigue (visual only)

Circadian Penalty	`μ = (delay_min/30)·rate`, capped at 0.95	Multiplicative: `final = raw·(1-μ)`

Opportunity Cost	`cost = –(idle/15)·1.5`	Additive penalty to IAT

Activation Boost	Push-ups/pull-ups → +0.1 θ (decaying)	Increases raw IAT via higher effective θ

VAD State	Velocity vs Accuracy over window	Affects heat gauge (PANIC +0.25, EXHAUSTION +0.15)

Shutdown Lockout	Rolling accuracy < threshold	Blocks new inputs, forces recovery

---
📁 File Structure
```
icarus-protocol/
├── main.py
├── models.py
├── utils.py
├── ui.py
├── requirements.txt
├── LICENSE
└── README.md
```
---
🧪 Example CSV Batch Upload Format

subject	difficulty_b	time_taken	correct	discrimination_a (optional)	guessing_c (optional)

Physics	1.2	45.0	1	1.0	0.25

Chemistry	-0.5	30.0	0	1.2	0.25

Math	2.1	75.0	1	1.5	0.25

Required columns: `subject`, `difficulty_b`, `time_taken`, `correct` 

Optional: `discrimination_a`, `guessing_c`, `topic`

---
📜 License

MIT License

---
📧 Contact

Author: Rishi Mistri 

GitHub: https://github.com/rishiMistri

LinkedIn: www.linkedin.com/in/rishi-mistri-366918304

---
Icarus Protocol – not a productivity tool, but a constraint system. A feedback controller.
