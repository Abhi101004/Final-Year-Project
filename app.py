import streamlit as st
import time
import joblib
import pandas as pd
import numpy as np
import subprocess
import sys
import os
from datetime import datetime  # NEW: for timestamp

# =========================================
# CONFIG: PATH TO SESSION "DATABASE" (CSV)
# =========================================
# Always write/read the CSV next to this app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_CSV = os.path.join(BASE_DIR, "ppd_sessions_history.csv")

# =========================================
# 🔗 CONFIG: UNITY GAME APPS (LOCAL .app FILES)
# =========================================
FOREST_WALK_APP = " "   # path to your built Unity app
FARMING_GAME_APP = "/Users/abhilavanya/Downloads/Pentafarm-main 2/FARM GAME.app"


def launch_forest_walk():
    """Launch the Forest Walk Unity app."""
    try:
        if sys.platform != "darwin":   # only macOS
            return False, "Forest Walk launch is only configured for macOS."

        if not os.path.exists(FOREST_WALK_APP):
            return False, f"Forest Walk app not found at: {FOREST_WALK_APP}"

        subprocess.Popen(["open", FOREST_WALK_APP])
        return True, None
    except Exception as e:
        return False, str(e)


def launch_farming_game():
    """Launch the Farming Game Unity app."""
    try:
        if sys.platform != "darwin":
            return False, "Farming Game launch is only configured for macOS."

        if not os.path.exists(FARMING_GAME_APP):
            return False, f"Farming Game app not found at: {FARMING_GAME_APP}"

        subprocess.Popen(["open", FARMING_GAME_APP])
        return True, None
    except Exception as e:
        return False, str(e)


# =========================================
# 🔔 macOS Notification Helper
# =========================================
def mac_notification(title: str, message: str):
    """
    Show a native macOS notification (Notification Center).
    Does nothing on non-macOS systems.
    """
    if sys.platform == "darwin":
        os.system(
            f'''osascript -e 'display notification "{message}" with title "{title}"' '''
        )


# =========================================
# 🗄️ SESSION HISTORY HELPERS (CSV "DATABASE")
# =========================================
def load_session_history() -> pd.DataFrame:
    """Load all past sessions from CSV, or return empty DataFrame."""
    if os.path.exists(SESSIONS_CSV):
        try:
            df = pd.read_csv(SESSIONS_CSV)
            # simple sanity check
            if "Session_Number" in df.columns:
                return df
            else:
                st.warning("Session history file exists but has no valid columns yet.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Could not read session history CSV: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def append_session_to_history(row: dict):
    """Append one completed session to the CSV database."""
    df_new = pd.DataFrame([row])
    file_exists = os.path.exists(SESSIONS_CSV)
    try:
        df_new.to_csv(SESSIONS_CSV, mode="a", header=not file_exists, index=False)
    except Exception as e:
        st.error(f"Could not write to session history CSV: {e}")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="PPD Mood & Therapy Companion",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
<style>
    :root {
        --primary-bg: #fef9f3;
        --primary-bg-soft: #fef5ed;
        --primary-bg-soft2: #f8f3f0;
        --primary-text: #2d1b4e;
        --secondary-text: #7e6ba6;
        --accent: #c6a7fe;
        --accent-strong: #5e4b8b;
        --card-bg: #ffffff;
    }

    .stApp {
        background: radial-gradient(circle at top left, var(--primary-bg) 0%, var(--primary-bg-soft) 40%, var(--primary-bg-soft2) 100%);
        color: var(--primary-text);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Make ALL normal text dark and legible */
    body, .stApp,
    label, p, span, li,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li {
        color: var(--primary-text) !important;
    }

    /* Cards */
    .wellness-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 26px 28px;
        box-shadow: 0 10px 35px rgba(94, 75, 139, 0.12);
        margin: 20px 0;
        border: 1.5px solid rgba(198, 167, 254, 0.45);
        backdrop-filter: blur(10px);
    }

    .mood-card {
        background: linear-gradient(145deg, #fff9f5 0%, #ffffff 60%, #f5ecff 100%);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        border: 1.5px solid #e4d8ff;
        transition: all 0.25s ease;
    }
    .mood-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 12px 28px rgba(198, 167, 254, 0.25);
        border-color: #c6a7fe;
    }

    .support-message {
        background: linear-gradient(135deg, #e9f7ef 0%, #f3fce9 100%);
        border-left: 5px solid #5da869;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0 24px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #1b5e20 !important;
        font-weight: 500;
    }

    .warning-message {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 16px 0 24px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #e65100 !important;
        font-weight: 500;
    }

    .results-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 50%, #f8e9ff 100%);
        border-radius: 22px;
        padding: 28px 30px 26px 30px;
        margin: 20px 0 10px 0;
        border: 1.8px solid #c6a7fe;
        box-shadow: 0 10px 28px rgba(94, 75, 139, 0.25);
    }

    .big-emoji { font-size: 48px; margin: 10px; }

    /* Progress steps */
    .progress-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #d7cdef;
        color: white;
        text-align: center;
        margin: 0 10px;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .progress-step.active {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        box-shadow: 0 4px 15px rgba(198, 167, 254, 0.4);
    }
    .progress-step.completed {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
    }

    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.08); opacity: 1; }
    }
    .breathe-circle {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: radial-gradient(circle at top, #c6a7fe 0%, #e1bee7 45%, #f3e5f5 100%);
        margin: 20px auto 8px auto;
        animation: breathe 4s ease-in-out infinite;
        box-shadow: 0 10px 30px rgba(198, 167, 254, 0.5);
    }
    .breathe-text {
        text-align:center;
        font-size: 13px;
        color: var(--secondary-text);
        margin-top: 6px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 25px;
        padding: 10px 26px;
        font-size: 15px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(198, 167, 254, 0.35);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-1.5px);
        box-shadow: 0 7px 22px rgba(198, 167, 254, 0.45);
        filter: brightness(1.03);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 14px 14px 0 0;
        padding: 10px 20px;
        font-size: 15px;
        font-weight: 600;
        color: var(--secondary-text);
        box-shadow: 0 -2px 0 rgba(0,0,0,0.04);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 18px rgba(198, 167, 254, 0.5);
    }

    /* ===== SELECT / DROPDOWN FIXES ===== */

    /* Closed select box (the white field) */
    div[data-baseweb="select"] > div {
        color: var(--primary-text) !important;
        font-size: 14px;
        background-color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #d5c5ff !important;
    }
    div[data-baseweb="select"] svg {
        color: var(--accent-strong) !important;
    }

    /* 🔹 The OPEN dropdown menu is rendered in a separate portal */
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #14121f !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid #d5c5ff !important;
    }

    /* Make ALL option text inside the menu WHITE */
    [data-baseweb="menu"] * ,
    [role="listbox"] li,
    [role="listbox"] li * {
        color: #ffffff !important;
    }

    /* Selected option highlight */
    [role="listbox"] li[aria-selected="true"] {
        background-color: #4b3a80 !important;
    }

    /* Sliders */
    [data-baseweb="slider"] > div {
        color: var(--primary-text) !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        box-shadow: 0 0 0 2px var(--accent) !important;
    }
    [data-baseweb="slider"] .rc-slider-track {
        background-color: var(--accent) !important;
    }

    /* Metrics text color fix */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricLabel"] {
        color: var(--primary-text) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helper functions
# -----------------------------
def compute_tmd(subscale_scores, K=100):
    T = subscale_scores.get("T", 0)
    D = subscale_scores.get("D", 0)
    A = subscale_scores.get("A", 0)
    F = subscale_scores.get("F", 0)
    C = subscale_scores.get("C", 0)
    V = subscale_scores.get("V", 0)
    return (T + D + A + F + C) - V + K


def compute_delta_tmd(pre_scores, post_scores, K=100):
    TMD_pre = compute_tmd(pre_scores, K=K)
    TMD_post = compute_tmd(post_scores, K=K)
    delta = TMD_pre - TMD_post
    return TMD_pre, TMD_post, delta


def get_mood_emoji(v):
    return ["😌", "🙂", "😐", "😟", "😰"][min(max(v, 0), 4)]


def get_vigor_emoji(v):
    return ["😴", "😪", "🙂", "😊", "✨"][min(max(v, 0), 4)]


def therapy_from_risk(ppd_percent: float):
    if ppd_percent >= 85:
        risk = "High Risk"
        therapy = "Clinician Escalation"
        detail = (
            "PPD risk is very high (≥ 85%). Please reach out to a clinician or "
            "mental health professional as soon as possible. This app does not "
            "replace professional care."
        )
    elif ppd_percent >= 65:
        risk = "Moderate Risk"
        therapy = "Nature Walk Therapy"
        detail = (
            "PPD risk is moderate (65–84%). Recommend guided **Nature Walk** "
            "therapy sessions with regular monitoring."
        )
    else:
        risk = "Low Risk"
        therapy = "User Choice: Nature Walk or Farming Game"
        detail = (
            "PPD risk is below 65%. You may choose between **Nature Walk** or "
            "**Farming Game** therapy as a gentle wellness activity."
        )
    return risk, therapy, detail


@st.cache_resource
def load_mdkr_objects():
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    dt = joblib.load("dt_model.pkl")
    rf = joblib.load("rf_model.pkl")
    knn = joblib.load("knn_model.pkl")
    meta_model = joblib.load("mdkr_meta_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return imputer, scaler, dt, rf, knn, meta_model, feature_cols


def preprocess_prams_for_model(df_new: pd.DataFrame, feature_cols):
    yes_no_cols = [
        "Feeling sad or Tearful",
        "Irritable towards baby & partner",
        "Feeling lonely or unsupported",
        "Feeling of guilt",
        "Problems of bonding with baby",
        "Suicide attempt",
    ]
    for col in yes_no_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].map(
                {
                    "Yes": 1,
                    "No": 0,
                    "Sometimes": 1,
                    "Maybe": 1,
                    "Not interested to say": 0.5,
                }
            )

    age_map = {
        "<20": 0,
        "20-25": 1,
        "25-30": 2,
        "30-35": 3,
        "35-40": 4,
        "40-45": 5,
        "45-50": 6,
        "50+": 7,
    }
    if "Age" in df_new.columns:
        df_new["Age"] = df_new["Age"].replace(age_map)

    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0

    return df_new[feature_cols]


def predict_ppd_and_therapy_from_answers(user_answers: dict):
    imputer, scaler, dt, rf, knn, meta_model, feature_cols = load_mdkr_objects()
    df_new = pd.DataFrame([user_answers])
    df_new = preprocess_prams_for_model(df_new, feature_cols)

    X_imp = imputer.transform(df_new)
    X_sc = scaler.transform(X_imp)

    p_dt = dt.predict_proba(X_sc)[:, 1]
    p_rf = rf.predict_proba(X_sc)[:, 1]
    p_knn = knn.predict_proba(X_sc)[:, 1]
    meta_features = np.column_stack([p_dt, p_rf, p_knn])

    ppd_prob = meta_model.predict_proba(meta_features)[:, 1][0]
    ppd_percent = ppd_prob * 100

    risk, therapy, detail = therapy_from_risk(ppd_percent)
    return ppd_percent, risk, therapy, detail


# -----------------------------
# Session state init
# -----------------------------
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "therapy_viewed" not in st.session_state:
    st.session_state.therapy_viewed = False

# -----------------------------
# 🔔 Check reminder timer (if any)
# -----------------------------
if "notify_ts" in st.session_state:
    if time.time() >= st.session_state["notify_ts"]:
        mac_notification(
            "PPD Therapy Reminder",
            "Please record your after-session mood in the PPD Companion app.",
        )
        st.toast("⏰ Time to record your after-session mood.", icon="🔔")
        del st.session_state["notify_ts"]

# -----------------------------
# Header + progress
# -----------------------------
st.markdown(
    """
<div style='text-align:center; padding: 12px 0 8px 0;'>
  <h1 style='color:#5e4b8b; font-size:40px; margin-bottom:4px;'>
    🌸 PPD Mood & Therapy Companion 🌸
  </h1>
  <p style='color:#7e6ba6; font-size:17px; max-width:620px; margin:0 auto;'>
    Lightweight companion to estimate PPD risk, assign therapy, and track mood before & after each session.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    step1 = (
        "completed"
        if "ppd_percent" in st.session_state
        else ("active" if st.session_state.current_step == 1 else "")
    )
    step2 = (
        "completed"
        if "TMD_pre" in st.session_state
        else ("active" if st.session_state.current_step == 2 else "")
    )
    step3 = (
        "completed"
        if st.session_state.get("therapy_viewed", False)
        else ("active" if st.session_state.current_step == 3 else "")
    )
    step4 = (
        "completed"
        if "TMD_post" in st.session_state
        else ("active" if st.session_state.current_step == 4 else "")
    )
    st.markdown(
        f"""
    <div style='text-align:center; margin: 8px 0 18px 0;'>
      <span class='progress-step {step1}'>1</span>
      <span style='color:#c6a7fe; font-weight:bold;'>━━</span>
      <span class='progress-step {step2}'>2</span>
      <span style='color:#c6a7fe; font-weight:bold;'>━━</span>
      <span class='progress-step {step3}'>3</span>
      <span style='color:#c6a7fe; font-weight:bold;'>━━</span>
      <span class='progress-step {step4}'>4</span>
      <br>
      <small style='color:#7e6ba6; font-size:13px;'>
        Risk Screener → Before Session Mood → Therapy Plan & Game → After Session + Summary
      </small>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Tabs
# -----------------------------
tab_screener, tab_before, tab_therapy, tab_after, tab_summary = st.tabs(
    [
        "📝 PPD Risk Screener",
        "🌅 Before Session",
        "🌿 Your Therapy Plan",
        "🌙 After Session",
        "📊 Session Summary",
    ]
)

# ---------- TAB 1: PPD RISK SCREENER ----------
with tab_screener:
    st.markdown(
        "<div class='support-message'>📝 Answer a few gentle questions so we can "
        "estimate your PPD risk and decide which therapy path fits best.</div>",
        unsafe_allow_html=True,
    )

    with st.form("ppd_form"):
        st.subheader("Age group")
        age = st.radio(
            "Age group",
            ["30-35", "35-40", "40-45", "45-50"],
            index=0,
            horizontal=True,
            key="age_radio"
        )

        st.subheader("Mood in the last 2 weeks")

        sad = st.radio(
            "Feeling sad or tearful",
            ["No", "Yes", "Sometimes", "Not interested to say"],
            index=0,
            horizontal=True,
            key="sad_radio"
        )

        irrit = st.radio(
            "Irritable towards baby & partner",
            ["No", "Yes", "Sometimes", "Not interested to say"],
            index=0,
            horizontal=True,
            key="irrit_radio"
        )

        lonely = st.radio(
            "Feeling lonely or unsupported",
            ["No", "Yes", "Sometimes", "Not interested to say"],
            index=0,
            horizontal=True,
            key="lonely_radio"
        )

        guilt = st.radio(
            "Feeling of guilt",
            ["No", "Yes", "Sometimes", "Not interested to say"],
            index=0,
            horizontal=True,
            key="guilt_radio"
        )

        bonding = st.radio(
            "Problems of bonding with baby",
            ["No", "Yes", "Sometimes", "Not interested to say"],
            index=0,
            horizontal=True,
            key="bonding_radio"
        )

        suicide = st.radio(
            "Any suicidal thoughts or attempts",
            ["No", "Yes", "Not interested to say"],
            index=0,
            horizontal=True,
            key="suicide_radio"
        )

        submitted = st.form_submit_button("🔍 Estimate PPD Risk")

    if submitted:
        user_answers = {
            "Age": age,
            "Feeling sad or Tearful": sad,
            "Irritable towards baby & partner": irrit,
            "Feeling lonely or unsupported": lonely,
            "Feeling of guilt": guilt,
            "Problems of bonding with baby": bonding,
            "Suicide attempt": suicide,
        }
        try:
            with st.spinner("Estimating your PPD risk gently..."):
                time.sleep(0.8)
                ppd_percent, risk, therapy, detail = predict_ppd_and_therapy_from_answers(
                    user_answers
                )

            st.session_state["ppd_percent"] = ppd_percent
            st.session_state["ppd_risk_level"] = risk
            st.session_state["assigned_therapy"] = therapy
            st.session_state["therapy_detail"] = detail
            st.session_state.current_step = max(st.session_state.current_step, 2)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated PPD Risk", f"{ppd_percent:.1f} %")
            with col2:
                st.metric("Risk Level", risk)

            risk_progress = min(max(int(ppd_percent), 0), 100)
            st.progress(risk_progress)

            st.info(f"**Assigned Therapy Path:** {therapy}")
            st.caption(
                "Next: go to **🌅 Before Session** tab and record your mood before starting the activity."
            )

            with st.expander("What does this estimate mean?"):
                st.write(detail)

        except Exception as e:
            st.error(
                "Prediction failed – ensure the preprocessing in `app.py` matches your notebook."
            )
            st.exception(e)

# ---------- TAB 2: BEFORE SESSION ----------
with tab_before:
    if "ppd_percent" not in st.session_state:
        st.markdown(
            """
        <div class='warning-message'>
          <h4>🧠 Please complete the PPD Risk Screener first</h4>
          <p>Go to the <b>📝 PPD Risk Screener</b> tab, answer the questions, and come back here.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class='support-message'>
          🌸 Before you start your therapy activity, let's gently record how you feel right now.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='wellness-card'>", unsafe_allow_html=True)
        st.subheader("How are you feeling right now?")
        st.caption("Rate each from 0 (not at all) to 4 (extremely).")

        colL, colR = st.columns(2)
        with colL:
            st.markdown("#### How calm or on-edge do you feel right now?")
            T_pre = st.slider(
                "Nervous / tense / on edge",
                0,
                4,
                st.session_state.get("T_pre", 0),
                key="T_pre_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(T_pre)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### How heavy or weighed-down do your emotions feel at this moment?")
            D_pre = st.slider(
                "Sad / hopeless / low",
                0,
                4,
                st.session_state.get("D_pre", 0),
                key="D_pre_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(D_pre)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### When small things happen around you, how quickly do they get on your nerves?")
            A_pre = st.slider(
                "Angry / irritated",
                0,
                4,
                st.session_state.get("A_pre", 0),
                key="A_pre_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(A_pre)}</div>",
                unsafe_allow_html=True,
            )

        with colR:
            st.markdown("#### How rested or drained do you feel right now?")
            F_pre = st.slider(
                "Exhausted / worn out",
                0,
                4,
                st.session_state.get("F_pre", 0),
                key="F_pre_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(F_pre)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### How clear or foggy does your mind feel right now?")
            C_pre = st.slider(
                "Confused / foggy",
                0,
                4,
                st.session_state.get("C_pre", 0),
                key="C_pre_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(C_pre)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### How strong is your spark or motivation to do small activities right now?")
            V_pre = st.slider(
                "Energetic / lively",
                0,
                4,
                st.session_state.get("V_pre", 0),
                key="V_pre_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_vigor_emoji(V_pre)}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div class='breathe-circle'></div>
        <div class='breathe-text'>
          Inhale as the circle grows, exhale as it shrinks. Try 3 gentle breaths before saving.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("💾 Save Before-Session Mood", use_container_width=True):
                scores = {
                    "T": T_pre,
                    "D": D_pre,
                    "A": A_pre,
                    "F": F_pre,
                    "C": C_pre,
                    "V": V_pre,
                }
                tmd = compute_tmd(scores)
                st.session_state["pre_scores"] = scores
                st.session_state["TMD_pre"] = tmd
                st.session_state.current_step = max(
                    st.session_state.current_step, 3
                )

                st.markdown(
                    """
                <div class='wellness-card' style='text-align:center;'>
                  <h3>✅ Before-session mood saved</h3>
                  <p style='margin-top:8px; color:#7e6ba6;'>
                    You're doing great. Take your time with the therapy activity.
                  </p>
                  <div class='breathe-circle'></div>
                  <p class='breathe-text'>
                    When you're ready, open the <b>🌿 Your Therapy Plan</b> tab to see your activity.
                  </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# ---------- TAB 3: YOUR THERAPY PLAN ----------
with tab_therapy:
    if "ppd_percent" not in st.session_state or "TMD_pre" not in st.session_state:
        st.markdown(
            """
        <div class='warning-message'>
          <h4>🌿 Therapy plan will appear after the first two steps</h4>
          <ol>
            <li>Complete the <b>📝 PPD Risk Screener</b>.</li>
            <li>Save your mood in the <b>🌅 Before Session</b> tab.</li>
          </ol>
          <p>After that, your personalised therapy plan will appear here.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.session_state.therapy_viewed = True
        st.session_state.current_step = max(st.session_state.current_step, 3)

        ppd_percent = st.session_state["ppd_percent"]
        risk = st.session_state["ppd_risk_level"]
        therapy = st.session_state["assigned_therapy"]
        detail = st.session_state["therapy_detail"]

        st.markdown(
            f"""
        <div class='wellness-card' style="
            background: radial-gradient(circle at top left, #e8d9ff 0%, #f7e4ff 35%, #fef9ff 100%);
            border-color: rgba(198,167,254,0.7);">
          <h2>🌿 Your Therapy Plan</h2>
          <p><b>PPD Risk:</b> {ppd_percent:.1f}% &nbsp;&nbsp;·&nbsp;&nbsp; <b>Level:</b> {risk}</p>
          <h3>Assigned: {therapy}</h3>
          <p style='max-width:800px;'>
            {detail}
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🎮 Start your session")

        # 🔗 ACTUAL CONNECTIONS TO UNITY GAMES HERE
        if therapy == "Clinician Escalation":
            st.warning(
                "Because your risk is high, this tool only shows supportive information. "
                "Please contact your clinician or a local mental health helpline for "
                "personalised care."
            )

        elif therapy == "Nature Walk Therapy":
            st.info(
                "You have been assigned **Nature Walk Therapy**. "
                "Click the button below to open the nature walk experience."
            )
            if st.button("🌲 Open Nature Walk Game", use_container_width=True):
                now = time.time()
                st.session_state["therapy_start_ts"] = now
                st.session_state["notify_ts"] = now + 15 * 60
                success, err = launch_forest_walk()
                if success:
                    st.success("Launching the Unity Forest Walk game… 🌲")
                else:
                    st.error(f"Could not launch Forest Walk: {err}")

        else:  # "User Choice: Nature Walk or Farming Game"
            st.info(
                "You can choose either **Nature Walk** or **Farming Game** therapy today."
            )
            choice = st.radio(
                "Which activity would you like to try today?",
                ["Nature Walk", "Farming Game"],
                index=0,
                horizontal=True,
            )
            colN, colF = st.columns(2)
            with colN:
                if st.button("🌲 Nature Walk Game", use_container_width=True):
                    now = time.time()
                    st.session_state["therapy_start_ts"] = now
                    st.session_state["notify_ts"] = now + 15 * 60
                    success, err = launch_forest_walk()
                    if success:
                        st.success("Launching the Unity Forest Walk game… 🌲")
                    else:
                        st.error(f"Could not launch Forest Walk: {err}")

            with colF:
                if st.button("🌾 Farming Game", use_container_width=True):
                    now = time.time()
                    st.session_state["therapy_start_ts"] = now
                    st.session_state["notify_ts"] = now + 15 * 60
                    success, err = launch_farming_game()
                    if success:
                        st.success("Launching the Unity Farming Game… 🌾")
                    else:
                        st.error(f"Could not launch Farming Game: {err}")

            st.caption(f"Selected today: **{choice}** (you can change this next time).")

        st.markdown(
            """
        <div class='support-message'>
          📌 After finishing your therapy activity, go to the <b>🌙 After Session</b> tab
          and record how you feel.
        </div>
        """,
            unsafe_allow_html=True,
        )

# ---------- TAB 4: AFTER SESSION ----------
with tab_after:
    if "TMD_pre" not in st.session_state or not st.session_state.get(
        "therapy_viewed", False
    ):
        st.markdown(
            """
        <div class='warning-message'>
          <h4>🌙 After-session check-in comes later</h4>
          <p>Please view your <b>🌿 Therapy Plan</b> and complete the activity first.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class='support-message'>
          🌙 Welcome back from your session. Let's gently check in on your mood now.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='wellness-card'>", unsafe_allow_html=True)
        st.subheader("How do you feel after your therapy session?")
        st.caption("Rate again from 0 (not at all) to 4 (extremely).")

        colL, colR = st.columns(2)
        with colL:
            st.markdown("#### How easy or difficult was it to stay relaxed while you were doing the activity?")
            T_post = st.slider(
                "Nervous / tense / on edge (after)",
                0,
                4,
                st.session_state.get("T_post", 0),
                key="T_post_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(T_post)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### How emotionally heavy or light did you feel after the session?")
            D_post = st.slider(
                "Sad / hopeless / low (after)",
                0,
                4,
                st.session_state.get("D_post", 0),
                key="D_post_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(D_post)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### How sensitive or reactive did you feel after the activity?")
            A_post = st.slider(
                "Angry / irritated (after)",
                0,
                4,
                st.session_state.get("A_post", 0),
                key="A_post_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(A_post)}</div>",
                unsafe_allow_html=True,
            )

        with colR:
            st.markdown("#### How drained did your body feel afterward?")
            F_post = st.slider(
                "Exhausted / worn out (after)",
                0,
                4,
                st.session_state.get("F_post", 0),
                key="F_post_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(F_post)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### How cloudy did your mind feel afterward?")
            C_post = st.slider(
                "Confused / foggy (after)",
                0,
                4,
                st.session_state.get("C_post", 0),
                key="C_post_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_mood_emoji(C_post)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("#### How much ‘get-up-and-go’ did you feel after finishing the session?")
            V_post = st.slider(
                "Energetic / lively (after)",
                0,
                4,
                st.session_state.get("V_post", 0),
                key="V_post_slider",
            )
            st.markdown(
                f"<div style='text-align:center;'>{get_vigor_emoji(V_post)}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("💾 Save After-Session Mood", use_container_width=True):
                scores = {
                    "T": T_post,
                    "D": D_post,
                    "A": A_post,
                    "F": F_post,
                    "C": C_post,
                    "V": V_post,
                }
                tmd = compute_tmd(scores)
                st.session_state["post_scores"] = scores
                st.session_state["TMD_post"] = tmd
                st.session_state.current_step = max(
                    st.session_state.current_step, 4
                )

                # 👉 compute therapy duration if we have a start time
                start_ts = st.session_state.get("therapy_start_ts")
                if start_ts is not None:
                    duration_sec = time.time() - start_ts
                    st.session_state["therapy_duration_min"] = duration_sec / 60.0
                else:
                    st.session_state["therapy_duration_min"] = None

                # clear reminder if still present
                if "notify_ts" in st.session_state:
                    del st.session_state["notify_ts"]

                #  BUILD ROW AND SAVE TO CSV "DATABASE"
                session_timestamp = datetime.now().isoformat(timespec="seconds")
                ppd_percent_val = st.session_state.get("ppd_percent", None)
                ppd_risk_level = st.session_state.get("ppd_risk_level", None)
                assigned_therapy = st.session_state.get("assigned_therapy", None)
                duration_min = st.session_state.get("therapy_duration_min", None)

                history_df = load_session_history()
                session_number = (history_df.shape[0] + 1) if not history_df.empty else 1

                # --- compute delta TMD (pre - post) ---
                delta_tmd = (
                    st.session_state.get("TMD_pre", None) - tmd
                    if st.session_state.get("TMD_pre", None) is not None
                    else None
                )

                # --- reward logic ---
                wellness_points = 0
                badge = None

                if delta_tmd is not None:
                    if delta_tmd >= 5:
                        wellness_points = 20
                        badge = "Big Mood Lift 🌈"
                    elif delta_tmd > 0:
                        wellness_points = 10
                        badge = "Gentle Improvement 💜"
                    else:
                        wellness_points = 5
                        badge = "You Showed Up 🌱"

                # --- build row for CSV history ---
                session_row = {
                    "Session_Number": session_number,
                    "Timestamp": session_timestamp,
                    "PPD_Risk_Percent": ppd_percent_val,
                    "PPD_Risk_Level": ppd_risk_level,
                    "Assigned_Therapy": assigned_therapy,
                    "TMD_Pre": st.session_state.get("TMD_pre", None),
                    "TMD_Post": tmd,
                    "Delta_TMD": delta_tmd,
                    "Therapy_Duration_Minutes": duration_min,
                    "Wellness_Points": wellness_points,
                    "Badge": badge,
                }

                append_session_to_history(session_row)

                # --- reward UI ---
                if delta_tmd is not None and delta_tmd > 0:
                    st.balloons()

                st.markdown(
                    f"""
                    <div class='wellness-card' style='text-align:center;'>
                      <h3>🎁 Today’s Reward</h3>
                      <p style='font-size:18px; margin-bottom:4px;'>
                        You earned <b>{wellness_points} Wellness Points</b>
                      </p>
                      <p style='font-size:16px;'>{badge}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.success(
                    "✅ After-session mood saved & stored. See the summary tab for results and history."
                )

# ---------- TAB 5: SUMMARY ----------
with tab_summary:
    if "TMD_pre" in st.session_state and "TMD_post" in st.session_state:
        TMD_pre = st.session_state["TMD_pre"]
        TMD_post = st.session_state["TMD_post"]
        _, _, delta = compute_delta_tmd(
            st.session_state["pre_scores"], st.session_state["post_scores"]
        )

        st.markdown("<div class='results-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center; margin-bottom:8px;'>🌈 Mood Change This Session</h2>",
            unsafe_allow_html=True,
        )

        col_top1, col_top2, col_top3 = st.columns(3)
        with col_top1:
            st.metric("Before-session TMD", f"{TMD_pre:.1f}")
        with col_top2:
            arrow = "⬇️" if delta > 0 else "⬆️" if delta < 0 else "➡️"
            label = "Improved" if delta > 0 else "Worsened" if delta < 0 else "No change"
            st.metric("ΔTMD (Pre - Post)", f"{delta:.1f}", label)
        with col_top3:
            st.metric("After-session TMD", f"{TMD_post:.1f}")

        # 👉 Show therapy duration if available
        duration_min = st.session_state.get("therapy_duration_min")
        if duration_min is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(
                "Therapy Session Duration",
                f"{duration_min:.1f} minutes"
            )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(
                f"""
            <div class='mood-card'>
              <div class='big-emoji'>🌅</div>
              <h3>Before Therapy</h3>
              <h2>{TMD_pre:.1f}</h2>
              <p>TMD score</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c2:
            arrow_icon = "⬇️" if delta > 0 else "⬆️" if delta < 0 else "➡️"
            st.markdown(
                f"""
            <div style='text-align:center; padding:40px 0 20px 0;'>
              <div style='font-size:64px;'>{arrow_icon}</div>
              <h2>{abs(delta):.1f} points</h2>
              <p>Change in mood disturbance (ΔTMD)</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
            <div class='mood-card'>
              <div class='big-emoji'>🌙</div>
              <h3>After Therapy</h3>
              <h2>{TMD_post:.1f}</h2>
              <p>TMD score</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # --- soft feedback text depending on delta ---
        if delta > 0:
            st.success(
                "Your TMD score decreased this session – that means overall distress "
                "was lower than before you started. 🌈"
            )
        elif delta < 0:
            st.warning(
                "Your TMD score was a bit higher after the session. "
                "That’s okay – some days are heavier, and that does not mean you failed. "
                "You still earned Wellness Points for showing up today 💜"
            )
        else:
            st.info(
                "Your TMD score stayed the same this session. Even maintaining is okay – "
                "consistency is what matters over time. 🌱"
            )

        # Simple bar chart for visual change (current session)
        chart_df = pd.DataFrame(
            {"TMD": [TMD_pre, TMD_post]}, index=["Before", "After"]
        )
        st.bar_chart(chart_df)

        # ===========================
        # 📈 LONGITUDINAL HISTORY + REWARDS
        # ===========================
        history_df = load_session_history()

        # Small debug so you can see if it's loading:
        st.caption(f"History CSV path: `{SESSIONS_CSV}` · Loaded rows: {history_df.shape[0]}")

        if not history_df.empty:
            # --- rewards overview ---
            if "Wellness_Points" in history_df.columns and "Badge" in history_df.columns:
                total_points = history_df["Wellness_Points"].fillna(0).sum()
                last_badge = history_df["Badge"].iloc[-1]
                total_sessions = history_df.shape[0]

                st.markdown("## 🎖️ Your Care Journey So Far")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Sessions Completed", int(total_sessions))
                with col_b:
                    st.metric("Total Wellness Points", int(total_points))
                with col_c:
                    st.metric("Latest Badge", last_badge)

                st.caption(
                    "These points & badges are gentle reminders that every check-in counts."
                )

                st.markdown("#### Badges Earned")
                st.dataframe(
                    history_df[["Session_Number", "Timestamp", "Badge"]],
                    use_container_width=True,
                )

            # --- mood history plots ---
            st.markdown("## 🧭 Mood Across All Sessions")

            history_plot = history_df.set_index("Session_Number")[
                ["TMD_Pre", "TMD_Post"]
            ]
            st.line_chart(history_plot)

            st.caption(
                "Trend of mood disturbance (TMD) before and after therapy across all sessions."
            )

            st.markdown("#### All Recorded Sessions")
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info(
                "No previous sessions stored yet. Complete one full session to start building history."
            )

        # Download this session summary only
        summary_row = {
            "PPD_Risk_Percent": st.session_state.get("ppd_percent", None),
            "PPD_Risk_Level": st.session_state.get("ppd_risk_level", None),
            "Assigned_Therapy": st.session_state.get("assigned_therapy", None),
            "TMD_Pre": TMD_pre,
            "TMD_Post": TMD_post,
            "Delta_TMD": delta,
            "Therapy_Duration_Minutes": duration_min,
        }
        summary_df = pd.DataFrame([summary_row])
        csv = summary_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Current Session Summary (CSV)",
            data=csv,
            file_name="ppd_session_summary_current.csv",
            mime="text/csv",
        )

        st.markdown("### 🔄 Start a new session")
        if st.button("Start New Session"):
            # NOTE: This does NOT delete the CSV history,
            # only clears in-memory session_state for a fresh run
            for key in list(st.session_state.keys()):
                if key not in ["_session", "_widget_id"]:
                    del st.session_state[key]
            st.rerun()
    else:
        st.markdown(
            """
        <div class='warning-message'>
          <h3>📊 Your session summary will appear here</h3>
          <p>Record both <b>before</b> and <b>after</b> session moods to see your TMD change.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown(
    """
<br><br>
<div style='text-align:center; padding: 18px; background: rgba(255, 255, 255, 0.9);
            border-radius: 15px; margin-top: 18px; border:1px solid #e6ddff;'>
  <p style='color:#7e6ba6; font-size:13px; margin-bottom:4px;'>
    🌸 This is a research prototype and does not replace professional diagnosis or treatment. <br>
    If you feel unsafe or in crisis, please contact your healthcare provider or a local crisis helpline immediately.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

