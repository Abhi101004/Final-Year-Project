import streamlit as st
import time
import joblib
import pandas as pd
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="PPD Mood & Therapy Companion",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# GLOBAL STYLES – CALM, LEGIBLE
# ============================================
st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #fff7ef 0%, #fef1e8 40%, #f9f3ff 100%);
        font-family: "Helvetica Neue", Arial, sans-serif;
    }

    /* Hide default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Cards */
    .wellness-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 26px 30px;
        box-shadow: 0 8px 26px rgba(181, 148, 255, 0.18);
        margin: 18px 0;
        border: 1.5px solid rgba(181, 148, 255, 0.32);
        backdrop-filter: blur(10px);
    }

    .mood-card {
        background: linear-gradient(135deg, #fffaf4 0%, #ffffff 100%);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        margin: 10px 0;
        border: 1.5px solid #f0e6ff;
        transition: all 0.25s ease;
    }
    .mood-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 24px rgba(181, 148, 255, 0.25);
        border-color: #c3a6ff;
    }

    .support-message {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border-left: 4px solid #81c784;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 18px 0;
        font-size: 15px;
        line-height: 1.6;
        color: #1b5e20 !important;
        font-weight: 500;
    }

    .warning-message {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ffb74d;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 18px 0;
        font-size: 15px;
        line-height: 1.6;
        color: #e65100 !important;
        font-weight: 500;
    }

    .results-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-radius: 22px;
        padding: 30px;
        margin: 20px 0;
        border: 2px solid #c6a7fe;
    }

    .big-emoji {
        font-size: 48px;
        margin-bottom: 6px;
    }

    /* Progress bubbles */
    .progress-step {
        display: inline-block;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #e0e0e0;
        color: white;
        text-align: center;
        line-height: 40px;
        margin: 0 7px;
        font-weight: 700;
    }
    .progress-step.active {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        box-shadow: 0 4px 15px rgba(198, 167, 254, 0.4);
    }
    .progress-step.completed {
        background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 28px;
        font-size: 17px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(198, 167, 254, 0.3);
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(198, 167, 254, 0.4);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 16px 16px 0 0;
        padding: 10px 22px;
        font-size: 15px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        color: white;
    }

    /* Labels & text darker for legibility */
    label, .stMarkdown, .stMarkdown p, .stMarkdown h1,
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stCaption, .stSelectbox label {
        color: #28173f !important;
    }

    .question-label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #28173f !important;
        margin-top: 10px;
        margin-bottom: 4px;
    }

    .therapy-hero {
        border-radius: 22px;
        padding: 32px 30px;
        margin-top: 10px;
        background-size: cover;
        background-position: center;
        color: #ffffff;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        position: relative;
        overflow: hidden;
    }
    .therapy-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(24,16,48,0.82), rgba(80,45,110,0.78));
    }
    .therapy-content {
        position: relative;
        z-index: 2;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# TMD / POMS HELPERS
# ============================================
def compute_tmd(subscale_scores, K=100):
    """Total Mood Disturbance: (T + D + A + F + C) − V + K"""
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
    return ["😌", "🙂", "😐", "😟", "😰"][min(max(int(v), 0), 4)]


def get_vigor_emoji(v):
    return ["😴", "😪", "🙂", "😊", "✨"][min(max(int(v), 0), 4)]


# ============================================
# THERAPY THRESHOLD LOGIC (FROM PAPER)
# ============================================
def therapy_from_risk(ppd_percent: float):
    """
    Threshold-Based Therapy Assignment from your LaTeX:
      P_PPD ≥ 85%          → Clinician Escalation
      65% ≤ P_PPD < 85%    → Nature Walk Therapy
      P_PPD < 65%          → User Choice (Nature Walk / Farming Game)
    """
    if ppd_percent >= 85:
        risk = "High Risk"
        therapy = "Clinician Escalation"
        detail = (
            "PPD risk is very high (≥ 85%). An urgent referral to a clinician or "
            "mental health professional is recommended."
        )
    elif ppd_percent >= 65:
        risk = "Moderate Risk"
        therapy = "Nature Walk Therapy"
        detail = (
            "PPD risk is moderate (65–84%). Recommend guided **Nature Walk Therapy** "
            "sessions with close monitoring."
        )
    else:
        risk = "Low Risk"
        therapy = "User Choice: Nature Walk or Farming Game"
        detail = (
            "PPD risk is below 65%. The mother can choose between **Nature Walk** or "
            "**Farming Game** therapy as a low-intensity wellness activity."
        )
    return risk, therapy, detail


def therapy_visuals(therapy_name: str):
    """
    Returns title, subtitle, description and background 'image feel'
    for the therapy assignment screen.
    """
    if therapy_name == "Clinician Escalation":
        bg_style = (
            "background-image: url('https://images.pexels.com/photos/3845761/pexels-photo-3845761.jpeg');"
        )
        title = "Clinician Support Recommended"
        subtitle = "A specialist will guide you through a personalised care plan."
        body = (
            "Your current PPD risk suggests you would benefit from speaking with a "
            "mental-health professional. This app can still be used alongside "
            "your clinical care to gently track your mood."
        )
    elif therapy_name == "Nature Walk Therapy":
        bg_style = (
            "background-image: url('https://images.pexels.com/photos/210617/pexels-photo-210617.jpeg');"
        )
        title = "Nature Walk Therapy"
        subtitle = "Slow, mindful walks through calming green spaces."
        body = (
            "During Nature Walk sessions, you will gently explore a soothing forest or "
            "park environment. The goal is to relax your body, slow your breathing, "
            "and feel quietly connected to nature."
        )
    else:  # User Choice: Nature Walk or Farming Game
        bg_style = (
            "background-image: url('https://images.pexels.com/photos/158827/field-cereals-wheat-ears-"
            "cereals-158827.jpeg');"
        )
        title = "Choice: Nature Walk or Farming Game"
        subtitle = "Pick the activity that feels most comforting today."
        body = (
            "For low-risk PPD, you can choose a calm **Nature Walk** experience or a "
            "gentle **Farming Game** where you care for crops and animals. Both are "
            "designed to be light, kind and non-overwhelming."
        )
    return bg_style, title, subtitle, body


# ============================================
# MDKR MODEL OBJECTS
# ============================================
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
    """
    Make sure this mirrors your notebook preprocessing.
    Here we use the same template we used earlier.
    """
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
                    "Not interested to say": 0,
                }
            )

    freq_map = {
        "Not at all": 0,
        "Several days": 1,
        "Two or more days a week": 2,
        "Nearly every day": 3,
        "Often": 2,
        "Not interested to say": 0,
    }
    # If you have any frequency columns, add them here:
    freq_cols = []  # e.g. ["Sleep difficulty", "Loss of interest"]
    for col in freq_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].replace(freq_map)

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

    # Ensure all feature columns exist
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


# ============================================
# SESSION STATE
# ============================================
if "stage" not in st.session_state:
    # 0: screener, 1: therapy plan, 2: pre, 3: post, 4: summary
    st.session_state.stage = 0

# ============================================
# MAIN APP
# ============================================
def main():
    # ----- HEADER -----
    st.markdown(
        """
        <div style='text-align: center; padding: 12px 0 4px 0;'>
            <h1 style='color: #5e4b8b; font-size: 40px; margin-bottom: 4px;'>
                🌸 Postpartum Mood & Therapy Companion
            </h1>
            <p style='color: #7e6ba6; font-size: 17px; font-weight: 300;'>
                A gentle, low-pressure space to screen PPD risk, assign therapy, and track mood.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- PROGRESS BAR -----
    labels = [
        "PPD Screener",
        "Therapy Plan",
        "Before Session",
        "After Session",
        "Summary",
    ]
    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        bubbles = []
        for i in range(5):
            if st.session_state.stage > i:
                cls = "completed"
            elif st.session_state.stage == i:
                cls = "active"
            else:
                cls = ""
            bubbles.append(f"<span class='progress-step {cls}'>{i+1}</span>")
        st.markdown(
            f"""
            <div style='text-align:center; margin: 18px 0 6px 0;'>
                {''.join(bubbles)}
                <br>
                <small style='color:#7e6ba6; font-size:13px;'>
                    { " → ".join(labels) }
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----- TABS -----
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📝 PPD Risk Screener",
            "🌿 Your Therapy Plan",
            "🌅 Before Session",
            "🌙 After Session",
            "📊 Session Summary",
        ]
    )

    # ===========================================================
    # TAB 0 – PPD RISK SCREENER  (PRAMS subset → MDKR)
    # ===========================================================
    with tab0:
        st.markdown(
            "<div class='support-message'>"
            "Please answer a few short questions about how you’ve been feeling. "
            "This helps estimate your **PPD risk** and choose the most gentle therapy path."
            "</div>",
            unsafe_allow_html=True,
        )

        with st.form("ppd_form"):
            st.markdown("### Basic information")
            age = st.selectbox(
                "Age group",
                ["30-35", "35-40", "40-45", "45-50"],
                index=0,
                help="Age band is used as a risk factor in the MDKR model.",
            )

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("### Mood in the last 2 weeks")

            sad = st.selectbox(
                "Feeling sad or tearful",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
            )
            irrit = st.selectbox(
                "Irritable towards baby & partner",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
            )
            lonely = st.selectbox(
                "Feeling lonely or unsupported",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
            )
            guilt = st.selectbox(
                "Feeling of guilt",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
            )
            bonding = st.selectbox(
                "Problems bonding with baby",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
            )
            suicide = st.selectbox(
                "Any suicidal thoughts or attempts",
                ["No", "Yes", "Not interested to say"],
                index=0,
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
                ppd_percent, risk, therapy, detail = predict_ppd_and_therapy_from_answers(
                    user_answers
                )
                st.session_state.ppd_percent = ppd_percent
                st.session_state.ppd_risk_level = risk
                st.session_state.assigned_therapy = therapy
                st.session_state.therapy_detail = detail
                st.session_state.stage = max(st.session_state.stage, 1)

                st.success(f"PPD Risk: {ppd_percent:.2f}%  •  Risk Level: {risk}")
                st.info(f"Assigned Therapy Path: **{therapy}**")
            except Exception as e:
                st.error(
                    "Prediction failed. Please make sure `app.py` preprocessing "
                    "matches the notebook."
                )
                st.exception(e)

    # ===========================================================
    # TAB 1 – THERAPY PLAN SCREEN (calming hero section)
    # ===========================================================
    with tab1:
        if "assigned_therapy" not in st.session_state:
            st.markdown(
                "<div class='warning-message'>"
                "Please complete the **PPD Risk Screener** first so we can assign a therapy."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            therapy_name = st.session_state.assigned_therapy
            risk_level = st.session_state.ppd_risk_level
            ppd_percent = st.session_state.ppd_percent
            detail = st.session_state.therapy_detail

            bg_style, title, subtitle, body = therapy_visuals(therapy_name)

            st.markdown(
                f"""
                <div class="therapy-hero" style="{bg_style}">
                    <div class="therapy-overlay"></div>
                    <div class="therapy-content">
                        <h2 style="font-size:30px; margin-bottom:4px;">🌿 Your Therapy Plan</h2>
                        <p style="font-size:15px; margin:0 0 14px 0;">PPD Risk: {ppd_percent:.1f}% &nbsp; • &nbsp; Level: {risk_level}</p>
                        <h3 style="font-size:26px; margin-bottom:4px;">{title}</h3>
                        <p style="font-size:17px; margin-bottom:10px;">{subtitle}</p>
                        <p style="font-size:15px; line-height:1.7; max-width:680px;">{body}</p>
                        <p style="font-size:14px; line-height:1.6; max-width:680px; opacity:0.9; margin-top:6px;">{detail}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class='support-message'>
                    📌 When you feel ready, click “Start Session” below. 
                    You’ll first record your **mood before the therapy**, then complete your assigned activity, 
                    and finally record your **mood after the session**.
                </div>
                """,
                unsafe_allow_html=True,
            )

            center = st.columns([1, 1, 1])[1]
            with center:
                if st.button("🌸 I'm ready – Start this session", use_container_width=True):
                    st.session_state.stage = max(st.session_state.stage, 2)
                    time.sleep(0.5)
                    st.rerun()

    # ===========================================================
    # TAB 2 – PRE-THERAPY TMD
    # ===========================================================
    with tab2:
        if st.session_state.stage < 2:
            st.markdown(
                "<div class='warning-message'>"
                "Please view **Your Therapy Plan** and click *Start this session* before filling this part."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='support-message'>
                    🌅 Before you begin your therapy activity, let's gently check in with your mood.
                    Please rate how you feel **right now** on each scale from 0 (not at all) to 4 (extremely).
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div class='wellness-card'>", unsafe_allow_html=True)
            st.markdown("### How are you feeling right now?")
            st.caption("There are no right or wrong answers – this is just for you.")

            colL, colR = st.columns(2)
            with colL:
                st.subheader("😰 Tension & Anxiety")
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

                st.subheader("😢 Sadness & Depression")
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

                st.subheader("😤 Anger & Frustration")
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
                st.subheader("😴 Fatigue & Tiredness")
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

                st.subheader("😵 Confusion & Uncertainty")
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

                st.subheader("✨ Energy & Vitality")
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
                    st.session_state.pre_scores = scores
                    st.session_state.TMD_pre = tmd
                    st.session_state.stage = max(st.session_state.stage, 3)

                    st.markdown(
                        """
                        <div class='wellness-card' style='text-align:center;'>
                            <h3>✅ Saved</h3>
                            <p>Now you can complete your assigned therapy activity in your own time.</p>
                            <p style='font-size:14px;'>When you finish, come back to the <b>After Session</b> tab to record how you feel.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ===========================================================
    # TAB 3 – POST-THERAPY TMD
    # ===========================================================
    with tab3:
        if "TMD_pre" not in st.session_state:
            st.markdown(
                "<div class='warning-message'>"
                "Please complete the **Before Session** mood check-in before filling this part."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='support-message'>
                    🌙 Welcome back. After finishing your therapy activity, let's check your mood once more.
                    Please rate how you feel **now**, using the same scales.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div class='wellness-card'>", unsafe_allow_html=True)
            st.markdown("### How do you feel after your therapy session?")
            st.caption("Use the same 0–4 scale as before.")

            colL, colR = st.columns(2)
            with colL:
                st.subheader("😰 Tension & Anxiety")
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

                st.subheader("😢 Sadness & Depression")
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

                st.subheader("😤 Anger & Frustration")
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
                st.subheader("😴 Fatigue & Tiredness")
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

                st.subheader("😵 Confusion & Uncertainty")
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

                st.subheader("✨ Energy & Vitality")
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
                if st.button("📊 Save & View Summary", use_container_width=True):
                    scores = {
                        "T": T_post,
                        "D": D_post,
                        "A": A_post,
                        "F": F_post,
                        "C": C_post,
                        "V": V_post,
                    }
                    tmd = compute_tmd(scores)
                    st.session_state.post_scores = scores
                    st.session_state.TMD_post = tmd
                    st.session_state.stage = max(st.session_state.stage, 4)
                    st.success("✅ Post-session mood saved. View your summary in the next tab.")

    # ===========================================================
    # TAB 4 – SESSION SUMMARY (ΔTMD only – NO big therapy block)
    # ===========================================================
    with tab4:
        if "TMD_pre" not in st.session_state or "TMD_post" not in st.session_state:
            st.markdown(
                "<div class='warning-message'>"
                "Once you complete both **Before** and **After** session check-ins, "
                "your summary will appear here."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            TMD_pre = st.session_state.TMD_pre
            TMD_post = st.session_state.TMD_post
            _, _, delta = compute_delta_tmd(
                st.session_state.pre_scores, st.session_state.post_scores
            )

            st.markdown("<div class='results-card'>", unsafe_allow_html=True)
            st.markdown(
                "<h2 style='text-align:center; color:#4a2f73;'>🌈 Mood Change This Session</h2>",
                unsafe_allow_html=True,
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
                arrow = "⬇️" if delta > 0 else "⬆️" if delta < 0 else "➡️"
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:40px 0;'>
                        <div style='font-size:64px;'>{arrow}</div>
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

            # Small note about therapy path (no big recommendation section)
            if "assigned_therapy" in st.session_state:
                therapy = st.session_state.assigned_therapy
                st.markdown(
                    f"""
                    <div class='support-message'>
                        🧠 <b>Therapy this session:</b> {therapy}.<br>
                        This summary shows how your mood shifted during that activity.
                        You and your clinician can use ΔTMD across sessions to judge what helps you most.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            colL, colR = st.columns(2)
            with colL:
                if st.button("🔄 Start a New Session", use_container_width=True):
                    # Clear session (model cache is separate)
                    st.session_state.clear()
                    st.rerun()

    # ----- FOOTER -----
    st.markdown(
        """
        <br><br>
        <div style='text-align:center; padding: 26px; background: rgba(255, 255, 255, 0.6);
                    border-radius: 16px; margin-top: 30px;'>
            <p style='color:#7e6ba6; font-size:13px;'>
                🌸 This is a research prototype inspired by your FYP architecture (MDKR + POMS TMD).<br>
                It is not a replacement for professional diagnosis. If you ever feel unsafe,
                please contact your healthcare provider or a local helpline immediately.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
