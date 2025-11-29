import streamlit as st
import time
import random
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mood Wellness Tracker",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS for Calming Design
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fef9f3 0%, #fef5ed 50%, #f8f3f0 100%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .wellness-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(198, 167, 254, 0.15);
        margin: 20px 0;
        border: 2px solid rgba(198, 167, 254, 0.3);
        backdrop-filter: blur(10px);
    }

    .mood-card {
        background: linear-gradient(135deg, #fff9f5 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        border: 2px solid #f0e6ff;
        transition: all 0.3s ease;
    }
    .mood-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(198, 167, 254, 0.2);
        border-color: #c6a7fe;
    }

    .support-message {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border-left: 4px solid #81c784;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #1b5e20 !important;
        font-weight: 500;
    }

    .warning-message {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ffb74d;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        font-size: 16px;
        line-height: 1.6;
        color: #e65100 !important;
        font-weight: 500;
    }

    .results-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 2px solid #c6a7fe;
    }

    .progress-step {
        display: inline-block;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #e0e0e0;
        color: white;
        text-align: center;
        line-height: 40px;
        margin: 0 10px;
        font-weight: bold;
    }
    .progress-step.active {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        box-shadow: 0 4px 15px rgba(198, 167, 254, 0.4);
    }
    .progress-step.completed {
        background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%);
    }

    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .breathe-circle {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(135deg, #c6a7fe 0%, #e1bee7 100%);
        margin: 20px auto;
        animation: breathe 4s ease-in-out infinite;
    }

    .big-emoji { font-size: 48px; margin: 10px; }

    .stButton > button {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(198, 167, 254, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(198, 167, 254, 0.4);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 15px 15px 0 0;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c6a7fe 0%, #b794f6 100%);
        color: white;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown h1,
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stCaption {
        color: #2d1b4e !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TMD Helper Functions (POMS)
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

def get_mood_emoji(v):  # 0–4
    return ["😌", "🙂", "😐", "😟", "😰"][min(max(v, 0), 4)]

def get_vigor_emoji(v):
    return ["😴", "😪", "🙂", "😊", "✨"][min(max(v, 0), 4)]

def therapy_from_risk(ppd_percent: float):
    """Same threshold rule from your LaTeX."""
    if ppd_percent >= 85:
        risk = "High Risk"
        therapy = "Clinician Escalation"
        detail = ("PPD risk is very high (≥85%). Immediate referral to a clinician "
                  "or mental health professional is recommended.")
    elif ppd_percent >= 65:
        risk = "Moderate Risk"
        therapy = "Nature Walk Therapy"
        detail = ("PPD risk is moderate (65–84%). Recommend guided Nature Walk VR "
                  "sessions with regular monitoring.")
    else:
        risk = "Low Risk"
        therapy = "User Choice: Nature Walk or Farming Game"
        detail = ("PPD risk is below 65%. The mother can choose between Nature Walk "
                  "or Farming Game therapy as a low-intensity wellness activity.")
    return risk, therapy, detail

# -----------------------------
# MDKR Loader + Preprocessing
# -----------------------------
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
    IMPORTANT: Make this match the function in your notebook.
    For now it's a simple template.
    """
    # Example yes/no mapping – ADAPT to your real columns
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
            df_new[col] = df_new[col].map({
                "Yes": 1,
                "No": 0,
                "Sometimes": 1,
                "Maybe": 1,
                "Not interested to say": 0
            })

    freq_map = {
        "Not at all": 0,
        "Several days": 1,
        "Two or more days a week": 2,
        "Nearly every day": 3,
        "Often": 2,
        "Not interested to say": 0
    }
    freq_cols = []  # fill this if your model uses any frequency columns
    for col in freq_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].replace(freq_map)

    age_map = {
        "<20": 0, "20-25": 1, "25-30": 2, "30-35": 3,
        "35-40": 4, "40-45": 5, "45-50": 6, "50+": 7
    }
    if "Age" in df_new.columns:
        df_new["Age"] = df_new["Age"].replace(age_map)

    # Ensure columns
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

    risk, therapy, _ = therapy_from_risk(ppd_percent)
    return ppd_percent, risk, therapy

# -----------------------------
# Session State Init
# -----------------------------
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

# -----------------------------
# Main App
# -----------------------------
def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #5e4b8b; font-size: 42px; margin-bottom: 10px;'>
                🌸 Mood Wellness Tracker 🌸
            </h1>
            <p style='color: #7e6ba6; font-size: 18px; font-weight: 300;'>
                A gentle space to track PPD risk, mood, and therapy response
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Progress indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pre_class = "completed" if "TMD_pre" in st.session_state else ("active" if st.session_state.current_step == 1 else "")
        post_class = "completed" if "TMD_post" in st.session_state else ("active" if st.session_state.current_step == 2 else "")
        results_class = "active" if "TMD_pre" in st.session_state and "TMD_post" in st.session_state else ""
        st.markdown(f"""
            <div style='text-align: center; margin: 30px 0;'>
                <span class='progress-step {pre_class}'>1</span>
                <span style='color: #c6a7fe;'>━━━</span>
                <span class='progress-step {post_class}'>2</span>
                <span style='color: #c6a7fe;'>━━━</span>
                <span class='progress-step {results_class}'>3</span>
                <br>
                <small style='color: #7e6ba6; font-size: 14px;'>
                    Before Therapy → After Therapy → Results & Recommendation
                </small>
            </div>
        """, unsafe_allow_html=True)

    # Tabs
    tab0, tab1, tab2, tab3 = st.tabs([
        "📝 PPD Risk Screener",
        "🌅 Before Therapy",
        "🌙 After Therapy",
        "📊 Results & Therapy"
    ])

    # ---------- TAB 0: PPD RISK SCREENER ----------
    with tab0:
        st.markdown(
            "<div class='support-message'>📝 Please answer a few questions so we can "
            "estimate your Postpartum Depression (PPD) risk and suggest a therapy path.</div>",
            unsafe_allow_html=True,
        )

        with st.form("ppd_form"):
            st.markdown("### Basic Questionnaire")

            age = st.selectbox("Age group", ["30-35", "35-40", "40-45", "45-50"], index=0)

            sad = st.selectbox(
                "Feeling sad or tearful in the last 2 weeks",
                ["No", "Yes", "Sometimes", "Not interested to say"]
            )
            irrit = st.selectbox(
                "Irritable towards baby & partner",
                ["No", "Yes", "Sometimes", "Not interested to say"]
            )
            lonely = st.selectbox(
                "Feeling lonely or unsupported",
                ["No", "Yes", "Sometimes", "Not interested to say"]
            )
            guilt = st.selectbox(
                "Feeling of guilt",
                ["No", "Yes", "Sometimes", "Not interested to say"]
            )
            bonding = st.selectbox(
                "Problems of bonding with baby",
                ["No", "Yes", "Sometimes", "Not interested to say"]
            )
            suicide = st.selectbox(
                "Any suicidal thoughts or attempts",
                ["No", "Yes", "Not interested to say"]
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
                ppd_percent, risk, therapy = predict_ppd_and_therapy_from_answers(user_answers)
                st.session_state["ppd_percent"] = ppd_percent
                st.session_state["ppd_risk_level"] = risk
                st.session_state["assigned_therapy"] = therapy

                st.success(f"PPD Risk: {ppd_percent:.2f}%  •  Risk Level: {risk}")
                st.info(f"Recommended Therapy Path: **{therapy}**")
            except Exception as e:
                st.error("Prediction failed – check that the preprocessing in app.py "
                         "matches your notebook.")
                st.exception(e)

    # ---------- TAB 1: PRE-THERAPY ----------
    with tab1:
        st.markdown("""
            <div class='support-message'>
            🌸 Before starting your therapy session, let's gently check how you feel right now.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='wellness-card'>", unsafe_allow_html=True)
        st.markdown("### How are you feeling right now?")
        st.caption("Rate each from 0 (not at all) to 4 (extremely).")

        colL, colR = st.columns(2)
        with colL:
            st.subheader("😰 Tension & Anxiety")
            T_pre = st.slider("Nervous / tense / on edge", 0, 4, st.session_state.get("T_pre", 0), key="T_pre")
            st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(T_pre)}</div>", unsafe_allow_html=True)

            st.subheader("😢 Sadness & Depression")
            D_pre = st.slider("Sad / hopeless / low", 0, 4, st.session_state.get("D_pre", 0), key="D_pre")
            st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(D_pre)}</div>", unsafe_allow_html=True)

            st.subheader("😤 Anger & Frustration")
            A_pre = st.slider("Angry / irritated", 0, 4, st.session_state.get("A_pre", 0), key="A_pre")
            st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(A_pre)}</div>", unsafe_allow_html=True)

        with colR:
            st.subheader("😴 Fatigue & Tiredness")
            F_pre = st.slider("Exhausted / worn out", 0, 4, st.session_state.get("F_pre", 0), key="F_pre")
            st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(F_pre)}</div>", unsafe_allow_html=True)

            st.subheader("😵 Confusion & Uncertainty")
            C_pre = st.slider("Confused / foggy", 0, 4, st.session_state.get("C_pre", 0), key="C_pre")
            st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(C_pre)}</div>", unsafe_allow_html=True)

            st.subheader("✨ Energy & Vitality")
            V_pre = st.slider("Energetic / lively", 0, 4, st.session_state.get("V_pre", 0), key="V_pre")
            st.markdown(f"<div style='text-align:center;'>{get_vigor_emoji(V_pre)}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            if st.button("💾 Save Pre-Therapy Mood", use_container_width=True):
                scores = {"T": T_pre, "D": D_pre, "A": A_pre,
                          "F": F_pre, "C": C_pre, "V": V_pre}
                tmd = compute_tmd(scores)
                st.session_state["pre_scores"] = scores
                st.session_state["TMD_pre"] = tmd
                st.session_state.current_step = 2

                st.markdown("""
                    <div class='wellness-card' style='text-align:center;'>
                    <h3>✅ Pre-therapy assessment saved</h3>
                    <div class='breathe-circle'></div>
                    <p>Take a deep breath... you're ready to begin your therapy session.</p>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(1.5)
                st.rerun()

    # ---------- TAB 2: POST-THERAPY ----------
    with tab2:
        if "TMD_pre" not in st.session_state:
            st.markdown("""
                <div class='warning-message'>
                <h4>⏸️ Pre-therapy mood not recorded</h4>
                <p>Please complete the "Before Therapy" check-in first.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='support-message'>
                🌙 Welcome back from therapy. Let's check how you feel now.
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='wellness-card'>", unsafe_allow_html=True)
            st.markdown("### How do you feel after your therapy session?")
            st.caption("Rate again from 0 to 4.")

            colL, colR = st.columns(2)
            with colL:
                st.subheader("😰 Tension & Anxiety")
                T_post = st.slider("Nervous / tense / on edge (after)", 0, 4,
                                   st.session_state.get("T_post", 0), key="T_post")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(T_post)}</div>", unsafe_allow_html=True)

                st.subheader("😢 Sadness & Depression")
                D_post = st.slider("Sad / hopeless / low (after)", 0, 4,
                                   st.session_state.get("D_post", 0), key="D_post")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(D_post)}</div>", unsafe_allow_html=True)

                st.subheader("😤 Anger & Frustration")
                A_post = st.slider("Angry / irritated (after)", 0, 4,
                                   st.session_state.get("A_post", 0), key="A_post")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(A_post)}</div>", unsafe_allow_html=True)

            with colR:
                st.subheader("😴 Fatigue & Tiredness")
                F_post = st.slider("Exhausted / worn out (after)", 0, 4,
                                   st.session_state.get("F_post", 0), key="F_post")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(F_post)}</div>", unsafe_allow_html=True)

                st.subheader("😵 Confusion & Uncertainty")
                C_post = st.slider("Confused / foggy (after)", 0, 4,
                                   st.session_state.get("C_post", 0), key="C_post")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(C_post)}</div>", unsafe_allow_html=True)

                st.subheader("✨ Energy & Vitality")
                V_post = st.slider("Energetic / lively (after)", 0, 4,
                                   st.session_state.get("V_post", 0), key="V_post")
                st.markdown(f"<div style='text-align:center;'>{get_vigor_emoji(V_post)}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1,1,1])
            with c2:
                if st.button("📊 Save & View Results", use_container_width=True):
                    scores = {"T": T_post, "D": D_post, "A": A_post,
                              "F": F_post, "C": C_post, "V": V_post}
                    tmd = compute_tmd(scores)
                    st.session_state["post_scores"] = scores
                    st.session_state["TMD_post"] = tmd
                    st.session_state.current_step = 3
                    st.success("✅ Post-therapy mood saved.")
                    time.sleep(1)
                    st.rerun()

    # ---------- TAB 3: RESULTS & THERAPY ----------
    with tab3:
        if "TMD_pre" in st.session_state and "TMD_post" in st.session_state:
            TMD_pre = st.session_state["TMD_pre"]
            TMD_post = st.session_state["TMD_post"]
            _, _, delta = compute_delta_tmd(
                st.session_state["pre_scores"],
                st.session_state["post_scores"]
            )

            st.markdown("<div class='results-card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;'>🌈 Mood Change This Session</h2>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                st.markdown(f"""
                    <div class='mood-card'>
                      <div class='big-emoji'>🌅</div>
                      <h3>Before Therapy</h3>
                      <h2>{TMD_pre:.1f}</h2>
                      <p>TMD score</p>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                arrow = "⬇️" if delta > 0 else "⬆️" if delta < 0 else "➡️"
                st.markdown(f"""
                    <div style='text-align:center; padding:50px 0;'>
                      <div style='font-size:64px;'>{arrow}</div>
                      <h2>{abs(delta):.1f} points</h2>
                      <p>Change in mood disturbance (ΔTMD)</p>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                    <div class='mood-card'>
                      <div class='big-emoji'>🌙</div>
                      <h3>After Therapy</h3>
                      <h2>{TMD_post:.1f}</h2>
                      <p>TMD score</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # PPD Risk & Therapy Recommendation
            st.markdown("### 🧠 Therapy Recommendation from PPD Risk (Track-2)")
            if "ppd_percent" in st.session_state:
                ppd_percent = st.session_state["ppd_percent"]
                risk_level = st.session_state["ppd_risk_level"]
                therapy = st.session_state["assigned_therapy"]
                _, _, detail = therapy_from_risk(ppd_percent)

                st.write(f"**PPD Risk:** {ppd_percent:.1f}%")
                st.write(f"**Risk Level:** {risk_level}")
                st.write(f"**Assigned Therapy Path:** {therapy}")
                st.info(detail)
            else:
                st.warning("PPD risk has not been estimated yet. "
                           "Please complete the **📝 PPD Risk Screener** tab.")

            # Reset button
            colL, colR = st.columns(2)
            with colL:
                if st.button("🔄 Start New Session", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        if key.startswith("TMD") or "scores" in key or key.endswith("_pre") or key.endswith("_post"):
                            del st.session_state[key]
                    st.session_state.current_step = 1
                    st.rerun()
        else:
            st.markdown("""
                <div class='warning-message'>
                <h3>📊 Your Results Will Appear Here</h3>
                <p>Complete both the pre-therapy and post-therapy mood check-ins to view your session summary.</p>
                </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <br><br>
        <div style='text-align:center; padding: 30px; background: rgba(255, 255, 255, 0.5);
                    border-radius: 15px; margin-top: 40px;'>
            <p style='color:#7e6ba6; font-size:14px;'>
                🌸 This tool is a research prototype. It does not replace professional diagnosis. 🌸<br>
                If you are in crisis, please contact your healthcare provider or a local helpline immediately.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
