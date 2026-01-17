import streamlit as st
import time
import joblib
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import sqlite3
import hashlib
import secrets
from datetime import datetime


# =========================================
# ✅ PATHS
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ppd_app.db")

# =========================================
# 🔗 CONFIG: UNITY GAME APPS (LOCAL .app FILES)
# =========================================
FOREST_WALK_APP = ""  # put your Forest Walk .app path here
FARMING_GAME_APP = "/Users/abhilavanya/Downloads/Pentafarm-main 2/FARM GAME.app"


# =========================================
# ✅ UNITY LAUNCHERS
# =========================================
def launch_forest_walk():
    """Launch the Forest Walk Unity app."""
    try:
        if sys.platform != "darwin":  # only macOS
            return False, "Forest Walk launch is only configured for macOS."

        if not FOREST_WALK_APP or not os.path.exists(FOREST_WALK_APP):
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

        if not FARMING_GAME_APP or not os.path.exists(FARMING_GAME_APP):
            return False, f"Farming Game app not found at: {FARMING_GAME_APP}"

        subprocess.Popen(["open", FARMING_GAME_APP])
        return True, None
    except Exception as e:
        return False, str(e)


# =========================================
# 🔔 macOS Notification Helper
# =========================================
def mac_notification(title: str, message: str):
    """Show a native macOS notification (Notification Center)."""
    if sys.platform == "darwin":
        os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')


# =========================================
# ✅ AUTH + DB (SQLite)
# =========================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def ensure_column(conn, table_name: str, column_name: str, col_def_sql: str):
    """
    If column_name is missing in table_name, adds it using: ALTER TABLE table ADD COLUMN col_def_sql
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [r[1] for r in cur.fetchall()]
    if column_name not in cols:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_def_sql}")
        conn.commit()


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('mother','admin')),
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        timestamp TEXT NOT NULL,

        session_number INTEGER,
        ppd_risk_percent REAL,
        ppd_risk_level TEXT,

        assigned_therapy TEXT,
        selected_therapy_actual TEXT,

        tmd_pre REAL,
        tmd_post REAL,
        delta_tmd REAL,

        therapy_duration_minutes REAL,
        wellness_points INTEGER,
        badge TEXT,

        FOREIGN KEY(username) REFERENCES users(username)
    )
    """)
    conn.commit()

    # If you already created tables earlier, ensure new columns exist
    ensure_column(conn, "sessions", "session_number", "session_number INTEGER")
    ensure_column(conn, "sessions", "selected_therapy_actual", "selected_therapy_actual TEXT")

    conn.close()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000)
        return dk.hex() == hashed
    except Exception:
        return False


def create_user(username: str, password: str, role: str = "mother") -> tuple[bool, str]:
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Username and password are required."

    if role not in ("mother", "admin"):
        return False, "Invalid role."

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE username=?", (username,))
    if cur.fetchone():
        conn.close()
        return False, "Username already exists."

    cur.execute(
        "INSERT INTO users(username, password_hash, role, created_at) VALUES(?,?,?,?)",
        (username, hash_password(password), role, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return True, "User created successfully."


def authenticate_user(username: str, password: str):
    username = (username or "").strip().lower()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, password_hash, role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    u, pw_hash, role = row
    if verify_password(password, pw_hash):
        return {"username": u, "role": role}
    return None


def any_admin_exists() -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    count = cur.fetchone()[0]
    conn.close()
    return count > 0


def load_all_users() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT username, role, created_at FROM users ORDER BY created_at DESC", conn)
    conn.close()
    return df


def load_mothers_list() -> list[str]:
    conn = get_conn()
    df = pd.read_sql_query("SELECT username FROM users WHERE role='mother' ORDER BY username ASC", conn)
    conn.close()
    return df["username"].tolist() if not df.empty else []


def load_user_sessions(username: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM sessions WHERE username=? ORDER BY id ASC",
        conn,
        params=(username,),
    )
    conn.close()
    return df


def load_sessions_for_admin(selected_username: str | None = None) -> pd.DataFrame:
    conn = get_conn()
    if selected_username and selected_username != "__ALL__":
        df = pd.read_sql_query(
            "SELECT * FROM sessions WHERE username=? ORDER BY id ASC",
            conn,
            params=(selected_username,),
        )
    else:
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id ASC", conn)
    conn.close()
    return df


def get_next_session_number(username: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sessions WHERE username=?", (username,))
    count = cur.fetchone()[0]
    conn.close()
    return int(count) + 1


def save_session(username: str, session_row: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions(
            username, timestamp, session_number,
            ppd_risk_percent, ppd_risk_level,
            assigned_therapy, selected_therapy_actual,
            tmd_pre, tmd_post, delta_tmd,
            therapy_duration_minutes, wellness_points, badge
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            username,
            session_row.get("Timestamp"),
            session_row.get("Session_Number"),
            session_row.get("PPD_Risk_Percent"),
            session_row.get("PPD_Risk_Level"),
            session_row.get("Assigned_Therapy"),
            session_row.get("Selected_Therapy_Actual"),
            session_row.get("TMD_Pre"),
            session_row.get("TMD_Post"),
            session_row.get("Delta_TMD"),
            session_row.get("Therapy_Duration_Minutes"),
            session_row.get("Wellness_Points"),
            session_row.get("Badge"),
        ),
    )
    conn.commit()
    conn.close()


# Initialize DB at startup
init_db()


# =========================================
# ✅ LOGIN UI
# =========================================
def admin_setup_screen():
    st.markdown("## 🛠️ First-time Setup (Create Admin / Doctor Account)")
    st.info("No admin account exists yet. Create your Doctor/Admin login now.")

    a_user = st.text_input("Admin Username", value="doctor")
    a_pass = st.text_input("Admin Password", type="password")
    a_pass2 = st.text_input("Confirm Password", type="password")

    if st.button("Create Admin Account"):
        if not a_pass or a_pass != a_pass2:
            st.error("Passwords do not match.")
            return
        ok, msg = create_user(a_user, a_pass, role="admin")
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


def login_screen():
    st.markdown("## 🔐 Welcome")

    tab_login, tab_signup = st.tabs(["Login", "Mother Sign Up"])

    with tab_login:
        st.markdown("### Login to your account")
        st.caption("Doctors and Mothers both login here")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", key="login_btn"):
            user = authenticate_user(username, password)
            if user:
                st.session_state["auth_user"] = user
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid username/password.")

    with tab_signup:
        st.markdown("### New Mother? Sign Up Here")
        st.caption("Sign up is only for Mothers. Doctor accounts are created during first-time setup.")
        new_user = st.text_input("Create Username", key="signup_user")
        new_pass = st.text_input("Create Password", type="password", key="signup_pass")
        new_pass2 = st.text_input("Confirm Password", type="password", key="signup_pass2")

        if st.button("Create Mother Account", key="signup_btn"):
            if new_pass != new_pass2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(new_user, new_pass, role="mother")
                if ok:
                    st.success("✅ Account created successfully! Now login from the Login tab.")
                else:
                    st.error(msg)



# =========================================
# Custom CSS
# =========================================
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

    body, .stApp,
    label, p, span, li,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li {
        color: var(--primary-text) !important;
    }

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

    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #ffffff !important;
        color: var(--primary-text) !important;
        border-radius: 10px !important;
        border: 1px solid #d5c5ff !important;
    }

    [data-baseweb="menu"] * ,
    [role="listbox"] li,
    [role="listbox"] li * {
        color: var(--primary-text) !important;
    }

    [role="listbox"] li[aria-selected="true"] {
        background-color: #f3e5f5 !important;
    }

    [data-baseweb="slider"] > div {
        color: var(--primary-text) !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        box-shadow: 0 0 0 2px var(--accent) !important;
    }
    [data-baseweb="slider"] .rc-slider-track {
        background-color: var(--accent) !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricLabel"] {
        color: var(--primary-text) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================
# Helper functions
# =========================================
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


# =========================================
# Session state init (flow)
# =========================================
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "therapy_viewed" not in st.session_state:
    st.session_state.therapy_viewed = False


# =========================================
# 🔔 Check reminder timer (if any)
# =========================================
if "notify_ts" in st.session_state:
    if time.time() >= st.session_state["notify_ts"]:
        mac_notification(
            "PPD Therapy Reminder",
            "Please record your after-session mood in the PPD Companion app.",
        )
        st.toast("⏰ Time to record your after-session mood.", icon="🔔")
        del st.session_state["notify_ts"]


# =========================================
# Check auth status
# =========================================
AUTH_USER = st.session_state.get("auth_user")
if not AUTH_USER:
    # Not logged in - show login/signup screen
    if not any_admin_exists():
        admin_setup_screen()
    else:
        login_screen()
    st.stop()

AUTH_USERNAME = AUTH_USER["username"]
AUTH_ROLE = AUTH_USER["role"]

# Logout button in sidebar
with st.sidebar:
    st.markdown(f"### Logged in as:")
    st.markdown(f"**{AUTH_USERNAME}** ({AUTH_ROLE})")
    if st.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# =========================================
# DOCTOR DASHBOARD FUNCTION
# =========================================
def doctor_admin_dashboard():
    """Admin/Doctor dashboard - view mother records by searching username"""
    st.markdown("## 🩺 Doctor Dashboard")
    st.markdown("---")
    
    # Search for mother by username
    st.markdown("### 🔍 Search Mother Records")
    
    mothers = load_mothers_list()
    if not mothers:
        st.info("No mother accounts exist yet. Mothers can sign up from the login page.")
    else:
        search_username = st.text_input("Enter Mother's Username", key="search_mother")
        
        if search_username:
            search_username = search_username.strip().lower()
            
            # Check if username exists
            if search_username not in mothers:
                st.warning(f"No mother account found with username: {search_username}")
            else:
                st.success(f"Viewing records for: **{search_username}**")
                
                # Load sessions for this mother
                df = load_user_sessions(search_username)
                
                if df.empty:
                    st.info(f"No therapy sessions recorded yet for {search_username}.")
                else:
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Sessions", len(df))
                    with col2:
                        if "wellness_points" in df.columns:
                            total_points = int(df["wellness_points"].fillna(0).sum())
                            st.metric("Total Wellness Points", total_points)
                    with col3:
                        if "ppd_risk_percent" in df.columns and not df["ppd_risk_percent"].isna().all():
                            latest_risk = df["ppd_risk_percent"].iloc[-1]
                            st.metric("Latest PPD Risk", f"{latest_risk:.1f}%")
                    
                    # Session history table
                    st.markdown("#### Session History")
                    st.dataframe(df, use_container_width=True)
                    
                    # Mood trend chart
                    st.markdown("#### Mood Trend (TMD)")
                    plot_df = df.copy()
                    if "session_number" in plot_df.columns:
                        plot_df["session_number"] = plot_df["session_number"].fillna(range(1, len(plot_df) + 1))
                        plot_df = plot_df.set_index("session_number")
                    else:
                        plot_df = plot_df.reset_index(drop=True)
                        plot_df.index = plot_df.index + 1
                    
                    if "tmd_pre" in plot_df.columns and "tmd_post" in plot_df.columns:
                        st.line_chart(plot_df[["tmd_pre", "tmd_post"]])
                    
                    # Download option
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"⬇️ Download {search_username}'s Records (CSV)",
                        data=csv,
                        file_name=f"{search_username}_sessions.csv",
                        mime="text/csv",
                    )
    
    st.markdown("---")
    
    # View all mothers list
    st.markdown("### 👥 All Registered Mothers")
    if mothers:
        for mother in mothers:
            st.markdown(f"- {mother}")
    else:
        st.info("No mothers registered yet.")


# =========================================
# Header + progress
# =========================================
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

# =========================================
# Role-based Interface
# =========================================
if AUTH_ROLE == "admin":
    # DOCTOR/ADMIN VIEW - Show only admin dashboard
    st.markdown("---")
    doctor_admin_dashboard()
    
else:
    # MOTHER VIEW - Show therapy workflow with progress
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

    # =========================================
    # Tabs for Mothers
    # =========================================
    tabs_list = [
        "📝 PPD Risk Screener",
        "🌅 Before Session",
        "🌿 Your Therapy Plan",
        "🌙 After Session",
        "📊 Session Summary",
    ]

    tabs = st.tabs(tabs_list)

    tab_screener = tabs[0]
    tab_before = tabs[1]
    tab_therapy = tabs[2]
    tab_after = tabs[3]
    tab_summary = tabs[4]


    # =========================================
    # ---------- TAB 1: PPD RISK SCREENER ----------
    # =========================================
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
                key="age_radio",
            )

            st.subheader("Mood in the last 2 weeks")

            sad = st.radio(
                "Feeling sad or tearful",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
                horizontal=True,
                key="sad_radio",
            )

            irrit = st.radio(
                "Irritable towards baby & partner",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
                horizontal=True,
                key="irrit_radio",
            )

            lonely = st.radio(
                "Feeling lonely or unsupported",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
                horizontal=True,
                key="lonely_radio",
            )

            guilt = st.radio(
                "Feeling of guilt",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
                horizontal=True,
                key="guilt_radio",
            )

            bonding = st.radio(
                "Problems of bonding with baby",
                ["No", "Yes", "Sometimes", "Not interested to say"],
                index=0,
                horizontal=True,
                key="bonding_radio",
            )

            suicide = st.radio(
                "Any suicidal thoughts or attempts",
                ["No", "Yes", "Not interested to say"],
                index=0,
                horizontal=True,
                key="suicide_radio",
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
                    ppd_percent, risk, therapy, detail = predict_ppd_and_therapy_from_answers(user_answers)

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
                st.caption("Next: go to **🌅 Before Session** tab and record your mood before starting the activity.")

                with st.expander("What does this estimate mean?"):
                    st.write(detail)

            except Exception as e:
                st.error("Prediction failed – ensure preprocessing in `app.py` matches your notebook.")
                st.exception(e)


    # =========================================
    # ---------- TAB 2: BEFORE SESSION ----------
    # =========================================
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
                T_pre = st.slider("Nervous / tense / on edge", 0, 4, st.session_state.get("T_pre", 0), key="T_pre_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(T_pre)}</div>", unsafe_allow_html=True)

                st.markdown("#### How heavy or weighed-down do your emotions feel at this moment?")
                D_pre = st.slider("Sad / hopeless / low", 0, 4, st.session_state.get("D_pre", 0), key="D_pre_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(D_pre)}</div>", unsafe_allow_html=True)

                st.markdown("#### When small things happen around you, how quickly do they get on your nerves?")
                A_pre = st.slider("Angry / irritated", 0, 4, st.session_state.get("A_pre", 0), key="A_pre_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(A_pre)}</div>", unsafe_allow_html=True)

            with colR:
                st.markdown("#### How rested or drained do you feel right now?")
                F_pre = st.slider("Exhausted / worn out", 0, 4, st.session_state.get("F_pre", 0), key="F_pre_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(F_pre)}</div>", unsafe_allow_html=True)

                st.markdown("#### How clear or foggy does your mind feel right now?")
                C_pre = st.slider("Confused / foggy", 0, 4, st.session_state.get("C_pre", 0), key="C_pre_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(C_pre)}</div>", unsafe_allow_html=True)

                st.markdown("#### How strong is your spark or motivation to do small activities right now?")
                V_pre = st.slider("Energetic / lively", 0, 4, st.session_state.get("V_pre", 0), key="V_pre_slider")
                st.markdown(f"<div style='text-align:center;'>{get_vigor_emoji(V_pre)}</div>", unsafe_allow_html=True)

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
                    scores = {"T": T_pre, "D": D_pre, "A": A_pre, "F": F_pre, "C": C_pre, "V": V_pre}
                    tmd = compute_tmd(scores)
                    st.session_state["pre_scores"] = scores
                    st.session_state["TMD_pre"] = tmd
                    st.session_state.current_step = max(st.session_state.current_step, 3)

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


    # =========================================
    # ---------- TAB 3: YOUR THERAPY PLAN ----------
    # =========================================
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

            if therapy == "Clinician Escalation":
                st.warning(
                    "Because your risk is high, this tool only shows supportive information. "
                    "Please contact your clinician or a local mental health helpline for personalised care."
                )
                st.session_state["selected_therapy_actual"] = "Clinician Escalation"

            elif therapy == "Nature Walk Therapy":
                st.info("You have been assigned **Nature Walk Therapy**. Click the button below to open it.")
                if st.button("🌲 Open Nature Walk Game", use_container_width=True):
                    st.session_state["selected_therapy_actual"] = "Nature Walk"
                    now = time.time()
                    st.session_state["therapy_start_ts"] = now
                    st.session_state["notify_ts"] = now + 15 * 60
                    success, err = launch_forest_walk()
                    if success:
                        st.success("Launching the Unity Forest Walk game… 🌲")
                    else:
                        st.error(f"Could not launch Forest Walk: {err}")

            else:  # User choice
                st.info("You can choose either **Nature Walk** or **Farming Game** therapy today.")
                choice = st.radio(
                    "Which activity would you like to try today?",
                    ["Nature Walk", "Farming Game"],
                    index=0,
                    horizontal=True,
                    key="choice_radio",
                )
                st.session_state["selected_therapy_actual"] = choice

                colN, colF = st.columns(2)
                with colN:
                    if st.button("🌲 Nature Walk Game", use_container_width=True):
                        st.session_state["selected_therapy_actual"] = "Nature Walk"
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
                        st.session_state["selected_therapy_actual"] = "Farming Game"
                        now = time.time()
                        st.session_state["therapy_start_ts"] = now
                        st.session_state["notify_ts"] = now + 15 * 60
                        success, err = launch_farming_game()
                        if success:
                            st.success("Launching the Unity Farming Game… 🌾")
                        else:
                            st.error(f"Could not launch Farming Game: {err}")

                st.caption(f"Selected today: **{st.session_state.get('selected_therapy_actual','Nature Walk')}**")

            st.markdown(
                """
            <div class='support-message'>
              📌 After finishing your therapy activity, go to the <b>🌙 After Session</b> tab
              and record how you feel.
            </div>
            """,
                unsafe_allow_html=True,
            )


    # =========================================
    # ---------- TAB 4: AFTER SESSION ----------
    # =========================================
    with tab_after:
        if "TMD_pre" not in st.session_state or not st.session_state.get("therapy_viewed", False):
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
                T_post = st.slider("Nervous / tense / on edge (after)", 0, 4, st.session_state.get("T_post", 0), key="T_post_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(T_post)}</div>", unsafe_allow_html=True)

                st.markdown("#### How emotionally heavy or light did you feel after the session?")
                D_post = st.slider("Sad / hopeless / low (after)", 0, 4, st.session_state.get("D_post", 0), key="D_post_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(D_post)}</div>", unsafe_allow_html=True)

                st.markdown("#### How sensitive or reactive did you feel after the activity?")
                A_post = st.slider("Angry / irritated (after)", 0, 4, st.session_state.get("A_post", 0), key="A_post_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(A_post)}</div>", unsafe_allow_html=True)

            with colR:
                st.markdown("#### How drained did your body feel afterward?")
                F_post = st.slider("Exhausted / worn out (after)", 0, 4, st.session_state.get("F_post", 0), key="F_post_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(F_post)}</div>", unsafe_allow_html=True)

                st.markdown("#### How cloudy did your mind feel afterward?")
                C_post = st.slider("Confused / foggy (after)", 0, 4, st.session_state.get("C_post", 0), key="C_post_slider")
                st.markdown(f"<div style='text-align:center;'>{get_mood_emoji(C_post)}</div>", unsafe_allow_html=True)

                st.markdown("#### How much 'get-up-and-go' did you feel after finishing the session?")
                V_post = st.slider("Energetic / lively (after)", 0, 4, st.session_state.get("V_post", 0), key="V_post_slider")
                st.markdown(f"<div style='text-align:center;'>{get_vigor_emoji(V_post)}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                if st.button("💾 Save After-Session Mood", use_container_width=True):
                    scores = {"T": T_post, "D": D_post, "A": A_post, "F": F_post, "C": C_post, "V": V_post}
                    tmd = compute_tmd(scores)
                    st.session_state["post_scores"] = scores
                    st.session_state["TMD_post"] = tmd
                    st.session_state.current_step = max(st.session_state.current_step, 4)

                    # Duration
                    start_ts = st.session_state.get("therapy_start_ts")
                    if start_ts is not None:
                        duration_sec = time.time() - start_ts
                        st.session_state["therapy_duration_min"] = duration_sec / 60.0
                    else:
                        st.session_state["therapy_duration_min"] = None

                    # Clear reminder
                    if "notify_ts" in st.session_state:
                        del st.session_state["notify_ts"]

                    # Build session row for DB
                    session_timestamp = datetime.now().isoformat(timespec="seconds")
                    ppd_percent_val = st.session_state.get("ppd_percent", None)
                    ppd_risk_level = st.session_state.get("ppd_risk_level", None)
                    assigned_therapy = st.session_state.get("assigned_therapy", None)
                    selected_actual = st.session_state.get("selected_therapy_actual", assigned_therapy)
                    duration_min = st.session_state.get("therapy_duration_min", None)

                    delta_tmd = (
                        st.session_state.get("TMD_pre", None) - tmd
                        if st.session_state.get("TMD_pre", None) is not None
                        else None
                    )

                    # Reward logic
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

                    # Session number per mother
                    session_number = get_next_session_number(AUTH_USERNAME)

                    session_row = {
                        "Session_Number": session_number,
                        "Timestamp": session_timestamp,
                        "PPD_Risk_Percent": ppd_percent_val,
                        "PPD_Risk_Level": ppd_risk_level,
                        "Assigned_Therapy": assigned_therapy,
                        "Selected_Therapy_Actual": selected_actual,
                        "TMD_Pre": st.session_state.get("TMD_pre", None),
                        "TMD_Post": tmd,
                        "Delta_TMD": delta_tmd,
                        "Therapy_Duration_Minutes": duration_min,
                        "Wellness_Points": wellness_points,
                        "Badge": badge,
                    }

                    save_session(AUTH_USERNAME, session_row)

                    if delta_tmd is not None and delta_tmd > 0:
                        st.balloons()

                    st.markdown(
                        f"""
                        <div class='wellness-card' style='text-align:center;'>
                          <h3>🎁 Today's Reward</h3>
                          <p style='font-size:18px; margin-bottom:4px;'>
                            You earned <b>{wellness_points} Wellness Points</b>
                          </p>
                          <p style='font-size:16px;'>{badge}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.success("✅ After-session mood saved & stored. See the summary tab for results and history.")


    # =========================================
    # ---------- TAB 5: SUMMARY ----------
    # =========================================
    with tab_summary:
        # Load history for the mother
        history_df = load_user_sessions(AUTH_USERNAME)

        # Current session summary (only if completed this run)
        if "TMD_pre" in st.session_state and "TMD_post" in st.session_state:
            TMD_pre = st.session_state["TMD_pre"]
            TMD_post = st.session_state["TMD_post"]
            _, _, delta = compute_delta_tmd(st.session_state["pre_scores"], st.session_state["post_scores"])

            st.markdown("<div class='results-card'>", unsafe_allow_html=True)
            st.markdown(
                "<h2 style='text-align:center; margin-bottom:8px;'>🌈 Mood Change This Session</h2>",
                unsafe_allow_html=True,
            )

            col_top1, col_top2, col_top3 = st.columns(3)
            with col_top1:
                st.metric("Before-session TMD", f"{TMD_pre:.1f}")
            with col_top2:
                label = "Improved" if delta > 0 else "Worsened" if delta < 0 else "No change"
                st.metric("ΔTMD (Pre - Post)", f"{delta:.1f}", label)
            with col_top3:
                st.metric("After-session TMD", f"{TMD_post:.1f}")

            duration_min = st.session_state.get("therapy_duration_min")
            if duration_min is not None:
                st.metric("Therapy Session Duration", f"{duration_min:.1f} minutes")

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

            if delta > 0:
                st.success("Your TMD score decreased this session – that means overall distress was lower than before. 🌈")
            elif delta < 0:
                st.warning(
                    "Your TMD score was higher after the session. That's okay – some days are heavier. "
                    "You still earned points for showing up 💜"
                )
            else:
                st.info("Your TMD score stayed the same. Consistency matters over time. 🌱")

            chart_df = pd.DataFrame({"TMD": [TMD_pre, TMD_post]}, index=["Before", "After"])
            st.bar_chart(chart_df)

            # Download current session summary
            summary_row = {
                "Username": AUTH_USERNAME,
                "PPD_Risk_Percent": st.session_state.get("ppd_percent", None),
                "PPD_Risk_Level": st.session_state.get("ppd_risk_level", None),
                "Assigned_Therapy": st.session_state.get("assigned_therapy", None),
                "Selected_Therapy_Actual": st.session_state.get("selected_therapy_actual", None),
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
                for key in list(st.session_state.keys()):
                    if key not in ["auth_user"]:
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

        # History for mother
        st.markdown("## 🧭 History")
        if history_df is None or history_df.empty:
            st.info("No sessions stored yet.")
        else:
            # normalize columns
            df = history_df.copy()

            # For plot: use session_number if available; otherwise generate
            if "session_number" in df.columns:
                df["session_number"] = df["session_number"].fillna(
                pd.Series(range(1, len(df) + 1), index=df.index)
                )
                df_plot = df.set_index("session_number")
            else:
                df_plot = df.reset_index(drop=True)
                df_plot.index = df_plot.index + 1

            # show points summary
            if "wellness_points" in df.columns:
                total_points = int(df["wellness_points"].fillna(0).sum())
                st.metric("Total Wellness Points", total_points)

            # TMD trend
            if "tmd_pre" in df_plot.columns and "tmd_post" in df_plot.columns:
                st.line_chart(df_plot[["tmd_pre", "tmd_post"]])
                st.caption("Trend of mood disturbance (TMD) before and after therapy across sessions.")
            else:
                st.caption("TMD columns not found for plotting (check DB schema / save step).")

            st.markdown("#### All Recorded Sessions")
            st.dataframe(df, use_container_width=True)


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