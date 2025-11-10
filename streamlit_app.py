import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sentinel Grid · Live Resilience Console",
    page_icon="🛰️",
    layout="wide"
)

# =========================
# STYLES
# =========================
st.markdown("""
<style>
body {
    background-color: #02040A;
    color: #f5f5f5;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}
hr {
    border: 0;
    border-top: 1px solid #222;
}
.block-container {
    padding-top: 0.5rem;
}
.metric-label {
    font-size: 0.75rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #e5e7eb;
}
.metric-sub {
    font-size: 0.75rem;
    color: #6b7280;
}
.resilience-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    background: linear-gradient(90deg, rgba(56,189,248,0.12), rgba(8,47,73,0.6));
    color: #7dd3fc;
    border: 1px solid rgba(56,189,248,0.3);
}
.resilience-score {
    font-size: 1.9rem;
    font-weight: 700;
}
.section-label {
    font-size: 0.75rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.panel {
    background: radial-gradient(circle at top left, rgba(56,189,248,0.05), transparent),
                #020817;
    border-radius: 18px;
    padding: 14px 16px 12px 16px;
    border: 1px solid rgba(75,85,99,0.7);
}
.panel-soft {
    background: #020817;
    border-radius: 16px;
    padding: 12px 14px;
    border: 1px solid rgba(31,41,55,0.9);
}
.tab-header button[role="tab"] {
    border-radius: 999px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div style='text-align: center; padding: 18px 0 10px 0;'>
    <img src='https://i.ibb.co/Z17tkjG6/wide-logo.png' width='260'/>
</div>
""", unsafe_allow_html=True)

top_left, top_mid, top_right = st.columns([1.4, 2, 1.6])
with top_mid:
    st.markdown(
        "<div style='text-align:center; font-size:22px; font-weight:500; color:#e5e7eb;'>"
        "Sentinel Grid · Live Resilience Console"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; font-size:12px; color:#6b7280;'>"
        "Simulated telemetry, anomaly detection & risk scoring for critical infrastructure."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# SESSION STATE INIT
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "sim_running" not in st.session_state:
    st.session_state.sim_running = False

# =========================
# HELPERS
# =========================
def generate_base_data(n_minutes: int = 240):
    timestamps = pd.date_range(datetime.now() - timedelta(minutes=n_minutes), periods=n_minutes, freq="T")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "voltage": np.random.normal(230, 3, n_minutes),
        "frequency": np.random.normal(50, 0.06, n_minutes),
        "temperature": np.random.normal(34, 1.5, n_minutes),
    })
    return df

def append_sim_row(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = generate_base_data(60)

    last_ts = df["timestamp"].iloc[-1]
    new_ts = last_ts + timedelta(minutes=1)

    v = df["voltage"].iloc[-1] + np.random.normal(0, 0.8)
    f = df["frequency"].iloc[-1] + np.random.normal(0, 0.015)
    t = df["temperature"].iloc[-1] + np.random.normal(0, 0.12)

    if np.random.rand() < 0.03:
        v += np.random.choice([8, -8])
    if np.random.rand() < 0.02:
        f += np.random.choice([0.25, -0.25])
    if np.random.rand() < 0.02:
        t += np.random.choice([4, 6])

    new_row = pd.DataFrame({
        "timestamp": [new_ts],
        "voltage": [v],
        "frequency": [f],
        "temperature": [t],
    })

    df = pd.concat([df, new_row], ignore_index=True)
    return df.tail(480)

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    df["anomaly"] = (
        (df["voltage"] > 245) | (df["voltage"] < 215) |
        (df["frequency"] > 50.2) | (df["frequency"] < 49.8) |
        (df["temperature"] > 40)
    )
    return df

def compute_resilience_index(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    window = df.tail(120)
    total = len(window)
    if total == 0:
        return 0
    anomalies = window["anomaly"].sum()
    anomaly_ratio = anomalies / total

    v_jitter = window["voltage"].std() / 10
    f_jitter = window["frequency"].std() * 5
    t_jitter = window["temperature"].std() / 8

    risk = anomaly_ratio * 3 + v_jitter + f_jitter + t_jitter
    idx = int(max(0, min(100, 100 - risk * 90)))
    return idx

def generate_ai_insight(df: pd.DataFrame, resilience_index: int) -> str:
    if df is None or df.empty:
        return "Awaiting telemetry. No infrastructure risk detected yet."
    recent = df.tail(60)
    a = recent["anomaly"].sum()
    if resilience_index > 88 and a == 0:
        return "Grid conditions nominal in the last hour. No significant anomalies detected. Sentinel Grid remains in standard readiness posture."
    if resilience_index > 75:
        return "Minor fluctuations detected but within safe thresholds. Recommend passive monitoring of nodes with recurring micro deviations."
    if resilience_index > 55:
        return "Elevated risk cluster emerging. Recommend pre-emptive inspection of affected feeders and verification of protection settings."
    if resilience_index > 35:
        return "Correlated anomalies across multiple metrics. Recommend standby incident response and load redistribution simulation."
    return "Critical pattern detected. Recommend immediate operator review, isolation simulation of suspect nodes, and execution of incident playbook."

# =========================
# SIDEBAR: DATA & SIM
# =========================
with st.sidebar:
    st.markdown("### Telemetry Input")
    st.write("Upload real CSV data or start with synthetic telemetry for the simulation.")

    uploaded_file = st.file_uploader(
        "Upload CSV (timestamp, voltage, frequency, temperature)",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if "timestamp" in df_upload.columns:
                df_upload["timestamp"] = pd.to_datetime(df_upload["timestamp"])
            st.session_state.df = df_upload
            st.session_state.sim_running = False
            st.success("Telemetry loaded.")
        except Exception:
            st.error("Could not parse CSV. Please check column names and formats.")

    if st.button("Generate synthetic telemetry", use_container_width=True):
        st.session_state.df = generate_base_data(240)
        st.session_state.sim_running = False
        st.success("Synthetic telemetry generated.")

    st.markdown("---")
    st.markdown("### Live Simulation")

    mode = st.radio(
        "Mode",
        ["Static analysis", "Live simulation"],
        index=1 if st.session_state.sim_running else 0
    )

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start", key="start_sim", use_container_width=True):
            if st.session_state.df is None:
                st.session_state.df = generate_base_data(120)
            st.session_state.sim_running = True
    with col_stop:
        if st.button("Stop", key="stop_sim", use_container_width=True):
            st.session_state.sim_running = False

    st.caption(
        "In this demo, live simulation advances the telemetry stream each run, "
        "emulating near real-time grid data."
    )

if st.session_state.df is None:
    st.info("Load telemetry from the sidebar or generate synthetic data to begin.")
    st.stop()

# =========================
# MAIN PLACEHOLDERS
# =========================
placeholder_metrics = st.empty()
placeholder_charts = st.empty()
placeholder_ai = st.empty()

# =========================
# RENDER FUNCTION
# =========================
def render_once():
    df = detect_anomalies(st.session_state.df)
    st.session_state.df = df

    # ---- METRICS ROW ----
    with placeholder_metrics.container():
        m1, m2, m3, m4 = st.columns([1.1, 1.1, 1.1, 1.2])

        total_points = len(df)
        total_anomalies = int(df["anomaly"].sum())
        anomaly_rate = (total_anomalies / total_points * 100) if total_points else 0
        resilience_index = compute_resilience_index(df)

        with m1:
            st.markdown('<div class="metric-label">Data Points</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{total_points:,}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Last 8 hours window</div>', unsafe_allow_html=True)

        with m2:
            st.markdown('<div class="metric-label">Anomalies Detected</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{total_anomalies}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-sub">{anomaly_rate:.2f}% of recent telemetry</div>', unsafe_allow_html=True)

        with m3:
            st.markdown('<div class="metric-label">Active Alerts</div>', unsafe_allow_html=True)
            active_alerts = max(1, int(total_anomalies * 0.1)) if total_anomalies > 0 else 0
            st.markdown(f'<div class="metric-value">{active_alerts}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-sub">Severity-weighted</div>', unsafe_allow_html=True)

        with m4:
            st.markdown('<div class="metric-label">Resilience Index</div>', unsafe_allow_html=True)
            color = (
                "#22c55e" if resilience_index > 80 else
                "#eab308" if resilience_index > 55 else
                "#f97316" if resilience_index > 35 else
                "#ef4444"
            )
            st.markdown(
                f'<div class="panel">'
                f'<div class="resilience-score" style="color:{color};">{resilience_index}</div>'
                f'<div class="metric-sub">0 = critical · 100 = highly resilient</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ---- CHARTS + TABLE ----
    with placeholder_charts.container():
        c1, c2 = st.columns([2.3, 1.7])

        with c1:
            st.markdown('<div class="section-label">Telemetry Streams</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(["Voltage", "Frequency", "Temperature"])
            df_plot = df.set_index("timestamp")
            with t1:
                st.line_chart(df_plot["voltage"], height=190)
            with t2:
                st.line_chart(df_plot["frequency"], height=190)
            with t3:
                st.line_chart(df_plot["temperature"], height=190)

        with c2:
            st.markdown('<div class="section-label">Recent Anomalies</div>', unsafe_allow_html=True)
            panel_df = df[df["anomaly"]].tail(6)[["timestamp", "voltage", "frequency", "temperature"]]
            if panel_df.empty:
                st.markdown(
                    '<div class="panel-soft">No critical anomalies in the latest window. Sentinel Grid is in monitoring mode.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="panel-soft">', unsafe_allow_html=True)
                st.dataframe(
                    panel_df.rename(columns={
                        "timestamp": "Time",
                        "voltage": "V",
                        "frequency": "Hz",
                        "temperature": "°C"
                    }),
                    use_container_width=True,
                    height=190
                )
                st.markdown('</div>', unsafe_allow_html=True)

    # ---- AI INSIGHT + PUBLISH ----
    with placeholder_ai.container():
        left, right = st.columns([2.3, 1.7])

        resilience_index = compute_resilience_index(df)
        ai_text = generate_ai_insight(df, resilience_index)

        with left:
            st.markdown('<div class="section-label">AI Situation Briefing</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel">{ai_text}</div>',
                unsafe_allow_html=True
            )

        with right:
            st.markdown('<div class="section-label">Integrity & Export</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-soft">'
                '<div class="resilience-pill">● Audit Trail Ready</div>'
                '<div style="font-size:12px; color:#9ca3af; margin-top:6px;">'
                'This demo simulates hashing the last 15 minutes of anomalies and '
                'pushing a commitment to a blockchain layer (e.g. Zcash, Solana, or BNB Chain) '
                'for tamper-evident logging.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
            if st.button("Simulate on-chain commit", key="commit_button", use_container_width=True):
                with st.spinner("Hashing anomalies and writing integrity proof..."):
                    time.sleep(1.0)
                st.success("Integrity proof recorded (simulated).")

# =========================
# STATIC OR LIVE
# =========================
if mode == "Static analysis" or not st.session_state.sim_running:
    render_once()
else:
    # Advance the stream a bit each run, then render ONCE (no duplicate widgets)
    for _ in range(40):
        st.session_state.df = append_sim_row(st.session_state.df)
    render_once()

# =========================
# FOOTER
# =========================
st.markdown("""
<hr/>
<div style='text-align: center; color: #6b7280; font-size: 11px; padding-top: 4px;'>
Sentinel Grid MVP · Monza Tech LLC · Simulated environment for demonstration purposes only.
</div>
""", unsafe_allow_html=True)