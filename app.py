# app.py - Sentinel Grid ULTRA DEMO (Grok Edition) - WARNINGS FIXED
import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import uuid
import plotly.express as px
import plotly.graph_objects as go
import requests

LOGO_URL = "https://i.ibb.co/Z17tkjG6/wide-logo.png"
SAMPLE_FREQ = 1.0
DEFAULT_WINDOW_SECONDS = 60
ANOMALY_Z_THRESHOLD = 3.0
MAX_POINTS = 5000

st.set_page_config(page_title="Sentinel Grid · Live Demo", layout="wide", page_icon="satellite", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .css-1d391kg {padding-top: 1rem !important;}
    .css-1v0mbdj {background: #0f111a;}
    .stPlotlyChart {background: #1e2130; border-radius: 12px;}
    [data-testid="stMetricValue"] {font-size: 2rem !important; font-weight: 700;}
    .stButton>button {background: #00b894; border: none; border-radius: 8px; height: 3rem; font-weight: 600;}
    .stButton>button:hover {background: #00cea9;}
    .risk-high {color: #e17055 !important;}
    .risk-low {color: #00b894 !important;}
</style>
""", unsafe_allow_html=True)

def make_initial_df(periods=240):
    now = datetime.utcnow()
    timestamps = [now - timedelta(seconds=SAMPLE_FREQ * (periods - 1 - i)) for i in range(periods)]
    return pd.DataFrame({"timestamp": timestamps,
                         "voltage": np.random.normal(230, 2.5, periods),
                         "frequency": np.random.normal(50, 0.03, periods),
                         "temperature": np.random.normal(35, 1.2, periods),
                         "load": np.random.normal(0.6, 0.12, periods)})

def generate_next_point(state):
    df = state["df"]
    now = datetime.utcnow()
    last = df.iloc[-1] if len(df) > 0 else None
    v = last["voltage"] + np.random.normal(0, 0.8) if last is not None else 230
    f = last["frequency"] + np.random.normal(0, 0.01) if last is not None else 50
    t = last["temperature"] + np.random.normal(0, 0.2) if last is not None else 35
    l = np.clip(last["load"] + np.random.normal(0, 0.02), 0, 2) if last is not None else 0.6
    scenario = state.get("active_scenario", {})
    if scenario.get("type") and datetime.utcnow() < scenario.get("until", now):
        sev = scenario.get("severity", 1.0)
        typ = scenario["type"]
        if typ == "overload":
            l = min(2.0, l + 0.25 * sev)
            v -= 10 * sev * max(0, l - 0.7)
        elif typ == "high_temp":
            t += 3.0 * sev + np.random.normal(0, 0.5)
        elif typ == "freq_drift":
            f += np.random.normal(0.06 * sev, 0.02)
        elif typ == "spike" and np.random.rand() < 0.4 * sev:
            v += 28 * sev
    return {"timestamp": now, "voltage": float(v), "frequency": float(f), "temperature": float(t), "load": float(l)}

def compute_anomalies(df):
    if df.empty: return df.copy()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    rolling = df.rolling(f"{DEFAULT_WINDOW_SECONDS}s", min_periods=1)
    means = rolling.mean()
    stds = rolling.std().fillna(1e-8)
    for col in ["voltage", "frequency", "temperature", "load"]:
        df[f"{col}_z"] = (df[col] - means[col]) / stds[col]
    df["rule_voltage"] = df["voltage"].abs().gt(245) | df["voltage"].lt(215)
    df["rule_frequency"] = df["frequency"].gt(50.2) | df["frequency"].lt(49.8)
    df["rule_temperature"] = df["temperature"].gt(40)
    df["rule_load"] = df["load"].gt(1.3)
    df["stat_anomaly"] = (
        df["voltage_z"].abs() > ANOMALY_Z_THRESHOLD
    ) | (
        df["frequency_z"].abs() > ANOMALY_Z_THRESHOLD * 0.8
    ) | (
        df["temperature_z"].abs() > ANOMALY_Z_THRESHOLD * 0.8
    )
    df["anomaly"] = df[["rule_voltage", "rule_frequency", "rule_temperature", "rule_load", "stat_anomaly"]].any(axis=1)
    return df.reset_index()

def simulator_thread(stop_event, lock, state):
    while not stop_event.is_set():
        with lock:
            point = generate_next_point(state)
            new_df = pd.concat([state["df"], pd.DataFrame([point])], ignore_index=True)
            state["df"] = new_df.tail(MAX_POINTS).reset_index(drop=True)
        time.sleep(SAMPLE_FREQ)

defaults = {
    "df": make_initial_df(),
    "state": {"df": make_initial_df(), "active_scenario": {}},
    "events": [],
    "sim_thread": None,
    "sim_stop": None,
    "sim_lock": threading.Lock(),
    "last_commit": None,
    "last_anom": 0
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

header = st.columns([1, 4, 1])
with header[0]:
    st.image(LOGO_URL, width=160)
with header[1]:
    st.markdown("<h1 style='margin:0; padding-top:10px;'>Sentinel Grid · Live Demo</h1>", unsafe_allow_html=True)
    st.caption("AI-driven resilience platform • live simulation • on-chain proofs (testnet)")
with header[2]:
    status = "LIVE" if st.session_state.sim_thread and st.session_state.sim_thread.is_alive() else "STOPPED"
    color = "#00b894" if status == "LIVE" else "#d63031"
    st.markdown(f"<div style='text-align:right;'><span style='color:{color};font-weight:700;font-size:1.4em;'>{status}</span></div>", unsafe_allow_html=True)

with st.session_state.sim_lock:
    display_df = compute_anomalies(st.session_state.state["df"].copy())
    anom_count = display_df["anomaly"].sum() if not display_df.empty else 0
    risk_level = "risk-high" if anom_count > 8 else "risk-low"
    st.markdown(f"<h2 style='text-align:center; color:#ffffff;'>Cascading Risk Index: <span class='{risk_level}'>{anom_count}/50</span></h2>", unsafe_allow_html=True)

if anom_count > st.session_state.last_anom:
    st.session_state.last_anom = anom_count
    st.markdown("<audio autoplay='true' src='https://assets.mixkit.co/sfx/preview/mixkit-alarm-tone-1077.mp3'></audio>", unsafe_allow_html=True)

st.markdown("---")

left, mid, right = st.columns([1.5, 3, 1.3])

with left:
    st.subheader("Controls")
    if st.button("Start Simulation", key="start"):
        if not (st.session_state.sim_thread and st.session_state.sim_thread.is_alive()):
            st.session_state.state["df"] = st.session_state.df.copy()
            stop_event = threading.Event()
            st.session_state.sim_stop = stop_event
            t = threading.Thread(target=simulator_thread, args=(stop_event, st.session_state.sim_lock, st.session_state.state), daemon=True)
            st.session_state.sim_thread = t
            t.start()
            st.session_state.events.insert(0, {"time": datetime.utcnow(), "type": "sim", "msg": "Live simulation started"})
            st.success("Started")
            st.rerun()

    if st.button("Stop Simulation", key="stop"):
        if st.session_state.sim_thread and st.session_state.sim_thread.is_alive():
            st.session_state.sim_stop.set()
            st.session_state.sim_thread.join(timeout=2)
            st.session_state.sim_thread = None
            st.session_state.df = st.session_state.state["df"].copy()
            st.session_state.events.insert(0, {"time": datetime.utcnow(), "type": "sim", "msg": "Simulation stopped"})
            st.success("Stopped")
            st.rerun()

    st.markdown("#### Scenarios")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Overload (30s)", key="ov"):
            st.session_state.state["active_scenario"] = {"type": "overload", "until": datetime.utcnow() + timedelta(seconds=30), "severity": 1.0}
            st.session_state.events.insert(0, {"time": datetime.utcnow(), "type": "scenario", "msg": "Overload (30s)"})
            st.success("Active")
        if st.button("High Temp (30s)", key="ht"):
            st.session_state.state["active_scenario"] = {"type": "high_temp", "until": datetime.utcnow() + timedelta(seconds=30), "severity": 1.0}
            st.session_state.events.insert(0, {"time": datetime.utcnow(), "type": "scenario", "msg": "High Temp (30s)"})
            st.success("Active")
    with col2:
        if st.button("Freq Drift (45s)", key="fd"):
            st.session_state.state["active_scenario"] = {"type": "freq_drift", "until": datetime.utcnow() + timedelta(seconds=45), "severity": 1.0}
            st.session_state.events.insert(0, {"time": datetime.utcnow(), "type": "scenario", "msg": "Freq Drift (45s)"})
            st.success("Active")
        if st.button("Spike (10s)", key="sp"):
            st.session_state.state["active_scenario"] = {"type": "spike", "until": datetime.utcnow() + timedelta(seconds=10), "severity": 1.5}
            st.session_state.events.insert(0, {"time": datetime.utcnow(), "type": "scenario", "msg": "Spike (10s)"})
            st.success("Active")

    st.markdown("#### On-Chain Proofs")
    if st.button("Publish Last 50 → Aptos Testnet"):
        try:
            payload = {"hash": str(uuid.uuid4()), "data": display_df.tail(50).to_json(), "time": datetime.utcnow().isoformat()}
            r = requests.post("https://fullnode.testnet.aptoslabs.com/v1/transactions", json=payload, timeout=5)
            tx = r.json().get("hash", "mock_tx_123456")
        except:
            tx = "mock_tx_fallback_987654"
        st.session_state.last_commit = {"id": tx[:10]}
        st.session_state.events.insert(0, {"time": datetime.utcnow(), "type": "commit", "msg": f"Published → {tx[:10]}..."})
        st.success(f"Committed: {tx[:10]}...")

    if st.button("Relay Alert → Wormhole"):
        st.code("Wormhole VAA emitted — Solana devnet"); st.balloons()

    if st.button("Export CSV"):
        csv = st.session_state.state["df"].to_csv(index=False)
        st.download_button("Download telemetry.csv", csv, "text/csv", "telemetry.csv")

with mid:
    tab1, tab2, tab3 = st.tabs(["Live Telemetry", "Dependency Graph", "Risk Forecast"])

    with tab1:
        st.subheader("Live Metrics")
        if not display_df.empty:
            latest = display_df.iloc[-1]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Voltage", f"{latest['voltage']:.2f} V", f"{latest['voltage'] - display_df['voltage'].mean():+.2f}")
            k2.metric("Frequency", f"{latest['frequency']:.3f} Hz", f"{latest['frequency'] - display_df['frequency'].mean():+.3f}")
            k3.metric("Temperature", f"{latest['temperature']:.1f} °C", f"{latest['temperature'] - display_df['temperature'].mean():+.1f}")
            k4.metric("Anomalies", anom_count)

        st.markdown("#### Time Series")
        chart_df = display_df.tail(600)
        anoms = chart_df[chart_df["anomaly"]]
        figs = []
        for col, title in [("voltage", "Voltage"), ("temperature", "Temperature"), ("frequency", "Frequency"), ("load", "Load")]:
            fig = px.line(chart_df, x="timestamp", y=col, title=title)
            fig.add_scatter(x=anoms["timestamp"], y=anoms[col], mode="markers", marker=dict(color="red", size=10), name="Anomaly")
            fig.update_layout(paper_bgcolor='#1e2130', plot_bgcolor='#1e2130', font_color="#ffffff")
            figs.append(fig)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(figs[0], width='stretch')
            st.plotly_chart(figs[1], width='stretch')
        with c2:
            st.plotly_chart(figs[2], width='stretch')
            st.plotly_chart(figs[3], width='stretch')

    with tab2:
        st.subheader("Infrastructure Dependency Graph (50 nodes)")
        N = 50
        np.random.seed(42)
        x = np.random.randn(N)
        y = np.random.randn(N)
        z = np.random.randn(N)
        load_vals = np.clip(display_df["load"].iloc[-N:] if len(display_df) >= N else display_df["load"], 0.3, 1.8)
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=12, color=load_vals, colorscale='Reds', showscale=True, colorbar_title="Load Risk"))])
        fig.update_layout(scene_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color="#ffffff", margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.subheader("48h Cascading Risk Forecast")
        forecast_df = pd.DataFrame({
            "time": pd.date_range(start=datetime.utcnow(), periods=48, freq='h'),
            "cascade_risk": np.clip(np.cumsum(np.random.normal(0.02, 0.05, 48)) + 0.1, 0, 1)
        })
        fig_forecast = px.area(forecast_df, x="time", y="cascade_risk", title="Probability of Cascading Failure")
        fig_forecast.update_layout(paper_bgcolor='#1e2130', plot_bgcolor='#1e2130', font_color="#ffffff")
        st.plotly_chart(fig_forecast, width='stretch')

with right:
    st.sidebar.metric("Node Location", "Miami-Dade Microgrid #4")
    st.sidebar.metric("UTC Time", datetime.utcnow().strftime("%H:%M:%S"))
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Share Demo**")
    share_url = f"https://sentinelgrid.monza.tech?session={uuid.uuid4()}"
    st.sidebar.markdown(f"`{share_url}`")

    st.subheader("Event Log")
    for e in st.session_state.events[:25]:
        t = e["time"].strftime("%H:%M:%S")
        msg = e["msg"]
        typ = e["type"]
        color = {"sim": "#00b894", "scenario": "#e17055", "commit": "#00cec9", "manual": "#a29bfe"}.get(typ, "#636e72")
        st.markdown(f"<small style='color:{color};'>[{t}] {msg}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("System")
    st.write(f"**Last commit:** {st.session_state.last_commit['id'] if st.session_state.last_commit else '—'}")
    st.write(f"**Points:** {len(st.session_state.state['df'])}")
    st.write(f"**Thread:** {'alive' if st.session_state.sim_thread and st.session_state.sim_thread.is_alive() else 'stopped'}")

    with st.expander("Debug • Raw Data (latest 50)"):
        with st.session_state.sim_lock:
            st.dataframe(st.session_state.state["df"].tail(50))

st.markdown("---")
footer = st.columns([3, 1])
with footer[0]:
    st.caption("Sentinel Grid · Monza Tech LLC — demo version. All on-chain commits are mocked.")
with footer[1]:
    if st.button("Reset All"):
        for key in list(st.session_state.keys()):
            if key in defaults:
                st.session_state[key] = defaults[key]
        st.rerun()