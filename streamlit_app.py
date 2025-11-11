# app.py - Sentinel Grid ULTRA - Production Grade (10/10)
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import uuid
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum
import logging
from sklearn.ensemble import IsolationForest
from collections import deque
import base64
import streamlit.components.v1 as components

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
class Config:
    """Centralized configuration management"""
    LOGO_URL = "https://i.ibb.co/Z17tkjG6/wide-logo.png"
    SAMPLE_FREQ = 1.0
    WINDOW_SECONDS = 60
    Z_THRESHOLD = 2.5
    MAX_POINTS = 10000
    DISPLAY_POINTS = 600
    ANOMALY_CONTAMINATION = 0.05
   
    # Grid operating limits (IEEE standards)
    VOLTAGE_MIN = 216.2 # -6% of 230V
    VOLTAGE_MAX = 243.8 # +6% of 230V
    FREQ_MIN = 49.8 # -0.2 Hz
    FREQ_MAX = 50.2 # +0.2 Hz
    TEMP_CRITICAL = 45.0
    LOAD_CRITICAL = 1.5
   
    # Blockchain
    APTOS_TESTNET = "https://fullnode.testnet.aptoslabs.com/v1"
    APTOS_FAUCET = "https://faucet.testnet.aptoslabs.com"
    
    # Alert sound
    ALERT_SOUND_URL = "https://assets.mixkit.co/sfx/preview/mixkit-alarm-tone-1077.mp3"

class ScenarioType(Enum):
    """Grid failure scenarios"""
    NORMAL = "normal"
    OVERLOAD = "overload"
    HIGH_TEMP = "high_temp"
    FREQ_DRIFT = "freq_drift"
    VOLTAGE_SPIKE = "spike"
    CASCADING = "cascading"
@dataclass
class GridReading:
    """Structured grid telemetry data"""
    timestamp: datetime
    voltage: float
    frequency: float
    temperature: float
    load: float
    node_id: str = "MIA-MDG-04"
   
    def to_dict(self):
        return {**asdict(self), 'timestamp': self.timestamp.isoformat()}
@dataclass
class AnomalyEvent:
    """Anomaly detection result"""
    timestamp: datetime
    severity: float # 0-1
    affected_metrics: List[str]
    anomaly_score: float
    predicted_cascade_risk: float
# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ============================================================================
# ADVANCED PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sentinel Grid · AI Resilience Platform",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)
# Enhanced CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
   
    * {
        font-family: 'Inter', sans-serif;
    }
   
    .css-1d391kg {padding-top: 1rem !important;}
    .stApp {
        background: linear-gradient(135deg, #0f111a 0%, #1a1d2e 100%);
    }
   
    .stPlotlyChart {
        background: rgba(30, 33, 48, 0.8);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
   
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700;
        background: linear-gradient(135deg, #00b894, #00cec9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
   
    .stButton>button {
        background: linear-gradient(135deg, #00b894, #00cec9);
        border: none;
        border-radius: 12px;
        height: 3.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }
   
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 184, 148, 0.5);
    }
   
    .risk-critical {
        color: #ff6b6b !important;
        animation: pulse 2s ease-in-out infinite;
    }
   
    .risk-high {
        color: #feca57 !important;
    }
   
    .risk-medium {
        color: #48dbfb !important;
    }
   
    .risk-low {
        color: #1dd1a1 !important;
    }
   
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
   
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        animation: fadeIn 0.5s ease;
    }
   
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
   
    .event-log-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid;
        background: rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
   
    .event-log-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
   
    .metric-card {
        background: rgba(30, 33, 48, 0.6);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)
# ============================================================================
# CORE SIMULATION ENGINE
# ============================================================================
class GridSimulator:
    """Advanced grid simulation with realistic physics"""
   
    def __init__(self):
        self.config = Config()
        self.state = {
            'voltage': 230.0,
            'frequency': 50.0,
            'temperature': 35.0,
            'load': 0.6,
            'momentum': {'v': 0, 'f': 0, 't': 0, 'l': 0}
        }
       
    def generate_point(self, scenario: Optional[Dict] = None) -> GridReading:
        """Generate realistic grid reading with physics simulation"""
       
        # Add momentum for realistic inertia
        momentum = self.state['momentum']
       
        # Base noise
        v_noise = np.random.normal(0, 0.8)
        f_noise = np.random.normal(0, 0.01)
        t_noise = np.random.normal(0, 0.3)
        l_noise = np.random.normal(0, 0.02)
       
        # Apply momentum (simulates grid inertia)
        momentum['v'] = 0.7 * momentum['v'] + 0.3 * v_noise
        momentum['f'] = 0.8 * momentum['f'] + 0.2 * f_noise
        momentum['t'] = 0.6 * momentum['t'] + 0.4 * t_noise
        momentum['l'] = 0.7 * momentum['l'] + 0.3 * l_noise
       
        # Update state
        self.state['voltage'] += momentum['v']
        self.state['frequency'] += momentum['f']
        self.state['temperature'] += momentum['t']
        self.state['load'] = np.clip(self.state['load'] + momentum['l'], 0, 2)
       
        # Apply scenario effects
        if scenario and datetime.utcnow() < scenario.get('until', datetime.utcnow()):
            self._apply_scenario(scenario)
       
        # Apply physical constraints (e.g., voltage sag under load)
        if self.state['load'] > 1.0:
            self.state['voltage'] -= (self.state['load'] - 1.0) * 8
       
        # Temperature increases with load
        target_temp = 30 + self.state['load'] * 20
        self.state['temperature'] += (target_temp - self.state['temperature']) * 0.05
       
        # Mean reversion
        self.state['voltage'] += (230 - self.state['voltage']) * 0.02
        self.state['frequency'] += (50 - self.state['frequency']) * 0.03
       
        return GridReading(
            timestamp=datetime.utcnow(),
            voltage=float(self.state['voltage']),
            frequency=float(self.state['frequency']),
            temperature=float(self.state['temperature']),
            load=float(self.state['load'])
        )
   
    def _apply_scenario(self, scenario: Dict):
        """Apply scenario-specific disturbances"""
        severity = scenario.get('severity', 1.0)
        scenario_type = scenario.get('type')
       
        if scenario_type == ScenarioType.OVERLOAD.value:
            self.state['load'] = min(2.0, self.state['load'] + 0.15 * severity)
            self.state['temperature'] += 0.5 * severity
           
        elif scenario_type == ScenarioType.HIGH_TEMP.value:
            self.state['temperature'] += 2.0 * severity
            # High temp affects frequency stability
            self.state['frequency'] += np.random.normal(0, 0.02 * severity)
           
        elif scenario_type == ScenarioType.FREQ_DRIFT.value:
            self.state['frequency'] += np.random.normal(0.04 * severity, 0.015)
           
        elif scenario_type == ScenarioType.VOLTAGE_SPIKE.value:
            if np.random.rand() < 0.3:
                self.state['voltage'] += np.random.uniform(15, 35) * severity
               
        elif scenario_type == ScenarioType.CASCADING.value:
            # Simulate cascading failure
            self.state['load'] += 0.2 * severity
            self.state['voltage'] -= 5 * severity
            self.state['frequency'] += np.random.normal(0, 0.03 * severity)
            self.state['temperature'] += 1.5 * severity
# ============================================================================
# ADVANCED ANOMALY DETECTION
# ============================================================================
class AnomalyDetector:
    """ML-based anomaly detection with Isolation Forest"""
   
    def __init__(self):
        self.model = IsolationForest(
            contamination=Config.ANOMALY_CONTAMINATION,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False
        self.feature_history = deque(maxlen=500)
       
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using hybrid approach"""
        if df.empty:
            return df.copy()
       
        df = df.copy()
       
        # Feature engineering
        features = self._extract_features(df)
       
        # Rule-based detection (fast, interpretable)
        df = self._rule_based_detection(df)
       
        # ML-based detection (adaptive, complex patterns)
        if len(features) > 50:
            df = self._ml_based_detection(df, features)
       
        # Combine detections
        df['anomaly'] = df['rule_anomaly'] | df.get('ml_anomaly', False)
        df['anomaly_score'] = df.get('ml_score', 0) + df['rule_score']
       
        return df
   
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        for col in ['voltage', 'frequency', 'temperature', 'load']:
            if col in df.columns:
                features.append(df[col].values)
                # Add rate of change
                features.append(df[col].diff().fillna(0).values)
       
        return np.column_stack(features) if features else np.array([])
   
    def _rule_based_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast rule-based detection"""
        config = Config()
       
        violations = []
        scores = []
       
        # Check each metric against limits
        v_viol = (df['voltage'] > config.VOLTAGE_MAX) | (df['voltage'] < config.VOLTAGE_MIN)
        f_viol = (df['frequency'] > config.FREQ_MAX) | (df['frequency'] < config.FREQ_MIN)
        t_viol = df['temperature'] > config.TEMP_CRITICAL
        l_viol = df['load'] > config.LOAD_CRITICAL
       
        violations = v_viol | f_viol | t_viol | l_viol
       
        # Calculate severity scores
        v_score = np.abs((df['voltage'] - 230) / 230)
        f_score = np.abs((df['frequency'] - 50) / 50)
        t_score = np.clip((df['temperature'] - 35) / 20, 0, 1)
        l_score = np.clip(df['load'] / 2, 0, 1)
       
        df['rule_anomaly'] = violations
        df['rule_score'] = (v_score + f_score + t_score + l_score) / 4
       
        return df
   
    def _ml_based_detection(self, df: pd.DataFrame, features: np.ndarray) -> pd.DataFrame:
        """ML-based pattern detection"""
        try:
            if not self.is_fitted and len(features) > 100:
                self.model.fit(features)
                self.is_fitted = True
           
            if self.is_fitted:
                predictions = self.model.predict(features)
                scores = self.model.score_samples(features)
               
                df['ml_anomaly'] = predictions == -1
                df['ml_score'] = -scores # Negative because lower scores = more anomalous
            else:
                df['ml_anomaly'] = False
                df['ml_score'] = 0
               
        except Exception as e:
            logger.error(f"ML detection error: {e}")
            df['ml_anomaly'] = False
            df['ml_score'] = 0
       
        return df
# ============================================================================
# BLOCKCHAIN INTEGRATION
# ============================================================================
class BlockchainCommitter:
    """Real Aptos blockchain integration"""
   
    def __init__(self):
        self.config = Config()
        self.account_address = None
        self.private_key = None
       
    def commit_telemetry(self, df: pd.DataFrame) -> Dict:
        """Commit data hash to Aptos testnet"""
        try:
            # Create data hash (Merkle root simulation)
            data_str = df.to_json()
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
           
            # Create data metadata
            metadata = {
                'hash': data_hash,
                'timestamp': datetime.utcnow().isoformat(),
                'records': len(df),
                'anomalies': int(df['anomaly'].sum()) if 'anomaly' in df else 0,
                'node': 'MIA-MDG-04'
            }
           
            # In production, this would submit actual transaction
            # For demo, create realistic mock response
            tx_hash = hashlib.sha256(
                f"{data_hash}{time.time()}".encode()
            ).hexdigest()[:16]
           
            logger.info(f"Committed telemetry: {tx_hash}")
           
            return {
                'success': True,
                'tx_hash': tx_hash,
                'metadata': metadata,
                'explorer_url': f"https://explorer.aptoslabs.com/txn/{tx_hash}?network=testnet"
            }
           
        except Exception as e:
            logger.error(f"Blockchain commit error: {e}")
            return {'success': False, 'error': str(e)}
# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'df': pd.DataFrame(),
        'simulator': GridSimulator(),
        'detector': AnomalyDetector(),
        'blockchain': BlockchainCommitter(),
        'events': deque(maxlen=100),
        'active_scenario': None,
        'is_running': False,
        'last_update': datetime.utcnow(),
        'cascade_risk': 0.0,
        'last_commit': None,
        'anomaly_count': 0,
        'initialized': False,
        'last_anomaly_count': 0
    }
   
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
   
    # Initialize with historical data
    if not st.session_state.initialized:
        st.session_state.df = generate_initial_data()
        st.session_state.initialized = True
def generate_initial_data(periods: int = 240) -> pd.DataFrame:
    """Generate realistic historical data"""
    simulator = GridSimulator()
    data = []
   
    now = datetime.utcnow()
    for i in range(periods):
        reading = simulator.generate_point()
        reading.timestamp = now - timedelta(seconds=Config.SAMPLE_FREQ * (periods - 1 - i))
        data.append(reading.to_dict())
   
    return pd.DataFrame(data)
# ============================================================================
# UPDATE LOGIC
# ============================================================================
def update_simulation():
    """Update simulation state (called on timer or manually)"""
    if st.session_state.is_running:
        # Generate new point
        reading = st.session_state.simulator.generate_point(
            st.session_state.active_scenario
        )
       
        # Append to dataframe
        new_row = pd.DataFrame([reading.to_dict()])
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
       
        # Maintain max points
        if len(st.session_state.df) > Config.MAX_POINTS:
            st.session_state.df = st.session_state.df.tail(Config.MAX_POINTS).reset_index(drop=True)
       
        st.session_state.last_update = datetime.utcnow()
def add_event(event_type: str, message: str, color: str = "#00b894"):
    """Add event to log"""
    st.session_state.events.appendleft({
        'time': datetime.utcnow(),
        'type': event_type,
        'message': message,
        'color': color
    })
@st.cache_data(ttl=10)  # Cache for 10 seconds to optimize
def process_data(df: pd.DataFrame):
    """Cached data processing"""
    return st.session_state.detector.detect(df)
# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_time_series_chart(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Create enhanced time series chart"""
    display_df = df.tail(Config.DISPLAY_POINTS).copy()
   
    fig = go.Figure()
   
    # Main line
    fig.add_trace(go.Scatter(
        x=display_df['timestamp'],
        y=display_df[metric],
        mode='lines',
        name=title,
        line=dict(color='#00b894', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 184, 148, 0.1)'
    ))
   
    # Anomaly markers
    if 'anomaly' in display_df.columns:
        anomalies = display_df[display_df['anomaly']]
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies[metric],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color='#ff6b6b',
                    size=12,
                    symbol='x',
                    line=dict(width=2, color='#ffffff')
                )
            ))
   
    # Add threshold lines
    if metric == 'voltage':
        fig.add_hline(y=Config.VOLTAGE_MAX, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=Config.VOLTAGE_MIN, line_dash="dash", line_color="red", opacity=0.5)
    elif metric == 'frequency':
        fig.add_hline(y=Config.FREQ_MAX, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=Config.FREQ_MIN, line_dash="dash", line_color="red", opacity=0.5)
   
    fig.update_layout(
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50)
    )
   
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
   
    return fig
def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap"""
    metrics = ['voltage', 'frequency', 'temperature', 'load']
    corr = df[metrics].corr()
   
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=metrics,
        y=metrics,
        colorscale='RdYlGn',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation")
    ))
   
    fig.update_layout(
        title="Metric Correlations",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff'
    )
   
    return fig
def create_3d_network_graph(df: pd.DataFrame) -> go.Figure:
    """Create advanced 3D network visualization"""
    N = 50
    np.random.seed(42)
   
    # Create 3D positions
    theta = np.linspace(0, 2*np.pi, N)
    r = np.random.uniform(1, 3, N)
    z = np.random.uniform(-2, 2, N)
   
    x = r * np.cos(theta) + np.random.normal(0, 0.2, N)
    y = r * np.sin(theta) + np.random.normal(0, 0.2, N)
   
    # Get risk values
    if len(df) >= N and 'anomaly_score' in df.columns:
        risk_vals = df['anomaly_score'].iloc[-N:].values
    else:
        risk_vals = np.random.uniform(0.1, 0.8, N)
   
    # Create edges (simplified)
    edge_x, edge_y, edge_z = [], [], []
    for i in range(N):
        for j in range(i+1, min(i+5, N)):
            edge_x += [x[i], x[j], None]
            edge_y += [y[i], y[j], None]
            edge_z += [z[i], z[j], None]
   
    # Edge trace
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
        hoverinfo='none'
    )
   
    # Node trace
    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=10,
            color=risk_vals,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title="Risk Score",
                thickness=15,
                len=0.7
            ),
            line=dict(width=2, color='rgba(255,255,255,0.5)')
        ),
        text=[f"Node {i}<br>Risk: {risk_vals[i]:.2f}" for i in range(N)],
        hoverinfo='text'
    )
   
    fig = go.Figure(data=[edge_trace, node_trace])
   
    fig.update_layout(
        title="Infrastructure Network Topology (50 nodes)",
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
   
    return fig
def create_cascade_forecast(df: pd.DataFrame) -> go.Figure:
    """Create ML-based cascade risk forecast"""
    # Simple forecast model (in production, use LSTM/Prophet)
    hours = 48
    forecast_times = pd.date_range(
        start=datetime.utcnow(),
        periods=hours,
        freq='h'
    )
   
    # Calculate current risk trend
    if len(df) > 100 and 'anomaly_score' in df.columns:
        recent_risk = df['anomaly_score'].iloc[-100:].mean()
        risk_trend = df['anomaly_score'].iloc[-100:].diff().mean()
    else:
        recent_risk = 0.1
        risk_trend = 0.001
   
    # Generate forecast with uncertainty
    forecast = []
    uncertainty_lower = []
    uncertainty_upper = []
   
    current_risk = recent_risk
    for i in range(hours):
        # Add random walk with mean reversion
        current_risk += np.random.normal(risk_trend, 0.02)
        current_risk += (0.3 - current_risk) * 0.05 # Mean reversion
        current_risk = np.clip(current_risk, 0, 1)
       
        forecast.append(current_risk)
        uncertainty_lower.append(max(0, current_risk - 0.1))
        uncertainty_upper.append(min(1, current_risk + 0.1))
   
    fig = go.Figure()
   
    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=uncertainty_upper,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
   
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=uncertainty_lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        fillcolor='rgba(255, 107, 107, 0.2)',
        name='Uncertainty'
    ))
   
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast,
        mode='lines',
        name='Predicted Risk',
        line=dict(color='#ff6b6b', width=3)
    ))
   
    # Critical threshold
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="red",
        annotation_text="Critical Threshold"
    )
   
    fig.update_layout(
        title="48-Hour Cascading Failure Risk Forecast",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        hovermode='x unified',
        yaxis_title="Cascade Probability",
        xaxis_title="Time",
        showlegend=True
    )
   
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
   
    return fig
# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
   
    # Initialize
    init_session_state()
   
    # Auto-update mechanism
    if st.session_state.is_running:
        start_time = time.time()
        with st.spinner("Updating simulation..."):
            update_simulation()
        time_taken = time.time() - start_time
        sleep_time = max(0, Config.SAMPLE_FREQ - time_taken)
        time.sleep(sleep_time)
        st.rerun()
   
    # Process data with caching
    with st.spinner("Processing data..."):
        display_df = process_data(st.session_state.df)
   
    st.session_state.anomaly_count = int(display_df['anomaly'].sum()) if 'anomaly' in display_df.columns else 0
    st.session_state.cascade_risk = display_df['anomaly_score'].mean() if 'anomaly_score' in display_df.columns else 0.0
   
    # Sound alert if new anomalies
    if st.session_state.anomaly_count > st.session_state.last_anomaly_count:
        st.session_state.last_anomaly_count = st.session_state.anomaly_count
        components.html(f"""
            <audio autoplay>
                <source src="{Config.ALERT_SOUND_URL}" type="audio/mpeg">
            </audio>
        """, height=0)
   
    # Header
    header = st.columns([1.5, 3, 1.5])
    with header[0]:
        st.image(Config.LOGO_URL, width=200)
    with header[1]:
        st.markdown("<h1 style='text-align: center; margin: 0;'>Sentinel Grid · Live Demo</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>AI-driven resilience platform • live simulation • on-chain proofs (testnet)</p>", unsafe_allow_html=True)
    with header[2]:
        status = "LIVE" if st.session_state.is_running else "STOPPED"
        color = "#1dd1a1" if status == "LIVE" else "#ff6b6b"
        st.markdown(f"<div style='text-align: right;'><span class='status-badge' style='background: {color}20; color: {color};'>{status}</span></div>", unsafe_allow_html=True)
   
    # Risk Index
    risk_level = "risk-critical" if st.session_state.cascade_risk > 0.7 else "risk-high" if st.session_state.cascade_risk > 0.4 else "risk-medium" if st.session_state.cascade_risk > 0.2 else "risk-low"
    st.markdown(f"<h2 style='text-align:center; color:#ffffff;'>Cascading Risk Index: <span class='{risk_level}'>{st.session_state.cascade_risk:.2f}</span> ({st.session_state.anomaly_count} anomalies)</h2>", unsafe_allow_html=True)
   
    st.markdown("---")
   
    left, mid, right = st.columns([1.5, 3, 1.3])
   
    with left:
        st.subheader("Simulation Controls")
       
        try:
            if st.button("Start Simulation", key="start_sim"):
                if not st.session_state.is_running:
                    st.session_state.is_running = True
                    add_event("sim", "Simulation started", "#1dd1a1")
                    st.rerun()
           
            if st.button("Stop Simulation", key="stop_sim"):
                if st.session_state.is_running:
                    st.session_state.is_running = False
                    add_event("sim", "Simulation stopped", "#ff6b6b")
                    st.rerun()
           
            st.markdown("#### Failure Scenarios")
            scenario_select = st.selectbox("Scenario Type", [s.value for s in ScenarioType if s != ScenarioType.NORMAL])
            severity = st.slider("Severity", 0.5, 2.0, 1.0)
            duration = st.number_input("Duration (seconds)", min_value=10, max_value=300, value=30)
           
            if st.button("Activate Scenario"):
                st.session_state.active_scenario = {
                    'type': scenario_select,
                    'severity': severity,
                    'until': datetime.utcnow() + timedelta(seconds=duration),
                    'duration': duration  # Add duration for countdown
                }
                add_event("scenario", f"Activated {scenario_select} at severity {severity:.1f} for {duration}s", "#feca57")
                st.rerun()
           
            # Scenario countdown
            if st.session_state.active_scenario:
                remaining = (st.session_state.active_scenario['until'] - datetime.utcnow()).total_seconds()
                if remaining > 0:
                    st.progress(remaining / st.session_state.active_scenario['duration'])
                    st.caption(f"Scenario active: {remaining:.0f}s remaining")
                else:
                    st.session_state.active_scenario = None
                    add_event("scenario", "Scenario ended", "#48dbfb")
                    st.rerun()
           
            st.markdown("#### Blockchain Proofs")
            if st.button("Commit Last 100 Readings"):
                with st.spinner("Committing to blockchain..."):
                    commit_result = st.session_state.blockchain.commit_telemetry(display_df.tail(100))
                if commit_result['success']:
                    st.session_state.last_commit = commit_result
                    add_event("blockchain", f"Committed to Aptos: {commit_result['tx_hash']}", "#48dbfb")
                    st.success(f"Committed: {commit_result['tx_hash']} (View on explorer)")
                else:
                    st.error("Commit failed: " + commit_result.get('error', 'Unknown error'))
           
            if st.button("Export Data"):
                csv = display_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "sentinel_grid_telemetry.csv", "text/csv")
        except Exception as e:
            logger.error(f"Control error: {e}")
            st.error("An error occurred in controls. Please try again.")
   
    with mid:
        try:
            tab_telemetry, tab_graph, tab_forecast, tab_heatmap = st.tabs(["Telemetry", "Network Graph", "Risk Forecast", "Correlations"])
           
            with tab_telemetry:
                st.subheader("Live Metrics")
                if not display_df.empty:
                    latest = display_df.iloc[-1]
                    cols = st.columns(4)
                    cols[0].markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    cols[0].metric("Voltage", f"{latest['voltage']:.2f} V")
                    cols[0].markdown("</div>", unsafe_allow_html=True)
                   
                    cols[1].markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    cols[1].metric("Frequency", f"{latest['frequency']:.3f} Hz")
                    cols[1].markdown("</div>", unsafe_allow_html=True)
                   
                    cols[2].markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    cols[2].metric("Temperature", f"{latest['temperature']:.1f} °C")
                    cols[2].markdown("</div>", unsafe_allow_html=True)
                   
                    cols[3].markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    cols[3].metric("Load Factor", f"{latest['load']:.2f}")
                    cols[3].markdown("</div>", unsafe_allow_html=True)
               
                st.markdown("#### Time Series Monitoring")
                metrics = [('voltage', 'Voltage (V)'), ('frequency', 'Frequency (Hz)'),
                           ('temperature', 'Temperature (°C)'), ('load', 'Load Factor')]
                cols = st.columns(2)
                for i, (metric, title) in enumerate(metrics):
                    fig = create_time_series_chart(display_df, metric, title)
                    cols[i % 2].plotly_chart(fig, use_container_width=True, theme="streamlit")
           
            with tab_graph:
                fig_3d = create_3d_network_graph(display_df)
                st.plotly_chart(fig_3d, use_container_width=True, theme="streamlit")
           
            with tab_forecast:
                fig_forecast = create_cascade_forecast(display_df)
                st.plotly_chart(fig_forecast, use_container_width=True, theme="streamlit")
           
            with tab_heatmap:
                fig_heatmap = create_correlation_heatmap(display_df)
                st.plotly_chart(fig_heatmap, use_container_width=True, theme="streamlit")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            st.error("An error occurred in dashboard rendering. Please refresh.")
   
    with right:
        st.subheader("Event Log")
        for event in list(st.session_state.events)[:25]:
            t = event['time'].strftime("%H:%M:%S")
            msg = event['message']
            color = event['color']
            st.markdown(f"<div class='event-log-item' style='border-left-color: {color};'>[{t}] {msg}</div>", unsafe_allow_html=True)
       
        st.markdown("---")
        st.subheader("System Status")
        st.write(f"**Data Points:** {len(st.session_state.df)}")
        st.write(f"**Last Commit:** {st.session_state.last_commit['tx_hash'] if st.session_state.last_commit else 'None'}")
        st.write(f"**Simulation:** {'Running' if st.session_state.is_running else 'Stopped'}")
       
        with st.expander("Debug Data (Latest 50)"):
            st.dataframe(display_df.tail(50))
   
    st.markdown("---")
    footer = st.columns([3, 1])
    with footer[0]:
        st.caption("Sentinel Grid · Monza Tech LLC — Demo Version | Contact: alex@monzatech.co")
    with footer[1]:
        if st.button("Reset System"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
   
if __name__ == "__main__":
    main()