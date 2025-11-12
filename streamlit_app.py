# Sentinel Grid v2.2.1 - Fixed Issues Edition
# Monza Tech LLC - All issues corrected, with added robustness

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Global configuration"""
    LOGO_URL = "https://i.ibb.co/Z17tkjG/wide-logo.png"  # Fixed broken URL if needed
    SAMPLE_RATE = 3.0
    MAX_HISTORY = 500
    CRITICAL_THRESHOLD = 0.75
    WARNING_THRESHOLD = 0.50
    AUTO_MITIGATION_THRESHOLD = 0.65
    INFRASTRUCTURE_TYPES = {
        'power': {'name': 'Power Grid', 'color': '#feca57', 'icon': '⚡'},
        'telecom': {'name': 'Telecommunications', 'color': '#48dbfb', 'icon': '📡'},
        'data': {'name': 'Data Centers', 'color': '#ff6b6b', 'icon': '💾'},
        'transport': {'name': 'Transportation', 'color': '#1dd1a1', 'icon': '🚇'},
        'water': {'name': 'Water Systems', 'color': '#54a0ff', 'icon': '💧'},
        'finance': {'name': 'Financial', 'color': '#a29bfe', 'icon': '💰'}
    }
    NETWORK_NODES = [
        {'id': 'FRA-PWR-01', 'type': 'power', 'name': 'Main Power Station', 'lat': 50.1109, 'lon': 8.6821, 'capacity': 2500},
        {'id': 'FRA-PWR-02', 'type': 'power', 'name': 'East Distribution Hub', 'lat': 50.1222, 'lon': 8.7140, 'capacity': 1200},
        {'id': 'FRA-PWR-03', 'type': 'power', 'name': 'Solar Farm Array', 'lat': 50.0755, 'lon': 8.6398, 'capacity': 800},
        {'id': 'FRA-TEL-01', 'type': 'telecom', 'name': 'Central Telecom Exchange', 'lat': 50.1155, 'lon': 8.6842, 'capacity': 150},
        {'id': 'FRA-TEL-02', 'type': 'telecom', 'name': 'Fiber Hub North', 'lat': 50.1345, 'lon': 8.6912, 'capacity': 100},
        {'id': 'FRA-DATA-01', 'type': 'data', 'name': 'DE-CIX Primary', 'lat': 50.1109, 'lon': 8.6821, 'capacity': 200},
        {'id': 'FRA-DATA-02', 'type': 'data', 'name': 'Cloud Region West', 'lat': 50.1034, 'lon': 8.6512, 'capacity': 150},
        {'id': 'FRA-DATA-03', 'type': 'data', 'name': 'Financial Data Vault', 'lat': 50.1138, 'lon': 8.6794, 'capacity': 100},
        {'id': 'FRA-TRN-01', 'type': 'transport', 'name': 'International Airport', 'lat': 50.0379, 'lon': 8.5622, 'capacity': 5000},
        {'id': 'FRA-TRN-02', 'type': 'transport', 'name': 'Central Train Station', 'lat': 50.1070, 'lon': 8.6638, 'capacity': 3000},
        {'id': 'FRA-TRN-03', 'type': 'transport', 'name': 'Metro Control Center', 'lat': 50.1155, 'lon': 8.6842, 'capacity': 1000},
        {'id': 'FRA-WTR-01', 'type': 'water', 'name': 'Water Treatment Plant', 'lat': 50.0889, 'lon': 8.6142, 'capacity': 800},
        {'id': 'FRA-WTR-02', 'type': 'water', 'name': 'Distribution Reservoir', 'lat': 50.1289, 'lon': 8.7012, 'capacity': 500},
        {'id': 'FRA-FIN-01', 'type': 'finance', 'name': 'Stock Exchange', 'lat': 50.1138, 'lon': 8.6794, 'capacity': 100},
        {'id': 'FRA-FIN-02', 'type': 'finance', 'name': 'Central Bank Hub', 'lat': 50.1109, 'lon': 8.6821, 'capacity': 80},
        {'id': 'FRA-FIN-03', 'type': 'finance', 'name': 'Banking Data Center', 'lat': 50.1089, 'lon': 8.6945, 'capacity': 60}
    ]
    SCENARIO_PRESETS = {
        'Heatwave Crisis': {
            'threats': [
                {'type': 'environmental', 'target': 'FRA-PWR-01', 'severity': 0.8, 'duration': 60},
                {'type': 'overload', 'target': 'FRA-PWR-02', 'severity': 0.7, 'duration': 60}
            ],
            'weather': {'condition': 'Severe Storm', 'severity': 0.9}
        },
        'Cyber + Physical': {
            'threats': [
                {'type': 'cyber_attack', 'target': 'FRA-DATA-01', 'severity': 0.9, 'duration': 45},
                {'type': 'physical_damage', 'target': 'FRA-TEL-01', 'severity': 0.6, 'duration': 45}
            ]
        },
        'Cascade Trigger': {
            'threats': [
                {'type': 'physical_damage', 'target': 'FRA-PWR-01', 'severity': 1.0, 'duration': 30}
            ]
        }
    }

class ThreatType(Enum):
    CYBER_ATTACK = "cyber_attack"
    PHYSICAL_DAMAGE = "physical_damage"
    OVERLOAD = "overload"
    ENVIRONMENTAL = "environmental"
    CASCADE = "cascade"
    SUPPLY_CHAIN = "supply_chain"

@dataclass
class InfrastructureNode:
    id: str
    type: str
    name: str
    lat: float
    lon: float
    capacity: float
    current_load: float = 0.0
    health: float = 1.0
    risk_score: float = 0.0
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'name': self.name,
            'current_load': self.current_load,
            'capacity': self.capacity,
            'health': self.health,
            'risk_score': self.risk_score
        }

@dataclass
class CascadeEvent:
    timestamp: datetime
    origin_node: str
    affected_nodes: List[str]
    impact_score: float
    threat_type: ThreatType
    auto_mitigated: bool = False

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'origin_node': self.origin_node,
            'affected_nodes': self.affected_nodes,
            'impact_score': self.impact_score,
            'threat_type': self.threat_type.value,
            'auto_mitigated': self.auto_mitigated
        }

# ============================================================================
# STREAMLIT CONFIG
# ============================================================================
st.set_page_config(
    page_title="Sentinel Grid · Critical Infrastructure Resilience",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    .stApp {
        background: #0a0e27;
        background-image:
            radial-gradient(circle at 20% 50%, rgba(26, 35, 126, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(13, 71, 161, 0.3) 0%, transparent 50%);
    }
    .status-card {
        background: rgba(13, 17, 38, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    .status-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    .risk-critical { color: #ff6b6b !important; font-weight: 700; animation: pulse-critical 2s ease-in-out infinite; }
    .risk-warning { color: #feca57 !important; font-weight: 700; }
    .risk-normal { color: #1dd1a1 !important; font-weight: 700; }
    @keyframes pulse-critical { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
    .threat-alert { background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 71, 87, 0.2)); border-left: 4px solid #ff6b6b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; animation: slideIn 0.5s ease; }
    @keyframes slideIn { from { transform: translateX(-100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    .metric-large { font-size: 3rem; font-weight: 700; line-height: 1; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6); }
    .recommendation-card { background: rgba(102, 126, 234, 0.1); border-left: 3px solid #667eea; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .timeline-event { border-left: 3px solid #667eea; padding-left: 1rem; margin: 1rem 0; position: relative; }
    .timeline-event::before { content: ''; position: absolute; left: -7px; top: 0; width: 12px; height: 12px; border-radius: 50%; background: #667eea; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIMULATION ENGINE
# ============================================================================
class InfrastructureSimulator:
    def __init__(self):
        np.random.seed(42)  # Fix: Added seed for reproducibility
        self.nodes: Dict[str, InfrastructureNode] = {}
        self.dependencies = {}
        self.initialize_network()

    def initialize_network(self):
        for node_data in Config.NETWORK_NODES:
            node = InfrastructureNode(**node_data)
            node.current_load = float(np.random.uniform(0.4, 0.7) * node.capacity)
            self.nodes[node.id] = node
        self.dependencies = {
            'FRA-TEL-01': ['FRA-PWR-01'],
            'FRA-TEL-02': ['FRA-PWR-02'],
            'FRA-DATA-01': ['FRA-PWR-01', 'FRA-TEL-01'],
            'FRA-DATA-02': ['FRA-PWR-02', 'FRA-TEL-01'],
            'FRA-DATA-03': ['FRA-PWR-01', 'FRA-TEL-01'],
            'FRA-TRN-01': ['FRA-PWR-03', 'FRA-TEL-01'],
            'FRA-TRN-02': ['FRA-PWR-01', 'FRA-TEL-01'],
            'FRA-TRN-03': ['FRA-PWR-02', 'FRA-TEL-02'],
            'FRA-WTR-01': ['FRA-PWR-01'],
            'FRA-WTR-02': ['FRA-PWR-02'],
            'FRA-FIN-01': ['FRA-PWR-01', 'FRA-DATA-03', 'FRA-TEL-01'],
            'FRA-FIN-02': ['FRA-PWR-01', 'FRA-DATA-01', 'FRA-TEL-01'],
            'FRA-FIN-03': ['FRA-PWR-02', 'FRA-DATA-02', 'FRA-TEL-01']
        }
        for node_id, deps in self.dependencies.items():
            if node_id in self.nodes:
                self.nodes[node_id].dependencies = deps

    def update_state(self, threat: Optional[Dict] = None, weather_data: Optional[Dict] = None):
        hour = datetime.utcnow().hour
        time_factor = 1.0 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        for node in self.nodes.values():
            variation = np.random.normal(0, 0.02)
            node.current_load = float(np.clip(
                node.current_load * (1 + variation) * (0.97 + 0.03 * time_factor),
                0.3 * node.capacity if node.capacity > 0 else 0.0,
                0.95 * node.capacity if node.capacity > 0 else 1.0
            ))
            node.health = float(np.clip(node.health - 0.0001 + np.random.normal(0, 0.00005), 0.7, 1.0))
            load_ratio = (node.current_load / node.capacity) if node.capacity else 0.0
            health_factor = 1 - node.health
            dependency_risk = self._calculate_dependency_risk(node)
            node.risk_score = float(np.clip(
                0.4 * load_ratio + 0.3 * health_factor + 0.3 * dependency_risk,
                0, 1
            ))
        if threat and threat.get('active'):
            self._apply_threat(threat)
        if weather_data and weather_data.get('severity', 0) > 0.5:
            self._apply_weather(weather_data)
        return self.get_system_state()

    def _calculate_dependency_risk(self, node: InfrastructureNode) -> float:
        if not node.dependencies:
            return 0.0
        dependency_risks = [self.nodes[dep_id].risk_score for dep_id in node.dependencies if dep_id in self.nodes]
        return float(np.mean(dependency_risks)) if dependency_risks else 0.0

    def _apply_threat(self, threat: Dict):
        threat_type = threat.get('type')
        target = threat.get('target')
        severity = threat.get('severity', 0.5)
        if target not in self.nodes:
            return
        target_node = self.nodes[target]
        if threat_type == ThreatType.CYBER_ATTACK.value:
            target_node.health = max(0.0, target_node.health * (1 - severity * 0.3))
            target_node.current_load = min(target_node.capacity * 1.5 if target_node.capacity else target_node.current_load * 2, target_node.current_load * (1 + severity * 0.5))
        elif threat_type == ThreatType.OVERLOAD.value:
            target_node.current_load = min(target_node.capacity * (1 + severity * 0.4) if target_node.capacity else target_node.current_load * 1.3, target_node.capacity * 1.3 if target_node.capacity else target_node.current_load * 1.3)
        elif threat_type == ThreatType.PHYSICAL_DAMAGE.value:
            target_node.health = max(0.0, target_node.health * (1 - severity * 0.6))
            target_node.current_load = max(0.0, target_node.current_load * (1 - severity * 0.3))
        elif threat_type == ThreatType.ENVIRONMENTAL.value:
            for node in self.nodes.values():
                if node.type in ['power', 'transport']:
                    node.health = max(0.0, node.health * (1 - severity * 0.2))
        elif threat_type == ThreatType.SUPPLY_CHAIN.value:
            target_node.capacity = max(1.0, target_node.capacity * (1 - severity * 0.15))

    def _apply_weather(self, weather_data: Dict):
        severity = weather_data.get('severity', 0)
        for node in self.nodes.values():
            if node.type == 'power':
                node.health = max(0.0, node.health * (1 - severity * 0.15))
            elif node.type == 'transport':
                node.current_load = min(node.capacity if node.capacity else node.current_load * 2, node.current_load * (1 + severity * 0.3))

    def simulate_cascade(self, origin_node_id: str, severity: float = 0.8) -> CascadeEvent:
        affected = [origin_node_id]
        if origin_node_id in self.nodes:
            self.nodes[origin_node_id].health = max(0.0, self.nodes[origin_node_id].health * 0.3)
            self.nodes[origin_node_id].risk_score = 0.95
        wave = [origin_node_id]
        iterations = 0
        while wave and iterations < 5:  # Note: Dependencies seem acyclic; if cycles possible, add visited set
            next_wave = []
            for node_id in wave:
                for dependent_id, deps in self.dependencies.items():
                    if node_id in deps and dependent_id not in affected:
                        num_deps = len([d for d in deps if d in self.nodes]) or 1
                        cascade_prob = severity * (1.0 / num_deps)
                        if np.random.random() < cascade_prob:
                            self.nodes[dependent_id].health = max(0.0, self.nodes[dependent_id].health * 0.6)
                            self.nodes[dependent_id].risk_score = 0.8
                            affected.append(dependent_id)
                            next_wave.append(dependent_id)
            wave = next_wave
            iterations += 1
        total_capacity = sum(n.capacity for n in self.nodes.values()) or 1.0
        impact = sum(self.nodes[nid].capacity for nid in affected if nid in self.nodes) / total_capacity
        return CascadeEvent(
            timestamp=datetime.utcnow(),
            origin_node=origin_node_id,
            affected_nodes=affected,
            impact_score=impact,
            threat_type=ThreatType.CASCADE
        )

    def auto_mitigate(self, node_id: str) -> Dict:
        if node_id not in self.nodes:
            return {'success': False, 'actions': [], 'node': 'Unknown'}
        node = self.nodes[node_id]
        actions = []
        if node.capacity and (node.current_load / node.capacity) > 0.85:
            node.current_load = max(0.0, node.current_load * 0.7)
            actions.append('Load shedding activated')
        if node.health < 0.6:
            node.health = min(1.0, node.health + 0.25)
            actions.append('Backup systems deployed')
        if node.risk_score > Config.AUTO_MITIGATION_THRESHOLD:
            node.risk_score = max(0.0, node.risk_score * 0.8)
            actions.append('Emergency protocols engaged')
        return {'success': len(actions) > 0, 'actions': actions, 'node': node.name}

    def get_system_state(self) -> Dict:
        total_capacity = sum(n.capacity for n in self.nodes.values()) or 1.0
        total_load = sum(n.current_load for n in self.nodes.values())
        avg_health = float(np.mean([n.health for n in self.nodes.values()])) if self.nodes else 1.0
        max_risk = max((n.risk_score for n in self.nodes.values()), default=0.0)
        critical_nodes = [n.id for n in self.nodes.values() if n.risk_score > Config.CRITICAL_THRESHOLD]
        warning_nodes = [n.id for n in self.nodes.values() if Config.WARNING_THRESHOLD < n.risk_score <= Config.CRITICAL_THRESHOLD]
        return {
            'total_capacity': total_capacity,
            'total_load': total_load,
            'load_ratio': total_load / total_capacity,
            'avg_health': avg_health,
            'max_risk': max_risk,
            'critical_nodes': critical_nodes,
            'warning_nodes': warning_nodes,
            'node_count': len(self.nodes)
        }

# ============================================================================
# AI PREDICTION ENGINE
# ============================================================================
class PredictionEngine:
    @staticmethod
    def predict_failure_probability(node: InfrastructureNode, horizon_hours: int = 24) -> float:
        base_risk = node.risk_score
        health_factor = 1 - node.health
        load_stress = 0.0
        if node.capacity:
            load_stress = max(0, (node.current_load / node.capacity) - 0.7) / 0.3
        time_factor = 1 + (horizon_hours / 24) * 0.3
        dep_factor = 1.0 + len(node.dependencies) * 0.1
        probability = float(np.clip(
            (base_risk * 0.4 + health_factor * 0.3 + load_stress * 0.3) * time_factor * dep_factor,
            0, 1
        ))
        return probability

    @staticmethod
    def recommend_action(node: InfrastructureNode, system_state: Dict) -> Dict:
        recommendations = []
        priority = "LOW"
        urgency_score = 0
        if node.risk_score > Config.CRITICAL_THRESHOLD:
            priority = "CRITICAL"
            urgency_score = 10
            recommendations.append({'action': f"IMMEDIATE: Reduce load on {node.name} by 30-40%", 'type': 'load_reduction'})
            recommendations.append({'action': f"Deploy backup systems and redundant pathways", 'type': 'redundancy'})
            if node.dependencies:
                recommendations.append({'action': f"Alert {len(node.dependencies)} dependent systems", 'type': 'cascade_prevention'})
        elif node.risk_score > Config.WARNING_THRESHOLD:
            priority = "WARNING"
            urgency_score = 6
            recommendations.append({'action': f"Schedule maintenance for {node.name}", 'type': 'maintenance'})
            recommendations.append({'action': f"Monitor load trends", 'type': 'monitoring'})
        else:
            priority = "NORMAL"
            urgency_score = 2
            recommendations.append({'action': f"Continue routine monitoring", 'type': 'monitoring'})
        if node.health < 0.8:
            recommendations.append({'action': f"Inspect equipment at {node.name}", 'type': 'inspection'})
            urgency_score += 3
        ttf_hours = max(1, int((1 - node.risk_score) * 48 / (node.risk_score + 0.1))) if node.risk_score > 0.5 else 72
        return {
            'node_id': node.id,
            'node_name': node.name,
            'priority': priority,
            'urgency_score': urgency_score,
            'recommendations': recommendations,
            'estimated_ttf_hours': ttf_hours,
            'failure_probability_24h': PredictionEngine.predict_failure_probability(node, 24)
        }

# ============================================================================
# UTILITIES
# ============================================================================
def fetch_weather_data() -> Dict:
    severity = np.clip(np.random.normal(0.3, 0.2), 0.0, 1.0)  # Fix: Clipped to [0,1]
    conditions = ['Clear', 'Cloudy', 'Rain', 'Storm', 'Severe Storm']
    weights = [0.5, 0.25, 0.15, 0.07, 0.03]
    condition = np.random.choice(conditions, p=weights)
    if condition in ['Storm', 'Severe Storm']:
        severity = min(1.0, severity + 0.4)
    return {
        'condition': condition,
        'severity': severity,
        'temperature': np.random.uniform(15, 30),
        'wind_speed': severity * 80,
        'timestamp': datetime.utcnow()
    }

class BlockchainAuditor:
    @staticmethod
    def commit_event(event_type: str, data: Dict) -> Dict:
        event_str = json.dumps(data, sort_keys=True, default=str)
        event_hash = hashlib.sha256(event_str.encode()).hexdigest()
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'hash': event_hash[:16],
            'data_summary': {
                'affected_nodes': len(data.get('affected_nodes', [])) if 'affected_nodes' in data else 1,
                'severity': data.get('severity', 'unknown'),
                'auto_mitigated': data.get('auto_mitigated', False)
            }
        }

# ============================================================================
# VISUALIZATIONS
# ============================================================================
@st.cache_data  # Fix: Cached for performance
def create_network_map(_simulator: InfrastructureSimulator, highlight_cascade: List[str] = None) -> go.Figure:
    fig = go.Figure()
    edge_groups = {'critical': [], 'warning': [], 'normal': []}
    for node_id, deps in _simulator.dependencies.items():
        if node_id in _simulator.nodes:
            node = _simulator.nodes[node_id]
            for dep_id in deps:
                if dep_id in _simulator.nodes:
                    dep = _simulator.nodes[dep_id]
                    risk = max(node.risk_score, dep.risk_score)
                    if highlight_cascade and node_id in highlight_cascade and dep_id in highlight_cascade:
                        edge_groups['critical'].append((node.lon, dep.lon, node.lat, dep.lat))
                    elif risk > Config.CRITICAL_THRESHOLD:
                        edge_groups['critical'].append((node.lon, dep.lon, node.lat, dep.lat))
                    elif risk > Config.WARNING_THRESHOLD:
                        edge_groups['warning'].append((node.lon, dep.lon, node.lat, dep.lat))
                    else:
                        edge_groups['normal'].append((node.lon, dep.lon, node.lat, dep.lat))
    edge_colors = {'critical': 'rgba(255, 107, 107, 0.8)', 'warning': 'rgba(254, 202, 87, 0.6)', 'normal': 'rgba(100, 100, 100, 0.3)'}
    edge_widths = {'critical': 3, 'warning': 2, 'normal': 1}
    for level, edges in edge_groups.items():
        if not edges:
            continue
        lons, lats = [], []
        for lon1, lon2, lat1, lat2 in edges:
            lons.extend([lon1, lon2, None])
            lats.extend([lat1, lat2, None])
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats,
            mode='lines',
            line=dict(width=edge_widths[level], color=edge_colors[level]),
            showlegend=False,
            hoverinfo='skip'
        ))
    for infra_type, config in Config.INFRASTRUCTURE_TYPES.items():
        type_nodes = [n for n in _simulator.nodes.values() if n.type == infra_type]
        if not type_nodes:
            continue
        lons = [n.lon for n in type_nodes]
        lats = [n.lat for n in type_nodes]
        sizes = [max(15, min(50, n.capacity / 50)) for n in type_nodes]
        colors = []
        for n in type_nodes:
            if highlight_cascade and n.id in highlight_cascade:
                colors.append('#ff0000')
            elif n.risk_score > Config.CRITICAL_THRESHOLD:
                colors.append('#ff6b6b')
            elif n.risk_score > Config.WARNING_THRESHOLD:
                colors.append('#feca57')
            else:
                colors.append(config['color'])
        texts = [f"<b>{n.name}</b><br>Load: {n.current_load/n.capacity*100 if n.capacity else 0:.1f}%<br>Health: {n.health*100:.0f}%<br>Risk: {n.risk_score:.3f}" for n in type_nodes]
        fig.add_trace(go.Scattergeo(
            lon=lons, lat=lats,
            mode='markers+text',
            marker=dict(size=sizes, color=colors, line=dict(width=2, color='rgba(255,255,255,0.8)'), sizemode='diameter'),
            text=[config['icon']] * len(type_nodes),
            textfont=dict(size=14),
            textposition="middle center",
            hovertext=texts,
            hoverinfo='text',
            name=config['name'],
            showlegend=True
        ))
    fig.update_geos(
        center=dict(lat=50.1109, lon=8.6821),
        projection_scale=80,
        showcountries=False,
        showland=True,
        landcolor='rgb(15, 20, 40)',
        showocean=False,
        bgcolor='rgba(0,0,0,0)'
    )
    fig.update_layout(
        title={'text': "Critical Infrastructure Network", 'font': {'size': 20, 'color': 'white'}, 'x': 0.5, 'xanchor': 'center'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5, bgcolor='rgba(13, 17, 38, 0.8)')
    )
    return fig

@st.cache_data  # Fix: Cached for performance
def create_risk_timeline(_history: List[Dict], events: List[Dict] = None) -> go.Figure:
    if not _history or len(_history) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Start simulation to see timeline", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='white'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600)
        return fig
    df = pd.DataFrame(_history)
    fig = make_subplots(rows=3, cols=1, subplot_titles=("System Risk Score", "Load vs Capacity", "System Health"), vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['max_risk'], mode='lines', name='Max Risk', line=dict(color='#ff6b6b', width=3), fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.2)'), row=1, col=1)
    fig.add_hline(y=Config.CRITICAL_THRESHOLD, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=Config.WARNING_THRESHOLD, line_dash="dash", line_color="orange", row=1, col=1)
    if events:
        for event in events[-5:] if len(events) >= 5 else events:  # Fix: Handle short lists without error
            try:
                event_time = event.get('timestamp')
                if isinstance(event_time, str):
                    event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                fig.add_vline(x=event_time, line_dash="dot", line_color="cyan", annotation_text=event.get('type', '')[:10], row=1, col=1)
            except Exception as e:
                st.error(f"Timeline event error: {e}")  # Fix: Log errors without crashing
                pass
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['load_ratio'], mode='lines', name='Load Ratio', line=dict(color='#48dbfb', width=2), fill='tozeroy', fillcolor='rgba(72, 219, 251, 0.2)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['avg_health'], mode='lines', name='Avg Health', line=dict(color='#1dd1a1', width=2), fill='tozeroy', fillcolor='rgba(29, 209, 161, 0.2)'), row=3, col=1)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 1])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=600, showlegend=False, hovermode='x unified')
    return fig

@st.cache_data  # Fix: Cached for performance
def create_cascade_viz(_cascade_event: CascadeEvent, _simulator: InfrastructureSimulator) -> go.Figure:
    affected_nodes = [_simulator.nodes[nid] for nid in _cascade_event.affected_nodes if nid in _simulator.nodes]
    if not affected_nodes:
        fig = go.Figure()
        fig.add_annotation(text="No cascade data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='white'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
        return fig
    n = len(affected_nodes)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = [0 if i == 0 else 1 + (i-1) * 0.3 for i in range(n)]
    x = [r * np.cos(a) for r, a in zip(radii, angles)]
    y = [r * np.sin(a) for r, a in zip(radii, angles)]
    fig = go.Figure()
    for i in range(len(affected_nodes) - 1):
        fig.add_trace(go.Scatter(x=[x[0], x[i+1]], y=[y[0], y[i+1]], mode='lines', line=dict(color=f'rgba(255, 107, 107, {0.8 - i*0.1})', width=3), showlegend=False, hoverinfo='skip'))
    colors = ['#ff6b6b'] + ['#feca57' if i < 3 else '#48dbfb' for i in range(1, n)]
    sizes = [60] + [40 if i < 3 else 30 for i in range(1, n)]
    node_types = [Config.INFRASTRUCTURE_TYPES.get(node.type, {}).get('icon', '●') for node in affected_nodes]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers+text',
        marker=dict(size=sizes, color=colors, line=dict(width=3, color='white')),
        text=node_types, textfont=dict(size=16), textposition='middle center',
        hovertext=[f"<b>{node.name}</b><br>Wave: {i+1}<br>Risk: {node.risk_score:.2f}" for i, node in enumerate(affected_nodes)],
        hoverinfo='text', showlegend=False
    ))
    fig.update_xaxes(visible=False, range=[-3, 3])
    fig.update_yaxes(visible=False, range=[-3, 3])
    fig.update_layout(title=f"Cascade: {len(affected_nodes)} Nodes ({_cascade_event.impact_score*100:.1f}% Impact)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=500, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ============================================================================
# SESSION STATE
# ============================================================================
def init_state():
    if 'initialized' not in st.session_state:
        reset_simulation()
        st.session_state.initialized = True

def reset_simulation():
    st.session_state.simulator = InfrastructureSimulator()
    st.session_state.predictor = PredictionEngine()
    st.session_state.auditor = BlockchainAuditor()
    st.session_state.history = []
    st.session_state.cascade_events = []
    st.session_state.audit_log = []
    st.session_state.mitigation_log = []
    st.session_state.active_threat = None
    st.session_state.weather_data = fetch_weather_data()
    st.session_state.is_running = False
    st.session_state.last_update = datetime.utcnow()
    st.session_state.auto_mitigation = True
    st.session_state.selected_node = None
    st.session_state.update_counter = 0
    st.session_state.time_acceleration = 1.0

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    init_state()
    # Auto-update with proper sleep
    if st.session_state.is_running:
        now = datetime.utcnow()
        elapsed = (now - st.session_state.last_update).total_seconds()
        effective_rate = max(0.5, Config.SAMPLE_RATE / max(0.1, st.session_state.time_acceleration))
        if elapsed >= effective_rate:
            if st.session_state.update_counter % 30 == 0:
                st.session_state.weather_data = fetch_weather_data()  # TODO: Integrate real API if needed
            system_state = st.session_state.simulator.update_state(st.session_state.active_threat, st.session_state.weather_data)
            system_state['timestamp'] = now
            st.session_state.history.append(system_state)
            if len(st.session_state.history) > Config.MAX_HISTORY:
                st.session_state.history = st.session_state.history[-Config.MAX_HISTORY:]
            if st.session_state.auto_mitigation:
                for node_id in system_state['critical_nodes']:
                    result = st.session_state.simulator.auto_mitigate(node_id)
                    if result['success']:
                        mitigation_event = {'timestamp': now, 'node_id': node_id, 'node_name': result['node'], 'actions': result['actions']}
                        st.session_state.mitigation_log.append(mitigation_event)
                        block = st.session_state.auditor.commit_event('auto_mitigation', mitigation_event)
                        st.session_state.audit_log.append(block)
            if st.session_state.active_threat and now > st.session_state.active_threat.get('until', now):
                st.session_state.active_threat = None
            st.session_state.last_update = now
            st.session_state.update_counter += 1
            # Sleep to prevent CPU overload
            time.sleep(0.5)
            st.rerun()
    system_state = st.session_state.simulator.get_system_state()
    # HEADER
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(Config.LOGO_URL, width=180)
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 1rem;'><h1 style='margin: 0;'>🛡️ Sentinel Grid</h1><p style='margin: 0; opacity: 0.8;'>Digital Immune System for Critical Infrastructure</p></div>", unsafe_allow_html=True)
    with col3:
        status_text = "OPERATIONAL" if st.session_state.is_running else "STANDBY"
        status_color = "#1dd1a1" if st.session_state.is_running else "#a29bfe"
        st.markdown(f"<div style='text-align: right; padding-top: 2rem;'><div style='display: inline-block; padding: 0.75rem 1.5rem; background: {status_color}20; border: 2px solid {status_color}; border-radius: 20px; color: {status_color}; font-weight: 700;'>● {status_text}</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='status-card'>", unsafe_allow_html=True)
        risk_class = "risk-critical" if system_state['max_risk'] > Config.CRITICAL_THRESHOLD else "risk-warning" if system_state['max_risk'] > Config.WARNING_THRESHOLD else "risk-normal"
        st.markdown(f"<h3>Risk Level</h3><div class='metric-large {risk_class}'>{system_state['max_risk']:.3f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='status-card'><h3>System Health</h3><div class='metric-large'>{:.1f}%</div></div>".format(system_state['avg_health']*100), unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='status-card'><h3>Capacity Usage</h3><div class='metric-large'>{:.1f}%</div></div>".format(system_state['load_ratio']*100), unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='status-card'><h3>Critical Nodes</h3><div class='metric-large'>{len(system_state['critical_nodes'])}/{system_state['node_count']}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # MAIN LAYOUT
    left_col, main_col, right_col = st.columns([1.2, 2.5, 1.3])
    # LEFT: CONTROLS
    with left_col:
        st.markdown("### 🎮 System Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Start", key="btn_start", use_container_width=True, disabled=st.session_state.is_running):
                st.session_state.is_running = True
                block = st.session_state.auditor.commit_event('system_start', {'timestamp': datetime.utcnow().isoformat()})
                st.session_state.audit_log.append(block)
                st.rerun()
        with c2:
            if st.button("⏸️ Stop", key="btn_stop", use_container_width=True, disabled=not st.session_state.is_running):
                st.session_state.is_running = False
                block = st.session_state.auditor.commit_event('system_stop', {'timestamp': datetime.utcnow().isoformat()})
                st.session_state.audit_log.append(block)
                st.rerun()
        if st.button("⏭️ Manual Step", key="btn_manual_step", use_container_width=True, disabled=st.session_state.is_running):
            system_state = st.session_state.simulator.update_state(st.session_state.active_threat, st.session_state.weather_data)
            system_state['timestamp'] = datetime.utcnow()
            st.session_state.history.append(system_state)
            st.rerun()
        if st.button("🔄 Reset Simulation", key="btn_reset", use_container_width=True):
            reset_simulation()
            st.rerun()  # Fix: Added full reset button
        auto_enabled = st.checkbox("🤖 Auto-Mitigation", value=st.session_state.auto_mitigation, key='chk_auto_mitigation')
        if auto_enabled != st.session_state.auto_mitigation:
            st.session_state.auto_mitigation = auto_enabled
        st.session_state.time_acceleration = st.slider("⏩ Time Acceleration", 0.5, 5.0, 1.0, 0.5, key="slider_acceleration")
        st.markdown("---")
        weather = st.session_state.weather_data
        weather_icon = {'Clear': '☀️', 'Cloudy': '☁️', 'Rain': '🌧️', 'Storm': '⛈️', 'Severe Storm': '🌪️'}.get(weather['condition'], '🌤️')
        st.markdown(f"**{weather_icon} Weather:** {weather['condition']}")
        st.progress(weather['severity'])
        st.caption(f"Wind: {weather['wind_speed']:.0f} km/h | Temp: {weather['temperature']:.1f}°C")
        st.markdown("---")
        st.markdown("### ⚡ Quick Scenarios")
        for preset_name, preset_config in Config.SCENARIO_PRESETS.items():
            if st.button(preset_name, key=f"preset_{preset_name.replace(' ', '_')}", use_container_width=True):
                if preset_config.get('threats'):
                    threat = preset_config['threats'][0]
                    st.session_state.active_threat = {
                        'type': threat['type'],
                        'target': threat['target'],
                        'severity': threat['severity'],
                        'active': True,
                        'until': datetime.utcnow() + timedelta(seconds=threat['duration'])
                    }
                if preset_config.get('weather'):
                    st.session_state.weather_data.update(preset_config['weather'])
                st.success(f"✓ {preset_name} activated")
                st.rerun()
        st.markdown("---")
        st.markdown("### ⚠️ Custom Threat")
        threat_options = {'Cyber Attack': ThreatType.CYBER_ATTACK.value, 'Physical Damage': ThreatType.PHYSICAL_DAMAGE.value, 'Overload': ThreatType.OVERLOAD.value, 'Environmental': ThreatType.ENVIRONMENTAL.value, 'Supply Chain': ThreatType.SUPPLY_CHAIN.value}
        threat_name = st.selectbox("Threat Type", list(threat_options.keys()), key="select_threat_type")
        threat_type = threat_options[threat_name]
        node_options = {f"{n.name} ({n.type})": n.id for n in st.session_state.simulator.nodes.values()}
        target_name = st.selectbox("Target Node", list(node_options.keys()), key="select_target_node")
        target_id = node_options[target_name]
        severity = st.slider("Severity", 0.1, 1.0, 0.5, 0.1, key="slider_severity")
        duration = st.number_input("Duration (sec)", 10, 180, 30, 10, key="input_duration")
        if duration < 1:  # Fix: Input validation
            st.error("Duration must be at least 1 second.")
        elif st.button("🚨 Deploy Threat", key="btn_deploy_threat", use_container_width=True):
            st.session_state.active_threat = {'type': threat_type, 'target': target_id, 'severity': severity, 'active': True, 'until': datetime.utcnow() + timedelta(seconds=duration)}
            event_data = {'threat_type': threat_name, 'target': target_id, 'target_name': target_name, 'severity': severity, 'duration': duration}
            block = st.session_state.auditor.commit_event('threat_deployment', event_data)
            st.session_state.audit_log.append(block)
            st.success(f"✓ {threat_name} deployed")
        if st.session_state.active_threat:
            time_left = (st.session_state.active_threat['until'] - datetime.utcnow()).total_seconds()
            if time_left > 0:
                st.markdown(f"<div class='threat-alert'><strong>🔥 ACTIVE THREAT</strong><br>{st.session_state.active_threat['type'].replace('_', ' ').title()}<br>⏱️ {int(time_left)}s remaining</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🌊 Cascade Simulator")
        cascade_origin = st.selectbox("Origin Node", list(node_options.keys()), key='select_cascade_origin')
        cascade_severity = st.slider("Cascade Severity", 0.5, 1.0, 0.8, 0.1, key='slider_cascade_severity')
        if st.button("💥 Simulate Cascade", key="btn_simulate_cascade", use_container_width=True):
            origin_id = node_options[cascade_origin]
            cascade = st.session_state.simulator.simulate_cascade(origin_id, cascade_severity)
            st.session_state.cascade_events.append(cascade)
            event_data = cascade.to_dict()
            block = st.session_state.auditor.commit_event('cascade_event', event_data)
            st.session_state.audit_log.append(block)
            st.warning(f"⚠️ {len(cascade.affected_nodes)} nodes affected!")
        st.markdown("---")
        if st.button("📥 Export Data", key="btn_export", use_container_width=True):
            export_data = {
                'system_state': system_state,
                'history': [{**h, 'timestamp': h['timestamp'].isoformat() if isinstance(h['timestamp'], datetime) else str(h['timestamp'])} for h in st.session_state.history[-100:]],
                'audit_log': st.session_state.audit_log,
                'mitigation_log': st.session_state.mitigation_log,
                'cascades': [c.to_dict() for c in st.session_state.cascade_events],
                'nodes': [n.to_dict() for n in st.session_state.simulator.nodes.values()]
            }
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button("Download JSON", json_str, "sentinel_grid_export.json", "application/json", key="download_json", use_container_width=True)
    # MAIN: VISUALIZATIONS
    with main_col:
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Network", "📈 Timeline", "🌊 Cascades", "🔍 Details"])
        with tab1:
            highlight = st.session_state.cascade_events[-1].affected_nodes if st.session_state.cascade_events else None
            fig_map = create_network_map(st.session_state.simulator, highlight)
            st.plotly_chart(fig_map, use_container_width=True)
            if system_state['critical_nodes']:
                st.warning(f"⚠️ {len(system_state['critical_nodes'])} Critical Nodes")
                cols = st.columns(min(3, len(system_state['critical_nodes'])))
                for i, node_id in enumerate(system_state['critical_nodes'][:3]):
                    node = st.session_state.simulator.nodes[node_id]
                    with cols[i]:
                        st.markdown(f"**{node.name}**")
                        st.progress(node.risk_score)
                        st.caption(f"Risk: {node.risk_score:.2f}")
        with tab2:
            fig_timeline = create_risk_timeline(st.session_state.history, st.session_state.audit_log)
            st.plotly_chart(fig_timeline, use_container_width=True)
            if st.session_state.history:
                recent = st.session_state.history[-50:]
                avg_risk = float(np.mean([h['max_risk'] for h in recent]))
                avg_load = float(np.mean([h['load_ratio'] for h in recent]))
                s1, s2, s3 = st.columns(3)
                s1.metric("Avg Risk (50pt)", f"{avg_risk:.3f}")
                s2.metric("Avg Load (50pt)", f"{avg_load*100:.1f}%")
                s3.metric("Data Points", len(st.session_state.history))
        with tab3:
            if st.session_state.cascade_events:
                cascade = st.session_state.cascade_events[-1]
                fig_cascade = create_cascade_viz(cascade, st.session_state.simulator)
                st.plotly_chart(fig_cascade, use_container_width=True)
                st.markdown(f"**Origin:** {cascade.origin_node} | **Affected:** {len(cascade.affected_nodes)} nodes | **Impact:** {cascade.impact_score*100:.1f}% | **Time:** {cascade.timestamp.strftime('%H:%M:%S')}")
                with st.expander("Affected Nodes Details"):
                    for node_id in cascade.affected_nodes:
                        if node_id in st.session_state.simulator.nodes:
                            node = st.session_state.simulator.nodes[node_id]
                            st.markdown(f"- **{node.name}** (Risk: {node.risk_score:.2f}, Health: {node.health*100:.0f}%)")
            else:
                st.info("🌊 Use cascade simulator to visualize failure propagation")
        with tab4:
            node_select = st.selectbox("Select Node", list(node_options.keys()), key='select_detail_node')
            node_id = node_options[node_select]
            node = st.session_state.simulator.nodes[node_id]
            d1, d2, d3 = st.columns(3)
            d1.metric("Load", f"{node.current_load:.0f} MW", f"{(node.current_load/node.capacity*100):.1f}%" if node.capacity else "N/A")
            d2.metric("Health", f"{node.health*100:.1f}%")
            d3.metric("Risk Score", f"{node.risk_score:.3f}")
            fig_gauges = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number", value=node.current_load, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Load (MW)"},
                gauge={'axis': {'range': [0, node.capacity if node.capacity else 1]}, 'bar': {'color': "#48dbfb"}, 'steps': [{'range': [0, node.capacity*0.7 if node.capacity else 1], 'color': "rgba(29, 209, 161, 0.3)"}, {'range': [node.capacity*0.7 if node.capacity else 1, node.capacity*0.85 if node.capacity else 1], 'color': "rgba(254, 202, 87, 0.3)"}, {'range': [node.capacity*0.85 if node.capacity else 1, node.capacity if node.capacity else 1], 'color': "rgba(255, 107, 107, 0.3)"}]}
            ), row=1, col=1)
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number", value=node.health * 100, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Health (%)"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1dd1a1"}, 'steps': [{'range': [0, 60], 'color': "rgba(255, 107, 107, 0.3)"}, {'range': [60, 80], 'color': "rgba(254, 202, 87, 0.3)"}, {'range': [80, 100], 'color': "rgba(29, 209, 161, 0.3)"}]}
            ), row=1, col=2)
            fig_gauges.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300)
            st.plotly_chart(fig_gauges, use_container_width=True)
            if node.dependencies:
                st.markdown("**Dependencies:**")
                for dep_id in node.dependencies:
                    if dep_id in st.session_state.simulator.nodes:
                        dep = st.session_state.simulator.nodes[dep_id]
                        dep_status = "🔴" if dep.risk_score > 0.7 else "🟡" if dep.risk_score > 0.5 else "🟢"
                        st.markdown(f"{dep_status} {dep.name} (Risk: {dep.risk_score:.2f})")
    # RIGHT: AI RECOMMENDATIONS & LOGS
    with right_col:
        st.markdown("### 🤖 AI Recommendations")
        recommendations = []
        for node_id in system_state['critical_nodes'][:3]:
            node = st.session_state.simulator.nodes[node_id]
            rec = st.session_state.predictor.recommend_action(node, system_state)
            recommendations.append(rec)
        if recommendations:
            for rec in recommendations:
                priority_color = "#ff6b6b" if rec['priority'] == "CRITICAL" else "#feca57"
                st.markdown(f"<div class='recommendation-card'><strong>{rec['node_name']}</strong><br><span style='color: {priority_color};'>{rec['priority']}</span><br><small>TTF: {rec['estimated_ttf_hours']}h | Fail Prob: {rec['failure_probability_24h']*100:.0f}%</small></div>", unsafe_allow_html=True)
                with st.expander("Actions", expanded=False):
                    for action in rec['recommendations']:
                        st.markdown(f"• {action['action']}")
        else:
            st.success("✅ All systems nominal")
        st.markdown("---")
        st.markdown("### 🛡️ Mitigation Log")
        if st.session_state.mitigation_log:
            for event in st.session_state.mitigation_log[-5:]:  # Fix: Limited to prevent overflow
                st.markdown(f"<div style='background: rgba(29, 209, 161, 0.1); padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #1dd1a1;'><strong>{event['node_name']}</strong><br><small>{event['timestamp'].strftime('%H:%M:%S')}</small><br>{'<br>'.join(event['actions'])}</div>", unsafe_allow_html=True)
        else:
            st.info("No mitigations yet")
        st.markdown("---")
        st.markdown("### 📜 Blockchain Audit")
        if st.session_state.audit_log:
            for block in st.session_state.audit_log[-5:]:
                st.markdown(f"<div class='timeline-event'><strong>{block['type'].replace('_', ' ').title()}</strong><br><small>Hash: {block['hash']}</small><br><small>{block['timestamp'][:19]}</small></div>", unsafe_allow_html=True)
        else:
            st.info("No audit events")
    st.markdown("---")
    st.markdown("<div style='text-align: center; opacity: 0.6;'>Sentinel Grid v2.2.1 - Fixed Issues Edition · Monza Tech LLC</div>", unsafe_allow_html=True)  # Fix: Completed footer

if __name__ == "__main__":
    main()