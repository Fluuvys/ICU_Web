import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN, Input, Dense, BatchNormalization, ReLU, Dropout, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

# =============================================================================
# MODEL ARCHITECTURE DEFINITIONS
# =============================================================================

class GRUDCell(Layer):
    def __init__(self, units, feature_dim, **kwargs):
        super(GRUDCell, self).__init__(**kwargs)
        self.units = units
        self.feature_dim = feature_dim
        self.state_size = units
        self.output_size = units

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            actual_feature_dim = input_shape[0][-1]
        else:
            actual_feature_dim = self.feature_dim
            
        self.W_z = self.add_weight(shape=(actual_feature_dim, self.units), name='W_z', initializer='glorot_uniform')
        self.U_z = self.add_weight(shape=(self.units, self.units), name='U_z', initializer='orthogonal')
        self.b_z = self.add_weight(shape=(self.units,), name='b_z', initializer='zeros')
        self.W_r = self.add_weight(shape=(actual_feature_dim, self.units), name='W_r', initializer='glorot_uniform')
        self.U_r = self.add_weight(shape=(self.units, self.units), name='U_r', initializer='orthogonal')
        self.b_r = self.add_weight(shape=(self.units,), name='b_r', initializer='zeros')
        self.W_h = self.add_weight(shape=(actual_feature_dim, self.units), name='W_h', initializer='glorot_uniform')
        self.U_h = self.add_weight(shape=(self.units, self.units), name='U_h', initializer='orthogonal')
        self.b_h = self.add_weight(shape=(self.units,), name='b_h', initializer='zeros')
        self.gamma_x_decay = self.add_weight(shape=(actual_feature_dim,), name='gamma_x_decay', initializer='ones')
        self.gamma_h_decay = self.add_weight(shape=(self.units,), name='gamma_h_decay', initializer='ones')
        self.mean_imputation = self.add_weight(shape=(actual_feature_dim,), name='mean_imputation', initializer='zeros')
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            x, m, delta_t = inputs
        else:
            x = inputs
            m = tf.ones_like(x)
            delta_t = tf.zeros((tf.shape(x)[0], 1))
        
        h_prev = states[0]
        
        if len(delta_t.shape) == 2 and delta_t.shape[-1] == 1:
            delta_t = tf.squeeze(delta_t, axis=-1)
        delta_t = tf.expand_dims(delta_t, axis=-1)
        
        gamma_x = tf.exp(-tf.maximum(0.0, self.gamma_x_decay) * delta_t)
        x_decayed = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.mean_imputation)
        
        gamma_h = tf.exp(-tf.maximum(0.0, self.gamma_h_decay) * delta_t)
        h_decayed = gamma_h * h_prev
        
        z = tf.sigmoid(K.dot(x_decayed, self.W_z) + K.dot(h_decayed, self.U_z) + self.b_z)
        r = tf.sigmoid(K.dot(x_decayed, self.W_r) + K.dot(h_decayed, self.U_r) + self.b_r)
        h_hat = tf.tanh(K.dot(x_decayed, self.W_h) + K.dot(r * h_decayed, self.U_h) + self.b_h)
        h_new = (1 - z) * h_decayed + z * h_hat
        return h_new, [h_new]

    def get_config(self):
        config = super(GRUDCell, self).get_config()
        config.update({'units': self.units, 'feature_dim': self.feature_dim})
        return config

class GRUDLayer(Layer):
    def __init__(self, units, **kwargs):
        super(GRUDLayer, self).__init__(**kwargs)
        self.units = units
        self.grud_cell = None
        
    def build(self, input_shape):
        values_shape, mask_shape, time_gaps_shape = input_shape
        feature_dim = values_shape[-1]
        self.grud_cell = GRUDCell(self.units, feature_dim)
        self.rnn = RNN(self.grud_cell, return_sequences=False)
        super().build(input_shape)
        
    def call(self, inputs):
        return self.rnn(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

def build_model(values_shape, mask_shape, time_gaps_shape, static_shape, l2_reg=0.01):
    values_input = Input(shape=values_shape, name='values')
    mask_input = Input(shape=mask_shape, name='mask')
    time_gaps_input = Input(shape=time_gaps_shape, name='time_gaps')
    static_input = Input(shape=static_shape, name='static')
    
    grud_inputs = (values_input, mask_input, time_gaps_input)
    grud_layer = GRUDLayer(64)(grud_inputs)
    
    static_x = Dense(32, kernel_regularizer=l2(l2_reg))(static_input)
    static_x = BatchNormalization()(static_x)
    static_x = ReLU()(static_x)
    static_x = Dropout(0.4)(static_x)
    
    combined_layer = Concatenate()([grud_layer, static_x])
    shared_dense = Dense(64, kernel_regularizer=l2(l2_reg))(combined_layer)
    shared_dense = BatchNormalization()(shared_dense)
    shared_dense = ReLU()(shared_dense)
    shared_dense = Dropout(0.5)(shared_dense)
    
    mortality_output = Dense(1, activation='sigmoid', name='mortality')(shared_dense)
    los_output = Dense(1, activation='linear', name='los')(shared_dense)
    
    model = Model(
        inputs={'values': values_input, 'mask': mask_input, 'time_gaps': time_gaps_input, 'static': static_input},
        outputs={'mortality': mortality_output, 'los': los_output}
    )
    return model

# =============================================================================
# CONSTANTS & CONFIGURATIONS
# =============================================================================

# FIX: This list MUST contain the 19 features the model expects, in the correct order.
# This ensures the input shape is always (None, 200, 19).
DYNAMIC_FEATURES_LIST = [
    'heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2', 
    'respiratory_rate', 'lactate', 'wbc', 'creatinine', 'glucose',
    'hemoglobin', 'platelet', 'sodium', 'potassium', 'bun', 
    'bicarbonate', 'ph', 'pco2', 'po2'
]

CLINICAL_RANGES = {
    'heart_rate': {'normal': (60, 100), 'unit': 'bpm', 'label': 'Heart Rate'},
    'systolic_bp': {'normal': (90, 140), 'unit': 'mmHg', 'label': 'Systolic BP'},
    'diastolic_bp': {'normal': (60, 90), 'unit': 'mmHg', 'label': 'Diastolic BP'},
    'temperature': {'normal': (36.5, 37.5), 'unit': '°C', 'label': 'Temperature'},
    'spo2': {'normal': (95, 100), 'unit': '%', 'label': 'SpO2'},
    'respiratory_rate': {'normal': (12, 20), 'unit': 'breaths/min', 'label': 'Resp. Rate'},
    'lactate': {'normal': (0.5, 2.0), 'unit': 'mmol/L', 'label': 'Lactate'},
    'wbc': {'normal': (4.0, 11.0), 'unit': '×10⁹/L', 'label': 'WBC Count'},
    'creatinine': {'normal': (0.6, 1.2), 'unit': 'mg/dL', 'label': 'Creatinine'},
    'glucose': {'normal': (70, 140), 'unit': 'mg/dL', 'label': 'Glucose'},
    'hemoglobin': {'normal': (12.0, 17.5), 'unit': 'g/dL', 'label': 'Hemoglobin'},
    'platelet': {'normal': (150, 450), 'unit': '×10⁹/L', 'label': 'Platelet Count'},
    'sodium': {'normal': (135, 145), 'unit': 'mEq/L', 'label': 'Sodium'},
    'potassium': {'normal': (3.5, 5.0), 'unit': 'mEq/L', 'label': 'Potassium'},
    'bun': {'normal': (7, 20), 'unit': 'mg/dL', 'label': 'BUN'},
    'bicarbonate': {'normal': (22, 29), 'unit': 'mEq/L', 'label': 'Bicarbonate'},
    'ph': {'normal': (7.35, 7.45), 'unit': '', 'label': 'pH'},
    'pco2': {'normal': (35, 45), 'unit': 'mmHg', 'label': 'pCO2'},
    'po2': {'normal': (80, 100), 'unit': 'mmHg', 'label': 'pO2'}
}


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_trained_model(model_path='better_model.h5'):
    try:
        shapes = {
            'values': (200, len(DYNAMIC_FEATURES_LIST)),
            'mask': (200, len(DYNAMIC_FEATURES_LIST)),
            'time_gaps': (200, 1),
            'static': (41,)
        }
        
        model = build_model(shapes['values'], shapes['mask'], shapes['time_gaps'], shapes['static'])
        model.load_weights(model_path)
        return model, True
    except Exception as e:
        st.warning(f"Could not load model: {e}. Using demo mode with simulated predictions.")
        return None, False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_risk_level(risk_score):
    if risk_score < 0.2:
        return "Low", "🟢", "#2ecc71"
    elif risk_score < 0.5:
        return "Moderate", "🟡", "#f39c12"
    else:
        return "High", "🔴", "#e74c3c"

def is_abnormal(value, feature_name):
    if feature_name in CLINICAL_RANGES:
        normal_range = CLINICAL_RANGES[feature_name]['normal']
        return value < normal_range[0] or value > normal_range[1]
    return False

def simulate_prediction(num_events, static_features=None):
    """Fallback simulation if model not loaded"""
    base_risk = 0.15
    event_factor = (num_events / 200) * 0.4
    noise = np.random.uniform(-0.05, 0.05)
    
    risk = min(max(base_risk + event_factor + noise, 0.05), 0.85)
    los = 2.0 + (num_events / 200) * 6.0 + np.random.uniform(-1, 1)
    
    return {'mortality': np.array([[risk]]), 'los': np.array([[max(1, los)]])}

def prepare_model_inputs(events_df, dynamic_features, static_features, max_events=200):
    """Prepare data for model prediction"""
    if len(events_df) == 0:
        return None
    
    events_df = events_df.sort_values('event_time').copy()
    
    mask = ~events_df[dynamic_features].isna().values
    values = events_df[dynamic_features].ffill().fillna(0).values
    
    time_diffs = events_df['event_time'].diff().dt.total_seconds() / 3600.0
    time_gaps = time_diffs.fillna(0).values.reshape(-1, 1)
    
    num_events = len(values)
    if num_events > max_events:
        values = values[:max_events]
        mask = mask[:max_events]
        time_gaps = time_gaps[:max_events]
    else:
        pad_len = max_events - num_events
        values = np.pad(values, ((0, pad_len), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_len), (0, 0)), 'constant')
        time_gaps = np.pad(time_gaps, ((0, pad_len), (0, 0)), 'constant')
    
    return {
        'values': np.expand_dims(values, 0),
        'mask': np.expand_dims(mask, 0),
        'time_gaps': np.expand_dims(time_gaps, 0),
        'static': np.expand_dims(static_features, 0)
    }

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = {
            'age': 70,
            'gender': 'M',
            'admission_type': 'Emergency'
        }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_risk_trajectory(predictions_history):
    if len(predictions_history) < 2:
        return None
    df = pd.DataFrame(predictions_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['risk'] * 100, mode='lines+markers', name='Mortality Risk',
        line=dict(color='#e74c3c', width=3), marker=dict(size=8)
    ))
    fig.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=20, y1=50, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    fig.update_layout(
        title="Risk Score Trajectory", xaxis_title="Time", yaxis_title="Mortality Risk (%)",
        yaxis_range=[0, 100], height=400, template="plotly_white"
    )
    return fig

def plot_feature_timeline(events_df, feature_name):
    if feature_name not in events_df.columns:
        return None
    data = events_df[['event_time', feature_name]].dropna()
    if len(data) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['event_time'], y=data[feature_name], mode='lines+markers',
        name=CLINICAL_RANGES.get(feature_name, {}).get('label', feature_name),
        line=dict(width=2), marker=dict(size=6)
    ))
    if feature_name in CLINICAL_RANGES:
        normal_range = CLINICAL_RANGES[feature_name]['normal']
        fig.add_hrect(
            y0=normal_range[0], y1=normal_range[1],
            fillcolor="green", opacity=0.1, line_width=0
        )
    fig.update_layout(
        title=f"{CLINICAL_RANGES.get(feature_name, {}).get('label', feature_name)} Timeline",
        xaxis_title="Time", yaxis_title=CLINICAL_RANGES.get(feature_name, {}).get('unit', ''),
        height=300, template="plotly_white"
    )
    return fig

def plot_feature_importance(current_values, dynamic_features):
    importance_scores, feature_labels = [], []
    for feat in dynamic_features:
        if feat in CLINICAL_RANGES and feat in current_values:
            value = current_values[feat]
            if pd.notna(value):
                abnormality = 0
                normal_range = CLINICAL_RANGES[feat]['normal']
                if value < normal_range[0]:
                    abnormality = (normal_range[0] - value) / normal_range[0]
                elif value > normal_range[1]:
                    abnormality = (value - normal_range[1]) / normal_range[1]
                if abnormality > 0:
                    importance_scores.append(min(abnormality * 100, 100))
                    feature_labels.append(CLINICAL_RANGES[feat]['label'])
    if not importance_scores:
        return None
    sorted_data = sorted(zip(importance_scores, feature_labels), reverse=True)[:5]
    scores, labels = zip(*sorted_data) if sorted_data else ([], [])
    fig = go.Figure(go.Bar(x=list(scores), y=list(labels), orientation='h', marker=dict(color='#3498db')))
    fig.update_layout(
        title="Top Contributing Factors (based on Abnormality)",
        xaxis_title="Abnormality Score", height=300, template="plotly_white"
    )
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="ICU Clinical Decision Support", layout="wide", page_icon="🏥")
    st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #3498db; }
    .high-risk { border-left-color: #e74c3c !important; background-color: #ffebee !important; }
    .moderate-risk { border-left-color: #f39c12 !important; background-color: #fff3e0 !important; }
    .low-risk { border-left-color: #2ecc71 !important; background-color: #e8f5e9 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    st.title("🏥 ICU Clinical Decision Support System")
    st.markdown("**Event-Based Real-Time Mortality & Length of Stay Prediction**")
    
    model, model_loaded = load_trained_model()
    
    if not model_loaded:
        st.info("📊 Demo Mode: Using simulated predictions for demonstration")
    
    with st.sidebar:
        st.header("Patient Information")
        age = st.number_input("Age", min_value=18, max_value=120, value=70)
        gender = st.selectbox("Gender", ["M", "F"])
        admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
        st.session_state.patient_info = {'age': age, 'gender': gender, 'admission_type': admission_type}
        
        st.markdown("---")
        st.header("Add Clinical Event")
        
        # FIX: Use all features for selection, not just the ones in CLINICAL_RANGES
        feature_options = [feat for feat in DYNAMIC_FEATURES_LIST if feat in CLINICAL_RANGES]
        selected_feature = st.selectbox("Select Measurement", feature_options, 
                                        format_func=lambda x: CLINICAL_RANGES[x]['label'])
        
        normal_range = CLINICAL_RANGES[selected_feature]['normal']
        st.caption(f"Normal range: {normal_range[0]}-{normal_range[1]} {CLINICAL_RANGES[selected_feature]['unit']}")
        value = st.number_input(
            f"Value ({CLINICAL_RANGES[selected_feature]['unit']})",
            min_value=0.0,
            value=float(np.mean(normal_range)),
            step=0.1
        )
        
        if st.button("➕ Add Measurement", type="primary", use_container_width=True):
            st.session_state.events.append({'event_time': datetime.now(), 'feature': selected_feature, 'value': value})
            st.success(f"Added {CLINICAL_RANGES[selected_feature]['label']}: {value}")
            st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Reset Patient", use_container_width=True):
            st.session_state.events, st.session_state.predictions = [], []
            st.rerun()
        
        st.markdown("---")
        st.caption(f"Total Events: {len(st.session_state.events)}")
        if st.session_state.events:
            st.caption(f"Time in ICU: {(datetime.now() - st.session_state.events[0]['event_time']).total_seconds() / 3600:.1f}h")
    
    if not st.session_state.events:
        st.info("👈 Add clinical measurements using the sidebar to begin monitoring")
        st.markdown("### Example Workflow")
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown("#### 1️⃣ Enter Patient Info")
        with col2: st.markdown("#### 2️⃣ Add Measurements")
        with col3: st.markdown("#### 3️⃣ Monitor Risk")
        return
    
    events_df = pd.DataFrame(st.session_state.events)
    events_pivot = events_df.pivot_table(index='event_time', columns='feature', values='value', aggfunc='last').reset_index()
    
    # FIX: Use the full DYNAMIC_FEATURES_LIST to ensure correct shape
    for col in DYNAMIC_FEATURES_LIST:
        if col not in events_pivot.columns:
            events_pivot[col] = np.nan
    events_pivot = events_pivot[['event_time'] + DYNAMIC_FEATURES_LIST]

    static_features = np.random.randn(41) 
    
    # FIX: Pass the correct full list of dynamic features
    model_input = prepare_model_inputs(events_pivot, DYNAMIC_FEATURES_LIST, static_features)
    
    if model_loaded and model_input:
        try:
            prediction = model.predict(model_input, verbose=0)
        except Exception as e:
            st.error(f"Model prediction failed: {e}. Falling back to simulation.")
            prediction = simulate_prediction(len(st.session_state.events))
    else:
        prediction = simulate_prediction(len(st.session_state.events))
    
    current_risk = float(prediction['mortality'][0][0])
    current_los = float(prediction['los'][0][0])
    
    if not st.session_state.predictions or st.session_state.predictions[-1]['num_events'] != len(st.session_state.events):
        st.session_state.predictions.append({
            'timestamp': datetime.now(), 'num_events': len(st.session_state.events),
            'risk': current_risk, 'los': current_los
        })
    
    risk_level, risk_icon, risk_color = get_risk_level(current_risk)
    
    st.markdown(f"""
    <div class="metric-card {risk_level.lower()}-risk">
        <h2>{risk_icon} Current Risk Assessment</h2>
        <h1 style="color: {risk_color}; margin: 10px 0;">{current_risk*100:.1f}%</h1>
        <p style="font-size: 18px; margin: 0;">Risk Level: <strong>{risk_level}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Predicted LOS", f"{current_los:.1f} days")
    with col2: st.metric("Events Processed", len(st.session_state.events))
    with col3:
        hours_in_icu = (datetime.now() - st.session_state.events[0]['event_time']).total_seconds() / 3600
        st.metric("Time in ICU", f"{hours_in_icu:.1f}h")
    with col4:
        if len(st.session_state.predictions) >= 2:
            risk_change = (st.session_state.predictions[-1]['risk'] - st.session_state.predictions[-2]['risk']) * 100
            st.metric("Risk Change", f"{risk_change:+.1f}%", delta_color="inverse" if risk_change > 0 else "normal")
        else:
            st.metric("Risk Change", "N/A")
    
    if current_risk > 0.5:
        st.error("🚨 **HIGH RISK ALERT**: Patient requires immediate clinical attention.")
    elif len(st.session_state.predictions) >= 2 and (st.session_state.predictions[-1]['risk'] - st.session_state.predictions[-2]['risk']) > 0.1:
        st.warning(f"⚠️ **Risk Increased by {(st.session_state.predictions[-1]['risk'] - st.session_state.predictions[-2]['risk'])*100:.0f}%**: Consider escalation of care.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Risk Trajectory", "🔬 Clinical Timeline", "📊 Contributing Factors", "💡 What-If Analysis"])
    
    with tab1:
        st.subheader("Risk Score Evolution")
        fig_risk = plot_risk_trajectory(st.session_state.predictions)
        if fig_risk: st.plotly_chart(fig_risk, use_container_width=True)
        else: st.info("Add more measurements to see the risk trajectory.")
    
    with tab2:
        st.subheader("Clinical Measurements Timeline")
        available_features = [f for f in DYNAMIC_FEATURES_LIST if f in events_pivot.columns and events_pivot[f].notna().any()]
        if available_features:
            selected_viz_feature = st.selectbox(
                "Select vital/lab to visualize", available_features,
                format_func=lambda x: CLINICAL_RANGES[x]['label'] if x in CLINICAL_RANGES else x
            )
            fig_timeline = plot_feature_timeline(events_pivot, selected_viz_feature)
            if fig_timeline: st.plotly_chart(fig_timeline, use_container_width=True)
        else: st.info("No measurements recorded yet.")
        st.subheader("Recent Events")
        recent_events = events_df.tail(10).copy()
        recent_events['feature_label'] = recent_events['feature'].map(lambda x: CLINICAL_RANGES.get(x, {}).get('label', x))
        recent_events['time'] = recent_events['event_time'].dt.strftime('%H:%M:%S')
        st.dataframe(recent_events[['time', 'feature_label', 'value']].rename(columns={'feature_label': 'Measurement'}).sort_values('time', ascending=False),
                     hide_index=True, use_container_width=True)
    
    with tab3:
        st.subheader("Key Contributing Factors")
        latest_values = events_pivot[DYNAMIC_FEATURES_LIST].ffill().iloc[-1].to_dict() if not events_pivot.empty else {}
        fig_importance = plot_feature_importance(latest_values, DYNAMIC_FEATURES_LIST)
        if fig_importance: st.plotly_chart(fig_importance, use_container_width=True)
        else: st.info("No abnormal values detected yet.")
        
        st.subheader("Current Abnormal Values")
        abnormal_found = False
        for feat in DYNAMIC_FEATURES_LIST:
            if feat in latest_values and pd.notna(latest_values[feat]) and feat in CLINICAL_RANGES:
                if is_abnormal(latest_values[feat], feat):
                    normal_range_str = CLINICAL_RANGES[feat]['normal']
                    st.warning(f"**{CLINICAL_RANGES[feat]['label']}**: {latest_values[feat]:.1f} {CLINICAL_RANGES[feat]['unit']} (Normal: {normal_range_str[0]}-{normal_range_str[1]})")
                    abnormal_found = True
        if not abnormal_found: st.success("✅ All monitored values are within normal ranges.")
    
    with tab4:
        st.subheader("What-If Scenario Analysis")
        available_whatif = [f for f in DYNAMIC_FEATURES_LIST if f in events_pivot.columns and events_pivot[f].notna().any() and f in CLINICAL_RANGES]
        if not available_whatif:
            st.info("Add measurements to enable 'What-If' analysis.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                whatif_feature = st.selectbox("Parameter to modify", available_whatif, format_func=lambda x: CLINICAL_RANGES[x]['label'], key="whatif")
                last_idx = events_pivot[whatif_feature].last_valid_index()
                current_val = events_pivot.loc[last_idx, whatif_feature] if last_idx is not None else 0
                normal_range_whatif = CLINICAL_RANGES[whatif_feature]['normal']
                simulated_value = st.slider(
                    f"Simulated {CLINICAL_RANGES[whatif_feature]['label']} ({CLINICAL_RANGES[whatif_feature]['unit']})",
                    min_value=float(normal_range_whatif[0] * 0.5), max_value=float(normal_range_whatif[1] * 1.5),
                    value=float(current_val), step=0.1
                )
            
            simulated_pivot = events_pivot.copy()
            if last_idx is not None:
                simulated_pivot.loc[last_idx, whatif_feature] = simulated_value

            simulated_input = prepare_model_inputs(simulated_pivot, DYNAMIC_FEATURES_LIST, static_features)
            
            if model_loaded and simulated_input:
                try: simulated_prediction = model.predict(simulated_input, verbose=0)
                except Exception: simulated_prediction = simulate_prediction(len(simulated_pivot))
            else: simulated_prediction = simulate_prediction(len(simulated_pivot))
            
            simulated_risk = float(simulated_prediction['mortality'][0][0])
            risk_delta = (simulated_risk - current_risk) * 100

            with col2:
                st.metric(
                    label="Simulated Mortality Risk", value=f"{simulated_risk*100:.1f}%",
                    delta=f"{risk_delta:+.1f}% vs Current",
                    delta_color="inverse" if risk_delta > 0 else "normal"
                )
                sim_level, sim_icon, _ = get_risk_level(simulated_risk)
                st.markdown(f"**Simulated Risk Level:** {sim_icon} {sim_level}")
                if abs(risk_delta) > 5: st.warning("This change has a significant impact on the predicted risk.")
                else: st.info("This change has a minor impact on the predicted risk.")

if __name__ == "__main__":
    main()

