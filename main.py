import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import pickle
import traceback

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Hệ Thống Dự Đoán Tử Vong ICU",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 10px 0;
}
/* --- ADD THESE NEW CLASSES --- */
.metric-value {
    color: black !important;
}
.metric-value-increase {
    color: #d62728 !important; /* Red */
}
.metric-value-decrease {
    color: #2ca02c !important; /* Green */
}
.metric-value-neutral {
    color: #1f77b4 !important; /* Blue */
}
/* --------------------------- */
.high-risk {
    border-left-color: #d62728 !important;
    background-color: #ffe6e6 !important;
}
.medium-risk {
    border-left-color: #ff7f0e !important;
    background-color: #fff4e6 !important;
}
.low-risk {
    border-left-color: #2ca02c !important;
    background-color: #e6ffe6 !important;
}
.feature-updated {
    background-color: #fff3cd;
    padding: 8px 12px;
    border-radius: 5px;
    border-left: 3px solid #ffc107;
    margin: 5px 0;
    font-weight: 500;
}
.feature-new {
    background-color: #d1ecf1;
    padding: 8px 12px;
    border-radius: 5px;
    border-left: 3px solid #17a2b8;
    margin: 5px 0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CUSTOM GRU-D LAYER
# ============================================================================
class GRUDCell(Layer):
    def __init__(self, units, feature_dim, **kwargs):
        super(GRUDCell, self).__init__(**kwargs)
        self.units = units
        self.feature_dim = feature_dim
        self.state_size = units
        self.output_size = units

    def build(self, input_shape):
        self.W_z = self.add_weight(shape=(self.feature_dim, self.units), name='W_z', initializer='glorot_uniform')
        self.U_z = self.add_weight(shape=(self.units, self.units), name='U_z', initializer='orthogonal')
        self.b_z = self.add_weight(shape=(self.units,), name='b_z', initializer='zeros')
        self.W_r = self.add_weight(shape=(self.feature_dim, self.units), name='W_r', initializer='glorot_uniform')
        self.U_r = self.add_weight(shape=(self.units, self.units), name='U_r', initializer='orthogonal')
        self.b_r = self.add_weight(shape=(self.units,), name='b_r', initializer='zeros')
        self.W_h = self.add_weight(shape=(self.feature_dim, self.units), name='W_h', initializer='glorot_uniform')
        self.U_h = self.add_weight(shape=(self.units, self.units), name='U_h', initializer='orthogonal')
        self.b_h = self.add_weight(shape=(self.units,), name='b_h', initializer='zeros')
        self.gamma_x_decay = self.add_weight(shape=(self.feature_dim,), name='gamma_x_decay', initializer='ones')
        self.gamma_h_decay = self.add_weight(shape=(self.units,), name='gamma_h_decay', initializer='ones')
        self.mean_imputation = self.add_weight(shape=(self.feature_dim,), name='mean_imputation', initializer='zeros', trainable=False)
        self.built = True

    def call(self, inputs, states):
        x = inputs[:, :self.feature_dim]
        m = inputs[:, self.feature_dim : 2 * self.feature_dim]
        delta_t = inputs[:, 2 * self.feature_dim:]
        
        h_prev = states[0]

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

# ============================================================================
# MODEL BUILDING FUNCTION
# ============================================================================
def build_model(dynamic_shape, static_shape, max_events=200):
    """Builds the Keras model from scratch."""
    values_input = keras.layers.Input(shape=(max_events, dynamic_shape), name='values')
    mask_input = keras.layers.Input(shape=(max_events, dynamic_shape), name='mask')
    time_gaps_input = keras.layers.Input(shape=(max_events, 1), name='time_gaps')
    static_input = keras.layers.Input(shape=(static_shape,), name='static')

    concatenated_input = keras.layers.Concatenate(axis=-1)([values_input, mask_input, time_gaps_input])

    grud_cell = GRUDCell(units=64, feature_dim=dynamic_shape)
    grud_output = keras.layers.RNN(grud_cell, return_sequences=False)(concatenated_input)

    static_x = keras.layers.Dense(32, activation='relu')(static_input)
    static_x = keras.layers.BatchNormalization()(static_x)
    static_x = keras.layers.Dropout(0.4)(static_x)

    combined = keras.layers.Concatenate()([grud_output, static_x])
    shared_dense = keras.layers.Dense(64, activation='relu')(combined)
    shared_dense = keras.layers.BatchNormalization()(shared_dense)
    shared_dense = keras.layers.Dropout(0.5)(shared_dense)
    
    mortality_output = keras.layers.Dense(1, activation='sigmoid', name='mortality')(shared_dense)
    los_output = keras.layers.Dense(1, activation='linear', name='los')(shared_dense)

    model = keras.Model(
        inputs={
            'values': values_input, 
            'mask': mask_input, 
            'time_gaps': time_gaps_input, 
            'static': static_input
        },
        outputs={
            'mortality': mortality_output, 
            'los': los_output
        }
    )
    return model

# ============================================================================
# DATA LOADING AND FEATURE DEFINITIONS
# ============================================================================
@st.cache_resource
def load_model_and_scaler():
    try:
        dynamic_feature_count = len(DYNAMIC_FEATURES)
        static_feature_count = len(STATIC_FEATURES)
        model = build_model(dynamic_shape=dynamic_feature_count, static_shape=static_feature_count)
        model.load_weights('better_model_weights.h5')

        with open('scaler.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data['scaler']
            feature_names = scaler_data['feature_names']
            
        return model, scaler, feature_names, None
    except Exception as e:
        return None, None, None, str(e)

DYNAMIC_FEATURES = [
    'Unnamed: 0',
    'heart_rate', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'temperature', 'spo2', 'respiratory_rate',
    'gcs_total', 'lactate', 'creatinine', 'wbc', 'hemoglobin', 'drug_vasopressor_inotropes',
    'drug_sedative_analgesic', 'drug_antibiotic_broad', 'drug_diuretic', 'drug_anticoagulant',
    'drug_corticosteroid'
]
CATEGORICAL_FEATURES = [
    'insurance_group_INS_medicaid', 'insurance_group_INS_medicare', 'insurance_group_INS_other',
    'ethnicity_group_ETH_asian', 'ethnicity_group_ETH_black', 'ethnicity_group_ETH_latino',
    'ethnicity_group_ETH_other', 'ethnicity_group_ETH_white', 'marital_group_MAR_divorced',
    'marital_group_MAR_married', 'marital_group_MAR_single', 'marital_group_MAR_unknown',
    'marital_group_MAR_widowed', 'admission_type_AMBULATORY OBSERVATION', 'admission_type_DIRECT EMER.',
    'admission_type_DIRECT OBSERVATION', 'admission_type_ELECTIVE', 'admission_type_EU OBSERVATION',
    'admission_type_EW EMER.', 'admission_type_OBSERVATION ADMIT',
    'admission_type_SURGICAL SAME DAY ADMISSION', 'admission_type_URGENT',
    'first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)', 'first_careunit_Coronary Care Unit (CCU)',
    'first_careunit_Intensive Care Unit (ICU)', 'first_careunit_Med/Surg',
    'first_careunit_Medical Intensive Care Unit (MICU)',
    'first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU)', 'first_careunit_Medicine',
    'first_careunit_Medicine/Cardiology Intermediate', 'first_careunit_Neuro Intermediate',
    'first_careunit_Neuro Stepdown', 'first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)',
    'first_careunit_Neurology', 'first_careunit_PACU', 'first_careunit_Surgery/Trauma',
    'first_careunit_Surgery/Vascular/Intermediate', 'first_careunit_Surgical Intensive Care Unit (SICU)',
    'first_careunit_Trauma SICU (TSICU)'
]
STATIC_FEATURES = ['age', 'GENDER_f'] + CATEGORICAL_FEATURES
FEATURE_LABELS = {
    'heart_rate': 'Nhịp tim (bpm)', 'systolic_bp': 'HA tâm thu (mmHg)', 'diastolic_bp': 'HA tâm trương (mmHg)',
    'mean_bp': 'HA trung bình (mmHg)', 'temperature': 'Nhiệt độ (°C)', 'spo2': 'SpO2 (%)',
    'respiratory_rate': 'Nhịp thở (/phút)', 'gcs_total': 'GCS Score', 'lactate': 'Lactate (mmol/L)',
    'creatinine': 'Creatinine (mg/dL)', 'wbc': 'WBC (×10⁹/L)', 'hemoglobin': 'Hemoglobin (g/dL)',
    'platelets': 'Tiểu cầu (×10⁹/L)',
    'drug_vasopressor_inotropes': 'Thuốc vận mạch',
    'drug_sedative_analgesic': 'An thần/giảm đau',
    'drug_antibiotic_broad': 'Kháng sinh phổ rộng',
    'drug_diuretic': 'Lợi tiểu',
    'drug_anticoagulant': 'Chống đông',
    'drug_corticosteroid': 'Corticosteroid'
}

def get_categorical_options():
    options = {}
    for feature in CATEGORICAL_FEATURES:
        parts = feature.split('_', 2)
        group, value = parts[0] + '_' + parts[1], parts[2]
        if group not in options: options[group] = []
        options[group].append(value)
    return options
categorical_options = get_categorical_options()

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def prepare_patient_for_prediction(patient_info, patient_events, scaler, feature_names, max_events=200):
    if not patient_events:
        return None

    df_events = pd.DataFrame(patient_events)
    df_events['event_time'] = pd.to_datetime(df_events['event_time'])
    df_events = df_events.sort_values('event_time')
    
    df = pd.DataFrame()
    all_features = DYNAMIC_FEATURES + STATIC_FEATURES
    for col in all_features:
        if col in df_events.columns:
            df[col] = df_events[col]
        elif col in patient_info.index:
            df[col] = patient_info[col]
        else:
            df[col] = 0
            
    if 'gender' in df.columns and 'GENDER_f' not in df.columns:
        df['GENDER_f'] = df['gender']
        df = df.drop(columns=['gender'])
    
    df_for_scaling = pd.DataFrame(columns=feature_names)
    df_for_scaling = pd.concat([df_for_scaling, df], ignore_index=True)
    df_for_scaling = df_for_scaling[feature_names].fillna(0)
    
    features_to_scale = [f for f in feature_names if f in df_for_scaling.columns and f not in ['id', 'name', 'hadm_id']]
    
    df_for_scaling[features_to_scale] = df_for_scaling[features_to_scale].astype(float)

    df_scaled = df_for_scaling.copy()
    try:
        df_scaled[features_to_scale] = scaler.transform(df_scaled[features_to_scale])
    except Exception as e:
        st.error(f"Scaling error: {e}. Check feature consistency.")
        return None

    dynamic_cols = [col for col in DYNAMIC_FEATURES if col in df_scaled.columns]
    mask = ~df_for_scaling[dynamic_cols].isna().values
    df_scaled[dynamic_cols] = df_scaled[dynamic_cols].ffill().fillna(0)
    dynamic_values = df_scaled[dynamic_cols].values

    time_diffs = df_events['event_time'].diff().dt.total_seconds().div(3600).fillna(0).values.reshape(-1, 1)
    
    static_cols_ordered = [sc for sc in STATIC_FEATURES if sc in df_scaled.columns]
    static_values = df_scaled[static_cols_ordered].iloc[0].values

    if len(dynamic_values) > max_events:
        dynamic_values, mask, time_diffs = dynamic_values[-max_events:], mask[-max_events:], time_diffs[-max_events:]
    else:
        pad_len = max_events - len(dynamic_values)
        dynamic_values = np.pad(dynamic_values, ((0, pad_len), (0, 0)))
        mask = np.pad(mask, ((0, pad_len), (0, 0)))
        time_diffs = np.pad(time_diffs, ((0, pad_len), (0, 0)))
    
    return {
        'values': np.expand_dims(dynamic_values, axis=0),
        'mask': np.expand_dims(mask.astype(float), axis=0),
        'time_gaps': np.expand_dims(time_diffs, axis=0),
        'static': np.expand_dims(static_values, axis=0)
    }

def predict_mortality(patient_info, patient_data, model, scaler, feature_names):
    try:
        x = prepare_patient_for_prediction(patient_info, patient_data, scaler, feature_names)
        if x is None: 
            return None, None
        
        predictions = model.predict(x, verbose=0)
        mortality_risk = float(predictions['mortality'][0][0])
        los_pred = float(predictions['los'][0][0])
        return mortality_risk, los_pred
    except Exception as e:
        st.error("Lỗi chi tiết trong quá trình dự đoán:")
        st.error(f"Loại lỗi (Error Type): {type(e)}")
        st.error(f"Thông báo lỗi (Error Message): {e}")
        st.code(traceback.format_exc()) 
        return None, None

# ============================================================================
# DATA INITIALIZATION FUNCTION
# ============================================================================
def load_initial_data():
    """Loads and processes a single CSV into patient and event data structures."""
    try:
        full_df = pd.read_csv('patient_data.csv')
        full_df['event_time'] = pd.to_datetime(full_df['event_time'])

        static_cols = ['hadm_id', 'age', 'GENDER_f'] + CATEGORICAL_FEATURES
        for col in static_cols:
            if col not in full_df.columns:
                full_df[col] = 0
        
        patients_df = full_df[static_cols].drop_duplicates(subset=['hadm_id']).copy()
        
        patients_df.rename(columns={'hadm_id': 'id', 'GENDER_f': 'gender'}, inplace=True)
        patients_df['name'] = patients_df['id'].apply(lambda x: f"Bệnh nhân {x}")
        
        patients_df['gender'] = patients_df['gender'].astype(int)

        st.session_state.patient_db = patients_df
        st.session_state.next_patient_id = patients_df['id'].max() + 1 if not patients_df.empty else 1

        event_cols = DYNAMIC_FEATURES + ['hadm_id', 'event_time']
        existing_event_cols = [col for col in event_cols if col in full_df.columns]
        events_df = full_df[existing_event_cols]
        
        base_events = {}
        for patient_id, group in events_df.groupby('hadm_id'):
            group = group.sort_values('event_time')
            base_events[patient_id] = group.to_dict('records')
        st.session_state.base_events = base_events
        
        patient_ids = patients_df['id'].tolist()
        st.session_state.patient_events = {pid: [] for pid in patient_ids}
        st.session_state.prediction_history = {pid: [] for pid in patient_ids}
        st.session_state.simulation_index = {pid: 0 for pid in patient_ids}
        st.success("Tải dữ liệu bệnh nhân cơ bản thành công!")
        
    except FileNotFoundError:
        st.warning("`patient_data.csv` not found. Starting with an empty database.")
        st.session_state.patient_db = pd.DataFrame(columns=['id', 'name', 'age', 'gender'] + CATEGORICAL_FEATURES)
        st.session_state.next_patient_id = 1
        st.session_state.base_events = {}
        st.session_state.patient_events = {}
        st.session_state.prediction_history = {}
        st.session_state.simulation_index = {}
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        st.stop()

# ============================================================================
# SESSION STATE INITIALIZATION & MODEL LOADING
# ============================================================================
if 'patient_db' not in st.session_state:
    load_initial_data()

model, scaler, feature_names, model_error = load_model_and_scaler()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🏥 Hệ Thống Dự Đoán ICU")
st.sidebar.markdown("---")
page = st.sidebar.radio("**Chọn Chức Năng:**", ["🏥 Đăng ký bệnh nhân", "🩺 Theo dõi thời gian thực"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.metric("📋 Số bệnh nhân", len(st.session_state.patient_db))
if model_error:
    st.sidebar.warning(f"⚠️ Lỗi Model: {model_error}")
else:
    st.sidebar.success("✅ Model đã sẵn sàng")

# ============================================================================
# PAGE 1: REGISTER PATIENT
# ============================================================================
if page == "🏥 Đăng ký bệnh nhân":
    st.title("🏥 Đăng Ký Bệnh Nhân Mới")

    with st.form("new_patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Thông tin cơ bản")
            name = st.text_input("Tên bệnh nhân:")
            age = st.number_input("Tuổi:", 18, 100, 65)
            gender = st.selectbox("Giới tính:", ["Nam", "Nữ"])

        with col2:
            st.subheader("Thông tin hành chính")
            insurance = st.selectbox("Bảo hiểm:", categorical_options['insurance_group'])
            ethnicity = st.selectbox("Dân tộc:", categorical_options['ethnicity_group'])
            marital = st.selectbox("Tình trạng hôn nhân:", categorical_options['marital_group'])

        with col3:
            st.subheader("Thông tin nhập viện")
            admission_type = st.selectbox("Loại nhập viện:", categorical_options['admission_type'])
            careunit = st.selectbox("Đơn vị chăm sóc:", categorical_options['first_careunit'])

        submitted = st.form_submit_button("➕ Thêm Bệnh Nhân", type="primary", use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("⚠️ Vui lòng nhập tên bệnh nhân")
        else:
            new_id = st.session_state.next_patient_id
            
            new_patient_data = {
                'id': new_id,
                'name': name,
                'age': age,
                'gender': 1 if gender == "Nữ" else 0
            }
            
            for feature in CATEGORICAL_FEATURES:
                new_patient_data[feature] = 0
            
            new_patient_data[f"insurance_group_{insurance}"] = 1
            new_patient_data[f"ethnicity_group_{ethnicity}"] = 1
            new_patient_data[f"marital_group_{marital}"] = 1
            new_patient_data[f"admission_type_{admission_type}"] = 1
            new_patient_data[f"first_careunit_{careunit}"] = 1

            new_row_df = pd.DataFrame([new_patient_data])
            st.session_state.patient_db = pd.concat([st.session_state.patient_db, new_row_df], ignore_index=True)
            
            st.session_state.patient_events[new_id] = []
            st.session_state.prediction_history[new_id] = []
            st.session_state.simulation_index[new_id] = 0
            st.session_state.next_patient_id += 1
            
            st.success(f"✅ Đã thêm bệnh nhân {name} (ID: {new_id})")
            st.balloons()

    st.markdown("---")
    st.subheader("📋 Danh Sách Bệnh Nhân")
    if not st.session_state.patient_db.empty:
        display_df = st.session_state.patient_db[['id', 'name', 'age', 'gender']].copy()
        display_df['gender'] = display_df['gender'].map({0: 'Nam', 1: 'Nữ'})
        display_df['Số sự kiện'] = display_df['id'].map(lambda x: len(st.session_state.patient_events.get(x, [])))
        display_df.columns = ['ID', 'Tên', 'Tuổi', 'Giới tính', 'Số sự kiện']
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Chưa có bệnh nhân nào.")

# ============================================================================
# PAGE 2: REAL-TIME MONITORING
# ============================================================================
else:
    st.title("🩺 Theo Dõi Thời Gian Thực & Dự Đoán")

    if model is None:
        st.error(f"⚠️ Không thể tải mô hình. Lỗi: {model_error}")
        st.stop()
    if st.session_state.patient_db.empty:
        st.warning("⚠️ Chưa có bệnh nhân nào. Hãy thêm ở trang 'Đăng ký bệnh nhân'.")
        st.stop()

    patient_id = st.selectbox(
        "🔎 Chọn Bệnh Nhân", st.session_state.patient_db['id'].tolist(),
        format_func=lambda x: f"ID {x}: {st.session_state.patient_db.loc[st.session_state.patient_db['id']==x, 'name'].iloc[0]}"
    )
    
    patient_info = st.session_state.patient_db.loc[st.session_state.patient_db['id'] == patient_id].iloc[0]
    
    st.markdown(f"### 👤 {patient_info['name']} | {patient_info['age']} tuổi | {'Nữ' if patient_info['gender'] == 1 else 'Nam'}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        is_base_patient = patient_id in st.session_state.base_events
        
        if st.button("📈 Tải Sự Kiện Tiếp Theo (Mô phỏng)", use_container_width=True, type="primary", disabled=not is_base_patient):
            sim_index = st.session_state.simulation_index.get(patient_id, 0)
            base_events_list = st.session_state.base_events[patient_id]
            
            if sim_index < len(base_events_list):
                new_event = base_events_list[sim_index].copy()
                st.session_state.patient_events[patient_id].append(new_event)
                st.session_state.simulation_index[patient_id] += 1
                st.success(f"✅ Đã tải sự kiện mô phỏng #{sim_index + 1}")
                st.rerun()
            else:
                st.info("Đã hết sự kiện mô phỏng cho bệnh nhân này.")
    
    with col2:
        auto_predict = st.checkbox("🔄 Tự động dự đoán", value=True)
        if not is_base_patient:
            st.caption("Chế độ mô phỏng chỉ dành cho bệnh nhân có sẵn (tải từ file).")

    # Prediction Logic
    events = st.session_state.patient_events.get(patient_id, [])
    history = st.session_state.prediction_history.get(patient_id, [])
    last_pred_count = history[-1]['event_count'] if history else -1

    if auto_predict and len(events) > 0 and len(events) != last_pred_count:
        with st.spinner("Đang dự đoán..."):
            mortality_risk, los_pred = predict_mortality(patient_info, events, model, scaler, feature_names)
            if mortality_risk is not None:
                prediction_record = {
                    'timestamp': datetime.now(),
                    'event_count': len(events),
                    'mortality_risk': mortality_risk,
                    'los_pred': los_pred
                }
                st.session_state.prediction_history[patient_id].append(prediction_record)
                st.rerun()

    # Display results and charts 
    if events:
        # ============================================================================
        # NEW SECTION: CUMULATIVE EVENTS TABLE
        # ============================================================================
        st.markdown("---")
        st.subheader("📊 Bảng Dữ Liệu Sự Kiện Liên Tục")
        
        # Create display dataframe with forward fill
        display_rows = []
        
        # Track last known values for forward filling
        last_values = {}
        
        for idx, event in enumerate(events, 1):
            row = {
                'Sự kiện': f"#{idx}",
                'Thời gian': event['event_time'].strftime('%H:%M:%S') if isinstance(event['event_time'], datetime) else str(event['event_time'])
            }
            
            # Add vital signs with forward fill
            vital_features = [
                ('heart_rate', 'Nhịp tim'),
                ('mean_bp', 'HA TB'),
                ('temperature', 'Nhiệt độ'),
                ('spo2', 'SpO2'),
                ('respiratory_rate', 'Nhịp thở'),
                ('gcs_total', 'GCS'),
                ('lactate', 'Lactate'),
                ('creatinine', 'Creatinine')
            ]
            
            for feature_key, display_key in vital_features:
                value = event.get(feature_key)
                # If value exists and is not None/NaN, update last known value
                if value is not None and value != '' and not (isinstance(value, float) and np.isnan(value)):
                    last_values[feature_key] = value
                    row[display_key] = f"{value}"
                # Otherwise, use forward filled value
                elif feature_key in last_values:
                    row[display_key] = f"{last_values[feature_key]} ↻"  # ↻ indicates forward filled
                else:
                    row[display_key] = '-'
            
            # Add drug indicators with forward fill
            drug_features = [
                ('drug_vasopressor_inotropes', 'Vận mạch'),
                ('drug_sedative_analgesic', 'An thần'),
                ('drug_antibiotic_broad', 'KS'),
                ('drug_diuretic', 'Lợi tiểu'),
                ('drug_anticoagulant', 'Chống đông'),
                ('drug_corticosteroid', 'Cortico')
            ]
            
            drugs = []
            for drug_key, drug_name in drug_features:
                value = event.get(drug_key)
                # Update last known value if present
                if value is not None and value != '':
                    last_values[drug_key] = value
                
                # Check if drug is active (using forward filled value if needed)
                if last_values.get(drug_key, 0) == 1:
                    drugs.append(drug_name)
            
            row['Thuốc'] = ', '.join(drugs) if drugs else '-'
            
            display_rows.append(row)
        
        # Create DataFrame and style it
        events_table_df = pd.DataFrame(display_rows)
        
        # Highlight the last row (most recent event)
        def highlight_last_row(s):
            return ['background-color: #fff3cd' if s.name == len(events_table_df) - 1 else '' for _ in s]
        
        styled_df = events_table_df.style.apply(highlight_last_row, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=min(400, 50 + len(events_table_df) * 35)
        )
        
        st.caption(f"📝 Tổng số sự kiện: **{len(events)}** | Sự kiện mới nhất được tô sáng | Ký hiệu ↻ = giá trị được giữ từ lần đo trước")
        
        # ============================================================================
        # EXISTING PREDICTION DISPLAY
        # ============================================================================
        st.markdown("---")
        if history:
            latest = history[-1]
            mortality_risk = latest['mortality_risk']
            los_pred = latest['los_pred']
            
            if mortality_risk >= 0.7: risk_class, risk_label, risk_emoji = "high-risk", "RẤT CAO ⚠️", "🔴"
            elif mortality_risk >= 0.3: risk_class, risk_label, risk_emoji = "medium-risk", "TRUNG BÌNH", "🟡"
            else: risk_class, risk_label, risk_emoji = "low-risk", "THẤP", "🟢"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="metric-card {risk_class}"><h2 class="metric-value" style="margin:0; color: black">{risk_emoji} {mortality_risk*100:.1f}%</h2><p style="margin:5px 0; color: black">Nguy cơ tử vong: <strong>{risk_label}</strong></p><p style="margin:0; font-size:0.9em; color: black">Dựa trên {latest['event_count']} sự kiện</p></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-card"><h2 class="metric-value" style="margin:0; color: black">📅 {los_pred:.1f} ngày</h2><p style="margin:5px 0; color: black">Dự đoán thời gian nằm ICU</p><p style="margin:0; font-size:0.9em; color: black">Length of Stay</p></div>""", unsafe_allow_html=True)
            with col3:
                if len(history) >= 2:
                    prev_risk = history[-2]['mortality_risk']
                    risk_change = mortality_risk - prev_risk
                    trend = "📈" if risk_change > 0 else "📉" if risk_change < 0 else "➡️"
                    
                    if risk_change > 0:
                        change_class = "metric-value-increase"
                    elif risk_change < 0:
                        change_class = "metric-value-decrease"
                    else:
                        change_class = "metric-value-neutral"
                else:
                    trend, risk_change, change_class = "➡️", 0, "metric-value-neutral"
                
                st.markdown(f"""<div class="metric-card"><h2 class="{change_class}" style="margin:0; color: black">{trend} {abs(risk_change)*100:.1f}%</h2><p style="margin:5px 0; color: black">Thay đổi nguy cơ</p><p style="margin:0; font-size:0.9em; color: black">So với lần đo trước</p></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Quỹ Đạo Nguy Cơ Tử Vong")
        if len(history) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h['event_count'] for h in history], y=[h['mortality_risk'] * 100 for h in history],
                mode='lines+markers', name='Nguy cơ tử vong', line=dict(color='#d62728', width=3),
                marker=dict(size=10), hovertemplate='Sự kiện %{x}<br>Nguy cơ: %{y:.1f}%<extra></extra>'
            ))
            fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
            fig.add_hrect(y0=30, y1=70, fillcolor="orange", opacity=0.1, line_width=0)
            fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
            fig.update_layout(xaxis_title="Số sự kiện quan sát", yaxis_title="Nguy cơ tử vong (%)", yaxis_range=[0, 100],
                              hovermode='x unified', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Event details and other charts
        st.markdown("---")
        st.subheader("📋 Chi Tiết & Diễn Biến")
        latest_event = events[-1]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🆕 Sự Kiện Gần Nhất")
            sub_c1, sub_c2 = st.columns(2)
            sub_c1.metric("❤️ Nhịp tim", f"{latest_event.get('heart_rate', 'N/A')} bpm")
            sub_c1.metric("🌡️ Nhiệt độ", f"{latest_event.get('temperature', 'N/A')}°C")
            sub_c1.metric("🫁 SpO2", f"{latest_event.get('spo2', 'N/A')}%")
            sub_c1.metric("🧪 Lactate", f"{latest_event.get('lactate', 'N/A')} mmol/L")
            sub_c2.metric("💉 HA trung bình", f"{latest_event.get('mean_bp', 'N/A')} mmHg")
            sub_c2.metric("💨 Nhịp thở", f"{latest_event.get('respiratory_rate', 'N/A')}/phút")
            sub_c2.metric("🧠 GCS", latest_event.get('gcs_total', 'N/A'))
            sub_c2.metric("🩸 Creatinine", f"{latest_event.get('creatinine', 'N/A')} mg/dL")

        with col2:
            st.markdown("#### 📈 Biểu Đồ Sinh Hiệu")
            df_plot = pd.DataFrame(events)
            vital_options = st.multiselect(
                "Chọn sinh hiệu để hiển thị:",
                options=['heart_rate', 'mean_bp', 'temperature', 'spo2', 'respiratory_rate', 'lactate', 'gcs_total', 'platelets'],
                default=['heart_rate', 'mean_bp'],
                format_func=lambda x: FEATURE_LABELS.get(x, x)
            )
            if vital_options:
                fig_vitals = go.Figure()
                for vital in vital_options:
                    if vital in df_plot.columns:
                        fig_vitals.add_trace(go.Scatter(
                            x=list(range(1, len(df_plot) + 1)), y=df_plot[vital],
                            mode='lines+markers', name=FEATURE_LABELS.get(vital, vital),
                            hovertemplate=f'{FEATURE_LABELS.get(vital, vital)}: %{{y}}<extra></extra>'
                        ))
                fig_vitals.update_layout(
                    xaxis_title="Số thứ tự sự kiện", yaxis_title="Giá trị", hovermode='x unified',
                    height=350, margin=dict(l=40, r=40, t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_vitals, use_container_width=True)
    else:
        st.info("📝 Chưa có sự kiện nào. Nhấn 'Tải Sự Kiện Tiếp Theo' để bắt đầu mô phỏng, hoặc nhập thủ công bên dưới.")
    
    # Manual input form
    st.markdown("---")
    st.subheader("✏️ Thêm Sự Kiện Thủ Công")
    with st.expander("📝 Nhập dữ liệu thủ công"):
        with st.form("manual_event_form"):
            st.markdown("**Sinh hiệu**")
            col1, col2, col3 = st.columns(3)
            with col1:
                hr = st.number_input("Nhịp tim (bpm)", 30, 200, 80)
                sbp = st.number_input("HA tâm thu", 70, 250, 120)
                dbp = st.number_input("HA tâm trương", 40, 150, 70)
                mbp = st.number_input("HA trung bình", 50, 180, 85)
            with col2:
                temp = st.number_input("Nhiệt độ (°C)", 35.0, 42.0, 37.0, 0.1)
                spo2 = st.number_input("SpO2 (%)", 50, 100, 95)
                resp = st.number_input("Nhịp thở", 5, 50, 16)
                gcs = st.slider("GCS Score", 3, 15, 13)
            with col3:
                lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 2.0, 0.1)
                creat = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.0, 0.1)
                wbc = st.number_input("WBC (×10⁹/L)", 0.0, 50.0, 10.0, 0.1)
                hb = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 12.0, 0.1)
                plt = st.number_input("Tiểu cầu (×10⁹/L)", 0.0, 1000.0, 250.0, 1.0)
            
            st.markdown("**Thuốc đang sử dụng**")
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                vaso = st.checkbox("Thuốc vận mạch")
                sed = st.checkbox("An thần/giảm đau")
            with d_col2:
                ab = st.checkbox("Kháng sinh phổ rộng")
                diu = st.checkbox("Lợi tiểu")
            with d_col3:
                ac = st.checkbox("Chống đông")
                cs = st.checkbox("Corticosteroid")
            
            submit_manual = st.form_submit_button("➕ Thêm Sự Kiện", use_container_width=True, type="primary")
            
            if submit_manual:
                manual_event = {
                    'event_time': datetime.now(),
                    'heart_rate': hr, 'systolic_bp': sbp, 'diastolic_bp': dbp, 'mean_bp': mbp,
                    'temperature': temp, 'spo2': spo2, 'respiratory_rate': resp, 'gcs_total': gcs,
                    'lactate': lactate, 'creatinine': creat, 'wbc': wbc, 'hemoglobin': hb, 'platelets': plt,
                    'drug_vasopressor_inotropes': int(vaso), 'drug_sedative_analgesic': int(sed),
                    'drug_antibiotic_broad': int(ab), 'drug_diuretic': int(diu),
                    'drug_anticoagulant': int(ac), 'drug_corticosteroid': int(cs)
                }
                st.session_state.patient_events[patient_id].append(manual_event)
                st.success("✅ Đã thêm sự kiện thủ công!")
                st.rerun()

    # Download Buttons
    if events:
        st.markdown("---")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            if history:
                history_df = pd.DataFrame(history)
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Tải Lịch Sử Dự Đoán (CSV)", data=csv,
                                  file_name=f"prediction_history_{patient_id}.csv", mime='text/csv', use_container_width=True)
        with dl_col2:
            events_df = pd.DataFrame(events)
            csv_events = events_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Tải Dữ Liệu Sự Kiện (CSV)", data=csv_events,
                              file_name=f"events_{patient_id}.csv", mime='text/csv', use_container_width=True)

# Footer
st.markdown("---")
st.caption("⚠️ **Lưu ý:** Ứng dụng này chỉ phục vụ mục đích nghiên cứu và demo. Không sử dụng cho chẩn đoán lâm sàng thực tế.")
