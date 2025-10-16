import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Conv2DTranspose, Concatenate, ReLU, UpSampling2D, GlobalAveragePooling2D, Dense, Multiply, Bidirectional, ConvLSTM2D, BatchNormalization, AveragePooling2D, Activation, Reshape, Add, LSTM, Dropout, Masking, Layer,MultiHeadAttention, LayerNormalization, RNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tqdm import tqdm
import pickle
# Configuration
CSV_FILE = "icu_patients_timeseries.csv"

# All time-dependent measurements
VITAL_SIGNS = {
    'temperature': 'Temperature (°C)',
    'heart_rate': 'Heart Rate (bpm)',
    'systolic_bp': 'Systolic BP (mmHg)',
    'diastolic_bp': 'Diastolic BP (mmHg)',
    'mean_bp': 'Mean BP (mmHg)',
    'respiratory_rate': 'Respiratory Rate (/min)',
    'spo2': 'SpO2 (%)',
    'gcs_total': 'GCS Total'
}

BLOOD_GAS = {
    'so2': 'O2 Saturation (%)',
    'po2': 'PaO2 (mmHg)',
    'pco2': 'PaCO2 (mmHg)',
    'fio2': 'FiO2 (%)',
    'ph': 'pH',
    'baseexcess': 'Base Excess (mEq/L)',
    'bicarbonate': 'Bicarbonate (mEq/L)',
    'totalco2': 'Total CO2 (mEq/L)'
}

LABS_HEMATOLOGY = {
    'hemoglobin': 'Hemoglobin (g/dL)',
    'hematocrit': 'Hematocrit (%)',
    'wbc': 'WBC (K/µL)',
    'platelet': 'Platelets (K/µL)',
    'rbc': 'RBC (M/µL)',
    'mch': 'MCH (pg)',
    'mchc': 'MCHC (g/dL)',
    'mcv': 'MCV (fL)',
    'rdw': 'RDW (%)'
}

LABS_CHEMISTRY = {
    'glucose': 'Glucose (mg/dL)',
    'sodium': 'Sodium (mEq/L)',
    'potassium': 'Potassium (mEq/L)',
    'chloride': 'Chloride (mEq/L)',
    'calcium': 'Calcium (mg/dL)',
    'lactate': 'Lactate (mmol/L)',
    'creatinine': 'Creatinine (mg/dL)',
    'bun': 'BUN (mg/dL)',
    'aniongap': 'Anion Gap (mEq/L)'
}

LABS_LIVER = {
    'albumin': 'Albumin (g/dL)',
    'globulin': 'Globulin (g/dL)',
    'total_protein': 'Total Protein (g/dL)',
    'bilirubin': 'Bilirubin (mg/dL)',
    'alt': 'ALT (U/L)',
    'ast': 'AST (U/L)'
}

MEDICATIONS = {
    'dopamine': 'Dopamine (mcg/kg/min)',
    'epinephrine': 'Epinephrine (mcg/min)',
    'norepinephrine': 'Norepinephrine (mcg/min)',
    'phenylephrine': 'Phenylephrine (mcg/min)',
    'vasopressin': 'Vasopressin (units/min)',
    'dobutamine': 'Dobutamine (mcg/kg/min)',
    'milrinone': 'Milrinone (mcg/kg/min)'
}

INTERVENTIONS = {
    'mechanical_vent': 'Mechanical Ventilation',
    'dialysis': 'Dialysis',
    'vasopressor': 'Any Vasopressor',
    'sedation': 'Sedation',
    'antibiotics': 'Antibiotics'
}

# Static patient info
STATIC_FIELDS = {
    'patient_id': 'Patient ID',
    'age': 'Age',
    'gender': 'Gender',
    'insurance': 'Insurance Type',
    'ethnicity': 'Ethnicity',
    'marital_status': 'Marital Status',
    'admission_type': 'Admission Type',
    'first_care_unit': 'First Care Unit',
    'icd_category': 'ICD Category'
}

GENDER_OPTIONS = ["Male", "Female"]
INSURANCE_OPTIONS = ["Medicaid", "Medicare", "Private", "Other"]
ETHNICITY_OPTIONS = ["Asian", "Black", "Latino", "White", "Other"]
MARITAL_OPTIONS = ["Divorced", "Married", "Single", "Unknown", "Widowed"]
ADMISSION_OPTIONS = ["Emergency", "Elective", "Urgent", "Observation"]
CARE_UNIT_OPTIONS = ["CVICU", "CCU", "MICU", "MICU/SICU", "Neuro ICU", "SICU", "TSICU"]
ICD_CATEGORIES = ["Circulatory", "Respiratory", "Infectious", "Injury", "Digestive", "Nervous", "Other"]



# Initialize CSV
def init_csv():
    if not os.path.exists(CSV_FILE):
        columns = ['timestamp', 'patient_id', 'event_number'] + \
                  list(STATIC_FIELDS.keys())[1:] + \
                  list(VITAL_SIGNS.keys()) + \
                  list(BLOOD_GAS.keys()) + \
                  list(LABS_HEMATOLOGY.keys()) + \
                  list(LABS_CHEMISTRY.keys()) + \
                  list(LABS_LIVER.keys()) + \
                  list(MEDICATIONS.keys()) + \
                  list(INTERVENTIONS.keys()) + \
                  ['urine_output', 'notes']
        df = pd.DataFrame(columns=columns)
        df.to_csv(CSV_FILE, index=False)

def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame()

def get_patient_list():
    df = load_data()
    if len(df) > 0:
        return sorted(df['patient_id'].unique().tolist())
    return []

def get_patient_static_info(patient_id):
    df = load_data()
    patient_data = df[df['patient_id'] == patient_id]
    if len(patient_data) > 0:
        return patient_data.iloc[0][list(STATIC_FIELDS.keys())].to_dict()
    return None

def get_patient_events(patient_id):
    df = load_data()
    return df[df['patient_id'] == patient_id].sort_values('timestamp', ascending=False)

def get_next_event_number(patient_id):
    df = load_data()
    patient_events = df[df['patient_id'] == patient_id]
    if len(patient_events) > 0:
        return patient_events['event_number'].max() + 1
    return 1

def save_event(event_data):
    df = load_data()
    new_row = pd.DataFrame([event_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return True

# Define model
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

    @tf.function
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

# --- Model Building Function ---
def build_model(values_shape, mask_shape, time_gaps_shape, static_shape, l2_reg=0.01):
    # Define all inputs
    values_input = Input(shape = values_shape)
    mask_input = Input(shape = mask_shape)
    time_gaps_input = Input(shape = time_gaps_shape)
    static_input = Input(shape = static_shape)
    
    combined_dynamic_input = Concatenate(axis=-1)([values_input, mask_input, time_gaps_input])
    
    # layers for time series features
    feature_dim = values_shape[-1]
    grud_layer = RNN(GRUDCell(64, feature_dim=feature_dim), return_sequences=False)(combined_dynamic_input)
    
    # Static stream
    static_x = Dense(32,kernel_regularizer = l2(l2_reg))(static_input)
    static_x = BatchNormalization()(static_x)
    static_x = ReLU()(static_x)
    static_x = Dropout(0.4)(static_x)
    
    # Combine static with time series
    combined_layer = Concatenate()([grud_layer, static_x])
    shared_dense = Dense(64, kernel_regularizer = l2(l2_reg))(combined_layer)
    shared_dense = BatchNormalization()(shared_dense)
    shared_dense = ReLU()(shared_dense)
    shared_dense = Dropout(0.5)(shared_dense)
    
    mortality_output = Dense(1, activation='sigmoid', name='mortality')(shared_dense)
    los_output = Dense(1, activation='linear', name='los')(shared_dense)
    
    model = Model(inputs = {'values': values_input,
                            'mask': mask_input,
                            'time_gaps': time_gaps_input,
                            'static': static_input},
                  outputs = {
                      'mortality': mortality_output,
                      'los': los_output
                  })
    return model

# --- Feature Definitions for Model ---
DYNAMIC_FEATURES = [
    'heart_rate', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'temperature', 'spo2', 'respiratory_rate',
    'gcs_total', 'lactate', 'creatinine', 'wbc', 'hemoglobin', 'platelet', 'glucose', 'sodium',
    'potassium', 'bun'
]
STATIC_FEATURES = [
    'age', 'gender_Female', 'gender_Male', 'insurance_Medicaid', 'insurance_Medicare', 'insurance_Other',
    'insurance_Private', 'ethnicity_Asian', 'ethnicity_Black', 'ethnicity_Latino', 'ethnicity_Other',
    'ethnicity_White', 'marital_status_Divorced', 'marital_status_Married', 'marital_status_Single',
    'marital_status_Unknown', 'marital_status_Widowed'
]
FEATURE_LABELS = {
    'heart_rate': 'Heart Rate (bpm)', 'systolic_bp': 'Systolic BP (mmHg)', 'diastolic_bp': 'Diastolic BP (mmHg)',
    'mean_bp': 'Mean BP (mmHg)', 'temperature': 'Temperature (°C)', 'spo2': 'SpO2 (%)',
    'respiratory_rate': 'Respiratory Rate (/min)', 'gcs_total': 'GCS Score', 'lactate': 'Lactate (mmol/L)',
    'creatinine': 'Creatinine (mg/dL)', 'wbc': 'WBC (K/µL)', 'hemoglobin': 'Hemoglobin (g/dL)',
    'platelet': 'Platelets (K/µL)', 'glucose': 'Glucose (mg/dL)', 'sodium': 'Sodium (mEq/L)',
    'potassium': 'Potassium (mEq/L)', 'bun': 'BUN (mg/dL)'
}

@st.cache_resource
def load_model_and_scaler():
    try:
        # model = build_model(dynamic_shape=len(DYNAMIC_FEATURES), static_shape=len(STATIC_FEATURES))
        model = build_model(
            values_shape=(200, len(DYNAMIC_FEATURES)),
            mask_shape=(200, len(DYNAMIC_FEATURES)),
            time_gaps_shape=(200, 1),
            static_shape=(len(STATIC_FEATURES),)
        )
        model.load_weights('better_model_weights.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
        return model, scaler_data, None
    except Exception as e:
        return None, None, str(e)

# --- Prediction & Data Preparation Functions ---
def prepare_patient_for_prediction(patient_events_df, scaler_data, max_events=200):
    if patient_events_df.empty:
        return None

    # 1. Create a full feature DataFrame from the events
    df = patient_events_df.copy()
    
    # 2. Handle static features (One-Hot Encoding)
    static_data = df.iloc[0]
    for feature in STATIC_FEATURES:
        if feature == 'age':
            df[feature] = static_data['age']
        else:
            parts = feature.split('_', 1)
            col, val = parts[0], parts[1]
            df[feature] = 1 if static_data[col] == val else 0
            
    # Ensure all expected columns are present, fill with 0 if not
    for col in DYNAMIC_FEATURES + STATIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # 3. Scale the features
    scaler = scaler_data['scaler']
    feature_names = scaler_data['feature_names']
    
    # Reorder df columns to match scaler's expected order and scale
    df_for_scaling = df[feature_names]
    df_scaled_values = scaler.transform(df_for_scaling)
    df_scaled = pd.DataFrame(df_scaled_values, columns=feature_names, index=df.index)

    # 4. Prepare inputs for the model
    dynamic_cols = [col for col in DYNAMIC_FEATURES if col in df_scaled.columns]
    mask = ~patient_events_df[dynamic_cols].isna().values
    dynamic_values = df_scaled[dynamic_cols].ffill().fillna(0).values

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_diffs = df['timestamp'].diff().dt.total_seconds().div(3600).fillna(0).values.reshape(-1, 1)

    static_cols_ordered = [sc for sc in STATIC_FEATURES if sc in df_scaled.columns]
    static_values = df_scaled[static_cols_ordered].iloc[0].values

    # 5. Pad sequences to max_events
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

def predict_mortality(patient_events_df, model, scaler_data):
    try:
        x = prepare_patient_for_prediction(patient_events_df, scaler_data)
        if x is None: return None, None
        predictions = model.predict(x, verbose=0)
        return float(predictions['mortality'][0][0]), float(predictions['los'][0][0])
    except Exception:
        st.error("Error during prediction. Details:")
        st.code(traceback.format_exc())
        return None, None

## Main function
def main():
    st.set_page_config(page_title="ICU Time-Series Data Collection", layout="wide")
    st.title("🏥 ICU Time-Series Data Collection System")
    
    init_csv()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Mode:", 
                       ["New Patient", "Add Event", "View Data", "Model Predictions", "Export Data"])
        
        st.divider()
        patient_count = len(get_patient_list())
        total_events = len(load_data())
        st.metric("Total Patients", patient_count)
        st.metric("Total Events", total_events)
    
    # PAGE 1: New Patient Registration
    if page == "New Patient":
        st.header("Register New Patient")
        
        with st.form("new_patient_form"):
            st.subheader("Patient Demographics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                patient_id = st.text_input("Patient ID*", placeholder="ICU-001")
                age = st.number_input("Age*", 18, 120, 65)
                gender = st.selectbox("Gender*", GENDER_OPTIONS)
            
            with col2:
                insurance = st.selectbox("Insurance*", INSURANCE_OPTIONS)
                ethnicity = st.selectbox("Ethnicity*", ETHNICITY_OPTIONS)
                marital_status = st.selectbox("Marital Status*", MARITAL_OPTIONS)
            
            with col3:
                admission_type = st.selectbox("Admission Type*", ADMISSION_OPTIONS)
                first_care_unit = st.selectbox("First Care Unit*", CARE_UNIT_OPTIONS)
                icd_category = st.selectbox("ICD Category*", ICD_CATEGORIES)
            
            st.divider()
            st.subheader("Initial Event Data")
            st.info("Fill in the initial measurements for this patient")
            
            # Vital Signs
            with st.expander("🫀 Vital Signs", expanded=True):
                cols = st.columns(4)
                vitals_data = {}
                for idx, (key, label) in enumerate(VITAL_SIGNS.items()):
                    with cols[idx % 4]:
                        vitals_data[key] = st.number_input(label, value=None, step=0.1)
            
            # Blood Gas
            with st.expander("💉 Blood Gas"):
                cols = st.columns(4)
                bloodgas_data = {}
                for idx, (key, label) in enumerate(BLOOD_GAS.items()):
                    with cols[idx % 4]:
                        bloodgas_data[key] = st.number_input(label, value=None, step=0.01)
            
            # Labs - Hematology
            with st.expander("🩸 Hematology"):
                cols = st.columns(4)
                heme_data = {}
                for idx, (key, label) in enumerate(LABS_HEMATOLOGY.items()):
                    with cols[idx % 4]:
                        heme_data[key] = st.number_input(label, value=None, step=0.1)
            
            # Labs - Chemistry
            with st.expander("🧪 Chemistry"):
                cols = st.columns(4)
                chem_data = {}
                for idx, (key, label) in enumerate(LABS_CHEMISTRY.items()):
                    with cols[idx % 4]:
                        chem_data[key] = st.number_input(label, value=None, step=0.1)
            
            # Labs - Liver
            with st.expander("🫘 Liver Function"):
                cols = st.columns(3)
                liver_data = {}
                for idx, (key, label) in enumerate(LABS_LIVER.items()):
                    with cols[idx % 3]:
                        liver_data[key] = st.number_input(label, value=None, step=0.1)
            
            # Medications
            with st.expander("💊 Medications"):
                cols = st.columns(4)
                meds_data = {}
                for idx, (key, label) in enumerate(MEDICATIONS.items()):
                    with cols[idx % 4]:
                        meds_data[key] = st.number_input(label, value=None, step=0.01)
            
            # Interventions
            with st.expander("🏥 Interventions"):
                cols = st.columns(5)
                interv_data = {}
                for idx, (key, label) in enumerate(INTERVENTIONS.items()):
                    with cols[idx % 5]:
                        interv_data[key] = st.checkbox(label)
            
            # Additional
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                urine_output = st.number_input("Urine Output (mL)", value=None, step=1.0)
            with col2:
                notes = st.text_area("Notes", height=100)
            
            submitted = st.form_submit_button("Register Patient", type="primary", use_container_width=True)
        
        if submitted:
            if not patient_id:
                st.error("Patient ID is required!")
            elif patient_id in get_patient_list():
                st.error(f"Patient {patient_id} already exists!")
            else:
                event_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'patient_id': patient_id,
                    'event_number': 1,
                    'age': age,
                    'gender': gender,
                    'insurance': insurance,
                    'ethnicity': ethnicity,
                    'marital_status': marital_status,
                    'admission_type': admission_type,
                    'first_care_unit': first_care_unit,
                    'icd_category': icd_category,
                    **vitals_data,
                    **bloodgas_data,
                    **heme_data,
                    **chem_data,
                    **liver_data,
                    **meds_data,
                    **{k: 1 if v else 0 for k, v in interv_data.items()},
                    'urine_output': urine_output,
                    'notes': notes
                }
                
                if save_event(event_data):
                    st.success(f"✅ Patient {patient_id} registered successfully!")
                    st.balloons()
                else:
                    st.error("Failed to save patient data!")
    
    # PAGE 2: Add Event
    elif page == "Add Event":
        st.header("Add New Event for Existing Patient")
        
        patients = get_patient_list()
        if not patients:
            st.warning("No patients registered yet. Please register a patient first.")
        else:
            selected_patient = st.selectbox("Select Patient", patients)
            
            if selected_patient:
                static_info = get_patient_static_info(selected_patient)
                next_event = get_next_event_number(selected_patient)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Patient ID", selected_patient)
                with col2:
                    st.metric("Age", static_info['age'])
                with col3:
                    st.metric("Next Event #", next_event)
                
                st.divider()
                
                with st.form(f"event_form_{selected_patient}"):
                    st.subheader(f"Event #{next_event} Data")
                    
                    # Vital Signs
                    with st.expander("🫀 Vital Signs", expanded=True):
                        cols = st.columns(4)
                        vitals_data = {}
                        for idx, (key, label) in enumerate(VITAL_SIGNS.items()):
                            with cols[idx % 4]:
                                vitals_data[key] = st.number_input(label, value=None, step=0.1, key=f"v_{key}")
                    
                    # Blood Gas
                    with st.expander("💉 Blood Gas"):
                        cols = st.columns(4)
                        bloodgas_data = {}
                        for idx, (key, label) in enumerate(BLOOD_GAS.items()):
                            with cols[idx % 4]:
                                bloodgas_data[key] = st.number_input(label, value=None, step=0.01, key=f"bg_{key}")
                    
                    # Labs - Hematology
                    with st.expander("🩸 Hematology"):
                        cols = st.columns(4)
                        heme_data = {}
                        for idx, (key, label) in enumerate(LABS_HEMATOLOGY.items()):
                            with cols[idx % 4]:
                                heme_data[key] = st.number_input(label, value=None, step=0.1, key=f"h_{key}")
                    
                    # Labs - Chemistry
                    with st.expander("🧪 Chemistry"):
                        cols = st.columns(4)
                        chem_data = {}
                        for idx, (key, label) in enumerate(LABS_CHEMISTRY.items()):
                            with cols[idx % 4]:
                                chem_data[key] = st.number_input(label, value=None, step=0.1, key=f"c_{key}")
                    
                    # Labs - Liver
                    with st.expander("🫘 Liver Function"):
                        cols = st.columns(3)
                        liver_data = {}
                        for idx, (key, label) in enumerate(LABS_LIVER.items()):
                            with cols[idx % 3]:
                                liver_data[key] = st.number_input(label, value=None, step=0.1, key=f"l_{key}")
                    
                    # Medications
                    with st.expander("💊 Medications"):
                        cols = st.columns(4)
                        meds_data = {}
                        for idx, (key, label) in enumerate(MEDICATIONS.items()):
                            with cols[idx % 4]:
                                meds_data[key] = st.number_input(label, value=None, step=0.01, key=f"m_{key}")
                    
                    # Interventions
                    with st.expander("🏥 Interventions"):
                        cols = st.columns(5)
                        interv_data = {}
                        for idx, (key, label) in enumerate(INTERVENTIONS.items()):
                            with cols[idx % 5]:
                                interv_data[key] = st.checkbox(label, key=f"i_{key}")
                    
                    # Additional
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        urine_output = st.number_input("Urine Output (mL)", value=None, step=1.0)
                    with col2:
                        notes = st.text_area("Notes", height=100)
                    
                    submitted = st.form_submit_button("Add Event", type="primary", use_container_width=True)
                
                if submitted:
                    event_data = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'patient_id': selected_patient,
                        'event_number': next_event,
                        **static_info,
                        **vitals_data,
                        **bloodgas_data,
                        **heme_data,
                        **chem_data,
                        **liver_data,
                        **meds_data,
                        **{k: 1 if v else 0 for k, v in interv_data.items()},
                        'urine_output': urine_output,
                        'notes': notes
                    }
                    
                    if save_event(event_data):
                        st.success(f"✅ Event #{next_event} added for patient {selected_patient}!")
                        st.rerun()
    # Page 3: Predictions
    # --- NEW PAGE 4: Model Predictions ---
    elif page == "Model Predictions":
        st.title("🤖 Real-Time ICU Predictions")
        
        model_choice = st.selectbox("Select model:", ["GRU-D", "Random Forest"])

        patients = get_patient_list()
        if not patients:
            st.info("No patients available. Please register a patient first.")
            st.stop()

        if model_choice == "GRU-D":
            if model_error:
                st.error(f"Cannot run predictions. Model failed to load: {model_error}")
            else:
                st.subheader("🧠 GRU-D Mortality & LOS Prediction")
                
                selected_patient = st.selectbox("Select Patient for Prediction", patients, key="grud_patient")
                
                if selected_patient:
                    events_df = get_patient_events(selected_patient)
                    st.metric("Total Events for Patient", len(events_df))

                    if len(events_df) < 2:
                        st.warning("At least 2 events are required to make a meaningful prediction.")
                    else:
                        # Slider to select number of events (only show if more than 2 events)
                        if len(events_df) > 2:
                            num_events_to_use = st.slider("Select number of events to use for prediction:", 
                                                          min_value=2, 
                                                          max_value=len(events_df), 
                                                          value=len(events_df), 
                                                          step=1)
                        else:
                            num_events_to_use = 2
                            st.info("Using both available events for prediction.")
                        
                        # Slice the dataframe based on slider
                        events_for_pred = events_df.head(num_events_to_use)
                        
                        if st.button("Run GRU-D Prediction", type="primary", use_container_width=True):
                            with st.spinner("Analyzing patient data and making predictions..."):
                                mortality_risk, los_pred = predict_mortality(events_for_pred, model, scaler_data)
                            
                            if mortality_risk is not None and los_pred is not None:
                                st.success("Prediction complete!")
                                
                                if mortality_risk >= 0.7: risk_label, risk_emoji = "HIGH RISK ⚠️", "🔴"
                                elif mortality_risk >= 0.3: risk_label, risk_emoji = "MEDIUM RISK", "🟡"
                                else: risk_label, risk_emoji = "LOW RISK", "🟢"
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(label=f"{risk_emoji} MORTALITY RISK", 
                                              value=f"{mortality_risk*100:.1f}%",
                                              help=f"The model predicts a {risk_label} probability of in-hospital mortality based on the first {num_events_to_use} events.")
                                with col2:
                                    st.metric(label="📅 PREDICTED LENGTH OF STAY",
                                              value=f"{los_pred:.1f} days",
                                              help="The model's estimate for the total length of the ICU stay.")

                                st.divider()
                                st.subheader("Key Vitals Trajectory")
                                
                                plot_df = events_for_pred.copy()
                                vital_options = st.multiselect(
                                    "Select vitals to display:",
                                    options=[k for k in FEATURE_LABELS.keys()],
                                    default=['heart_rate', 'mean_bp', 'lactate', 'gcs_total']
                                )

                                if vital_options:
                                    fig = go.Figure()
                                    for vital in vital_options:
                                        if vital in plot_df.columns:
                                            fig.add_trace(go.Scatter(
                                                x=plot_df['event_number'],
                                                y=plot_df[vital],
                                                mode='lines+markers',
                                                name=FEATURE_LABELS.get(vital, vital)
                                            ))
                                    fig.update_layout(
                                        xaxis_title="Event Number",
                                        yaxis_title="Value",
                                        hovermode='x unified',
                                        height=400,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Prediction failed. Please check the logs or input data.")

        elif model_choice == "Random Forest":
            import pickle
            import traceback

            @st.cache_resource
            def load_rf_model():
                try:
                    with open("rf_mortality_model.pkl", "rb") as f:
                        model_artifacts = pickle.load(f)
                    
                    # Extract components from the saved artifacts
                    rf_model = model_artifacts['model']
                    scaler = model_artifacts['scaler']
                    feature_names = model_artifacts['feature_names']
                    dynamic_features = model_artifacts['dynamic_features']
                    static_features = model_artifacts['static_features']
                    
                    return rf_model, scaler, feature_names, dynamic_features, static_features, None
                except Exception as e:
                    return None, None, None, None, None, str(e)

            def create_aggregate_features(patient_events, dynamic_features):
                """Create aggregate features matching the training script"""
                features = {}
                
                # Event count features
                features['num_events'] = len(patient_events)
                features['event_density'] = len(patient_events) / 200.0
                
                # For each dynamic feature, compute aggregates
                for feature in dynamic_features:
                    if feature in patient_events.columns:
                        values = patient_events[feature].dropna()
                    else:
                        values = pd.Series([])
                    
                    if len(values) == 0:
                        features[f'{feature}_count'] = 0
                        features[f'{feature}_mean'] = 0
                        features[f'{feature}_std'] = 0
                        features[f'{feature}_min'] = 0
                        features[f'{feature}_max'] = 0
                        features[f'{feature}_first'] = 0
                        features[f'{feature}_last'] = 0
                        features[f'{feature}_trend'] = 0
                        features[f'{feature}_range'] = 0
                    else:
                        features[f'{feature}_count'] = len(values)
                        features[f'{feature}_mean'] = values.mean()
                        features[f'{feature}_std'] = values.std() if len(values) > 1 else 0
                        features[f'{feature}_min'] = values.min()
                        features[f'{feature}_max'] = values.max()
                        features[f'{feature}_first'] = values.iloc[0]
                        features[f'{feature}_last'] = values.iloc[-1]
                        
                        if len(values) > 1:
                            features[f'{feature}_trend'] = values.iloc[-1] - values.iloc[0]
                        else:
                            features[f'{feature}_trend'] = 0
                        
                        features[f'{feature}_range'] = values.max() - values.min()
                
                return features

            rf_model, rf_scaler, feature_names, dynamic_features_rf, static_features_rf, load_error = load_rf_model()

            if load_error:
                st.error(f"⚠️ Failed to load RF model: {load_error}")
            else:
                st.success("✅ Random Forest model loaded successfully.")
                with st.expander("🔍 Model Info"):
                    st.write(f"Total features: {len(feature_names)}")
                    st.write(f"Dynamic features: {len(dynamic_features_rf)}")
                    st.write(f"Static features: {len(static_features_rf)}")
                
                st.subheader("🌳 Random Forest Mortality Prediction")
                
                # Patient selection
                selected_patient = st.selectbox("Select Patient for Prediction", patients, key="rf_patient")
                
                if selected_patient:
                    events_df = get_patient_events(selected_patient)
                    
                    # Display patient info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Patient ID", selected_patient)
                    with col2:
                        st.metric("Total Events", len(events_df))
                    with col3:
                        static_info = get_patient_static_info(selected_patient)
                        st.metric("Age", static_info['age'])
                    
                    if len(events_df) < 1:
                        st.warning("At least 1 event is required to make a prediction.")
                    else:
                        # Slider to select number of events to use (only show if more than 1 event)
                        if len(events_df) > 1:
                            num_events_to_use = st.slider(
                                "Select number of events to use for prediction:", 
                                min_value=1, 
                                max_value=len(events_df), 
                                value=len(events_df), 
                                step=1,
                                key="rf_slider"
                            )
                        else:
                            num_events_to_use = 1
                            st.info("Using the only available event for prediction.")
                        
                        # Show preview of events being used
                        with st.expander("📋 Preview Events Being Used"):
                            st.dataframe(events_df.head(num_events_to_use)[['timestamp', 'event_number', 'heart_rate', 'mean_bp', 'temperature', 'spo2']], use_container_width=True)
                        
                        if st.button("Run Random Forest Prediction", type="primary", use_container_width=True):
                            with st.spinner("Preparing patient data and running prediction..."):
                                try:
                                    # Get events to use for prediction
                                    events_for_pred = events_df.head(num_events_to_use)
                                    
                                    # Create static features dictionary matching training format
                                    static_dict = {}
                                    static_dict['age'] = float(static_info['age'])
                                    static_dict['GENDER_f'] = 1 if static_info['gender'] == 'Female' else 0
                                    
                                    # Insurance features (one-hot)
                                    insurance_val = static_info.get('insurance', '').lower()
                                    for ins_type in ['medicaid', 'medicare', 'other']:
                                        static_dict[f'insurance_group_INS_{ins_type}'] = 1 if insurance_val == ins_type else 0
                                    
                                    # Ethnicity features (one-hot)
                                    ethnicity_val = static_info.get('ethnicity', '').lower()
                                    for eth_type in ['asian', 'black', 'latino', 'other', 'white']:
                                        static_dict[f'ethnicity_group_ETH_{eth_type}'] = 1 if ethnicity_val == eth_type else 0
                                    
                                    # Marital status features (one-hot)
                                    marital_val = static_info.get('marital_status', '').lower()
                                    for mar_type in ['divorced', 'married', 'single', 'unknown', 'widowed']:
                                        static_dict[f'marital_group_MAR_{mar_type}'] = 1 if marital_val == mar_type else 0
                                    
                                    # Admission type features (one-hot) - add all possible types
                                    admission_types = ['AMBULATORY OBSERVATION', 'DIRECT EMER.', 'DIRECT OBSERVATION', 
                                                       'ELECTIVE', 'EU OBSERVATION', 'EW EMER.', 'OBSERVATION ADMIT',
                                                       'SURGICAL SAME DAY ADMISSION', 'URGENT']
                                    admission_val = static_info.get('admission_type', '')
                                    for adm_type in admission_types:
                                        static_dict[f'admission_type_{adm_type}'] = 1 if admission_val.upper() == adm_type else 0
                                    
                                    # First care unit features (one-hot)
                                    care_units = ['Cardiac Vascular Intensive Care Unit (CVICU)', 
                                                  'Coronary Care Unit (CCU)',
                                                  'Intensive Care Unit (ICU)',
                                                  'Medical Intensive Care Unit (MICU)',
                                                  'Medical/Surgical Intensive Care Unit (MICU/SICU)',
                                                  'Medicine', 'Medicine/Cardiology Intermediate',
                                                  'Neuro Intermediate', 'Neuro Stepdown',
                                                  'Neuro Surgical Intensive Care Unit (Neuro SICU)',
                                                  'Neurology', 'PACU', 'Surgery/Trauma',
                                                  'Surgery/Vascular/Intermediate',
                                                  'Surgical Intensive Care Unit (SICU)',
                                                  'Trauma SICU (TSICU)']
                                    care_unit_val = static_info.get('first_care_unit', '')
                                    for unit in care_units:
                                        static_dict[f'first_careunit_{unit}'] = 1 if care_unit_val == unit else 0
                                    
                                    # Create aggregate features from events
                                    agg_features = create_aggregate_features(events_for_pred, dynamic_features_rf)
                                    
                                    # Combine static and aggregate features
                                    all_features_dict = {**static_dict, **agg_features}
                                    
                                    # Convert to DataFrame
                                    sample_df = pd.DataFrame([all_features_dict])
                                    
                                    # Ensure all training features are present
                                    for feat in feature_names:
                                        if feat not in sample_df.columns:
                                            sample_df[feat] = 0
                                    
                                    # Reorder columns to match training
                                    sample_df = sample_df[feature_names]
                                    
                                    # Scale features
                                    scaled_features = rf_scaler.transform(sample_df)
                                    
                                    # Make prediction
                                    proba = rf_model.predict_proba(scaled_features)[0][1]
                                    prediction = rf_model.predict(scaled_features)[0]
                                    
                                    st.success("Prediction complete!")
                                    
                                    # Display results
                                    if proba >= 0.7: risk_label, risk_emoji = "HIGH RISK ⚠️", "🔴"
                                    elif proba >= 0.3: risk_label, risk_emoji = "MEDIUM RISK", "🟡"
                                    else: risk_label, risk_emoji = "LOW RISK", "🟢"
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            label=f"{risk_emoji} MORTALITY RISK", 
                                            value=f"{proba*100:.1f}%",
                                            help=f"Random Forest predicts {risk_label}"
                                        )
                                    with col2:
                                        prediction_text = "High Risk" if prediction == 1 else "Low Risk"
                                        st.metric(
                                            label="CLASSIFICATION",
                                            value=prediction_text,
                                            help="Binary classification result"
                                        )
                                    
                                    # Progress bar for visual representation
                                    st.divider()
                                    st.subheader("Risk Assessment")
                                    st.progress(proba)
                                    
                                    # =========================================================
                                    # =========== UPDATED CODE BLOCK STARTS HERE ==============
                                    # =========================================================
                                    with st.expander("🔬 Top 30 Risk Factors with Patient Values"):
                                        # 1. Define the "pretty names" from your image
                                        top_features_pretty_names = [
                                            "GCS (min)", "Urineoutput", "Resp rate (min)", "Ventilation", "Activity (bed)",
                                            "CCT score", "GNBRI", "Code status", "Age", "BUN (min)", "PreICULOS day",
                                            "Heart rate (min)", "Aniongap (min)", "SpO2 (min)", "BUN/Creatinine",
                                            "Shock index", "Temp (min)", "Platelet (min)", "Lactate (min)", "AST (min)",
                                            "SBP (min)", "PTT (min)", "ALT (min)", "Chloride (min)", "Base excess (min)",
                                            "Neutrophils (min)", "PaO2/FiO2 (no-vent)", "eGFR", "PaCO2 (min)", "MBP (min)"
                                        ]

                                        # 2. Map pretty names to the actual feature names calculated in the script
                                        feature_mapping = {
                                            "GCS (min)": "gcs_total_min",
                                            "Resp rate (min)": "respiratory_rate_min",
                                            "Age": "age",
                                            "BUN (min)": "bun_min",
                                            "Heart rate (min)": "heart_rate_min",
                                            "SpO2 (min)": "spo2_min",
                                            "Temp (min)": "temperature_min",
                                            "Platelet (min)": "platelet_min",
                                            "Lactate (min)": "lactate_min",
                                            "SBP (min)": "systolic_bp_min",
                                            "MBP (min)": "mean_bp_min",
                                            # Note: Add more mappings here if you add the features to your data collection
                                        }

                                        # 3. Display in two columns
                                        col1, col2 = st.columns(2)
                                        
                                        # Function to display a single feature
                                        def display_feature(pretty_name, rank):
                                            code_name = feature_mapping.get(pretty_name)
                                            value = "N/A"
                                            if code_name and code_name in all_features_dict:
                                                raw_value = all_features_dict[code_name]
                                                if pd.notna(raw_value):
                                                    value = f"{raw_value:.1f}"
                                            st.markdown(f"**{rank}. {pretty_name}:** `{value}`")

                                        with col1:
                                            for i in range(15): # First 15 features
                                                display_feature(top_features_pretty_names[i], i + 1)
                                        with col2:
                                            for i in range(15, 30): # Next 15 features
                                                display_feature(top_features_pretty_names[i], i + 1)
                                    # =========================================================
                                    # ============ UPDATED CODE BLOCK ENDS HERE ===============
                                    # =========================================================
                                    
                                    # Define latest_event before using it
                                    latest_event = events_for_pred.iloc[0]

                                    st.divider()
                                    st.subheader("Key Vitals from Latest Event")
                                    
                                    vital_cols = st.columns(4)
                                    key_vitals = ['heart_rate', 'mean_bp', 'temperature', 'spo2', 'lactate', 'creatinine', 'gcs_total', 'wbc']
                                    for idx, vital in enumerate(key_vitals[:8]):
                                        with vital_cols[idx % 4]:
                                            value = latest_event.get(vital, 'N/A')
                                            if pd.notna(value):
                                                st.metric(FEATURE_LABELS.get(vital, vital), f"{value:.1f}")
                                            else:
                                                st.metric(FEATURE_LABELS.get(vital, vital), "N/A")
                                    
                                    # Trend visualization
                                    st.divider()
                                    st.subheader("Vital Signs Trend")
                                    vital_options = st.multiselect(
                                        "Select vitals to display:",
                                        options=[k for k in FEATURE_LABELS.keys() if k in events_for_pred.columns],
                                        default=['heart_rate', 'mean_bp', 'spo2'],
                                        key="rf_vitals"
                                    )
                                    
                                    if vital_options:
                                        fig = go.Figure()
                                        for vital in vital_options:
                                            if vital in events_for_pred.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=events_for_pred['event_number'],
                                                    y=events_for_pred[vital],
                                                    mode='lines+markers',
                                                    name=FEATURE_LABELS.get(vital, vital)
                                                ))
                                        fig.update_layout(
                                            xaxis_title="Event Number",
                                            yaxis_title="Value",
                                            hovermode='x unified',
                                            height=400,
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                except Exception as e:
                                    st.error(f"Prediction failed: {str(e)}")
                                    st.code(traceback.format_exc())
    
    # PAGE 3: View Data
    elif page == "View Data":
        st.header("View Patient Data")
        
        patients = get_patient_list()
        if not patients:
            st.info("No patient data available yet.")
        else:
            selected_patient = st.selectbox("Select Patient to View", patients)
            
            if selected_patient:
                events_df = get_patient_events(selected_patient)
                
                st.subheader(f"Patient: {selected_patient}")
                st.metric("Total Events", len(events_df))
                
                st.divider()
                
                # Show data in tabs
                tab1, tab2, tab3 = st.tabs(["📋 All Events", "📈 Latest Values", "🔍 Search"])
                
                with tab1:
                    st.dataframe(events_df, use_container_width=True, height=600)
                
                with tab2:
                    if len(events_df) > 0:
                        latest = events_df.iloc[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**Vital Signs**")
                            for key in VITAL_SIGNS.keys():
                                if pd.notna(latest[key]):
                                    st.metric(VITAL_SIGNS[key], f"{latest[key]}")
                        
                        with col2:
                            st.write("**Key Labs**")
                            for key in ['lactate', 'creatinine', 'wbc', 'platelet']:
                                if pd.notna(latest[key]):
                                    st.metric(LABS_CHEMISTRY.get(key) or LABS_HEMATOLOGY.get(key), f"{latest[key]}")
                        
                        with col3:
                            st.write("**Interventions**")
                            for key in INTERVENTIONS.keys():
                                if latest[key] == 1:
                                    st.success(f"✓ {INTERVENTIONS[key]}")
                
                with tab3:
                    search_col = st.selectbox("Search by column", events_df.columns.tolist())
                    search_val = st.text_input("Search value")
                    if search_val:
                        filtered = events_df[events_df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
                        st.dataframe(filtered, use_container_width=True)
    
    # PAGE 4: Export Data
    elif page == "Export Data":
        st.header("Export Patient Data")
        
        df = load_data()
        
        if len(df) == 0:
            st.info("No data to export yet.")
        else:
            st.subheader("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export All Data**")
                csv_all = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download All Events (CSV)",
                    data=csv_all,
                    file_name=f"icu_all_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.write("**Export by Patient**")
                patients = get_patient_list()
                if patients:
                    export_patient = st.selectbox("Select Patient", patients)
                    if export_patient:
                        patient_df = df[df['patient_id'] == export_patient]
                        csv_patient = patient_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"📥 Download {export_patient} Events",
                            data=csv_patient,
                            file_name=f"icu_{export_patient}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            st.divider()
            st.subheader("Preview Data")
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df)} records")

if __name__ == "__main__":
    main()
