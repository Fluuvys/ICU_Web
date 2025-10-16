import streamlit as st
import pandas as pd
from datetime import datetime
import os

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

def main():
    st.set_page_config(page_title="ICU Time-Series Data Collection", layout="wide")
    st.title("🏥 ICU Time-Series Data Collection System")
    
    init_csv()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Mode:", 
                       ["📝 New Patient", "➕ Add Event", "📊 View Data", "💾 Export Data"])
        
        st.divider()
        patient_count = len(get_patient_list())
        total_events = len(load_data())
        st.metric("Total Patients", patient_count)
        st.metric("Total Events", total_events)
    
    # PAGE 1: New Patient Registration
    if page == "📝 New Patient":
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
    elif page == "➕ Add Event":
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
    
    # PAGE 3: View Data
    elif page == "📊 View Data":
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
    elif page == "💾 Export Data":
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
