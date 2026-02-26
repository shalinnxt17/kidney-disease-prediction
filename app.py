import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="CKD Prediction System",
    layout="wide"
)

st.title("🩺 Chronic Kidney Disease Prediction System")
st.write("Predict CKD using Machine Learning (SVM & Decision Tree)")

# -----------------------------
# Safe base directory (IMPORTANT FOR STREAMLIT CLOUD)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load models safely
# -----------------------------
@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(BASE_DIR, "models", "imputer.pkl"), "rb") as f:
        imputer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "models", "svm_model.pkl"), "rb") as f:
        svm_model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "models", "dt_model.pkl"), "rb") as f:
        dt_model = pickle.load(f)

    return scaler, imputer, svm_model, dt_model


try:
    scaler, imputer, svm_model, dt_model = load_models()
except Exception as e:
    st.error("❌ Failed to load model files")
    st.exception(e)
    st.stop()

# -----------------------------
# Input fields (12 features example)
# -----------------------------
st.sidebar.header("Patient Inputs")

age = st.sidebar.number_input("Age", 1, 120, 45)
bp = st.sidebar.number_input("Blood Pressure", 50, 200, 80)
sg = st.sidebar.number_input("Specific Gravity", 1.000, 1.030, 1.020, format="%.3f")
al = st.sidebar.number_input("Albumin", 0, 5, 1)
su = st.sidebar.number_input("Sugar", 0, 5, 0)
bgr = st.sidebar.number_input("Blood Glucose Random", 50, 500, 120)
bu = st.sidebar.number_input("Blood Urea", 1, 200, 36)
sc = st.sidebar.number_input("Serum Creatinine", 0.1, 15.0, 1.2)
sod = st.sidebar.number_input("Sodium", 100, 170, 140)
pot = st.sidebar.number_input("Potassium", 2.0, 8.0, 4.5)
hemo = st.sidebar.number_input("Hemoglobin", 3.0, 20.0, 12.0)
pcv = st.sidebar.number_input("Packed Cell Volume", 10, 60, 40)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict"):
    input_data = np.array([[age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv]])

    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    svm_pred = svm_model.predict(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]

    st.subheader("🔍 Prediction Result")

    if svm_pred == 1:
        st.error("⚠️ CKD Detected (SVM)")
    else:
        st.success("✅ No CKD Detected (SVM)")

    if dt_pred == 1:
        st.error("⚠️ CKD Detected (Decision Tree)")
    else:
        st.success("✅ No CKD Detected (Decision Tree)")