import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Load models & preprocessors
# -----------------------------
svm_model = pickle.load(open("models/svm_model.pkl", "rb"))
dt_model = pickle.load(open("models/dt_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
imputer = pickle.load(open("models/imputer.pkl", "rb"))

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Chronic Kidney Disease Prediction System")
st.write("Predict Chronic Kidney Disease (CKD) using Machine Learning")

# -----------------------------
# Sidebar inputs (12 FEATURES)
# -----------------------------
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 120, 45)
bp = st.sidebar.number_input("Blood Pressure", 50, 200, 80)
sg = st.sidebar.number_input("Specific Gravity", 1.000, 1.030, 1.020, step=0.001)
al = st.sidebar.number_input("Albumin", 0, 5, 1)
su = st.sidebar.number_input("Sugar", 0, 5, 0)

bgr = st.sidebar.number_input("Blood Glucose Random", 50, 500, 120)
bu = st.sidebar.number_input("Blood Urea", 1, 300, 40)
sc = st.sidebar.number_input("Serum Creatinine", 0.1, 20.0, 1.2)
sod = st.sidebar.number_input("Sodium", 100, 200, 140)
pot = st.sidebar.number_input("Potassium", 2.0, 7.0, 4.5)

hemo = st.sidebar.number_input("Hemoglobin", 3.0, 20.0, 12.0)
pcv = st.sidebar.number_input("Packed Cell Volume", 10, 60, 40)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict"):

    # Arrange inputs in training order
    input_data = np.array([[
        age, bp, sg, al, su,
        bgr, bu, sc, sod, pot,
        hemo, pcv
    ]])

    # Preprocessing
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    # Predictions
    svm_pred = svm_model.predict(input_data)[0]
    svm_prob = svm_model.predict_proba(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]

    # -----------------------------
    # Result display
    # -----------------------------
    st.subheader("🔍 Prediction Result")

    if svm_pred == 1:
        st.error("⚠️ CKD Detected")
        confidence = svm_prob[1] * 100
    else:
        st.success("✅ No CKD Detected")
        confidence = svm_prob[0] * 100

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.info(
        "ℹ️ Confidence represents the probability output of the SVM model. "
        "Lower confidence indicates the patient lies close to the decision boundary."
    )

    # -----------------------------
    # Model comparison (Text)
    # -----------------------------
    st.subheader("🤖 Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "SVM Prediction",
            "CKD" if svm_pred == 1 else "Not CKD"
        )

    with col2:
        st.metric(
            "Decision Tree Prediction",
            "CKD" if dt_pred == 1 else "Not CKD"
        )

    # -----------------------------
    # Graph 1: Accuracy Comparison
    # -----------------------------
    st.subheader("📊 Model Accuracy Comparison")

    models = ["SVM", "Decision Tree"]
    accuracies = [0.9875, 0.9875]  # from your training output

    fig1, ax1 = plt.subplots()
    ax1.bar(models, accuracies)
    ax1.set_ylim(0.9, 1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Comparison")

    st.pyplot(fig1)

    # -----------------------------
    # Graph 2: Prediction Probability
    # -----------------------------
    st.subheader("📊 SVM Prediction Probability")

    labels = ["Not CKD", "CKD"]
    probabilities = svm_prob

    fig2, ax2 = plt.subplots()
    ax2.bar(labels, probabilities)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction Confidence")

    st.pyplot(fig2)