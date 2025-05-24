import streamlit as st
import numpy as np
import pandas as pd
import joblib

# UI settings
st.set_page_config(page_title="Hypertension Risk Predictor", layout="centered")
st.title("🩺 Hypertension Risk Predictor")
st.markdown("Predict the likelihood of developing hypertension based on clinical and lifestyle features.")


#load model
log_model = joblib.load("logistic_regression_auc_0.9485.pkl")
rf_model = joblib.load("random_forest_auc_0.9533.pkl")
xgb_model = joblib.load("xgboost_auc_0.9479.pkl")
scaler = joblib.load("scaler.pkl")

# Model dictionary with AUC
model_dict = {
    "Random Forest": (rf_model, 0.9533),
    "Logistic Regression": (log_model, 0.9485),
    "XGBoost": (xgb_model, 0.9479)
}

# Sidebar model selection
model_choice = st.sidebar.selectbox("Select Model", list(model_dict.keys()), index=0)
model, auc_score = model_dict[model_choice]

# User Inputs
st.markdown("### Enter Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 45)
    cigs = st.slider("Cigarettes Per Day", 0, 60, 0)
    totChol = st.number_input("Total Cholesterol", min_value=100.0, max_value=600.0, value=200.0)
    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
    heartRate = st.slider("Heart Rate", 40, 150, 75)

with col2:
    sysBP = st.slider("Systolic BP", 80, 200, 120)
    diaBP = st.slider("Diastolic BP", 50, 120, 80)
    glucose = st.number_input("Glucose", min_value=50.0, max_value=300.0, value=100.0)

    male = st.selectbox("Sex", ["Male", "Female"])
    currentSmoker = st.selectbox("Current Smoker?", ["Yes", "No"])
    BPMeds = st.selectbox("On BP Medications?", ["Yes", "No"])
    diabetes = st.selectbox("Has Diabetes?", ["Yes", "No"])

    import streamlit as st
    import numpy as np
    import pandas as pd
    import joblib

    # Load models
    log_model = joblib.load("logistic_regression_auc_0.9485.pkl")
    rf_model = joblib.load("random_forest_auc_0.9533.pkl")
    xgb_model = joblib.load("xgboost_auc_0.9479.pkl")
    scaler = joblib.load("scaler.pkl")

    # Model dictionary with AUC
    model_dict = {
        "Random Forest": (rf_model, 0.9533),
        "Logistic Regression": (log_model, 0.9485),
        "XGBoost": (xgb_model, 0.9479)
    }

    # UI settings
    st.set_page_config(page_title="Hypertension Risk Predictor", layout="centered")
    st.title("🩺 Hypertension Risk Predictor")
    st.markdown("Predict the likelihood of developing hypertension based on clinical and lifestyle features.")

    # Sidebar model selection
    model_choice = st.sidebar.selectbox("Select Model", list(model_dict.keys()), index=0)
    model, auc_score = model_dict[model_choice]

    # User Inputs
    st.markdown("### Enter Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 45)
        cigs = st.slider("Cigarettes Per Day", 0, 60, 0)
        totChol = st.number_input("Total Cholesterol", min_value=100.0, max_value=600.0, value=200.0)
        BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
        heartRate = st.slider("Heart Rate", 40, 150, 75)

    with col2:
        sysBP = st.slider("Systolic BP", 80, 200, 120)
        diaBP = st.slider("Diastolic BP", 50, 120, 80)
        glucose = st.number_input("Glucose", min_value=50.0, max_value=300.0, value=100.0)

        male = st.selectbox("Sex", ["Male", "Female"])
        currentSmoker = st.selectbox("Current Smoker?", ["Yes", "No"])
        BPMeds = st.selectbox("On BP Medications?", ["Yes", "No"])
        diabetes = st.selectbox("Has Diabetes?", ["Yes", "No"])

    # Encoding
    male = 1 if male == "Male" else 0
    currentSmoker = 1 if currentSmoker == "Yes" else 0
    BPMeds = 1 if BPMeds == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0

    # Input Data
    input_data = pd.DataFrame([[
        male, age, currentSmoker, cigs, BPMeds,
        diabetes, totChol, sysBP, diaBP, BMI,
        heartRate, glucose
    ]], columns=[
        'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
        'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
        'heartRate', 'glucose'
    ])

    # Prediction
    if st.button("🔍 Predict Hypertension Risk"):
        if model_choice == "Logistic Regression":
            input_scaled = scaler.transform(input_data)
            prob = model.predict_proba(input_scaled)[0][1]
        else:
            prob = model.predict_proba(input_data)[0][1]

        st.markdown(f"<h2 style='color:#EF476F;'>Hypertension Risk: {prob:.2%}</h2>", unsafe_allow_html=True)
        st.info(f"🔢 Model Used: {model_choice} | AUC Score: {auc_score:.4f}")

        if prob > 0.5:
            st.error("⚠️ High Risk: Clinical attention recommended.")
        else:
            st.success("✅ Low Risk: Maintain a healthy lifestyle.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Part of the NCD Risk Suite • Built by Oluwafemi</p>", unsafe_allow_html=True)