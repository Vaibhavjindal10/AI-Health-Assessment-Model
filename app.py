import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load('risk_model.pkl')

st.title("🩺 Medical Risk Predictor")

st.write("Enter patient details to check their risk category:")

# Input fields
hr = st.number_input("Heart Rate", min_value=30, max_value=200 , value=78)
rr = st.number_input("Respiratory Rate", min_value=5, max_value=50, value=18)
bt = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
spo2 = st.number_input("Oxygen Saturation (%)", min_value=50, max_value=100, value=98)
# sbp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
# dbp = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
# hrv = st.number_input("Derived HRV", min_value=0, max_value=100, value=40)
pulse_pressure = st.number_input("Derived Pulse Pressure", min_value=0, max_value=150, value=40)
# bmi = st.number_input("Derived BMI", min_value=10.0, max_value=50.0, value=24.0)
map_val = st.number_input("Derived MAP", min_value=50.0, max_value=150.0, value=93.0)

# Encode gender
gender_code = 1 if gender == "Male" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[hr, rr, bt, spo2, age, gender_code, weight, height,
                            pulse_pressure,  map_val]])

    # input_data = input_data.drop(columns=['Derived_HRV', 'Derived_BMI', 'Systolic Blood Pressure', 'Diastolic Blood Pressure'])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.subheader("🔍 Prediction Result:")
    st.write(f"**Risk Category:** {'High Risk' if prediction == 1 else 'Low Risk'}")
    st.write(f"**Confidence:** {prob[prediction]*100:.2f}%")

