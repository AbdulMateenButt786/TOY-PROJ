import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


st.set_page_config(page_title="Placement Predictor", page_icon="🎓")
st.title("Student Placement Predictor")

st.markdown("Enter Student Details Below:")

cgpa = st.number_input("Enter CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, step=0.01)
iq = st.number_input("Enter IQ (e.g., 70 - 160)", min_value=50, max_value=200, step=1)


if st.button("Predict Placement"):
    input_data = np.array([[cgpa, iq]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.success("The student is likely to be placed!")
    else:
        st.error("The student is not likely to be placed.")
