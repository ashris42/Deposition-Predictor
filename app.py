import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Deposition Pattern Predictor", layout="centered")

# Load model
model = joblib.load("SVM_SPLIT_02_MODEL.joblib")
scaler = joblib.load("scaler_Split_02.joblib")

# Label mapping
label_map = {
    0: "Coffee Stain",
    1: "Inverse Coffee Stain",
    2: "No Coffee Stain"
}

# Title
st.title("Deposition Pattern Predictor")

st.write("A machine learning model to predict deposition patterns")


# Inputs (no +/- buttons)
thickness = st.text_input("Film Thickness(mm)")
energy = st.text_input("Surface Energy(mJ/m^2)")
roughness = st.text_input("Surface Roughness (SDR %)")


# Predict
if st.button("Predict"):
    try:
        thickness_val = float(thickness)
        energy_val = float(energy)
        roughness_val = float(roughness)

        X = np.array([[thickness_val, energy_val, roughness_val]])

        # Apply scaler
        X = scaler.transform(X)

        pred = model.predict(X)[0]
        label = label_map[pred]

        # ✅ Green output box
        st.success(f"Predicted Pattern: {label}")

    except:
        st.error("Please enter valid numeric values.")
        
# LOGO
import base64

def get_base64(img_file):
    with open(img_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("logo.png")

st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{img_base64}" width="200">
    </div>
    """,
    unsafe_allow_html=True
)

st.write("This ML model was developed by Aditya Sinha and Pratyush Padhy under the supervision of Professor Arnab Dutta (in collaboration with Professor Nandini Bhandaru) of BITS Pilani, Hyderabad Campus. This webapp can be used to predict deposition patterns. The classes being predicted are - Coffee Stain, Inverse Coffee Stain and No Coffee Stain.")
 
st.write("Kindly ensure that all feature values are positive. The input requirements for the features are as follows:")

st.write("""
1. Film thickness must be entered in millimeters (mm)

2. Surface energy must be provided in mJ/m²

3. Surface roughness must be given as SDR %

""")

st.write(
"This web application has been developed as part of academic work. "
"We do not take responsibility for any decisions or outcomes based on its predictions."
)

st.write("For any queries, please send an email to - f20230942@hyderabad.bits-pilani.ac.in (Aditya Sinha), f20231264@hyderabad.bits-pilani.ac.in (Pratyush Padhy) or arnabdutta@hyderabad.bits-pilani.ac.in (Arnab Dutta)")      
