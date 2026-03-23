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

st.write(
    "Enter film thickness, surface energy, and surface roughness to predict the deposition pattern."
)

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