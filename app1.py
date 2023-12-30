import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib

# Load the scaler
scaler = joblib.load("scaler.pkl")  # Load your saved scaler

model = tf.keras.models.load_model("model.h5")

def predict_residues(LL, PI, DPI, CF):
    # Preprocess the input using the loaded scaler
    input_data = np.array([[LL, PI, DPI, CF]])
    scaled_input = scaler.transform(input_data)

    # Make a prediction using the model
    prediction = model.predict(scaled_input)

    return prediction  # Return the prediction

def main():
    st.title("Residual Shear Strength of Clay")
    st.markdown(
        """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Streamlit Geotech ML App</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    LL = st.text_input("LL", "Input Liquid Limit")
    PI = st.text_input("PI", "Input Plasticity Index")
    DPI = st.text_input("DPI", "Input Change in PI")
    CF = st.text_input("CF", "Input Clay Fraction")

    try:
        LL = float(LL)
        PI = float(PI)
        DPI = float(DPI)
        CF = float(CF)
    except ValueError:
        st.error("Please enter numeric values for LL, PI, DPI, and CF.")
        st.stop()

    result = ""
    if st.button("Predict"):
        result = predict_residues(LL, PI, DPI, CF)
        st.success("The predicted residual shear strength is {}".format(result))

    if st.button("About"):
        st.text("This app is used for predicting the residual shear strength of clay")
        st.text("Built with Streamlit")

if __name__ == "__main__":
    main()
