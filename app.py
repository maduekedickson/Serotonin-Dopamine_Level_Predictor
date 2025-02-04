import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("serotonin_dopamine_model.pkl", "rb") as f:
    models = pickle.load(f)
    serotonin_model = models["serotonin_model"]
    dopamine_model = models["dopamine_model"]

# Streamlit UI
st.image("images.jpg")
st.title("Serotonin & Dopamine Level Predictor")
st.write("Enter the feature values to predict neurotransmitter levels.")

# User input fields
drd2 = st.number_input("DRD2_rs1800497 (0, 1, or 2)", min_value=0, max_value=2, value=1)
slc6a4 = st.number_input("SLC6A4_rs25531 (0, 1, or 2)", min_value=0, max_value=2, value=1)
comt = st.number_input("COMT_rs4680 (0, 1, or 2)", min_value=0, max_value=2, value=1)
maoa = st.number_input("MAOA_rs6323 (0, 1, or 2)", min_value=0, max_value=2, value=1)
htr2a = st.number_input("HTR2A_rs6313 (0, 1, or 2)", min_value=0, max_value=2, value=1)

drd2_methyl = st.number_input("DRD2 Methylation Level", value=50.0)
slc6a4_methyl = st.number_input("SLC6A4 Methylation Level", value=50.0)
comt_methyl = st.number_input("COMT Methylation Level", value=50.0)
maoa_methyl = st.number_input("MAOA Methylation Level", value=50.0)
htr2a_methyl = st.number_input("HTR2A Methylation Level", value=50.0)

drug_exposure = st.number_input("Drug Exposure Years", min_value=0, max_value=50, value=5)
susceptibility_score = st.number_input("Addiction Susceptibility Score", min_value=0.0, max_value=1.0, value=0.5)

# Convert input to numpy array
features = np.array([[drd2, slc6a4, comt, maoa, htr2a,
                      drd2_methyl, slc6a4_methyl, comt_methyl, maoa_methyl, htr2a_methyl,
                      drug_exposure, susceptibility_score]])

# Predict
if st.button("Predict"): 
    serotonin_pred = serotonin_model.predict(features)[0]
    dopamine_pred = dopamine_model.predict(features)[0]
    
    st.success(f"Predicted Serotonin Level: {serotonin_pred:.2f}")
    st.success(f"Predicted Dopamine Level: {dopamine_pred:.2f}")
