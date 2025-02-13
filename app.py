import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("serotonin_dopamine_model.pkl", "rb") as f:
    models = pickle.load(f)
    serotonin_model = models["serotonin_model"]
    dopamine_model = models["dopamine_model"]

# App Header
st.image("images.jpg")
st.title("Serotonin & Dopamine Level Predictor")

# Model Description
st.markdown("""
### About This Model  
This AI model predicts serotonin and dopamine levels based on genetic markers, methylation levels, and drug exposure history.  
- **Serotonin & Dopamine:** Key neurotransmitters that regulate mood, motivation, and cognitive function.  
- **Genetic Markers:** Variations in genes like **DRD2, SLC6A4, COMT, MAOA, and HTR2A** influence neurotransmitter production.  
- **Methylation Levels:** Epigenetic modifications affecting gene expression.  
- **Drug Exposure & Susceptibility Score:** Indicators of potential neurotransmitter imbalances.  

These predictions can provide insights into the impact of genetic and environmental factors on neurotransmitter levels.
""")

st.write("Enter the feature values to predict neurotransmitter levels.")

# User input fields
st.markdown("### Genetic Markers (Enter 0, 1, or 2)")
drd2 = st.number_input("DRD2_rs1800497", min_value=0, max_value=2, value=1)
slc6a4 = st.number_input("SLC6A4_rs25531", min_value=0, max_value=2, value=1)
comt = st.number_input("COMT_rs4680", min_value=0, max_value=2, value=1)
maoa = st.number_input("MAOA_rs6323", min_value=0, max_value=2, value=1)
htr2a = st.number_input("HTR2A_rs6313", min_value=0, max_value=2, value=1)

st.markdown("### Epigenetic Markers (Methylation Levels)")
drd2_methyl = st.number_input("DRD2 Methylation Level", value=50.0)
slc6a4_methyl = st.number_input("SLC6A4 Methylation Level", value=50.0)
comt_methyl = st.number_input("COMT Methylation Level", value=50.0)
maoa_methyl = st.number_input("MAOA Methylation Level", value=50.0)
htr2a_methyl = st.number_input("HTR2A Methylation Level", value=50.0)

st.markdown("### Environmental Factors")
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

    # Interpretation
    st.markdown("""
    ### Understanding the Predictions  
    - **Higher serotonin levels** are linked to positive mood, while lower levels may indicate depression or anxiety.  
    - **Higher dopamine levels** are associated with motivation and reward, while lower levels may be linked to addiction risk or cognitive issues.  
    """)

