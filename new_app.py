import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("synthetic_genetic_epigenetic_data.csv")

# Define features and target variables
X = df.drop(columns=["Serotonin_Level", "Dopamine_Level"])
y_serotonin = df["Serotonin_Level"]
y_dopamine = df["Dopamine_Level"]

# Split into training and testing sets
X_train, X_test, y_train_serotonin, y_test_serotonin = train_test_split(X, y_serotonin, test_size=0.2, random_state=42)
X_train, X_test, y_train_dopamine, y_test_dopamine = train_test_split(X, y_dopamine, test_size=0.2, random_state=42)

# Train models
model_serotonin = RandomForestRegressor(n_estimators=100, random_state=42)
model_dopamine = RandomForestRegressor(n_estimators=100, random_state=42)

model_serotonin.fit(X_train, y_train_serotonin)
model_dopamine.fit(X_train, y_train_dopamine)

# Save models
with open("serotonin_dopamine_model.pkl", "wb") as f:
    pickle.dump({"serotonin_model": model_serotonin, "dopamine_model": model_dopamine}, f)

print("Model saved successfully!")
