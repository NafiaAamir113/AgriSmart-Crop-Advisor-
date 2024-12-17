# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import streamlit as st

# Title and description
st.set_page_config(
    page_title="AgriSmart Crop Advisor 🌾",
    page_icon="🌱",
    layout="centered",
)

st.title("🌾 AgriSmart Crop Advisor 🌱")
st.write(
    "Welcome to **AgriSmart Crop Advisor**! This app helps farmers and agricultural consultants predict the best crop to grow based on soil, weather, and location data. 🚜💡"
)
st.subheader("Upload Your Crop Data CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_and_preprocess_data(file):
    if file is not None:
        data = pd.read_csv(file)
        
        # Handle missing values
        data = data.dropna()
        
        # Encode label column
        label_encoder = LabelEncoder()
        data["label"] = label_encoder.fit_transform(data["label"])
        
        # One-hot encode the 'district' column
        data = pd.get_dummies(data, columns=["district"], drop_first=True)
        
        # Features and Target
        X = data.drop("label", axis=1)
        y = data["label"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model, label_encoder, X.columns, data
    else:
        st.warning("Please upload a CSV file first.")
        return None, None, None, None

if uploaded_file is not None:
    model, label_encoder, feature_columns, data = load_and_preprocess_data(uploaded_file)
else:
    st.stop()  # Prevent further execution until a file is uploaded

# Load Data and Train Model 
@st.cache_data
def load_and_preprocess_data():
    # Step 2: Load the dataset
    file_path = "Crop(Distric level).csv"  
    data = pd.read_csv(file_path)

    # Handle missing values
    data = data.dropna()

    # Encode label column
    label_encoder = LabelEncoder()
    data["label"] = label_encoder.fit_transform(data["label"])

    # One-hot encode the 'district' column
    data = pd.get_dummies(data, columns=["district"], drop_first=True)

    # Features and Target
    X = data.drop("label", axis=1)
    y = data["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, label_encoder, X.columns, data

# Load and preprocess data
model, label_encoder, feature_columns, data = load_and_preprocess_data()

# User Input Interface
st.subheader("🌟 Enter Input Values for Crop Prediction 🌟")

# Input fields for user
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌱 Nitrogen (N):", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    P = st.number_input("🌿 Phosphorus (P):", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    K = st.number_input("🍂 Potassium (K):", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
    ph = st.number_input("🧪 Soil pH:", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

with col2:
    temperature = st.number_input("🌡️ Temperature (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("💧 Humidity (%):", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    rainfall = st.number_input("🌧️ Rainfall (mm):", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
    district_input = st.text_input("📍 District Name (e.g., ryk):", "ryk")

# Submit button
if st.button("🚀 Predict Crop"):
    # Prepare input data
    new_data_raw = pd.DataFrame(
        {
            "N": [N],
            "P": [P],
            "K": [K],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall],
        }
    )

    # Add one-hot encoded district columns
    for col in feature_columns:
        if "district_" in col:
            new_data_raw[col] = 1 if col == f"district_{district_input}" else 0

    # Add missing columns with 0 (if any)
    for col in feature_columns:
        if col not in new_data_raw.columns:
            new_data_raw[col] = 0

    # Reorder columns to match training data
    new_data_raw = new_data_raw[feature_columns]

    # Predict crop type
    new_data_encoded = model.predict(new_data_raw)
    predicted_crop = label_encoder.inverse_transform(new_data_encoded)

    # Display result
    st.success(f"🌾 **Predicted Crop Type:** {predicted_crop[0]} 🌱")
    st.balloons()

# Footer 
st.write("**Developed by Us3 with 💚 for Smart Agriculture 🚜.**")
st.markdown("Stay sustainable, stay productive! 🌍")


