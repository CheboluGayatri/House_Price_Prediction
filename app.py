import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = os.path.join("models", "house_price_model.joblib")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please make sure the model is saved at 'models/house_price_model.joblib'.")
    st.stop()

model_bundle = joblib.load(MODEL_PATH)
model, feature_names = model_bundle

# ----------------------------
# Streamlit UI Config
# ----------------------------
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

# ----------------------------
# Add Background Image and Custom CSS
# ----------------------------
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background-image: url('https://png.pngtree.com/background/20250422/original/pngtree-a-3d-model-of-house-on-architectural-blueprints-surrounded-by-rising-picture-image_16420659.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    /* Main content container */
    .css-18e3th9 {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
    }

    /* Title */
    .stApp h1 {
        color: #ffffff;
        font-weight: 800;
        text-shadow: 2px 2px 4px #000000;
    }

    /* Subtitle / description */
    .stApp p, .stApp .markdown-text-container {
        color: #ffffff;
        font-weight: 600;
        text-shadow: 1px 1px 3px #000000;
    }

    /* Prediction result styling */
    .stAlertSuccess {
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: #b7f0c1 !important; /* light green */
        font-weight: 800;
        font-size: 24px;
        text-align: center;
        border-radius: 10px;
        padding: 15px;
    }

    /* Footer */
    .stApp .stMarkdown p {
        color: #ffffff;
        font-weight: 600;
        font-size: 16px;
        text-align: center;
        text-shadow: 1px 1px 2px #000000;
    }

    /* Sidebar style */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
    }
    .css-1d391kg label, .css-1d391kg div {
        color: #ffffff !important;
    }

    /* Button styling */
    div.stButton > button:first-child {
        background-color: #00b386;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 8px 20px;
        text-align: center;
    }
    div.stButton > button:first-child:hover {
        background-color: #00e6a6;
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Title and description
# ----------------------------
st.title("🏡 House Price Prediction App")
st.markdown("Fill in the details below to predict the house price.")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Enter House Details")

# Numeric Inputs
area = st.sidebar.number_input("Total Area (sq ft)", min_value=100.0, max_value=10000.0, step=10.0)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4])
stories = st.sidebar.selectbox("Number of Stories", [1, 2, 3, 4])
parking = st.sidebar.selectbox("Parking Spaces", [0, 1, 2, 3])

# Categorical Inputs
mainroad = st.sidebar.radio("Main Road Access?", ['yes', 'no'])
guestroom = st.sidebar.radio("Guest Room?", ['yes', 'no'])
basement = st.sidebar.radio("Basement?", ['yes', 'no'])
hotwaterheating = st.sidebar.radio("Hot Water Heating?", ['yes', 'no'])
airconditioning = st.sidebar.radio("Air Conditioning?", ['yes', 'no'])
prefarea = st.sidebar.radio("Preferred Area?", ['yes', 'no'])
furnishingstatus = st.sidebar.radio("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# ----------------------------
# Prepare Input Data
# ----------------------------
input_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad_yes': int(mainroad == 'yes'),
    'guestroom_yes': int(guestroom == 'yes'),
    'basement_yes': int(basement == 'yes'),
    'hotwaterheating_yes': int(hotwaterheating == 'yes'),
    'airconditioning_yes': int(airconditioning == 'yes'),
    'prefarea_yes': int(prefarea == 'yes'),
    'furnishingstatus_semi-furnished': int(furnishingstatus == 'semi-furnished'),
    'furnishingstatus_unfurnished': int(furnishingstatus == 'unfurnished')
}

try:
    input_df = pd.DataFrame([input_data], columns=feature_names)
except Exception as e:
    st.error("⚠️ Feature mismatch! Your model expects different features.")
    st.write("Expected features:", feature_names)
    st.stop()

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🏠 Estimated House Price: ₹ {round(prediction):,}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("🔧 Made with ❤️ using Streamlit | Gayatri")
