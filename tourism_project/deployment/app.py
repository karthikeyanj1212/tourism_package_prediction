import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ---------------------------------------------------------
# Load trained model (pipeline with preprocessing inside)
# ---------------------------------------------------------
# TODO: change these to your actual repo + file on Hugging Face
MODEL_REPO_ID = "KarthiKeyanJ1212/tourism_prediction_model"
MODEL_FILENAME = "best_tourism_prediction_model_v1.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = joblib.load(model_path)

# ---------------------------------------------------------
# Streamlit App UI
# ---------------------------------------------------------
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This app predicts whether a customer is likely to purchase the **Wellness Tourism Package**
based on their profile and interaction details.
Please fill in the customer information below.
""")

st.sidebar.header("Customer Details")

# -------------- Customer demographic inputs --------------
age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=30, step=1)

typeof_contact = st.sidebar.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

city_tier = st.sidebar.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.sidebar.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

designation = st.sidebar.selectbox(
    "Designation",
    ["Executive", "Senior Manager", "Manager", "AVP", "VP"]
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=200000,
    value=25000,
    step=500
)

# -------------- Travel & family details --------------
st.sidebar.header("Travel & Family Details")

num_person_visiting = st.sidebar.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

num_children_visiting = st.sidebar.number_input(
    "Number of Children Visiting (Age < 5)",
    min_value=0,
    max_value=5,
    value=0,
    step=1
)

preferred_property_star = st.sidebar.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

num_trips = st.sidebar.number_input(
    "Average Number of Trips per Year",
    min_value=0,
    max_value=20,
    value=2,
    step=1
)

passport = st.sidebar.selectbox(
    "Has Passport?",
    ["No", "Yes"]
)

own_car = st.sidebar.selectbox(
    "Owns a Car?",
    ["No", "Yes"]
)

# -------------- Interaction details --------------
st.sidebar.header("Sales Interaction Details")

product_pitched = st.sidebar.selectbox(
    "Product Pitched",
    # Common product categories in this dataset
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

pitch_satisfaction = st.sidebar.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

num_followups = st.sidebar.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

duration_of_pitch = st.sidebar.number_input(
    "Duration of Pitch (minutes)",
    min_value=1,
    max_value=200,
    value=30,
    step=1
)

# ---------------------------------------------------------
# Prepare input DataFrame
# Column names MUST match what was used in training
# (excluding 'Unnamed: 0', 'CustomerID', 'ProdTaken')
# ---------------------------------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

st.subheader("Customer Input Summary")
st.write(input_data)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("Predict Package Purchase"):
    # model is assumed to be a pipeline that handles preprocessing
    pred = model.predict(input_data)[0]

    # If model supports predict_proba, show probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]  # probability of class 1
    else:
        proba = None

    result = "Will Purchase the Package" if pred == 1 else "Will NOT Purchase the Package"

    st.subheader("Prediction Result")
    st.success(f"The model predicts: **{result}**")

    if proba is not None:
        st.write(f"Estimated probability of purchase: **{proba:.2%}**")
