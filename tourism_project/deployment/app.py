import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from the Hugging Face model hub
model_path = hf_hub_download(repo_id="sheetalkarande/tourism-package-model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the newly introduced
Wellness Tourism Package based on their profile and interaction details.
Please enter the customer information below to get a prediction.
""")

# User input - Customer Details
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=10, value=3, step=1)
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
num_trips = st.number_input("Number Of Trips (per year)", min_value=0, max_value=50, value=2, step=1)
passport = st.selectbox("Holds a Passport?", [0, 1])
own_car = st.selectbox("Owns a Car?", [0, 1])
num_children_visiting = st.number_input("Number Of Children Visiting (below age 5)", min_value=0, max_value=10, value=1, step=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=20000.0, step=100.0)

# User input - Customer Interaction Data
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
num_followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=3, step=1)
duration_of_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)

# Assemble input into DataFrame (column names must match training features)
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase the Package" if prediction == 1 else "Will Not Purchase the Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
