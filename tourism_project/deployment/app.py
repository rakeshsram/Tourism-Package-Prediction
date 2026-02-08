import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="rakesh1715/Tourism-Package-Prediction", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")

st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
age = st.number_input("Age (in years)", min_value=18, max_value=100, value=25)
typeofContract = st.selectbox("Type of Contract", ["Self Enquiry", "Company Invited"])
cityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
durationOfPitch = st.number_input("Duration Of Pitch", min_value=5, max_value=150, value=25)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", 'Free Lancer', 'Large Business'])
gender = st.selectbox("Gender", ["Male", "Female"])
numberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=6, value=1)
numberOfFollowups = st.number_input("Number Of Followups", min_value=1, max_value=10, value=1)
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", 'Deluxe', 'King', 'Super Deluxe'])
preferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=1)
maritalStatus = st.selectbox("Marital Status", ["Married", "Single", 'Divorced','Unmarried'])
numberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=50, value=1)
passport = st.selectbox("Passport", ["0", "1"])
pitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=1)
ownCar = st.selectbox("Own Car", ["0", "1"])
numberOfChildrenVisiting = st.number_input("Number Of Children Visiting",
                                            min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Senior Manager", 'Manager', 'AVP', 'VP'])
monthlyIncome = st.number_input("Monthly Income",
                                min_value=0, max_value=100000, value=0)

# Assemble input into DataFrame

input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofContact,
    'CityTier': cityTier,
    'DurationOfPitch': durationOfPitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberOfPersonVisiting,
    'NumberOfFollowups': numberOfFollowups,
    'ProductPitched': productPitched,
    'PreferredPropertyStar': preferredPropertyStar,
    'MaritalStatus': maritalStatus,
    'NumberOfTrips': numberOfTrips,
    'Passport': passport,
    'PitchSatisfactionScore': pitchSatisfactionScore,
    'OwnCar': ownCar,
    'NumberOfChildrenVisiting': numberOfChildrenVisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyIncome

}])


if st.button("Predict Tourism Package Taken"):
    prediction = model.predict(input_data)[0]
    result = "Package Taken" if prediction == 1 else "Package Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
