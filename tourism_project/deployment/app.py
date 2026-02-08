import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# Download and load the model
model_path = hf_hub_download(repo_id="rakesh1715/Tourism-Package-Prediction",
                             filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")

st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter the required fields to get a prediction.
""")

# User input
age = st.number_input("Age (in years)", min_value=18, max_value=100, value=25)
typeofContact = st.selectbox("Type of Contract", ["Self Enquiry", "Company Invited"])
cityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", 'Free Lancer', 'Large Business'])
gender = st.selectbox("Gender", ["Male", "Female"])
numberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, value=1)
preferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=1)
maritalStatus = st.selectbox("Marital Status", ["Married", "Single", 'Divorced'])
numberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=50, value=1)
passport = st.selectbox("Passport", ["Yes", "No"])
ownCar = st.selectbox("Own Car", ["Yes", "No"])
numberOfChildrenVisiting = st.number_input("Number Of Children Visiting",
                                           min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Executive", "Senior Manager", 'Manager', 'AVP', 'VP'])
monthlyIncome = st.number_input("Monthly Income",
                                min_value=0, value=0)
# pitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=1)
# durationOfPitch = st.number_input("Duration Of Pitch", min_value=5, value=25)
# numberOfFollowups = st.number_input("Number Of Followups", min_value=1, value=1)
# ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", 'Deluxe', 'Super Deluxe', 'King'])

# Assemble input into DataFrame

input_data = pd.DataFrame([{
        'Age': age,
        'CityTier': str(cityTier),
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': numberOfPersonVisiting,
        'PreferredPropertyStar': preferredPropertyStar,
        'MaritalStatus': maritalStatus,
        'NumberOfTrips': numberOfTrips,
        'Passport': "1" if passport == "Yes" else "0",
        'TypeofContact': typeofContact,
        'OwnCar': "1" if ownCar == "Yes" else "0",
        'NumberOfChildrenVisiting': numberOfChildrenVisiting,
        'Designation': designation,
        'MonthlyIncome': monthlyIncome
        # 'DurationOfPitch': durationOfPitch,
        # 'NumberOfFollowups': numberOfFollowups,
        # 'ProductPitched': productPitched,
        # 'PitchSatisfactionScore': pitchSatisfactionScore,

}])

if st.button("Predict Tourism Package Taken"):
    prediction = model.predict(input_data)[0]
    result = "Package Taken" if prediction == 1 else "Package Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
