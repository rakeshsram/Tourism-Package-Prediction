# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/rakesh1715/Tourism-Package-Prediction/tourism.csv"
data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop 'Unnamed: 0' and 'CustomerID' columns
data.drop(['Unnamed: 0', 'CustomerID'], axis=1, inplace=True)

# Remove duplicate from the data
data.drop_duplicates(inplace=True)

# Dara transformation
data["Gender"] = data["Gender"].str.replace("Fe Male","Female")
print(data["Gender"].value_counts())

data["MaritalStatus"] = data["MaritalStatus"].str.replace("Unmarried","Single")
print(data["MaritalStatus"].value_counts())

data["ProdTaken"] = data["ProdTaken"].astype("category")
data["TypeofContact"] = data["TypeofContact"].astype("category")
data["Occupation"] = data["Occupation"].astype("category")
data["Gender"] = data["Gender"].astype("category")
data["ProductPitched"] = data["ProductPitched"].astype("category")
data["MaritalStatus"] = data["MaritalStatus"].astype("category")
data["Passport"] = data["Passport"].astype("category")
data["OwnCar"] = data["OwnCar"].astype("category")
data["Designation"] = data["Designation"].astype("category")
data["CityTier"] = data["CityTier"].astype("category")
data["NumberOfPersonVisiting"] = data["NumberOfPersonVisiting"].astype("category")
data["NumberOfFollowups"] = data["NumberOfFollowups"].astype("category")
data["PreferredPropertyStar"] = data["PreferredPropertyStar"].astype("category")
data["PitchSatisfactionScore"] = data["PitchSatisfactionScore"].astype("category")
data["NumberOfChildrenVisiting"] = data["NumberOfChildrenVisiting"].astype("category")
data["NumberOfTrips"] = data["NumberOfTrips"].astype("int64")
data["DurationOfPitch"] = data["DurationOfPitch"].astype("int64")

# Modify the ordinal values for columns: ProductPitched, Designation to Label Encoding
product_order = {
    'Basic': 1,
    'Standard': 2,
    'Deluxe': 3,
    'Super Deluxe': 4,
    'King': 5
}

# Replace the original column with encoded values
data['ProductPitched'] = data['ProductPitched'].map(product_order)

designation_order = {
    'Executive': 1,
    'Manager': 2,
    'Senior Manager': 3,
    'AVP': 4,
    'VP': 5
}
# Replace the original column with encoded values
data['Designation'] = data['Designation'].map(designation_order)

# Modify the npminal values for columns: Occupation, Gender, MaritalStatus, TypeofContact to One-Hot Encoding
nominal_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus']
data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)
print("One-Hot Encoding complete. Original nominal columns have been replaced by new binary columns.")

# Split the data into train and test sets
Y = data["ProdTaken"]
X = data.drop("ProdTaken" , axis=1)

# creating dummy variables
X = pd.get_dummies(X, drop_first=True)

# splitting in training and test set
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 42, stratify=Y)

files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="rakesh1715/Tourism_Package_Prediction",
        repo_type="dataset",
    )
