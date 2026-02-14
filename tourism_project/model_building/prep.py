# for creating a folder
import os
# for data manipulation
import pandas as pd
# for converting text data in to numerical representation
# for hugging face space authentication to upload files
from huggingface_hub import HfApi
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/rakesh1715/Tourism-Package-Prediction/tourism.csv"
data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

"""
Observations from data interpretation:
- There are no null values
- There are missing values
- There are no duplicates
- Since the objective is to build models on data of the existing customers which can be used to target new customers, we can drop the customer interaction data from the dataset as those features will not be available for new customers.
- column - Unnamed can be removed
- Following fields as per data dictionary must be converted to categorical:
  - ProdTaken
  - CityTier
  - Passport
  - OwnCar
- Gender 'Female' is recorded as 'Fe male' - needs conversion
- Marital status column can be updated by treating UnMarried as Single
- Model Evaluation Metric:
  - Recall score should be maximized. Greater the Recall score, higher the chances of predicting the potential customers who may purchase the new travel package.
"""

# Drop redundant fields
data.drop(
        [
                "Unnamed: 0",
                "CustomerID",
                # "DurationOfPitch",
                # "NumberOfFollowups",
                # "ProductPitched",
                # "PitchSatisfactionScore",
        ],
        axis=1,
        inplace=True,
)

# Remove duplicate from the data
data.drop_duplicates(inplace=True)

# Data transformation for gender as "Female" is interpreted as "FE male" in some records
data["Gender"] = data["Gender"].str.replace("Fe Male", "Female")

# Merge MaritalStatus of UnMarried and Single
data["MaritalStatus"] = data["MaritalStatus"].str.replace("Unmarried", "Single")

# Convert to categorical types
data["ProdTaken"] = data["ProdTaken"].astype("category")
data["CityTier"] = data["CityTier"].astype("category")
data["Passport"] = data["Passport"].astype("category")
data["OwnCar"] = data["OwnCar"].astype("category")

data["TypeofContact"] = data["TypeofContact"].astype("category")
data["Occupation"] = data["Occupation"].astype("category")
data["Gender"] = data["Gender"].astype("category")
data["MaritalStatus"] = data["MaritalStatus"].astype("category")
data["Designation"] = data["Designation"].astype("category")

data["NumberOfPersonVisiting"] = data["NumberOfPersonVisiting"].astype("int64")
data["PreferredPropertyStar"] = data["PreferredPropertyStar"].astype("float")
data["NumberOfChildrenVisiting"] = data["NumberOfChildrenVisiting"].astype("int64")
data["NumberOfTrips"] = data["NumberOfTrips"].astype("int64")

# Split the data into train and test sets
Y = data["ProdTaken"]
X = data.drop("ProdTaken", axis=1)

# splitting in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

for file_path in files:
    api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],  # just the filename
            repo_id="rakesh1715/Tourism-Package-Prediction",
            repo_type="dataset",
    )
