import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/rakesh1715/Tourism-Package-Prediction/X_train.csv"
Xtest_path = "hf://datasets/rakesh1715/Tourism-Package-Prediction/X_test.csv"
ytrain_path = "hf://datasets/rakesh1715/Tourism-Package-Prediction/y_train.csv"
ytest_path = "hf://datasets/rakesh1715/Tourism-Package-Prediction/y_test.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# One-hot encode and scale numeric features

ordinal_cols = [
        "CityTier",
        "PreferredPropertyStar",
        # "PitchSatisfactionScore",
        # "ProductPitched",
        "Designation"
]

nominal_cols = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "MaritalStatus"
]

numeric_cols = [
        "Age",
        # "DurationOfPitch",
        "NumberOfPersonVisiting",
        # "NumberOfFollowups",
        "NumberOfTrips",
        "NumberOfChildrenVisiting",
        "MonthlyIncome"
]

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
        (
                OrdinalEncoder(
                        categories=[
                                ["1", "2", "3"],  # CityTier
                                [1, 2, 3, 4, 5],  # PreferredPropertyStar
                                # [1, 2, 3, 4, 5],    # PitchSatisfactionScore,
                                # ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],    # ProductPitched
                                ["Executive", "Manager", "Senior Manager", "AVP", "VP"]  # Designation
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                ),
                ordinal_cols
        ),
        (
                OneHotEncoder(handle_unknown="ignore"),
                nominal_cols
        ),
        (
                StandardScaler(),
                numeric_cols
        )
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
        'xgbclassifier__n_estimators': [50, 75, 100],
        'xgbclassifier__max_depth': [2, 3, 4],
        'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
        'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
        'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
            "train_accuracy": train_report['accuracy'],
            "train_precision": train_report['1']['precision'],
            "train_recall": train_report['1']['recall'],
            "train_f1-score": train_report['1']['f1-score'],
            "test_accuracy": test_report['accuracy'],
            "test_precision": test_report['1']['precision'],
            "test_recall": test_report['1']['recall'],
            "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "rakesh1715/Tourism-Package-Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
            path_or_fileobj="best_tourism_package_prediction_model_v1.joblib",
            path_in_repo="best_tourism_package_prediction_model_v1.joblib",
            repo_id=repo_id,
            repo_type=repo_type,
    )
