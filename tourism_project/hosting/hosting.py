import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
        folder_path="tourism_project/deployment",  # the local folder containing your files
        repo_id="rakesh1715/Tourism-Package-Prediction",  # the target repo
        repo_type="space",  # dataset, model, or space
        path_in_repo="",  # optional: subfolder path inside the repo
)
