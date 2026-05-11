"""Upload dataset.json to Hugging Face dataset repo."""

from pathlib import Path
from huggingface_hub import login, HfApi

REPO_ID = "HikkenNoAce/Intent_Decomposition_to_Sub_Intents_for_O_RAN_networks"
DATASET_FILE = "dataset.json"
REPO_TYPE = "dataset"

if __name__ == "__main__":
    print("Logging in to Hugging Face...")
    login()

    dataset_path = Path(DATASET_FILE)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

    print(f"Uploading {DATASET_FILE} to {REPO_ID}...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo="dataset.json",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    print(f"✓ Successfully uploaded dataset.json to {REPO_ID}")
    print(f"Access it at: https://huggingface.co/datasets/{REPO_ID}")
