"""One-command script to upload the friday-desktop-intent dataset to Hugging Face."""

import sys
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else input("Enter your Hugging Face Repo ID (e.g. username/friday-desktop-intent): ").strip()
    if not repo_id:
        print("[-] Repo ID required.")
        return

    print(f"[*] Target Hugging Face Repository: {repo_id}")
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"[+] Verified/created Hugging Face dataset repo: {repo_id}")
    except Exception as e:
        print(f"[-] Make sure you are logged in (`huggingface-cli login`). Error: {e}")
        return

    folder_to_upload = str(Path(__file__).parent)
    print(f"[*] Uploading '{folder_to_upload}' to Hugging Face...")

    api.upload_folder(
        folder_path=folder_to_upload,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload FRIDAY DesktopIntent multi-subset dataset & 293-test benchmark",
        ignore_patterns=["upload_to_hf.py"],
    )

    print("\n[+] Successfully uploaded to Hugging Face!")
    print(f"[+] View your dataset at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
