#!/usr/bin/env python3
"""
Script to push Solar Regatta repository to Hugging Face Spaces.

Usage:
    python push_to_huggingface.py

You will be prompted to:
1. Log in to Hugging Face (if not already logged in)
2. Provide your Hugging Face username
3. Provide a repository name (default: solar-regatta)
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import subprocess

def main():
    print("=" * 70)
    print("Solar Regatta - Hugging Face Spaces Uploader")
    print("=" * 70)
    print()

    # Get Hugging Face credentials
    print("Step 1: Authentication")
    print("-" * 70)

    # Check if already logged in
    try:
        api = HfApi()
        whoami = api.whoami()
        username = whoami['name']
        print(f"✓ Already logged in as: {username}")
    except Exception as e:
        print("You need to log in to Hugging Face.")
        print("Please run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        return

    print()

    # Get repository details
    print("Step 2: Repository Configuration")
    print("-" * 70)

    repo_name = input(f"Repository name (default: solar-regatta): ").strip()
    if not repo_name:
        repo_name = "solar-regatta"

    repo_id = f"{username}/{repo_name}"
    print(f"Repository ID: {repo_id}")
    print()

    # Choose repo type
    print("Choose repository type:")
    print("1. Space (recommended - for interactive demos)")
    print("2. Model (for ML models)")
    print("3. Dataset (for datasets)")
    choice = input("Enter choice (1-3, default: 1): ").strip()

    if choice == "2":
        repo_type = "model"
    elif choice == "3":
        repo_type = "dataset"
    else:
        repo_type = "space"

    print(f"Using repo type: {repo_type}")
    print()

    # Create repository
    print("Step 3: Creating Repository")
    print("-" * 70)

    try:
        if repo_type == "space":
            url = create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="static",
                exist_ok=True,
                private=False
            )
        else:
            url = create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                exist_ok=True,
                private=False
            )
        print(f"✓ Repository created/verified: {url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    print()

    # Prepare files
    print("Step 4: Preparing Files")
    print("-" * 70)

    # Get current directory
    current_dir = Path.cwd()

    # Copy HF_README.md to README.md for Hugging Face
    if (current_dir / "HF_README.md").exists():
        # Backup original README
        if (current_dir / "README.md").exists():
            shutil.copy(current_dir / "README.md", current_dir / "README_ORIGINAL.md")
            print("✓ Backed up original README.md to README_ORIGINAL.md")

        # Copy HF README
        shutil.copy(current_dir / "HF_README.md", current_dir / "README.md")
        print("✓ Using HF_README.md as README.md for Hugging Face")

    print()

    # Upload using git
    print("Step 5: Uploading to Hugging Face")
    print("-" * 70)

    try:
        # Add HF remote if not exists
        result = subprocess.run(
            ["git", "remote", "get-url", "huggingface"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Remote doesn't exist, add it
            hf_url = f"https://huggingface.co/spaces/{repo_id}"
            if repo_type == "model":
                hf_url = f"https://huggingface.co/{repo_id}"
            elif repo_type == "dataset":
                hf_url = f"https://huggingface.co/datasets/{repo_id}"

            subprocess.run(
                ["git", "remote", "add", "huggingface", hf_url],
                check=True
            )
            print(f"✓ Added Hugging Face remote: {hf_url}")
        else:
            print("✓ Hugging Face remote already exists")

        # Commit README changes
        subprocess.run(["git", "add", "README.md"], check=True)
        try:
            subprocess.run(
                ["git", "commit", "-m", "Update README for Hugging Face"],
                capture_output=True
            )
        except:
            pass  # No changes to commit

        # Push to Hugging Face
        print("Pushing to Hugging Face...")
        result = subprocess.run(
            ["git", "push", "huggingface", "main"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ Successfully pushed to Hugging Face!")
        else:
            print(f"Error pushing: {result.stderr}")
            # Try force push
            print("Attempting force push...")
            result = subprocess.run(
                ["git", "push", "-f", "huggingface", "main"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✓ Successfully force pushed to Hugging Face!")
            else:
                print(f"Error: {result.stderr}")
                return

        # Restore original README
        if (current_dir / "README_ORIGINAL.md").exists():
            shutil.copy(current_dir / "README_ORIGINAL.md", current_dir / "README.md")
            (current_dir / "README_ORIGINAL.md").unlink()
            print("✓ Restored original README.md")

    except Exception as e:
        print(f"Error uploading: {e}")
        # Restore original README on error
        if (current_dir / "README_ORIGINAL.md").exists():
            shutil.copy(current_dir / "README_ORIGINAL.md", current_dir / "README.md")
            (current_dir / "README_ORIGINAL.md").unlink()
        return

    print()
    print("=" * 70)
    print("✓ Successfully uploaded to Hugging Face!")
    print("=" * 70)

    if repo_type == "space":
        print(f"View your Space at: https://huggingface.co/spaces/{repo_id}")
    elif repo_type == "model":
        print(f"View your Model at: https://huggingface.co/{repo_id}")
    else:
        print(f"View your Dataset at: https://huggingface.co/datasets/{repo_id}")

    print()
    print("Next steps:")
    print("1. Visit the URL above to see your repository")
    print("2. Update the README.md metadata (emoji, tags, etc.)")
    print("3. Share with the community!")

if __name__ == "__main__":
    main()
