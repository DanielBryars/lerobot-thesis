#!/usr/bin/env python3
"""Standalone script to upload a dataset to HuggingFace Hub."""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, list_repo_files


def upload_dataset(local_path: Path, repo_id: str, dry_run: bool = False):
    """Upload a dataset folder to HuggingFace Hub."""

    print(f"Local path: {local_path}")
    print(f"Repo ID: {repo_id}")
    print()

    # List all local files
    print("=== Local files ===")
    local_files = sorted(local_path.rglob("*"))
    for f in local_files:
        if f.is_file():
            rel_path = f.relative_to(local_path)
            size = f.stat().st_size
            print(f"  {rel_path} ({size:,} bytes)")
    print()

    if dry_run:
        print("DRY RUN - not uploading")
        return

    # Create repo if needed
    api = HfApi()
    print("Creating repo (if needed)...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Upload entire folder
    print("\nUploading folder...")
    upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("\nUpload complete!")
    print()

    # List remote files to verify
    print("=== Remote files (after upload) ===")
    remote_files = sorted(list_repo_files(repo_id, repo_type="dataset"))
    for f in remote_files:
        print(f"  {f}")
    print()

    # Check for missing files
    local_rel_paths = set()
    for f in local_files:
        if f.is_file():
            rel_path = str(f.relative_to(local_path)).replace("\\", "/")
            local_rel_paths.add(rel_path)

    remote_set = set(remote_files)

    missing = local_rel_paths - remote_set
    if missing:
        print("=== MISSING FILES (local but not remote) ===")
        for f in sorted(missing):
            print(f"  {f}")
    else:
        print("All local files are present on remote!")

    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace")
    parser.add_argument("local_path", type=Path, help="Path to local dataset folder")
    parser.add_argument("repo_id", type=str, help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")

    args = parser.parse_args()

    if not args.local_path.exists():
        print(f"Error: {args.local_path} does not exist")
        return 1

    upload_dataset(args.local_path, args.repo_id, args.dry_run)
    return 0


if __name__ == "__main__":
    exit(main())
