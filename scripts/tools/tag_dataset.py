#!/usr/bin/env python3
"""Tag a HuggingFace dataset with its codebase version (required by LeRobot)."""

import argparse
import json
from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="Tag dataset with codebase version")
    parser.add_argument("repo_id", help="HuggingFace dataset repo ID (e.g. danbhf/sim_pick_place_157ep)")
    args = parser.parse_args()

    # Download and read info.json
    print(f"Fetching info.json from {args.repo_id}...")
    info_path = hf_hub_download(args.repo_id, "meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        info = json.load(f)

    version = info.get("codebase_version", "v2.0")
    print(f"Version: {version}")

    # Create tag
    hub_api = HfApi()
    try:
        hub_api.create_tag(args.repo_id, tag=version, repo_type="dataset")
        print(f"Tagged {args.repo_id} with {version}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Tag {version} already exists")
        else:
            raise


if __name__ == "__main__":
    main()
