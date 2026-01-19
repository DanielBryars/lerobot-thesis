#!/usr/bin/env python3
"""Fix missing image stats in LeRobot dataset on HuggingFace."""

import argparse
import json
import tempfile
import os
from huggingface_hub import HfApi, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="Add missing image stats to dataset")
    parser.add_argument("repo_id", help="HuggingFace dataset repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without uploading")
    args = parser.parse_args()

    # Download current stats
    print(f"Downloading stats.json from {args.repo_id}...")
    stats_path = hf_hub_download(args.repo_id, "meta/stats.json", repo_type="dataset", force_download=True)

    with open(stats_path) as f:
        stats = json.load(f)

    print(f"Current keys: {list(stats.keys())}")

    # Download info.json to get image feature names
    info_path = hf_hub_download(args.repo_id, "meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        info = json.load(f)

    # Find image features
    features = info.get("features", {})
    image_keys = [k for k in features.keys() if "image" in k.lower()]
    print(f"Image features found: {image_keys}")

    # Add placeholder stats for missing image keys
    # These get overwritten by ImageNet stats when use_imagenet_stats=True
    placeholder = {
        "mean": [[[0.0]], [[0.0]], [[0.0]]],
        "std": [[[1.0]], [[1.0]], [[1.0]]],
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[255.0]], [[255.0]], [[255.0]]]
    }

    added = []
    for key in image_keys:
        if key not in stats:
            stats[key] = placeholder
            added.append(key)
            print(f"  Added: {key}")

    if not added:
        print("No missing image stats - nothing to fix!")
        return

    print(f"\nAdded {len(added)} image stats entries")

    if args.dry_run:
        print("\nDry run - not uploading. New stats would be:")
        print(json.dumps(stats, indent=2))
        return

    # Upload fixed stats
    with tempfile.TemporaryDirectory() as tmpdir:
        fixed_path = os.path.join(tmpdir, "stats.json")
        with open(fixed_path, "w") as f:
            json.dump(stats, f, indent=2)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=fixed_path,
            path_in_repo="meta/stats.json",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add missing image stats for LeRobot compatibility"
        )
        print(f"\nUploaded fixed stats.json to {args.repo_id}")


if __name__ == "__main__":
    main()
