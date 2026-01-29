#!/usr/bin/env python3
"""
Generate and upload episode_scenes.json for datasets that are missing it.

For re-recorded datasets with confuser blocks, this:
1. Downloads the source dataset's episode_scenes.json (for duplo positions)
2. Adds confuser position info (default or marks as randomized)
3. Uploads to the target dataset
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

# Default confuser position from so101_with_confuser.xml
DEFAULT_CONFUSER = {
    "position": {"x": 0.25, "y": 0.05, "z": 0.0096},
    "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
}


def main():
    parser = argparse.ArgumentParser(description="Upload episode_scenes.json to HuggingFace datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Target HuggingFace dataset ID")
    parser.add_argument("--source", type=str, help="Source dataset to get duplo positions from (auto-detected from rerecord_metadata)")
    parser.add_argument("--confuser-randomized", action="store_true", help="Mark confuser as randomized (positions unknown)")
    parser.add_argument("--copies", type=int, default=1, help="Number of copies per source episode")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload, just show what would be done")
    args = parser.parse_args()

    api = HfApi()

    # Get rerecord metadata to find source dataset
    source_dataset = args.source
    if not source_dataset:
        try:
            meta_path = hf_hub_download(args.dataset, 'meta/rerecord_metadata.json', repo_type='dataset')
            with open(meta_path) as f:
                meta = json.load(f)
            source_dataset = meta.get("rerecord_info", {}).get("source_dataset")
            print(f"Auto-detected source dataset: {source_dataset}")
        except Exception as e:
            print(f"ERROR: Could not get source dataset. Specify with --source. Error: {e}")
            return

    # Load source episode_scenes
    print(f"\nLoading episode_scenes from source: {source_dataset}")
    try:
        scenes_path = hf_hub_download(source_dataset, 'meta/episode_scenes.json', repo_type='dataset')
        with open(scenes_path) as f:
            source_scenes = json.load(f)
        print(f"  Loaded {len(source_scenes)} episodes from source")
    except Exception as e:
        print(f"ERROR: Could not load episode_scenes.json from source: {e}")
        return

    # Generate new episode_scenes
    new_scenes = {}
    source_ep_count = len(source_scenes)

    for copy_idx in range(args.copies):
        for src_ep_idx in range(source_ep_count):
            output_ep_idx = copy_idx * source_ep_count + src_ep_idx

            # Get source scene info
            src_scene = source_scenes.get(str(src_ep_idx), {})

            # Create new scene info
            new_scene = {
                "scene_xml": src_scene.get("scene_xml", "scenes/so101_with_confuser.xml"),
                "objects": {}
            }

            # Copy duplo info from source
            if "objects" in src_scene and "duplo" in src_scene["objects"]:
                new_scene["objects"]["duplo"] = src_scene["objects"]["duplo"]

            # Add confuser info
            if args.confuser_randomized:
                new_scene["objects"]["confuser"] = {
                    "position": {"x": "randomized", "y": "randomized", "z": 0.0096},
                    "quaternion": {"w": "randomized", "x": 0.0, "y": 0.0, "z": "randomized"},
                    "note": "Position was randomized during re-recording"
                }
            else:
                # Check if source already has confuser info
                if "objects" in src_scene and "confuser" in src_scene["objects"]:
                    new_scene["objects"]["confuser"] = src_scene["objects"]["confuser"]
                else:
                    new_scene["objects"]["confuser"] = DEFAULT_CONFUSER

            # Copy bowl info if present
            if "objects" in src_scene and "bowl" in src_scene["objects"]:
                new_scene["objects"]["bowl"] = src_scene["objects"]["bowl"]

            new_scenes[str(output_ep_idx)] = new_scene

    print(f"\nGenerated {len(new_scenes)} episode scenes")

    if args.dry_run:
        print("\n[DRY RUN] Would upload episode_scenes.json with content:")
        print(json.dumps(dict(list(new_scenes.items())[:3]), indent=2))
        print(f"... and {len(new_scenes) - 3} more episodes")
        return

    # Save locally first
    local_path = Path(f"episode_scenes_{args.dataset.replace('/', '_')}.json")
    with open(local_path, 'w') as f:
        json.dump(new_scenes, f, indent=2)
    print(f"Saved to {local_path}")

    # Upload to HuggingFace
    print(f"\nUploading to {args.dataset}...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo="meta/episode_scenes.json",
        repo_id=args.dataset,
        repo_type="dataset",
    )
    print("Upload complete!")

    # Cleanup local file
    local_path.unlink()


if __name__ == "__main__":
    main()
