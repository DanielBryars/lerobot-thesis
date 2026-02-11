#!/usr/bin/env python3
"""
Extract the first frame (overhead camera) from each episode of a LeRobot dataset.

Saves images as PNG files, one per episode, for use with the high-level planner.

Usage:
    # Extract from the 220ep dataset (default)
    python scripts/high-level-planner/extract_first_frames.py

    # Extract specific episodes
    python scripts/high-level-planner/extract_first_frames.py --episodes 0 5 10 50

    # Custom dataset and output directory
    python scripts/high-level-planner/extract_first_frames.py \
        --dataset danbhf/sim_pick_place_157ep \
        --output frames/

    # Use wrist cam instead
    python scripts/high-level-planner/extract_first_frames.py --camera wrist_cam
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract_first_frames(
    dataset_id: str,
    output_dir: Path,
    camera: str = "overhead_cam",
    episodes: list[int] | None = None,
):
    """Extract the first frame from each episode.

    Args:
        dataset_id: HuggingFace dataset ID
        output_dir: Directory to save PNGs
        camera: Camera name to extract
        episodes: Specific episode indices, or None for all
    """
    print(f"Loading dataset: {dataset_id}")
    dataset = LeRobotDataset(dataset_id)

    # Get episode boundaries from the hf_dataset
    all_episode_indices = dataset.hf_dataset["episode_index"]
    num_episodes = dataset.meta.total_episodes

    if episodes is None:
        episodes = list(range(num_episodes))

    print(f"Dataset has {num_episodes} episodes, extracting {len(episodes)} frames")
    print(f"Camera: {camera}")

    # Find the first frame index for each requested episode
    first_frame_map = {}
    for i, ep_idx in enumerate(all_episode_indices):
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        if ep_idx in episodes and ep_idx not in first_frame_map:
            first_frame_map[ep_idx] = i

    missing = set(episodes) - set(first_frame_map.keys())
    if missing:
        print(f"WARNING: episodes not found in dataset: {sorted(missing)}")

    # Load episode_scenes.json for metadata if available
    episode_scenes = {}
    try:
        from huggingface_hub import hf_hub_download
        scenes_path = hf_hub_download(
            dataset_id, "meta/episode_scenes.json", repo_type="dataset"
        )
        with open(scenes_path) as f:
            episode_scenes = json.load(f)
        print(f"Loaded scene metadata for {len(episode_scenes)} episodes")
    except Exception:
        print("No episode_scenes.json found (scene metadata won't be embedded)")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_key = f"observation.images.{camera}"

    manifest = []

    for ep_idx in sorted(first_frame_map.keys()):
        frame_idx = first_frame_map[ep_idx]
        sample = dataset[frame_idx]

        if image_key not in sample:
            available = [k for k in sample.keys() if "images" in k]
            print(f"ERROR: '{image_key}' not found. Available: {available}")
            return

        img_tensor = sample[image_key]  # shape: [C, H, W], float [0, 1]
        # Convert to PIL
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(img_np)

        filename = f"episode_{ep_idx:04d}.png"
        img.save(output_dir / filename)

        # Build manifest entry
        entry = {"episode": ep_idx, "file": filename}
        ep_key = str(ep_idx)
        if ep_key in episode_scenes:
            scene = episode_scenes[ep_key]
            try:
                duplo_pos = scene["objects"]["duplo"]["position"]
                entry["duplo_position"] = {
                    "x": duplo_pos["x"],
                    "y": duplo_pos["y"],
                }
            except (KeyError, TypeError):
                pass
            try:
                bowl_pos = scene["objects"]["bowl"]["position"]
                entry["bowl_position"] = {
                    "x": bowl_pos["x"],
                    "y": bowl_pos["y"],
                }
            except (KeyError, TypeError):
                pass
        manifest.append(entry)

        print(f"  Episode {ep_idx:3d} -> {filename} ({img.width}x{img.height})")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_id,
                "camera": camera,
                "num_frames": len(manifest),
                "frames": manifest,
            },
            f,
            indent=2,
        )
    print(f"\nSaved {len(manifest)} frames to {output_dir}/")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract first frames from LeRobot dataset episodes")
    parser.add_argument(
        "--dataset",
        default="danbhf/sim_pick_place_2pos_220ep_v2",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--output",
        default="scripts/high-level-planner/frames",
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "--camera",
        default="overhead_cam",
        help="Camera name to extract (default: overhead_cam)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific episode indices to extract (default: all)",
    )
    args = parser.parse_args()

    extract_first_frames(
        dataset_id=args.dataset,
        output_dir=Path(args.output),
        camera=args.camera,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
