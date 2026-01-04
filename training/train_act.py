#!/usr/bin/env python
"""
Train an ACT (Action Chunking Transformer) policy on a LeRobot dataset.

Based on LeRobot examples and lerobot-scratch training scripts.

Usage:
    python train_act.py danbhf/sim_pick_place_20251229_101340
    python train_act.py danbhf/sim_pick_place_20251229_101340 --steps 50000
    python train_act.py danbhf/sim_pick_place_20251229_101340 --batch_size 4 --steps 5000
"""

# Suppress noisy warnings before importing anything else
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")

import argparse
from pathlib import Path
from datetime import datetime
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

# Add project root to path for shared utilities
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Import shared utilities
from utils.constants import MOTOR_NAMES
from utils.ik_solver import IKSolver


class CachedDataset(torch.utils.data.Dataset):
    """Wrapper that pre-caches all dataset samples in memory for fast access.

    This eliminates GPU idle time caused by video decoding during training.
    All frames are decoded once at startup and kept in CPU memory.
    """

    def __init__(self, dataset, resize_images_to=None):
        self.resize_to = resize_images_to
        print(f"\nPre-caching {len(dataset)} samples to memory...")
        if resize_images_to:
            print(f"  Resizing images to {resize_images_to}x{resize_images_to}")

        # Pre-load all samples
        self.samples = []
        for i in tqdm(range(len(dataset)), desc="Caching dataset"):
            sample = dataset[i]
            cached_sample = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    # Resize images if requested
                    if resize_images_to and "images" in k and v.dim() >= 3:
                        if v.dim() == 4:  # [n_obs_steps, C, H, W]
                            v = F.interpolate(v, size=(resize_images_to, resize_images_to),
                                            mode='bilinear', align_corners=False)
                        elif v.dim() == 3:  # [C, H, W]
                            v = F.interpolate(v.unsqueeze(0), size=(resize_images_to, resize_images_to),
                                            mode='bilinear', align_corners=False).squeeze(0)
                    cached_sample[k] = v.clone()
                else:
                    cached_sample[k] = v
            self.samples.append(cached_sample)

        print(f"Cached {len(self.samples)} samples")

        # Estimate memory usage
        total_bytes = 0
        for sample in self.samples[:1]:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.element_size() * v.numel()
        total_gb = (total_bytes * len(self.samples)) / (1024**3)
        print(f"Estimated cache size: {total_gb:.2f} GB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def cycle(dataloader):
    """Infinite dataloader iterator."""
    while True:
        for batch in dataloader:
            yield batch


def prepare_obs_for_policy(obs: dict, device: torch.device) -> dict:
    """Convert simulation observation to policy input format."""
    import numpy as np
    batch = {}

    # Extract state (joint positions)
    state = []
    for motor in MOTOR_NAMES:
        key = f"{motor}.pos"
        state.append(obs.get(key, 0.0))
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    # Camera images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            # Check if this is a depth image (1 channel, float32)
            if value.shape[2] == 1 and value.dtype == np.float32:
                # Depth image: normalize by 1.0m for tabletop range
                img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float()
                img = img / 1.0  # Divide by 1m (meters)
            else:
                # RGB image: normalize to [0, 1]
                img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def run_evaluation(policy, preprocessor, postprocessor, device, num_episodes: int, randomize: bool = True, fps: int = 30):
    """Run evaluation episodes in simulation.

    Returns:
        success_rate: float between 0 and 1
        avg_steps: average steps per episode
        avg_time: average time per episode in seconds
        ik_failure_rate: IK failure rate (None if not using EE action space)
        avg_ik_error: Average IK error in mm (None if not using EE action space)
    """
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    # Check action dimension from policy config to determine if EE actions
    action_dim = policy.config.output_features["action"].shape[0]
    is_ee_action_space = (action_dim == 8)

    # Initialize IK solver if needed (uses shared utility)
    ik_solver = None
    if is_ee_action_space:
        ik_solver = IKSolver()
        print(f"  Using EE action space (8-dim) with IK conversion")

    # Check if policy uses depth by looking for depth cameras in input features
    depth_cameras = []
    sim_cameras = ["wrist_cam", "overhead_cam"]
    for key in policy.config.input_features:
        if key.startswith("observation.images.") and key.endswith("_depth"):
            # Extract camera name (e.g., "overhead_cam" from "observation.images.overhead_cam_depth")
            cam_name = key.replace("observation.images.", "").replace("_depth", "")
            if cam_name not in depth_cameras:
                depth_cameras.append(cam_name)

    if depth_cameras:
        print(f"  Using depth cameras: {depth_cameras}")

    # Initialize simulation (no VR for speed)
    config = SO100SimConfig(
        sim_cameras=sim_cameras,
        depth_cameras=depth_cameras,
        enable_vr=False,
        camera_width=640,
        camera_height=480,
    )
    sim_robot = SO100Sim(config)
    sim_robot.connect()

    successes = 0
    total_steps = 0
    total_time = 0.0
    max_steps = 300

    policy.eval()

    for ep in range(num_episodes):
        policy.reset()
        sim_robot.reset_scene(randomize=randomize, pos_range=0.04, rot_range=np.pi)

        # Reset IK stats for this evaluation run
        if ik_solver:
            ik_solver.reset_stats()

        ep_start = time.time()
        for step in range(max_steps):
            obs = sim_robot.get_observation()
            batch = prepare_obs_for_policy(obs, device)
            batch = preprocessor(batch)

            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action)

            # Convert action to dict - handle different tensor shapes
            action_np = action.cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np.flatten()

            # Convert EE actions to joint actions if needed
            if is_ee_action_space:
                action_np = action_np[:8]  # Take first 8 values (EE action)
                # Use IKSolver which returns normalized values for sim
                joint_action, _, _ = ik_solver.ee_to_joint_action(action_np, return_normalized=True)
            else:
                joint_action = action_np[:6]  # Take first 6 values (joint action)

            action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(6)}
            sim_robot.send_action(action_dict)

            if sim_robot.is_task_complete():
                successes += 1
                total_steps += step + 1
                total_time += time.time() - ep_start
                break
        else:
            total_steps += max_steps
            total_time += time.time() - ep_start

    sim_robot.disconnect()

    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    avg_time = total_time / num_episodes

    # Report IK stats if using EE action space
    if is_ee_action_space and ik_solver:
        stats = ik_solver.get_stats()
        if stats["total_count"] > 0:
            print(f"  IK stats: {stats['failure_count']}/{stats['total_count']} failures "
                  f"({100*stats['failure_rate']:.2f}%), avg error: {stats['avg_error_mm']:.2f}mm")
            return success_rate, avg_steps, avg_time, stats["failure_rate"], stats["avg_error_mm"]

    return success_rate, avg_steps, avg_time, None, None


def main():
    parser = argparse.ArgumentParser(description="Train ACT policy on LeRobot dataset")
    parser.add_argument("dataset", type=str, help="HuggingFace dataset repo ID")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--chunk_size", type=int, default=100, help="ACT chunk size (default: 100)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log_freq", type=int, default=100, help="Log frequency (default: 100)")
    parser.add_argument("--save_freq", type=int, default=5000, help="Checkpoint save frequency (default: 5000)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--wandb_project", type=str, default="lerobot-thesis", help="WandB project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--eval_episodes", type=int, default=0, help="Evaluation episodes per checkpoint (0=disabled)")
    parser.add_argument("--eval_randomize", action="store_true", help="Randomize object position during eval")
    parser.add_argument("--cache_dataset", action="store_true", help="Pre-cache dataset in memory (eliminates GPU idle time)")
    parser.add_argument("--cache_image_size", type=int, default=None, help="Resize images during caching (e.g., 224)")
    parser.add_argument("--use_joint_actions", action="store_true", help="Use action_joints instead of action (for EE datasets with preserved joint actions)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated camera names to use (e.g., 'wrist_cam,overhead_cam,overhead_cam_depth'). Default: all available")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/train/act_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset metadata to get feature shapes
    print(f"Loading dataset metadata: {args.dataset}")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)

    # Get features for policy
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    # Handle action field selection for EE datasets with preserved joint actions
    if args.use_joint_actions:
        if 'action_joints' not in output_features:
            print("ERROR: --use_joint_actions specified but dataset has no 'action_joints' field")
            sys.exit(1)
        # Use action_joints as the action, remove original action
        print("Using 'action_joints' (6-dim joint space) instead of 'action'")
        if 'action' in output_features:
            del output_features['action']
        # Rename action_joints to action for the policy
        output_features['action'] = output_features.pop('action_joints')
    else:
        # Default: use 'action', exclude 'action_joints'
        if 'action_joints' in output_features:
            del output_features['action_joints']

    # Exclude action_joints from input features (it's only for comparison during playback)
    input_features = {key: ft for key, ft in features.items()
                      if key not in output_features and key != 'action_joints' and key != 'action'}

    # Filter cameras if --cameras specified
    if args.cameras:
        selected_cams = [c.strip() for c in args.cameras.split(",")]
        print(f"Selected cameras: {selected_cams}")
        # Keep only observation.images.* keys that match selected cameras
        input_features = {
            key: ft for key, ft in input_features.items()
            if not key.startswith("observation.images.")
            or any(cam in key for cam in selected_cams)
        }

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Create ACT policy config
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
    )

    # Create policy
    print("Creating ACT policy...")
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    # Create pre/post processors for normalization
    # If using joint actions, rename stats key from action_joints to action
    stats = dataset_metadata.stats
    if args.use_joint_actions:
        import copy
        stats = copy.deepcopy(dict(stats))  # Deep copy to avoid modifying original
        if 'action_joints' in stats:
            stats['action'] = stats.pop('action_joints')
        elif 'action' in stats:
            # Remove EE action stats if present (wrong shape)
            del stats['action']
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Set up delta timestamps for ACT
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in range(args.chunk_size)],
    }
    # If using joint actions, also need to load action_joints with chunk timestamps
    if args.use_joint_actions:
        delta_timestamps["action_joints"] = [i / dataset_metadata.fps for i in range(args.chunk_size)]
    for key in input_features:
        delta_timestamps[key] = [0.0]

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw_dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(raw_dataset)} frames")

    # Optionally cache dataset in memory to eliminate video decoding bottleneck
    if args.cache_dataset:
        dataset = CachedDataset(raw_dataset, resize_images_to=args.cache_image_size)
        # With cached dataset, no workers needed and no pin_memory
        num_workers = 0
        pin_memory = False
    else:
        dataset = raw_dataset
        num_workers = args.num_workers
        pin_memory = device.type != "cpu"

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-7
    )

    # Initialize WandB
    if not args.no_wandb:
        print("Initializing WandB...")
        wandb.init(
            project=args.wandb_project,
            name=f"act_{args.dataset.split('/')[-1]}",
            config={
                "dataset": args.dataset,
                "training_steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "chunk_size": args.chunk_size,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "dataset_frames": len(dataset),
                "dataset_fps": dataset_metadata.fps,
                "device": str(device),
                "cache_dataset": args.cache_dataset,
                "cache_image_size": args.cache_image_size,
            },
        )

    # Training info
    print()
    print("=" * 60)
    print("ACT Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Device: {device}")
    print(f"Dataset caching: {'enabled' if args.cache_dataset else 'disabled'}")
    print(f"WandB: {'disabled' if args.no_wandb else args.wandb_project}")
    print("=" * 60)
    print()

    # Training loop
    print("Starting training...")
    step = 0
    best_loss = float('inf')
    running_loss = 0.0
    running_kl_loss = 0.0
    start_time = time.time()

    data_iter = cycle(dataloader)
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        batch = next(data_iter)

        # Move batch to device and preprocess
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # If using joint actions from EE dataset, rename action_joints to action
        if args.use_joint_actions and 'action_joints' in batch:
            batch['action'] = batch.pop('action_joints')

        batch = preprocessor(batch)

        # Fix tensor dimensions for ACT model expectations:
        # State: ACT expects [batch, state_dim], not [batch, n_obs_steps, state_dim]
        if "observation.state" in batch and isinstance(batch["observation.state"], torch.Tensor):
            if batch["observation.state"].dim() == 3:
                batch["observation.state"] = batch["observation.state"].squeeze(1)

        # Forward pass
        loss, output_dict = policy.forward(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)

        optimizer.step()
        scheduler.step()

        # Update metrics
        running_loss += loss.item()
        if "kld_loss" in output_dict:
            running_kl_loss += output_dict["kld_loss"]

        step += 1
        pbar.update(1)

        # Log to WandB every step
        if not args.no_wandb:
            log_dict = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }
            if "kld_loss" in output_dict:
                log_dict["train/kl_loss"] = output_dict["kld_loss"]
            wandb.log(log_dict, step=step)

        # Logging
        if step % args.log_freq == 0:
            avg_loss = running_loss / args.log_freq
            avg_kl_loss = running_kl_loss / args.log_freq if running_kl_loss > 0 else 0
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_minutes = (args.steps - step) / steps_per_sec / 60

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'kl': f'{avg_kl_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'eta': f'{eta_minutes:.1f}m'
            })

            # Log averaged metrics to WandB
            if not args.no_wandb:
                wandb.log({
                    "train/avg_loss": avg_loss,
                    "train/avg_kl_loss": avg_kl_loss,
                    "train/steps_per_sec": steps_per_sec,
                    "train/eta_minutes": eta_minutes,
                }, step=step)

            running_loss = 0.0
            running_kl_loss = 0.0

            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                if not args.no_wandb:
                    wandb.log({"train/best_loss": best_loss}, step=step)

        # Save checkpoint
        if step % args.save_freq == 0 or step == args.steps:
            checkpoint_dir = output_dir / f"checkpoint_{step:06d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Saving checkpoint to {checkpoint_dir}")
            policy.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)

            # Save training state for resuming
            torch.save({
                'step': step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_dir / "training_state.pt")

            if not args.no_wandb:
                wandb.log({"checkpoint/step": step}, step=step)

            # Run evaluation if enabled
            if args.eval_episodes > 0:
                print(f"\n  Running {args.eval_episodes} evaluation episodes...")
                policy.eval()
                success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error = run_evaluation(
                    policy, preprocessor, postprocessor, device,
                    num_episodes=args.eval_episodes,
                    randomize=args.eval_randomize
                )
                policy.train()
                print(f"  Eval: {success_rate*100:.1f}% success, {avg_steps:.1f} avg steps, {avg_time:.2f}s avg time")

                if not args.no_wandb:
                    log_data = {
                        "eval/success_rate": success_rate,
                        "eval/avg_steps": avg_steps,
                        "eval/avg_time": avg_time,
                    }
                    if ik_failure_rate is not None:
                        log_data["eval/ik_failure_rate"] = ik_failure_rate
                    if avg_ik_error is not None:
                        log_data["eval/avg_ik_error_mm"] = avg_ik_error
                    wandb.log(log_data, step=step)

    pbar.close()

    # Save final model
    print()
    print("Saving final model...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model: {final_dir}")

    # Final evaluation
    if args.eval_episodes > 0:
        print()
        print(f"Running final evaluation ({args.eval_episodes} episodes)...")
        policy.eval()
        success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error = run_evaluation(
            policy, preprocessor, postprocessor, device,
            num_episodes=args.eval_episodes,
            randomize=args.eval_randomize
        )
        print(f"Final success rate: {success_rate*100:.1f}%")
        print(f"Final avg steps: {avg_steps:.1f}")
        print(f"Final avg time: {avg_time:.2f}s")
        if ik_failure_rate is not None:
            print(f"Final IK failure rate: {ik_failure_rate*100:.2f}%")
        if avg_ik_error is not None:
            print(f"Final avg IK error: {avg_ik_error:.2f}mm")

        if not args.no_wandb:
            log_data = {
                "final/success_rate": success_rate,
                "final/avg_steps": avg_steps,
                "final/avg_time": avg_time,
            }
            if ik_failure_rate is not None:
                log_data["final/ik_failure_rate"] = ik_failure_rate
            if avg_ik_error is not None:
                log_data["final/avg_ik_error_mm"] = avg_ik_error
            wandb.log(log_data)

    # Log final metrics and finish WandB
    if not args.no_wandb:
        wandb.log({
            "final/total_time_minutes": elapsed / 60,
            "final/best_loss": best_loss,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
