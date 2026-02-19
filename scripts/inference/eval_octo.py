#!/usr/bin/env python
"""
Evaluate a fine-tuned Octo-Small model in simulation.

Wraps Octo model in a policy class with select_action() interface
compatible with run_evaluation() and run_pickup_episodes().

Inference loop:
1. Get observation from sim
2. Build Octo observation dict (2-frame history buffer)
3. Run model forward -> get action_horizon delta joint actions
4. Convert deltas to absolute: absolute[0] = state + delta[0],
   absolute[i] = absolute[i-1] + delta[i]
5. Execute all actions, then re-predict (receding horizon)

Usage:
    # Full pick-and-place eval at training positions
    python scripts/inference/eval_octo.py outputs/train/octo_XXXX/final \
        --episodes 50

    # At specific block position
    python scripts/inference/eval_octo.py outputs/train/octo_XXXX/final \
        --episodes 20 --block-x 0.213 --block-y 0.254

    # Pickup-only spatial grid
    python scripts/inference/eval_octo.py outputs/train/octo_XXXX/final \
        --episodes 5 --pickup-only --grid-size 5

    # With visualization
    python scripts/inference/eval_octo.py outputs/train/octo_XXXX/final \
        --episodes 10 --visualize
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.training import run_evaluation, prepare_obs_for_policy
from utils.constants import MOTOR_NAMES, NUM_JOINTS


class OctoPolicy:
    """Wraps a fine-tuned Octo-Small model as a robot policy.

    Implements select_action(batch) compatible with run_evaluation()
    and run_pickup_episodes().

    Handles:
    - 2-frame observation history buffering
    - Delta-to-absolute action conversion
    - Receding horizon execution (predict action_horizon steps, execute all, re-predict)
    """

    PRIMARY_SIZE = 256
    WRIST_SIZE = 128

    def __init__(self, model_path: str, device: str = "cuda"):
        model_path = Path(model_path)
        self.device = torch.device(device)

        # Load training metadata
        meta_path = model_path / "training_metadata.json"
        with open(meta_path) as f:
            self.training_meta = json.load(f)

        # Load dataset stats (for action denormalization if needed)
        stats_path = model_path / "dataset_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.dataset_stats = json.load(f)
        else:
            self.dataset_stats = {}

        self.action_horizon = self.training_meta.get("action_horizon", 4)
        self.action_dim = self.training_meta.get("action_dim", NUM_JOINTS)
        self.use_wrist_cam = self.training_meta.get("use_wrist_cam", True)
        self.use_proprio = self.training_meta.get("use_proprio", True)
        self.instruction = self.training_meta.get("instruction",
                                                   "Pick up the block and place it in the bowl")

        # Load Octo model
        from octo.model.octo_model_pt import OctoModelPt

        # Try loading as our checkpoint format first
        model_weights_path = model_path / "model.pt"
        if model_weights_path.exists():
            # Our custom checkpoint format
            pretrained_path = self.training_meta.get("pretrained_path",
                                                      "hf://rail-berkeley/octo-small-1.5")
            self._load_from_custom_checkpoint(model_path, pretrained_path)
        else:
            # Try OctoModelPt.load_pretrained format
            result = OctoModelPt.load_pretrained(str(model_path))
            self.model = result["octo_model"]

        self.model.to(self.device)
        self.model.eval()

        # Camera names from training metadata
        camera_names = self.training_meta.get("cameras", ["overhead_cam"])

        # Build a minimal config for run_evaluation() camera detection
        input_features = {}
        for cam_name in camera_names:
            img_size = self.WRIST_SIZE if "wrist" in cam_name else self.PRIMARY_SIZE
            input_features[f"observation.images.{cam_name}"] = SimpleNamespace(
                shape=(3, img_size, img_size)
            )
        output_features = {
            "action": SimpleNamespace(shape=(self.action_dim,))
        }
        self.config = SimpleNamespace(
            input_features=input_features,
            output_features=output_features,
            chunk_size=self.action_horizon,
            n_action_steps=self.action_horizon,
        )

        # History buffer (stores last 2 observations)
        self._obs_history = deque(maxlen=2)
        # Action queue for receding horizon
        self._action_queue = []
        # Previous target for delta accumulation
        self._prev_target = None

        # Pre-create task (cached, same for all steps)
        self._task = self.model.create_tasks(texts=[self.instruction], device=self.device)

        print(f"  Action horizon: {self.action_horizon}")
        print(f"  Cameras: {camera_names}")
        print(f"  Proprio: {'enabled' if self.use_proprio else 'disabled'}")
        print(f"  Wrist cam: {'enabled' if self.use_wrist_cam else 'disabled'}")

    def _load_from_custom_checkpoint(self, model_path: Path, pretrained_path: str):
        """Load model from our custom checkpoint format (model.pt + config)."""
        from octo.model.octo_model_pt import OctoModelPt
        from octo.utils.spec import ModuleSpec
        from octo.model.components.action_heads_pt import L1ActionHeadPt

        # Rebuild config from pretrained + modifications
        meta = OctoModelPt.load_config_and_meta_from_jax(pretrained_path)

        if not self.use_wrist_cam and "wrist" in meta["config"]["model"]["observation_tokenizers"]:
            del meta["config"]["model"]["observation_tokenizers"]["wrist"]

        if self.use_proprio:
            from octo.model.components.tokenizers_pt import LowdimObsTokenizerPt
            meta["config"]["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
                LowdimObsTokenizerPt,
                n_bins=256,
                bin_type="normal",
                low=-2.0,
                high=2.0,
                obs_keys=["proprio"],
            )

        num_tokens = {"primary": 256, "language": 16, "action": 1}
        if self.use_wrist_cam:
            num_tokens["wrist"] = 64
        if self.use_proprio:
            num_tokens["proprio"] = NUM_JOINTS
        meta["config"]["model"]["num_tokens_dict"] = num_tokens

        meta["config"]["model"]["heads"]["action"] = ModuleSpec.create(
            L1ActionHeadPt,
            input_dim=384,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            readout_key="readout_action",
        )

        self.model = OctoModelPt.from_config(**meta, verbose=False)
        weights = torch.load(model_path / "model.pt", weights_only=True)
        self.model.load_state_dict(weights)
        print(f"  Loaded custom checkpoint from {model_path}")

    def eval(self):
        self.model.eval()
        return self

    def reset(self):
        """Reset between episodes."""
        self._obs_history.clear()
        self._action_queue = []
        self._prev_target = None

    def _process_primary_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Resize primary camera image to 256x256. Input: (1, 3, H, W) or (3, H, W)."""
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return F.interpolate(img_tensor, size=(self.PRIMARY_SIZE, self.PRIMARY_SIZE),
                             mode="bilinear", align_corners=False)

    def _process_wrist_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Resize wrist camera image to 128x128. Input: (1, 3, H, W) or (3, H, W)."""
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return F.interpolate(img_tensor, size=(self.WRIST_SIZE, self.WRIST_SIZE),
                             mode="bilinear", align_corners=False)

    def _build_observation(self, batch: dict) -> dict:
        """Convert policy batch to Octo observation dict with T=2 history."""
        # Extract current observation
        current_obs = {}

        # Primary camera
        primary_key = "observation.images.overhead_cam"
        if primary_key in batch:
            img = batch[primary_key]
            if img.dim() == 4:
                img = img[0]  # Remove batch dim -> (3, H, W)
            current_obs["image_primary"] = self._process_primary_image(img)  # (1, 3, 256, 256)

        # Wrist camera
        if self.use_wrist_cam:
            wrist_key = "observation.images.wrist_cam"
            if wrist_key in batch:
                img = batch[wrist_key]
                if img.dim() == 4:
                    img = img[0]
                current_obs["image_wrist"] = self._process_wrist_image(img)

        # Proprioception
        if self.use_proprio:
            state = batch.get("observation.state")
            if state is not None:
                if state.dim() == 2:
                    state = state[0]  # Remove batch dim
                current_obs["proprio"] = state.unsqueeze(0)  # (1, D)

        # Add to history
        self._obs_history.append(current_obs)

        # Build T=2 observation (pad with repeat if only 1 frame)
        if len(self._obs_history) < 2:
            hist = [current_obs, current_obs]  # Repeat first frame
            pad_mask = torch.tensor([[False, True]], dtype=torch.bool, device=self.device)
        else:
            hist = list(self._obs_history)
            pad_mask = torch.tensor([[True, True]], dtype=torch.bool, device=self.device)

        # Stack along time dimension: (B=1, T=2, ...)
        observations = {"timestep_pad_mask": pad_mask}

        if "image_primary" in current_obs:
            observations["image_primary"] = torch.stack(
                [hist[0]["image_primary"].squeeze(0), hist[1]["image_primary"].squeeze(0)],
                dim=0
            ).unsqueeze(0).to(self.device)  # (1, 2, 3, H, W)

        if self.use_wrist_cam and "image_wrist" in current_obs:
            observations["image_wrist"] = torch.stack(
                [hist[0]["image_wrist"].squeeze(0), hist[1]["image_wrist"].squeeze(0)],
                dim=0
            ).unsqueeze(0).to(self.device)

        if self.use_proprio and "proprio" in current_obs:
            observations["proprio"] = torch.stack(
                [hist[0]["proprio"].squeeze(0), hist[1]["proprio"].squeeze(0)],
                dim=0
            ).unsqueeze(0).to(self.device)  # (1, 2, D)

        return observations

    def _delta_to_absolute(self, deltas: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute joint positions.

        Args:
            deltas: (horizon, D) delta actions
            current_state: (D,) current joint positions

        Returns:
            (horizon, D) absolute joint targets
        """
        absolute = np.zeros_like(deltas)
        arm_dim = NUM_JOINTS - 1  # 5 arm joints, gripper stays absolute

        # First action: relative to current state
        absolute[0, :arm_dim] = current_state[:arm_dim] + deltas[0, :arm_dim]
        if deltas.shape[1] > arm_dim:
            absolute[0, arm_dim:] = deltas[0, arm_dim:]  # Gripper absolute

        # Subsequent: relative to previous target
        for i in range(1, len(deltas)):
            absolute[i, :arm_dim] = absolute[i - 1, :arm_dim] + deltas[i, :arm_dim]
            if deltas.shape[1] > arm_dim:
                absolute[i, arm_dim:] = deltas[i, arm_dim:]

        return absolute

    @torch.no_grad()
    def _generate_actions(self, batch: dict) -> np.ndarray:
        """Run model inference and return (horizon, action_dim) absolute actions."""
        observations = self._build_observation(batch)

        # Get current state for delta->absolute conversion
        state = batch.get("observation.state")
        if state is not None:
            if isinstance(state, torch.Tensor):
                current_state = state.cpu().numpy().flatten()[:self.action_dim]
            else:
                current_state = np.array(state).flatten()[:self.action_dim]
        else:
            current_state = np.zeros(self.action_dim)

        # Run model
        # sample_actions returns (B, horizon, D)
        from octo.utils.train_utils_pt import tree_map
        actions = self.model.sample_actions(
            observations=observations,
            tasks=self._task,
            unnormalization_statistics=None,  # We handle denorm ourselves
            generator=torch.Generator(self.device).manual_seed(0),
        )

        # Convert to numpy (B=1, horizon, D) -> (horizon, D)
        if isinstance(actions, torch.Tensor):
            delta_actions = actions[0].cpu().numpy()
        else:
            delta_actions = np.array(actions[0])

        # Convert deltas to absolute
        absolute_actions = self._delta_to_absolute(delta_actions, current_state)

        return absolute_actions

    def select_action(self, batch: dict) -> torch.Tensor:
        """Return a single action. Generates a full horizon chunk when queue is empty."""
        if not self._action_queue:
            actions = self._generate_actions(batch)
            self._action_queue = list(actions)

        action = self._action_queue.pop(0)
        return torch.from_numpy(action).float().unsqueeze(0)  # (1, D)


# ---------------------------------------------------------------------------
# No-op processors (OctoPolicy handles its own processing)
# ---------------------------------------------------------------------------
def noop_preprocessor(batch):
    return batch


def noop_postprocessor(action):
    return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Octo-Small model")
    parser.add_argument("path", type=str, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--visualize", action="store_true", help="Show camera feeds")
    parser.add_argument("--mujoco-viewer", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--block-x", type=float, default=None, help="Fixed block X")
    parser.add_argument("--block-y", type=float, default=None, help="Fixed block Y")

    # Pickup-only mode
    parser.add_argument("--pickup-only", action="store_true", help="Pickup-only evaluation")
    parser.add_argument("--grid-size", type=int, default=None, help="Spatial grid size")
    parser.add_argument("--lift-height", type=float, default=0.05, help="Lift height (m)")
    parser.add_argument("--x-min", type=float, default=0.10)
    parser.add_argument("--x-max", type=float, default=0.35)
    parser.add_argument("--y-min", type=float, default=0.08)
    parser.add_argument("--y-max", type=float, default=0.38)

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model_path = Path(args.path)

    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}")
        sys.exit(1)

    print(f"Loading Octo-Small from: {model_path}")
    policy = OctoPolicy(str(model_path), device=device)

    if args.pickup_only:
        _run_pickup_eval(policy, args, device)
    else:
        _run_full_eval(policy, args, device)


def _run_full_eval(policy, args, device):
    """Full pick-and-place evaluation."""
    results = run_evaluation(
        policy=policy,
        preprocessor=noop_preprocessor,
        postprocessor=noop_postprocessor,
        device=torch.device(device),
        num_episodes=args.episodes,
        randomize=True,
        action_dim=policy.action_dim,
        max_steps=args.max_steps,
        verbose=True,
        analyze_failures=True,
        visualize=args.visualize,
        mujoco_viewer=args.mujoco_viewer,
        block_x=args.block_x,
        block_y=args.block_y,
    )

    success_rate, avg_steps, avg_time, _, _, failure_summary = results

    print("\n" + "=" * 60)
    print("OCTO-SMALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model: {args.path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Success Rate: {success_rate * 100:.1f}%")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Avg Time: {avg_time:.2f}s")
    print("=" * 60)

    if failure_summary:
        print("\nFailure Analysis:")
        for key, value in failure_summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


def _run_pickup_eval(policy, args, device):
    """Pickup-only evaluation."""
    from scripts.experiments.eval_pickup_model_spatial import run_pickup_episodes
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    camera_names = policy.training_meta.get("cameras", ["overhead_cam"])
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"

    config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=camera_names,
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(config)
    sim.connect()

    # Build position list
    if args.block_x is not None and args.block_y is not None:
        positions = [(args.block_x, args.block_y)]
    elif args.grid_size:
        xs = np.linspace(args.x_min, args.x_max, args.grid_size)
        ys = np.linspace(args.y_min, args.y_max, args.grid_size)
        positions = [(x, y) for x in xs for y in ys]
    else:
        positions = [(0.213, 0.254), (0.213, -0.047)]

    print(f"\nOCTO PICKUP EVALUATION")
    print(f"  Positions: {len(positions)}")
    print(f"  Episodes per position: {args.episodes}")
    print()

    all_results = []
    for pos_idx, (x, y) in enumerate(positions):
        print(f"  Position {pos_idx + 1}/{len(positions)}: ({x:.3f}, {y:.3f})...", end=" ", flush=True)

        succ, total, details, _ = run_pickup_episodes(
            sim, policy, noop_preprocessor, noop_postprocessor,
            torch.device(device),
            block_x=x, block_y=y,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            viewer=args.mujoco_viewer,
            lift_height=args.lift_height,
        )

        rate = succ / total if total > 0 else 0
        approach_count = sum(1 for d in details if d["approached"])
        print(f"{rate * 100:.0f}% ({succ}/{total}), approached={approach_count}")
        all_results.append({"x": x, "y": y, "rate": rate, "succ": succ, "total": total})

    sim.disconnect()

    # Summary
    print("\n" + "=" * 60)
    print("OCTO PICKUP SUMMARY")
    print("=" * 60)
    rates = [r["rate"] for r in all_results]
    total_succ = sum(r["succ"] for r in all_results)
    total_eps = sum(r["total"] for r in all_results)
    print(f"  Overall: {total_succ}/{total_eps} ({np.mean(rates) * 100:.1f}%)")
    print(f"  Positions >0%: {sum(1 for r in rates if r > 0)}/{len(rates)}")
    print(f"  Positions 100%: {sum(1 for r in rates if r >= 1.0)}/{len(rates)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
