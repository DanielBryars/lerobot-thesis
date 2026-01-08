# Claude Code Notes

## Code Organization Rules

1. **No code duplication** - Shared functions go in `utils/` modules
2. **Training utilities** - All shared training code goes in `utils/training.py`
3. **Constants** - All constants go in `utils/constants.py`
4. **Conversions** - Unit conversions go in `utils/conversions.py`

## Key Functions in utils/training.py

- `save_checkpoint()` - Save checkpoints with training_metadata (REQUIRED)
- `load_checkpoint()` - Load checkpoints
- `run_evaluation()` - Run evaluation episodes (used by both training scripts AND eval.py)
- `prepare_obs_for_policy()` - Prepare observations for policy inference
- `CachedDataset` - Dataset wrapper for memory caching

## Training Metadata

Every checkpoint MUST include `training_metadata.json` with:
- `dataset_repo_id` - HuggingFace dataset ID
- `scene` - MuJoCo scene XML used
- `cameras` - List of camera names
- `camera_resolutions` - Dict of camera resolutions
- `action_space` - Action space description
- `action_dim` - Action dimension
- `chunk_size` - Action chunk size
- `fps` - Dataset FPS

## Folder Structure

```
scripts/
  inference/   - Evaluation scripts (eval.py)
  training/    - Training scripts (train_act.py, train_smolvla.py)
  remote/      - Remote training (Dockerfile, train_remote.py)
  tools/       - Utility scripts (teleop, FK/IK testing, etc.)
utils/         - Shared Python modules
```
