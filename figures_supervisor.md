# Figures (Supervisor Set)

Keep this list short in the main write-up; link the rest in an appendix.

## Must-Show (Existing)

1) Data scaling curve
- `outputs/experiments/data_scaling/data_scaling_results.png`

2) Spatial generalization heatmaps (single-position ACT baseline)
- `outputs/experiments/spatial_heatmap_wide.png`
- `outputs/experiments/spatial_heatmap_fine.png`
- `outputs/experiments/spatial_success_vs_distance.png`

3) Confuser failure (mode collapse visual)
- `outputs/experiments/200K Model Colapse half way between blocks.png`

4) Best subtask+coords model spatial evaluation
- `outputs/experiments/spatial_7b_heatmap.png`
- `outputs/experiments/spatial_7b_distance.png`
- `outputs/experiments/spatial_7b_comparison.png`

5) Blinkering mechanism visualizations
- `outputs/experiments/blinkering_mask_visualization.png`
- `outputs/experiments/blinkering_attention_pattern.png`

6) Temporal ensembling visualization
- `outputs/experiments/temporal_ensemble_viz.png`

## Pickup Spatial Evidence (Existing)

- Videos: `outputs/experiments/pickup_recordings/*.mp4`

Suggested: embed 1 success + 1 fail in appendix.
- `outputs/experiments/pickup_recordings/pickup_near_train_pos2_SUCCESS.mp4`
- `outputs/experiments/pickup_recordings/pickup_novel_between_FAIL.mp4`

## Placeholders (Generate If Needed)

- FIG-PH-01: SmolVLA checkpoint vs success (show 0% across all checkpoints, both variants)
- FIG-PH-02: Pi0 before/after postprocessor (action magnitude + one rollout trajectory)
- FIG-PH-03: Confuser: ResNet vs ViT success comparison bar chart
- FIG-PH-04: Subtask+coords: selective vs full coords ablation (success, pick, drop rates)
