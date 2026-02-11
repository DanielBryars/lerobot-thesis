# Dataset & Model Index

All HuggingFace artifacts under the [`danbhf`](https://huggingface.co/danbhf) namespace.

---

## Raw Recording Datasets

Individual 20-episode teleoperation recording sessions.

### Position 1 (default ~0.217, 0.225)

| Dataset | Episodes | Frames | Avg Duration | Start Pos | Date |
|---------|----------|--------|--------------|-----------|------|
| [sim_pick_place_20251229_101340](https://huggingface.co/datasets/danbhf/sim_pick_place_20251229_101340) | 20 | 3,276 | 5.5s | Fixed | 2025-12-29 |
| [sim_pick_place_20251229_144730](https://huggingface.co/datasets/danbhf/sim_pick_place_20251229_144730) | 20 | 3,783 | 6.3s | Fixed | 2025-12-29 |
| [sim_pick_place_20260116_000212](https://huggingface.co/datasets/danbhf/sim_pick_place_20260116_000212) | 20 | 3,066 | 5.1s | Fixed | 2026-01-16 |
| [sim_pick_place_20260116_001731](https://huggingface.co/datasets/danbhf/sim_pick_place_20260116_001731) | 20 | 2,796 | 4.7s | Random | 2026-01-16 |
| [sim_pick_place_20260116_002742](https://huggingface.co/datasets/danbhf/sim_pick_place_20260116_002742) | 20 | 2,695 | 4.5s | Random | 2026-01-16 |
| [sim_pick_place_20260117_182521](https://huggingface.co/datasets/danbhf/sim_pick_place_20260117_182521) | 18 | 2,362 | 4.4s | Random | 2026-01-17 |
| [sim_pick_place_20260117_184751](https://huggingface.co/datasets/danbhf/sim_pick_place_20260117_184751) | 20 | 2,638 | 4.4s | Random | 2026-01-17 |
| [sim_pick_place_20260117_190108](https://huggingface.co/datasets/danbhf/sim_pick_place_20260117_190108) | 19 | 2,418 | 4.3s | Random | 2026-01-17 |

### Position 2 (0.337, -0.015)

| Dataset | Episodes | Frames | Date |
|---------|----------|--------|------|
| [sim_pick_place_20260124_142023](https://huggingface.co/datasets/danbhf/sim_pick_place_20260124_142023) | 20 | - | 2026-01-24 |
| [sim_pick_place_20260124_181337](https://huggingface.co/datasets/danbhf/sim_pick_place_20260124_181337) | 20 | 2,828 | 2026-01-24 |
| [sim_pick_place_20260124_183145](https://huggingface.co/datasets/danbhf/sim_pick_place_20260124_183145) | 20 | 2,769 | 2026-01-24 |
| [sim_pick_place_20260124_191458](https://huggingface.co/datasets/danbhf/sim_pick_place_20260124_191458) | 20 | 2,747 | 2026-01-24 |
| [sim_pick_place_20260124_192424](https://huggingface.co/datasets/danbhf/sim_pick_place_20260124_192424) | 20 | 2,193 | 2026-01-24 |

### Gap-filling (various positions)

| Dataset | Episodes | Description |
|---------|----------|-------------|
| [sim_pick_place_targeted_20260125_124212](https://huggingface.co/datasets/danbhf/sim_pick_place_targeted_20260125_124212) | 20 | 20 episodes at 20 unique positions for gap-filling |

### Real Robot

| Dataset | Description | Date |
|---------|-------------|------|
| [real_pick_place_0001](https://huggingface.co/datasets/danbhf/real_pick_place_0001) | First real robot recording: pick up block and place in bowl | 2026-01-17 |

---

## Merged / Processed Datasets

Training-ready datasets built from the raw recordings above.

| Dataset | Episodes | Frames | Action Space | Description | Used In |
|---------|----------|--------|--------------|-------------|---------|
| [sim_pick_place_merged_40ep](https://huggingface.co/datasets/danbhf/sim_pick_place_merged_40ep) | 40 | 6,559 | Joint (6-dim) | Merged from 2x 20251229 recordings | Exp 2, 5, 7-11, Pi0 finetune |
| [sim_pick_place_merged_40ep_ee](https://huggingface.co/datasets/danbhf/sim_pick_place_merged_40ep_ee) | 40 | 6,559 | EE (8-dim) | FK-converted to end-effector space (had quaternion sign bug) | Exp 12 |
| [sim_pick_place_merged_40ep_ee_2](https://huggingface.co/datasets/danbhf/sim_pick_place_merged_40ep_ee_2) | 40 | 6,559 | EE (8-dim) | Fixed version with quaternion continuity | Exp 15, 16 |
| [sim_pick_place_40ep_rgbd_ee](https://huggingface.co/datasets/danbhf/sim_pick_place_40ep_rgbd_ee) | 40 | 6,559 | EE (8-dim) | Re-recorded with D435-style depth camera (RGBD) | Exp 1, 1b, 17-21, SmolVLA |
| [sim_pick_place_157ep](https://huggingface.co/datasets/danbhf/sim_pick_place_157ep) | 157 | 22,534 | Joint (6-dim) | All 8 position-1 recordings merged. Main single-position dataset | Exp 3, 4, 5-11, ACT-ViT, pickup coords |
| [sim_pick_place_157ep_pi0](https://huggingface.co/datasets/danbhf/sim_pick_place_157ep_pi0) | 157 | 22,534 | Joint (6-dim) | Pi0-ready version with gripper normalized to [0-1] | Exp 4d |
| [sim_pick_place_pos2_100ep](https://huggingface.co/datasets/danbhf/sim_pick_place_pos2_100ep) | 100 | 13,700 | Joint (6-dim) | All 5 position-2 recordings merged | act_pos2_100ep training |
| [sim_pick_place_2pos_200ep_v2](https://huggingface.co/datasets/danbhf/sim_pick_place_2pos_200ep_v2) | 200 | 28,816 | Joint (6-dim) | 100ep pos1 + 100ep pos2 merged | Multi-position training |
| [sim_pick_place_2pos_220ep_v2](https://huggingface.co/datasets/danbhf/sim_pick_place_2pos_220ep_v2) | 220 | ~31,210 | Joint (6-dim) | 200ep + 20 gap-fill episodes. Main multi-position dataset | Exp 12, spatial gen, blinkering |
| [sim_pick_place_2pos_220ep_confuser](https://huggingface.co/datasets/danbhf/sim_pick_place_2pos_220ep_confuser) | 220 | ~31,210 | Joint (6-dim) | Re-recorded with white confuser block at fixed pos (0.25, 0.05) | Confuser exp 1, A-B-A |
| [sim_pick_place_2pos_220ep_confuser_rand](https://huggingface.co/datasets/danbhf/sim_pick_place_2pos_220ep_confuser_rand) | 220 | - | Joint (6-dim) | Confuser randomized +/-3cm and full rotation | Confuser exp 2 |
| [sim_pick_place_220ep_confuser_5x](https://huggingface.co/datasets/danbhf/sim_pick_place_220ep_confuser_5x) | 1,100 | - | Joint (6-dim) | 220ep x 5 copies, confuser at random workspace positions | Confuser exp 3 |
| [sim_pick_place_220ep_confuser_mixed_5x](https://huggingface.co/datasets/danbhf/sim_pick_place_220ep_confuser_mixed_5x) | 1,100 | - | Joint (6-dim) | 4:1 confuser:no-confuser ratio mixed dataset | Exp 5c/5d, ACT-ViT pickup coords |

### Superseded Datasets

| Dataset | Notes |
|---------|-------|
| [sim_pick_place_2pos_200ep](https://huggingface.co/datasets/danbhf/sim_pick_place_2pos_200ep) | Early 2-position merge, superseded by v2 |
| [sim_pick_place_2pos_220ep](https://huggingface.co/datasets/danbhf/sim_pick_place_2pos_220ep) | Had video chunk indexing issue, superseded by v2 |

---

## Dataset Lineage

```
Raw 20251229 (2x20ep) --> merged_40ep (joint)
                            |-> merged_40ep_ee (EE, quat bug)
                            |     |-> merged_40ep_ee_2 (quat fixed)
                            |           |-> 40ep_rgbd_ee (re-recorded with depth)
                            |-> openpi_sim_pick_place (local NPZ)

Raw 20260116-17 (6x~20ep) + Raw 20251229 --> 157ep (joint)
                                                |-> 157ep_pi0 (gripper [0-1])

Raw 20260124 (5x20ep) --> pos2_100ep

157ep (pos1) + pos2_100ep --> 2pos_200ep_v2
  + targeted gap-fill (20ep) --> 2pos_220ep_v2 (main multi-position dataset)
                                   |-> 2pos_220ep_confuser (fixed confuser)
                                   |-> 2pos_220ep_confuser_rand (random confuser)
                                   |-> 220ep_confuser_5x (5x augmented, 1100ep)
                                   |-> 220ep_confuser_mixed_5x (4:1 confuser:clean)
```

---

## Trained Models

Model checkpoints uploaded to HuggingFace.

| Model | Trained On | Description |
|-------|-----------|-------------|
| [act_sim_pick_place_100k](https://huggingface.co/danbhf/act_sim_pick_place_100k) | 40ep_rgbd_ee | ACT baseline, RGBD + EE space |
| [smolvla_sim_pick_place_200k](https://huggingface.co/danbhf/smolvla_sim_pick_place_200k) | 40ep_rgbd_ee | SmolVLA, RGBD + EE space |
| [smolvla_so101_200k](https://huggingface.co/danbhf/smolvla_so101_200k) | merged_40ep | SmolVLA, RGB + joint space |
| [pi0_so101_lerobot](https://huggingface.co/danbhf/pi0_so101_lerobot) | 157ep | Pi0, 5K steps |
| [pi0_so101_lerobot_20k](https://huggingface.co/danbhf/pi0_so101_lerobot_20k) | 157ep | Pi0, 20K steps |
| [pi0_so101_pick_place_157](https://huggingface.co/danbhf/pi0_so101_pick_place_157) | 157ep_pi0 | Pi0 on Pi0-format dataset |
| [pi0_so101_test](https://huggingface.co/danbhf/pi0_so101_test) | 157ep | Pi0 test/debug run |
| [pi0_so101_20260110](https://huggingface.co/danbhf/pi0_so101_20260110) | merged_40ep | Pi0 early experiment |
| [act_joint_space_90pct](https://huggingface.co/danbhf/act_joint_space_90pct) | merged_40ep | ACT joint-space, 73.3% success |
| [act_ee_space_90pct](https://huggingface.co/danbhf/act_ee_space_90pct) | merged_40ep_ee_2 | ACT EE-space, 63.3% success |
| [act_pos2_100ep](https://huggingface.co/danbhf/act_pos2_100ep) | pos2_100ep | ACT on position 2 only |
| [act_2pos_200ep](https://huggingface.co/danbhf/act_2pos_200ep) | 2pos_200ep_v2 | ACT on both positions |
| [act_2pos_220ep_100k](https://huggingface.co/danbhf/act_2pos_220ep_100k) | 2pos_220ep_v2 | ACT on 220ep, best at 60K (95%) |
| [act_confuser_220ep](https://huggingface.co/danbhf/act_confuser_220ep) | 2pos_220ep_confuser | ACT with fixed confuser |
| [act_confuser_rand_220ep](https://huggingface.co/danbhf/act_confuser_rand_220ep) | 2pos_220ep_confuser_rand | ACT with randomized confuser |
| [act_confuser_5x_1100ep](https://huggingface.co/danbhf/act_confuser_5x_1100ep) | 220ep_confuser_5x | ACT on 1100ep confuser-augmented |
| [act_pickup_coords_157ep](https://huggingface.co/danbhf/act_pickup_coords_157ep) | 157ep | ACT with pickup coords, no confuser |
| [act_pickup_coords_confuser_mixed](https://huggingface.co/danbhf/act_pickup_coords_confuser_mixed) | 220ep_confuser_mixed_5x | ACT + pickup coords, 50K steps |
| [act_pickup_coords_confuser_mixed_200k](https://huggingface.co/danbhf/act_pickup_coords_confuser_mixed_200k) | 220ep_confuser_mixed_5x | ACT + pickup coords, 200K steps |
| [act_pickup_coords_pos1_confuser](https://huggingface.co/danbhf/act_pickup_coords_pos1_confuser) | 220ep_confuser_mixed_5x (pos1) | ACT + pickup coords, pos1 filtered |
| [act_vit_157ep](https://huggingface.co/danbhf/act_vit_157ep) | 157ep | ACT-ViT (ViT-B/16), 157ep |
| [act_vit_pickup_coords_157ep](https://huggingface.co/danbhf/act_vit_pickup_coords_157ep) | 157ep | ACT-ViT + pickup coords, 157ep |
| [act_vit_pickup_coords_confuser_mixed](https://huggingface.co/danbhf/act_vit_pickup_coords_confuser_mixed) | 220ep_confuser_mixed_5x | ACT-ViT + pickup coords, mixed confuser |
