# Thesis Experiments Timeline (Core)

Audience: supervisor. Chronological, headline-level. Focus: core modelling/experiment findings (not tooling).

Columns: ID | When | Model / Dataset | Question / Change | Eval | Result | Takeaway / Next

---

| ID | When | Model / Dataset | Question / Change | Eval | Result | Takeaway / Next |
|---:|:-----|:-----------------|:------------------|:-----|:-------|:----------------|
| C01 | 2026-01-06..07 | ACT (ResNet), joint-space vs EE-space | Establish baselines; compare joint vs EE (IK) actions | 30 eps | Joint 73.3% vs EE 63.3% (IK fail ~28%) | Joint-space more reliable; EE-space limited by IK failures. |
| C02 | 2026-01-08..10 | SmolVLA (RGBD+EE; then RGB+joints) | Test compact VLA on task | Many checkpoints | 0% success across all checkpoints (both variants) | SmolVLA recipe/inference likely not viable here without deeper changes; keep as negative result. |
| C03 | 2026-01-10 | Pi0 (openpi/JAX, 40ep) | Test VLA flow model on small dataset | 10 eps | 0% (moves somewhat; gripper regresses to mean) | VLA needs more data/recipe; gripper is failure mode at low data. |
| C04 | 2026-01-18 | ACT (157 episodes) | Data scaling sanity check at fixed position | 10 eps/checkpoint | Best checkpoint_045000 = 100% (10/10); final 80% | More data improves peak success; mild overfit after peak. |
| C05 | 2026-01-19..21 | Pi0 (LeRobot PyTorch, 157ep) | Re-attempt Pi0 with working training stack | Sim eval + viz | Initially 0%; root cause = missing action denormalization; after fix robot moves; first successes observed later | Pre/post-processing is critical; VLA can appear “dead” with a single missing postprocessor. |


| C06 | 2026-01-20 | ACT chunking rollout length | Does shorter execution horizon help? | 10 eps/setting | n_action_steps 1-5 => 0%; sweet spot ~20 => 100% | There is an optimal replan frequency; too-frequent replanning destabilizes. |
| C07 | 2026-01-21..22 | ACT spatial generalization (single-pos) | Quantify spatial generalization radius | 5x5 + 7x7 grids, 990..2630 eps | Limited: ~50% threshold around ~7cm; far OOD mostly 0–10% | BC policies memorize workspace region; need data diversity or explicit conditioning. |
| C08 | 2026-01-22 | Temporal ensembling (ACT) | Does ensembling reduce drops? | 50 eps/condition | 82% -> 90%; drops 7 -> 1; never-pick increased | Ensembling smooths transport but can hurt pickup; trade-off. |


| C09 | 2026-01-22..?? | Data scaling experiment (ACT) | Success vs dataset size | Multi-run sweep | Plots produced | Use for thesis: success increases with data; include curve with checkpoints. |
| C10 | 2026-01-31 | Confuser dataset + ViT backbone | Can ViT avoid mode collapse vs ResNet under confuser mixing? | 50 eps | ViT-B/16 ~72% on confuser (1 cam) | ViT improves disambiguation/pick rate; backbone matters more than loss. |
| C11 | 2026-02-01 | ViT-B/32 vs ViT-B/16 | Is coarser patching acceptable? | 50 eps | ViT-B/32 ~60% vs ViT-B/16 ~72% | Spatial resolution matters for manipulation. |
| C12 | 2026-02-01 | Frozen ViT backbone | Do we need fine-tuning? | 50 eps | Frozen backbone ~74% (slightly better) | Pretrained vision features transfer well; freezing can regularize. |
| C13 | 2026-02-02..03 | 2-camera ViT + camera embeddings + chunk_size | Why did 2 cameras degrade? | 50 eps | Chunk_size confounded; chunk 100 => 36%, chunk 50 => 58% | Chunk size interacts with input complexity; control confounds in comparisons. |
| C14 | 2026-02-03 | Coords conditioning (confuser dataset) | Do explicit pickup XY coords help? | 50 eps | 72% -> 76% | Coords give modest gains; main benefit reduces drops. |
| C15 | 2026-02-03 | Ensembling on 2-camera model | Does ensembling generalize across setups? | 50 eps | 58% -> 50% (drop rate improved but timeouts rose) | Ensembling is not universally beneficial; depends on observation complexity. |
| C16 | 2026-02-05 | Subtask conditioning (157ep) | Does subtask token help? | 50 eps | 84% (subtask only) | Subtask conditioning provides useful mode info. |
| C17 | 2026-02-05 | Subtask + coords (157ep) + selective masking | When are coords helpful? | 50 eps | 86% full coords; 90% selective coords; pick rate 100% | Use coords for navigation phases; mask for fine manipulation. |
| C18 | 2026-02-07 | Full spatial grid for best 157ep model (7b) | How does 7b generalize spatially? | 7x7, 245 eps | 18.4% overall; ~8cm 50% radius | Conditioning helps vs single-pos ACT baseline but still limited without diverse positions. |
| C19 | 2026-02-05 | Delta actions (with chunking) | Does delta action space improve generalization? | 10 eps | 0% (fails; chunk/delta mismatch) | Delta actions clash with chunked position control; errors compound within chunk. |
| C20 | 2026-02-08 | Per-subtask failure analysis (7b) | Which phase is bottleneck? | 20 eps | Approach/Pickup/Transport 100%; Drop 80% | Drop/release is primary bottleneck at training position. |
| C21 | 2026-02-08..09 | State fix + “blinkering” (157ep) 2x2 | Remove position leakage; force wrist reliance | 20 eps + pickup grid | Task success drops (90% -> 55–65%); spatial pickup 12% -> 4% -> 0% | Masking/bug-fix alone can’t create invariance; data diversity dominates. |
| C22 | 2026-02-09 | Diverse-position training (220ep, fix_state+blinkering+subtask+coords) | Does data diversity enable invariant pickup? | 20 eps + pickup tests | Task 65% with blinkering (35% w/o); pickup invariance depends on approach method | Diverse data changes outcome: blinkering helps drop; pickup invariance emerges only with in-distribution approach states. |
| C23 | 2026-02-09 | Pickup spatial methods (natural approach vs IK teleport vs teleported-start) | Is PICK_UP a reusable primitive? | 5x5 grid | IK teleport 0%; natural approach: 100% at all reachable positions; teleported-start: 7/25 at 100% | PICK_UP can be position-invariant given in-distribution pre-grasp state; navigation is the limiter. |
| C24 | 2026-02-?? | Completion head + subtask chunks (220ep) | Can we learn transitions / reduce subtask label bleed? | 20 eps | 14a (masked actions) fails; 14b (aux completion only) succeeds and improves drops; final 85% at training positions | Don’t mask action supervision; auxiliary completion loss acts as useful regularizer; completion-based resets hurt.

---

Notes:
- “IK teleport” refers to jumping to a pose via IK before running PICK_UP; it is out-of-distribution for a 5-DOF arm and invalid as a generalization test.
- “Blinkering” = masking overhead camera tokens during PICK_UP/DROP to force reliance on wrist camera.
