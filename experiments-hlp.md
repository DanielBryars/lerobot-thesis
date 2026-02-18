# High-Level Planning Experiments

## Overview

Experiments with high-level task planning for the pick-and-place robot. The planner takes an overhead camera image, detects objects, and generates a sequence of actions (MOVE-TO, PICKUP, DROP) to accomplish the task.

## Pipeline

1. **Scene scanning** (`scan_scene.py`): GroundingDINO detects objects and returns normalized image coordinates
2. **Task planning** (`create-subtasks.py`): Given detected objects, generate action sequence
3. **Execution**: Low-level ACT policy executes each subtask

## Experiment 1: GroundingDINO Object Detection

**Goal**: Evaluate GroundingDINO-tiny for detecting blocks and bowls in overhead sim images.

**Setup**:
- Model: `IDEA-Research/grounding-dino-tiny`
- Labels: `["white block", "bowl"]`
- Test image: `157-episode-1-starting-frame.png` (overhead cam, 157ep dataset)
- Threshold: 0.3

**Results**: *(pending - run scan_scene.py)*

```bash
python scripts/high-level-planner/scan_scene.py \
    --image scripts/high-level-planner/157-episode-1-starting-frame.png \
    --labels "white block" "bowl" \
    --annotate
```

## Experiment 2: Batch Detection on 220ep Dataset

**Goal**: Test detection across varied block positions from the 220ep dataset.

**Setup**:
- Frames extracted at episodes 0, 50, 100, 150, 200
- Ground truth positions available in manifest.json

```bash
python scripts/high-level-planner/scan_scene.py \
    --manifest scripts/high-level-planner/frames/manifest.json \
    --labels "white block" "bowl" \
    --annotate \
    --output-json scripts/high-level-planner/frames/detections.json
```

**Results**: *(pending)*

## Notes

- GroundingDINO operates on RGB images, no depth needed
- Coordinates are normalized [0,1] image space (top-left origin)
- The ChatGPT vision baseline (`chat-gpt/create-subtasks.py`) also returns normalized coords for comparison
- Detection threshold may need tuning for sim images (synthetic textures differ from real)
