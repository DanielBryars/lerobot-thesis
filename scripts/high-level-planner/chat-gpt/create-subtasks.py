#!/usr/bin/env python3
"""
Call OpenAI Responses API with an image + a parameterised task, and get back:
- white_block_center: {x,y} normalised to image
- bin_center: {x,y} normalised to image
- plan: list of steps using only MOVE-TO/PICKUP/DROP
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(p: str) -> Path:
    """
    Resolve image path robustly:
    - If absolute, use it
    - Else resolve relative to the script directory (not the current working directory)
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def image_to_data_url(path_str: str) -> str:
    path = resolve_path(path_str)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            f"(You passed: {path_str})\n"
            f"Tip: run with --image <absolute_path> or place the file relative to the script folder:\n"
            f"  {SCRIPT_DIR}"
        )

    lower = path.name.lower()
    if lower.endswith(".png"):
        mime = "image/png"
    elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
        mime = "image/jpeg"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    else:
        raise ValueError("Unsupported image type. Use .png, .jpg/.jpeg, or .webp")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_prompt(task: str) -> str:
    return f"""
You are controlling a robot arm using ONLY these actions:
- MOVE-TO(x, y) where x,y are normalised image coordinates in [0,1] with origin at top-left.
- PICKUP
- DROP

Task: {task}

From the image:
1) Estimate the (x,y) centre of the white lego block mentioned in the task.
2) Estimate the (x,y) centre of the bin mentioned in the task.
3) Produce a plan as a list of steps using ONLY the allowed actions.

Constraints:
- If an action is MOVE-TO, it MUST include x and y.
- If an action is PICKUP or DROP, set x and y to null.
- Return ONLY the JSON object matching the schema.
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        help="Path to input image (png/jpg/webp). Relative paths are resolved relative to this script's folder.",
        # fixed spelling: episode
        default="../157-episode-1-starting-frame.png",
    )
    parser.add_argument(
        "--task",
        help="Task instruction, e.g. 'Put the white lego block into the bin'",
        default="Put the white lego block on the left into the bin on the right",
    )
    parser.add_argument("--model", help="Model id", default="gpt-5")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        return 2

    data_url = image_to_data_url(args.image)
    client = OpenAI()

    # Strict JSON schema output (Structured Outputs)
    # NOTE: OpenAI strict mode does NOT support: minimum/maximum, minItems,
    # allOf, if/then, not, or anyOf (except for nullable types).
    # Plan items use nullable x/y: MOVE-TO gets numbers, PICKUP/DROP get null.
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "white_block_center": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
            },
            "bin_center": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
            },
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {"type": "string", "enum": ["MOVE-TO", "PICKUP", "DROP"]},
                        "x": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                        "y": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    },
                    "required": ["action", "x", "y"],
                },
            },
        },
        "required": ["white_block_center", "bin_center", "plan"],
    }

    t0 = time.perf_counter()
    resp = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt(args.task)},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "robot_arm_plan",
                "strict": True,
                "schema": schema,
            }
        },
    )
    elapsed = time.perf_counter() - t0

    raw = resp.output_text
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        print("ERROR: Model did not return valid JSON.", file=sys.stderr)
        print(raw)
        return 3

    print(json.dumps(obj, indent=2))

    usage = resp.usage
    print(f"\n--- Stats ---", file=sys.stderr)
    print(f"Time:           {elapsed:.2f}s", file=sys.stderr)
    print(f"Input tokens:   {usage.input_tokens}", file=sys.stderr)
    print(f"Output tokens:  {usage.output_tokens}", file=sys.stderr)
    print(f"Total tokens:   {usage.total_tokens}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
