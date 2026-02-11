#!/usr/bin/env python
"""
Visualize the blinkering attention mask used in ACT-ViT.

Creates a diagram showing which tokens are masked for each subtask phase,
illustrating how blinkering forces the model to rely on wrist camera only
during PICK_UP and DROP phases.

Usage:
    python scripts/tools/visualize_blinkering.py
    python scripts/tools/visualize_blinkering.py --output blinkering_mask.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent


def create_blinkering_visualization(output_path: str = None):
    """Create a visualization of the blinkering attention mask."""

    # Token layout for ACT-ViT with 2 cameras (ViT-B/16, 224x224 → 14x14 = 196 patches each)
    n_1d = 3  # latent + state + env_state
    n_patches = 196  # per camera (14x14 grid of 16x16 patches)
    n_cameras = 2
    total_tokens = n_1d + n_cameras * n_patches  # 3 + 392 = 395

    subtasks = ["MOVE_TO_SOURCE", "PICK_UP", "MOVE_TO_DEST", "DROP"]
    subtask_colors_bg = ["#e8f5e9", "#fff3e0", "#e3f2fd", "#fce4ec"]
    mask_color = "#d32f2f"  # Red for masked

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), gridspec_kw={"hspace": 0.4})
    fig.suptitle("ACT-ViT Blinkering Attention Mask\n(True = masked, token is ignored in attention)",
                 fontsize=14, fontweight="bold", y=0.98)

    for i, (subtask, ax, bg_color) in enumerate(zip(subtasks, axes, subtask_colors_bg)):
        # Build mask for this subtask
        mask = np.zeros(total_tokens)
        is_blinkered = subtask in ("PICK_UP", "DROP")
        if is_blinkered:
            overhead_start = n_1d + n_patches  # = 199
            mask[overhead_start:] = 1.0

        # Create the visualization as a heatmap-like bar
        ax.set_xlim(0, total_tokens)
        ax.set_ylim(0, 1)

        # Draw token blocks
        # 1D tokens
        for j in range(n_1d):
            labels = ["Latent", "State", "Env\nState"]
            color = "#4caf50"  # Green = active
            rect = mpatches.FancyBboxPatch((j, 0.1), 0.9, 0.8,
                                            boxstyle="round,pad=0.05",
                                            facecolor=color, edgecolor="black", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(j + 0.45, 0.5, labels[j], ha="center", va="center",
                    fontsize=5, fontweight="bold", color="white")

        # Wrist camera patches (always active)
        wrist_start = n_1d
        wrist_end = n_1d + n_patches
        rect = mpatches.FancyBboxPatch((wrist_start, 0.1), n_patches, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor="#2196f3", edgecolor="black", linewidth=1)
        ax.add_patch(rect)
        ax.text(wrist_start + n_patches / 2, 0.5,
                f"Wrist Camera\n196 patches (14×14)\nALWAYS ACTIVE",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        # Overhead camera patches (masked during PICK_UP/DROP)
        overhead_start = wrist_end
        overhead_end = overhead_start + n_patches
        if is_blinkered:
            oh_color = mask_color
            oh_label = f"Overhead Camera\n196 patches (14×14)\nMASKED"
            oh_style = "round,pad=0.1"
        else:
            oh_color = "#ff9800"
            oh_label = f"Overhead Camera\n196 patches (14×14)\nACTIVE"
            oh_style = "round,pad=0.1"

        rect = mpatches.FancyBboxPatch((overhead_start, 0.1), n_patches, 0.8,
                                        boxstyle=oh_style,
                                        facecolor=oh_color, edgecolor="black", linewidth=1)
        ax.add_patch(rect)
        ax.text(overhead_start + n_patches / 2, 0.5, oh_label,
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        # Add hatching for masked region
        if is_blinkered:
            hatch_rect = mpatches.FancyBboxPatch(
                (overhead_start, 0.1), n_patches, 0.8,
                boxstyle="round,pad=0.1",
                facecolor="none", edgecolor="white", linewidth=0, hatch="////"
            )
            ax.add_patch(hatch_rect)

        # Subtask label
        status = " [BLINKERED]" if is_blinkered else ""
        ax.set_title(f"Subtask {i}: {subtask}{status}", fontsize=11,
                     fontweight="bold", loc="left",
                     color=mask_color if is_blinkered else "#333333")

        # Axis formatting
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Token index labels
        ax.set_xticks([0, n_1d, wrist_start + n_patches / 2, overhead_start, overhead_start + n_patches / 2, total_tokens])
        ax.set_xticklabels(["0", f"{n_1d}", f"idx {n_1d + n_patches//2}", f"{overhead_start}", f"idx {overhead_start + n_patches//2}", f"{total_tokens}"],
                           fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4caf50", edgecolor="black", label="1D tokens (latent, state, env_state)"),
        mpatches.Patch(facecolor="#2196f3", edgecolor="black", label="Wrist camera patches (position-invariant, egocentric)"),
        mpatches.Patch(facecolor="#ff9800", edgecolor="black", label="Overhead camera patches (active)"),
        mpatches.Patch(facecolor=mask_color, edgecolor="black", hatch="////", label="Overhead camera patches (MASKED by blinkering)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    plt.subplots_adjust(bottom=0.12, top=0.90)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        default_path = "outputs/experiments/blinkering_mask_visualization.png"
        Path(default_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {default_path}")
    plt.close()


def create_attention_pattern_visualization(output_path: str = None):
    """Create a visualization showing the attention pattern (what can attend to what)."""

    n_1d = 3
    n_patches = 196
    total = n_1d + 2 * n_patches  # 395

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Encoder Self-Attention: Normal vs Blinkered (key_padding_mask)\n"
                 "Green = can attend, Red = blocked (key masked)",
                 fontsize=13, fontweight="bold")

    for ax_idx, (ax, title, blinkered) in enumerate(zip(
        axes, ["Normal (MOVE_TO_SOURCE / MOVE_TO_DEST)", "Blinkered (PICK_UP / DROP)"], [False, True]
    )):
        # Build attention mask matrix (what query tokens can attend to what key tokens)
        # In self-attention: mask[i,j]=True means query i CANNOT attend to key j
        attn_mask = np.zeros((total, total))

        if blinkered:
            overhead_start = n_1d + n_patches
            # key_padding_mask masks KEY (column) positions only
            # No query token can attend TO overhead keys (columns masked)
            attn_mask[:, overhead_start:] = 1.0
            # NOTE: Overhead tokens as QUERIES can still attend to non-masked keys
            # (rows are NOT masked by key_padding_mask)
            # Their outputs are ignored because:
            # 1. No other encoder token attends to them (column mask)
            # 2. Decoder uses memory_key_padding_mask to ignore them too

        # Visualize with a compact representation (downsample for readability)
        # Group tokens: [1D(3), Wrist(196), Overhead(196)]
        group_sizes = [n_1d, n_patches, n_patches]
        group_labels = ["1D\n(3)", "Wrist\n(196)", "Overhead\n(196)"]
        n_groups = len(group_sizes)

        # Aggregate: for each group pair, what fraction is masked?
        group_mask = np.zeros((n_groups, n_groups))
        row_start = 0
        for gi, gs in enumerate(group_sizes):
            col_start = 0
            for gj, gs2 in enumerate(group_sizes):
                block = attn_mask[row_start:row_start+gs, col_start:col_start+gs2]
                group_mask[gi, gj] = block.mean()
                col_start += gs2
            row_start += gs

        # Draw heatmap
        im = ax.imshow(group_mask, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(group_labels, fontsize=9)
        ax.set_xlabel("Key (attended to)", fontsize=10)
        ax.set_ylabel("Query (attending)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Add text annotations
        for gi in range(n_groups):
            for gj in range(n_groups):
                val = group_mask[gi, gj]
                text = "BLOCKED" if val > 0.5 else "OK"
                color = "white" if val > 0.5 else "black"
                ax.text(gj, gi, text, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    plt.tight_layout()

    path = output_path or "outputs/experiments/blinkering_attention_pattern.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    create_blinkering_visualization(args.output)
    create_attention_pattern_visualization()
