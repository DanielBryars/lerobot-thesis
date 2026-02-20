#!/usr/bin/env python3
"""Patch octo-pytorch text_processing.py to handle missing FlaxAutoModel."""
import os

fpath = "/root/octo-pytorch/octo/data/utils/text_processing.py"
with open(fpath, "r") as f:
    content = f.read()

# Replace the problematic import line
old = "from transformers import AutoTokenizer, FlaxAutoModel  # lazy import"
new = """from transformers import AutoTokenizer  # lazy import
        try:
            from transformers import FlaxAutoModel
        except ImportError:
            FlaxAutoModel = None  # Not available in PyTorch-only install"""

if old in content:
    content = content.replace(old, new)
    with open(fpath, "w") as f:
        f.write(content)
    print("Patched text_processing.py successfully")
else:
    print("Pattern not found - checking if already patched")
    for i, line in enumerate(content.split('\n')):
        if 'FlaxAutoModel' in line or 'AutoTokenizer' in line:
            print(f"  Line {i+1}: {line}")
