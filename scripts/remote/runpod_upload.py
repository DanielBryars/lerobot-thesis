#!/usr/bin/env python3
"""Upload files to RunPod via SFTP (paramiko)."""

import os
import sys
import paramiko

HOST = "ssh.runpod.io"
USER = "w2k5as06fwr3jq-64411d37"
KEY_FILE = r"C:\Users\bryar\.ssh\id_ed25519_runpod"
REMOTE_BASE = "/root/lerobot-thesis"


def upload_files(local_base: str, files: list):
    """Upload a list of files (relative paths) to RunPod."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(KEY_FILE)

    print(f"Connecting to {USER}@{HOST}...")
    client.connect(HOST, username=USER, pkey=key, timeout=30)
    sftp = client.open_sftp()

    for rel_path in files:
        local_path = os.path.join(local_base, rel_path)
        remote_path = f"{REMOTE_BASE}/{rel_path}".replace("\\", "/")

        # Ensure remote directory exists
        remote_dir = os.path.dirname(remote_path).replace("\\", "/")
        _mkdirs(sftp, remote_dir)

        print(f"  {rel_path} -> {remote_path}")
        sftp.put(local_path, remote_path)

    sftp.close()
    client.close()
    print(f"\nUploaded {len(files)} files.")


def _mkdirs(sftp, remote_dir: str):
    """Recursively create remote directories."""
    dirs_to_create = []
    current = remote_dir
    while current and current != "/":
        try:
            sftp.stat(current)
            break  # exists
        except FileNotFoundError:
            dirs_to_create.append(current)
            current = os.path.dirname(current).replace("\\", "/")

    for d in reversed(dirs_to_create):
        try:
            sftp.mkdir(d)
        except Exception:
            pass  # may already exist due to race


if __name__ == "__main__":
    local_base = r"E:\git\ai\lerobot-thesis"

    # Files to upload
    files = [
        "scripts/remote/patch_text_processing.py",
        "scripts/training/train_octo.py",
        "scripts/inference/eval_octo.py",
        "scripts/tools/convert_to_delta_actions.py",
        "utils/octo_dataset.py",
        "utils/training.py",
        "utils/constants.py",
    ]

    upload_files(local_base, files)
