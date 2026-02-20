#!/usr/bin/env python3
"""Push local files to RunPod via SSH PTY using base64 encoding.

Since RunPod doesn't support SFTP, we encode files as base64 and
decode them on the remote end.
"""

import base64
import io
import os
import re
import sys
import time
import paramiko

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HOST = "ssh.runpod.io"
USER = "w2k5as06fwr3jq-64411d37"
KEY_FILE = r"C:\Users\bryar\.ssh\id_ed25519_runpod"
REMOTE_BASE = "/root/lerobot-thesis"
LOCAL_BASE = r"E:\git\ai\lerobot-thesis"

PROMPT_RE = re.compile(r'root@[a-z0-9]+:[^\n]*#\s*$')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\[\?[0-9]*[a-zA-Z]|\]0;[^\x07\n]*')


def clean_output(data: str) -> str:
    return ANSI_RE.sub('', data)


def wait_for_prompt(channel, timeout: int = 60) -> str:
    output = ""
    start = time.time()
    while time.time() - start < timeout:
        if channel.recv_ready():
            data = channel.recv(65536).decode("utf-8", errors="replace")
            clean = clean_output(data)
            output += clean
            if PROMPT_RE.search(output.rstrip()):
                return output
        elif channel.exit_status_ready():
            break
        else:
            time.sleep(0.2)
    return output


def send_cmd(channel, cmd, timeout=60, quiet=False):
    """Send a command and wait for prompt."""
    channel.send(cmd + "\n")
    time.sleep(0.1)
    output = wait_for_prompt(channel, timeout)
    if not quiet:
        # Print just the last relevant lines
        lines = output.strip().split('\n')
        for line in lines[-3:]:
            if line.strip():
                print(f"    {line.strip()}")
    return output


def push_file(channel, local_path, remote_path):
    """Push a single file to remote via base64 encoding."""
    with open(local_path, "rb") as f:
        content = f.read()

    b64 = base64.b64encode(content).decode("ascii")
    size_kb = len(content) / 1024

    print(f"  {os.path.relpath(local_path, LOCAL_BASE)} ({size_kb:.1f} KB)")

    # Ensure remote directory exists
    remote_dir = os.path.dirname(remote_path).replace("\\", "/")
    send_cmd(channel, f"mkdir -p {remote_dir}", quiet=True)

    # Split base64 into chunks (max command line length ~64KB, use 4KB chunks to be safe)
    chunk_size = 4096
    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]

    # Write first chunk (overwrite)
    send_cmd(channel, f"echo '{chunks[0]}' > /tmp/_b64_transfer", quiet=True)

    # Append remaining chunks
    for chunk in chunks[1:]:
        send_cmd(channel, f"echo '{chunk}' >> /tmp/_b64_transfer", quiet=True)

    # Decode and move to target
    send_cmd(channel, f"cat /tmp/_b64_transfer | tr -d '\\n' | base64 -d > {remote_path}", quiet=True)
    send_cmd(channel, f"rm /tmp/_b64_transfer", quiet=True)

    # Verify size
    output = send_cmd(channel, f"wc -c < {remote_path}", quiet=True)
    return True


def main():
    # Files relative to REMOTE_BASE (/root/lerobot-thesis)
    files = [
        "scripts/remote/profile_forward.py",
    ]

    # Extra files with explicit (local_path, remote_path) pairs
    extra_files = [
    ]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(KEY_FILE)

    print(f"Connecting to {USER}@{HOST}...")
    client.connect(HOST, username=USER, pkey=key, timeout=30)
    channel = client.invoke_shell(term="xterm", width=200, height=50)

    # Wait for initial prompt
    wait_for_prompt(channel, timeout=15)

    total = len(files) + len(extra_files)
    print(f"\nPushing {total} files...")

    for rel_path in files:
        local_path = os.path.join(LOCAL_BASE, rel_path)
        remote_path = f"{REMOTE_BASE}/{rel_path}".replace("\\", "/")
        push_file(channel, local_path, remote_path)

    for local_path, remote_path in extra_files:
        push_file(channel, local_path, remote_path)

    # Clean exit
    channel.send("exit\n")
    time.sleep(1)
    channel.close()
    client.close()
    print(f"\nDone! Pushed {total} files.")


if __name__ == "__main__":
    main()
