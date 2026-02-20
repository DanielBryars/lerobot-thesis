#!/usr/bin/env python3
"""Run commands on RunPod via SSH using paramiko (handles PTY requirement).

Usage:
    python runpod_ssh.py                          # Quick test
    python runpod_ssh.py commands.sh              # Run commands from file
    python runpod_ssh.py -c "cmd1" "cmd2" "cmd3"  # Run individual commands
"""

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

# Regex to detect shell prompt (root@xxx:/path#)
PROMPT_RE = re.compile(r'root@[a-z0-9]+:[^\n]*#\s*$')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\[\?[0-9]*[a-zA-Z]|\]0;[^\x07\n]*')


def clean_output(data: str) -> str:
    """Strip ANSI escape codes."""
    return ANSI_RE.sub('', data)


def wait_for_prompt(channel, timeout: int = 300) -> str:
    """Read from channel until we see a shell prompt. Returns accumulated output."""
    output = ""
    start = time.time()
    while time.time() - start < timeout:
        if channel.recv_ready():
            data = channel.recv(65536).decode("utf-8", errors="replace")
            clean = clean_output(data)
            print(clean, end="", flush=True)
            output += clean
            # Check if we see a prompt at the end
            if PROMPT_RE.search(output.rstrip()):
                return output
        elif channel.exit_status_ready():
            break
        else:
            time.sleep(0.3)
    return output


def run_commands(commands: list, timeout_per_cmd: int = 300):
    """Connect to RunPod and run commands one at a time, waiting for prompt between each."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    key = paramiko.Ed25519Key.from_private_key_file(KEY_FILE)
    print(f"Connecting to {USER}@{HOST}...")
    client.connect(HOST, username=USER, pkey=key, timeout=30)

    # Open interactive channel with PTY
    channel = client.invoke_shell(term="xterm", width=200, height=50)

    # Wait for initial banner + prompt
    wait_for_prompt(channel, timeout=15)

    # Execute each command and wait for prompt
    for cmd in commands:
        cmd = cmd.strip()
        if not cmd or cmd.startswith('#'):
            continue
        channel.send(cmd + "\n")
        time.sleep(0.2)
        wait_for_prompt(channel, timeout=timeout_per_cmd)

    # Clean exit
    channel.send("exit\n")
    time.sleep(1)
    while channel.recv_ready():
        data = channel.recv(65536).decode("utf-8", errors="replace")
        print(clean_output(data), end="", flush=True)

    channel.close()
    client.close()
    print("\n[Connection closed]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-c":
        # Commands as arguments
        cmds = sys.argv[2:]
    elif len(sys.argv) > 1:
        # Commands from file (one per line, or && separated)
        with open(sys.argv[1]) as f:
            cmds = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    cmds.append(line)
    else:
        cmds = ["hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && echo DONE"]

    run_commands(cmds)
