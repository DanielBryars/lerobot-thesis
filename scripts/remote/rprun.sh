#!/bin/bash
# Run a command on RunPod via SSH
# Usage: bash rprun.sh "command to run"
SSH_KEY="$HOME/.ssh/id_ed25519_runpod"
SSH_HOST="aa28z8y673hfm5-64411b67@ssh.runpod.io"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

CMD="$1 ; exit"
echo "$CMD" | ssh -tt $SSH_OPTS -i "$SSH_KEY" "$SSH_HOST" 2>&1 | sed 's/\r//g' | sed -n '/\$ .*; exit/,/Connection to.*closed/p' | head -n -1 | tail -n +2
