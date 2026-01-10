#!/bin/bash
# Complete setup script for training Pi0 on SO-101 with LeRobot v3.0 datasets
# Run this on the remote instance ONCE before training
#
# Usage:
#   bash /app/lerobot-thesis/scripts/pi0/setup_openpi_so101.sh

set -e

echo "=========================================="
echo "OpenPI SO-101 Training Setup"
echo "=========================================="
echo ""

# Fix Windows line endings if needed
echo "[0/7] Fixing line endings..."
sed -i 's/\r$//' /app/lerobot-thesis/scripts/pi0/*.sh 2>/dev/null || true
sed -i 's/\r$//' /app/lerobot-thesis/scripts/pi0/*.py 2>/dev/null || true

# Step 1: Update lerobot dependency version
echo "[1/7] Updating lerobot version constraint to >=0.4.0..."
sed -i 's/"lerobot",/"lerobot>=0.4.0",/' /app/openpi/pyproject.toml

# Step 2: Remove numpy<2.0.0 constraint from main pyproject.toml
echo "[2/7] Removing numpy version ceiling from openpi..."
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' /app/openpi/pyproject.toml

# Step 3: Remove numpy<2.0.0 constraint from openpi-client
echo "[3/7] Removing numpy version ceiling from openpi-client..."
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' /app/openpi/packages/openpi-client/pyproject.toml

# Step 4: Remove the lerobot git pin from [tool.uv.sources]
echo "[4/7] Removing lerobot git pin (will use PyPI instead)..."
sed -i '/^lerobot = { git/d' /app/openpi/pyproject.toml

# Step 5: Remove the rlds dependency group (conflicts with numpy>=2.0)
echo "[5/7] Removing rlds dependency group (not needed for LeRobot format)..."
# Remove the entire rlds group block
sed -i '/^rlds = \[/,/^\]/d' /app/openpi/pyproject.toml
# Also remove dlimp source since we removed rlds
sed -i '/^dlimp = { git/d' /app/openpi/pyproject.toml

# Step 6: Delete lock file and resync dependencies
echo "[6/7] Regenerating lock file and syncing dependencies..."
echo "      (This may take a few minutes...)"
cd /app/openpi
rm -f uv.lock
uv sync

# Verify lerobot version
echo ""
echo "=== Dependency Verification ==="
LEROBOT_VERSION=$(.venv/bin/python -c "import lerobot; print(lerobot.__version__)" 2>/dev/null)
NUMPY_VERSION=$(.venv/bin/python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
echo "LeRobot version: $LEROBOT_VERSION"
echo "NumPy version:   $NUMPY_VERSION"

if [[ "$LEROBOT_VERSION" < "0.4.0" ]]; then
    echo "ERROR: LeRobot version too old. Expected >=0.4.0, got $LEROBOT_VERSION"
    exit 1
fi
echo "OK: Dependencies OK"

# Step 7: Patch openpi config.py to add SO-101 support
echo ""
echo "[7/7] Adding SO-101 training configuration..."
cd /app/lerobot-thesis
.venv/bin/python scripts/pi0/patch_openpi_so101.py || python3 scripts/pi0/patch_openpi_so101.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Compute normalization stats:"
echo "   cd /app/openpi && uv run scripts/compute_norm_stats.py --config-name=pi0_so101"
echo ""
echo "2. Run training:"
echo "   cd /app/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \\"
echo "       uv run scripts/train.py pi0_so101 \\"
echo "       --exp-name=so101_pick_place \\"
echo "       --overrides num_train_steps=20000"
echo ""
echo "3. Monitor with WandB:"
echo "   Training logs at: https://wandb.ai"
echo ""
