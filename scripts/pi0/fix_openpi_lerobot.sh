#!/bin/bash
# Fix openpi to support lerobot v3 datasets (>=0.4.0)
#
# openpi pins lerobot==0.1.0 which lacks:
# - cameras module (lerobot.cameras.configs)
# - v3 dataset format support
#
# This script patches openpi's pyproject.toml to:
# 1. Change lerobot pin from ==0.1.0 to >=0.4.0
# 2. Remove lerobot git source (conflicts with PyPI)
# 3. Remove numpy<2.0.0 constraint (needed for newer lerobot)
# 4. Remove rlds dependency group (conflicts with numpy>=2)
#
# Usage:
#   cd ~/openpi  # or wherever openpi is cloned
#   bash /path/to/fix_openpi_lerobot.sh
#
# After running, lerobot 0.4.x will be installed with uv sync

set -e

echo "Fixing openpi for lerobot v3 support..."

# Check we're in openpi directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found. Run this from the openpi root directory."
    exit 1
fi

if ! grep -q "openpi" pyproject.toml; then
    echo "ERROR: This doesn't look like the openpi repository."
    exit 1
fi

# 1. Change lerobot pin to >=0.4.0
echo "  [1/6] Updating lerobot version constraint..."
sed -i 's/"lerobot",/"lerobot>=0.4.0",/' pyproject.toml

# 2. Remove lerobot git source
echo "  [2/6] Removing lerobot git source..."
sed -i '/^lerobot = { git/d' pyproject.toml

# 3. Remove numpy<2.0.0 constraint in main pyproject.toml
echo "  [3/6] Removing numpy<2.0.0 constraint..."
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' pyproject.toml

# 4. Remove numpy<2.0.0 constraint in openpi-client
echo "  [4/6] Fixing openpi-client numpy constraint..."
if [ -f "packages/openpi-client/pyproject.toml" ]; then
    sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' packages/openpi-client/pyproject.toml
fi

# 5. Remove rlds dependency group (conflicts with numpy>=2)
echo "  [5/6] Removing rlds dependency group..."
sed -i '/^rlds = \[/,/^\]/d' pyproject.toml
sed -i '/^dlimp = { git/d' pyproject.toml

# 6. Sync dependencies
echo "  [6/6] Running uv sync..."
uv sync

# Verify
echo ""
echo "Verifying lerobot version..."
VERSION=$(uv run python -c "import lerobot; print(lerobot.__version__)" 2>/dev/null || echo "FAILED")

if [[ "$VERSION" == "0.4"* ]]; then
    echo "SUCCESS: lerobot $VERSION installed"
else
    echo "WARNING: lerobot version is $VERSION (expected 0.4.x)"
fi
