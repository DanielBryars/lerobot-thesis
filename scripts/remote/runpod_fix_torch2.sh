# Force install PyTorch nightly with CUDA 12.8
source /root/octo_env/bin/activate

# Force reinstall to get the cu128 nightly build
pip install --pre --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-deps 2>&1 | tail -10; echo "TORCH_INSTALL_DONE"

# Install deps that may have been dropped
pip install filelock sympy networkx jinja2 triton 2>&1 | tail -3; echo "DEPS_DONE"

# Verify
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda)" 2>&1; echo "TORCH_VERIFY_DONE"
