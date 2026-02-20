# Install PyTorch nightly with CUDA 12.8 (sm_120 / Blackwell support)
source /root/octo_env/bin/activate

# Install PyTorch nightly with CUDA 12.8
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -5; echo "TORCH_INSTALL_DONE"

# Verify
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); x = torch.randn(2,2).cuda(); print('CUDA tensor OK:', x.shape)" 2>&1; echo "TORCH_VERIFY_DONE"
