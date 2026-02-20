# Full reinstall PyTorch nightly with CUDA 12.8 (including all deps)
source /root/octo_env/bin/activate
pip install --pre --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -20; echo "TORCH_INSTALL_DONE"
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')" 2>&1; echo "TORCH_VERIFY_DONE"
python3 -c "import torch; x = torch.randn(2,2).cuda(); print('CUDA tensor OK:', x.shape, x.device)" 2>&1; echo "CUDA_TEST_DONE"
