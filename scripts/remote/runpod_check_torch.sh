# Check PyTorch CUDA compatibility
source /root/octo_env/bin/activate
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')" 2>&1; echo "TORCH_CHECK_DONE"

# Check CUDA toolkit version
nvcc --version 2>&1 | tail -2; echo "NVCC_DONE"
