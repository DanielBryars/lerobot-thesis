# Check PyTorch CUDA compatibility and launch training if ready
source /root/octo_env/bin/activate
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')" 2>&1; echo "TORCH_CHECK_DONE"
python3 -c "import torch; x = torch.randn(2,2).cuda(); print('CUDA tensor OK:', x.shape, x.device)" 2>&1; echo "CUDA_TEST_DONE"
