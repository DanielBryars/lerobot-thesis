# Fix numpy version and launch training
source /root/octo_env/bin/activate
# Downgrade numpy for compatibility with tensorflow/scipy
pip install "numpy==1.26.4" "opencv-python-headless<4.11" 2>&1 | tail -5; echo "NUMPY_FIX_DONE"
# Verify torch still works with numpy 1.26
python3 -c "import numpy; print('numpy:', numpy.__version__); import torch; print('torch:', torch.__version__); x = torch.randn(2,2).cuda(); print('CUDA OK:', x.device)" 2>&1; echo "TORCH_VERIFY_DONE"
# Check that imageio/av still work
python3 -c "import imageio; import av; print('imageio OK, av OK')" 2>&1; echo "IMAGEIO_VERIFY_DONE"
# Check transformers
python3 -c "import transformers; print('transformers:', transformers.__version__)" 2>&1; echo "TRANSFORMERS_VERIFY_DONE"
