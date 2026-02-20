# Fix numpy incompatibility - downgrade to 1.26.x (compatible with JAX, TF, and PyTorch)
source /root/octo_env/bin/activate

# Install numpy 1.26 (last 1.x version, works with everything)
pip install "numpy==1.26.4" 2>&1 | tail -5; echo "NUMPY_INSTALL_DONE"

# Reinstall opencv-python-headless with older version that supports numpy<2
pip install "opencv-python-headless<4.11" 2>&1 | tail -3; echo "CV2_INSTALL_DONE"

# Verify everything works
python3 -c "import numpy; print('numpy:', numpy.__version__)" 2>&1; echo "CHECK1"
python3 -c "import cv2; print('cv2:', cv2.__version__)" 2>&1; echo "CHECK2"
python3 -c "from octo.model.octo_model_pt import OctoModelPt; print('OctoModelPt OK')" 2>&1; echo "CHECK3"
