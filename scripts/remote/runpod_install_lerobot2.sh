# Install lerobot from PyPI in the Python 3.10 venv
source /root/octo_env/bin/activate

# Install lerobot (standard HuggingFace LeRobot package)
pip install lerobot 2>&1 | tail -20; echo "LEROBOT_INSTALL_DONE"

# Verify import
python3 -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobotDataset import OK')" 2>&1; echo "VERIFY_DONE"
