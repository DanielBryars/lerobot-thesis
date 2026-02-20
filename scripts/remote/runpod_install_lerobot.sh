# Install lerobot in the Python 3.10 venv
source /root/octo_env/bin/activate
cd /root/lerobot-thesis

# Install lerobot (editable, minimal - no robot hardware deps)
pip install -e src/lerobot 2>&1 | tail -20; echo "LEROBOT_INSTALL_DONE"

# Also install lerobot_robot_sim
pip install -e src/lerobot_robot_sim 2>&1 | tail -5; echo "SIM_INSTALL_DONE"

# Verify import
python3 -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobotDataset import OK')" 2>&1; echo "VERIFY_DONE"
