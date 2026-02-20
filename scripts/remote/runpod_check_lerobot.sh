# Check if lerobot is installed
source /root/octo_env/bin/activate
pip show lerobot 2>&1 | head -5; echo "PIP_SHOW_DONE"
python3 -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobotDataset import OK')" 2>&1; echo "VERIFY_DONE"
