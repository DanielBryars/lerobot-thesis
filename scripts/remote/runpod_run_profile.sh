# Run forward pass profiler
source /root/octo_env/bin/activate
cd /root/lerobot-thesis
python3 scripts/remote/profile_forward.py 2>&1; echo "PROFILE_SCRIPT_DONE"
