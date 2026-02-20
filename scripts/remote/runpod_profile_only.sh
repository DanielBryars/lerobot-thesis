# Kill training, run profiler
source /root/octo_env/bin/activate
pkill -f train_octo 2>/dev/null; echo "KILLED"
sleep 3
cd /root/lerobot-thesis
python3 scripts/remote/profile_forward.py 2>&1; echo "PROFILE_DONE"
