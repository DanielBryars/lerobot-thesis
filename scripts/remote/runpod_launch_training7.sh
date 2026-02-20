# Launch Octo training variant A (with patched attention mask)
source /root/octo_env/bin/activate
pkill -f train_octo 2>/dev/null; echo "KILLED_OLD"
sleep 3
cd /root/lerobot-thesis
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python3 -u scripts/training/train_octo.py \
    danbhf/sim_pick_place_2pos_220ep_v2_delta \
    --no-wrist-cam --no-proprio \
    --run-name octo_A_overhead_only_v2 \
    --output-dir outputs/train/octo_A_v2 \
    --batch-size 64 \
    --steps 50000 \
    --save-freq 5000 \
    --log-freq 100 \
    > /root/training_octo_A_v2.log 2>&1 &
echo "LAUNCHED PID=$!"
sleep 5
ps aux | grep train_octo | grep -v grep
echo "LAST_10_LINES:"
tail -10 /root/training_octo_A_v2.log
echo "LAUNCH_DONE"
