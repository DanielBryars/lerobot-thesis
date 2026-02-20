# Check training status
source /root/octo_env/bin/activate
ps aux | grep train_octo | grep -v grep | head -1; echo "PS_DONE"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; echo "GPU_DONE"
wc -l /root/training_octo_A.log; echo "LINES_DONE"
tail -40 /root/training_octo_A.log 2>&1; echo "LOG_DONE"
