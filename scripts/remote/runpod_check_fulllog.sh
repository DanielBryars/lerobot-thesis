# Check full log and GPU status
source /root/octo_env/bin/activate
wc -l /root/training_octo_A.log; echo "LINE_COUNT_DONE"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; echo "GPU_CHECK_DONE"
tail -5 /root/training_octo_A.log 2>&1; echo "TAIL_DONE"
