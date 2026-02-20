# Check training status - more log lines
source /root/octo_env/bin/activate
ps aux | grep train_octo | grep -v grep; echo "PS_CHECK_DONE"
tail -120 /root/training_octo_A.log 2>&1; echo "LOG_CHECK_DONE"
