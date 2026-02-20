# Read the full log (first 70 lines) to check T5 caching
source /root/octo_env/bin/activate
head -60 /root/training_octo_A.log 2>&1; echo "HEAD_DONE"
