# Run GPU benchmark
source /root/octo_env/bin/activate
cd /root/lerobot-thesis
python3 scripts/remote/gpu_benchmark.py 2>&1; echo "BENCHMARK_SCRIPT_DONE"
