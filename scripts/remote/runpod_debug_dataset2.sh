# Debug dataset columns (single-line commands)
source /root/octo_env/bin/activate
cd /root/lerobot-thesis
python3 -c "from huggingface_hub import hf_hub_download; import pandas as pd; pf = hf_hub_download('danbhf/sim_pick_place_2pos_220ep_v2_delta', 'data/chunk-000/file-000.parquet', repo_type='dataset'); df = pd.read_parquet(pf); print('Columns:', list(df.columns)); print('Shape:', df.shape)" 2>&1; echo "DEBUG_DONE"
