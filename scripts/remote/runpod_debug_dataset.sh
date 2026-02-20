# Debug dataset columns
source /root/octo_env/bin/activate
cd /root/lerobot-thesis

# Check what columns exist in the dataset
python3 -c "
from huggingface_hub import hf_hub_download
import pandas as pd
pf = hf_hub_download('danbhf/sim_pick_place_2pos_220ep_v2_delta', 'data/chunk-000/file-000.parquet', repo_type='dataset')
df = pd.read_parquet(pf)
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print('First row sample:')
for col in df.columns:
    val = df[col].iloc[0]
    if isinstance(val, list):
        print(f'  {col}: list len={len(val)}')
    elif isinstance(val, dict):
        print(f'  {col}: dict keys={list(val.keys())}')
    else:
        print(f'  {col}: {type(val).__name__} = {val}')
" 2>&1; echo "DEBUG_DONE"
