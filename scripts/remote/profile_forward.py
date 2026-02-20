"""Profile Octo forward pass to find the CPU bottleneck."""
import sys
import time
import warnings
import logging
sys.path.insert(0, '/root/lerobot-thesis')
sys.path.insert(0, '/root/octo-pytorch')
warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)

import torch
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device('cuda')
print(f'Device: {torch.cuda.get_device_name(0)}')
mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'VRAM: {mem:.1f} GiB')

# Load model
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.spec import ModuleSpec
from octo.model.components.action_heads_pt import L1ActionHeadPt

print("Loading model...")
meta = OctoModelPt.load_config_and_meta_from_jax("hf://rail-berkeley/octo-small-1.5")
if "wrist" in meta["config"]["model"]["observation_tokenizers"]:
    del meta["config"]["model"]["observation_tokenizers"]["wrist"]
meta["config"]["model"]["num_tokens_dict"] = {"primary": 256, "language": 16, "action": 1}
meta["config"]["model"]["heads"]["action"] = ModuleSpec.create(
    L1ActionHeadPt, input_dim=384, action_horizon=4, action_dim=6, readout_key="readout_action"
)
model = OctoModelPt.from_config(**meta, verbose=False)
model.load_weights_from_jax("hf://rail-berkeley/octo-small-1.5", skip_keys_regex=".*hf_model")
model.to(device)
model.train()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Use smaller batch to avoid OOM during profiling
B = 16
print(f"\nBatch size: {B}")

obs = {
    "image_primary": torch.randn(B, 2, 3, 256, 256, device=device),
    "timestep_pad_mask": torch.ones(B, 2, dtype=torch.bool, device=device),
}
task = model.create_tasks(texts=["Pick up the block"] * B, device=device)

gt_actions = torch.randn(B, 2, 4, 6, device=device)
action_pad_mask = torch.ones(B, 2, 4, 6, dtype=torch.bool, device=device)

def mem_report():
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"alloc={alloc:.2f}G reserved={reserved:.2f}G"

# Warm up
print(f"Warming up... ({mem_report()})")
torch.cuda.synchronize()
with torch.no_grad():
    _, _ = model(obs, task, obs["timestep_pad_mask"], action_pad_mask=action_pad_mask,
                 gt_actions=gt_actions, train=True)
torch.cuda.synchronize()
torch.cuda.empty_cache()
print(f"Warm-up done. ({mem_report()})")

# Time full forward (with grad for realistic training scenario)
print("\n--- Full forward timing (with grad) ---")
torch.cuda.synchronize()
t0 = time.time()
_, head_out = model(obs, task, obs["timestep_pad_mask"], action_pad_mask=action_pad_mask,
                    gt_actions=gt_actions, train=True)
torch.cuda.synchronize()
fwd_time = time.time()-t0
print(f"Full forward: {fwd_time:.3f}s")

# Backward timing
loss = head_out["action"]["loss"]
torch.cuda.synchronize()
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
bwd_time = time.time()-t0
print(f"Backward: {bwd_time:.3f}s")
print(f"Fwd/Bwd ratio: {fwd_time/bwd_time:.1f}x")

# Clear grads
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()

# Now profile individual components
print(f"\n--- Component timing (B={B}) ---")
transformer = model.module.octo_transformer

# Time task tokenizer (language = T5)
print("\n[Task tokenizers]")
torch.cuda.synchronize()
for name, tok in transformer.task_tokenizers.items():
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = tok(obs, task, train=True)
    torch.cuda.synchronize()
    elapsed = time.time()-t0
    lang_type = type(task.get('language_instruction', None)) if hasattr(task, 'get') else 'N/A'
    print(f"  task_tok_{name}: {elapsed:.3f}s  (lang_instr_type={lang_type})")
torch.cuda.empty_cache()

# Time observation tokenizer (ViT)
print("\n[Observation tokenizers]")
torch.cuda.synchronize()
for name, tok in transformer.observation_tokenizers.items():
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = tok(obs, task, train=True)
    torch.cuda.synchronize()
    elapsed = time.time()-t0
    print(f"  obs_tok_{name}: {elapsed:.3f}s  output_shape={out.shape if hasattr(out, 'shape') else type(out)}")
torch.cuda.empty_cache()

# Time the BlockTransformer layers specifically
print("\n[BlockTransformer forward]")
# First, prepare the token groups as the transformer would
token_group_dict = {}
for name, tok in transformer.task_tokenizers.items():
    with torch.no_grad():
        token_group_dict[name] = tok(obs, task, train=True)
for name, tok in transformer.observation_tokenizers.items():
    with torch.no_grad():
        token_group_dict[name] = tok(obs, task, train=True)

# Readout tokens
readout_keys = list(model.module.octo_transformer.heads.keys())
print(f"  Readout keys: {readout_keys}")

# Check if there's a readout token method
if hasattr(transformer, 'readout_action'):
    print(f"  readout_action type: {type(transformer.readout_action)}")

# Time just the transformer body
# We need to find the actual transformer forward
print(f"  Token groups: {list(token_group_dict.keys())}")
for k, v in token_group_dict.items():
    if hasattr(v, 'shape'):
        print(f"    {k}: shape={v.shape}")
    elif isinstance(v, dict):
        print(f"    {k}: dict with keys={list(v.keys())}")
    else:
        print(f"    {k}: type={type(v)}")

# Pre-encode language and test again
print("\n--- Pre-encoding T5 ---")
lang_tok = transformer.task_tokenizers["language"]
lang_input = task["language_instruction"]
print(f"  language_instruction type: {type(lang_input)}")
if isinstance(lang_input, dict):
    print(f"  Keys: {list(lang_input.keys())}")
    t5_device = next(lang_tok.hf_model.parameters()).device
    print(f"  T5 device: {t5_device}")
    print(f"  T5 params: {sum(p.numel() for p in lang_tok.hf_model.parameters()):,}")

    lang_device = {k: v.to(t5_device) if isinstance(v, torch.Tensor) else v for k, v in lang_input.items()}

    # Time T5 encoding
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        t5_out = lang_tok.hf_model(**lang_device).last_hidden_state
    torch.cuda.synchronize()
    t5_time = time.time()-t0
    print(f"  T5 encoding time: {t5_time:.3f}s")
    print(f"  T5 output shape: {t5_out.shape}")

    task["language_instruction"] = t5_out.detach().to(device)
elif isinstance(lang_input, torch.Tensor):
    print(f"  Already a tensor: {lang_input.shape}")

torch.cuda.empty_cache()

# Full forward with pre-encoded T5
print("\n--- Forward with pre-encoded T5 ---")
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
t0 = time.time()
_, head_out2 = model(obs, task, obs["timestep_pad_mask"], action_pad_mask=action_pad_mask,
                     gt_actions=gt_actions, train=True)
torch.cuda.synchronize()
fwd2_time = time.time()-t0
print(f"Forward (pre-encoded T5): {fwd2_time:.3f}s")

loss2 = head_out2["action"]["loss"]
torch.cuda.synchronize()
t0 = time.time()
loss2.backward()
torch.cuda.synchronize()
bwd2_time = time.time()-t0
print(f"Backward (pre-encoded T5): {bwd2_time:.3f}s")
print(f"Fwd/Bwd ratio: {fwd2_time/bwd2_time:.1f}x")

# Third run to confirm
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
torch.cuda.synchronize()
t0 = time.time()
_, head_out3 = model(obs, task, obs["timestep_pad_mask"], action_pad_mask=action_pad_mask,
                     gt_actions=gt_actions, train=True)
torch.cuda.synchronize()
fwd3_time = time.time()-t0
print(f"Forward (3rd run): {fwd3_time:.3f}s")

# Profile with torch.profiler for detailed breakdown
print("\n--- Detailed profiling with torch.profiler ---")
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    _, head_out4 = model(obs, task, obs["timestep_pad_mask"], action_pad_mask=action_pad_mask,
                         gt_actions=gt_actions, train=True)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print("\n--- CPU time breakdown ---")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

print(f"\n--- Summary ---")
print(f"Full forward (with T5 dict):       {fwd_time:.3f}s")
print(f"Full forward (pre-encoded T5):      {fwd2_time:.3f}s")
print(f"Full forward (3rd run, pre-encoded): {fwd3_time:.3f}s")
print(f"Backward:                            {bwd_time:.3f}s / {bwd2_time:.3f}s")
print(f"T5 encoding alone:                   {t5_time:.3f}s" if 't5_time' in dir() else "T5 was already tensor")

print("\nPROFILE_DONE")
