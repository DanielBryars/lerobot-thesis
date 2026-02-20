# Debug action head shapes by reading the octo-pytorch source
source /root/octo_env/bin/activate
echo "=== action_heads_pt.py L1ActionHeadPt.loss ==="
sed -n '150,210p' /root/octo-pytorch/octo/model/components/action_heads_pt.py 2>&1; echo "ACTION_HEADS_DONE"
echo "=== octo_module_pt.py forward (relevant section) ==="
sed -n '490,540p' /root/octo-pytorch/octo/model/octo_module_pt.py 2>&1; echo "MODULE_DONE"
echo "=== octo_model_pt.py forward ==="
sed -n '480,530p' /root/octo-pytorch/octo/model/octo_model_pt.py 2>&1; echo "MODEL_DONE"
