# Debug what's slow in the forward pass
source /root/octo_env/bin/activate

echo "=== OctoTransformerPt forward (lines 203-300) ==="
sed -n '203,300p' /root/octo-pytorch/octo/model/octo_module_pt.py 2>&1; echo "TRANSFORMER_FWD_DONE"

echo "=== Check if _verify_shapes is expensive ==="
sed -n '771,830p' /root/octo-pytorch/octo/model/octo_model_pt.py 2>&1; echo "VERIFY_DONE"

echo "=== Check observation tokenizer ==="
grep -n "class.*ImageTokenizer" /root/octo-pytorch/octo/model/components/tokenizers_pt.py; echo "IMG_TOK_GREP"
