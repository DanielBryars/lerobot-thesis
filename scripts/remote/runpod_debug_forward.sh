# Debug the forward pass - check transformer and task tokenizer code
source /root/octo_env/bin/activate
echo "=== OctoTransformerPt forward ==="
grep -n "def forward" /root/octo-pytorch/octo/model/octo_module_pt.py | head -10; echo "GREP1"
echo "=== Task tokenizer processing ==="
grep -rn "class.*TaskTokenizer" /root/octo-pytorch/octo/model/ | head -5; echo "GREP2"
echo "=== Text processing encode ==="
grep -n "def encode" /root/octo-pytorch/octo/data/utils/text_processing.py | head -5; echo "GREP3"
echo "=== LanguageTokenizerPt ==="
grep -n "class LanguageTokenizerPt" /root/octo-pytorch/octo/model/components/tokenizers_pt.py; echo "GREP4"
sed -n '/class LanguageTokenizerPt/,/class /p' /root/octo-pytorch/octo/model/components/tokenizers_pt.py 2>&1 | head -50; echo "TOKENIZER_DONE"
echo "=== OctoTransformerPt forward body ==="
grep -n "class OctoTransformerPt" /root/octo-pytorch/octo/model/octo_module_pt.py; echo "GREP5"
sed -n '1,100p' /root/octo-pytorch/octo/model/octo_module_pt.py 2>&1; echo "TRANSFORMER_DONE"
