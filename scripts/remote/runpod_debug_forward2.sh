# Debug the T5 encoding path
source /root/octo_env/bin/activate
echo "=== Full LanguageTokenizerPt forward ==="
sed -n '245,310p' /root/octo-pytorch/octo/model/components/tokenizers_pt.py 2>&1; echo "LANG_TOK_DONE"
echo "=== text_processing.py HFTokenizer.encode ==="
sed -n '35,100p' /root/octo-pytorch/octo/data/utils/text_processing.py 2>&1; echo "TEXT_PROC_DONE"
echo "=== _np2pt function ==="
grep -n "_np2pt" /root/octo-pytorch/octo/model/octo_model_pt.py | head -5; echo "NP2PT_GREP"
sed -n '1,30p' /root/octo-pytorch/octo/model/octo_model_pt.py 2>&1; echo "NP2PT_DONE"
