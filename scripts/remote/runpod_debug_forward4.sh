# Read ImageTokenizerPt - it might be the slow part
source /root/octo_env/bin/activate
echo "=== ImageTokenizerPt ==="
sed -n '96,215p' /root/octo-pytorch/octo/model/components/tokenizers_pt.py 2>&1; echo "IMG_TOK_DONE"
echo "=== Block transformer forward ==="
sed -n '300,400p' /root/octo-pytorch/octo/model/octo_module_pt.py 2>&1; echo "BLK_TRANS_DONE"
