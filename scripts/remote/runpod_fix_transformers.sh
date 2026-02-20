# Fix transformers/huggingface_hub incompatibility
source /root/octo_env/bin/activate

# Downgrade transformers to 4.x (compatible with huggingface_hub 0.x)
pip install "transformers<5" 2>&1 | tail -5; echo "TRANSFORMERS_FIX_DONE"

# Verify
python3 -c "import transformers; print('transformers:', transformers.__version__)" 2>&1; echo "CHECK1"
python3 -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('t5-base'); print('t5-base tokenizer OK')" 2>&1; echo "CHECK2"
