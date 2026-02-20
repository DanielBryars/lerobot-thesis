echo "=== Re-clone lerobot-thesis ==="
cd /root
mv lerobot-thesis lerobot-thesis-old 2>/dev/null
git clone https://github.com/danbhf/lerobot-thesis.git
ls /root/lerobot-thesis/scripts/training/train_octo.py
ls /root/lerobot-thesis/utils/octo_dataset.py
echo "=== Clone DONE ==="
