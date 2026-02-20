# Fix JAX/numpy incompatibility by patching octo typing.py to not import JAX
source /root/octo_env/bin/activate

# Show the typing.py content
head -20 /root/octo-pytorch/octo/utils/typing.py; echo "TYPING_CONTENT_DONE"

# Check numpy version
python3 -c "import numpy; print('numpy:', numpy.__version__)" 2>&1; echo "NUMPY_CHECK_DONE"
