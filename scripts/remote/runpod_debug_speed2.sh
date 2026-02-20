# Check create_tasks implementation
source /root/octo_env/bin/activate
sed -n '121,190p' /root/octo-pytorch/octo/model/octo_model_pt.py 2>&1; echo "CREATE_TASKS_DONE"
